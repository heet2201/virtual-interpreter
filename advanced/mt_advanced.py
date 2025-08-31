"""
Advanced Machine Translation implementation with fine-tuning and domain adaptation
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    AdamW
)
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
from dataclasses import dataclass
import pickle

from ..configs.config import TRANSLATION_CONFIG, TRAINING_CONFIG, SUPPORTED_LANGUAGES
from ..utils.evaluation import TranslationEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TranslationExample:
    """Data class for translation training examples"""
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    domain: Optional[str] = None
    quality_score: Optional[float] = None


class TranslationDataset(Dataset):
    """Dataset for translation fine-tuning"""
    
    def __init__(
        self,
        examples: List[TranslationExample],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        lang_codes: Dict[str, str] = None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_codes = lang_codes or {}
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        try:
            # Prepare source text with language prefix
            src_lang_code = self.lang_codes.get(example.source_language, example.source_language)
            source_text = example.source_text
            
            # Set source language for NLLB models
            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = src_lang_code
            
            # Tokenize source
            source_encoding = self.tokenizer(
                source_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Tokenize target
            target_encoding = self.tokenizer(
                example.target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Prepare labels (replace padding tokens with -100)
            labels = target_encoding["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": source_encoding["input_ids"].squeeze(),
                "attention_mask": source_encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze(),
                "source_language": example.source_language,
                "target_language": example.target_language,
                "domain": example.domain or "general"
            }
            
        except Exception as e:
            logger.error(f"Error processing translation example {idx}: {e}")
            # Return empty tensors as fallback
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.full((self.max_length,), -100, dtype=torch.long),
                "source_language": "en",
                "target_language": "en",
                "domain": "general"
            }


class DomainAdapter(nn.Module):
    """Domain adaptation layer for translation models"""
    
    def __init__(self, hidden_size: int, num_domains: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        
        # Domain-specific parameters
        self.domain_embeddings = nn.Embedding(num_domains, hidden_size)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains)
        )
        
        # Adaptation layers
        self.adaptation_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, hidden_states, domain_ids=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if domain_ids is None:
            # Predict domain from hidden states
            pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
            domain_logits = self.domain_classifier(pooled)
            domain_ids = torch.argmax(domain_logits, dim=-1)
        
        # Get domain embeddings
        domain_embeds = self.domain_embeddings(domain_ids)  # [batch_size, hidden_size]
        domain_embeds = domain_embeds.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Combine hidden states with domain embeddings
        combined = torch.cat([hidden_states, domain_embeds], dim=-1)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply adaptation
        adapted = self.adaptation_layer(combined)  # [batch_size, seq_len, hidden_size]
        
        return adapted + hidden_states  # Residual connection


class AdvancedTranslator:
    """Advanced Machine Translation with fine-tuning and domain adaptation"""
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        enable_domain_adaptation: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Advanced Translation system
        
        Args:
            model_name: HuggingFace model name
            device: Device for computation
            enable_domain_adaptation: Enable domain adaptation
            cache_dir: Cache directory for models
        """
        self.model_name = model_name or TRANSLATION_CONFIG["advanced"]["model_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_domain_adaptation = enable_domain_adaptation
        self.cache_dir = cache_dir
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.domain_adapter = None
        
        # Training components
        self.trainer = None
        self.evaluator = TranslationEvaluator()
        
        # Language and domain mappings
        self.lang_codes = self._get_language_codes()
        self.domain_to_id = {}
        self.id_to_domain = {}
        
        # Performance tracking
        self.training_history = []
        self.evaluation_results = {}
        
        self.load_model()
    
    def _get_language_codes(self) -> Dict[str, str]:
        """Get language codes for the model"""
        # NLLB language codes
        return {
            "en": "eng_Latn",
            "es": "spa_Latn", 
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "nl": "nld_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ar": "arb_Arab"
        }
    
    def load_model(self):
        """Load translation model and tokenizer"""
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            
            logger.info("Advanced translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        domain: Optional[str] = None,
        max_length: Optional[int] = None,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Advanced translation with additional options
        
        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code  
            domain: Domain for adaptation
            max_length: Maximum output length
            num_beams: Beam search size
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            return_scores: Return generation scores
            
        Returns:
            Translation result with additional information
        """
        try:
            # Validate languages
            if source_lang not in self.lang_codes or target_lang not in self.lang_codes:
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")
            
            # Skip if same language
            if source_lang == target_lang:
                return {
                    "translation": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "confidence": 1.0,
                    "domain": domain
                }
            
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Get language codes
            src_code = self.lang_codes[source_lang]
            tgt_code = self.lang_codes[target_lang]
            
            # Set source language
            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = src_code
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Prepare generation kwargs
            generation_kwargs = {
                "max_length": max_length or TRANSLATION_CONFIG["advanced"]["max_length"],
                "num_beams": num_beams,
                "temperature": temperature,
                "do_sample": do_sample,
                "forced_bos_token_id": self.tokenizer.lang_code_to_id.get(tgt_code),
                "return_dict_in_generate": True,
                "output_scores": return_scores,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate translation
            with torch.no_grad():
                # Apply domain adaptation if enabled
                if self.enable_domain_adaptation and self.domain_adapter and domain:
                    domain_id = self.domain_to_id.get(domain, 0)
                    # This would require modifying the forward pass - simplified here
                    generated = self.model.generate(**inputs, **generation_kwargs)
                else:
                    generated = self.model.generate(**inputs, **generation_kwargs)
            
            # Decode translation
            translation = self.tokenizer.decode(
                generated.sequences[0], 
                skip_special_tokens=True
            )
            
            # Post-process
            translation = self._postprocess_text(translation)
            
            # Calculate confidence
            confidence = self._calculate_confidence(generated, inputs)
            
            result = {
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "confidence": confidence,
                "domain": domain,
                "generation_params": {
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "do_sample": do_sample
                }
            }
            
            # Add scores if requested
            if return_scores and hasattr(generated, 'scores'):
                result["generation_scores"] = [score.cpu().numpy() for score in generated.scores]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced translation: {e}")
            return {"error": str(e)}
    
    def prepare_training_data(
        self,
        data_path: str,
        domains: List[str] = None,
        validation_split: float = 0.1,
        quality_threshold: float = 0.0
    ) -> Tuple[TranslationDataset, TranslationDataset]:
        """
        Prepare training data for fine-tuning
        
        Args:
            data_path: Path to training data
            domains: List of domains for adaptation
            validation_split: Validation split ratio
            quality_threshold: Minimum quality score
            
        Returns:
            Training and validation datasets
        """
        try:
            # Load examples
            examples = self._load_translation_examples(data_path, quality_threshold)
            
            # Setup domain mappings
            if domains:
                self.domain_to_id = {domain: i for i, domain in enumerate(domains)}
                self.id_to_domain = {i: domain for i, domain in enumerate(domains)}
            
            # Split data
            n_train = int(len(examples) * (1 - validation_split))
            train_examples = examples[:n_train]
            val_examples = examples[n_train:]
            
            # Create datasets
            train_dataset = TranslationDataset(
                train_examples, self.tokenizer, lang_codes=self.lang_codes
            )
            val_dataset = TranslationDataset(
                val_examples, self.tokenizer, lang_codes=self.lang_codes
            )
            
            logger.info(f"Prepared {len(train_dataset)} training and {len(val_dataset)} validation samples")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def fine_tune(
        self,
        train_dataset: TranslationDataset,
        val_dataset: TranslationDataset,
        output_dir: str,
        num_epochs: int = None,
        learning_rate: float = None,
        batch_size: int = None,
        gradient_accumulation_steps: int = None,
        warmup_steps: int = None,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        early_stopping_patience: int = 3,
        use_domain_adaptation: bool = False
    ) -> Dict[str, Any]:
        """
        Fine-tune the translation model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory
            num_epochs: Training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            save_strategy: Save strategy
            evaluation_strategy: Evaluation strategy
            early_stopping_patience: Early stopping patience
            use_domain_adaptation: Use domain adaptation
            
        Returns:
            Training results
        """
        try:
            # Set default parameters
            num_epochs = num_epochs or TRAINING_CONFIG["num_epochs"]
            learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
            batch_size = batch_size or TRAINING_CONFIG["batch_size"]
            gradient_accumulation_steps = gradient_accumulation_steps or TRAINING_CONFIG["gradient_accumulation_steps"]
            warmup_steps = warmup_steps or TRAINING_CONFIG["warmup_steps"]
            
            # Initialize domain adapter if requested
            if use_domain_adaptation and len(self.domain_to_id) > 0:
                hidden_size = self.model.config.hidden_size
                num_domains = len(self.domain_to_id)
                self.domain_adapter = DomainAdapter(hidden_size, num_domains)
                self.domain_adapter.to(self.device)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=TRAINING_CONFIG["weight_decay"],
                warmup_steps=warmup_steps,
                logging_steps=50,
                save_steps=500,
                eval_steps=500,
                save_strategy=save_strategy,
                evaluation_strategy=evaluation_strategy,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                push_to_hub=False,
                report_to=[],
                dataloader_drop_last=True,
                fp16=self.device == "cuda",
                dataloader_num_workers=2,
                remove_unused_columns=False,
            )
            
            # Data collator
            def data_collator(features):
                batch = self.tokenizer.pad(
                    {
                        "input_ids": [f["input_ids"] for f in features],
                        "attention_mask": [f["attention_mask"] for f in features],
                        "labels": [f["labels"] for f in features]
                    },
                    padding=True,
                    return_tensors="pt"
                )
                return batch
            
            # Compute metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                
                # Decode predictions and labels
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Compute BLEU score
                bleu_scores = []
                for pred, label in zip(decoded_preds, decoded_labels):
                    score = self.evaluator.bleu_score(label, pred)
                    bleu_scores.append(score)
                
                return {
                    "bleu": np.mean(bleu_scores),
                    "bleu_std": np.std(bleu_scores)
                }
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            )
            
            # Train model
            logger.info("Starting translation model fine-tuning...")
            train_result = self.trainer.train()
            
            # Save model and tokenizer
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save domain mappings if used
            if self.domain_to_id:
                with open(Path(output_dir) / "domain_mappings.json", "w") as f:
                    json.dump({
                        "domain_to_id": self.domain_to_id,
                        "id_to_domain": self.id_to_domain
                    }, f, indent=2)
            
            # Save domain adapter if used
            if self.domain_adapter:
                torch.save(
                    self.domain_adapter.state_dict(),
                    Path(output_dir) / "domain_adapter.pt"
                )
            
            # Store training history
            self.training_history.append(train_result)
            
            logger.info("Fine-tuning completed successfully")
            
            return {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "output_dir": output_dir
            }
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def _load_translation_examples(
        self, 
        data_path: str, 
        quality_threshold: float
    ) -> List[TranslationExample]:
        """Load translation examples from data path"""
        examples = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Load from JSON lines file
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    quality = data.get("quality_score", 1.0)
                    
                    if quality >= quality_threshold:
                        examples.append(TranslationExample(
                            source_text=data["source_text"],
                            target_text=data["target_text"],
                            source_language=data["source_language"],
                            target_language=data["target_language"],
                            domain=data.get("domain"),
                            quality_score=quality
                        ))
        
        elif data_path.is_dir():
            # Load from directory with parallel files
            for src_file in data_path.glob("*.src"):
                tgt_file = src_file.with_suffix(".tgt")
                if tgt_file.exists():
                    with open(src_file, 'r', encoding='utf-8') as sf, \
                         open(tgt_file, 'r', encoding='utf-8') as tf:
                        
                        src_lines = sf.readlines()
                        tgt_lines = tf.readlines()
                        
                        for src_line, tgt_line in zip(src_lines, tgt_lines):
                            examples.append(TranslationExample(
                                source_text=src_line.strip(),
                                target_text=tgt_line.strip(),
                                source_language="en",  # Default
                                target_language="es",  # Default
                                quality_score=1.0
                            ))
        
        return examples
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for translation"""
        text = text.strip()
        # Add any domain-specific preprocessing here
        return text
    
    def _postprocess_text(self, text: str) -> str:
        """Postprocess translated text"""
        text = text.strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text
    
    def _calculate_confidence(self, generated, inputs) -> float:
        """Calculate translation confidence score"""
        # Simplified confidence calculation
        input_length = inputs['input_ids'].shape[1]
        output_length = generated.sequences.shape[1]
        
        # Length-based heuristic
        length_ratio = min(output_length / max(input_length, 1), 2.0)
        confidence = min(0.9, 0.4 + 0.3 * length_ratio)
        
        return confidence
    
    def evaluate_model(
        self, 
        test_dataset: TranslationDataset,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """Evaluate translation model"""
        try:
            dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            all_predictions = []
            all_references = []
            
            self.model.eval()
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"]
                    
                    # Generate predictions
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode
                    predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                    references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
            
            # Calculate metrics
            results = self.evaluator.evaluate_batch(all_references, all_predictions)
            
            self.evaluation_results = results
            
            logger.info(f"Evaluation completed - BLEU: {results['bleu_mean']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"error": str(e)}
    
    def save_model(self, output_dir: str):
        """Save model, tokenizer, and additional components"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training history
        with open(output_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save domain mappings
        if self.domain_to_id:
            with open(output_path / "domain_mappings.json", "w") as f:
                json.dump({
                    "domain_to_id": self.domain_to_id,
                    "id_to_domain": self.id_to_domain
                }, f, indent=2)
        
        # Save domain adapter
        if self.domain_adapter:
            torch.save(self.domain_adapter.state_dict(), output_path / "domain_adapter.pt")
        
        logger.info(f"Advanced translation model saved to {output_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "domain_adaptation_enabled": self.enable_domain_adaptation,
            "supported_languages": list(self.lang_codes.keys()),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        if self.domain_to_id:
            info["domains"] = list(self.domain_to_id.keys())
        
        if self.training_history:
            info["training_history"] = self.training_history
        
        if self.evaluation_results:
            info["evaluation_results"] = self.evaluation_results
        
        return info


def create_advanced_translator(
    model_name: str = None,
    enable_domain_adaptation: bool = False,
    device: str = None
) -> AdvancedTranslator:
    """
    Factory function to create Advanced Translator
    
    Args:
        model_name: Model name
        enable_domain_adaptation: Enable domain adaptation
        device: Computation device
        
    Returns:
        AdvancedTranslator instance
    """
    return AdvancedTranslator(
        model_name=model_name,
        device=device,
        enable_domain_adaptation=enable_domain_adaptation
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create advanced translator
        translator = create_advanced_translator(enable_domain_adaptation=True)
        
        print("Advanced Translation Model Info:")
        print(json.dumps(translator.get_model_info(), indent=2))
        
        # Example translation
        # result = translator.translate(
        #     "Hello, how are you today?",
        #     source_lang="en",
        #     target_lang="es", 
        #     domain="conversational",
        #     num_beams=5,
        #     return_scores=True
        # )
        # print("Translation Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have sufficient GPU memory and required dependencies")
