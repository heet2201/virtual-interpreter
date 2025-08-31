"""
Advanced Automatic Speech Recognition (ASR) implementation with fine-tuning capabilities
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    WhisperConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import librosa
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
from dataclasses import dataclass
import evaluate

from ..configs.config import ASR_CONFIG, AUDIO_CONFIG, TRAINING_CONFIG
from ..utils.audio_utils import load_audio, normalize_audio
from ..utils.evaluation import ASREvaluator

logger = logging.getLogger(__name__)


@dataclass
class ASRTrainingExample:
    """Data class for ASR training examples"""
    audio_path: str
    transcript: str
    language: str
    duration: float


class ASRDataset(Dataset):
    """Dataset for ASR fine-tuning"""
    
    def __init__(
        self, 
        examples: List[ASRTrainingExample],
        processor: WhisperProcessor,
        max_length: int = 30
    ):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        try:
            # Load and preprocess audio
            audio, sr = load_audio(example.audio_path, target_sr=16000)
            
            # Limit audio length
            max_samples = self.max_length * sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Process with Whisper processor
            inputs = self.processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Process transcript
            with self.processor.as_target_processor():
                labels = self.processor(
                    example.transcript,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).input_ids
            
            return {
                "input_features": inputs.input_features.squeeze(),
                "labels": labels.squeeze(),
                "language": example.language
            }
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            # Return empty tensors as fallback
            return {
                "input_features": torch.zeros((80, 3000)),  # Whisper mel-spectrogram shape
                "labels": torch.zeros((1,), dtype=torch.long),
                "language": "en"
            }


class AdvancedASR:
    """Advanced ASR with fine-tuning capabilities"""
    
    def __init__(
        self, 
        model_name: str = None,
        device: Optional[str] = None,
        enable_fine_tuning: bool = True
    ):
        """
        Initialize Advanced ASR system
        
        Args:
            model_name: Whisper model name/path
            device: Device for computation
            enable_fine_tuning: Whether to enable fine-tuning features
        """
        self.model_name = model_name or ASR_CONFIG["advanced"]["model_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_fine_tuning = enable_fine_tuning
        
        # Model components
        self.processor = None
        self.model = None
        self.config = None
        
        # Training components
        self.trainer = None
        self.evaluator = ASREvaluator()
        
        # Performance tracking
        self.training_history = []
        self.evaluation_results = {}
        
        self.load_model()
    
    def load_model(self):
        """Load Whisper model and processor"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Load processor and model
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Get model config
            self.config = self.model.config
            
            logger.info("Advanced Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def transcribe(
        self, 
        audio_input, 
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        beam_size: int = 5,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Advanced transcription with more options
        
        Args:
            audio_input: Audio file path or numpy array
            language: Source language
            task: 'transcribe' or 'translate'
            temperature: Sampling temperature
            beam_size: Beam search size
            return_probabilities: Whether to return token probabilities
            
        Returns:
            Transcription results with additional information
        """
        try:
            # Process audio
            if isinstance(audio_input, str):
                audio, sr = load_audio(audio_input, target_sr=16000)
            else:
                audio = audio_input
                sr = 16000
            
            # Preprocess audio
            audio = normalize_audio(audio)
            
            # Process with Whisper processor
            inputs = self.processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate with advanced options
            generation_kwargs = {
                "max_new_tokens": 448,
                "num_beams": beam_size,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "return_dict_in_generate": True,
                "output_scores": return_probabilities,
            }
            
            if language:
                generation_kwargs["language"] = language
            if task:
                generation_kwargs["task"] = task
            
            with torch.no_grad():
                generated = self.model.generate(inputs, **generation_kwargs)
            
            # Decode results
            transcription = self.processor.batch_decode(
                generated.sequences, 
                skip_special_tokens=True
            )[0]
            
            result = {
                "text": transcription.strip(),
                "language": language or "auto",
                "task": task,
                "beam_size": beam_size,
                "temperature": temperature
            }
            
            # Add probabilities if requested
            if return_probabilities and hasattr(generated, 'scores'):
                result["token_probabilities"] = self._extract_token_probabilities(
                    generated.scores, generated.sequences[0]
                )
                result["confidence"] = self._calculate_confidence_from_probs(
                    result["token_probabilities"]
                )
            else:
                result["confidence"] = 0.85  # Default confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced transcription: {e}")
            return {"error": str(e)}
    
    def prepare_training_data(
        self, 
        data_path: str,
        validation_split: float = 0.1,
        max_duration: float = 30.0
    ) -> Tuple[ASRDataset, ASRDataset]:
        """
        Prepare training data from a directory or manifest file
        
        Args:
            data_path: Path to training data
            validation_split: Fraction for validation
            max_duration: Maximum audio duration
            
        Returns:
            Training and validation datasets
        """
        try:
            # Load training examples
            examples = self._load_training_examples(data_path, max_duration)
            
            # Split into train and validation
            n_train = int(len(examples) * (1 - validation_split))
            train_examples = examples[:n_train]
            val_examples = examples[n_train:]
            
            # Create datasets
            train_dataset = ASRDataset(train_examples, self.processor)
            val_dataset = ASRDataset(val_examples, self.processor)
            
            logger.info(f"Prepared {len(train_dataset)} training and {len(val_dataset)} validation samples")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def fine_tune(
        self,
        train_dataset: ASRDataset,
        val_dataset: ASRDataset,
        output_dir: str,
        num_epochs: int = None,
        learning_rate: float = None,
        batch_size: int = None,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        early_stopping_patience: int = 3
    ) -> Dict[str, Any]:
        """
        Fine-tune the Whisper model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            save_strategy: When to save model
            evaluation_strategy: When to evaluate
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training results
        """
        if not self.enable_fine_tuning:
            raise RuntimeError("Fine-tuning not enabled")
        
        try:
            # Set training parameters
            num_epochs = num_epochs or TRAINING_CONFIG["num_epochs"]
            learning_rate = learning_rate or TRAINING_CONFIG["learning_rate"]
            batch_size = batch_size or TRAINING_CONFIG["batch_size"]
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
                learning_rate=learning_rate,
                warmup_steps=TRAINING_CONFIG["warmup_steps"],
                logging_steps=50,
                save_steps=500,
                eval_steps=500,
                save_strategy=save_strategy,
                evaluation_strategy=evaluation_strategy,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                push_to_hub=False,
                report_to=[],  # Disable wandb/tensorboard
                dataloader_drop_last=True,
                remove_unused_columns=False,
                fp16=self.device == "cuda",
            )
            
            # Data collator
            def data_collator(features):
                batch = {}
                
                # Collect input features
                input_features = [f["input_features"] for f in features]
                labels = [f["labels"] for f in features]
                
                # Pad input features
                max_len = max(f.shape[-1] for f in input_features)
                batch_features = []
                for f in input_features:
                    if f.shape[-1] < max_len:
                        padded = torch.nn.functional.pad(f, (0, max_len - f.shape[-1]))
                        batch_features.append(padded)
                    else:
                        batch_features.append(f)
                
                batch["input_features"] = torch.stack(batch_features)
                
                # Pad labels
                if labels[0].dim() > 0:
                    max_label_len = max(l.shape[0] for l in labels)
                    padded_labels = []
                    for l in labels:
                        if l.shape[0] < max_label_len:
                            padded = torch.nn.functional.pad(l, (0, max_label_len - l.shape[0]), value=-100)
                            padded_labels.append(padded)
                        else:
                            padded_labels.append(l)
                    batch["labels"] = torch.stack(padded_labels)
                else:
                    batch["labels"] = torch.tensor([l.item() if l.dim() > 0 else l for l in labels])
                
                return batch
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            )
            
            # Start training
            logger.info("Starting fine-tuning...")
            train_result = self.trainer.train()
            
            # Save model
            self.trainer.save_model()
            self.processor.save_pretrained(output_dir)
            
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
    
    def evaluate_model(
        self, 
        test_dataset: ASRDataset,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            test_dataset: Test dataset
            batch_size: Evaluation batch size
            
        Returns:
            Evaluation results
        """
        try:
            dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            all_predictions = []
            all_references = []
            
            self.model.eval()
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move to device
                    input_features = batch["input_features"].to(self.device)
                    labels = batch["labels"]
                    
                    # Generate predictions
                    generated = self.model.generate(
                        input_features,
                        max_new_tokens=448,
                        num_beams=5
                    )
                    
                    # Decode predictions and references
                    predictions = self.processor.batch_decode(generated, skip_special_tokens=True)
                    references = self.processor.batch_decode(labels, skip_special_tokens=True)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
            
            # Calculate metrics
            results = self.evaluator.evaluate_batch(all_references, all_predictions)
            
            self.evaluation_results = results
            
            logger.info(f"Evaluation completed - WER: {results['wer_mean']:.3f}, CER: {results['cer_mean']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"error": str(e)}
    
    def _load_training_examples(
        self, 
        data_path: str, 
        max_duration: float
    ) -> List[ASRTrainingExample]:
        """Load training examples from data path"""
        examples = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Load from manifest file (JSON lines format)
            with open(data_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("duration", 0) <= max_duration:
                        examples.append(ASRTrainingExample(
                            audio_path=data["audio_path"],
                            transcript=data["transcript"],
                            language=data.get("language", "en"),
                            duration=data.get("duration", 0.0)
                        ))
        
        elif data_path.is_dir():
            # Load from directory structure
            for audio_file in data_path.glob("**/*.wav"):
                transcript_file = audio_file.with_suffix(".txt")
                if transcript_file.exists():
                    with open(transcript_file, 'r') as f:
                        transcript = f.read().strip()
                    
                    # Get duration
                    audio, sr = load_audio(str(audio_file))
                    duration = len(audio) / sr
                    
                    if duration <= max_duration:
                        examples.append(ASRTrainingExample(
                            audio_path=str(audio_file),
                            transcript=transcript,
                            language="en",  # Default language
                            duration=duration
                        ))
        
        return examples
    
    def _extract_token_probabilities(
        self, 
        scores: List[torch.Tensor], 
        sequence: torch.Tensor
    ) -> List[float]:
        """Extract token probabilities from generation scores"""
        probabilities = []
        
        for i, score in enumerate(scores):
            if i < len(sequence) - 1:  # Skip last token
                token_id = sequence[i + 1]  # Skip BOS token
                probs = torch.softmax(score[0], dim=-1)
                token_prob = probs[token_id].item()
                probabilities.append(token_prob)
        
        return probabilities
    
    def _calculate_confidence_from_probs(self, token_probs: List[float]) -> float:
        """Calculate confidence score from token probabilities"""
        if not token_probs:
            return 0.0
        
        # Geometric mean of probabilities
        log_probs = [np.log(max(p, 1e-10)) for p in token_probs]
        return np.exp(np.mean(log_probs))
    
    def save_model(self, output_dir: str):
        """Save model and processor"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        # Save training history
        with open(output_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_fine_tuned_model(self, model_dir: str):
        """Load a fine-tuned model"""
        try:
            self.processor = WhisperProcessor.from_pretrained(model_dir)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            
            # Load training history if available
            history_file = Path(model_dir) / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.training_history = json.load(f)
            
            logger.info(f"Fine-tuned model loaded from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "fine_tuning_enabled": self.enable_fine_tuning,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        if self.training_history:
            info["training_history"] = self.training_history
        
        if self.evaluation_results:
            info["evaluation_results"] = self.evaluation_results
        
        return info


def create_advanced_asr(
    model_name: str = None,
    enable_fine_tuning: bool = True,
    device: str = None
) -> AdvancedASR:
    """
    Factory function to create Advanced ASR system
    
    Args:
        model_name: Whisper model name
        enable_fine_tuning: Enable fine-tuning capabilities
        device: Computation device
        
    Returns:
        AdvancedASR instance
    """
    return AdvancedASR(
        model_name=model_name,
        device=device,
        enable_fine_tuning=enable_fine_tuning
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create advanced ASR system
        asr = create_advanced_asr(enable_fine_tuning=True)
        
        print("Advanced ASR Model Info:")
        print(json.dumps(asr.get_model_info(), indent=2))
        
        # Example transcription
        # result = asr.transcribe(
        #     "path/to/audio.wav",
        #     language="en",
        #     beam_size=5,
        #     return_probabilities=True
        # )
        # print("Transcription Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have the required dependencies and GPU/CPU resources")
