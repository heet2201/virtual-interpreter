"""
Basic Machine Translation (MT) implementation using HuggingFace NLLB models
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

from ..configs.config import TRANSLATION_CONFIG, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class BasicTranslator:
    """Basic Machine Translation implementation using NLLB/M2M100 models"""
    
    def __init__(self, model_name: str = None, device: Optional[str] = None):
        """
        Initialize the Basic Translation system
        
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name or TRANSLATION_CONFIG["basic"]["model_name"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.lang_codes = self._get_language_codes()
        self.load_model()
    
    def _get_language_codes(self) -> Dict[str, str]:
        """Get language codes mapping for the model"""
        # NLLB language codes mapping
        nllb_codes = {
            "en": "eng_Latn",  # English
            "es": "spa_Latn",  # Spanish
            "fr": "fra_Latn",  # French
            "de": "deu_Latn",  # German
            "it": "ita_Latn",  # Italian
            "pt": "por_Latn",  # Portuguese
            "nl": "nld_Latn",  # Dutch
            "ru": "rus_Cyrl",  # Russian
            "zh": "zho_Hans",  # Chinese (Simplified)
            "ja": "jpn_Jpan",  # Japanese
            "ko": "kor_Hang",  # Korean
            "ar": "arb_Arab",  # Arabic
        }
        return nllb_codes
    
    def load_model(self):
        """Load the translation model"""
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading translation model: {e}")
            raise
    
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Translate text from source language to target language
        
        Args:
            text: Input text to translate
            source_lang: Source language code (e.g., 'en', 'es')
            target_lang: Target language code
            max_length: Maximum length of translation
            
        Returns:
            Dictionary containing translation results
        """
        try:
            # Validate languages
            if source_lang not in self.lang_codes or target_lang not in self.lang_codes:
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")
            
            # Skip translation if source and target are the same
            if source_lang == target_lang:
                return {
                    "translation": text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "confidence": 1.0,
                    "model": self.model_name
                }
            
            # Preprocess text
            text = self._preprocess_text(text)
            if not text.strip():
                return {"translation": "", "source_lang": source_lang, "target_lang": target_lang}
            
            # Get NLLB language codes
            src_code = self.lang_codes[source_lang]
            tgt_code = self.lang_codes[target_lang]
            
            # Set source language in tokenizer
            self.tokenizer.src_lang = src_code
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            max_len = max_length or TRANSLATION_CONFIG["basic"]["max_length"]
            
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                    max_length=max_len,
                    num_beams=TRANSLATION_CONFIG["basic"]["num_beams"],
                    do_sample=False,
                    early_stopping=True
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            # Post-process translation
            translation = self._postprocess_text(translation)
            
            # Calculate confidence (approximation)
            confidence = self._calculate_confidence(inputs, generated_tokens)
            
            result = {
                "translation": translation,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "confidence": confidence,
                "model": self.model_name
            }
            
            logger.info(f"Translation completed: {text[:50]}... -> {translation[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            raise
    
    def batch_translate(
        self, 
        texts: List[str], 
        source_lang: str, 
        target_lang: str
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts
        
        Args:
            texts: List of input texts
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translation results
        """
        results = []
        for text in texts:
            try:
                result = self.translate(text, source_lang, target_lang)
                results.append(result)
            except Exception as e:
                logger.error(f"Error translating text '{text[:50]}...': {e}")
                results.append({
                    "translation": "",
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "error": str(e)
                })
        
        return results
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection (placeholder - would use a proper detector in production)
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # This is a simplified implementation
        # In production, use a proper language detection library like langdetect
        
        # Simple heuristics based on character patterns
        if re.search(r'[а-яё]', text.lower()):
            return 'ru'
        elif re.search(r'[一-龯]', text):
            return 'zh'
        elif re.search(r'[ひらがなカタカナ]', text):
            return 'ja'
        elif re.search(r'[가-힣]', text):
            return 'ko'
        elif re.search(r'[ا-ي]', text):
            return 'ar'
        else:
            # Default to English for Latin script
            return 'en'
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        # Basic preprocessing
        text = text.strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _postprocess_text(self, text: str) -> str:
        """Postprocess translated text"""
        # Basic postprocessing
        text = text.strip()
        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text
    
    def _calculate_confidence(self, inputs: Dict, generated_tokens: torch.Tensor) -> float:
        """
        Calculate translation confidence (approximation)
        
        Args:
            inputs: Tokenized inputs
            generated_tokens: Generated token IDs
            
        Returns:
            Confidence score between 0 and 1
        """
        # This is a simplified confidence calculation
        # In practice, you might use model scores, attention weights, etc.
        
        input_length = inputs['input_ids'].shape[1]
        output_length = generated_tokens.shape[1]
        
        # Simple heuristic: longer inputs with reasonable output length tend to be more reliable
        length_ratio = min(output_length / max(input_length, 1), 2.0)
        confidence = min(0.9, 0.5 + 0.2 * length_ratio)
        
        return confidence
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.lang_codes.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "supported_languages": self.get_supported_languages(),
            "language_pairs": len(self.get_supported_languages()) ** 2
        }


class MultiDirectionalTranslator:
    """Wrapper for handling multiple translation directions"""
    
    def __init__(self, model_name: str = None):
        """Initialize multi-directional translator"""
        self.translator = BasicTranslator(model_name)
        self.supported_langs = self.translator.get_supported_languages()
    
    def translate_to_english(self, text: str, source_lang: str) -> Dict[str, Any]:
        """Translate from any supported language to English"""
        return self.translator.translate(text, source_lang, "en")
    
    def translate_from_english(self, text: str, target_lang: str) -> Dict[str, Any]:
        """Translate from English to any supported language"""
        return self.translator.translate(text, "en", target_lang)
    
    def get_translation_pairs(self) -> List[Tuple[str, str]]:
        """Get all supported translation pairs"""
        pairs = []
        for src in self.supported_langs:
            for tgt in self.supported_langs:
                if src != tgt:
                    pairs.append((src, tgt))
        return pairs


def create_translator(model_name: str = None) -> BasicTranslator:
    """
    Factory function to create BasicTranslator
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        BasicTranslator instance
    """
    return BasicTranslator(model_name)


def create_bidirectional_translator() -> MultiDirectionalTranslator:
    """
    Factory function to create MultiDirectionalTranslator
    
    Returns:
        MultiDirectionalTranslator instance
    """
    return MultiDirectionalTranslator()


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create translator
    translator = create_translator()
    
    # Display model info
    print("Model Info:", translator.get_model_info())
    
    # Example translation
    try:
        result = translator.translate(
            "Hello, how are you today?", 
            source_lang="en", 
            target_lang="es"
        )
        print("Translation:", result["translation"])
        print("Confidence:", result["confidence"])
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install required dependencies and have sufficient memory/GPU")
