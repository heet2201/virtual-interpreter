"""
Basic Automatic Speech Recognition (ASR) implementation using OpenAI Whisper
"""
import torch
import whisper
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import librosa
import soundfile as sf

from ..configs.config import ASR_CONFIG, AUDIO_CONFIG, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class BasicASR:
    """Basic ASR implementation using Whisper model"""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the Basic ASR system
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run the model on (cuda/cpu)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file for ASR
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=AUDIO_CONFIG["sample_rate"])
            
            # Ensure audio is not too long (avoid memory issues)
            max_samples = AUDIO_CONFIG["max_duration"] * AUDIO_CONFIG["sample_rate"]
            if len(audio) > max_samples:
                logger.warning(f"Audio too long, truncating to {AUDIO_CONFIG['max_duration']} seconds")
                audio = audio[:max_samples]
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def transcribe(self, audio_input, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_input: Audio file path or numpy array
            language: Source language code (optional, will auto-detect if None)
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Handle different input types
            if isinstance(audio_input, str):
                audio = self.preprocess_audio(audio_input)
            elif isinstance(audio_input, np.ndarray):
                audio = audio_input
            else:
                raise ValueError("Audio input must be file path or numpy array")
            
            # Prepare transcription options
            options = {}
            if language and language in [lang["whisper"] for lang in SUPPORTED_LANGUAGES.values()]:
                options["language"] = language
            
            # Transcribe
            logger.info("Starting transcription...")
            result = self.model.transcribe(audio, **options)
            
            # Extract results
            transcription_result = {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "confidence": self._calculate_confidence(result.get("segments", [])),
            }
            
            logger.info(f"Transcription completed: {transcription_result['text'][:100]}...")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
    
    def transcribe_realtime(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe audio chunk for real-time processing
        
        Args:
            audio_chunk: Audio chunk as numpy array
            
        Returns:
            Transcribed text
        """
        try:
            result = self.model.transcribe(audio_chunk, language=None)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error in real-time transcription: {e}")
            return ""
    
    def _calculate_confidence(self, segments) -> float:
        """
        Calculate average confidence score from segments
        
        Args:
            segments: Whisper segments with timestamps
            
        Returns:
            Average confidence score
        """
        if not segments:
            return 0.0
        
        # Whisper doesn't provide explicit confidence, estimate from other metrics
        # Use average probability if available, otherwise use segment consistency
        confidences = []
        for segment in segments:
            if "avg_logprob" in segment:
                # Convert log probability to confidence (rough approximation)
                conf = max(0.0, min(1.0, np.exp(segment["avg_logprob"])))
                confidences.append(conf)
        
        return np.mean(confidences) if confidences else 0.8  # Default confidence
    
    def batch_transcribe(self, audio_files: list, language: Optional[str] = None) -> list:
        """
        Transcribe multiple audio files
        
        Args:
            audio_files: List of audio file paths
            language: Source language code
            
        Returns:
            List of transcription results
        """
        results = []
        for audio_file in audio_files:
            try:
                result = self.transcribe(audio_file, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Error transcribing {audio_file}: {e}")
                results.append({"text": "", "error": str(e)})
        
        return results
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return list(SUPPORTED_LANGUAGES.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "supported_languages": self.get_supported_languages(),
        }


def create_asr_system(model_size: str = "base") -> BasicASR:
    """
    Factory function to create BasicASR system
    
    Args:
        model_size: Whisper model size
        
    Returns:
        BasicASR instance
    """
    return BasicASR(model_size=model_size)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create ASR system
    asr = create_asr_system("base")
    
    # Display model info
    print("Model Info:", asr.get_model_info())
    
    # Example transcription (uncomment to test with actual audio file)
    # result = asr.transcribe("path/to/audio/file.wav")
    # print("Transcription:", result["text"])
    # print("Language:", result["language"])
    # print("Confidence:", result["confidence"])
