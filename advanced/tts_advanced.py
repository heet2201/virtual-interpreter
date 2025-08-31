"""
Advanced Text-to-Speech (TTS) implementation using Coqui TTS and neural vocoders
"""
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List, Tuple
import tempfile
import json
import pickle

try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.manage import ModelManager
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

try:
    import nemo
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

from ..configs.config import TTS_CONFIG, SUPPORTED_LANGUAGES
from ..utils.audio_utils import normalize_audio, audio_quality_check

logger = logging.getLogger(__name__)


class VoiceCloner:
    """Voice cloning functionality for TTS"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.reference_embeddings = {}
        
    def create_voice_embedding(
        self, 
        reference_audio: str, 
        speaker_name: str,
        duration_threshold: float = 10.0
    ) -> Dict[str, Any]:
        """
        Create voice embedding from reference audio
        
        Args:
            reference_audio: Path to reference audio file
            speaker_name: Name for the speaker
            duration_threshold: Minimum duration for good embedding
            
        Returns:
            Voice embedding information
        """
        try:
            # Load and validate audio
            audio, sr = librosa.load(reference_audio, sr=22050)
            duration = len(audio) / sr
            
            if duration < duration_threshold:
                logger.warning(f"Audio duration {duration:.2f}s is below recommended {duration_threshold}s")
            
            # Preprocess audio
            audio = normalize_audio(audio, target_level=-20.0)
            
            # Quality check
            quality = audio_quality_check(audio, sr)
            
            if quality["quality_score"] < 0.7:
                logger.warning(f"Low audio quality detected: {quality['quality_score']:.2f}")
            
            # Create embedding (simplified - would use actual voice encoder)
            embedding = self._extract_voice_embedding(audio, sr)
            
            # Store embedding
            self.reference_embeddings[speaker_name] = {
                "embedding": embedding,
                "quality_score": quality["quality_score"],
                "duration": duration,
                "audio_path": reference_audio
            }
            
            return {
                "speaker_name": speaker_name,
                "embedding_created": True,
                "quality_score": quality["quality_score"],
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error creating voice embedding: {e}")
            return {"error": str(e)}
    
    def _extract_voice_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract voice embedding from audio (placeholder)"""
        # This would use a real voice encoder like SpeakerNet, ECAPA-TDNN, etc.
        # For now, return a random embedding as placeholder
        return np.random.randn(512).astype(np.float32)
    
    def get_speaker_embedding(self, speaker_name: str) -> Optional[np.ndarray]:
        """Get stored speaker embedding"""
        if speaker_name in self.reference_embeddings:
            return self.reference_embeddings[speaker_name]["embedding"]
        return None
    
    def list_speakers(self) -> List[str]:
        """List available speakers"""
        return list(self.reference_embeddings.keys())


class AdvancedTTS:
    """Advanced TTS with neural models and voice cloning"""
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        use_gpu: bool = True,
        enable_voice_cloning: bool = False
    ):
        """
        Initialize Advanced TTS system
        
        Args:
            model_name: TTS model name
            device: Computation device
            use_gpu: Whether to use GPU
            enable_voice_cloning: Enable voice cloning features
        """
        self.model_name = model_name or TTS_CONFIG["advanced"]["model"]
        self.device = device or ("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_voice_cloning = enable_voice_cloning
        
        # Model components
        self.tts_model = None
        self.vocoder = None
        self.voice_cloner = None
        
        # Language support
        self.supported_languages = []
        
        # Audio settings
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "total_duration": 0.0,
            "avg_rtf": 0.0  # Real-time factor
        }
        
        self.temp_dir = Path(tempfile.gettempdir()) / "advanced_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize TTS models"""
        try:
            logger.info("Initializing Advanced TTS models...")
            
            if COQUI_AVAILABLE:
                self._initialize_coqui_tts()
            elif NEMO_AVAILABLE:
                self._initialize_nemo_tts()
            else:
                raise ImportError("No advanced TTS library available (Coqui TTS or NeMo)")
            
            # Initialize voice cloner if enabled
            if self.enable_voice_cloning:
                self.voice_cloner = VoiceCloner()
                logger.info("Voice cloning enabled")
            
            logger.info("Advanced TTS initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing TTS models: {e}")
            raise
    
    def _initialize_coqui_tts(self):
        """Initialize Coqui TTS models"""
        try:
            # Load TTS model
            self.tts_model = TTS(
                model_name=self.model_name,
                progress_bar=False,
                gpu=self.use_gpu
            )
            
            # Get supported languages
            if hasattr(self.tts_model, 'languages'):
                self.supported_languages = self.tts_model.languages
            else:
                self.supported_languages = list(SUPPORTED_LANGUAGES.keys())
            
            logger.info(f"Coqui TTS model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Coqui TTS: {e}")
            raise
    
    def _initialize_nemo_tts(self):
        """Initialize NeMo TTS models"""
        try:
            # Load FastPitch model for mel-spectrogram generation
            self.tts_model = FastPitchModel.from_pretrained("tts_en_fastpitch")
            
            # Load HiFiGAN vocoder
            self.vocoder = HifiGanModel.from_pretrained("tts_hifigan")
            
            if self.use_gpu:
                self.tts_model = self.tts_model.cuda()
                self.vocoder = self.vocoder.cuda()
            
            logger.info("NeMo TTS models loaded")
            
        except Exception as e:
            logger.error(f"Error initializing NeMo TTS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        output_path: Optional[str] = None,
        speed: float = 1.0,
        emotion: Optional[str] = None,
        style: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Advanced text-to-speech synthesis
        
        Args:
            text: Input text
            language: Target language
            speaker: Speaker name (for multi-speaker models)
            speaker_wav: Reference audio for voice cloning
            output_path: Output file path
            speed: Speech speed multiplier
            emotion: Emotion style (if supported)
            style: Speaking style (if supported)
            
        Returns:
            Synthesis results
        """
        import time
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not text.strip():
                return {"error": "Empty text provided"}
            
            if language not in self.supported_languages:
                logger.warning(f"Language {language} not supported, using default")
                language = "en"
            
            # Prepare output path
            if output_path is None:
                output_path = self.temp_dir / f"synthesis_{hash(text)}.wav"
            else:
                output_path = Path(output_path)
            
            # Synthesize using appropriate method
            if COQUI_AVAILABLE and self.tts_model:
                audio = self._synthesize_coqui(
                    text, language, speaker, speaker_wav, speed, emotion
                )
            elif NEMO_AVAILABLE and self.tts_model:
                audio = self._synthesize_nemo(text, speed)
            else:
                raise RuntimeError("No TTS model available")
            
            # Save audio
            sf.write(output_path, audio, self.sample_rate)
            
            # Calculate metrics
            duration = len(audio) / self.sample_rate
            synthesis_time = time.time() - start_time
            rtf = synthesis_time / duration  # Real-time factor
            
            # Update statistics
            self._update_stats(duration, rtf)
            
            # Quality assessment
            quality = audio_quality_check(audio, self.sample_rate)
            
            result = {
                "success": True,
                "output_file": str(output_path),
                "duration": duration,
                "synthesis_time": synthesis_time,
                "rtf": rtf,
                "quality_score": quality["quality_score"],
                "language": language,
                "speaker": speaker,
                "model": self.model_name
            }
            
            logger.info(f"Synthesis completed - Duration: {duration:.2f}s, RTF: {rtf:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            return {"error": str(e)}
    
    def _synthesize_coqui(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        speaker_wav: Optional[str],
        speed: float,
        emotion: Optional[str]
    ) -> np.ndarray:
        """Synthesize using Coqui TTS"""
        try:
            synthesis_kwargs = {
                "text": text,
                "language": language if language in self.supported_languages else "en"
            }
            
            # Add speaker if specified
            if speaker and hasattr(self.tts_model, 'speakers') and speaker in self.tts_model.speakers:
                synthesis_kwargs["speaker"] = speaker
            
            # Add speaker reference for voice cloning
            if speaker_wav and Path(speaker_wav).exists():
                synthesis_kwargs["speaker_wav"] = speaker_wav
            
            # Synthesize
            audio = self.tts_model.tts(**synthesis_kwargs)
            
            # Convert to numpy array if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            # Apply speed modification
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in Coqui synthesis: {e}")
            raise
    
    def _synthesize_nemo(self, text: str, speed: float) -> np.ndarray:
        """Synthesize using NeMo models"""
        try:
            # Tokenize text
            parsed = self.tts_model.parse(text)
            
            # Generate mel-spectrogram
            with torch.no_grad():
                spectrogram = self.tts_model.generate_spectrogram(tokens=parsed)
                
                # Apply speed modification to spectrogram
                if speed != 1.0:
                    # Adjust spectrogram length for speed
                    new_length = int(spectrogram.shape[-1] / speed)
                    spectrogram = torch.nn.functional.interpolate(
                        spectrogram.unsqueeze(0), 
                        size=new_length, 
                        mode='linear'
                    ).squeeze(0)
                
                # Generate audio using vocoder
                audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            
            # Convert to numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy().flatten()
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in NeMo synthesis: {e}")
            raise
    
    def batch_synthesize(
        self,
        texts: List[str],
        language: str = "en",
        speakers: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch synthesis for multiple texts
        
        Args:
            texts: List of texts to synthesize
            language: Target language
            speakers: List of speakers (optional)
            output_dir: Output directory
            
        Returns:
            List of synthesis results
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.temp_dir
        
        results = []
        speakers = speakers or [None] * len(texts)
        
        for i, (text, speaker) in enumerate(zip(texts, speakers)):
            try:
                output_file = output_path / f"batch_synthesis_{i}.wav"
                
                result = self.synthesize(
                    text=text,
                    language=language,
                    speaker=speaker,
                    output_path=str(output_file)
                )
                
                result["batch_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch synthesis {i}: {e}")
                results.append({
                    "error": str(e),
                    "batch_index": i,
                    "text": text
                })
        
        return results
    
    def create_custom_voice(
        self,
        speaker_name: str,
        reference_audios: List[str],
        transcripts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create custom voice from reference audio
        
        Args:
            speaker_name: Name for the custom voice
            reference_audios: List of reference audio files
            transcripts: Corresponding transcripts (if available)
            
        Returns:
            Voice creation results
        """
        if not self.enable_voice_cloning:
            return {"error": "Voice cloning not enabled"}
        
        try:
            # Process each reference audio
            embeddings = []
            quality_scores = []
            
            for i, audio_path in enumerate(reference_audios):
                result = self.voice_cloner.create_voice_embedding(
                    audio_path, 
                    f"{speaker_name}_{i}"
                )
                
                if "error" not in result:
                    embedding = self.voice_cloner.get_speaker_embedding(f"{speaker_name}_{i}")
                    embeddings.append(embedding)
                    quality_scores.append(result["quality_score"])
            
            if not embeddings:
                return {"error": "No valid embeddings created"}
            
            # Average embeddings for better representation
            final_embedding = np.mean(embeddings, axis=0)
            avg_quality = np.mean(quality_scores)
            
            # Store final embedding
            self.voice_cloner.reference_embeddings[speaker_name] = {
                "embedding": final_embedding,
                "quality_score": avg_quality,
                "num_references": len(embeddings),
                "reference_audios": reference_audios
            }
            
            return {
                "speaker_name": speaker_name,
                "voice_created": True,
                "num_references": len(embeddings),
                "avg_quality": avg_quality
            }
            
        except Exception as e:
            logger.error(f"Error creating custom voice: {e}")
            return {"error": str(e)}
    
    def fine_tune_voice(
        self,
        speaker_name: str,
        training_data: List[Tuple[str, str]],  # (audio_path, transcript)
        num_epochs: int = 100,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Fine-tune TTS model for specific speaker
        
        Args:
            speaker_name: Speaker name
            training_data: List of (audio_path, transcript) tuples
            num_epochs: Training epochs
            learning_rate: Learning rate
            
        Returns:
            Fine-tuning results
        """
        # This would implement actual fine-tuning
        # For now, return placeholder
        return {
            "message": "Fine-tuning not implemented in this version",
            "speaker_name": speaker_name,
            "training_samples": len(training_data)
        }
    
    def _update_stats(self, duration: float, rtf: float):
        """Update generation statistics"""
        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_duration"] += duration
        
        # Update average RTF
        n = self.generation_stats["total_generations"]
        current_avg = self.generation_stats["avg_rtf"]
        self.generation_stats["avg_rtf"] = (current_avg * (n - 1) + rtf) / n
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "use_gpu": self.use_gpu,
            "voice_cloning_enabled": self.enable_voice_cloning,
            "supported_languages": self.supported_languages,
            "sample_rate": self.sample_rate,
            "generation_stats": self.generation_stats
        }
        
        if self.voice_cloner:
            info["available_speakers"] = self.voice_cloner.list_speakers()
        
        return info
    
    def save_voice_models(self, output_dir: str):
        """Save voice cloning models and embeddings"""
        if not self.enable_voice_cloning or not self.voice_cloner:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = output_path / "voice_embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump(self.voice_cloner.reference_embeddings, f)
        
        logger.info(f"Voice models saved to {output_dir}")
    
    def load_voice_models(self, model_dir: str):
        """Load voice cloning models and embeddings"""
        if not self.enable_voice_cloning:
            self.voice_cloner = VoiceCloner()
        
        model_path = Path(model_dir)
        
        # Load embeddings
        embeddings_file = model_path / "voice_embeddings.pkl"
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                self.voice_cloner.reference_embeddings = pickle.load(f)
            
            logger.info(f"Voice models loaded from {model_dir}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in self.temp_dir.glob("*.wav"):
                file.unlink()
            logger.info("Cleaned up temporary TTS files")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")


def create_advanced_tts(
    model_name: str = None,
    enable_voice_cloning: bool = False,
    use_gpu: bool = True
) -> AdvancedTTS:
    """
    Factory function to create Advanced TTS system
    
    Args:
        model_name: TTS model name
        enable_voice_cloning: Enable voice cloning
        use_gpu: Use GPU acceleration
        
    Returns:
        AdvancedTTS instance
    """
    return AdvancedTTS(
        model_name=model_name,
        use_gpu=use_gpu,
        enable_voice_cloning=enable_voice_cloning
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create advanced TTS system
        tts = create_advanced_tts(enable_voice_cloning=True)
        
        print("Advanced TTS Model Info:")
        print(json.dumps(tts.get_model_info(), indent=2))
        
        # Example synthesis
        # result = tts.synthesize(
        #     text="Hello, this is an example of advanced text-to-speech synthesis.",
        #     language="en",
        #     speed=1.0,
        #     emotion="neutral"
        # )
        # print("Synthesis Result:", result)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Install Coqui TTS: pip install TTS")
        print("Or install NeMo: pip install nemo_toolkit[all]")
