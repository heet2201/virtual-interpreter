"""
Basic Text-to-Speech (TTS) implementation using gTTS and pyttsx3
"""
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import io

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..configs.config import TTS_CONFIG, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class BasicTTS:
    """Basic Text-to-Speech implementation using gTTS and pyttsx3"""
    
    def __init__(self, engine: str = "gtts", voice_speed: float = 1.0):
        """
        Initialize the Basic TTS system
        
        Args:
            engine: TTS engine to use ('gtts' or 'pyttsx3')
            voice_speed: Speech rate (1.0 = normal speed)
        """
        self.engine_name = engine
        self.voice_speed = voice_speed
        self.engine = None
        self.temp_dir = Path(tempfile.gettempdir()) / "virtual_interpreter_tts"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Language mapping for gTTS
        self.gtts_lang_map = {
            "en": "en",
            "es": "es",
            "fr": "fr", 
            "de": "de",
            "it": "it",
            "pt": "pt",
            "nl": "nl",
            "ru": "ru",
            "zh": "zh",
            "ja": "ja",
            "ko": "ko",
            "ar": "ar",
        }
        
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize the selected TTS engine"""
        try:
            if self.engine_name == "gtts":
                if not GTTS_AVAILABLE:
                    raise ImportError("gTTS not available. Install with: pip install gtts")
                logger.info("Using Google Text-to-Speech (gTTS)")
                
            elif self.engine_name == "pyttsx3":
                if not PYTTSX3_AVAILABLE:
                    raise ImportError("pyttsx3 not available. Install with: pip install pyttsx3")
                
                self.engine = pyttsx3.init()
                # Set speech rate
                rate = self.engine.getProperty('rate')
                self.engine.setProperty('rate', rate * self.voice_speed)
                logger.info("Using pyttsx3 TTS engine")
                
            else:
                raise ValueError(f"Unsupported TTS engine: {self.engine_name}")
                
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            raise
    
    def text_to_speech(
        self, 
        text: str, 
        language: str = "en", 
        output_file: Optional[str] = None,
        play_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Convert text to speech
        
        Args:
            text: Input text to convert
            language: Language code for TTS
            output_file: Path to save audio file (optional)
            play_audio: Whether to play audio immediately
            
        Returns:
            Dictionary containing TTS results
        """
        try:
            if not text.strip():
                return {"error": "Empty text provided"}
            
            # Validate language
            if language not in self.gtts_lang_map:
                logger.warning(f"Language {language} not supported, using English")
                language = "en"
            
            if self.engine_name == "gtts":
                result = self._gtts_synthesize(text, language, output_file, play_audio)
            elif self.engine_name == "pyttsx3":
                result = self._pyttsx3_synthesize(text, language, output_file, play_audio)
            else:
                raise ValueError(f"Unsupported engine: {self.engine_name}")
            
            logger.info(f"TTS synthesis completed for text: {text[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in text-to-speech synthesis: {e}")
            return {"error": str(e)}
    
    def _gtts_synthesize(
        self, 
        text: str, 
        language: str, 
        output_file: Optional[str] = None,
        play_audio: bool = False
    ) -> Dict[str, Any]:
        """Synthesize speech using gTTS"""
        try:
            # Create gTTS object
            tts = gTTS(
                text=text,
                lang=self.gtts_lang_map[language],
                slow=TTS_CONFIG["basic"]["slow"]
            )
            
            # Determine output file
            if output_file is None:
                output_file = self.temp_dir / f"tts_output_{hash(text)}.mp3"
            
            # Save audio file
            tts.save(str(output_file))
            
            result = {
                "success": True,
                "engine": "gtts",
                "language": language,
                "output_file": str(output_file),
                "text": text,
            }
            
            # Play audio if requested
            if play_audio:
                self._play_audio(output_file)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in gTTS synthesis: {e}")
            return {"error": str(e), "engine": "gtts"}
    
    def _pyttsx3_synthesize(
        self, 
        text: str, 
        language: str, 
        output_file: Optional[str] = None,
        play_audio: bool = False
    ) -> Dict[str, Any]:
        """Synthesize speech using pyttsx3"""
        try:
            # Set voice if available
            self._set_voice_by_language(language)
            
            if output_file:
                # Save to file
                self.engine.save_to_file(text, str(output_file))
                self.engine.runAndWait()
            
            if play_audio:
                # Speak directly
                self.engine.say(text)
                self.engine.runAndWait()
            
            result = {
                "success": True,
                "engine": "pyttsx3",
                "language": language,
                "output_file": str(output_file) if output_file else None,
                "text": text,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pyttsx3 synthesis: {e}")
            return {"error": str(e), "engine": "pyttsx3"}
    
    def _set_voice_by_language(self, language: str):
        """Set voice based on language (pyttsx3)"""
        try:
            voices = self.engine.getProperty('voices')
            
            # Simple language to voice mapping
            voice_patterns = {
                "en": ["english", "en"],
                "es": ["spanish", "es"],
                "fr": ["french", "fr"],
                "de": ["german", "de"],
                "it": ["italian", "it"],
                "pt": ["portuguese", "pt"],
            }
            
            if language in voice_patterns:
                patterns = voice_patterns[language]
                for voice in voices:
                    voice_name = voice.name.lower()
                    if any(pattern in voice_name for pattern in patterns):
                        self.engine.setProperty('voice', voice.id)
                        logger.info(f"Set voice to: {voice.name}")
                        return
            
            logger.info(f"No specific voice found for {language}, using default")
            
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
    
    def _play_audio(self, audio_file: str):
        """Play audio file using pygame or system command"""
        try:
            if PYGAME_AVAILABLE:
                pygame.mixer.init()
                pygame.mixer.music.load(str(audio_file))
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    
            else:
                # Fallback to system command
                import subprocess
                import platform
                
                system = platform.system()
                if system == "Darwin":  # macOS
                    subprocess.run(["afplay", str(audio_file)])
                elif system == "Linux":
                    subprocess.run(["aplay", str(audio_file)])
                elif system == "Windows":
                    os.startfile(str(audio_file))
                    
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def batch_synthesize(
        self, 
        texts: List[str], 
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Synthesize multiple texts
        
        Args:
            texts: List of texts to synthesize
            language: Language code
            
        Returns:
            List of synthesis results
        """
        results = []
        for i, text in enumerate(texts):
            try:
                output_file = self.temp_dir / f"batch_tts_{i}_{hash(text)}.mp3"
                result = self.text_to_speech(
                    text, 
                    language, 
                    output_file=str(output_file),
                    play_audio=False
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error synthesizing text {i}: {e}")
                results.append({"error": str(e), "text": text})
        
        return results
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices (pyttsx3 only)"""
        if self.engine_name != "pyttsx3" or not self.engine:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            return [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "language": getattr(voice, 'languages', ['unknown']),
                    "gender": getattr(voice, 'gender', 'unknown')
                }
                for voice in voices
            ]
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return []
    
    def set_voice_properties(self, rate: Optional[int] = None, volume: Optional[float] = None):
        """Set voice properties (pyttsx3 only)"""
        if self.engine_name != "pyttsx3" or not self.engine:
            return
        
        try:
            if rate is not None:
                self.engine.setProperty('rate', rate)
            if volume is not None:
                self.engine.setProperty('volume', volume)
        except Exception as e:
            logger.error(f"Error setting voice properties: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.gtts_lang_map.keys())
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files"""
        try:
            for file in self.temp_dir.glob("*.mp3"):
                file.unlink()
            for file in self.temp_dir.glob("*.wav"):
                file.unlink()
            logger.info("Cleaned up temporary TTS files")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get TTS engine information"""
        info = {
            "engine": self.engine_name,
            "supported_languages": self.get_supported_languages(),
            "voice_speed": self.voice_speed,
        }
        
        if self.engine_name == "pyttsx3" and self.engine:
            info["available_voices"] = len(self.get_available_voices())
        
        return info


class MultiEngineTTS:
    """TTS system that can fallback between engines"""
    
    def __init__(self):
        """Initialize multi-engine TTS"""
        self.engines = {}
        self.primary_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available TTS engines"""
        # Try to initialize gTTS
        try:
            self.engines["gtts"] = BasicTTS("gtts")
            self.primary_engine = "gtts"
            logger.info("gTTS engine available")
        except Exception as e:
            logger.warning(f"gTTS not available: {e}")
        
        # Try to initialize pyttsx3
        try:
            self.engines["pyttsx3"] = BasicTTS("pyttsx3")
            if not self.primary_engine:
                self.primary_engine = "pyttsx3"
            logger.info("pyttsx3 engine available")
        except Exception as e:
            logger.warning(f"pyttsx3 not available: {e}")
        
        if not self.engines:
            raise RuntimeError("No TTS engines available")
    
    def synthesize(self, text: str, language: str = "en", **kwargs) -> Dict[str, Any]:
        """Synthesize with fallback to different engines"""
        for engine_name, engine in self.engines.items():
            try:
                result = engine.text_to_speech(text, language, **kwargs)
                if "error" not in result:
                    return result
            except Exception as e:
                logger.warning(f"Engine {engine_name} failed: {e}")
        
        return {"error": "All TTS engines failed"}


def create_tts_system(engine: str = "gtts") -> BasicTTS:
    """
    Factory function to create BasicTTS system
    
    Args:
        engine: TTS engine to use
        
    Returns:
        BasicTTS instance
    """
    return BasicTTS(engine=engine)


def create_multi_engine_tts() -> MultiEngineTTS:
    """
    Factory function to create MultiEngineTTS
    
    Returns:
        MultiEngineTTS instance
    """
    return MultiEngineTTS()


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create TTS system
    try:
        tts = create_tts_system("gtts")
        
        # Display engine info
        print("Engine Info:", tts.get_engine_info())
        
        # Example synthesis
        result = tts.text_to_speech(
            "Hello, this is a test of the text-to-speech system.",
            language="en",
            play_audio=False  # Set to True to play audio
        )
        
        if "error" not in result:
            print("Synthesis successful!")
            print("Output file:", result.get("output_file"))
        else:
            print("Error:", result["error"])
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install required dependencies: pip install gtts pygame")
