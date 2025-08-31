"""
Basic End-to-End Virtual Interpreter Pipeline
Integrates ASR, Machine Translation, and TTS
"""
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import tempfile
import json

from .asr_basic import BasicASR
from .mt_basic import BasicTranslator
from .tts_basic import BasicTTS
from ..configs.config import SUPPORTED_LANGUAGES, QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)


class VirtualInterpreterPipeline:
    """
    Basic Virtual Interpreter Pipeline combining ASR, MT, and TTS
    """
    
    def __init__(
        self, 
        asr_model_size: str = "base",
        translation_model: str = None,
        tts_engine: str = "gtts",
        device: Optional[str] = None
    ):
        """
        Initialize the Virtual Interpreter Pipeline
        
        Args:
            asr_model_size: Whisper model size for ASR
            translation_model: HuggingFace model for translation
            tts_engine: TTS engine to use
            device: Device for computations
        """
        self.device = device
        self.asr_model_size = asr_model_size
        self.translation_model = translation_model
        self.tts_engine = tts_engine
        
        # Initialize components
        self.asr = None
        self.translator = None
        self.tts = None
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_processing_time": 0.0,
            "component_times": {"asr": 0.0, "mt": 0.0, "tts": 0.0}
        }
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing Virtual Interpreter Pipeline...")
            
            # Initialize ASR
            logger.info("Loading ASR model...")
            self.asr = BasicASR(model_size=self.asr_model_size, device=self.device)
            
            # Initialize Translator
            logger.info("Loading Translation model...")
            self.translator = BasicTranslator(model_name=self.translation_model, device=self.device)
            
            # Initialize TTS
            logger.info("Loading TTS engine...")
            self.tts = BasicTTS(engine=self.tts_engine)
            
            logger.info("Pipeline initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def interpret(
        self, 
        audio_input, 
        source_language: str,
        target_language: str,
        output_audio_file: Optional[str] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Complete interpretation pipeline: Audio -> Text -> Translation -> Audio
        
        Args:
            audio_input: Input audio (file path or numpy array)
            source_language: Source language code
            target_language: Target language code
            output_audio_file: Path for output audio file
            return_intermediates: Whether to return intermediate results
            
        Returns:
            Dictionary containing interpretation results
        """
        start_time = time.time()
        
        try:
            # Validate languages
            if not self._validate_languages(source_language, target_language):
                return {"error": f"Unsupported language pair: {source_language} -> {target_language}"}
            
            result = {
                "source_language": source_language,
                "target_language": target_language,
                "success": True,
                "processing_time": {}
            }
            
            # Step 1: Speech-to-Text (ASR)
            asr_start = time.time()
            logger.info(f"Step 1: Transcribing audio ({source_language})")
            
            asr_result = self.asr.transcribe(audio_input, language=source_language)
            
            if not asr_result.get("text"):
                return {"error": "ASR failed - no text extracted", "success": False}
            
            asr_time = time.time() - asr_start
            result["processing_time"]["asr"] = asr_time
            result["transcription"] = asr_result["text"]
            result["asr_confidence"] = asr_result.get("confidence", 0.0)
            
            logger.info(f"ASR completed in {asr_time:.2f}s: {asr_result['text'][:100]}...")
            
            # Check ASR quality
            if result["asr_confidence"] < QUALITY_THRESHOLDS["asr_confidence"]:
                logger.warning(f"Low ASR confidence: {result['asr_confidence']:.2f}")
            
            # Step 2: Machine Translation
            mt_start = time.time()
            logger.info(f"Step 2: Translating text ({source_language} -> {target_language})")
            
            # Skip translation if source and target languages are the same
            if source_language == target_language:
                translation_result = {
                    "translation": asr_result["text"],
                    "confidence": 1.0
                }
                mt_time = 0.001  # Minimal time for no-op
            else:
                translation_result = self.translator.translate(
                    asr_result["text"],
                    source_language,
                    target_language
                )
                mt_time = time.time() - mt_start
            
            result["processing_time"]["mt"] = mt_time
            result["translation"] = translation_result.get("translation", "")
            result["translation_confidence"] = translation_result.get("confidence", 0.0)
            
            logger.info(f"Translation completed in {mt_time:.2f}s: {result['translation'][:100]}...")
            
            # Step 3: Text-to-Speech (TTS)
            tts_start = time.time()
            logger.info(f"Step 3: Synthesizing speech ({target_language})")
            
            tts_result = self.tts.text_to_speech(
                result["translation"],
                target_language,
                output_file=output_audio_file,
                play_audio=False
            )
            
            tts_time = time.time() - tts_start
            result["processing_time"]["tts"] = tts_time
            
            if "error" in tts_result:
                logger.error(f"TTS failed: {tts_result['error']}")
                result["tts_error"] = tts_result["error"]
            else:
                result["output_audio_file"] = tts_result.get("output_file")
            
            logger.info(f"TTS completed in {tts_time:.2f}s")
            
            # Calculate total processing time
            total_time = time.time() - start_time
            result["processing_time"]["total"] = total_time
            
            # Update performance statistics
            self._update_performance_stats(total_time, asr_time, mt_time, tts_time, success=True)
            
            # Add intermediate results if requested
            if return_intermediates:
                result["intermediates"] = {
                    "asr_result": asr_result,
                    "translation_result": translation_result,
                    "tts_result": tts_result
                }
            
            logger.info(f"Complete interpretation pipeline finished in {total_time:.2f}s")
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            self._update_performance_stats(error_time, 0, 0, 0, success=False)
            
            logger.error(f"Pipeline error: {e}")
            return {
                "error": str(e),
                "success": False,
                "processing_time": {"total": error_time}
            }
    
    def interpret_realtime_chunk(
        self, 
        audio_chunk: np.ndarray,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """
        Process audio chunk for real-time interpretation
        
        Args:
            audio_chunk: Audio chunk as numpy array
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Dictionary containing chunk interpretation results
        """
        try:
            # Real-time ASR
            transcription = self.asr.transcribe_realtime(audio_chunk)
            
            if not transcription.strip():
                return {"text": "", "translation": "", "confidence": 0.0}
            
            # Real-time translation
            if source_language != target_language:
                translation_result = self.translator.translate(
                    transcription, source_language, target_language
                )
                translation = translation_result.get("translation", "")
                confidence = translation_result.get("confidence", 0.0)
            else:
                translation = transcription
                confidence = 1.0
            
            return {
                "text": transcription,
                "translation": translation,
                "confidence": confidence,
                "source_language": source_language,
                "target_language": target_language
            }
            
        except Exception as e:
            logger.error(f"Real-time chunk processing error: {e}")
            return {"error": str(e)}
    
    def batch_interpret(
        self, 
        audio_files: List[str],
        language_pairs: List[Tuple[str, str]],
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple audio files with different language pairs
        
        Args:
            audio_files: List of audio file paths
            language_pairs: List of (source, target) language tuples
            output_dir: Directory for output files
            
        Returns:
            List of interpretation results
        """
        if len(audio_files) != len(language_pairs):
            raise ValueError("Number of audio files must match number of language pairs")
        
        results = []
        output_path = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        output_path.mkdir(exist_ok=True)
        
        for i, (audio_file, (src_lang, tgt_lang)) in enumerate(zip(audio_files, language_pairs)):
            try:
                output_file = output_path / f"interpreted_{i}_{src_lang}_to_{tgt_lang}.mp3"
                
                result = self.interpret(
                    audio_file,
                    src_lang,
                    tgt_lang,
                    output_audio_file=str(output_file)
                )
                
                result["input_file"] = audio_file
                result["batch_index"] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                results.append({
                    "error": str(e),
                    "input_file": audio_file,
                    "batch_index": i,
                    "success": False
                })
        
        return results
    
    def get_quality_assessment(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of interpretation results
        
        Args:
            result: Interpretation result dictionary
            
        Returns:
            Quality assessment
        """
        assessment = {
            "overall_quality": "unknown",
            "asr_quality": "unknown",
            "translation_quality": "unknown",
            "recommendations": []
        }
        
        try:
            # ASR quality assessment
            asr_confidence = result.get("asr_confidence", 0.0)
            if asr_confidence >= 0.8:
                assessment["asr_quality"] = "high"
            elif asr_confidence >= 0.6:
                assessment["asr_quality"] = "medium"
            else:
                assessment["asr_quality"] = "low"
                assessment["recommendations"].append("Consider using higher quality audio or different ASR model")
            
            # Translation quality assessment
            translation_confidence = result.get("translation_confidence", 0.0)
            if translation_confidence >= 0.7:
                assessment["translation_quality"] = "high"
            elif translation_confidence >= 0.5:
                assessment["translation_quality"] = "medium"
            else:
                assessment["translation_quality"] = "low"
                assessment["recommendations"].append("Translation quality is low, consider manual review")
            
            # Overall quality
            if assessment["asr_quality"] == "high" and assessment["translation_quality"] == "high":
                assessment["overall_quality"] = "high"
            elif assessment["asr_quality"] == "low" or assessment["translation_quality"] == "low":
                assessment["overall_quality"] = "low"
            else:
                assessment["overall_quality"] = "medium"
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
        
        return assessment
    
    def _validate_languages(self, source_lang: str, target_lang: str) -> bool:
        """Validate that language pair is supported"""
        supported = list(SUPPORTED_LANGUAGES.keys())
        supported_codes = [lang["code"] for lang in SUPPORTED_LANGUAGES.values()]
        
        return (source_lang in supported_codes and target_lang in supported_codes)
    
    def _update_performance_stats(
        self, 
        total_time: float, 
        asr_time: float, 
        mt_time: float, 
        tts_time: float,
        success: bool
    ):
        """Update performance statistics"""
        self.performance_stats["total_requests"] += 1
        if success:
            self.performance_stats["successful_requests"] += 1
        
        # Update average processing time
        current_avg = self.performance_stats["avg_processing_time"]
        n = self.performance_stats["total_requests"]
        self.performance_stats["avg_processing_time"] = (current_avg * (n-1) + total_time) / n
        
        # Update component times
        self.performance_stats["component_times"]["asr"] = (
            self.performance_stats["component_times"]["asr"] * (n-1) + asr_time
        ) / n
        self.performance_stats["component_times"]["mt"] = (
            self.performance_stats["component_times"]["mt"] * (n-1) + mt_time
        ) / n
        self.performance_stats["component_times"]["tts"] = (
            self.performance_stats["component_times"]["tts"] * (n-1) + tts_time
        ) / n
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration information"""
        return {
            "asr_model": self.asr_model_size,
            "translation_model": self.translation_model or "default",
            "tts_engine": self.tts_engine,
            "device": self.device,
            "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
            "performance_stats": self.get_performance_stats()
        }
    
    def save_session_log(self, filepath: str):
        """Save session performance log"""
        try:
            log_data = {
                "pipeline_info": self.get_pipeline_info(),
                "performance_stats": self.get_performance_stats(),
                "timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
                
            logger.info(f"Session log saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving session log: {e}")


def create_interpreter_pipeline(
    asr_model_size: str = "base",
    translation_model: str = None,
    tts_engine: str = "gtts",
    device: Optional[str] = None
) -> VirtualInterpreterPipeline:
    """
    Factory function to create Virtual Interpreter Pipeline
    
    Args:
        asr_model_size: Whisper model size
        translation_model: Translation model name
        tts_engine: TTS engine
        device: Computation device
        
    Returns:
        VirtualInterpreterPipeline instance
    """
    return VirtualInterpreterPipeline(
        asr_model_size=asr_model_size,
        translation_model=translation_model,
        tts_engine=tts_engine,
        device=device
    )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create interpreter pipeline
        interpreter = create_interpreter_pipeline(
            asr_model_size="base",
            tts_engine="gtts"
        )
        
        # Display pipeline info
        print("Pipeline Info:")
        print(json.dumps(interpreter.get_pipeline_info(), indent=2))
        
        # Example interpretation (uncomment to test with actual audio)
        # result = interpreter.interpret(
        #     "path/to/audio/file.wav",
        #     source_language="en",
        #     target_language="es",
        #     output_audio_file="output.mp3"
        # )
        # 
        # print("\nInterpretation Result:")
        # print(json.dumps(result, indent=2))
        # 
        # # Quality assessment
        # quality = interpreter.get_quality_assessment(result)
        # print("\nQuality Assessment:")
        # print(json.dumps(quality, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed and models are available")
