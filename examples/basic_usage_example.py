#!/usr/bin/env python3
"""
Basic Usage Example for Virtual Interpreter System

This example demonstrates how to use the basic Virtual Interpreter pipeline
for end-to-end speech-to-speech translation.
"""

import logging
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from basic.pipeline_basic import create_interpreter_pipeline
from utils.audio_utils import audio_quality_check

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main example function"""
    
    print("=" * 60)
    print("Virtual Interpreter - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create the interpreter pipeline
    print("\n1. Initializing Virtual Interpreter Pipeline...")
    
    try:
        interpreter = create_interpreter_pipeline(
            asr_model_size="base",  # Options: tiny, base, small, medium, large
            tts_engine="gtts"       # Options: gtts, pyttsx3
        )
        
        print("✅ Pipeline initialized successfully!")
        
        # Display pipeline information
        pipeline_info = interpreter.get_pipeline_info()
        print(f"   - ASR Model: {pipeline_info['asr_model']}")
        print(f"   - Translation Model: {pipeline_info['translation_model']}")
        print(f"   - TTS Engine: {pipeline_info['tts_engine']}")
        print(f"   - Device: {pipeline_info['device']}")
        print(f"   - Supported Languages: {len(pipeline_info['supported_languages'])}")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("\nMake sure you have installed all required dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    # Step 2: Example interpretation (without actual audio file)
    print("\n2. Example Interpretation Process...")
    
    # Since we don't have actual audio files, we'll demonstrate the workflow
    print("   📝 In a real scenario, you would:")
    print("   - Provide an audio file path: 'input_audio.wav'")
    print("   - Specify source language: 'en' (English)")
    print("   - Specify target language: 'es' (Spanish)")
    print("   - Optionally specify output path: 'output_audio.mp3'")
    
    # Example code (commented out as we don't have audio files)
    """
    # Uncomment and modify this section when you have audio files to test:
    
    result = interpreter.interpret(
        audio_input="path/to/input_audio.wav",    # Input audio file
        source_language="en",                      # Source language
        target_language="es",                      # Target language  
        output_audio_file="path/to/output.mp3",   # Output audio file
        return_intermediates=True                  # Return intermediate results
    )
    
    if result["success"]:
        print(f"✅ Interpretation completed!")
        print(f"   Original text: {result['transcription']}")
        print(f"   Translation: {result['translation']}")
        print(f"   Processing time: {result['processing_time']['total']:.2f}s")
        print(f"   Output audio: {result['output_audio_file']}")
        
        # Quality assessment
        quality = interpreter.get_quality_assessment(result)
        print(f"   Quality: {quality['overall_quality']}")
        
    else:
        print(f"❌ Interpretation failed: {result.get('error', 'Unknown error')}")
    """
    
    # Step 3: Show supported language pairs
    print("\n3. Supported Language Pairs...")
    
    supported_languages = [
        ("English", "en"), ("Spanish", "es"), ("French", "fr"),
        ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
        ("Dutch", "nl"), ("Russian", "ru"), ("Chinese", "zh"),
        ("Japanese", "ja"), ("Korean", "ko"), ("Arabic", "ar")
    ]
    
    print("   The system supports bidirectional translation between:")
    for name, code in supported_languages:
        print(f"   - {name} ({code})")
    
    print(f"\n   Total possible translation pairs: {len(supported_languages) * (len(supported_languages) - 1)}")
    
    # Step 4: Performance characteristics
    print("\n4. Performance Characteristics...")
    
    performance_stats = interpreter.get_performance_stats()
    print(f"   - Total requests processed: {performance_stats['total_requests']}")
    print(f"   - Success rate: {performance_stats['success_rate']:.1%}")
    print(f"   - Average processing time: {performance_stats['avg_processing_time']:.2f}s")
    
    component_times = performance_stats['component_times']
    print("   - Component breakdown:")
    print(f"     • ASR: {component_times['asr']:.2f}s")
    print(f"     • Translation: {component_times['mt']:.2f}s") 
    print(f"     • TTS: {component_times['tts']:.2f}s")
    
    # Step 5: Real-time usage example
    print("\n5. Real-time Processing Example...")
    
    print("   For real-time audio processing, you would typically:")
    print("   - Capture audio in chunks (e.g., 1-2 second segments)")
    print("   - Process each chunk with interpret_realtime_chunk()")
    print("   - Buffer and combine results for coherent output")
    
    # Example real-time code (commented out)
    """
    import numpy as np
    
    # Simulate audio chunk (replace with actual audio capture)
    audio_chunk = np.random.randn(16000)  # 1 second at 16kHz
    
    realtime_result = interpreter.interpret_realtime_chunk(
        audio_chunk=audio_chunk,
        source_language="en",
        target_language="es"
    )
    
    if "error" not in realtime_result:
        print(f"Real-time result: {realtime_result['translation']}")
    """
    
    # Step 6: Batch processing example
    print("\n6. Batch Processing Example...")
    
    print("   For processing multiple files:")
    """
    audio_files = [
        "audio1.wav", "audio2.wav", "audio3.wav"
    ]
    
    language_pairs = [
        ("en", "es"),  # English to Spanish
        ("fr", "en"),  # French to English  
        ("de", "it")   # German to Italian
    ]
    
    batch_results = interpreter.batch_interpret(
        audio_files=audio_files,
        language_pairs=language_pairs,
        output_dir="./batch_output"
    )
    
    for i, result in enumerate(batch_results):
        if result["success"]:
            print(f"File {i+1}: {result['translation']}")
        else:
            print(f"File {i+1}: Error - {result['error']}")
    """
    
    # Step 7: Tips for best results
    print("\n7. Tips for Best Results...")
    
    print("   Audio Quality:")
    print("   - Use clear, high-quality audio (16kHz or higher)")
    print("   - Minimize background noise")
    print("   - Ensure speakers are close to microphone")
    print("   - Keep audio segments under 30 seconds for optimal processing")
    
    print("\n   Language Considerations:")
    print("   - Specify source language when known (improves accuracy)")
    print("   - For best results, use high-resource language pairs")
    print("   - Consider domain-specific terminology")
    
    print("\n   Performance Optimization:")
    print("   - Use GPU acceleration when available")
    print("   - Consider using smaller models for faster processing")
    print("   - Implement caching for repeated translations")
    
    # Step 8: Troubleshooting
    print("\n8. Common Issues and Solutions...")
    
    print("   If you encounter issues:")
    print("   - Check that all dependencies are installed")
    print("   - Ensure sufficient memory (8GB+ recommended)")
    print("   - Verify audio file formats are supported (WAV, MP3, M4A)")
    print("   - Check internet connection (required for model downloads)")
    print("   - Monitor GPU memory usage if using CUDA")
    
    print("\n" + "=" * 60)
    print("Example completed! Ready to process your audio files.")
    print("=" * 60)
    
    # Save session log
    try:
        interpreter.save_session_log("session_log.json")
        print("📋 Session log saved to session_log.json")
    except Exception as e:
        print(f"Note: Could not save session log: {e}")


def demonstrate_component_usage():
    """Demonstrate individual component usage"""
    
    print("\n" + "=" * 60)
    print("Individual Component Usage Examples")
    print("=" * 60)
    
    # ASR Example
    print("\n🎤 ASR (Speech Recognition) Example:")
    """
    from basic.asr_basic import create_asr_system
    
    asr = create_asr_system("base")
    
    # Transcribe audio file
    result = asr.transcribe("audio_file.wav", language="en")
    print(f"Transcript: {result['text']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Language: {result['language']}")
    """
    
    # Translation Example  
    print("\n🌍 Translation Example:")
    """
    from basic.mt_basic import create_translator
    
    translator = create_translator()
    
    # Translate text
    result = translator.translate(
        text="Hello, how are you today?",
        source_lang="en", 
        target_lang="es"
    )
    print(f"Translation: {result['translation']}")
    print(f"Confidence: {result['confidence']}")
    """
    
    # TTS Example
    print("\n🔊 TTS (Speech Synthesis) Example:")
    """
    from basic.tts_basic import create_tts_system
    
    tts = create_tts_system("gtts")
    
    # Synthesize speech
    result = tts.text_to_speech(
        text="Hola, ¿cómo estás hoy?",
        language="es",
        output_file="output_speech.mp3"
    )
    print(f"Audio file: {result['output_file']}")
    """


if __name__ == "__main__":
    try:
        main()
        demonstrate_component_usage()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and try again.")
