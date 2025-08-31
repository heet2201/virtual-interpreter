"""
Virtual Interpreter System

A comprehensive AI-powered real-time interpretation system combining
Automatic Speech Recognition (ASR), Machine Translation (MT), and 
Text-to-Speech (TTS) for seamless multilingual communication.

Usage:
    from virtual_interpreter.basic import create_interpreter_pipeline
    
    # Create basic pipeline
    interpreter = create_interpreter_pipeline()
    
    # Interpret audio file
    result = interpreter.interpret(
        audio_input="input.wav",
        source_language="en", 
        target_language="es",
        output_audio_file="output.mp3"
    )

Author: Case Study Implementation
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Virtual Interpreter Case Study"
__license__ = "MIT"

# Import main components for easy access
from .basic.pipeline_basic import create_interpreter_pipeline
from .basic.asr_basic import create_asr_system
from .basic.mt_basic import create_translator
from .basic.tts_basic import create_tts_system

# Advanced components (optional imports)
try:
    from .advanced.asr_advanced import create_advanced_asr
    from .advanced.mt_advanced import create_advanced_translator
    from .advanced.tts_advanced import create_advanced_tts
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

# Configuration
from .configs.config import SUPPORTED_LANGUAGES

__all__ = [
    # Basic components
    "create_interpreter_pipeline",
    "create_asr_system", 
    "create_translator",
    "create_tts_system",
    # Configuration
    "SUPPORTED_LANGUAGES",
    # Version info
    "__version__",
    "__author__",
    "__license__",
]

# Add advanced components if available
if ADVANCED_AVAILABLE:
    __all__.extend([
        "create_advanced_asr",
        "create_advanced_translator", 
        "create_advanced_tts"
    ])

def get_info():
    """Get system information"""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "advanced_features_available": ADVANCED_AVAILABLE
    }
