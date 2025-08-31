"""
Configuration settings for Virtual Interpreter System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Supported languages (high-resource languages)
SUPPORTED_LANGUAGES = {
    "english": {"code": "en", "whisper": "en", "tts": "en"},
    "spanish": {"code": "es", "whisper": "es", "tts": "es"},
    "french": {"code": "fr", "whisper": "fr", "tts": "fr"},
    "german": {"code": "de", "whisper": "de", "tts": "de"},
    "italian": {"code": "it", "whisper": "it", "tts": "it"},
    "portuguese": {"code": "pt", "whisper": "pt", "tts": "pt"},
    "dutch": {"code": "nl", "whisper": "nl", "tts": "nl"},
    "russian": {"code": "ru", "whisper": "ru", "tts": "ru"},
    "chinese": {"code": "zh", "whisper": "zh", "tts": "zh"},
    "japanese": {"code": "ja", "whisper": "ja", "tts": "ja"},
    "korean": {"code": "ko", "whisper": "ko", "tts": "ko"},
    "arabic": {"code": "ar", "whisper": "ar", "tts": "ar"},
}

# Model configurations
ASR_CONFIG = {
    "basic": {
        "model_name": "openai/whisper-base",
        "model_size": "base",  # tiny, base, small, medium, large
    },
    "advanced": {
        "model_name": "openai/whisper-large-v3",
        "model_size": "large-v3",
    }
}

TRANSLATION_CONFIG = {
    "basic": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "max_length": 512,
        "num_beams": 4,
    },
    "advanced": {
        "model_name": "facebook/nllb-200-3.3B",
        "max_length": 512,
        "num_beams": 4,
    }
}

TTS_CONFIG = {
    "basic": {
        "engine": "gtts",  # Google Text-to-Speech
        "slow": False,
    },
    "advanced": {
        "engine": "coqui",  # Coqui TTS
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
    }
}

# Audio settings
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "format": "wav",
    "channels": 1,
    "max_duration": 30,  # seconds
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "timeout": 300,
}

# Training settings
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "gradient_accumulation_steps": 2,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
}

# GPU settings
GPU_CONFIG = {
    "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
    "mixed_precision": True,
    "compile": False,  # PyTorch 2.0 compile
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "interpreter.log",
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "asr_confidence": 0.7,
    "translation_bleu": 0.25,
    "audio_quality": 0.8,
}
