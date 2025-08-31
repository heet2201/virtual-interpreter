# Virtual Interpreter System - Case Study Implementation

## Overview

This repository contains a comprehensive implementation of an AI-powered, real-time interpretation system that combines Automatic Speech Recognition (ASR), Machine Translation (MT), and Text-to-Speech (TTS) to enable seamless multilingual communication.

## System Architecture

The Virtual Interpreter system is designed with both **basic** and **advanced** implementations:

- **Basic Implementation**: Uses existing open-source models with minimal fine-tuning
- **Advanced Implementation**: Includes fine-tuning capabilities, domain adaptation, and voice cloning

### Core Components

1. **Automatic Speech Recognition (ASR)**
   - Basic: OpenAI Whisper models
   - Advanced: Fine-tunable Whisper with confidence scoring

2. **Machine Translation (MT)**
   - Basic: NLLB-200 models from Meta
   - Advanced: Fine-tunable NLLB with domain adaptation

3. **Text-to-Speech (TTS)**
   - Basic: Google TTS (gTTS) and pyttsx3
   - Advanced: Coqui TTS with voice cloning capabilities

## Supported Languages

The system supports **12 high-resource languages** with bidirectional translation:

| Language | Code | ASR | MT | TTS |
|----------|------|-----|----|----|
| English  | en   | ✓   | ✓  | ✓   |
| Spanish  | es   | ✓   | ✓  | ✓   |
| French   | fr   | ✓   | ✓  | ✓   |
| German   | de   | ✓   | ✓  | ✓   |
| Italian  | it   | ✓   | ✓  | ✓   |
| Portuguese | pt | ✓   | ✓  | ✓   |
| Dutch    | nl   | ✓   | ✓  | ✓   |
| Russian  | ru   | ✓   | ✓  | ✓   |
| Chinese  | zh   | ✓   | ✓  | ✓   |
| Japanese | ja   | ✓   | ✓  | ✓   |
| Korean   | ko   | ✓   | ✓  | ✓   |
| Arabic   | ar   | ✓   | ✓  | ✓   |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for models

### Basic Installation

```bash
# Clone repository
git clone <repository_url>
cd virtual_interpreter

# Install dependencies
pip install -r requirements.txt

# For advanced TTS features
pip install TTS

# For GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Downloads

Models will be automatically downloaded on first use. To pre-download:

```python
from virtual_interpreter.basic import create_interpreter_pipeline

# This will download necessary models
pipeline = create_interpreter_pipeline()
```

## Quick Start

### Basic Usage

```python
from virtual_interpreter.basic import create_interpreter_pipeline

# Create pipeline
interpreter = create_interpreter_pipeline(
    asr_model_size="base",
    tts_engine="gtts"
)

# Interpret audio file
result = interpreter.interpret(
    audio_input="input_audio.wav",
    source_language="en",
    target_language="es",
    output_audio_file="output_audio.mp3"
)

print(f"Transcription: {result['transcription']}")
print(f"Translation: {result['translation']}")
print(f"Processing time: {result['processing_time']['total']:.2f}s")
```

### Advanced Usage

```python
from virtual_interpreter.advanced import create_advanced_asr, create_advanced_translator

# Create advanced components
asr = create_advanced_asr(enable_fine_tuning=True)
translator = create_advanced_translator(enable_domain_adaptation=True)

# Transcribe with advanced options
result = asr.transcribe(
    "audio.wav",
    language="en",
    beam_size=5,
    return_probabilities=True
)

# Translate with domain specification
translation = translator.translate(
    result["text"],
    source_lang="en",
    target_lang="es",
    domain="medical",
    num_beams=5
)
```

## Training and Fine-tuning

### ASR Fine-tuning

```python
from virtual_interpreter.advanced.asr_advanced import create_advanced_asr

# Create ASR system
asr = create_advanced_asr(enable_fine_tuning=True)

# Prepare training data
train_dataset, val_dataset = asr.prepare_training_data(
    data_path="training_data/",
    validation_split=0.1
)

# Fine-tune model
results = asr.fine_tune(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="./fine_tuned_asr",
    num_epochs=10,
    learning_rate=1e-4
)
```

### Translation Fine-tuning

```python
from virtual_interpreter.advanced.mt_advanced import create_advanced_translator

# Create translator
translator = create_advanced_translator(enable_domain_adaptation=True)

# Prepare domain-specific data
train_dataset, val_dataset = translator.prepare_training_data(
    data_path="translation_data.jsonl",
    domains=["medical", "legal", "technical"],
    validation_split=0.1
)

# Fine-tune with domain adaptation
results = translator.fine_tune(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir="./fine_tuned_mt",
    use_domain_adaptation=True,
    num_epochs=5
)
```

## Performance Metrics

### Quality Thresholds

- **ASR Confidence**: > 0.7 (recommended)
- **Translation BLEU**: > 0.25 (acceptable)
- **Audio Quality**: > 0.8 (good)

### Real-time Performance

| Component | Processing Time (RTF) | Memory Usage |
|-----------|----------------------|--------------|
| ASR Basic | 0.1-0.3x             | 1-2GB        |
| ASR Advanced | 0.2-0.5x           | 2-4GB        |
| MT Basic  | 0.05-0.1x            | 2-3GB        |
| MT Advanced | 0.1-0.2x           | 4-6GB        |
| TTS Basic | 0.1-0.3x             | 0.5-1GB      |
| TTS Advanced | 0.2-0.8x           | 2-4GB        |

*RTF: Real-Time Factor (< 1.0 means faster than real-time)*

## Evaluation

### Automatic Evaluation

```python
from virtual_interpreter.utils.evaluation import PipelineEvaluator

evaluator = PipelineEvaluator()

# Evaluate end-to-end performance
evaluation = evaluator.evaluate_end_to_end(
    input_audio="test_audio.wav",
    reference_transcript="Hello, how are you?",
    reference_translation="Hola, ¿cómo estás?",
    pipeline_result=result
)

print(f"Overall Score: {evaluation['overall_score']:.3f}")
```

### Metrics

- **ASR**: Word Error Rate (WER), Character Error Rate (CER)
- **MT**: BLEU score, METEOR score, Semantic similarity
- **TTS**: Naturalness, Intelligibility, Audio quality
- **Pipeline**: End-to-end latency, Overall quality score

## Configuration

### Model Configuration

```python
# configs/config.py

ASR_CONFIG = {
    "basic": {"model_name": "openai/whisper-base"},
    "advanced": {"model_name": "openai/whisper-large-v3"}
}

TRANSLATION_CONFIG = {
    "basic": {"model_name": "facebook/nllb-200-distilled-600M"},
    "advanced": {"model_name": "facebook/nllb-200-3.3B"}
}
```

### Audio Settings

```python
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "max_duration": 30  # seconds
}
```

## API Documentation

### REST API (Optional)

```python
from fastapi import FastAPI
from virtual_interpreter.api import create_app

app = create_app()

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /interpret` - Full interpretation pipeline
- `POST /transcribe` - ASR only
- `POST /translate` - MT only  
- `POST /synthesize` - TTS only

## Directory Structure

```
virtual_interpreter/
├── basic/                 # Basic implementations
│   ├── asr_basic.py
│   ├── mt_basic.py
│   ├── tts_basic.py
│   └── pipeline_basic.py
├── advanced/              # Advanced implementations
│   ├── asr_advanced.py
│   ├── mt_advanced.py
│   ├── tts_advanced.py
│   └── pipeline_advanced.py
├── configs/               # Configuration files
│   └── config.py
├── utils/                 # Utility modules
│   ├── audio_utils.py
│   └── evaluation.py
├── docs/                  # Documentation
├── tests/                 # Test files
└── models/                # Stored models
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific component tests
python -m pytest tests/test_asr.py
python -m pytest tests/test_mt.py
python -m pytest tests/test_tts.py

# Performance benchmarks
python -m pytest tests/test_performance.py
```

## Limitations and Known Issues

1. **Real-time Performance**: Advanced models may not achieve real-time performance on CPU
2. **Memory Usage**: Large models require significant GPU memory (8GB+)
3. **Language Support**: Quality varies across languages
4. **Domain Adaptation**: Requires domain-specific training data
5. **Voice Cloning**: Needs high-quality reference audio (10+ seconds)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{virtual_interpreter_2024,
  title={Virtual Interpreter: AI-Powered Real-time Multilingual Communication},
  author={Case Study Implementation},
  year={2024}
}
```

## Support

For questions and issues:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the example notebooks in `examples/`
