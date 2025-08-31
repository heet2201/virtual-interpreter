"""
Audio processing utilities for the Virtual Interpreter
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import Tuple, Optional, List
import io

logger = logging.getLogger(__name__)


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio(audio_data: np.ndarray, file_path: str, sample_rate: int = 16000):
    """
    Save audio data to file
    
    Args:
        audio_data: Audio data as numpy array
        file_path: Output file path
        sample_rate: Sample rate
    """
    try:
        sf.write(file_path, audio_data, sample_rate)
        logger.info(f"Audio saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving audio to {file_path}: {e}")
        raise


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target level in dB
    
    Args:
        audio: Input audio
        target_level: Target level in dB
        
    Returns:
        Normalized audio
    """
    # Calculate current RMS level
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    
    # Calculate target RMS
    target_rms = 10 ** (target_level / 20)
    
    # Apply scaling
    scaling_factor = target_rms / rms
    return audio * scaling_factor


def remove_silence(
    audio: np.ndarray, 
    sr: int, 
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.1
) -> np.ndarray:
    """
    Remove silence from audio
    
    Args:
        audio: Input audio
        sr: Sample rate
        threshold_db: Silence threshold in dB
        min_silence_duration: Minimum silence duration to remove
        
    Returns:
        Audio with silence removed
    """
    try:
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio))
        
        # Find non-silent regions
        non_silent = audio_db > threshold_db
        
        # Remove short silence gaps
        min_samples = int(min_silence_duration * sr)
        
        # Simple approach: keep samples above threshold
        return audio[non_silent]
        
    except Exception as e:
        logger.error(f"Error removing silence: {e}")
        return audio


def split_audio_chunks(
    audio: np.ndarray, 
    sr: int, 
    chunk_duration: float = 30.0,
    overlap: float = 0.1
) -> List[np.ndarray]:
    """
    Split audio into chunks for processing
    
    Args:
        audio: Input audio
        sr: Sample rate
        chunk_duration: Chunk duration in seconds
        overlap: Overlap between chunks (fraction)
        
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * chunk_samples)
    step_samples = chunk_samples - overlap_samples
    
    chunks = []
    start = 0
    
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Pad if chunk is too short
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
        chunks.append(chunk)
        start += step_samples
    
    return chunks


def apply_noise_reduction(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply basic noise reduction using spectral subtraction
    
    Args:
        audio: Input audio
        sr: Sample rate
        
    Returns:
        Noise-reduced audio
    """
    try:
        # Simple spectral subtraction approach
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first few frames
        noise_frames = min(5, magnitude.shape[1])
        noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Subtract noise estimate
        alpha = 2.0  # Over-subtraction factor
        magnitude_denoised = magnitude - alpha * noise_estimate
        
        # Apply minimum floor
        magnitude_denoised = np.maximum(magnitude_denoised, 0.1 * magnitude)
        
        # Reconstruct audio
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised)
        
        return audio_denoised
        
    except Exception as e:
        logger.error(f"Error in noise reduction: {e}")
        return audio


def detect_voice_activity(
    audio: np.ndarray, 
    sr: int, 
    frame_length: int = 1024,
    hop_length: int = 512
) -> np.ndarray:
    """
    Simple voice activity detection
    
    Args:
        audio: Input audio
        sr: Sample rate
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
        
    Returns:
        Boolean array indicating voice activity
    """
    try:
        # Calculate energy
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Simple threshold-based VAD
        threshold = np.mean(energy) * 0.1
        vad = energy > threshold
        
        return vad
        
    except Exception as e:
        logger.error(f"Error in voice activity detection: {e}")
        return np.ones(len(audio) // hop_length, dtype=bool)


def convert_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Convert sample rate of audio
    
    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        logger.error(f"Error converting sample rate: {e}")
        return audio


def audio_quality_check(audio: np.ndarray, sr: int) -> dict:
    """
    Check audio quality metrics
    
    Args:
        audio: Input audio
        sr: Sample rate
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        # Calculate various quality metrics
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        
        # Signal-to-noise ratio estimate
        energy = np.mean(audio ** 2)
        noise_estimate = np.mean(audio[:int(0.1 * sr)] ** 2)  # First 100ms as noise estimate
        snr = 10 * np.log10(energy / max(noise_estimate, 1e-10))
        
        # Clipping detection
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        return {
            "rms": float(rms),
            "peak": float(peak),
            "snr_estimate": float(snr),
            "clipping_ratio": float(clipping_ratio),
            "zero_crossing_rate": float(zcr),
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "quality_score": min(1.0, max(0.0, (snr + 20) / 40))  # Rough quality score
        }
        
    except Exception as e:
        logger.error(f"Error in audio quality check: {e}")
        return {"error": str(e)}
