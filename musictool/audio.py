"""Audio loading and simple filtering helpers."""

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 22050


def load_audio(path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load any audio file librosa/soundfile can read as mono float32."""
    y, sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"No audio data in {path}")
    return y.astype(np.float32), sr


def lowpass(y: np.ndarray, sr: int, cutoff: float, order: int = 6) -> np.ndarray:
    sos = butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def highpass(y: np.ndarray, sr: int, cutoff: float, order: int = 6) -> np.ndarray:
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)
