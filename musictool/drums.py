"""Drum transcription for electronic music.

Onsets are detected on the percussive signal, then each hit is classified by
which frequency bands "light up" at that moment (multi-label, so a kick and a
hi-hat on the same 16th both get written). Output uses General MIDI drum notes.
"""

import numpy as np
import librosa
from scipy.signal import find_peaks

from .notes import Note

KICK, SNARE, CLOSED_HAT, OPEN_HAT = 36, 38, 42, 46

N_FFT = 1024
HOP = 256

BANDS = {
    "low": (30.0, 150.0),  # kick body / sub
    "mid": (180.0, 2500.0),  # snare / clap body
    "high": (6000.0, 11000.0),  # hats, rides, shakers
}


def _band_flux(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # Linear magnitude (not dB): loud transients dominate, so a bass note's soft
    # attack in the kick band doesn't look like a kick.
    band = S[(freqs >= lo) & (freqs < hi)]
    return np.maximum(0.0, np.diff(band, axis=1, prepend=band[:, :1])).sum(axis=0)


def _normalise(x: np.ndarray) -> np.ndarray:
    ref = np.percentile(x, 99.5)
    return x / ref if ref > 1e-9 else np.zeros_like(x)


def transcribe_drums(y: np.ndarray, sr: int, sensitivity: float = 0.5) -> list[Note]:
    """Return drum hits as GM drum notes.

    ``sensitivity`` (0-1) lowers the per-band thresholds as it increases.
    """
    if np.max(np.abs(y)) < 1e-4:
        return []

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    hop_s = HOP / sr

    flux = {name: _normalise(_band_flux(S, freqs, lo, hi)) for name, (lo, hi) in BANDS.items()}
    high_energy = S[(freqs >= BANDS["high"][0]) & (freqs < BANDS["high"][1])].sum(axis=0)

    combined = flux["low"] + flux["mid"] + flux["high"]
    onsets, _ = find_peaks(combined, height=0.12, prominence=0.08, distance=max(1, int(0.04 / hop_s)))

    threshold = float(np.interp(sensitivity, [0.0, 1.0], [0.55, 0.15]))
    n_frames = combined.shape[0]
    notes: list[Note] = []
    for f in onsets:
        win = slice(max(0, f - 1), min(n_frames, f + 3))
        strength = {name: float(flux[name][win].max()) for name in BANDS}
        t = f * hop_s

        def add(pitch: int, value: float, length: float = 0.1):
            velocity = int(np.clip(40 + 87 * min(value, 1.0), 1, 127))
            notes.append(Note(t, t + length, pitch, velocity))

        is_snare = strength["mid"] >= threshold * 1.2 and strength["mid"] > 0.6 * strength["low"]
        if strength["low"] >= threshold:
            add(KICK, strength["low"])
        if is_snare:
            add(SNARE, strength["mid"])
        if strength["high"] >= threshold and not (is_snare and strength["high"] < 1.3 * strength["mid"]):
            pitch = OPEN_HAT if _decay_time(high_energy, f, hop_s) > 0.14 else CLOSED_HAT
            add(pitch, strength["high"])

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _decay_time(energy: np.ndarray, frame: int, hop_s: float) -> float:
    """Seconds until band energy drops below 25% of its peak after an onset."""
    end = min(len(energy), frame + int(0.5 / hop_s))
    seg = energy[frame:end]
    if seg.size == 0:
        return 0.0
    peak_idx = int(seg[: max(1, int(0.03 / hop_s))].argmax())
    peak = seg[peak_idx]
    below = np.nonzero(seg[peak_idx:] < 0.25 * peak)[0]
    n = below[0] if below.size else seg.size - peak_idx
    return n * hop_s
