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
    if sr != 22050:  # band edges and thresholds are tuned at 22.05 kHz
        y, sr = librosa.resample(y, orig_sr=sr, target_sr=22050), 22050

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


# --- Neural drum transcription (ADTOF) ---------------------------------------

# ADTOF classes (its GM output pitches) -> the GM notes we write.
ADTOF_CLASSES = [(35, KICK), (38, SNARE), (47, 47), (42, CLOSED_HAT), (49, 49)]  # kick, snare, toms, hats, cymbals
ADTOF_SR = 44100
ADTOF_FPS = 100
# ADTOF's thresholds were tuned on mostly acoustic kits. Clickless 808/909-style
# sine kicks only reach ~0.15-0.2 kick activation, so we lower the kick threshold
# (0.22 -> 0.14). On a 909-ish kick this adds no false hits; on a pure sine kick it
# lifts kick F1 from 0.72 to 0.97 (bench/ + tests/synth.py).
ADTOF_THRESHOLD_SCALE = [0.65, 1.0, 1.0, 1.0, 1.0]

_adtof = None


def adtof_available() -> bool:
    try:
        import adtof_pytorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load_adtof():
    global _adtof
    if _adtof is None:
        import contextlib
        import io

        import torch
        from adtof_pytorch import calculate_n_bins, create_frame_rnn_model, get_default_weights_path, load_pytorch_weights
        from adtof_pytorch.audio import create_adtof_processor

        from .gpu import device

        model = create_frame_rnn_model(calculate_n_bins())
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_pytorch_weights(model, get_default_weights_path(), strict=False)
        model.eval().to(device())
        _adtof = (model, create_adtof_processor(), torch)
    return _adtof


def transcribe_drums_adtof(y: np.ndarray, sr: int, sensitivity: float = 0.5) -> list[Note]:
    """Drum transcription with the ADTOF Frame-RNN (Zehren et al.), a CNN + bi-GRU
    trained on ~114 h of real, crowd-annotated music.

    Writes kick, snare, toms, closed hi-hat and cymbal; velocity follows the
    network's confidence.
    """
    if np.max(np.abs(y)) < 1e-4:
        return []
    from adtof_pytorch import FRAME_RNN_THRESHOLDS, PeakPicker

    from .gpu import device

    model, processor, torch = _load_adtof()
    if sr != ADTOF_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ADTOF_SR)
    spec = processor.apply_filterbank(processor.compute_stft(y.astype(np.float32))).T[None, :, :, None]
    with torch.no_grad():
        act = model(torch.from_numpy(np.ascontiguousarray(spec)).float().to(device())).cpu().numpy()[0]

    # sensitivity 0.5 = the paper's tuned thresholds; 0 = x1.6 stricter, 1 = x0.5 looser
    scale = float(np.interp(sensitivity, [0.0, 0.5, 1.0], [1.6, 1.0, 0.5]))
    thresholds = [min(0.95, t * c * scale) for t, c in zip(FRAME_RNN_THRESHOLDS, ADTOF_THRESHOLD_SCALE)]
    picker = PeakPicker(thresholds=thresholds, fps=ADTOF_FPS)
    peaks = picker.pick(act, labels=[c for c, _ in ADTOF_CLASSES])[0]

    notes = []
    for i, (cls, pitch) in enumerate(ADTOF_CLASSES):
        for t in peaks[cls]:
            frame = min(act.shape[0] - 1, int(round(t * ADTOF_FPS)))
            conf = float(act[max(0, frame - 1) : frame + 2, i].max())
            velocity = int(np.clip(35 + 92 * conf, 1, 127))
            notes.append(Note(float(t), float(t) + 0.1, pitch, velocity))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
