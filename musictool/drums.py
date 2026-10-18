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


# --- Drum-kit separation (DrumSep) ------------------------------------------

DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
RIDE, CRASH, TOMS = 51, 49, 47
DRUMSEP_PITCH = {"kick": KICK, "snare": SNARE, "toms": TOMS, "hh": CLOSED_HAT, "ride": RIDE, "crash": CRASH}


def drumsep_available() -> bool:
    from .separation import neural_available

    return neural_available()


def _local_peaks(y: np.ndarray, times: np.ndarray, sr: int, win: float = 0.04) -> np.ndarray:
    n = int(win * sr)
    return np.array([np.abs(y[int(t * sr): int(t * sr) + n]).max(initial=0.0) for t in times])


def transcribe_drums_drumsep(y: np.ndarray, sr: int, sensitivity: float = 0.5) -> list[Note]:
    """Split the drum stem into kick / snare / toms / hi-hat / ride / crash with
    MDX23C DrumSep (aufr33 & jarredou), then detect onsets on each piece.

    Class comes from *which stem* a hit lands in, so it doesn't depend on the
    kit sounding acoustic (ADTOF misses many electronic snares). A hit only
    counts if that stem carries a real share of the drum stem's level at that
    moment, which rejects bleed between the separated pieces.
    """
    if np.max(np.abs(y)) < 1e-4:
        return []
    from .separation import NEURAL_SR, run_model

    if sr != NEURAL_SR:
        y, sr = librosa.resample(y, orig_sr=sr, target_sr=NEURAL_SR), NEURAL_SR
    parts = run_model(DRUMSEP_MODEL, y)

    # sensitivity 0.5 -> a piece must reach 22% of the drum stem's local peak
    share = float(np.interp(sensitivity, [0.0, 0.5, 1.0], [0.4, 0.22, 0.1]))
    notes: list[Note] = []
    for name, part in parts.items():
        pitch = DRUMSEP_PITCH.get(name)
        if pitch is None or np.max(np.abs(part)) < 1e-4:
            continue
        env = librosa.onset.onset_strength(y=part, sr=sr, hop_length=256)
        times = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=256, units="time", backtrack=False)
        if not len(times):
            continue
        peak = _local_peaks(part, times, sr)
        total = _local_peaks(y, times, sr) + 1e-9
        keep = (peak / total >= share) & (peak >= 0.05 * np.percentile(peak, 95))
        ref = np.percentile(peak[keep], 95) if keep.any() else 1.0
        hop_s = 256 / sr
        if name == "hh":
            hh_env = librosa.feature.rms(y=part, frame_length=1024, hop_length=256)[0]
        for t, p, k in zip(times, peak, keep):
            if not k:
                continue
            note_pitch = pitch
            if name == "hh" and _decay_time(hh_env, int(t / hop_s), hop_s) > 0.14:
                note_pitch = OPEN_HAT
            velocity = int(np.clip(30 + 97 * min(1.0, p / ref), 1, 127))
            notes.append(Note(float(t), float(t) + 0.1, note_pitch, velocity))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


_GROUP = {KICK: "kick", SNARE: "snare", CLOSED_HAT: "hat", OPEN_HAT: "hat", TOMS: "toms", CRASH: "cym", RIDE: "cym"}


def transcribe_drums_ensemble(y: np.ndarray, sr: int, sensitivity: float = 0.5) -> list[Note]:
    """DrumSep hits, plus any ADTOF hit DrumSep has no same-instrument hit for (+-30 ms).

    On bench/ the union beats both (snare 0.94, hats 0.93); on a real Aphex Twin
    track it catches ~2x ADTOF's snares at full hit recall.
    """
    base = transcribe_drums_drumsep(y, sr, sensitivity)
    extra = [
        n
        for n in transcribe_drums_adtof(y, sr, sensitivity)
        if not any(_GROUP.get(m.pitch) == _GROUP.get(n.pitch) and abs(m.start - n.start) <= 0.03 for m in base)
    ]
    return sorted(base + extra, key=lambda n: (n.start, n.pitch))
