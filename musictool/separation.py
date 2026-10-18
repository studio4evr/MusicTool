"""Split a mix into drums / bass / melodic (/ vocal) parts before transcription.

* ``hpss`` - fast, dependency-free harmonic/percussive separation (median
             filtering) plus a low/high band split. Stems leak into each other.
* neural   - Meta's Hybrid Transformer Demucs v4, run through ``audio-separator``
             (PyTorch, uses an NVIDIA GPU when present). Far cleaner stems; the
             model is downloaded on first use.

``separate`` returns ``(stems, sample_rate)``. Stem keys: ``drums``, ``bass``,
``melodic`` and, for neural models, ``vocals``.
"""

import contextlib
import io
import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from .audio import lowpass, highpass

os.environ.setdefault("TQDM_DISABLE", "1")  # audio-separator's download/progress bars

BASS_CUTOFF_HZ = 200.0
NEURAL_SR = 44100

# key -> (audio-separator model file, label). Best first; SDR from the model registry.
NEURAL_MODELS = {
    "demucs-ft": ("htdemucs_ft.yaml", "Demucs v4 fine-tuned"),  # drums 10.0 dB, bass 12.0 dB, 4x slower
    "demucs": ("htdemucs.yaml", "Demucs v4"),  # drums 9.4 dB, bass 11.6 dB
}

STEM_NAMES = {"Drums": "drums", "Bass": "bass", "Other": "melodic", "Vocals": "vocals"}


def model_dir() -> Path:
    base = os.environ.get("MUSICTOOL_MODEL_DIR") or Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "musictool" / "models"
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def neural_available() -> bool:
    try:
        import audio_separator  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def demucs_available() -> bool:  # kept for the web API / older callers
    return neural_available()


def best_separation() -> str:
    return "demucs-ft" if neural_available() else "hpss"


def separation_label(method: str) -> str:
    return NEURAL_MODELS[method][1] if method in NEURAL_MODELS else "harmonic/percussive DSP"


def separate_hpss(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 2.0))
    return {
        "drums": percussive,
        "bass": lowpass(harmonic, sr, BASS_CUTOFF_HZ),
        "melodic": highpass(harmonic, sr, BASS_CUTOFF_HZ * 0.6),
    }


_separators: dict[str, object] = {}


def _separator(model_file: str, output_dir: str):
    """Load each model once per process; only the output folder changes per call."""
    sep = _separators.get(model_file)
    if sep is None:
        from audio_separator.separator import Separator

        with contextlib.redirect_stdout(io.StringIO()):
            sep = Separator(
                log_level=logging.ERROR,  # it warns about ONNX CUDA, which we don't use
                model_file_dir=str(model_dir()),
                output_dir=output_dir,
                output_format="WAV",
                sample_rate=NEURAL_SR,
                # shifts=1: one pass instead of two; ~2x faster for ~0.1 dB SDR
                demucs_params={"segment_size": "Default", "shifts": 1, "overlap": 0.25, "segments_enabled": True},
            )
            sep.load_model(model_filename=model_file)
        _separators[model_file] = sep
    sep.output_dir = output_dir
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = output_dir
    return sep


def run_model(model_file: str, audio) -> dict[str, np.ndarray]:
    """Run any audio-separator model on a file path or a mono 44.1 kHz array.

    Returns mono stems keyed by the model's lower-cased stem names
    (e.g. "drums", "other" or "kick", "hh").
    """
    with tempfile.TemporaryDirectory(prefix="musictool-stems-") as out:
        if isinstance(audio, np.ndarray):
            src = Path(out) / "input.wav"
            sf.write(src, audio, NEURAL_SR)
        else:
            src = Path(audio)
        sep = _separator(model_file, out)
        with contextlib.redirect_stdout(io.StringIO()):
            files = sep.separate(str(src))
        stems = {}
        for f in files:
            f = Path(f) if Path(f).is_absolute() else Path(out) / f
            match = re.search(r"_\(([^)]+)\)", f.stem)
            if match:
                y, _ = librosa.load(str(f), sr=NEURAL_SR, mono=True)
                stems[match.group(1).lower()] = y.astype(np.float32)
    return stems


def separate_neural(path, method: str) -> dict[str, np.ndarray]:
    model_file, _ = NEURAL_MODELS[method]
    stems = {STEM_NAMES[k.capitalize()]: v for k, v in run_model(model_file, path).items() if k.capitalize() in STEM_NAMES}
    missing = {"drums", "bass", "melodic"} - stems.keys()
    if missing:
        raise RuntimeError(f"Stem separation did not produce: {', '.join(sorted(missing))}")
    return stems


def separate(y: np.ndarray, sr: int, method: str = "hpss", path=None) -> tuple[dict[str, np.ndarray], int]:
    if method == "hpss":
        return separate_hpss(y, sr), sr
    if method in NEURAL_MODELS:
        if not neural_available():
            raise RuntimeError("Neural separation needs PyTorch + audio-separator (run install.bat and choose GPU)")
        if path is None:
            raise ValueError("Neural separation needs the source file path")
        return separate_neural(path, method), NEURAL_SR
    raise ValueError(f"Unknown separation method: {method}")
