"""Split a mix into drums / bass / melodic parts before transcription.

Two methods are available:

* ``hpss``   - fast, dependency-free harmonic/percussive source separation
               (median filtering) followed by a low/high band split.
* ``demucs`` - Meta's Hybrid Transformer Demucs neural stem separator.
               Much cleaner stems, but requires ``pip install demucs`` (PyTorch).
"""

import numpy as np
import librosa

from .audio import lowpass, highpass

BASS_CUTOFF_HZ = 200.0


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def separate_hpss(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 2.0))
    return {
        "drums": percussive,
        "bass": lowpass(harmonic, sr, BASS_CUTOFF_HZ),
        "melodic": highpass(harmonic, sr, BASS_CUTOFF_HZ * 0.6),
    }


_demucs_model = None


def separate_demucs(path, sr: int) -> dict[str, np.ndarray]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    global _demucs_model
    if _demucs_model is None:
        _demucs_model = get_model("htdemucs")
        _demucs_model.eval()
    model = _demucs_model

    wav, _ = librosa.load(str(path), sr=model.samplerate, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    wav = wav[: model.audio_channels]
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    tensor = torch.from_numpy((wav - mean) / std).float()[None]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        sources = apply_model(model, tensor, device=device, split=True, overlap=0.25)[0]
    sources = sources.cpu().numpy() * std + mean

    stems = {}
    for name, source in zip(model.sources, sources):
        mono = librosa.resample(source.mean(0), orig_sr=model.samplerate, target_sr=sr)
        stems["melodic" if name == "other" else name] = mono.astype(np.float32)
    return stems


def separate(y: np.ndarray, sr: int, method: str = "hpss", path=None) -> dict[str, np.ndarray]:
    if method == "hpss":
        return separate_hpss(y, sr)
    if method == "demucs":
        if not demucs_available():
            raise RuntimeError("Demucs is not installed. Run: pip install demucs")
        if path is None:
            raise ValueError("Demucs separation needs the source file path")
        return separate_demucs(path, sr)
    raise ValueError(f"Unknown separation method: {method}")
