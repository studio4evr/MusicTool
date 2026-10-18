"""Multi-instrument transcription with YourMT3+ (Chang et al., MLSP 2024).

YourMT3+ is an MT3-style audio-to-MIDI transformer (Perceiver-TF encoder,
mixture-of-experts, multi-channel T5 decoder) trained on ten datasets including
Slakh's rendered software instruments. Unlike Basic Pitch it also says *which
instrument* plays each note (13 classes: synth lead, synth pad, piano, organ,
bass, strings, ...), so different synths come out as different tracks.

The code and checkpoint are not on PyPI. ``install.bat`` fetches them into
``<model dir>/YourMT3`` (the Hugging Face space ``mimbres/YourMT3``, pinned).
The checkpoint is a full pickle (``torch.load(weights_only=False)``), so it is
only ever loaded from that pinned download.
"""

import contextlib
import io
import os
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from .notes import Note

_MOE = ["-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1",
        "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
        "-ac", "spec", "-hop", "300", "-atc", "1"]
# The authors' demo-app checkpoints: name -> (checkpoint, loader args).
VARIANTS = {
    "moe-nops": ("mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt", _MOE),
    "moe-ps": ("mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt", _MOE),
    "ptf-ps": ("ptf_mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt",
               ["-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
                "-ac", "spec", "-hop", "300", "-atc", "1"]),
}
DEFAULT_VARIANT = "moe-nops"


def ckpt_path(variant: str = DEFAULT_VARIANT) -> Path:
    exp, file = VARIANTS[variant][0].split("@")
    return Path("amt/logs/2024") / exp / "checkpoints" / file


def loader_args(variant: str = DEFAULT_VARIANT) -> list[str]:
    checkpoint, args = VARIANTS[variant]
    return [checkpoint, "-p", "2024", *args, "-pr", "16"]

# YourMT3's class names -> (our track name, GM program to write). On electronic music
# the acoustic classes mean "a synth that sounds like this", hence the renames.
CLASS_TRACKS = {
    "Acoustic Piano": ("Keys", 4),  # electric piano
    "Electric Piano": ("Keys", 4),
    "Chromatic Percussion": ("Bells / Mallets", 11),
    "Organ": ("Organ", 16),
    "Guitar (clean)": ("Pluck", 24),
    "Guitar (distortion)": ("Distorted Lead", 30),
    "Bass": ("Bass", 38),
    "Strings": ("Strings", 50),  # synth strings
    "Brass": ("Brass Synth", 62),  # synth brass
    "Reed": ("Wind Synth", 64),
    "Pipe": ("Wind Synth", 73),
    "Synth Lead": ("Synth Lead", 81),
    "Synth Pad": ("Synth Pad", 89),
    "Singing Voice": ("Vocals", 54),
    "Singing Voice (chorus)": ("Vocals", 54),
}
TRACK_PROGRAMS = {name: program for name, program in CLASS_TRACKS.values()}


SPACE = "mimbres/YourMT3"
SPACE_REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"  # the revision bench/ was measured with


def download(variant: str = DEFAULT_VARIANT) -> Path:
    """Fetch the YourMT3 code + one checkpoint (~560 MB) into the model dir."""
    from huggingface_hub import snapshot_download

    from .separation import model_dir

    target = model_dir() / "YourMT3"
    snapshot_download(
        repo_id=SPACE,
        repo_type="space",
        revision=SPACE_REVISION,
        local_dir=target,
        allow_patterns=["model_helper.py", "amt/src/**", ckpt_path(variant).as_posix()],
    )
    return target


def ymt3_dir(variant: str = DEFAULT_VARIANT) -> Path | None:
    """Where the YourMT3 space (with this checkpoint) was downloaded, or None."""
    from .separation import model_dir

    candidates = [os.environ.get("MUSICTOOL_YMT3_DIR"), model_dir() / "YourMT3",
                  Path(__file__).resolve().parents[1] / "third_party" / "YourMT3"]
    for c in candidates:
        if c and (Path(c) / ckpt_path(variant)).is_file():
            return Path(c)
    return None


def ymt3_available(variant: str = DEFAULT_VARIANT) -> bool:
    try:
        import lightning  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return ymt3_dir(variant) is not None


_models: dict[str, object] = {}
_pristine_cfg = None


def _load(variant: str = DEFAULT_VARIANT):
    if variant in _models:
        return _models[variant]
    root = ymt3_dir(variant)
    if root is None:
        raise RuntimeError("YourMT3+ is not installed (run install.bat)")
    for p in (str(root), str(root / "amt" / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.modules.setdefault("wandb", types.SimpleNamespace(Table=lambda **kw: None))  # training-only logging

    from .gpu import device

    # The authors' loader resolves checkpoint paths relative to the repo and is chatty.
    with contextlib.chdir(root), warnings.catch_warnings(), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        import copy

        import model.init_train as init_train
        from model_helper import load_model_checkpoint

        # update_config() edits the module-level default audio/model configs in place, so
        # a second checkpoint would load with the first one's architecture (and, as loading
        # is strict=False, silently drop the mismatched weights). Give every load a fresh copy.
        global _pristine_cfg
        if _pristine_cfg is None:
            _pristine_cfg = copy.deepcopy((init_train.default_audio_cfg, init_train.default_model_cfg))
        init_train.default_audio_cfg, init_train.default_model_cfg = copy.deepcopy(_pristine_cfg)

        model = load_model_checkpoint(args=loader_args(variant), device="cpu")
    _models[variant] = model.to(device()).eval()
    return _models[variant]


def transcribe_mt3(y: np.ndarray, sr: int, batch_size: int = 8, variant: str = DEFAULT_VARIANT) -> dict[str, list[Note]]:
    """Transcribe a mix (or stem) into ``{instrument family: notes}``.

    Drum notes come back under "Drums" (GM drum pitches).
    """
    if np.max(np.abs(y)) < 1e-4:
        return {}
    import librosa
    import torch

    from .gpu import device

    model = _load(variant)
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    target_sr = model.audio_cfg["sample_rate"]
    frames = model.audio_cfg["input_frames"]
    audio = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr) if sr != target_sr else y
    segments = slice_padded_array(audio[None, :], frames, frames)
    segments = torch.from_numpy(segments.astype("float32")).to(device()).unsqueeze(1)  # (n, 1, frames)

    with torch.no_grad(), warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        pred_tokens, _ = model.inference_file(bsz=batch_size, audio_segments=segments)
        starts = [frames * i / target_sr for i in range(segments.shape[0])]
        per_channel = []
        for ch in range(model.task_manager.num_decoding_channels):
            tokens_ch = [arr[:, ch, :] for arr in pred_tokens]
            zipped, _, _ = model.task_manager.detokenize_list_batches(tokens_ch, starts, return_events=True)
            notes_ch, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
            per_channel.append(notes_ch)
        notes = mix_notes(per_channel)

    inverse = model.midi_output_inverse_vocab or {}
    tracks: dict[str, list[Note]] = defaultdict(list)
    for n in notes:
        if n.offset <= n.onset:
            continue
        velocity = int(np.clip(n.velocity if n.velocity > 1 else 100, 1, 127))
        if n.is_drum:
            tracks["Drums"].append(Note(float(n.onset), float(n.onset) + 0.1, int(n.pitch), velocity))
        else:
            class_name = inverse.get(int(n.program), (None, "Synth Lead"))[1]
            track = CLASS_TRACKS.get(class_name, ("Synth Lead", 81))[0]
            tracks[track].append(Note(float(n.onset), float(n.offset), int(n.pitch), velocity))
    for v in tracks.values():
        v.sort(key=lambda n: (n.start, n.pitch))
    return dict(tracks)


if __name__ == "__main__":  # python -m musictool.mt3 --download
    import argparse

    ap = argparse.ArgumentParser(description="Manage the YourMT3+ model.")
    ap.add_argument("--download", action="store_true", help="download YourMT3+ (~560 MB)")
    ap.add_argument("--variant", choices=list(VARIANTS), default=DEFAULT_VARIANT)
    a = ap.parse_args()
    if a.download:
        print("YourMT3+", a.variant, "installed in", download(a.variant))
