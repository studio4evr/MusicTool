"""Score engine combinations against bench/synth_track.py's ground truth.

    python bench/benchmark.py            # all configs
    python bench/benchmark.py --quick    # baseline vs best only

Metrics are mir_eval F1 scores (50 ms onset tolerance; pitched notes must also
match pitch exactly, offsets ignored). Output is unquantised so the grid
doesn't hide timing errors.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mir_eval.onset import f_measure as onset_f
from mir_eval.transcription import precision_recall_f1_overlap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from musictool import TranscriptionOptions, transcribe  # noqa: E402
from musictool.drums import KICK as KICK_NOTE, SNARE as SNARE_NOTE  # noqa: E402
import synth_track  # noqa: E402

HAT_NOTES = {42, 46}

CONFIGS = {
    "baseline (hpss/dsp/pyin)": dict(separation="hpss", drum_engine="dsp", mono_engine="pyin"),
    "demucs + dsp/pyin": dict(separation="demucs-ft", drum_engine="dsp", mono_engine="pyin"),
    "demucs + adtof/pyin": dict(separation="demucs-ft", drum_engine="adtof", mono_engine="pyin"),
    "demucs + adtof/crepe": dict(separation="demucs-ft", drum_engine="adtof", mono_engine="crepe"),
    "demucs + drumsep/basic-pitch": dict(separation="demucs-ft", drum_engine="drumsep", mono_engine="basic-pitch"),
    "default: demucs + ensemble/basic-pitch": dict(separation="demucs-ft", drum_engine="ensemble", mono_engine="basic-pitch"),
}
QUICK = ["baseline (hpss/dsp/pyin)", "default: demucs + ensemble/basic-pitch"]


def note_f1(ref, est):
    if not ref or not est:
        return 0.0
    ref_i = np.array([[s, e] for s, e, _ in ref])
    ref_p = np.array([440 * 2 ** ((p - 69) / 12) for *_, p in ref])
    est_i = np.array([[n.start, max(n.end, n.start + 0.01)] for n in est])
    est_p = np.array([440 * 2 ** ((n.pitch - 69) / 12) for n in est])
    return precision_recall_f1_overlap(ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None)[2]


def score(result, truth):
    tracks = {t.name: t.notes for t in result.tracks}
    drums = tracks.get("Drums", [])

    def hits(pitches):
        return np.array(sorted(n.start for n in drums if n.pitch in pitches))

    out = {
        "kick": onset_f(np.array(truth["kick"]), hits({KICK_NOTE}), window=0.05)[0],
        "snare": onset_f(np.array(truth["snare"]), hits({SNARE_NOTE}), window=0.05)[0],
        "hihat": onset_f(np.array(sorted(set(truth["hat"]))), hits(HAT_NOTES), window=0.05)[0],
        "bass": note_f1(truth["bass"], tracks.get("Bass", [])),
        # every pitched track except Bass: instrument-split output has many synth tracks
        "synths": note_f1(truth["synths"], [n for t in result.tracks if not t.is_drum and t.name != "Bass" for n in t.notes]),
    }
    out["mean"] = float(np.mean(list(out.values())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    y, sr, truth = synth_track.render(seed=args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.wav"
        sf.write(path, y, sr)
        names = QUICK if args.quick else list(CONFIGS)
        rows = []
        for name in names:
            t = time.time()
            res = transcribe(path, TranscriptionOptions(quantize=None, **CONFIGS[name]))
            rows.append((name, score(res, truth), time.time() - t, res.bpm))

    cols = ["kick", "snare", "hihat", "bass", "synths", "mean"]
    print(f"\n{'config':38}" + "".join(f"{c:>8}" for c in cols) + f"{'secs':>7}{'bpm':>8}")
    for name, s, secs, bpm in rows:
        print(f"{name:38}" + "".join(f"{s[c]:8.2f}" for c in cols) + f"{secs:7.1f}{bpm:8.1f}")
    print(f"\n(true tempo {132.0} BPM, {len(y) / sr:.0f} s of audio)")


if __name__ == "__main__":
    main()
