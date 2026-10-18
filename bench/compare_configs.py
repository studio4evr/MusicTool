"""Run engine configurations on a real track + reference MIDI and on the synthetic
benchmark, side by side.

    python bench/compare_configs.py song.wav reference.mid [--out DIR]
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import reference as R  # noqa: E402
import synth_track  # noqa: E402
from musictool import TranscriptionOptions, transcribe  # noqa: E402

BASE = dict(separation="demucs-ft", drum_engine="ensemble")
CONFIGS = {
    "default (auto)": dict(),
    "default, drop unlabelled": dict(unlabelled_min_velocity=None),
    "synth pitch from C3": dict(synth_min_pitch=48),
    "synth velocity from 40": dict(synth_min_velocity=40),
    "synth C3 + vel 40": dict(synth_min_pitch=48, synth_min_velocity=40),
    "synth C3 + vel 55": dict(synth_min_pitch=48, synth_min_velocity=55),
    "v5: hybrid(moe) + BP bass": dict(synth_engine="hybrid", mono_engine="basic-pitch", mt3_variant="moe-nops", mt3_input="no-drums"),
    "hybrid(ptf) + BP bass": dict(synth_engine="hybrid", mono_engine="basic-pitch", mt3_variant="ptf-ps"),
    "hybrid(ptf) + MT3 bass": dict(synth_engine="hybrid", mono_engine="yourmt3", mt3_variant="ptf-ps"),
    "layers + MT3 bass": dict(synth_engine="layers", mono_engine="yourmt3", mt3_variant="ptf-ps"),
    "MT3 only (ptf)": dict(synth_engine="yourmt3", mono_engine="yourmt3", mt3_variant="ptf-ps"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("reference")
    ap.add_argument("--out", default="output/compare")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only or list(CONFIGS)

    mids = {}
    for name in names:
        res = transcribe(args.song, TranscriptionOptions(**BASE, **CONFIGS[name]))
        mids[name] = res.write(out / (re.sub(r"[^\w]+", "_", name).strip("_") + ".mid"))

    ref = R.load(Path(args.reference))
    anchor = R.load(next(iter(mids.values())))
    scale, offset = R.fit_alignment(ref[0]["kick"], anchor[0]["kick"])
    offset = R.refine_offset_with_pitch(scale, offset, ref, anchor)
    scale, offset = R.polish_with_kicks(scale, offset, ref[0]["kick"], anchor[0]["kick"])
    ref = R.apply(scale, offset, *ref)
    end = max(e for _, e, _ in ref[1] + ref[2])

    y, sr, truth = synth_track.render(seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        bench_wav = Path(tmp) / "bench.wav"
        sf.write(bench_wav, y, sr)
        bench = {name: benchmark.score(transcribe(bench_wav, TranscriptionOptions(quantize=None, **BASE, **CONFIGS[name])), truth)
                 for name in names}

    print(f"\n{'config':28}| vs reference: {'kick':>5} {'snare':>5} {'hihat':>5} {'bass':>5} {'synth':>5} {'s_pc':>5} {'frame':>5}"
          f" | bench: {'bass':>5} {'synth':>5} {'mean':>5}")
    for name in names:
        s = R.score(ref, R.load(mids[name]), end)
        b = bench[name]
        print(f"{name:28}| {'':13} {s['kick']:5.2f} {s['snare']:5.2f} {s['hihat']:5.2f} {s['bass']:5.2f} {s['synth']:5.2f}"
              f" {s['synth_pc']:5.2f} {s['synth_frame']:5.2f} | {'':6} {b['bass']:5.2f} {b['synths']:5.2f} {b['mean']:5.2f}")


if __name__ == "__main__":
    main()
