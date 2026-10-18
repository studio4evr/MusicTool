"""Score transcriptions against a reference MIDI of the same track (e.g. a
hand-made or commercial transcription).

The reference may be on a different timeline (another tempo, an offset, a
double-time grid), so it is first aligned: t_audio = scale * t_ref + offset,
fitted on kick drums (the most reliable shared landmarks).

    python bench/reference.py reference.mid ours.mid [more.mid ...]

Scores are F1 (1.0 = matches the reference). Drums: onset F1 per class.
Pitched: note F1 (onset within TOL and exact pitch), "pc" = pitch class
(ignores octave errors), "frame" = per-20 ms piano-roll F1 (are the right
pitches sounding).
"""

import argparse
from pathlib import Path

import numpy as np
import pretty_midi
from mir_eval.onset import f_measure as onset_f
from mir_eval.transcription import precision_recall_f1_overlap

TOL = 0.06  # the reference and our output are both grid-quantised; allow half a 32nd at ~130 BPM
FRAME = 0.02

DRUM_CLASS = {35: "kick", 36: "kick", 38: "snare", 39: "snare", 40: "snare",  # clap counted with snare
              42: "hihat", 44: "hihat", 46: "hihat",
              49: "cymbal", 51: "cymbal", 52: "cymbal", 55: "cymbal", 57: "cymbal", 59: "cymbal"}


def drum_class(pitch: int) -> str:
    return DRUM_CLASS.get(pitch, "perc")


def load(path: Path):
    """-> drums {class: onsets}, bass [(s, e, p)], synth [(s, e, p)]"""
    pm = pretty_midi.PrettyMIDI(str(path))
    drums, bass, synth = {}, [], []
    for inst in pm.instruments:
        if inst.is_drum:
            for n in inst.notes:
                drums.setdefault(drum_class(n.pitch), []).append(n.start)
        else:
            is_bass = inst.name == "Bass" or (not inst.name and 32 <= inst.program < 40)
            (bass if is_bass else synth).extend((n.start, n.end, n.pitch) for n in inst.notes)
    return {k: np.array(sorted(v)) for k, v in drums.items()}, sorted(bass), sorted(synth)


def fit_alignment(ref_kicks: np.ndarray, kicks: np.ndarray) -> tuple[float, float]:
    """scale, offset maximising kick matches within 30 ms."""
    best = (0, 1.0, 0.0)
    for scale in np.arange(0.80, 1.25, 0.0005):
        diffs = (kicks[:, None] - scale * ref_kicks[None, :]).ravel()
        diffs = diffs[np.abs(diffs) < 3.0]
        if not diffs.size:
            continue
        hist, edges = np.histogram(diffs, bins=np.arange(-3.0, 3.0 + 0.01, 0.01))
        i = int(hist.argmax())
        if hist[i] > best[0]:
            best = (hist[i], scale, (edges[i] + edges[i + 1]) / 2)
    _, scale, offset = best
    # refine offset with the median of matched differences
    shifted = scale * ref_kicks + offset
    d = np.array([kicks[np.abs(kicks - t).argmin()] - t for t in shifted])
    offset += float(np.median(d[np.abs(d) < 0.03])) if np.any(np.abs(d) < 0.03) else 0.0
    return float(scale), float(offset)


def refine_offset_with_pitch(scale, offset, ref, est) -> float:
    """Kick patterns repeat every bar, so the kick fit can land whole bars off.
    Re-pick the offset (within +-8 s) by how many pitched onsets also match in
    pitch class (exact pitch would punish octave errors)."""
    def keyed(notes):
        return {pc: np.array(sorted(s for s, _, p in notes if p % 12 == pc)) for pc in range(12)}

    rk, ek = keyed(ref[1] + ref[2]), keyed(est[1] + est[2])

    def matches(off):
        total = 0
        for pc in range(12):
            r, e = scale * rk[pc] + off, ek[pc]
            if len(r) and len(e):
                idx = np.clip(np.searchsorted(e, r), 1, len(e) - 1)
                d = np.minimum(np.abs(e[idx] - r), np.abs(e[idx - 1] - r))
                total += int((d <= 0.03).sum())
        return total

    cands = offset + np.arange(-8.0, 8.0, 0.005)
    return float(max(cands, key=matches))


def polish_with_kicks(scale, offset, ref_kicks, kicks):
    """Least-squares fit through kick pairs matched within 60 ms: removes the
    residual constant lag and the slow drift of the coarse scale grid."""
    t = scale * ref_kicks + offset
    j = np.clip(np.searchsorted(kicks, t), 1, len(kicks) - 1)
    near = np.where(np.abs(kicks[j] - t) < np.abs(kicks[j - 1] - t), kicks[j], kicks[j - 1])
    ok = np.abs(near - t) < 0.06
    if ok.sum() < 20:
        return scale, offset
    a, b = np.polyfit(ref_kicks[ok], near[ok], 1)
    return float(a), float(b)


def apply(scale, offset, drums, bass, synth):
    f = lambda t: scale * t + offset
    # notes the alignment puts before the audio starts can't be matched; drop them
    return ({k: f(v)[f(v) >= 0] for k, v in drums.items()},
            [(f(s), f(e), p) for s, e, p in bass if f(s) >= 0],
            [(f(s), f(e), p) for s, e, p in synth if f(s) >= 0])


def note_f1(ref, est, pitch_class=False):
    if not ref or not est:
        return 0.0
    mod = (lambda p: 60 + p % 12) if pitch_class else (lambda p: p)
    hz = lambda ps: 440 * 2 ** ((np.array([mod(p) for p in ps]) - 69) / 12)
    ri = np.array([[s, max(e, s + 0.01)] for s, e, _ in ref])
    ei = np.array([[s, max(e, s + 0.01)] for s, e, _ in est])
    return precision_recall_f1_overlap(ri, hz([p for *_, p in ref]), ei, hz([p for *_, p in est]),
                                       onset_tolerance=TOL, offset_ratio=None)[2]


def frame_f1(ref, est, end):
    n = int(end / FRAME) + 1

    def roll(notes):
        r = np.zeros((128, n), bool)
        for s, e, p in notes:
            r[p, max(0, int(s / FRAME)): max(0, int(e / FRAME)) + 1] = True
        return r

    a, b = roll(ref), roll(est)
    tp = (a & b).sum()
    if not tp:
        return 0.0
    P, R = tp / b.sum(), tp / a.sum()
    return float(2 * P * R / (P + R))


def score(ref, est, end):
    rd, rb, rs = ref
    ed, eb, es = est
    out = {}
    def dedupe(t, gap=0.03):  # e.g. a snare and a clap on the same hit = one "snare" onset
        t = np.sort(t)
        return t[np.concatenate([[True], np.diff(t) > gap])] if len(t) else t

    for cls in ("kick", "snare", "hihat", "cymbal", "perc"):
        r, e = dedupe(rd.get(cls, np.array([]))), dedupe(ed.get(cls, np.array([])))
        out[cls] = onset_f(r, e, window=TOL)[0] if len(r) and len(e) else 0.0
    out["bass"] = note_f1(rb, eb)
    out["bass_pc"] = note_f1(rb, eb, pitch_class=True)
    out["synth"] = note_f1(rs, es)
    out["synth_pc"] = note_f1(rs, es, pitch_class=True)
    out["synth_frame"] = frame_f1(rs, es, end)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("midi", nargs="+")
    ap.add_argument("--anchor", help="MIDI whose kicks to align the reference to (default: first midi)")
    args = ap.parse_args()

    ref = load(Path(args.reference))
    anchor = load(Path(args.anchor or args.midi[0]))
    scale, offset = fit_alignment(ref[0].get("kick", np.array([])), anchor[0].get("kick", np.array([])))
    offset = refine_offset_with_pitch(scale, offset, ref, anchor)
    scale, offset = polish_with_kicks(scale, offset, ref[0].get("kick", np.array([])), anchor[0].get("kick", np.array([])))
    ref = apply(scale, offset, *ref)
    end = max([max((e for _, e, _ in ref[1] + ref[2]), default=0)] + [v.max() for v in ref[0].values() if len(v)])
    print(f"alignment: t_audio = {scale:.4f} * t_ref {offset:+.3f} s")
    counts = {k: len(v) for k, v in ref[0].items()}
    print(f"reference: drums {counts}, bass {len(ref[1])}, synth {len(ref[2])}\n")

    cols = ["kick", "snare", "hihat", "cymbal", "perc", "bass", "bass_pc", "synth", "synth_pc", "synth_frame"]
    print(f"{'midi':28}" + "".join(f"{c:>12}" for c in cols))
    for m in args.midi:
        s = score(ref, load(Path(m)), end)
        print(f"{Path(m).parent.name + '/' + Path(m).name:28}" + "".join(f"{s[c]:12.2f}" for c in cols))


if __name__ == "__main__":
    main()
