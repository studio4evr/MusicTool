"""Split a synth note list into per-instrument tracks using YourMT3+'s labels.

Basic Pitch finds pitches well (real-track synth coverage 0.94) but says nothing
about *which* instrument plays them. YourMT3+ labels instruments but, on dense
electronic music, misses most notes (coverage ~0.3). So: keep Basic Pitch's
notes, borrow YourMT3+'s instrument for each, and add the YourMT3+ notes Basic
Pitch missed.
"""

from collections import defaultdict

from .notes import Note

OTHER = "Synths"


def _overlap(a: Note, b: Note) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def assign_instruments(
    notes: list[Note],
    labelled: dict[str, list[Note]],
    *,
    add_missing: bool = True,
    onset_tol: float = 0.05,
    # 4 s / an octave labels ~2/3 of the notes on a real track; wider windows force
    # labels onto unrelated notes, narrower ones leave most of them unlabelled.
    context_s: float = 4.0,
    context_semitones: int = 12,
    min_share: float = 0.03,
) -> dict[str, list[Note]]:
    """Return ``{instrument: notes}``.

    For each note: the instrument of an overlapping labelled note at the same
    pitch; otherwise the instrument of the nearest labelled note within
    ``context_s`` seconds and ``context_semitones`` (the same line continuing);
    otherwise ``OTHER``. With ``add_missing``, labelled notes with no
    same-pitch note starting within ``onset_tol`` are added too. Instruments
    with under ``min_share`` of the notes are folded into ``OTHER``.
    """
    ref = sorted(((fam, n) for fam, ns in labelled.items() if fam not in ("Drums", "Bass") for n in ns),
                 key=lambda fn: fn[1].start)
    starts = [n.start for _, n in ref]
    import bisect

    def window(t0: float, t1: float):
        lo = bisect.bisect_left(starts, t0)
        hi = bisect.bisect_right(starts, t1)
        return ref[lo:hi]

    out: dict[str, list[Note]] = defaultdict(list)
    for n in notes:
        cands = window(n.start - 4.0, n.end)  # labelled notes can be long; look back generously
        same = [(fam, m) for fam, m in cands if m.pitch == n.pitch and _overlap(n, m) > 0]
        if same:
            fam = max(same, key=lambda fm: _overlap(n, fm[1]))[0]
        else:
            near = [(abs(m.start - n.start) + 0.05 * abs(m.pitch - n.pitch), fam)
                    for fam, m in window(n.start - context_s, n.start + context_s)
                    if abs(m.pitch - n.pitch) <= context_semitones]
            fam = min(near)[1] if near else OTHER
        out[fam].append(n)

    if add_missing:
        for fam, m in ref:
            if not any(abs(n.start - m.start) <= onset_tol and n.pitch == m.pitch
                       for n in notes if abs(n.start - m.start) <= onset_tol):
                out[fam].append(m)

    total = sum(len(v) for v in out.values()) or 1
    for fam in [f for f, v in out.items() if f != OTHER and len(v) / total < min_share]:
        out[OTHER].extend(out.pop(fam))
    for v in out.values():
        v.sort(key=lambda n: (n.start, n.pitch))
    return {k: v for k, v in out.items() if v}


def _drop_harmonics(notes: list[Note], tol: float = 0.03) -> list[Note]:
    """Drop notes an octave / 12th / two octaves above a louder note starting together
    (Basic Pitch often reports a bright synth's overtones as extra notes)."""
    import bisect

    notes = sorted(notes, key=lambda n: n.start)
    starts = [n.start for n in notes]
    keep = []
    for n in notes:
        lo, hi = bisect.bisect_left(starts, n.start - tol), bisect.bisect_right(starts, n.start + tol)
        if not any(notes[i].velocity >= n.velocity and n.pitch - notes[i].pitch in (12, 19, 24) for i in range(lo, hi)):
            keep.append(n)
    return keep


def lead_and_pad(stem, sr: int) -> dict[str, list[Note]]:
    """Split the "other" stem into the two layers most electronic tracks are built from:

    * Synth Lead - short notes (<= 0.3 s) above G#3, one at a time: arps, riffs, stabs.
      Notes down to 30 ms are kept, so 32nd-note arpeggios survive.
    * Synth Pad  - sustained notes (>= 0.25 s), fragments up to 0.3 s apart joined.
    """
    from .pitch import make_monophonic, merge_fragments, transcribe_pitched

    lead = transcribe_pitched(stem, sr, onset_threshold=0.4, frame_threshold=0.25, min_note_ms=30, min_freq=55.0)
    lead = [n for n in lead if n.pitch >= 56 and n.end - n.start <= 0.3]
    lead = make_monophonic(_drop_harmonics(lead))

    pad = transcribe_pitched(stem, sr, onset_threshold=0.25, frame_threshold=0.15, min_note_ms=30, min_freq=55.0)
    pad = merge_fragments(pad, max_gap=0.3)
    pad = _drop_harmonics([n for n in pad if n.end - n.start >= 0.25])
    return {"Synth Lead": lead, "Synth Pad": sorted(pad, key=lambda n: (n.start, n.pitch))}
