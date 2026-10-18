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


class _Index:
    def __init__(self, notes):
        import bisect

        self._bisect = bisect
        self.notes = sorted(notes, key=lambda n: n.start)
        self.starts = [n.start for n in self.notes]

    def match(self, note: Note, tol: float = 0.05) -> list[Note]:
        lo = self._bisect.bisect_left(self.starts, note.start - tol)
        hi = self._bisect.bisect_right(self.starts, note.start + tol)
        return [self.notes[i] for i in range(lo, hi) if self.notes[i].pitch == note.pitch]


def _sounding(index: "_Index", note: Note, lookback: float = 0.5) -> list[Note]:
    """Notes of `index` at the same pitch that are sounding when `note` starts."""
    import bisect

    lo = bisect.bisect_left(index.starts, note.start - lookback)
    hi = bisect.bisect_right(index.starts, note.start + 0.08)
    return [index.notes[i] for i in range(lo, hi)
            if index.notes[i].pitch == note.pitch and index.notes[i].end > note.start]


def vote_synths(base: list[Note], passes: list[dict[str, list[Note]]], min_votes: int = 2,
                durations: list[Note] | None = None, sustain_rescue: float | None = None) -> dict[str, list[Note]]:
    """Keep synth notes that at least `min_votes` sources agree on (same pitch, onset
    within 50 ms). Sources: Basic Pitch (`base`) and each YourMT3+ pass.

    Independent models rarely make the *same* mistake, so agreement filters out
    the extra notes (reverb, delays, textures) any single model adds. On a real
    track scored against a commercial transcription, 2-of-5 voting lifted synth
    F1 from ~0.25 (Basic Pitch split by YourMT3+ labels) to 0.33.

    Instrument = the majority YourMT3+ label among the agreeing notes; length =
    the longest agreeing note, extended by Basic Pitch (`durations`), since
    YourMT3+ cuts synth notes short.
    """
    from collections import Counter

    sources = [[(None, n) for n in base]] + [
        [(label, n) for label, ns in p.items() if label not in ("Drums", "Bass") for n in ns] for p in passes
    ]
    indexes = [_Index([n for _, n in src]) for src in sources]
    label_of = {id(n): label for src in sources for label, n in src}
    dur_index = _Index(durations if durations is not None else base)

    pool = sorted((n for src in sources for _, n in src), key=lambda n: (n.start, n.pitch))
    kept: list[Note] = []
    out: dict[str, list[Note]] = {}
    for n in pool:
        if any(m.pitch == n.pitch and n.start - m.start <= 0.05 for m in kept[-40:]):
            continue  # already decided for this note
        matches = [idx.match(n) for idx in indexes]
        votes = sum(bool(m) for m in matches)
        rescued = (sustain_rescue is not None and label_of[id(n)] is None and n.end - n.start >= sustain_rescue)
        if votes < min_votes and not rescued:
            continue
        agreeing = [m for ms in matches for m in ms]
        labels = Counter(label_of[id(m)] for m in agreeing if label_of[id(m)] is not None)
        label = labels.most_common(1)[0][0] if labels else OTHER
        end = max([m.end for m in agreeing] + [m.end for m in _sounding(dur_index, n)])
        velocity = max(m.velocity for m in agreeing)
        kept.append(n)
        out.setdefault(label, []).append(Note(n.start, max(end, n.start + 0.03), n.pitch, min(127, velocity)))
    return out


def find_downbeat(stem, sr: int, bpm: float, beat_phase: float, snares: list[float]) -> float:
    """Time of a bar line (beat 1), from beat phase + harmonic change + snare placement.

    Chords mostly change on beat 1, so the beat offset (mod 4) with the most chroma
    change in the "other" stem wins, among offsets that don't carry the backbeat
    (snares sit on beats 2 and 4, so a snare-heavy offset can't be beat 1).
    """
    import librosa
    import numpy as np

    beat = 60.0 / bpm
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=stem, sr=sr, hop_length=hop)
    novelty = np.r_[0.0, np.linalg.norm(np.diff(chroma, axis=1), axis=0)]
    beats = np.arange(beat_phase, len(stem) / sr, beat)
    frames = np.clip((beats * sr / hop).astype(int), 0, len(novelty) - 1)
    per_beat = np.array([novelty[max(0, f - 2): f + 3].max() for f in frames])
    change = np.array([per_beat[i::4].mean() if len(per_beat[i::4]) else 0.0 for i in range(4)])

    snare_share = np.zeros(4)
    for t in snares:
        k = int(round((t - beat_phase) / beat))
        if abs((t - beat_phase) / beat - k) < 0.25:
            snare_share[k % 4] += 1
    if snare_share.sum() >= 8:
        backbeat = snare_share > 0.6 * snare_share.max()
        if not backbeat.all():
            change[backbeat] = -1.0
    return beat_phase + int(np.argmax(change)) * beat


def bar_chord_pads(stem, sr: int, bpm: float, downbeat: float, share: float = 0.5, min_pitch: int = 48) -> list[Note]:
    """Pads as one chord per bar: pitches Basic Pitch hears sustained through at least
    `share` of a bar, written from bar line to bar line (how pads are usually charted).
    On a real track: pad F1 0.38 against a commercial transcription, vs ~0.15 when
    taking Basic Pitch's own pad onsets, which smear over the attack."""
    import numpy as np

    from .pitch import merge_fragments, transcribe_pitched

    notes = transcribe_pitched(stem, sr, onset_threshold=0.25, frame_threshold=0.15, min_note_ms=100, min_freq=55.0)
    cands = [n for n in merge_fragments(notes, max_gap=0.3) if n.end - n.start >= 0.25 and min_pitch <= n.pitch < 84]
    bar = 4 * 60.0 / bpm
    t0 = downbeat - bar * int(downbeat // bar)
    out = []
    for b in np.arange(t0, len(stem) / sr, bar):
        e = b + bar
        cover: dict[int, float] = {}
        vel: dict[int, int] = {}
        for n in cands:
            ov = min(n.end, e) - max(n.start, b)
            if ov > 0:
                cover[n.pitch] = cover.get(n.pitch, 0.0) + ov
                vel[n.pitch] = max(vel.get(n.pitch, 0), n.velocity)
        out += [Note(float(b), float(e), p, vel[p]) for p, c in cover.items() if c >= share * bar]
    return out
