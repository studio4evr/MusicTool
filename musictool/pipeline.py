"""End-to-end pipeline: audio file -> separated stems -> notes -> MIDI."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pretty_midi

from .audio import load_audio
from .drums import (
    adtof_available,
    drumsep_available,
    transcribe_drums,
    transcribe_drums_adtof,
    transcribe_drums_drumsep,
    transcribe_drums_ensemble,
)
from .notes import Note
from .pitch import (
    make_monophonic,
    merge_fragments,
    remove_octave_bleed,
    transcribe_monophonic,
    transcribe_pitched,
)
from .instruments import lead_and_pad
from .mt3 import ymt3_available
from .separation import best_separation, separate, separation_label
from .timing import estimate_tempo, grid_phase, grid_step, quantize

ProgressFn = Callable[[str, float], None]


@dataclass
class TranscriptionOptions:
    # "auto" picks the best engine that is installed; see resolve_engines().
    separation: str = "auto"  # "auto", "hpss", or a neural model key from separation.NEURAL_MODELS
    drum_engine: str = "auto"  # "auto", "ensemble", "drumsep", "adtof" (neural) or "dsp"
    mono_engine: str = "auto"  # bass/vocal lines: "auto", "crepe" (neural), "pyin" or "basic-pitch"
    # synths: "auto", "hybrid" (Basic Pitch notes split into instruments by YourMT3+),
    # "yourmt3" (YourMT3+ alone) or "basic-pitch" (one Synths track)
    synth_engine: str = "auto"
    # YourMT3+ checkpoints (mt3.VARIANTS) and what they listen to ("full" mix or "no-drums").
    # Measured on bench/ and against a commercial transcription of a real track
    # (bench/compare_configs.py): the MoE model labels synth instruments best (synth F1
    # 0.38 bench / 0.20 real), the pitch-shift-trained PTF model finds bass far better
    # (bass F1 0.64 vs 0.34 for Basic Pitch on the real track, 0.80 vs 0.84 bench).
    mt3_variant: str = "moe-nops"  # instrument labels for the synth split
    mt3_input: str = "no-drums"
    bass_mt3_variant: str = "ptf-ps"  # bass, when mono_engine == "yourmt3"
    bass_mt3_input: str = "full"
    # Hybrid: Basic Pitch notes YourMT3+ hears no instrument near. None = drop them,
    # otherwise keep those at or above this velocity in an "Synths" track.
    unlabelled_min_velocity: int | None = 0
    hybrid_add_missing: bool = True  # hybrid: also add the YourMT3+ notes Basic Pitch missed
    # extra YourMT3+ passes, (variant, input), merged into the instrument labels
    extra_label_passes: tuple[tuple[str, str], ...] = ()
    # YourMT3+ bass: fill its silent gaps with Basic Pitch notes from these sources
    # ("stem" = the Demucs bass stem, "low-synth" = synth notes below synth_min_pitch).
    bass_fill: tuple[str, ...] = ()
    # Clean-up of the synth tracks. None = auto (on for neural stems, off for HPSS), 0 = off.
    # With the bass on its own track, synth notes below C3 mostly double it, and quiet
    # Basic Pitch notes are mostly reverb tails / overtones. Dropping both: synth F1 0.37 ->
    # 0.48 on bench/, 0.20 -> 0.26 against a commercial transcription of a real track.
    synth_min_pitch: int | None = None  # 48 = C3
    synth_min_velocity: int | None = None  # 40
    bpm: float | None = None  # None = detect automatically
    quantize: int | None = 16  # grid division, None/0 = off
    drums: bool = True
    bass: bool = True
    melodic: bool = True
    # None = the tuned per-stem defaults in PROFILES; set a value to override every stem.
    onset_threshold: float | None = None  # Basic Pitch note onset confidence
    frame_threshold: float | None = None  # Basic Pitch note sustain confidence
    min_note_ms: float | None = None
    drum_sensitivity: float = 0.5
    stems_dir: Path | None = None  # if set, the separated stems are written here as WAV


DRUM_ENGINES = {
    "ensemble": ("DrumSep + ADTOF", transcribe_drums_ensemble),
    "drumsep": ("DrumSep", transcribe_drums_drumsep),
    "adtof": ("ADTOF neural net", transcribe_drums_adtof),
    "dsp": ("DSP", transcribe_drums),
}


def _best_drum_engine() -> str:
    if drumsep_available() and adtof_available():
        return "ensemble"
    if drumsep_available():
        return "drumsep"
    return "adtof" if adtof_available() else "dsp"


def resolve_engines(opts: TranscriptionOptions) -> TranscriptionOptions:
    """Replace every "auto" with the best engine available on this machine."""
    separation = best_separation() if opts.separation == "auto" else opts.separation
    if opts.mono_engine != "auto":
        mono = opts.mono_engine
    elif separation != "hpss" and ymt3_available(opts.bass_mt3_variant):
        mono = "yourmt3"  # see bass_mt3_variant
    elif separation != "hpss":
        mono = "basic-pitch"  # on clean Demucs bass stems it beats CREPE (bench F1 0.85 vs 0.68)
    else:
        mono = "pyin"  # Basic Pitch makes octave errors on HPSS's leaky low end
    if opts.synth_engine != "auto":
        synth = opts.synth_engine
    else:  # the hybrid needs clean stems to subtract the drums from the mix
        synth = "hybrid" if separation != "hpss" and ymt3_available() else "basic-pitch"
    return replace(
        opts,
        separation=separation,
        drum_engine=_best_drum_engine() if opts.drum_engine == "auto" else opts.drum_engine,
        mono_engine=mono,
        synth_engine=synth,
        synth_min_pitch=(48 if separation != "hpss" else 0) if opts.synth_min_pitch is None else opts.synth_min_pitch,
        synth_min_velocity=(40 if separation != "hpss" else 0) if opts.synth_min_velocity is None else opts.synth_min_velocity,
    )


@dataclass
class Track:
    name: str
    program: int  # General MIDI program number (0-127)
    is_drum: bool
    notes: list[Note] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "program": self.program,
            "is_drum": self.is_drum,
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class TranscriptionResult:
    tracks: list[Track]
    bpm: float
    duration: float
    engines: dict[str, str] = field(default_factory=dict)  # which model handled each step

    def to_midi(self, track_names: list[str] | None = None) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        for track in self.tracks:
            if track_names is not None and track.name not in track_names:
                continue
            inst = pretty_midi.Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
            for n in track.notes:
                inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end))
            pm.instruments.append(inst)
        return pm

    def write(self, path, track_names: list[str] | None = None) -> Path:
        path = Path(path)
        self.to_midi(track_names).write(str(path))
        return path

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "duration": round(self.duration, 3),
            "engines": self.engines,
            "tracks": [t.to_dict() for t in self.tracks],
        }


@dataclass(frozen=True)
class StemProfile:
    name: str
    program: int  # GM program
    fmin: float
    fmax: float | None
    mono: bool
    onset: float = 0.5  # Basic Pitch thresholds
    frame: float = 0.3
    min_note_ms: float = 80.0


# Tuned with bench/benchmark.py (ground truth) + bench/realtrack.py (a real Aphex Twin track).
PROFILES = {
    # Clean neural stems. Synths go down to 55 Hz (110 Hz dropped half of the real track's
    # synth parts) with looser thresholds: real-track synth coverage 0.50 -> 0.94, bench F1
    # unchanged. The vocal stem is transcribed polyphonically: on instrumental electronic
    # music it mostly holds synth leads. Bass via Basic Pitch with 50 ms notes: bench F1
    # 0.85 vs 0.68 for CREPE, same real-track coverage.
    "neural": {
        "bass": StemProfile("Bass", 38, 27.5, 220.0, True, min_note_ms=50.0),
        "melodic": StemProfile("Synths", 81, 55.0, None, False, onset=0.4, frame=0.25),
        "vocals": StemProfile("Vocals", 54, 55.0, None, False, onset=0.4, frame=0.25),
    },
    # HPSS leaks bass harmonics upwards, so synths stay above 110 Hz.
    "dsp": {
        "bass": StemProfile("Bass", 38, 27.5, 220.0, True),
        "melodic": StemProfile("Synths", 81, 110.0, None, False),
        "vocals": StemProfile("Vocals", 54, 70.0, 1100.0, True),
    },
}


def transcribe(path, options: TranscriptionOptions | None = None, progress: ProgressFn | None = None) -> TranscriptionResult:
    """Transcribe an audio file into a multi-track ``TranscriptionResult``."""
    opts = resolve_engines(options or TranscriptionOptions())
    report = progress or (lambda stage, frac: None)

    report("Loading audio", 0.02)
    y, sr = load_audio(path)
    duration = len(y) / sr

    report("Detecting tempo", 0.06)
    detected_bpm, first_beat = estimate_tempo(y, sr)
    bpm = float(opts.bpm) if opts.bpm else detected_bpm

    neural = opts.separation != "hpss"
    report(f"Separating stems ({separation_label(opts.separation)})", 0.1)
    if neural:
        stems, stem_sr = _cached((_file_key(path), "stems", opts.separation),
                                 lambda: separate(y, sr, opts.separation, path=path))
    else:
        stems, stem_sr = separate(y, sr, opts.separation, path=path)
    if opts.stems_dir:
        import soundfile as sf

        Path(opts.stems_dir).mkdir(parents=True, exist_ok=True)
        for key, audio in stems.items():
            sf.write(Path(opts.stems_dir) / f"{Path(path).stem}_{key}.wav", audio, stem_sr)

    tracks: list[Track] = []
    if opts.drums and "drums" in stems:
        label, drum_fn = DRUM_ENGINES[opts.drum_engine]
        report(f"Transcribing drums ({label})", 0.4)
        drum_notes = _cached((_file_key(path), "drums", opts.separation, opts.drum_engine, opts.drum_sensitivity),
                             lambda: drum_fn(stems["drums"], stem_sr, opts.drum_sensitivity))
        tracks.append(Track("Drums", 0, True, [replace(n) for n in drum_notes]))

    mt3_stems = stems if neural else None
    labelled = bass_labelled = None
    if opts.melodic and opts.synth_engine in ("hybrid", "yourmt3"):
        report(f"Labelling instruments (YourMT3+ {opts.mt3_variant})", 0.45)
        labelled = _run_mt3(path, mt3_stems, opts.mt3_variant, opts.mt3_input)
        for variant, source in opts.extra_label_passes:
            labelled = _merge_labels(labelled, _run_mt3(path, mt3_stems, variant, source))
    if opts.bass and opts.mono_engine == "yourmt3":
        report(f"Transcribing bass (YourMT3+ {opts.bass_mt3_variant})", 0.5)
        bass_labelled = _run_mt3(path, mt3_stems, opts.bass_mt3_variant, opts.bass_mt3_input)

    mt3_bass_track = None
    profiles = PROFILES["neural" if neural else "dsp"]
    wanted = [k for k in profiles if k in stems and (k != "bass" or opts.bass) and (k == "bass" or opts.melodic)]
    if opts.synth_engine == "layers":
        wanted = [k for k in wanted if k == "bass"]
    for i, key in enumerate(wanted):
        p = profiles[key]
        frac = 0.55 + 0.4 * i / len(wanted)
        if key == "bass" and opts.mono_engine == "yourmt3":
            mt3_bass = list(bass_labelled.get("Bass", []))
            if _coverage(mt3_bass, stems[key], stem_sr) >= 0.3:
                if "stem" in opts.bass_fill:
                    stem_bass = transcribe_pitched(stems[key], stem_sr, min_note_ms=p.min_note_ms, min_freq=p.fmin,
                                                   max_freq=p.fmax, monophonic=True)
                    mt3_bass += _in_gaps(stem_bass, mt3_bass)
                tracks.append(Track(p.name, p.program, False, sorted(mt3_bass, key=lambda n: n.start)))
                mt3_bass_track = tracks[-1]
                continue
            # YourMT3+ didn't call this sound a bass (e.g. a plain synth bass on a sparse
            # track): fall back to Basic Pitch on the Demucs bass stem.
        min_ms = opts.min_note_ms if opts.min_note_ms is not None else p.min_note_ms
        if p.mono and opts.mono_engine in ("pyin", "crepe"):
            label = "CREPE neural net" if opts.mono_engine == "crepe" else "pYIN"
            report(f"Transcribing {p.name.lower()} ({label})", frac)
            notes = transcribe_monophonic(
                stems[key], stem_sr, min_freq=p.fmin, max_freq=p.fmax, min_note_ms=min_ms, engine=opts.mono_engine
            )
        else:
            report(f"Transcribing {p.name.lower()} (Basic Pitch neural net)", frac)
            notes = transcribe_pitched(
                stems[key],
                stem_sr,
                onset_threshold=opts.onset_threshold if opts.onset_threshold is not None else p.onset,
                frame_threshold=opts.frame_threshold if opts.frame_threshold is not None else p.frame,
                min_note_ms=min_ms,
                min_freq=p.fmin,
                max_freq=p.fmax,
                monophonic=p.mono,
            )
        if not p.mono:
            notes = merge_fragments(notes)
        tracks.append(Track(p.name, p.program, False, notes))
    mono_tracks = {p.name for p in profiles.values() if p.mono}

    if opts.melodic and opts.synth_engine == "layers" and "melodic" in stems:
        report("Transcribing lead and pad layers (Basic Pitch)", 0.85)
        for name, notes in lead_and_pad(stems["melodic"], stem_sr).items():
            tracks.append(Track(name, 81 if name == "Synth Lead" else 89, False, notes))
        mono_tracks = ({"Bass"} & mono_tracks) | {"Synth Lead"}
    if opts.melodic and opts.synth_engine in ("hybrid", "yourmt3"):
        report("Splitting synths by instrument", 0.9)
        tracks = _split_synths_by_instrument(tracks, labelled, opts.synth_engine, opts.unlabelled_min_velocity,
                                             opts.hybrid_add_missing)
        mono_tracks = {"Bass"} & mono_tracks

    if opts.synth_min_pitch or opts.synth_min_velocity:
        low_synth = []
        for t in tracks:
            if t.is_drum or t.name == "Bass":
                continue
            low_synth += [n for n in t.notes if n.pitch < opts.synth_min_pitch and n.velocity >= opts.synth_min_velocity]
            t.notes = [n for n in t.notes if n.pitch >= opts.synth_min_pitch and n.velocity >= opts.synth_min_velocity]
        tracks = [t for t in tracks if t.is_drum or t.name == "Bass" or t.notes]
        if mt3_bass_track is not None and "low-synth" in opts.bass_fill:
            # A bass line Demucs filed under "other" (common with synth basses).
            mt3_bass_track.notes = sorted(mt3_bass_track.notes + _in_gaps(low_synth, mt3_bass_track.notes),
                                          key=lambda n: n.start)

    bass_track = next((t for t in tracks if t.name == "Bass"), None)
    if bass_track is not None and not neural:  # neural stems don't leak bass harmonics
        for track in tracks:
            if track.name == "Synths":
                track.notes = remove_octave_bleed(track.notes, bass_track.notes)

    if opts.quantize:
        report("Quantising", 0.97)
        step = grid_step(bpm, opts.quantize)
        reference = next((t for t in tracks if t.is_drum and t.notes), None)
        onsets = [n.start for t in ([reference] if reference else tracks) for n in t.notes]
        phase = grid_phase(onsets, step) if onsets else first_beat
        for track in tracks:
            track.notes = quantize(track.notes, bpm, opts.quantize, phase)
            if track.name in mono_tracks:
                track.notes = make_monophonic(track.notes)

    report("Done", 1.0)
    engines = {
        "separation": opts.separation,
        "drums": opts.drum_engine,
        "bass": opts.mono_engine,
        "synths": opts.synth_engine if opts.melodic else "-",
    }
    return TranscriptionResult(tracks=tracks, bpm=bpm, duration=duration, engines=engines)


SYNTH_SOURCE_TRACKS = ("Synths", "Vocals")


def _merge_labels(a: dict[str, list[Note]], b: dict[str, list[Note]], tol: float = 0.05) -> dict[str, list[Note]]:
    """Union of two YourMT3+ passes; a note of `b` is skipped if `a` has the same pitch within tol."""
    import bisect

    flat = sorted((n for k, v in a.items() if k not in ("Drums", "Bass") for n in v), key=lambda n: n.start)
    starts = [n.start for n in flat]
    out = {k: list(v) for k, v in a.items()}
    for k, v in b.items():
        if k in ("Drums", "Bass"):
            continue
        for n in v:
            lo, hi = bisect.bisect_left(starts, n.start - tol), bisect.bisect_right(starts, n.start + tol)
            if not any(flat[i].pitch == n.pitch for i in range(lo, hi)):
                out.setdefault(k, []).append(n)
    return out


def _in_gaps(candidates: list[Note], line: list[Note], margin: float = 0.05) -> list[Note]:
    """Candidates that start while no note of `line` is sounding (+- margin)."""
    import bisect

    line = sorted(line, key=lambda n: n.start)
    starts = [n.start for n in line]
    out = []
    for c in candidates:
        i = bisect.bisect_right(starts, c.start + margin)
        sounding = any(line[j].start - margin <= c.start <= line[j].end + margin for j in range(max(0, i - 8), i))
        if not sounding:
            out.append(c)
    return out


def _coverage(notes: list[Note], stem, sr: int, floor_db: float = -30.0) -> float:
    """Share of the time `stem` is audible (within floor_db of its loud level) that has a note."""
    import librosa
    import numpy as np

    hop = 1024
    rms = librosa.feature.rms(y=stem, frame_length=2 * hop, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    audible = db > np.percentile(db, 95) + floor_db
    if not audible.any():
        return 1.0
    active = np.zeros_like(audible)
    for n in notes:
        active[int(n.start * sr / hop): int(n.end * sr / hop) + 1] = True
    return float((audible & active).sum() / audible.sum())


_cache: dict = {}


def _cached(key, fn):
    """Memoise the slow neural steps per (file, mtime, settings), so re-running a
    file with other options (web UI, benchmarks) skips separation and YourMT3+."""
    if key not in _cache:
        if len(_cache) >= 12:
            _cache.pop(next(iter(_cache)))
        _cache[key] = fn()
    return _cache[key]


def _file_key(path) -> tuple:
    p = Path(path).resolve()
    return (str(p), p.stat().st_mtime_ns)


def _run_mt3(path, stems: dict | None, variant: str, source: str) -> dict[str, list[Note]]:
    source = source if stems is not None else "full"
    key = (_file_key(path), "mt3", variant, source)
    return {k: list(v) for k, v in _cached(key, lambda: _run_mt3_uncached(path, stems, variant, source)).items()}


def _run_mt3_uncached(path, stems: dict | None, variant: str, source: str) -> dict[str, list[Note]]:
    """YourMT3+ on the full mix, or on the mix minus the Demucs drum stem."""
    import librosa

    from .mt3 import transcribe_mt3
    from .separation import NEURAL_SR

    mix, _ = librosa.load(str(path), sr=NEURAL_SR, mono=True)
    if source == "no-drums" and stems is not None and "drums" in stems:
        n = min(len(mix), len(stems["drums"]))
        mix = mix[:n] - stems["drums"][:n]
    return transcribe_mt3(mix, NEURAL_SR, variant=variant)


def _split_synths_by_instrument(tracks: list[Track], labelled: dict[str, list[Note]], engine: str,
                                unlabelled_min_velocity: int | None = 0, add_missing: bool = True) -> list[Track]:
    """Replace the Synths / Vocals tracks with one track per instrument."""
    from .instruments import OTHER, assign_instruments
    from .mt3 import TRACK_PROGRAMS

    kept = [t for t in tracks if t.name not in SYNTH_SOURCE_TRACKS]
    if engine == "yourmt3":
        groups = {k: v for k, v in labelled.items() if k not in ("Drums", "Bass")}
    else:
        base = [n for t in tracks if t.name in SYNTH_SOURCE_TRACKS for n in t.notes]
        groups = assign_instruments(base, labelled, add_missing=add_missing)
        rest = groups.pop(OTHER, [])
        if unlabelled_min_velocity is not None:
            rest = [n for n in rest if n.velocity >= unlabelled_min_velocity]
            if rest:
                groups[OTHER] = rest
    ordered = sorted(groups.items(), key=lambda kv: (kv[0] == OTHER, -len(kv[1])))
    return kept + [Track(name, TRACK_PROGRAMS.get(name, 81), False, notes) for name, notes in ordered]
