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
from .separation import best_separation, separate, separation_label
from .timing import estimate_tempo, grid_phase, grid_step, quantize

ProgressFn = Callable[[str, float], None]


@dataclass
class TranscriptionOptions:
    # "auto" picks the best engine that is installed; see resolve_engines().
    separation: str = "auto"  # "auto", "hpss", or a neural model key from separation.NEURAL_MODELS
    drum_engine: str = "auto"  # "auto", "ensemble", "drumsep", "adtof" (neural) or "dsp"
    mono_engine: str = "auto"  # bass/vocal lines: "auto", "crepe" (neural), "pyin" or "basic-pitch"
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
    elif separation != "hpss":
        mono = "basic-pitch"  # on clean Demucs bass stems it beats CREPE (bench F1 0.85 vs 0.68)
    else:
        mono = "pyin"  # Basic Pitch makes octave errors on HPSS's leaky low end
    return replace(
        opts,
        separation=separation,
        drum_engine=_best_drum_engine() if opts.drum_engine == "auto" else opts.drum_engine,
        mono_engine=mono,
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
        tracks.append(Track("Drums", 0, True, drum_fn(stems["drums"], stem_sr, opts.drum_sensitivity)))

    profiles = PROFILES["neural" if neural else "dsp"]
    wanted = [k for k in profiles if k in stems and (k != "bass" or opts.bass) and (k == "bass" or opts.melodic)]
    for i, key in enumerate(wanted):
        p = profiles[key]
        frac = 0.55 + 0.4 * i / len(wanted)
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
    engines = {"separation": opts.separation, "drums": opts.drum_engine, "bass": opts.mono_engine, "synths": "basic-pitch"}
    return TranscriptionResult(tracks=tracks, bpm=bpm, duration=duration, engines=engines)
