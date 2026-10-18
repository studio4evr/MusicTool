"""End-to-end pipeline: audio file -> separated stems -> notes -> MIDI."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import pretty_midi

from .audio import load_audio
from .drums import adtof_available, transcribe_drums, transcribe_drums_adtof
from .notes import Note
from .pitch import (
    crepe_available,
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
    drum_engine: str = "auto"  # "auto", "adtof" (neural) or "dsp"
    mono_engine: str = "auto"  # bass/vocal lines: "auto", "crepe" (neural), "pyin" or "basic-pitch"
    bpm: float | None = None  # None = detect automatically
    quantize: int | None = 16  # grid division, None/0 = off
    drums: bool = True
    bass: bool = True
    melodic: bool = True
    onset_threshold: float = 0.5  # Basic Pitch note onset confidence
    frame_threshold: float = 0.3  # Basic Pitch note sustain confidence
    min_note_ms: float = 80.0
    drum_sensitivity: float = 0.5
    stems_dir: Path | None = None  # if set, the separated stems are written here as WAV


def resolve_engines(opts: TranscriptionOptions) -> TranscriptionOptions:
    """Replace every "auto" with the best engine available on this machine."""
    return replace(
        opts,
        separation=best_separation() if opts.separation == "auto" else opts.separation,
        drum_engine=("adtof" if adtof_available() else "dsp") if opts.drum_engine == "auto" else opts.drum_engine,
        mono_engine=("crepe" if crepe_available() else "pyin") if opts.mono_engine == "auto" else opts.mono_engine,
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


# Track name -> (GM program, is_drum, Basic Pitch frequency range, monophonic)
PITCHED_STEMS = {
    "bass": ("Bass", 38, (27.5, 220.0), True),  # Synth Bass 1
    "melodic": ("Synths", 81, (110.0, None), False),  # Lead 2 (sawtooth)
    "vocals": ("Vocals", 54, (70.0, 1100.0), True),  # Synth Voice
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
        if opts.drum_engine == "adtof":
            report("Transcribing drums (ADTOF neural net)", 0.45)
            drum_notes = transcribe_drums_adtof(stems["drums"], stem_sr, opts.drum_sensitivity)
        else:
            report("Transcribing drums", 0.45)
            drum_notes = transcribe_drums(stems["drums"], stem_sr, opts.drum_sensitivity)
        tracks.append(Track("Drums", 0, True, drum_notes))

    wanted = [k for k in PITCHED_STEMS if k in stems and (k != "bass" or opts.bass) and (k == "bass" or opts.melodic)]
    for i, key in enumerate(wanted):
        name, program, (fmin, fmax), mono = PITCHED_STEMS[key]
        frac = 0.55 + 0.4 * i / len(wanted)
        if mono and opts.mono_engine in ("pyin", "crepe"):
            label = "CREPE neural net" if opts.mono_engine == "crepe" else "pYIN"
            report(f"Transcribing {name.lower()} ({label})", frac)
            notes = transcribe_monophonic(
                stems[key], stem_sr, min_freq=fmin, max_freq=fmax, min_note_ms=opts.min_note_ms, engine=opts.mono_engine
            )
        else:
            report(f"Transcribing {name.lower()} (Basic Pitch neural net)", frac)
            notes = transcribe_pitched(
                stems[key],
                stem_sr,
                onset_threshold=opts.onset_threshold,
                frame_threshold=opts.frame_threshold,
                min_note_ms=opts.min_note_ms,
                min_freq=fmin,
                max_freq=fmax or None,
                monophonic=mono,
            )
        if not mono:
            notes = merge_fragments(notes)
        tracks.append(Track(name, program, False, notes))

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
            if track.name in ("Bass", "Vocals"):
                track.notes = make_monophonic(track.notes)

    report("Done", 1.0)
    engines = {"separation": opts.separation, "drums": opts.drum_engine, "bass": opts.mono_engine, "synths": "basic-pitch"}
    return TranscriptionResult(tracks=tracks, bpm=bpm, duration=duration, engines=engines)
