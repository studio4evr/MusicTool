"""End-to-end pipeline: audio file -> separated stems -> notes -> MIDI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pretty_midi

from .audio import load_audio
from .drums import transcribe_drums
from .notes import Note
from .pitch import (
    make_monophonic,
    merge_fragments,
    remove_octave_bleed,
    transcribe_monophonic,
    transcribe_pitched,
)
from .separation import separate
from .timing import estimate_tempo, grid_phase, grid_step, quantize

ProgressFn = Callable[[str, float], None]


@dataclass
class TranscriptionOptions:
    separation: str = "hpss"  # "hpss" or "demucs"
    bpm: float | None = None  # None = detect automatically
    quantize: int | None = 16  # grid division, None/0 = off
    drums: bool = True
    bass: bool = True
    melodic: bool = True
    onset_threshold: float = 0.5  # Basic Pitch note onset confidence
    frame_threshold: float = 0.3  # Basic Pitch note sustain confidence
    min_note_ms: float = 80.0
    drum_sensitivity: float = 0.5
    mono_engine: str = "pyin"  # bass/vocal line engine: "pyin" or "basic-pitch"


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
    opts = options or TranscriptionOptions()
    report = progress or (lambda stage, frac: None)

    report("Loading audio", 0.02)
    y, sr = load_audio(path)
    duration = len(y) / sr

    report("Detecting tempo", 0.08)
    detected_bpm, first_beat = estimate_tempo(y, sr)
    bpm = float(opts.bpm) if opts.bpm else detected_bpm

    report(f"Separating stems ({opts.separation})", 0.15)
    stems = separate(y, sr, opts.separation, path=path)

    tracks: list[Track] = []
    if opts.drums and "drums" in stems:
        report("Transcribing drums", 0.35)
        tracks.append(Track("Drums", 0, True, transcribe_drums(stems["drums"], sr, opts.drum_sensitivity)))

    wanted = [k for k in PITCHED_STEMS if k in stems and (k != "bass" or opts.bass) and (k == "bass" or opts.melodic)]
    for i, key in enumerate(wanted):
        name, program, (fmin, fmax), mono = PITCHED_STEMS[key]
        if mono and opts.mono_engine == "pyin":
            report(f"Transcribing {name.lower()} (pYIN)", 0.45 + 0.5 * i / len(wanted))
            notes = transcribe_monophonic(stems[key], sr, min_freq=fmin, max_freq=fmax, min_note_ms=opts.min_note_ms)
        else:
            report(f"Transcribing {name.lower()} (Basic Pitch AI)", 0.45 + 0.5 * i / len(wanted))
            notes = transcribe_pitched(
                stems[key],
                sr,
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
    if bass_track is not None and opts.separation == "hpss":
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
    return TranscriptionResult(tracks=tracks, bpm=bpm, duration=duration)
