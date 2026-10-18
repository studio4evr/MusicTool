import pretty_midi
import pytest

import synth
from conftest import DSP
from musictool import Note, TranscriptionOptions, transcribe
from musictool.pitch import make_monophonic, merge_fragments, remove_octave_bleed
from musictool.timing import estimate_tempo, grid_phase, quantize

TOL = 0.05  # seconds


def f1(found: list[float], expected: list[float], tol: float = TOL) -> float:
    unmatched = list(expected)
    hits = 0
    for t in found:
        match = next((e for e in unmatched if abs(e - t) <= tol), None)
        if match is not None:
            unmatched.remove(match)
            hits += 1
    if not found or not expected:
        return 0.0
    precision, recall = hits / len(found), hits / len(expected)
    return 2 * precision * recall / (precision + recall) if hits else 0.0


def track_named(result, name):
    return next(t for t in result.tracks if t.name == name)


def test_tempo_detection(track):
    _, y, sr, _ = track
    bpm, _ = estimate_tempo(y, sr)
    assert bpm == pytest.approx(synth.BPM, abs=0.5)


@pytest.mark.parametrize("pitch,kind", [(36, "kick"), (38, "snare"), (42, "hat")])
def test_drums(result, track, pitch, kind):
    truth = track[3][kind]
    found = [n.start for n in track_named(result, "Drums").notes if n.pitch == pitch]
    assert f1(found, truth) >= 0.9


def test_bass_line(result, track):
    truth = track[3]["bass"]
    notes = track_named(result, "Bass").notes
    correct = sum(
        any(abs(n.start - t) <= TOL and n.pitch == p for n in notes) for t, p in truth
    )
    assert correct / len(truth) >= 0.9
    assert len(notes) <= len(truth) * 1.2


def test_synth_chords(result, track):
    truth = track[3]["chords"]
    notes = track_named(result, "Synths").notes
    found = sum(any(abs(n.start - t) <= TOL and n.pitch == p for n in notes) for t, p in truth)
    assert found / len(truth) >= 0.75


def test_quantised_output_is_on_grid(track):
    res = transcribe(track[0], TranscriptionOptions(quantize=16, melodic=False, **DSP))
    step = 60 / res.bpm / 4
    starts = [n.start for t in res.tracks for n in t.notes]
    phase = grid_phase(starts, step)
    assert all(abs(((s - phase) / step) - round((s - phase) / step)) < 1e-6 for s in starts)


def test_midi_file(result, tmp_path):
    path = result.write(tmp_path / "out.mid")
    pm = pretty_midi.PrettyMIDI(str(path))
    names = [i.name for i in pm.instruments]
    assert names == ["Drums", "Bass", "Synths"]
    assert pm.instruments[0].is_drum
    assert sum(len(i.notes) for i in pm.instruments) == sum(len(t.notes) for t in result.tracks)

    single = pretty_midi.PrettyMIDI(str(result.write(tmp_path / "bass.mid", ["Bass"])))
    assert [i.name for i in single.instruments] == ["Bass"]


def test_track_toggles(track):
    res = transcribe(track[0], TranscriptionOptions(drums=False, melodic=False, **DSP))
    assert [t.name for t in res.tracks] == ["Bass"]


def test_silence(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(22050 * 2, dtype=np.float32), 22050)
    res = transcribe(path, TranscriptionOptions(**DSP))
    assert all(not t.notes for t in res.tracks)


def test_quantize_snaps_and_dedupes():
    notes = [Note(0.01, 0.2, 60, 100), Note(0.02, 0.1, 60, 90), Note(0.13, 0.14, 62, 80)]
    q = quantize(notes, bpm=120, division=16)  # 0.125 s grid
    assert [(n.start, n.pitch) for n in q] == [(0.0, 60), (0.125, 62)]
    assert q[1].end == pytest.approx(0.25)  # minimum one grid step long


def test_grid_phase_handles_wraparound():
    step = 0.125
    onsets = [k * step - 0.005 for k in range(1, 20)] + [k * step + 0.005 for k in range(1, 20)]
    phase = grid_phase(onsets, step)
    assert min(phase, step - phase) < 1e-3


def test_make_monophonic():
    notes = [Note(0, 1, 40, 50), Note(0.01, 0.5, 52, 90), Note(0.5, 1, 43, 80)]
    mono = make_monophonic(notes)
    assert [(n.pitch, n.end) for n in mono] == [(52, 0.5), (43, 1)]


def test_merge_fragments():
    notes = [Note(0, 0.5, 60, 70), Note(0.51, 1.0, 60, 90), Note(1.5, 2, 60, 60)]
    merged = merge_fragments(notes)
    assert [(n.start, n.end, n.velocity) for n in merged] == [(0, 1.0, 90), (1.5, 2, 60)]


def test_remove_octave_bleed():
    bass = [Note(0, 0.5, 36, 100)]
    melodic = [Note(0.01, 0.45, 48, 60), Note(0.01, 0.45, 51, 60), Note(1.0, 1.5, 48, 60)]
    kept = remove_octave_bleed(melodic, bass)
    assert [(n.start, n.pitch) for n in kept] == [(0.01, 51), (1.0, 48)]
