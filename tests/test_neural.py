"""Neural engines (Demucs, ADTOF, CREPE). Skipped unless the GPU extras are installed."""

import pytest

from musictool import TranscriptionOptions, transcribe
from musictool.drums import adtof_available, transcribe_drums_adtof
from musictool.instruments import OTHER, assign_instruments
from musictool.mt3 import ymt3_available
from musictool.notes import Note
from musictool.pitch import crepe_available, transcribe_monophonic
from musictool.separation import neural_available

from test_pipeline import TOL, f1, track_named

needs_neural = pytest.mark.skipif(
    not (neural_available() and adtof_available() and crepe_available()), reason="GPU extras not installed"
)


@pytest.fixture(scope="module")
def neural_result(track):
    return transcribe(track[0], TranscriptionOptions(quantize=None))


@needs_neural
def test_auto_picks_neural_engines(neural_result):
    assert neural_result.engines == {
        "separation": "demucs-ft",
        "drums": "ensemble",
        "bass": "yourmt3" if ymt3_available("ptf-ps") else "basic-pitch",
        "synths": "vote" if ymt3_available("moe-ps") else ("hybrid" if ymt3_available() else "basic-pitch"),
    }


@needs_neural
@pytest.mark.parametrize("pitch,kind", [(36, "kick"), (38, "snare"), (42, "hat")])
def test_neural_drums(neural_result, track, pitch, kind):
    # tests/synth.py's kit (pure-sine kick, 50 ms noise hats) is far from what the
    # drum models were trained on; bench/benchmark.py measures the realistic case
    # (kick 0.96, snare 0.94, hats 0.92).
    found = [n.start for n in track_named(neural_result, "Drums").notes if n.pitch in (pitch, 46 if pitch == 42 else pitch)]
    assert f1(found, track[3][kind]) >= 0.8


@needs_neural
def test_neural_bass(neural_result, track):
    notes = track_named(neural_result, "Bass").notes
    correct = sum(any(abs(n.start - t) <= TOL and n.pitch == p for n in notes) for t, p in track[3]["bass"])
    assert correct / len(track[3]["bass"]) >= 0.9


@pytest.mark.skipif(not crepe_available(), reason="torchcrepe not installed")
def test_crepe_finds_a_saw_bass():
    # Regression test: with fmin/fmax set, torchcrepe's viterbi decoder reported
    # ~0 voicing confidence, so CREPE transcribed nothing at all.
    import numpy as np

    sr = 22050
    t = np.arange(int(0.5 * sr)) / sr
    y = np.concatenate([0.4 * (2 * ((t * 440 * 2 ** ((p - 69) / 12)) % 1) - 1) for p in (36, 39, 43, 41)])
    notes = transcribe_monophonic(y.astype(np.float32), sr, min_freq=27.5, max_freq=220, engine="crepe")
    assert [n.pitch for n in notes] == [36, 39, 43, 41]


@pytest.mark.skipif(not adtof_available(), reason="ADTOF not installed")
def test_adtof_silence():
    import numpy as np

    assert transcribe_drums_adtof(np.zeros(44100, dtype=np.float32), 44100) == []


def test_assign_instruments():
    labelled = {
        "Synth Lead": [Note(0.0, 1.0, 72, 90)],
        "Synth Pad": [Note(0.0, 4.0, 60, 80), Note(0.0, 4.0, 64, 80)],
        "Drums": [Note(0.0, 0.1, 36, 100)],
    }
    notes = [
        Note(0.0, 1.0, 72, 100),  # same pitch, overlapping -> Synth Lead
        Note(0.5, 3.0, 64, 100),  # same pitch, overlapping -> Synth Pad
        Note(1.5, 2.0, 74, 100),  # no overlap; nearest labelled note in the line -> Synth Lead
        Note(30.0, 31.0, 40, 100),  # nothing nearby -> unlabelled
    ]
    tracks = assign_instruments(notes, labelled, min_share=0.0)
    names = {n.pitch: name for name, ns in tracks.items() for n in ns}
    assert names[72] == "Synth Lead" and names[74] == "Synth Lead"
    assert names[64] == "Synth Pad" and names[40] == OTHER
    assert names[60] == "Synth Pad"  # a labelled note Basic Pitch missed is added
    assert "Drums" not in tracks


@pytest.mark.skipif(not ymt3_available(), reason="YourMT3+ not installed")
def test_hybrid_splits_synths_into_instruments(neural_result):
    names = [t.name for t in neural_result.tracks]
    assert "Drums" in names and "Bass" in names
    pitched = [t for t in neural_result.tracks if not t.is_drum and t.name != "Bass"]
    assert pitched and sum(len(t.notes) for t in pitched) > 0


def test_vote_synths():
    from musictool.instruments import vote_synths

    base = [Note(0.0, 0.5, 60, 80), Note(1.0, 1.2, 64, 80), Note(2.0, 4.0, 67, 80)]
    passes = [
        {"Synth Lead": [Note(0.01, 0.1, 60, 100)], "Bass": [Note(1.0, 1.1, 64, 100)]},
        {"Synth Lead": [Note(0.02, 0.1, 60, 100)], "Synth Pad": [Note(3.0, 3.1, 72, 100)]},
    ]
    out = vote_synths(base, passes, min_votes=2, sustain_rescue=1.0)  # rescue is opt-in
    pitches = {n.pitch: (lab, n) for lab, ns in out.items() for n in ns}
    assert pitches[60][0] == "Synth Lead" and pitches[60][1].end == 0.5  # 3 votes, Basic Pitch length
    assert 64 not in pitches  # one vote (a Bass-labelled pass note doesn't count for synths)
    assert 72 not in pitches  # one vote
    assert 67 in pitches  # long Basic Pitch note rescued


def test_bar_chord_pads_start_on_bar_lines():
    import numpy as np
    from musictool.instruments import bar_chord_pads

    sr, bpm = 22050, 120.0
    bar = 2.0
    t = np.arange(int(4 * bar * sr)) / sr
    y = np.zeros_like(t)
    for i, chord in enumerate(([60, 64, 67], [62, 65, 69], [60, 64, 67], [62, 65, 69])):
        seg = (t >= i * bar) & (t < (i + 1) * bar)
        for p in chord:
            y[seg] += 0.2 * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * t[seg])
    pads = bar_chord_pads(y.astype(np.float32), sr, bpm, downbeat=0.0)
    assert pads and all(abs(n.start / bar - round(n.start / bar)) < 1e-6 for n in pads)
    first = sorted(n.pitch for n in pads if n.start == 0.0)
    assert {60, 64, 67} <= set(first)
