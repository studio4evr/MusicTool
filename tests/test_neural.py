"""Neural engines (Demucs, ADTOF, CREPE). Skipped unless the GPU extras are installed."""

import pytest

from musictool import TranscriptionOptions, transcribe
from musictool.drums import adtof_available, transcribe_drums_adtof
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
        "bass": "basic-pitch",
        "synths": "basic-pitch",
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
