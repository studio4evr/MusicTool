import sys
from pathlib import Path

import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import synth  # noqa: E402


@pytest.fixture(scope="session")
def track(tmp_path_factory):
    y, sr, truth = synth.render(bars=4)
    path = tmp_path_factory.mktemp("audio") / "synthetic.wav"
    sf.write(path, y, sr)
    return path, y, sr, truth


# The CPU engines, pinned so results don't depend on what's installed.
DSP = dict(separation="hpss", drum_engine="dsp", mono_engine="pyin")


@pytest.fixture(scope="session")
def result(track):
    from musictool import TranscriptionOptions, transcribe

    return transcribe(track[0], TranscriptionOptions(quantize=None, **DSP))
