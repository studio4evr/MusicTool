# MusicTool

AI transcription of **electronic music audio → multi-track MIDI**.
Drop in a track (WAV, MP3, FLAC, OGG, AIFF) and get a `.mid` file with separate
**Drums**, **Bass** and **Synths** tracks, tempo-synced and quantised to the grid,
ready to drag into Ableton, FL Studio, Logic, Bitwig, and so on.

![web app](docs/screenshot.png)

## How it works

| Stage | Method |
|---|---|
| Stem separation | Harmonic/percussive separation + band split (fast, default), or **Demucs** neural stem separation (optional, best quality) |
| Synths / chords | **Spotify Basic Pitch**, a neural polyphonic transcription model, run via ONNX Runtime (no TensorFlow needed) |
| Bass (and vocals with Demucs) | **pYIN** probabilistic f0 tracking (default; more accurate on low monophonic bass), or Basic Pitch |
| Drums | Multi-band transient detection → kick / snare / closed hat / open hat (GM drum map 36/38/42/46) |
| Tempo | Beat tracking + least-squares fit of the beat grid |
| Quantisation | Grid phase is estimated from the drum hits, then notes snap to 1/8, 1/16 or 1/32 |
| Clean-up | Fragmented sustained notes are merged, and octave bleed from the bass is removed from the synth track |

## Windows quick start

1. Install **Python 3.12** from <https://www.python.org/downloads/windows/>.
   In the installer, tick **"Add python.exe to PATH"**.
2. Download this repository (green **Code** button → **Download ZIP**) and unzip it.
3. Double-click **`install.bat`**. It sets everything up in a `.venv` folder,
   which takes a few minutes the first time.
4. Double-click **`start.bat`**. MusicTool opens in your browser. Close the
   black window, or press Ctrl+C in it, to stop the app.

You can also **drag audio files onto `transcribe.bat`**. It writes a `.mid` file,
plus one per track, next to each audio file without opening the browser.

If Windows SmartScreen warns about the `.bat` files, click *More info → Run anyway*.
They only create the `.venv` folder and install Python packages into it.

## Install (macOS / Linux / manual)

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --no-deps basic-pitch   # only its ONNX model is needed; avoids TensorFlow pins
pip install -e .
pip install resampy mir_eval scikit-learn   # used by basic-pitch's inference code

# optional: neural stem separation (installs PyTorch, large download)
pip install demucs
```

## Use it

**Web app**

```bash
musictool serve --open     # http://127.0.0.1:8000
```

Upload a file, tweak the options, and watch progress. You can then preview the
result on a piano roll, play the MIDI back with a built-in synth (alongside the
original audio if you like), and download the full MIDI or individual tracks.

**Command line**

```bash
musictool track.mp3                       # writes track.mid
musictool track.wav -o out.mid --separate-tracks   # also out_drums.mid, out_bass.mid, ...
musictool track.wav --separation demucs --quantize 32 --bpm 174
musictool --help
```

**Python**

```python
from musictool import transcribe, TranscriptionOptions

result = transcribe("track.wav", TranscriptionOptions(quantize=16))
print(result.bpm, [(t.name, len(t.notes)) for t in result.tracks])
result.write("track.mid")
```

## Tips

* Use `--separation demucs` on dense, full mixes. It gives far cleaner bass and synth parts.
* There's too much noise in the synth track? Lower the *Note sensitivity* or raise `--onset-threshold`.
* Hats or percussion are missing? Raise *Drum sensitivity*.
* The tempo detection fold assumes 85–180 BPM. For drum & bass or halftime, set `--bpm` explicitly.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The tests render a synthetic electronic track with known kick, snare, hat,
bass and chord notes. They check tempo, per-instrument accuracy, the MIDI
output, the CLI and the web API.
