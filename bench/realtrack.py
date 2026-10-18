"""Score a transcription of a *real* track against its own separated stems.

There is no ground-truth MIDI for real music, so this measures agreement with
the audio instead:

* drums  - recall / precision of drum hits against onsets detected in the
           Demucs drum stem (50 ms window)
* pitched - "coverage": of the time a stem is audibly playing, how much has a
           note sounding (recall), and how much note time falls where the stem
           is audible (precision); plus chroma similarity between the MIDI and
           the stem where both are active (are they the right pitch classes?)

    python bench/realtrack.py song.wav out.mid [more.mid ...] [--stems DIR]

Stems are computed once with Demucs and cached in --stems (default: next to
the first MIDI file).
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 44100
HOP = 1024


def load_stems(song: Path, stems_dir: Path) -> dict[str, np.ndarray]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    files = {k: stems_dir / f"{k}.wav" for k in ("drums", "bass", "melodic", "vocals")}
    if not all(f.exists() for f in files.values()):
        from musictool.separation import separate_neural

        for k, v in separate_neural(song, "demucs-ft").items():
            sf.write(files[k], v, SR)
    return {k: librosa.load(str(f), sr=SR, mono=True)[0] for k, f in files.items()}


def drum_scores(drum_stem: np.ndarray, hits: np.ndarray) -> tuple[float, float]:
    env = librosa.onset.onset_strength(y=drum_stem, sr=SR, hop_length=512)
    ref = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=512, units="time", backtrack=False)
    hits = np.unique(np.round(hits, 2))
    if not len(ref) or not len(hits):
        return 0.0, 0.0
    recall = np.mean([np.min(np.abs(hits - t)) <= 0.05 for t in ref])
    precision = np.mean([np.min(np.abs(ref - t)) <= 0.05 for t in hits])
    return float(recall), float(precision)


def audible(stem: np.ndarray, n_frames: int, floor_db: float = -30.0) -> np.ndarray:
    rms = librosa.feature.rms(y=stem, frame_length=2 * HOP, hop_length=HOP)[0][:n_frames]
    db = 20 * np.log10(rms + 1e-9)
    ref = np.percentile(db, 95)
    mask = db > ref + floor_db
    return np.pad(mask, (0, max(0, n_frames - len(mask))))


def note_mask(notes, n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, bool)
    for n in notes:
        m[int(n.start * SR / HOP): int(n.end * SR / HOP) + 1] = True
    return m[:n_frames]


def chroma_similarity(stem: np.ndarray, notes, n_frames: int) -> float:
    if not notes:
        return 0.0
    audio = librosa.feature.chroma_cqt(y=stem, sr=SR, hop_length=HOP)[:, :n_frames]
    roll = np.zeros((12, audio.shape[1]))
    for n in notes:
        roll[n.pitch % 12, int(n.start * SR / HOP): int(n.end * SR / HOP) + 1] += n.velocity / 127
    both = (roll.sum(0) > 0) & (audio.max(0) > 0.1)
    if not both.any():
        return 0.0
    a, r = audio[:, both], roll[:, both]
    cos = (a * r).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(r, axis=0) + 1e-9)
    return float(cos.mean())


def score(stems: dict, midi_path: Path) -> dict[str, float]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    n_frames = 1 + len(stems["drums"]) // HOP
    tracks = {i.name: i.notes for i in pm.instruments}
    drums = next((i.notes for i in pm.instruments if i.is_drum), [])
    out = {}
    out["drum_R"], out["drum_P"] = drum_scores(stems["drums"], np.array([n.start for n in drums]))
    pitched = [i for i in pm.instruments if not i.is_drum]
    for name, stem_keys, insts in (
        ("bass", ["bass"], [i for i in pitched if i.name == "Bass"]),
        ("synth", ["melodic", "vocals"], [i for i in pitched if i.name != "Bass"]),  # every instrument track
    ):
        stem = sum(stems[k] for k in stem_keys)
        notes = [n for i in insts for n in i.notes]
        aud, act = audible(stem, n_frames), note_mask(notes, n_frames)
        out[f"{name}_R"] = float((aud & act).sum() / max(1, aud.sum()))
        out[f"{name}_P"] = float((aud & act).sum() / max(1, act.sum()))
        out[f"{name}_chroma"] = chroma_similarity(stem, notes, n_frames)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("midi", nargs="+")
    ap.add_argument("--stems")
    args = ap.parse_args()
    stems_dir = Path(args.stems) if args.stems else Path(args.midi[0]).parent / "eval_stems"
    stems = load_stems(Path(args.song), stems_dir)
    cols = ["drum_R", "drum_P", "bass_R", "bass_P", "bass_chroma", "synth_R", "synth_P", "synth_chroma"]
    print(f"{'midi':34}" + "".join(f"{c:>13}" for c in cols))
    for m in args.midi:
        s = score(stems, Path(m))
        print(f"{Path(m).parent.name + '/' + Path(m).name:34}" + "".join(f"{s[c]:13.2f}" for c in cols))


if __name__ == "__main__":
    main()
