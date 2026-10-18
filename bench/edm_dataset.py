"""Render a synthetic electronic-music dataset with exact per-instrument MIDI, in
YourMT3's Slakh-style training format (16 kHz stems + note / note-event files +
index JSON), for fine-tuning it on synth-heavy music.

    python bench/edm_dataset.py --out data/edm --tracks 400

Each track picks a key, scale, tempo (110-175 BPM), chord progression and a random
subset of instruments: drums, bass (acid / sub / FM / square), lead (arp or riff,
16ths or 32nds, saw / square / supersaw / pulse, with delay), pad (detuned saws,
sidechained), pluck stabs and FM bells. Echoes of delays are not labelled.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

SR = 16000
SCALES = {"minor": [0, 2, 3, 5, 7, 8, 10], "major": [0, 2, 4, 5, 7, 9, 11], "dorian": [0, 2, 3, 5, 7, 9, 10],
          "phrygian": [0, 1, 3, 5, 7, 8, 10], "harmonic": [0, 2, 3, 5, 7, 8, 11]}
# GM programs per role (YourMT3's mc13 classes: lead 80-87, pad 88-95, bass 32-39, chromatic perc 8-15)
PROGRAMS = {"bass": [38, 39, 33], "lead": [80, 81, 84, 87], "pad": [88, 89, 90, 91, 95], "pluck": [82, 85, 86], "bells": [8, 11]}


def hz(p):
    return 440.0 * 2 ** ((np.asarray(p) - 69) / 12)


class Song:
    def __init__(self, rng):
        self.rng = rng
        self.bpm = float(rng.uniform(110, 175))
        self.beat = 60.0 / self.bpm
        self.six = self.beat / 4
        self.bars = int(rng.choice([12, 16, 16, 20]))
        self.root = int(rng.integers(0, 12))
        self.scale = SCALES[rng.choice(list(SCALES))]
        self.n = int((self.bars * 4 + 4) * self.beat * SR)
        degrees = rng.choice([[0, 5, 3, 4], [0, 3, 4, 4], [0, 6, 5, 4], [0, 0, 5, 3], [0, 4, 5, 3], [0, 2, 5, 4]])
        self.chord_degrees = [int(d) for d in degrees]
        self.bars_per_chord = int(rng.choice([1, 2]))

    def degree(self, d, octave):
        sc = self.scale
        return 12 * octave + self.root + sc[d % 7] + 12 * (d // 7)

    def chord(self, bar, octave=4, size=3):
        d = self.chord_degrees[(bar // self.bars_per_chord) % len(self.chord_degrees)]
        return [self.degree(d + 2 * k, octave) for k in range(size)]


def env_adsr(length, a, d, s, r, sr=SR):
    n = int(length * sr)
    t = np.arange(n) / sr
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = int(r * sr)
    tail = e[-1] * np.exp(-np.arange(rel) / max(r * sr / 5, 1))
    return np.concatenate([e, tail])


def osc(kind, f, n, rng, detune=0.0):
    t = np.arange(n) / SR
    ph = (f * t + rng.random()) % 1.0
    if kind == "saw":
        return 2 * ph - 1
    if kind == "square":
        return np.sign(ph - 0.5)
    if kind == "pulse":
        return np.where(ph < 0.25, 1.0, -1.0)
    if kind == "sine":
        return np.sin(2 * np.pi * ph)
    if kind == "tri":
        return 2 * np.abs(2 * ph - 1) - 1
    if kind == "supersaw":
        return sum(2 * ((f * (1 + d) * t + rng.random()) % 1) - 1 for d in (-0.012, -0.005, 0, 0.006, 0.011)) / 3
    if kind == "fm":
        return np.sin(2 * np.pi * f * t + 2.5 * np.exp(-t * 6) * np.sin(2 * np.pi * f * 2 * t))
    raise ValueError(kind)


def lowpass(x, cutoff):
    sos = butter(2, float(np.clip(cutoff, 60, SR / 2 - 200)), "lowpass", fs=SR, output="sos")
    return sosfilt(sos, x)


def place(buf, sig, t0, gain=1.0):
    i = int(round(t0 * SR))
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    buf[i: i + len(sig)] += gain * sig


def render_bass(song, rng):
    buf = np.zeros(song.n); notes = []
    kind = rng.choice(["acid", "sub", "fm", "square"])
    rhythm = rng.choice(["offbeat", "rolling", "eighths", "syncopated"])
    steps = {"offbeat": [2, 6, 10, 14], "rolling": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],
             "eighths": [0, 2, 4, 6, 8, 10, 12, 14], "syncopated": [0, 3, 6, 10, 12, 14]}[rhythm]
    for bar in range(song.bars):
        root = song.chord(bar, octave=2, size=1)[0]
        for s in steps:
            p = root + (12 if rng.random() < 0.12 else 0) + (int(rng.choice([0, 7, 3])) if rng.random() < 0.15 else 0)
            t0 = (bar * 16 + s) * song.six
            length = song.six * float(rng.uniform(0.5, 0.95))
            n = int(length * SR)
            if kind == "acid":
                sig = np.tanh(2.5 * lowpass(osc("saw", hz(p), n, rng), rng.uniform(300, 2500)))
            elif kind == "sub":
                sig = osc("sine", hz(p), n, rng) + 0.2 * osc("tri", hz(p) * 2, n, rng)
            elif kind == "fm":
                sig = osc("fm", hz(p), n, rng)
            else:
                sig = lowpass(osc("square", hz(p), n, rng), 900)
            sig = sig * env_adsr(length, 0.003, 0.15, 0.6, 0.0)[:n]
            place(buf, sig, t0, 0.5)
            notes.append((t0, t0 + length, p))
    return buf, notes


def render_lead(song, rng):
    buf = np.zeros(song.n); notes = []
    kind = rng.choice(["saw", "square", "supersaw", "pulse", "tri"])
    style = rng.choice(["arp16", "arp32", "riff"])
    cutoff = rng.uniform(1500, 6000)
    active = sorted(rng.choice(song.bars, size=max(4, song.bars * 3 // 4), replace=False))
    for bar in active:
        ch = song.chord(bar, octave=int(rng.choice([4, 5])), size=3)
        if style.startswith("arp"):
            div = 16 if style == "arp16" else 32
            order = rng.choice(["up", "down", "updown", "random"])
            pat = ch + [ch[0] + 12]
            if order == "down":
                pat = pat[::-1]
            elif order == "updown":
                pat = pat + pat[-2:0:-1]
            step = song.beat * 4 / div
            for k in range(div):
                if rng.random() < 0.1:
                    continue
                p = int(rng.choice(pat)) if order == "random" else pat[k % len(pat)]
                t0 = bar * 4 * song.beat + k * step
                length = step * float(rng.uniform(0.5, 0.9))
                notes.append((t0, t0 + length, p))
        else:
            k = 0
            while k < 16:
                dur = int(rng.choice([1, 2, 2, 3, 4]))
                if rng.random() < 0.75:
                    p = song.degree(int(rng.integers(0, 9)), 5)
                    t0 = (bar * 16 + k) * song.six
                    notes.append((t0, t0 + dur * song.six * 0.9, p))
                k += dur
    for t0, t1, p in notes:
        n = int((t1 - t0) * SR)
        sig = lowpass(osc(kind, hz(p), n, rng), cutoff) * env_adsr(t1 - t0, 0.003, 0.1, 0.7, 0.0)[:n]
        place(buf, sig, t0, 0.18)
    if rng.random() < 0.7:  # tempo-synced delay, echoes unlabelled
        d = int(song.beat * float(rng.choice([0.75, 0.5, 0.375])) * SR)
        wet = np.zeros_like(buf)
        for k, g in ((1, 0.4), (2, 0.18)):
            wet[k * d:] += g * buf[: len(buf) - k * d]
        buf += wet
    return buf, notes


def render_pad(song, rng):
    buf = np.zeros(song.n); notes = []
    kind = rng.choice(["supersaw", "saw", "tri"])
    pump = rng.random() < 0.6
    for bar in range(0, song.bars, song.bars_per_chord):
        if rng.random() < 0.15:
            continue
        ch = song.chord(bar, octave=int(rng.choice([3, 4])), size=int(rng.choice([3, 4])))
        t0 = bar * 4 * song.beat
        length = song.bars_per_chord * 4 * song.beat * 0.98
        n = int(length * SR)
        e = env_adsr(length, float(rng.uniform(0.02, 0.4)), 0.5, 0.8, 0.0)[:n]
        for p in ch:
            sig = lowpass(osc(kind, hz(p), n, rng), rng.uniform(1200, 3500)) * e
            if pump:
                t = np.arange(n) / SR
                sig *= 1 - 0.7 * np.exp(-((t % song.beat) / 0.09))
            place(buf, sig, t0, 0.07)
            notes.append((t0, t0 + length, p))
    return buf, notes


def render_pluck(song, rng):
    buf = np.zeros(song.n); notes = []
    steps = sorted(rng.choice(16, size=int(rng.integers(3, 7)), replace=False))
    for bar in range(song.bars):
        if rng.random() < 0.3:
            continue
        ch = song.chord(bar, octave=4, size=3)
        for s in steps:
            t0 = (bar * 16 + int(s)) * song.six
            length = song.six * float(rng.uniform(1.0, 2.0))
            n = int(length * SR)
            for p in ch:
                sig = lowpass(osc("saw", hz(p), n, rng), 3000) * np.exp(-np.arange(n) / SR * 18)
                place(buf, sig, t0, 0.08)
                notes.append((t0, t0 + length, p))
    return buf, notes


def render_bells(song, rng):
    buf = np.zeros(song.n); notes = []
    for bar in range(song.bars):
        for s in (0, 6, 10):
            if rng.random() < 0.5:
                continue
            p = song.degree(int(rng.integers(0, 8)), 5)
            t0 = (bar * 16 + s) * song.six
            length = song.beat
            n = int(length * 1.5 * SR)
            sig = osc("fm", hz(p), n, rng) * np.exp(-np.arange(n) / SR * 5)
            place(buf, sig, t0, 0.12)
            notes.append((t0, t0 + length, p))
    return buf, notes


def render_drums(song, rng):
    buf = np.zeros(song.n); notes = []
    hp = butter(4, 6500, "highpass", fs=SR, output="sos")
    bp = butter(2, [200, 4500], "bandpass", fs=SR, output="sos")

    def noise(sec):
        return rng.standard_normal(int(sec * SR))

    def kick():
        t = np.arange(int(0.35 * SR)) / SR
        f = 45 + float(rng.uniform(90, 200)) * np.exp(-t * 30)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)

    def snare():
        t = np.arange(int(0.2 * SR)) / SR
        return (sosfilt(bp, noise(0.2)) + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 18)

    def clap():
        t = np.arange(int(0.18 * SR)) / SR
        return sosfilt(bp, noise(0.18)) * (np.exp(-t * 25) + 0.6 * np.exp(-((t - 0.012) % 0.011) * 300) * (t < 0.035))

    def hat(open_):
        t = np.arange(int((0.3 if open_ else 0.05) * SR)) / SR
        return sosfilt(hp, noise(len(t) / SR)) * np.exp(-t * (12 if open_ else 90))

    four = rng.random() < 0.7
    use_clap = rng.random() < 0.5
    hat_div = int(rng.choice([2, 4]))
    for bar in range(song.bars):
        for s in range(16):
            t0 = (bar * 16 + s) * song.six
            if (four and s % 4 == 0) or (not four and s in (0, 6, 10)) or (s == 14 and rng.random() < 0.15):
                place(buf, kick(), t0, 0.9); notes.append((t0, t0 + 0.1, 36))
            if s in (4, 12):
                if use_clap:
                    place(buf, clap(), t0, 0.5); notes.append((t0, t0 + 0.1, 39))
                else:
                    place(buf, snare(), t0, 0.5); notes.append((t0, t0 + 0.1, 38))
            if s % (16 // (4 * hat_div // 2)) == 2 or (hat_div == 4 and s % 2 == 1):
                op = s % 4 == 2 and rng.random() < 0.4
                place(buf, hat(op), t0, 0.2 if op else 0.15); notes.append((t0, t0 + 0.1, 46 if op else 42))
    return buf, notes


RENDER = {"drums": render_drums, "bass": render_bass, "lead": render_lead, "pad": render_pad,
          "pluck": render_pluck, "bells": render_bells}


def render_track(seed: int):
    rng = np.random.default_rng(seed)
    song = Song(rng)
    roles = ["drums", "bass", "lead", "pad"]
    roles = [r for r in roles if rng.random() < 0.9] + [r for r in ("pluck", "bells") if rng.random() < 0.35]
    if not any(r in roles for r in ("lead", "pad", "pluck", "bells")):
        roles.append("lead")
    stems = []
    for role in roles:
        audio, notes = RENDER[role](song, rng)
        program = 128 if role == "drums" else int(rng.choice(PROGRAMS[role]))  # 128 = YourMT3's drum program
        stems.append(dict(role=role, program=program, is_drum=role == "drums", audio=audio, notes=notes))
    # shared reverb on melodic stems, bus saturation, normalise
    t = np.arange(int(1.5 * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-t * 4); ir[0] = 1; ir /= np.abs(ir).sum() / 6
    for st in stems:
        if not st["is_drum"] and st["role"] != "bass":
            st["audio"] = st["audio"] + 0.25 * fftconvolve(st["audio"], ir)[: song.n]
    mix = sum(st["audio"] for st in stems)
    peak = np.max(np.abs(np.tanh(1.3 * mix))) or 1.0
    gain = 0.95 / (np.max(np.abs(mix)) or 1.0)
    for st in stems:
        st["audio"] = st["audio"] * gain
    mix = mix * gain
    return song, stems, mix


def write_dataset(out: Path, n_tracks: int, split_ratio=(0.9, 0.05, 0.05), seed0: int = 1000):
    src = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src))
    from musictool.mt3 import ymt3_dir
    root = ymt3_dir()
    sys.path.insert(0, str(root / "amt" / "src"))
    from utils.note_event_dataclasses import Note as YNote
    from utils.note2event import mix_notes, note2note_event

    out = out.resolve()  # the trainer runs from the YourMT3 folder, so index paths must be absolute
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": {}, "validation": {}, "test": {}}
    for i in range(n_tracks):
        song, stems, mix = render_track(seed0 + i)
        tid = f"edm{i:05d}"
        d = out / tid
        d.mkdir(exist_ok=True)
        n_frames = len(mix)
        audio_array = np.stack([st["audio"].astype(np.float16) for st in stems])
        stem_content = {"mtrack_id": tid, "program": np.array([st["program"] for st in stems], dtype=np.int64),
                        "is_drum": np.array([int(st["is_drum"]) for st in stems], dtype=np.int64),
                        "n_frames": n_frames, "audio_array": audio_array}
        np.save(d / f"{tid}_stem.npy", stem_content)
        sf.write(d / f"{tid}_mix.wav", mix.astype(np.float32), SR, subtype="PCM_16")
        mixed = []
        for st in stems:
            ys = sorted((YNote(is_drum=st["is_drum"], program=st["program"],
                               onset=float(a), offset=float(max(b, a + 0.01)), pitch=int(p), velocity=1)
                         for a, b, p in st["notes"]), key=lambda x: (x.onset, x.pitch))
            mixed = mix_notes((mixed, ys), True, True, True)
        duration = n_frames / SR
        notes = {"mtrack_id": tid, "program": stem_content["program"], "is_drum": stem_content["is_drum"],
                 "duration_sec": duration, "notes": mixed}
        events = {**{k: v for k, v in notes.items() if k != "notes"},
                  "note_events": note2note_event(mixed, sort=True, return_activity=True)}
        np.save(d / f"{tid}_notes.npy", notes, allow_pickle=True)
        np.save(d / f"{tid}_note_events.npy", events, allow_pickle=True)
        r = (i % 20) / 20
        split = "train" if r < split_ratio[0] else ("validation" if r < split_ratio[0] + split_ratio[1] else "test")
        splits[split][len(splits[split])] = {
            "mtrack_id": tid, "n_frames": n_frames, "stem_file": str(d / f"{tid}_stem.npy"),
            "mix_audio_file": str(d / f"{tid}_mix.wav"), "notes_file": str(d / f"{tid}_notes.npy"),
            "note_events_file": str(d / f"{tid}_note_events.npy"), "midi_file": "",
            "program": stem_content["program"].tolist(), "is_drum": stem_content["is_drum"].tolist()}
        if (i + 1) % 25 == 0:
            print(f"rendered {i + 1}/{n_tracks}", flush=True)
    idx = out / "yourmt3_indexes"
    idx.mkdir(exist_ok=True)
    for split, fl in splits.items():
        (idx / f"edm_{split}_file_list.json").write_text(json.dumps(fl, indent=1))
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/edm")
    ap.add_argument("--tracks", type=int, default=400)
    a = ap.parse_args()
    write_dataset(Path(a.out), a.tracks)
