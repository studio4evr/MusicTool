"""Render a short synthetic electronic track with known ground-truth notes."""

import numpy as np
from scipy.signal import butter, sosfilt

SR = 22050
BPM = 128.0
BEAT = 60.0 / BPM

BASS_LINE = [36, 36, 39, 41, 36, 36, 43, 41]  # one note per beat (8th-note length)
CHORDS = [[60, 63, 67], [58, 62, 65]]  # one chord per bar


def _saw(freq, t):
    return 2.0 * ((t * freq) % 1.0) - 1.0


def _hz(pitch):
    return 440.0 * 2 ** ((pitch - 69) / 12)


def render(bars: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_beats = bars * 4
    y = np.zeros(int((n_beats + 1) * BEAT * SR), dtype=np.float64)
    truth = {"kick": [], "snare": [], "hat": [], "bass": [], "chords": []}

    def place(sig, t0, gain):
        i = int(t0 * SR)
        y[i : i + len(sig)] += gain * sig[: len(y) - i]

    hp = butter(4, 7000, "highpass", fs=SR, output="sos")
    bp = butter(2, [200, 3000], "bandpass", fs=SR, output="sos")
    lp = butter(2, 900, "lowpass", fs=SR, output="sos")

    for b in range(n_beats):
        t0 = b * BEAT
        # kick: pitch-swept sine
        t = np.arange(int(0.3 * SR)) / SR
        f = 45 + 110 * np.exp(-t * 30)
        place(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9), t0, 0.9)
        truth["kick"].append(t0)
        # offbeat closed hat
        t = np.arange(int(0.05 * SR)) / SR
        place(sosfilt(hp, rng.standard_normal(len(t))) * np.exp(-t * 80), t0 + BEAT / 2, 0.35)
        truth["hat"].append(t0 + BEAT / 2)
        # snare on 2 and 4
        if b % 4 in (1, 3):
            t = np.arange(int(0.2 * SR)) / SR
            body = sosfilt(bp, rng.standard_normal(len(t))) + 0.5 * np.sin(2 * np.pi * 190 * t)
            place(body * np.exp(-t * 18), t0, 0.6)
            truth["snare"].append(t0)
        # bass: 8th note on the offbeat, sidechain style
        pitch = BASS_LINE[b % len(BASS_LINE)]
        t = np.arange(int(BEAT / 2 * SR)) / SR
        env = np.minimum(1, t * 200) * np.minimum(1, (BEAT / 2 - t) * 100)
        place(sosfilt(lp, _saw(_hz(pitch), t)) * env, t0 + BEAT / 2, 0.45)
        truth["bass"].append((t0 + BEAT / 2, pitch))

    for bar in range(bars):
        t0 = bar * 4 * BEAT
        t = np.arange(int(3.5 * BEAT * SR)) / SR
        env = np.minimum(1, t * 40) * np.minimum(1, (3.5 * BEAT - t) * 40)
        for p in CHORDS[bar % len(CHORDS)]:
            tone = np.sin(2 * np.pi * _hz(p) * t) + 0.3 * np.sin(4 * np.pi * _hz(p) * t)
            place(tone * env, t0, 0.12)
            truth["chords"].append((t0, p))

    y /= np.max(np.abs(y)) * 1.05
    return y.astype(np.float32), SR, truth
