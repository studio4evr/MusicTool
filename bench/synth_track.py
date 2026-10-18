"""Render a realistic-ish electronic track with exact ground-truth notes.

Harder than tests/synth.py on purpose: acid bass with filter sweeps and glides,
detuned supersaw chords with sidechain pumping, a delayed arpeggio lead,
snare rolls, open hats, reverb and bus saturation.
"""

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
KICK, SNARE, HAT = "kick", "snare", "hat"


def _hz(p):
    return 440.0 * 2 ** ((p - 69) / 12)


def _saw(phase):
    return 2.0 * (phase % 1.0) - 1.0


def _reverb_ir(seconds, rng, decay=4.0):
    t = np.arange(int(seconds * SR)) / SR
    ir = rng.standard_normal(len(t)) * np.exp(-t * decay)
    ir[0] = 1.0
    return ir / np.abs(ir).sum() * 6


def render(bpm: float = 132.0, bars: int = 16, seed: int = 1):
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    six = beat / 4  # 16th
    n = int((bars * 4 + 2) * beat * SR)
    drums, bass, pads, lead = (np.zeros(n) for _ in range(4))
    truth = {KICK: [], SNARE: [], HAT: [], "bass": [], "synths": []}

    def place(buf, sig, t0, gain=1.0):
        i = int(round(t0 * SR))
        if i >= n:
            return
        sig = sig[: n - i]
        buf[i : i + len(sig)] += gain * sig

    hp = butter(4, 7000, "highpass", fs=SR, output="sos")
    bp = butter(2, [180, 4000], "bandpass", fs=SR, output="sos")

    def kick():
        t = np.arange(int(0.35 * SR)) / SR
        f = 48 + 160 * np.exp(-t * 35)
        click = np.exp(-t * 900) * rng.standard_normal(len(t)) * 0.3
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) + click

    def snare():
        t = np.arange(int(0.25 * SR)) / SR
        return (sosfilt(bp, rng.standard_normal(len(t))) * 0.9 + 0.5 * np.sin(2 * np.pi * 200 * t)) * np.exp(-t * 16)

    def hat(open_=False):
        t = np.arange(int((0.35 if open_ else 0.05) * SR)) / SR
        return sosfilt(hp, rng.standard_normal(len(t))) * np.exp(-t * (10 if open_ else 90))

    # --- drums: four-on-the-floor, 16th hats w/ accents, open hats, snare rolls
    for bar in range(bars):
        for s in range(16):
            t = (bar * 16 + s) * six
            if s % 4 == 0 or (bar % 4 == 3 and s == 14):
                place(drums, kick(), t, 0.9)
                truth[KICK].append(t)
            if s in (4, 12):
                place(drums, snare(), t, 0.55)
                truth[SNARE].append(t)
            if s % 4 == 2:
                place(drums, hat(open_=True), t, 0.22)
                truth[HAT].append(t)
            elif s % 2 == 1 or bar % 2:
                place(drums, hat(), t, 0.25 if s % 4 == 0 else 0.15)
                truth[HAT].append(t)
        if bar % 8 == 7:  # 32nd-note snare roll over the last beat
            for k in range(8):
                t = (bar * 16 + 12) * six + k * six / 2
                if k:  # k=0 is the regular backbeat already placed
                    place(drums, snare(), t, 0.25 + 0.04 * k)
                    truth[SNARE].append(t)

    # --- acid bass: 16th pattern, resonant sweep, slides, octave jumps
    pattern = [36, None, 36, 48, None, 39, 36, None, 43, None, 41, 36, None, 46, 36, 48]
    slides = {3, 11}
    phase = 0.0
    for bar in range(bars):
        root_shift = [0, 0, -2, 3][bar % 4]
        for s, p in enumerate(pattern):
            if p is None:
                continue
            p = p + root_shift
            t0 = (bar * 16 + s) * six
            length = six * (1.9 if s in slides else 0.8)
            tt = np.arange(int(length * SR)) / SR
            f = np.full(len(tt), _hz(p))
            if s in slides:  # glide from the previous pitch
                prev = _hz(p - 12)
                g = np.minimum(1, tt / 0.06)
                f = prev * (1 - g) + _hz(p) * g
            ph = phase + np.cumsum(f) / SR
            phase = ph[-1]
            cutoff = 300 + 2200 * (0.5 + 0.5 * np.sin(2 * np.pi * (bar * 16 + s) / 64)) * np.exp(-tt * 12)
            sig = _saw(ph)
            # one-pole-ish resonant LP approximation: average of two filters at the start cutoff
            sos = butter(2, float(np.clip(cutoff[0], 80, 8000)), "lowpass", fs=SR, output="sos")
            sig = np.tanh(2.2 * sosfilt(sos, sig))
            env = np.minimum(1, tt * 400) * np.minimum(1, (length - tt) * 300)
            place(bass, sig * env, t0, 0.42)
            truth["bass"].append((t0, t0 + length, p))

    # --- detuned supersaw pads (one chord per bar, sidechained to the kick)
    chords = [[60, 63, 67, 70], [58, 62, 65, 69], [56, 60, 63, 67], [63, 67, 70, 74]]
    for bar in range(bars):
        t0 = bar * 16 * six
        length = 16 * six * 0.95
        tt = np.arange(int(length * SR)) / SR
        sig = np.zeros(len(tt))
        for p in chords[bar % 4]:
            for det in (-0.12, 0.0, 0.11):
                sig += _saw(tt * _hz(p + det) + rng.random())
            truth["synths"].append((t0, t0 + length, p))
        sos = butter(2, 2500, "lowpass", fs=SR, output="sos")
        pump = 1 - 0.7 * np.exp(-((tt % beat) / 0.09))
        env = np.minimum(1, tt * 30) * np.minimum(1, (length - tt) * 30)
        place(pads, sosfilt(sos, sig) * env * pump, t0, 0.035)

    # --- square arpeggio lead with dotted-8th delay, 16ths, every other 2 bars
    arp = [72, 75, 79, 84, 79, 75]
    for bar in range(bars):
        if bar % 4 < 2:
            continue
        for s in range(16):
            p = arp[s % len(arp)] + [0, 0, -2, 3][bar % 4]
            t0 = (bar * 16 + s) * six
            length = six * 0.6
            tt = np.arange(int(length * SR)) / SR
            sq = np.sign(np.sin(2 * np.pi * _hz(p) * tt)) * np.exp(-tt * 8)
            place(lead, sq, t0, 0.07)
            truth["synths"].append((t0, t0 + length, p))
    delayed = np.zeros(n)
    d = int(0.75 * beat * SR)
    delayed[d:] = lead[:-d] * 0.35
    lead += delayed

    # --- bus: reverb on snare/pads/lead, saturation
    ir = _reverb_ir(1.8, rng)
    wet = fftconvolve(pads + lead + 0.3 * drums, ir)[:n]
    mix = drums + bass + pads + lead + 0.25 * wet
    mix = np.tanh(1.4 * mix) / np.tanh(1.4)
    mix /= np.max(np.abs(mix)) * 1.02
    for k in (KICK, SNARE, HAT):
        truth[k].sort()
    return mix.astype(np.float32), SR, truth
