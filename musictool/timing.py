"""Tempo estimation and grid quantisation."""

import numpy as np
import librosa

from .notes import Note


def estimate_tempo(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (bpm, time of first beat in seconds).

    Electronic music sits mostly in 110-180 BPM, so the estimate is folded
    into the 85-180 range to avoid half/double-time errors.
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=256, start_bpm=124, units="time")
    beats = np.asarray(beats, dtype=float)
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 120.0
    first_beat = float(beats[0]) if len(beats) else 0.0
    if len(beats) >= 4:
        # Least-squares line through the beat grid: far more precise than the
        # frame-quantised tempo estimate, and electronic music has a fixed tempo.
        slope, intercept = np.polyfit(np.arange(len(beats)), beats, 1)
        if slope > 0:
            bpm = 60.0 / slope
            first_beat = float(intercept) % slope
    if bpm <= 0:
        bpm = 120.0
    refined = _fit_beat_grid(y, sr, _fold(bpm))
    if refined is not None:
        bpm, first_beat = refined
        # Polish: track beats locked to the chosen tempo and regress through them.
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, bpm=bpm, tightness=800, units="time")
        if len(beats) >= 8:
            slope, intercept = np.polyfit(np.arange(len(beats)), np.asarray(beats, dtype=float), 1)
            if slope > 0 and abs(60.0 / slope / bpm - 1) < 0.01:
                bpm, first_beat = 60.0 / slope, float(intercept) % slope
    return float(round(bpm, 2)), first_beat


def _fold(bpm: float) -> float:
    while bpm < 85:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


_RATIOS = (1.0, 2.0, 0.5, 1.5, 2 / 3, 4 / 3, 3 / 4)
HOP = 256


def _fit_beat_grid(y: np.ndarray, sr: int, bpm: float) -> tuple[float, float] | None:
    """Pick the metrical level and refine tempo + phase by fitting a beat grid.

    Beat trackers sometimes lock onto a dotted or triplet pulse (a 132 BPM track
    read as 88) and their tempo is only accurate to ~1%, which drifts by beats
    over a whole track. For each related tempo (x1.5, x2/3, ...) we search a
    +-4% window, coarse then fine, for the grid whose beats land hardest on
    low-frequency onsets: in dance music the kick marks the beat.
    """
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, n_mels=32, fmax=250)
    if env.size < 64 or env.mean() <= 0:
        return None
    env = env / env.mean()
    env = np.maximum(env, np.maximum(np.roll(env, 1), np.roll(env, -1)))  # tolerate +-1 frame

    def score(cand: float) -> tuple[float, float]:
        period = 60.0 / cand * sr / HOP
        phases = np.linspace(0, period, 32, endpoint=False)
        idx = np.round(phases[:, None] + np.arange(0, env.size / period - 1)[None, :] * period).astype(int)
        means = env[np.clip(idx, 0, env.size - 1)].mean(axis=1)
        best = int(means.argmax())
        prior = np.exp(-0.5 * (np.log2(cand / 128.0) / 0.8) ** 2)  # mild preference for ~128
        return float(means[best]) * (0.9 + 0.1 * prior), phases[best] * HOP / sr

    best = None
    for cand in (bpm * r for r in _RATIOS):
        if not 85 <= cand <= 180:
            continue
        top = cand
        for width, steps in ((0.04, 81), (0.002, 41), (0.0002, 41)):
            grid = top * (1 + np.linspace(-width, width, steps))
            top = max(grid, key=lambda c: score(c)[0])
        s, phase = score(top)
        if best is None or s > best[0]:
            best = (s, top, phase)
    return (best[1], best[2]) if best else None


def grid_step(bpm: float, division: int) -> float:
    return 60.0 / bpm * 4.0 / division


def grid_phase(onsets: list[float], step: float) -> float:
    """Offset of the grid that best fits the onsets (circular mean of onset mod step).

    Beat trackers often lock onto the off-beat; the notes themselves are a more
    reliable reference for where the grid lines fall.
    """
    if not onsets:
        return 0.0
    angles = 2 * np.pi * (np.asarray(onsets) % step) / step
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean / (2 * np.pi) * step) % step)


def quantize(notes: list[Note], bpm: float, division: int, phase: float = 0.0) -> list[Note]:
    """Snap note starts/ends to a 1/``division`` note grid (e.g. 16 = 16ths)."""
    grid = grid_step(bpm, division)
    phase = phase % grid

    def snap(t: float) -> float:
        return max(0.0, phase + round((t - phase) / grid) * grid)

    result: list[Note] = []
    seen: set[tuple[float, int]] = set()
    for n in notes:
        start = snap(n.start)
        end = max(snap(n.end), start + grid)
        key = (round(start, 6), n.pitch)
        if key in seen:  # two hits collapsed onto the same grid slot
            continue
        seen.add(key)
        result.append(Note(start, end, n.pitch, n.velocity))
    return result
