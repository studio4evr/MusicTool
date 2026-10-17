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
    while bpm < 85:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return float(round(bpm, 2)), first_beat


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
