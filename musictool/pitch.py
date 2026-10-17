"""Polyphonic pitch transcription with Spotify's Basic Pitch neural network.

Basic Pitch is a lightweight CNN trained for automatic music transcription.
We run its ONNX export through onnxruntime, so TensorFlow is not required.
"""

import contextlib
import io
import logging
import os
import tempfile
import warnings

import numpy as np
import soundfile as sf

from .notes import Note

_model = None


def _load_model():
    global _model
    if _model is None:
        # basic_pitch logs a warning for every optional backend that is missing.
        logging.getLogger().setLevel(logging.ERROR)
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model

        _model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    return _model


def transcribe_pitched(
    y: np.ndarray,
    sr: int,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 80.0,
    min_freq: float | None = None,
    max_freq: float | None = None,
    monophonic: bool = False,
) -> list[Note]:
    """Transcribe pitched notes from a (preferably separated) mono signal."""
    if np.max(np.abs(y)) < 1e-4:
        return []

    from basic_pitch.inference import predict

    model = _load_model()
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_path, y, sr)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            _, _, events = predict(
                tmp_path,
                model,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=min_note_ms,
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
            )
    finally:
        os.remove(tmp_path)

    notes = [
        Note(
            start=float(start),
            end=float(end),
            pitch=int(pitch),
            velocity=int(np.clip(round(float(amp) * 127), 1, 127)),
        )
        for start, end, pitch, amp, _ in events
    ]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return make_monophonic(notes) if monophonic else notes


def make_monophonic(notes: list[Note]) -> list[Note]:
    """Keep one note at a time (bass lines): the loudest of simultaneous onsets,
    and cut each note off where the next one starts."""
    by_onset: list[Note] = []
    for note in sorted(notes, key=lambda n: n.start):
        if by_onset and note.start - by_onset[-1].start < 0.03:
            if note.velocity > by_onset[-1].velocity:
                by_onset[-1] = note
            continue
        by_onset.append(note)

    result = []
    for i, note in enumerate(by_onset):
        end = note.end
        if i + 1 < len(by_onset):
            end = min(end, by_onset[i + 1].start)
        if end - note.start > 0.02:
            result.append(Note(note.start, end, note.pitch, note.velocity))
    return result


def transcribe_monophonic(
    y: np.ndarray,
    sr: int,
    *,
    min_freq: float,
    max_freq: float,
    min_note_ms: float = 80.0,
    hop_length: int = 256,
) -> list[Note]:
    """Transcribe a single-voice line (bass, vocal) with probabilistic YIN.

    pYIN is an HMM-smoothed f0 tracker; on monophonic synth bass it is more
    accurate than Basic Pitch, which tends to make octave errors below ~100 Hz.
    """
    import librosa
    from scipy.signal import find_peaks

    if np.max(np.abs(y)) < 1e-4:
        return []

    f0, voiced, _ = librosa.pyin(
        y, fmin=min_freq, fmax=max_freq, sr=sr, frame_length=2048, hop_length=hop_length
    )
    midi = np.where(voiced, np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), -1).astype(int)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0][: len(midi)]
    peak_rms = float(rms.max()) or 1.0

    # Onsets let us split repeated notes of the same pitch.
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames, _ = find_peaks(env, prominence=0.2 * float(env.max() or 1.0), distance=3)
    onset_set = set(onset_frames.tolist())

    segments: list[list[int]] = []  # [start_frame, end_frame, pitch]
    for i, m in enumerate(midi):
        cur = segments[-1] if segments else None
        continuing = cur is not None and cur[1] >= i - 2 and cur[2] == m and m > 0
        if continuing and not (i in onset_set and i - cur[0] > 2):
            cur[1] = i
        elif m > 0:
            segments.append([i, i, m])

    frame_s = hop_length / sr
    notes = []
    for start, end, pitch in segments:
        dur = (end - start + 1) * frame_s
        if dur * 1000 < min_note_ms:
            continue
        level = float(rms[start : end + 1].max()) / peak_rms
        velocity = int(np.clip(round(30 + 97 * level), 1, 127))
        notes.append(Note(start * frame_s, (end + 1) * frame_s, int(pitch), velocity))
    return notes


def merge_fragments(notes: list[Note], max_gap: float = 0.03) -> list[Note]:
    """Join same-pitch notes separated by a tiny gap.

    Basic Pitch re-triggers sustained pads whenever another instrument's
    transient (a kick, a sidechain pump) hits; this stitches them back together.
    """
    merged: list[Note] = []
    last_by_pitch: dict[int, Note] = {}
    for n in sorted(notes, key=lambda n: n.start):
        prev = last_by_pitch.get(n.pitch)
        if prev is not None and n.start - prev.end <= max_gap:
            prev.end = max(prev.end, n.end)
            prev.velocity = max(prev.velocity, n.velocity)
            continue
        copy = Note(n.start, n.end, n.pitch, n.velocity)
        merged.append(copy)
        last_by_pitch[n.pitch] = copy
    return merged


def remove_octave_bleed(notes: list[Note], bass: list[Note], tolerance: float = 0.06) -> list[Note]:
    """Drop notes that are just an octave harmonic of a concurrent bass note."""
    if not bass:
        return notes

    def is_bleed(n: Note) -> bool:
        for b in bass:
            if b.start - tolerance > n.start:
                break
            if (
                n.pitch - b.pitch in (12, 24)
                and abs(n.start - b.start) <= tolerance
                and n.end <= b.end + tolerance
            ):
                return True
        return False

    bass = sorted(bass, key=lambda b: b.start)
    return [n for n in notes if not is_bleed(n)]
