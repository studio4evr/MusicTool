from dataclasses import dataclass


@dataclass
class Note:
    start: float  # seconds
    end: float  # seconds
    pitch: int  # MIDI note number
    velocity: int  # 1-127

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "pitch": self.pitch,
            "velocity": self.velocity,
        }
