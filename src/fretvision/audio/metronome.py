"""Low-jitter metronome click generator and beat-phase clock."""

from __future__ import annotations

import math
import threading

import numpy as np
import sounddevice as sd

TIME_SIGNATURES: dict[str, tuple[int, int]] = {
    "4/4": (4, 4),
    "3/4": (3, 4),
    "6/8": (6, 8),
}


class MetronomeEngine:
    """Real-time audio click track using sounddevice.OutputStream."""

    def __init__(
        self,
        bpm: int = 100,
        time_signature: str = "4/4",
        sample_rate: int = 44100,
        volume: float = 0.35,
    ) -> None:
        self.bpm = max(40, min(240, int(bpm)))
        self.time_signature = time_signature if time_signature in TIME_SIGNATURES else "4/4"
        self.sample_rate = int(sample_rate)
        self.volume = float(volume)

        self._lock = threading.Lock()
        self._running = False
        self._stream: sd.OutputStream | None = None
        self._sample_pos: int = 0
        self._last_error: str | None = None

        self._accent_click = self._synth_click(1250.0, 0.026)
        self._beat_click = self._synth_click(880.0, 0.020)

    def _synth_click(self, freq_hz: float, duration_s: float) -> np.ndarray:
        n = max(64, int(round(duration_s * self.sample_rate)))
        t = np.arange(n, dtype=np.float32) / float(self.sample_rate)
        env = np.exp(-t * 140.0).astype(np.float32)
        return (np.sin(2.0 * math.pi * freq_hz * t) * env * self.volume).astype(np.float32)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def beats_per_bar(self) -> int:
        return TIME_SIGNATURES.get(self.time_signature, (4, 4))[0]

    def set_tempo(self, bpm: int, time_signature: str | None = None) -> None:
        with self._lock:
            self.bpm = max(40, min(240, int(bpm)))
            if time_signature and time_signature in TIME_SIGNATURES:
                self.time_signature = time_signature

    def start(self) -> bool:
        """Start audio output click stream (returns False cleanly if no output device)."""
        self.stop()
        self._last_error = None
        with self._lock:
            self._sample_pos = 0
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=self._callback,
            )
            stream.start()
            self._stream = stream
            self._running = True
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._running = False
            return False

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        _ = (time_info, status)
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            samples_per_beat = max(1, int(round((60.0 / self.bpm) * self.sample_rate)))
            bpb = self.beats_per_bar()
            start_s = self._sample_pos
            end_s = start_s + frames

            first_beat_idx = start_s // samples_per_beat
            last_beat_idx = (end_s - 1) // samples_per_beat
            for b_idx in range(first_beat_idx, last_beat_idx + 1):
                beat_sample = b_idx * samples_per_beat
                click = self._accent_click if (b_idx % bpb == 0) else self._beat_click
                # Write overlapping portion of click into output buffer
                c_start = max(0, start_s - beat_sample)
                o_start = max(0, beat_sample - start_s)
                copy_len = min(len(click) - c_start, frames - o_start)
                if copy_len > 0:
                    out[o_start : o_start + copy_len] += click[c_start : c_start + copy_len]

            self._sample_pos = end_s
        outdata[:, 0] = np.clip(out, -0.95, 0.95)
