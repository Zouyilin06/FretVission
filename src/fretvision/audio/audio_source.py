"""Audio input capture via sounddevice and circular ring buffer."""

from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioDevice:
    """Available audio input device."""

    index: int
    name: str
    host_api: str
    default_samplerate: int
    max_input_channels: int

    @property
    def label(self) -> str:
        return f"{self.name} [{self.host_api}]"


def enumerate_audio_devices() -> list[AudioDevice]:
    """Return available audio input devices reported by PortAudio/sounddevice."""
    devices: list[AudioDevice] = []
    try:
        hostapis = sd.query_hostapis()
        all_devs = sd.query_devices()
    except Exception:
        return devices

    for idx, dev in enumerate(all_devs):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue
        host_idx = int(dev.get("hostapi", 0))
        host_name = str(hostapis[host_idx].get("name", "")) if 0 <= host_idx < len(hostapis) else ""
        devices.append(
            AudioDevice(
                index=idx,
                name=str(dev.get("name", f"Input {idx}")),
                host_api=host_name,
                default_samplerate=int(dev.get("default_samplerate", 44100)),
                max_input_channels=max_in,
            )
        )
    return devices


class AudioRingBuffer:
    """Thread-safe circular float32 buffer for real-time audio windows."""

    def __init__(self, capacity: int = 16384) -> None:
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        """Append mono float32 samples into the ring buffer."""
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = len(arr)
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                self._buf[:] = arr[-self.capacity :]
                self._write_pos = 0
            else:
                end = self._write_pos + n
                if end <= self.capacity:
                    self._buf[self._write_pos : end] = arr
                else:
                    first = self.capacity - self._write_pos
                    self._buf[self._write_pos :] = arr[:first]
                    self._buf[: n - first] = arr[first:]
                self._write_pos = end % self.capacity
            self._total_written += n

    def read_latest(self, n_samples: int) -> np.ndarray:
        """Return the most recent n_samples (zero-padded on the left if underfilled)."""
        n = min(int(n_samples), self.capacity)
        with self._lock:
            avail = min(n, self._total_written)
            out = np.zeros(n, dtype=np.float32)
            if avail == 0:
                return out
            start = (self._write_pos - avail) % self.capacity
            if start < self._write_pos:
                out[n - avail :] = self._buf[start : self._write_pos]
            else:
                first = self.capacity - start
                out[n - avail : n - avail + first] = self._buf[start:]
                out[n - avail + first :] = self._buf[: self._write_pos]
            return out

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written


class AudioSource:
    """Live microphone/audio-interface stream backed by sounddevice."""

    def __init__(
        self,
        device_index: int | None = None,
        preferred_sample_rate: int = 44100,
        hop_size: int = 512,
    ) -> None:
        self.device_index = device_index
        self.preferred_sample_rate = int(preferred_sample_rate)
        self.hop_size = int(hop_size)
        self.sample_rate: int = self.preferred_sample_rate
        self.ring = AudioRingBuffer(capacity=16384)

        self._stream: sd.InputStream | None = None
        self._last_error: str | None = None

    def open(self) -> bool:
        """Open the live audio input stream."""
        self.close()
        self._last_error = None
        try:
            dev_info = sd.query_devices(self.device_index, kind="input")
            native_sr = int(dev_info.get("default_samplerate", self.preferred_sample_rate))
        except Exception:
            native_sr = self.preferred_sample_rate

        # Prefer native samplerate on Windows WASAPI/DirectSound to avoid opening failure
        candidates = [native_sr]
        if self.preferred_sample_rate not in candidates:
            candidates.append(self.preferred_sample_rate)
        for sr in (48000, 44100):
            if sr not in candidates:
                candidates.append(sr)

        for sr in candidates:
            try:
                stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=self.hop_size,
                    latency="low",
                    callback=self._callback,
                )
                stream.start()
                self._stream = stream
                self.sample_rate = sr
                return True
            except Exception as exc:
                self._last_error = str(exc)
        return False

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        _ = (frames, time_info, status)
        if indata.size > 0:
            self.ring.push(indata[:, 0])

    def read_window(self, n_samples: int = 4096) -> np.ndarray:
        """Return latest mono audio samples."""
        return self.ring.read_latest(n_samples)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def close(self) -> None:
        """Close audio input stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a PCM WAV file as normalized mono float32 samples and sample rate."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr
