"""Real-time musical key and scale detection from live guitar note/chord events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .music import pc_to_name

# Temperley (1999/2007) key profiles tuned for polyphonic chord progressions & rock/pop guitar (C = 0).
_PROFILE_MAJOR = np.array(
    [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
    dtype=np.float64,
)
_PROFILE_MINOR = np.array(
    [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.8, 3.8],
    dtype=np.float64,
)

# Scale interval sets relative to tonic (0..11) for guitar scale highlighting.
SCALE_INTERVALS: dict[str, tuple[int, ...]] = {
    "Major": (0, 2, 4, 5, 7, 9, 11),
    "Minor": (0, 2, 3, 5, 7, 8, 10),
    "Minor Pentatonic": (0, 3, 5, 7, 10),
    "Major Pentatonic": (0, 2, 4, 7, 9),
    "Dorian": (0, 2, 3, 5, 7, 9, 10),
    "Mixolydian": (0, 2, 4, 5, 7, 9, 10),
}


def _zscore(vec: np.ndarray) -> np.ndarray:
    centered = vec - np.mean(vec)
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 1e-12 else centered


_MAJOR_TEMPLATES = np.stack([_zscore(np.roll(_PROFILE_MAJOR, i)) for i in range(12)])
_MINOR_TEMPLATES = np.stack([_zscore(np.roll(_PROFILE_MINOR, i)) for i in range(12)])


@dataclass(frozen=True)
class KeyCandidate:
    """Ranked musical key candidate."""

    tonic_pc: int
    mode: str  # "Major" or "Minor"
    score: float

    @property
    def name(self) -> str:
        return f"{pc_to_name(self.tonic_pc)} {self.mode}"

    @property
    def relative_name(self) -> str:
        if self.mode == "Major":
            return f"{pc_to_name((self.tonic_pc - 3) % 12)} Minor"
        return f"{pc_to_name((self.tonic_pc + 3) % 12)} Major"


@dataclass(frozen=True)
class KeyEstimate:
    """Current smoothed musical key & scale estimate."""

    tonic_pc: int
    mode: str
    display_name: str
    relative_name: str
    confidence: float
    scale_label: str
    scale_pcs: frozenset[int]
    top_candidates: tuple[KeyCandidate, ...]
    active_note_count: int


class KeyEstimator:
    """Tracks pitch-class energy over time to estimate the current musical key."""

    def __init__(
        self,
        half_life_s: float = 6.0,
        switch_margin: float = 0.055,
        dwell_seconds: float = 1.0,
    ) -> None:
        self.half_life_s = max(1.0, float(half_life_s))
        self.switch_margin = float(switch_margin)
        self.dwell_seconds = float(dwell_seconds)

        self._pc_weights = np.zeros(12, dtype=np.float64)
        self._bass_weights = np.zeros(12, dtype=np.float64)
        self._last_time_s: float | None = None
        self._note_events = 0

        self._current_key: tuple[int, str] | None = None
        self._pending_key: tuple[int, str] | None = None
        self._pending_since_s: float = 0.0

    def reset(self) -> None:
        """Clear accumulated harmonic memory."""
        self._pc_weights.fill(0.0)
        self._bass_weights.fill(0.0)
        self._last_time_s = None
        self._note_events = 0
        self._current_key = None
        self._pending_key = None
        self._pending_since_s = 0.0

    def update(
        self,
        timestamp_s: float,
        active_midis: Iterable[int],
        weights: Iterable[float] | None = None,
        is_onset: bool = False,
    ) -> KeyEstimate | None:
        """Ingest active MIDI pitches at timestamp_s and return current key estimate."""
        if self._last_time_s is not None and timestamp_s > self._last_time_s:
            dt = min(5.0, timestamp_s - self._last_time_s)
            decay = math.exp(-math.log(2.0) * dt / self.half_life_s)
            self._pc_weights *= decay
            self._bass_weights *= decay
        self._last_time_s = timestamp_s

        midis_list = sorted(int(m) for m in active_midis)
        if midis_list:
            w_list = (
                list(weights)
                if weights is not None
                else [1.0] * len(midis_list)
            )
            onset_mult = 2.0 if is_onset else 0.45
            # Deduplicate octave-doubled strings within a single chord frame
            frame_pc_weights: dict[int, float] = {}
            for midi_val, w in zip(midis_list, w_list):
                pc = midi_val % 12
                frame_pc_weights[pc] = max(frame_pc_weights.get(pc, 0.0), max(0.1, float(w)))
            for pc, w in frame_pc_weights.items():
                self._pc_weights[pc] += w * onset_mult
            # Give extra tonic-resolution evidence to the lowest sounding pitch (bass root)
            bass_pc = midis_list[0] % 12
            self._bass_weights[bass_pc] += 0.85 * onset_mult
            if is_onset:
                self._note_events += 1

        return self.current_estimate(timestamp_s)

    def current_estimate(self, timestamp_s: float | None = None) -> KeyEstimate | None:
        """Compute current key estimate without adding new note evidence."""
        total_energy = float(np.sum(self._pc_weights))
        if total_energy < 0.8:
            return None

        combined = self._pc_weights + 0.45 * self._bass_weights
        hist_norm = _zscore(combined)
        distinct_pcs = int(np.sum(self._pc_weights > (0.06 * np.max(self._pc_weights))))

        candidates: list[KeyCandidate] = []
        bass_total = float(np.sum(self._bass_weights)) + 1e-9
        for tonic in range(12):
            bass_affinity = 0.08 * (self._bass_weights[tonic] / bass_total)
            maj_corr = float(np.dot(hist_norm, _MAJOR_TEMPLATES[tonic])) + bass_affinity
            min_corr = float(np.dot(hist_norm, _MINOR_TEMPLATES[tonic])) + bass_affinity

            # If sparse pentatonic playing (<=5 distinct pitch classes), give slight boost
            # when tones strictly fit Minor/Major Pentatonic of that tonic.
            if distinct_pcs <= 5:
                active_pcs = {
                    i for i in range(12)
                    if self._pc_weights[i] > 0.12 * np.max(self._pc_weights)
                }
                min_pent = {(tonic + s) % 12 for s in SCALE_INTERVALS["Minor Pentatonic"]}
                maj_pent = {(tonic + s) % 12 for s in SCALE_INTERVALS["Major Pentatonic"]}
                if active_pcs and active_pcs.issubset(min_pent):
                    min_corr += 0.06
                if active_pcs and active_pcs.issubset(maj_pent):
                    maj_corr += 0.06

            candidates.append(KeyCandidate(tonic_pc=tonic, mode="Major", score=maj_corr))
            candidates.append(KeyCandidate(tonic_pc=tonic, mode="Minor", score=min_corr))

        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[0]
        now_s = timestamp_s if timestamp_s is not None else (self._last_time_s or 0.0)

        # Hysteresis prevents jumping on brief passing chromatic tones
        chosen_key = (top.tonic_pc, top.mode)
        if self._current_key is None:
            self._current_key = chosen_key
        elif chosen_key != self._current_key:
            curr_score = next(
                c.score for c in candidates if (c.tonic_pc, c.mode) == self._current_key
            )
            if top.score - curr_score >= self.switch_margin:
                if self._pending_key != chosen_key:
                    self._pending_key = chosen_key
                    self._pending_since_s = now_s
                elif (now_s - self._pending_since_s) >= self.dwell_seconds:
                    self._current_key = chosen_key
                    self._pending_key = None
            else:
                self._pending_key = None

        cur_tonic, cur_mode = self._current_key
        cur_candidate = next(
            c for c in candidates if (c.tonic_pc, c.mode) == self._current_key
        )
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        # Map correlation + separation into intuitive [0..1] confidence
        conf_raw = 0.55 * max(0.0, cur_candidate.score) + 0.45 * min(
            1.0, max(0.0, (cur_candidate.score - second_score) * 3.0 + 0.35)
        )
        # Ramp confidence smoothly on very sparse (<3) note samples
        evidence_factor = min(1.0, max(0.25, distinct_pcs / 4.0))
        confidence = float(max(0.1, min(0.99, conf_raw * evidence_factor)))

        # Choose most descriptive scale label (Pentatonic vs full Major/Minor)
        active_pcs = {
            i for i in range(12)
            if self._pc_weights[i] > 0.10 * np.max(self._pc_weights)
        }
        scale_label = cur_mode
        if cur_mode == "Minor":
            pent = {(cur_tonic + s) % 12 for s in SCALE_INTERVALS["Minor Pentatonic"]}
            if 3 <= len(active_pcs) <= 5 and active_pcs.issubset(pent):
                scale_label = "Minor Pentatonic"
            scale_pcs = frozenset((cur_tonic + s) % 12 for s in SCALE_INTERVALS[scale_label])
        else:
            pent = {(cur_tonic + s) % 12 for s in SCALE_INTERVALS["Major Pentatonic"]}
            if 3 <= len(active_pcs) <= 5 and active_pcs.issubset(pent):
                scale_label = "Major Pentatonic"
            scale_pcs = frozenset((cur_tonic + s) % 12 for s in SCALE_INTERVALS[scale_label])

        return KeyEstimate(
            tonic_pc=cur_tonic,
            mode=cur_mode,
            display_name=cur_candidate.name,
            relative_name=cur_candidate.relative_name,
            confidence=confidence,
            scale_label=f"{pc_to_name(cur_tonic)} {scale_label}",
            scale_pcs=scale_pcs,
            top_candidates=tuple(candidates[:3]),
            active_note_count=self._note_events,
        )
