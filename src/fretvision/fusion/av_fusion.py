"""Audio-visual sensor fusion resolving exact (string, fret) positions, chords, and keys."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fretvision.audio.pitch_detector import AudioAnalysisResult, DetectedPitch
from fretvision.theory.key_detector import KeyEstimate, KeyEstimator
from fretvision.theory.music import (
    ChordMatch,
    GuitarPosition,
    base_open_midis,
    get_valid_positions,
    midi_to_name,
    recognize_chord,
)
from fretvision.vision.hand_tracker import FingerObservation, HandTrackingResult


@dataclass(frozen=True)
class ActiveNote:
    """Resolved note on the guitar neck with fusion source metadata."""

    string_idx: int      # 0 (Low E) .. 5 (High E)
    fret: int            # 0 .. max_fret
    midi: int
    note_name: str
    confidence: float
    source: str          # "av_fused", "audio_only", "visual_hover"
    finger_label: str | None = None
    cents_offset: float = 0.0
    bend_semitones: int = 0
    technique: str | None = None  # "b", "h", "p", "/"

    @property
    def string_number(self) -> int:
        return 6 - self.string_idx


@dataclass
class FusedFrameResult:
    """Complete musical state for a single synchronized video+audio tick."""

    timestamp_s: float
    sounding_notes: tuple[ActiveNote, ...]
    hover_notes: tuple[ActiveNote, ...]
    chord: ChordMatch | None
    key_estimate: KeyEstimate | None
    rms_db: float
    is_onset: bool
    dominant_cents: float


def build_visual_pitch_priors(
    hand_result: HandTrackingResult | None,
    tuning_name: str,
    capo: int = 0,
    max_fret: int = 22,
    custom_tuning: tuple[int, int, int, int, int, int] | None = None,
) -> dict[int, float]:
    """Convert visual fretboard finger probabilities into expected MIDI pitch priors."""
    if hand_result is None or not hand_result.hand_present:
        return {}

    mat = hand_result.press_prob_matrix
    base_midis = base_open_midis(tuning_name, custom_tuning)
    priors: dict[int, float] = {}

    for s_idx, open_midi in enumerate(base_midis):
        capo_midi = open_midi + capo
        col = mat[s_idx]
        max_col = float(np.max(col[capo + 1 : max_fret + 1])) if max_fret > capo else 0.0
        if max_col < 0.25:
            # No finger blocking this string above capo -> open string plausible
            priors[capo_midi] = max(priors.get(capo_midi, 0.0), 0.35)

        for f in range(max(1, capo + 1), min(max_fret + 1, mat.shape[1])):
            p = float(col[f])
            if p >= 0.25:
                midi = capo_midi + (f - capo)
                priors[midi] = max(priors.get(midi, 0.0), p)

    return priors


class AudioVisualFusion:
    """Fuses audio multi-pitch detections with camera hand/fretboard tracking."""

    def __init__(
        self,
        tuning_name: str = "Standard (EADGBE)",
        capo: int = 0,
        max_fret: int = 22,
        key_half_life_s: float = 6.0,
    ) -> None:
        self.tuning_name = tuning_name
        self.capo = int(capo)
        self.max_fret = int(max_fret)
        self.custom_tuning: tuple[int, int, int, int, int, int] | None = None
        self.key_estimator = KeyEstimator(half_life_s=key_half_life_s)

        self._prev_sounding: dict[tuple[int, int], int] = {}  # (s, f) -> midi
        self._prev_by_string: dict[int, tuple[int, str | None]] = {}  # string_idx -> (fret, finger_label)
        self._pending_audio_only: dict[int, int] = {}  # midi -> consecutive frame count
        self._last_mono_anchor: tuple[int, int, int] | None = None  # (s, f, base_midi)

    def reset_key(self) -> None:
        """Reset accumulated key detection context."""
        self.key_estimator.reset()

    def set_guitar_config(
        self,
        tuning_name: str,
        capo: int,
        max_fret: int = 22,
        custom_tuning: tuple[int, int, int, int, int, int] | None = None,
    ) -> None:
        """Update tuning, capo, or fret span."""
        self.tuning_name = tuning_name
        self.capo = max(0, min(12, int(capo)))
        self.max_fret = int(max_fret)
        self.custom_tuning = custom_tuning
        self._prev_sounding.clear()
        self._prev_by_string.clear()
        self._pending_audio_only.clear()
        self._last_mono_anchor = None

    def fuse(
        self,
        timestamp_s: float,
        audio_result: AudioAnalysisResult,
        hand_result: HandTrackingResult | None,
        fretboard_calibrated: bool,
    ) -> FusedFrameResult:
        """Combine current audio pitches and visual finger positions into resolved notes & key."""
        base_midis = base_open_midis(self.tuning_name, self.custom_tuning)
        has_visual = bool(
            fretboard_calibrated and hand_result is not None and hand_result.hand_present
        )

        # Build finger-lookup map at discrete (s, f)
        finger_map: dict[tuple[int, int], FingerObservation] = {}
        hover_notes: list[ActiveNote] = []
        hand_frets: list[int] = []

        if has_visual and hand_result is not None:
            for obs in hand_result.fingers:
                if not obs.in_bounds or obs.string_idx is None or obs.fret_idx is None:
                    continue
                if obs.fret_idx < self.capo:
                    continue
                key = (obs.string_idx, obs.fret_idx)
                if key not in finger_map or obs.confidence > finger_map[key].confidence:
                    finger_map[key] = obs
                if obs.fret_idx > self.capo:
                    hand_frets.append(obs.fret_idx)
                midi = base_midis[obs.string_idx] + obs.fret_idx
                hover_notes.append(
                    ActiveNote(
                        string_idx=obs.string_idx,
                        fret=obs.fret_idx,
                        midi=midi,
                        note_name=midi_to_name(midi),
                        confidence=obs.confidence,
                        source="visual_hover",
                        finger_label=obs.finger_label,
                    )
                )

        hand_center_fret = float(np.median(hand_frets)) if hand_frets else None
        sounding_notes: list[ActiveNote] = []

        if not audio_result.is_silent and audio_result.pitches:
            raw_notes = self._assign_pitches(
                audio_result.pitches,
                base_midis,
                hand_result if has_visual else None,
                finger_map,
                hand_center_fret,
            )
            # Temporal glitch filter for uncorroborated non-onset audio_only spikes
            prev_midis = set(self._prev_sounding.values())
            new_pending: dict[int, int] = {}
            for note in raw_notes:
                if (
                    note.source == "av_fused"
                    or audio_result.is_onset
                    or note.midi in prev_midis
                    or note.confidence >= 0.85
                ):
                    sounding_notes.append(note)
                else:
                    cnt = self._pending_audio_only.get(note.midi, 0) + 1
                    new_pending[note.midi] = cnt
                    if cnt >= 2:
                        sounding_notes.append(note)
            self._pending_audio_only = new_pending
            sounding_notes = self._annotate_techniques(sounding_notes, audio_result.is_onset)
        else:
            self._pending_audio_only.clear()

        # Update temporal continuity & bend anchor
        self._prev_sounding = {(n.string_idx, n.fret): n.midi for n in sounding_notes}
        self._prev_by_string = {n.string_idx: (n.fret, n.finger_label) for n in sounding_notes}
        if len(sounding_notes) == 1 and sounding_notes[0].bend_semitones == 0:
            n0 = sounding_notes[0]
            self._last_mono_anchor = (n0.string_idx, n0.fret, n0.midi)
        elif not sounding_notes:
            self._last_mono_anchor = None

        # Remove hover markers that overlap confirmed sounding positions
        sounding_keys = {(n.string_idx, n.fret) for n in sounding_notes}
        hover_filtered = tuple(h for h in hover_notes if (h.string_idx, h.fret) not in sounding_keys)

        # Chord and Key updates from sounding pitches
        active_midis = [n.midi for n in sounding_notes]
        active_weights = [n.confidence for n in sounding_notes]
        chord = recognize_chord(active_midis) if len(active_midis) >= 2 else None
        key_est = self.key_estimator.update(
            timestamp_s=timestamp_s,
            active_midis=active_midis,
            weights=active_weights,
            is_onset=audio_result.is_onset,
        )

        return FusedFrameResult(
            timestamp_s=timestamp_s,
            sounding_notes=tuple(sounding_notes),
            hover_notes=hover_filtered,
            chord=chord,
            key_estimate=key_est,
            rms_db=audio_result.rms_db,
            is_onset=audio_result.is_onset,
            dominant_cents=audio_result.dominant_cents,
        )

    def _assign_pitches(
        self,
        pitches: tuple[DetectedPitch, ...],
        base_midis: tuple[int, int, int, int, int, int],
        hand_result: HandTrackingResult | None,
        finger_map: dict[tuple[int, int], FingerObservation],
        hand_center_fret: float | None,
    ) -> list[ActiveNote]:
        """Globally solve string/fret assignment under physical string-uniqueness constraint."""
        # Check for single-note string bend from anchored fret
        if len(pitches) == 1 and self._last_mono_anchor is not None and hand_result is not None:
            p0 = pitches[0]
            a_s, a_f, a_midi = self._last_mono_anchor
            delta_semi = p0.midi - a_midi
            if a_f > self.capo and delta_semi in (1, 2):
                vis_prob = float(hand_result.press_prob_matrix[a_s, min(a_f, hand_result.press_prob_matrix.shape[1] - 1)])
                if vis_prob >= 0.40:
                    f_obs = finger_map.get((a_s, a_f))
                    return [
                        ActiveNote(
                            string_idx=a_s,
                            fret=a_f,
                            midi=p0.midi,
                            note_name=midi_to_name(p0.midi),
                            confidence=p0.confidence,
                            source="av_fused",
                            finger_label=f_obs.finger_label if f_obs else None,
                            cents_offset=p0.cents_offset,
                            bend_semitones=delta_semi,
                            technique="b",
                        )
                    ]

        # Build scored candidates per pitch
        pitch_candidates: list[tuple[DetectedPitch, list[tuple[GuitarPosition, float, str]]]] = []
        for pitch in pitches:
            positions = get_valid_positions(
                pitch.midi,
                base_midis,
                max_fret=self.max_fret,
                capo=self.capo,
            )
            if not positions:
                continue

            scored: list[tuple[GuitarPosition, float, str]] = []
            for pos in positions:
                score, src = self._score_position(
                    pos,
                    hand_result,
                    hand_center_fret,
                )
                scored.append((pos, score, src))
            scored.sort(key=lambda item: item[1], reverse=True)
            pitch_candidates.append((pitch, scored))

        if not pitch_candidates:
            return []

        # Exact search over up to 6 active pitches ensuring at most 1 note per string
        best_total = -1e9
        best_assignment: list[tuple[DetectedPitch, GuitarPosition, str, float]] = []

        def _search(
            idx: int,
            used_strings: int,
            acc_score: float,
            acc: list[tuple[DetectedPitch, GuitarPosition, str, float]],
        ) -> None:
            nonlocal best_total, best_assignment
            if idx == len(pitch_candidates):
                if acc_score > best_total:
                    best_total = acc_score
                    best_assignment = list(acc)
                return

            pitch, cands = pitch_candidates[idx]
            assigned_any = False
            for pos, score, src in cands:
                bit = 1 << pos.string_idx
                if used_strings & bit:
                    continue
                assigned_any = True
                acc.append((pitch, pos, src, score))
                _search(idx + 1, used_strings | bit, acc_score + score, acc)
                acc.pop()

            # Allow dropping a weak conflicting duplicate pitch rather than failing whole chord
            if not assigned_any:
                _search(idx + 1, used_strings, acc_score - 0.5, acc)

        _search(0, 0, 0.0, [])

        notes: list[ActiveNote] = []
        for pitch, pos, src, _score in best_assignment:
            f_obs = finger_map.get((pos.string_idx, pos.fret))
            notes.append(
                ActiveNote(
                    string_idx=pos.string_idx,
                    fret=pos.fret,
                    midi=pitch.midi,
                    note_name=midi_to_name(pitch.midi),
                    confidence=pitch.confidence,
                    source=src,
                    finger_label=f_obs.finger_label if f_obs else None,
                    cents_offset=pitch.cents_offset,
                    bend_semitones=0,
                )
            )
        notes.sort(key=lambda n: n.string_idx)
        return notes

    def _score_position(
        self,
        pos: GuitarPosition,
        hand_result: HandTrackingResult | None,
        hand_center_fret: float | None,
    ) -> tuple[float, str]:
        """Score a candidate (string, fret) given visual evidence and guitar ergonomics."""
        score = 0.0
        source = "audio_only"

        if hand_result is not None and hand_result.hand_present:
            mat = hand_result.press_prob_matrix
            if pos.fret == self.capo:
                # Open string: penalize if a finger presses higher up on the same string
                hi_slice = mat[pos.string_idx, self.capo + 1 :]
                max_blocking = float(np.max(hi_slice)) if hi_slice.size > 0 else 0.0
                if max_blocking >= 0.55:
                    score -= 0.85
                elif max_blocking < 0.25:
                    score += 0.40
                    source = "av_fused"
            else:
                f_idx = min(pos.fret, mat.shape[1] - 1)
                vis_prob = float(mat[pos.string_idx, f_idx])
                if vis_prob >= 0.25:
                    score += 2.4 * vis_prob
                    source = "av_fused"
                else:
                    score -= 0.20
                # Same-string physics: a higher pressed fret blocks any lower fret from sounding
                if f_idx + 1 < mat.shape[1]:
                    hi_block = float(np.max(mat[pos.string_idx, f_idx + 1 :]))
                    if hi_block >= 0.55:
                        score -= 0.75

        # Ergonomic hand-span preference
        if hand_center_fret is not None and pos.fret > self.capo:
            dist = abs(pos.fret - hand_center_fret)
            if dist > 2.5:
                score -= 0.14 * (dist - 2.5)
        elif pos.fret > self.capo:
            # Mild open-position bias in pure audio mode
            score -= 0.02 * pos.fret

        # Temporal continuity hysteresis
        if (pos.string_idx, pos.fret) in self._prev_sounding:
            score += 0.35

        return score, source

    def _annotate_techniques(
        self,
        notes: list[ActiveNote],
        is_onset: bool,
    ) -> list[ActiveNote]:
        """Tag legato transitions (hammer-on, pull-off, slide) on the same string."""
        if not self._prev_by_string or is_onset:
            return notes
        annotated: list[ActiveNote] = []
        for n in notes:
            if n.technique is not None:
                annotated.append(n)
                continue
            prev = self._prev_by_string.get(n.string_idx)
            if prev is None:
                annotated.append(n)
                continue
            prev_fret, prev_finger = prev
            delta = n.fret - prev_fret
            tech: str | None = None
            if delta != 0:
                same_finger = (
                    n.finger_label is not None
                    and prev_finger is not None
                    and n.finger_label == prev_finger
                )
                if same_finger or abs(delta) >= 4:
                    tech = "/"
                elif delta > 0:
                    tech = "h"
                elif delta < 0:
                    tech = "p"
            if tech is not None:
                annotated.append(
                    ActiveNote(
                        string_idx=n.string_idx,
                        fret=n.fret,
                        midi=n.midi,
                        note_name=n.note_name,
                        confidence=n.confidence,
                        source=n.source,
                        finger_label=n.finger_label,
                        cents_offset=n.cents_offset,
                        bend_semitones=n.bend_semitones,
                        technique=tech,
                    )
                )
            else:
                annotated.append(n)
        return annotated
