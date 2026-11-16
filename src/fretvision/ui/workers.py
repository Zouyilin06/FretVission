"""Background Qt worker threads for synchronized video, audio, fusion, and model setup."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from fretvision.audio.audio_source import AudioSource
from fretvision.audio.pitch_detector import HybridHarmonicPitchDetector
from fretvision.config import AppConfig
from fretvision.fusion.av_fusion import (
    AudioVisualFusion,
    FusedFrameResult,
    build_visual_pitch_priors,
)
from fretvision.fusion.tab_recorder import TabColumn, TabRecorder
from fretvision.theory.music import midi_to_freq
from fretvision.vision.fretboard_model import FretboardModel
from fretvision.vision.hand_tracker import FingerObservation, HandTracker, HandTrackingResult
from fretvision.vision.model_assets import ensure_hand_landmarker_model
from fretvision.vision.neck_tracker import (
    NeckTracker,
    NeckTrackingStatus,
    auto_guess_neck_corners,
)
from fretvision.vision.video_source import VideoSource


@dataclass
class EngineSnapshot:
    """Thread-safe snapshot sent from ProcessingWorker to MainWindow."""

    frame_bgr: np.ndarray | None
    fretboard: FretboardModel | None
    hand_result: HandTrackingResult | None
    neck_status: NeckTrackingStatus
    tracked_points: int
    fused: FusedFrameResult
    tab_columns: list[TabColumn]
    video_fps: float
    audio_latency_ms: float
    tuning_name: str
    demo_mode: bool


class HandHistoryBuffer:
    """Timestamped ring buffer of recent hand-tracking observations for AV sync."""

    def __init__(self, max_age_s: float = 0.5) -> None:
        self.max_age_s = float(max_age_s)
        self._items: deque[tuple[float, HandTrackingResult | None]] = deque(maxlen=60)

    def push(self, timestamp_s: float, result: HandTrackingResult | None) -> None:
        self._items.append((float(timestamp_s), result))
        cutoff = timestamp_s - self.max_age_s
        while len(self._items) > 1 and self._items[0][0] < cutoff:
            self._items.popleft()

    def nearest(self, target_ts: float) -> HandTrackingResult | None:
        if not self._items:
            return None
        best_ts, best_res = min(self._items, key=lambda pair: abs(pair[0] - target_ts))
        _ = best_ts
        return best_res

    def clear(self) -> None:
        self._items.clear()


class ModelDownloadWorker(QThread):
    """Downloads MediaPipe hand_landmarker.task off the GUI thread."""

    status_message = Signal(str)
    completed = Signal(bool, str)

    def run(self) -> None:
        try:
            path = ensure_hand_landmarker_model(progress_cb=self.status_message.emit)
            self.completed.emit(True, str(path))
        except Exception as exc:
            self.completed.emit(False, str(exc))


class ProcessingWorker(QThread):
    """Runs camera + hand/neck tracking + audio pitch analysis + fusion off the GUI thread."""

    snapshot_ready = Signal(object)  # EngineSnapshot
    error_occurred = Signal(str)
    status_message = Signal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._lock = threading.RLock()
        self._running = False

        self.video_source = VideoSource(
            source=config.camera.source,
            width=config.camera.width,
            height=config.camera.height,
            target_fps=config.camera.target_fps,
            mirror=config.camera.mirror,
            rotation_deg=config.camera.rotation_deg,
        )
        self.audio_source = AudioSource(
            device_index=config.audio.device_index,
            preferred_sample_rate=config.audio.preferred_sample_rate,
            hop_size=config.audio.hop_size,
        )
        self.neck_tracker = NeckTracker(tracking_mode=config.profile.tracking_mode)
        self.hand_tracker = HandTracker(max_fret=config.profile.guitar.max_fret)
        self.pitch_detector = HybridHarmonicPitchDetector(
            noise_gate_db=config.audio.noise_gate_db,
            onset_flux_threshold=config.audio.onset_flux_threshold,
        )
        self.fusion = AudioVisualFusion(
            tuning_name=config.profile.guitar.tuning_name,
            capo=config.profile.guitar.capo,
            max_fret=config.profile.guitar.max_fret,
            key_half_life_s=config.key_half_life_s,
        )
        self.tab_recorder = TabRecorder()

        if config.profile.corners_px is not None:
            try:
                self.neck_tracker.set_calibration(
                    np.asarray(config.profile.corners_px, dtype=np.float32),
                    start_fret=config.profile.start_fret,
                    end_fret=config.profile.end_fret,
                    max_fret=config.profile.guitar.max_fret,
                )
            except ValueError:
                config.profile.corners_px = None

        self._demo_mode = False
        self._demo_start_s = 0.0
        self._av_sync_offset_ms = float(config.av_sync_offset_ms)
        self._hand_history = HandHistoryBuffer(max_age_s=0.5)
        self._last_frame_ts = 0.0
        self._last_hand_result: HandTrackingResult | None = None
        self._last_frame_bgr: np.ndarray | None = None
        self._last_neck_status = NeckTrackingStatus.UNCALIBRATED
        self._last_tracked_pts = 0
        self._fps_ema = 0.0
        self._last_video_tick_s = 0.0

    def set_demo_mode(self, enabled: bool) -> None:
        with self._lock:
            self._demo_mode = bool(enabled)
            self._demo_start_s = time.monotonic()
            self.fusion.reset_key()
            self.tab_recorder.clear()
            if enabled:
                # Set synthetic 12-fret calibration quad inside 640x360 frame
                demo_corners = np.array(
                    [[70, 120], [560, 110], [560, 240], [70, 230]],
                    dtype=np.float32,
                )
                self.neck_tracker.set_calibration(demo_corners, start_fret=0, end_fret=12, max_fret=22)
                self.neck_tracker.tracking_mode = "static"
                self.status_message.emit("Demo Mode active: playing synthetic G-Em-C-D progression + A Minor riff")
            else:
                self.neck_tracker.tracking_mode = self.config.profile.tracking_mode
                if self.config.profile.corners_px is not None:
                    self.neck_tracker.set_calibration(
                        np.asarray(self.config.profile.corners_px, dtype=np.float32),
                        start_fret=self.config.profile.start_fret,
                        end_fret=self.config.profile.end_fret,
                        max_fret=self.config.profile.guitar.max_fret,
                    )
                else:
                    self.neck_tracker.clear()

    def set_video_source(
        self,
        source: int | str,
        mirror: bool | None = None,
        rotation_deg: int | None = None,
    ) -> None:
        with self._lock:
            self.config.camera.source = source
            if mirror is not None:
                self.config.camera.mirror = mirror
            if rotation_deg is not None:
                self.config.camera.rotation_deg = rotation_deg
            self.video_source.source = source
            self.video_source.mirror = self.config.camera.mirror
            self.video_source.rotation_deg = self.config.camera.rotation_deg
            ok = self.video_source.open()
            if not ok and self.video_source.last_error:
                self.error_occurred.emit(self.video_source.last_error)
            else:
                self.status_message.emit(f"Video connected: {source}")

    def update_video_orientation(self, mirror: bool, rotation_deg: int) -> None:
        with self._lock:
            self.config.camera.mirror = mirror
            self.config.camera.rotation_deg = rotation_deg
            self.video_source.mirror = mirror
            self.video_source.rotation_deg = rotation_deg

    def set_audio_device(self, device_index: int | None) -> None:
        with self._lock:
            self.config.audio.device_index = device_index
            self.audio_source.device_index = device_index
            ok = self.audio_source.open()
            if not ok and self.audio_source.last_error:
                self.error_occurred.emit(f"Audio error: {self.audio_source.last_error}")
            else:
                self.status_message.emit(
                    f"Audio stream active ({self.audio_source.sample_rate} Hz)"
                )

    def set_calibration(self, corners_px: np.ndarray, start_fret: int, end_fret: int) -> None:
        with self._lock:
            try:
                self.neck_tracker.set_calibration(
                    corners_px,
                    start_fret=start_fret,
                    end_fret=end_fret,
                    max_fret=self.config.profile.guitar.max_fret,
                )
            except ValueError as exc:
                self.status_message.emit(f"Invalid calibration: {exc}")
                return
            self.config.profile.corners_px = corners_px.tolist()
            self.config.profile.start_fret = int(start_fret)
            self.config.profile.end_fret = int(end_fret)
            self.config.profile.save()
            self.status_message.emit(f"Fretboard calibrated (frets {start_fret}–{end_fret})")

    def set_av_sync_offset(self, offset_ms: float) -> None:
        with self._lock:
            self._av_sync_offset_ms = float(max(-300.0, min(300.0, offset_ms)))
            self.config.av_sync_offset_ms = self._av_sync_offset_ms

    def auto_guess_calibration(self, start_fret: int, end_fret: int) -> bool:
        with self._lock:
            frame = self._last_frame_bgr
            if frame is None:
                return False
            corners = auto_guess_neck_corners(frame)
            if corners is None:
                return False
            self.set_calibration(corners, start_fret, end_fret)
            return True

    def set_tracking_mode(self, mode: str) -> None:
        with self._lock:
            self.config.profile.tracking_mode = mode  # type: ignore[assignment]
            self.neck_tracker.tracking_mode = mode

    def reset_neck_to_anchor(self) -> None:
        with self._lock:
            self.neck_tracker.reset_to_anchor()

    def clear_calibration(self) -> None:
        with self._lock:
            self.config.profile.corners_px = None
            self.neck_tracker.clear()

    def set_guitar_setup(self, tuning_name: str, capo: int, handedness: str) -> None:
        with self._lock:
            self.config.profile.guitar.tuning_name = tuning_name
            self.config.profile.guitar.capo = capo
            self.config.profile.guitar.handedness = handedness  # type: ignore[assignment]
            self.fusion.set_guitar_config(tuning_name, capo, self.config.profile.guitar.max_fret)
            self.config.profile.save()

    def set_noise_gate(self, db: float) -> None:
        with self._lock:
            self.config.audio.noise_gate_db = float(db)
            self.pitch_detector.noise_gate_db = float(db)

    def reset_key(self) -> None:
        with self._lock:
            self.fusion.reset_key()

    def set_tab_paused(self, paused: bool) -> None:
        with self._lock:
            self.tab_recorder.set_paused(paused)

    def clear_tab(self) -> None:
        with self._lock:
            self.tab_recorder.clear()

    def export_ascii_tab(self, path: str, tuning_name: str) -> str:
        with self._lock:
            return str(self.tab_recorder.export_ascii(path, tuning_name=tuning_name))

    def export_json_tab(self, path: str, tuning_name: str) -> str:
        with self._lock:
            return str(self.tab_recorder.export_json(path, tuning_name=tuning_name))

    def reload_hand_model(self) -> None:
        with self._lock:
            self.hand_tracker.close()
            self.hand_tracker.initialize()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        with self._lock:
            if not self.video_source.open() and self.video_source.last_error:
                self.status_message.emit(
                    "No default webcam opened — pick a camera/URL or try Demo Mode."
                )
            if not self.audio_source.open() and self.audio_source.last_error:
                self.status_message.emit(
                    f"Audio standby ({self.audio_source.last_error})"
                )
            self.hand_tracker.initialize()

        try:
            while self._running:
                t_start = time.monotonic()
                snapshot = self._step(t_start)
                if snapshot is not None:
                    self.snapshot_ready.emit(snapshot)

                elapsed = time.monotonic() - t_start
                sleep_s = max(0.004, 0.028 - elapsed)
                time.sleep(sleep_s)
        finally:
            self.video_source.close()
            self.audio_source.close()
            self.hand_tracker.close()

    def _step(self, now_s: float) -> EngineSnapshot | None:
        with self._lock:
            if self._demo_mode:
                return self._step_demo(now_s)

            frame_bgr, frame_ts = self.video_source.read_latest()
            if frame_bgr is not None and frame_ts > self._last_frame_ts:
                self._last_frame_ts = frame_ts
                self._last_frame_bgr = frame_bgr
                if self._last_video_tick_s > 0.0:
                    dt = max(1e-3, now_s - self._last_video_tick_s)
                    inst_fps = 1.0 / dt
                    self._fps_ema = inst_fps if self._fps_ema == 0.0 else 0.8 * self._fps_ema + 0.2 * inst_fps
                self._last_video_tick_s = now_s

                # Run hand detection first so hand boxes mask out optical flow features
                ts_ms = int(now_s * 1000)
                hand_res = self.hand_tracker.process_frame(
                    frame_bgr,
                    ts_ms,
                    self.neck_tracker.model,
                )
                neck_res = self.neck_tracker.update(frame_bgr, hand_masks=hand_res.hand_bboxes)
                if neck_res.model is not None and hand_res.hand_present:
                    hand_res = self.hand_tracker.project_to_fretboard(hand_res, neck_res.model)

                self._last_hand_result = hand_res
                self._hand_history.push(frame_ts, hand_res)
                self._last_neck_status = neck_res.status
                self._last_tracked_pts = neck_res.tracked_point_count

            t_aud0 = time.perf_counter()
            audio_delay_ms = max(0.0, self._av_sync_offset_ms)
            audio_buf = self.audio_source.read_window(4096, delay_ms=audio_delay_ms)
            if self._av_sync_offset_ms < 0.0:
                sync_hand = self._hand_history.nearest(now_s + (self._av_sync_offset_ms / 1000.0))
                if sync_hand is None:
                    sync_hand = self._last_hand_result
            else:
                sync_hand = self._last_hand_result

            priors = build_visual_pitch_priors(
                sync_hand,
                self.config.profile.guitar.tuning_name,
                capo=self.config.profile.guitar.capo,
                max_fret=self.config.profile.guitar.max_fret,
            )
            audio_res = self.pitch_detector.analyze(
                audio_buf,
                self.audio_source.sample_rate,
                visual_pitch_priors=priors,
            )
            audio_ms = (time.perf_counter() - t_aud0) * 1000.0

            fused = self.fusion.fuse(
                timestamp_s=now_s,
                audio_result=audio_res,
                hand_result=sync_hand,
                fretboard_calibrated=self.neck_tracker.is_calibrated,
            )
            self.tab_recorder.ingest(fused)

            tab_snapshot = [
                TabColumn(c.timestamp_s, dict(c.frets), c.chord_label)
                for c in self.tab_recorder.columns
            ]
            return EngineSnapshot(
                frame_bgr=self._last_frame_bgr,
                fretboard=self.neck_tracker.model,
                hand_result=self._last_hand_result,
                neck_status=self._last_neck_status,
                tracked_points=self._last_tracked_pts,
                fused=fused,
                tab_columns=tab_snapshot,
                video_fps=self._fps_ema,
                audio_latency_ms=audio_ms,
                tuning_name=self.config.profile.guitar.tuning_name,
                demo_mode=False,
            )

    def _step_demo(self, now_s: float) -> EngineSnapshot:
        """Synthesize realistic guitar chords & melody audio + animated neck visuals."""
        t = now_s - self._demo_start_s
        # Cycle: G major chord -> Em chord -> C major chord -> D major chord -> G-A-B-D lead riff
        events: list[tuple[str, tuple[int, ...], tuple[tuple[int, int, str], ...]]] = [
            ("G", (43, 47, 50, 55, 59, 67), ((0, 3, "2"), (1, 2, "1"), (5, 3, "3"))),
            ("Em", (40, 47, 52, 55, 59, 64), ((1, 2, "2"), (2, 2, "3"))),
            ("C", (48, 52, 55, 60, 64), ((1, 3, "3"), (2, 2, "2"), (4, 1, "1"))),
            ("D", (50, 57, 62, 66), ((3, 2, "1"), (4, 3, "3"), (5, 2, "2"))),
            ("G4", (67,), ((5, 3, "2"),)),
            ("A4", (69,), ((5, 5, "4"),)),
            ("B4", (71,), ((5, 7, "4"),)),
            ("D5", (74,), ((5, 10, "4"),)),
        ]
        slot = int(t / 1.1) % len(events)
        _, midis, fretted = events[slot]

        # Synthesize harmonic guitar waveform
        sr = 44100
        n_samp = 4096
        tt = np.arange(n_samp, dtype=np.float64) / sr
        sig = np.zeros(n_samp, dtype=np.float64)
        for m in midis:
            f0 = midi_to_freq(float(m))
            for h_idx in range(1, 5):
                sig += (0.28 / (h_idx ** 0.85)) * np.sin(2.0 * np.pi * f0 * h_idx * tt)
        sig = (sig / max(1.0, np.max(np.abs(sig)))) * 0.35

        # Draw synthetic dark studio neck background for the camera widget
        frame = np.full((360, 640, 3), (22, 26, 34), dtype=np.uint8)
        fb = self.neck_tracker.model
        assert fb is not None
        poly = fb.get_visible_neck_polygon(0, 12).astype(np.int32)
        cv2.fillPoly(frame, [poly], (36, 28, 24))
        cv2.putText(
            frame,
            "SYNTHETIC DEMO FEED (G - Em - C - D + G Major Lead)",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 210, 255),
            1,
            cv2.LINE_AA,
        )

        # Build synthetic HandTrackingResult matching fretted fingers
        press_mat = np.zeros((6, 25), dtype=np.float32)
        obs_list: list[FingerObservation] = []
        for s_idx, f_idx, f_lbl in fretted:
            px, py = fb.fret_string_center_px(s_idx, f_idx)
            press_mat[s_idx, f_idx] = 0.95
            obs_list.append(
                FingerObservation(
                    finger_label=f_lbl,
                    pixel_xy=(px, py),
                    string_float=float(s_idx),
                    fret_float=float(f_idx),
                    string_idx=s_idx,
                    fret_idx=f_idx,
                    in_bounds=True,
                    confidence=0.95,
                )
            )
            cv2.circle(frame, (int(px), int(py)), 9, (16, 185, 129), -1)

        hand_res = HandTrackingResult(
            hand_present=True,
            landmarks_px=[],
            hand_bboxes=[],
            fingers=obs_list,
            press_prob_matrix=press_mat,
            model_ready=True,
        )

        priors = build_visual_pitch_priors(hand_res, "Standard (EADGBE)", capo=0, max_fret=22)
        audio_res = self.pitch_detector.analyze(sig.astype(np.float32), sr, visual_pitch_priors=priors)
        # Force clean onset at slot transitions
        slot_phase = (t % 1.1)
        if slot_phase < 0.06:
            from dataclasses import replace

            audio_res = replace(audio_res, is_onset=True)

        fused = self.fusion.fuse(now_s, audio_res, hand_res, fretboard_calibrated=True)
        self.tab_recorder.ingest(fused)

        tab_snapshot = [
            TabColumn(c.timestamp_s, dict(c.frets), c.chord_label)
            for c in self.tab_recorder.columns
        ]
        return EngineSnapshot(
            frame_bgr=frame,
            fretboard=fb,
            hand_result=hand_res,
            neck_status=NeckTrackingStatus.STATIC,
            tracked_points=0,
            fused=fused,
            tab_columns=tab_snapshot,
            video_fps=30.0,
            audio_latency_ms=1.8,
            tuning_name="Standard (EADGBE)",
            demo_mode=True,
        )
