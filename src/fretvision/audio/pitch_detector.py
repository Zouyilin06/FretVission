"""Hybrid low-latency pitch detection: vectorized YIN + Harmonic-Template NNLS."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from fretvision.theory.music import freq_to_midi_and_cents, midi_to_freq, midi_to_name


@dataclass(frozen=True)
class DetectedPitch:
    """Detected active pitch in the current audio frame."""

    midi: int
    freq_hz: float
    cents_offset: float
    confidence: float
    is_monophonic: bool = False

    @property
    def note_name(self) -> str:
        return midi_to_name(self.midi)


@dataclass(frozen=True)
class AudioAnalysisResult:
    """Output from a single audio analysis tick."""

    pitches: tuple[DetectedPitch, ...]
    rms_db: float
    is_onset: bool
    is_silent: bool
    dominant_cents: float
    spectrum_bands: tuple[float, ...] = ()


def yin_pitch(
    samples: np.ndarray,
    sample_rate: int,
    fmin: float = 65.0,
    fmax: float = 1400.0,
    threshold: float = 0.16,
) -> tuple[float | None, float]:
    """Fast FFT-accelerated YIN monophonic fundamental frequency estimator.

    Returns (freq_hz_or_None, confidence_0_to_1).
    """
    x = np.asarray(samples, dtype=np.float64)
    n = len(x)
    if n < 512:
        return None, 0.0

    # Window size W = n // 2 so lag tau runs up to n // 2
    w = n // 2
    tau_min = max(2, int(math.floor(sample_rate / fmax)))
    tau_max = min(w - 2, int(math.ceil(sample_rate / fmin)))
    if tau_max <= tau_min:
        return None, 0.0

    x = x - np.mean(x)
    # Cross-correlation via FFT: acf[tau] = sum_{j=0}^{W-1} x[j] * x[j + tau]
    nfft = 1 << (n - 1).bit_length() * 1
    if nfft < n + w:
        nfft *= 2
    # Pad x[:W] reversed or correlate x[:W] with full x
    x_head = x[:w]
    fft_x = np.fft.rfft(x, n=nfft)
    fft_head = np.fft.rfft(x_head, n=nfft)
    acf = np.fft.irfft(fft_x * np.conj(fft_head), n=nfft)[: w + 1]

    # Cumulative squared energies for sliding windows x[tau : tau + W]
    sq = x * x
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    energy_head = csum[w]
    taus = np.arange(0, w + 1)
    energy_shift = csum[taus + w] - csum[taus]

    diff = np.maximum(0.0, energy_head + energy_shift - 2.0 * acf)

    # Cumulative Mean Normalized Difference Function (CMNDF)
    cmndf = np.ones(w + 1, dtype=np.float64)
    running_sum = np.cumsum(diff[1:])
    idx = np.arange(1, w + 1, dtype=np.float64)
    valid = running_sum > 1e-12
    cmndf[1:][valid] = diff[1:][valid] * idx[valid] / running_sum[valid]

    sub = cmndf[tau_min : tau_max + 1]
    below = np.where(sub < threshold)[0]
    if len(below) > 0:
        # Walk forward to the local minimum of the first dip below threshold
        k = int(below[0])
        while k + 1 < len(sub) and sub[k + 1] < sub[k]:
            k += 1
        best_tau = tau_min + k
        best_val = float(sub[k])
    else:
        k = int(np.argmin(sub))
        best_tau = tau_min + k
        best_val = float(sub[k])
        if best_val > 0.35:
            return None, 0.0

    # Parabolic interpolation around best_tau
    if 1 <= best_tau < w:
        s0, s1, s2 = cmndf[best_tau - 1], cmndf[best_tau], cmndf[best_tau + 1]
        denom = 2.0 * (s0 - 2.0 * s1 + s2)
        if abs(denom) > 1e-12:
            shift = np.clip((s0 - s2) / denom, -0.5, 0.5)
            tau_refined = best_tau + float(shift)
        else:
            tau_refined = float(best_tau)
    else:
        tau_refined = float(best_tau)

    freq = sample_rate / max(1e-6, tau_refined)
    conf = max(0.0, min(1.0, 1.0 - best_val))
    return float(freq), float(conf)


class PitchDetector(ABC):
    """Abstract interface for pitch detection backends."""

    @abstractmethod
    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        visual_pitch_priors: dict[int, float] | None = None,
    ) -> AudioAnalysisResult:
        """Analyze an audio window and return active pitches, onset flag, and RMS."""


def _hann_kernel_mag(df_hz: np.ndarray, window_len: int, sample_rate: int) -> np.ndarray:
    """Analytical Fourier magnitude of a Hann window at frequency offset df_hz."""
    k = df_hz * (window_len / float(sample_rate))
    return np.abs(0.5 * np.sinc(k) + 0.25 * np.sinc(k - 1.0) + 0.25 * np.sinc(k + 1.0))


class HybridHarmonicPitchDetector(PitchDetector):
    """Real-time guitar pitch detector combining multi-res FFT, Harmonic NNLS, and YIN."""

    def __init__(
        self,
        midi_min: int = 36,  # C2 (covers Drop C and below)
        midi_max: int = 89,  # F6 (25th fret high E)
        bins_per_semitone: int = 3,
        noise_gate_db: float = -48.0,
        onset_flux_threshold: float = 0.25,
    ) -> None:
        self.midi_min = int(midi_min)
        self.midi_max = int(midi_max)
        self.bins_per_semitone = int(bins_per_semitone)
        self.noise_gate_db = float(noise_gate_db)
        self.onset_flux_threshold = float(onset_flux_threshold)

        # Log-frequency grid spanning MIDI 36 .. 108 (up to ~4186 Hz for upper harmonics)
        self.grid_midi_min = float(self.midi_min)
        self.grid_midi_max = 108.0
        n_semis = int(self.grid_midi_max - self.grid_midi_min)
        self.n_log_bins = n_semis * self.bins_per_semitone + 1
        self.log_midis = np.linspace(self.grid_midi_min, self.grid_midi_max, self.n_log_bins)
        self.log_freqs = np.array([midi_to_freq(m) for m in self.log_midis], dtype=np.float64)

        self.candidate_midis = np.arange(self.midi_min, self.midi_max + 1, dtype=int)
        self._dict_cache: dict[int, np.ndarray] = {}
        self._prev_spec: np.ndarray | None = None
        self._prev_rms: float = 0.0

    def _get_dictionary(self, sample_rate: int) -> np.ndarray:
        """Build or return cached (n_log_bins, n_candidates) Hann-matched harmonic templates."""
        sr = int(sample_rate)
        if sr in self._dict_cache:
            return self._dict_cache[sr]

        n_cand = len(self.candidate_midis)
        W = np.zeros((self.n_log_bins, n_cand), dtype=np.float64)
        alpha = np.clip((self.log_freqs - 180.0) / 140.0, 0.0, 1.0)

        for col, midi in enumerate(self.candidate_midis):
            f0 = midi_to_freq(float(midi))
            for h_idx in range(1, 6):
                fh = f0 * h_idx
                if fh > self.log_freqs[-1]:
                    break
                kl = _hann_kernel_mag(self.log_freqs - fh, 4096, sr)
                ks = _hann_kernel_mag(self.log_freqs - fh, 2048, sr)
                lobe = (1.0 - alpha) * kl + alpha * ks
                amp = 1.0 / (h_idx ** 0.78)
                if midi < 45 and h_idx == 1:
                    amp *= 0.90
                W[:, col] += amp * lobe

            norm = np.linalg.norm(W[:, col])
            if norm > 1e-12:
                W[:, col] /= norm

        self._dict_cache[sr] = W
        return W

    def _compute_multires_spectrum(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Blend 4096-sample (zero-padded 8192) low-freq FFT and 2048-sample high-freq FFT."""
        n = len(samples)
        if n >= 4096:
            long_seg = samples[-4096:]
        else:
            long_seg = np.pad(samples, (4096 - n, 0))
        short_seg = long_seg[-2048:]

        win_long = np.hanning(4096)
        win_short = np.hanning(2048)

        mag_long = np.abs(np.fft.rfft(long_seg * win_long, n=8192)) / (np.sum(win_long) * 0.5)
        freqs_long = np.fft.rfftfreq(8192, d=1.0 / sample_rate)

        mag_short = np.abs(np.fft.rfft(short_seg * win_short, n=4096)) / (np.sum(win_short) * 0.5)
        freqs_short = np.fft.rfftfreq(4096, d=1.0 / sample_rate)

        interp_long = np.interp(self.log_freqs, freqs_long, mag_long)
        interp_short = np.interp(self.log_freqs, freqs_short, mag_short)

        # Smooth crossover around 220 Hz (A3): crisp low-E resolution + fast high-note onset
        alpha = np.clip((self.log_freqs - 180.0) / 140.0, 0.0, 1.0)
        return (1.0 - alpha) * interp_long + alpha * interp_short

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        visual_pitch_priors: dict[int, float] | None = None,
    ) -> AudioAnalysisResult:
        """Detect single or polyphonic pitches, onset, RMS level, and bend cents."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(x) < 512:
            return AudioAnalysisResult(
                pitches=(),
                rms_db=-96.0,
                is_onset=False,
                is_silent=True,
                dominant_cents=0.0,
            )

        rms = float(np.sqrt(np.mean(x[-2048:] ** 2) + 1e-12))
        rms_db = 20.0 * math.log10(rms + 1e-9)

        if rms_db < self.noise_gate_db:
            self._prev_spec = None
            self._prev_rms = rms
            return AudioAnalysisResult(
                pitches=(),
                rms_db=rms_db,
                is_onset=False,
                is_silent=True,
                dominant_cents=0.0,
            )

        spec = self._compute_multires_spectrum(x, sample_rate)
        spec_bands = self._downsample_spectrum_bands(spec, n_bands=24)
        # Positive spectral flux detects both loud attacks and quiet high-string plucks over ringing bass
        is_onset = False
        if self._prev_spec is not None:
            flux = float(np.sum(np.maximum(0.0, spec - self._prev_spec)))
            norm_flux = flux / (float(np.sum(self._prev_spec)) + 1e-6)
            if (norm_flux > self.onset_flux_threshold and rms > self._prev_rms * 1.03) or (
                norm_flux > self.onset_flux_threshold * 1.45
            ):
                is_onset = True
        else:
            is_onset = True

        self._prev_spec = spec.copy()
        self._prev_rms = rms

        # Solve Harmonic NNLS: spec ~= W @ weights
        dictionary = self._get_dictionary(sample_rate)
        weights, _ = nnls(dictionary, spec)

        # Reject broadband percussive muted strums / pick scrapes via spectral flatness
        spec_pos = spec + 1e-12
        flatness = math.exp(float(np.mean(np.log(spec_pos)))) / (float(np.mean(spec_pos)) + 1e-12)
        total_w = float(np.sum(weights)) + 1e-12
        raw_max_w = float(np.max(weights)) if len(weights) else 0.0
        if flatness > 0.50 and (raw_max_w / total_w) < 0.16:
            return AudioAnalysisResult(
                pitches=(),
                rms_db=rms_db,
                is_onset=False,
                is_silent=True,
                dominant_cents=0.0,
            )

        # Apply subtle visual prior boost when fingers isolate specific notes
        if visual_pitch_priors:
            for idx, m in enumerate(self.candidate_midis):
                prior = visual_pitch_priors.get(int(m), 0.0)
                if prior > 0.0:
                    weights[idx] *= 1.0 + 0.35 * min(1.0, prior)

        # Fundamental & harmonic verification to prevent ghost octaves/fifths and mainlobe bleed
        max_w = float(np.max(weights)) if len(weights) else 0.0
        if max_w < 1e-4:
            return AudioAnalysisResult(
                pitches=(),
                rms_db=rms_db,
                is_onset=is_onset,
                is_silent=True,
                dominant_cents=0.0,
            )

        spec_max = float(np.max(spec)) + 1e-9
        candidates: list[tuple[int, float]] = []
        for idx, w_val in enumerate(weights):
            if w_val < max_w * 0.24:
                continue
            midi = int(self.candidate_midis[idx])
            has_vis = bool(visual_pitch_priors and visual_pitch_priors.get(midi, 0.0) > 0.35)

            # Suppress mainlobe shoulder bleed from adjacent semitones unless locally peaked
            left_w = float(weights[idx - 1]) if idx > 0 else 0.0
            right_w = float(weights[idx + 1]) if idx + 1 < len(weights) else 0.0
            adj_max = max(left_w, right_w)
            f0_bin = int(round((midi - self.grid_midi_min) * self.bins_per_semitone))
            f0_energy = float(np.max(spec[max(0, f0_bin - 1) : min(self.n_log_bins, f0_bin + 2)]))
            f0_left = float(spec[max(0, f0_bin - self.bins_per_semitone)])
            f0_right = float(spec[min(self.n_log_bins - 1, f0_bin + self.bins_per_semitone)])
            is_f0_peak = f0_energy >= max(f0_left, f0_right) * 0.96

            if adj_max > w_val * 1.12 and not is_f0_peak and not has_vis:
                continue

            h2_bin = int(round((midi + 12.0 - self.grid_midi_min) * self.bins_per_semitone))
            h2_energy = (
                float(np.max(spec[max(0, h2_bin - 1) : min(self.n_log_bins, h2_bin + 2)]))
                if 0 <= h2_bin < self.n_log_bins
                else 0.0
            )
            h2_left = float(spec[max(0, h2_bin - self.bins_per_semitone)]) if 0 <= h2_bin < self.n_log_bins else 0.0
            h2_right = float(spec[min(self.n_log_bins - 1, h2_bin + self.bins_per_semitone)]) if 0 <= h2_bin < self.n_log_bins else 0.0
            is_h2_peak = h2_energy > max(h2_left, h2_right) * 1.02

            h3_bin = int(round((midi + 19.01955 - self.grid_midi_min) * self.bins_per_semitone))
            h3_energy = (
                float(np.max(spec[max(0, h3_bin - 1) : min(self.n_log_bins, h3_bin + 2)]))
                if 0 <= h3_bin < self.n_log_bins
                else 0.0
            )
            min_f0_req = spec_max * (0.14 if midi < 45 else 0.16)
            has_low_harmonic_pair = (
                midi < 45
                and f0_energy >= spec_max * 0.06
                and is_h2_peak
                and h2_energy > spec_max * 0.22
                and h3_energy > spec_max * 0.20
            )
            if (f0_energy < min_f0_req or (f0_energy < spec_max * 0.22 and not is_f0_peak)) and not has_low_harmonic_pair:
                continue
            candidates.append((midi, float(w_val)))

        # Sort strongest first and prune octave-above harmonic shadows unless strongly supported
        candidates.sort(key=lambda item: item[1], reverse=True)
        filtered: list[tuple[int, float]] = []
        chosen_midis: dict[int, float] = {}
        for midi, w_val in candidates:
            has_vis = bool(visual_pitch_priors and visual_pitch_priors.get(midi, 0.0) > 0.35)
            # Suppress adjacent semitone shoulder that isn't visually present
            if not has_vis and (
                chosen_midis.get(midi - 1, 0.0) > w_val * 1.25
                or chosen_midis.get(midi + 1, 0.0) > w_val * 1.25
            ):
                continue
            # If an octave below (midi - 12) is much stronger, require visual prior or strong weight
            parent_w = chosen_midis.get(midi - 12, 0.0)
            if parent_w > 0.0 and w_val < parent_w * 0.36 and not has_vis:
                continue
            # Suppress twelfth above (midi - 19 = 3rd harmonic) unless very strong or visually supported
            fifth_parent_w = chosen_midis.get(midi - 19, 0.0)
            if fifth_parent_w > 0.0 and w_val < fifth_parent_w * 0.36 and not has_vis:
                continue
            filtered.append((midi, w_val))
            chosen_midis[midi] = w_val
            if len(filtered) >= 6:
                break

        # Monophonic fast path via YIN (never collapse genuine 2-note power chords / double-stops)
        is_effectively_mono = len(filtered) <= 1 or (
            len(filtered) == 2 and filtered[1][1] < filtered[0][1] * 0.35
        )
        yin_freq, yin_conf = yin_pitch(x[-3072:] if len(x) >= 3072 else x, sample_rate)
        if yin_freq is not None and yin_conf >= 0.78 and is_effectively_mono:
            y_midi, y_cents = freq_to_midi_and_cents(yin_freq)
            if self.midi_min <= y_midi <= self.midi_max:
                # Confirm YIN agrees with strongest NNLS note (or is within a bend of +-2 semitones)
                if not filtered or abs(y_midi - filtered[0][0]) <= 2:
                    pitch = DetectedPitch(
                        midi=y_midi,
                        freq_hz=yin_freq,
                        cents_offset=y_cents,
                        confidence=min(0.99, yin_conf),
                        is_monophonic=True,
                    )
                    return AudioAnalysisResult(
                        pitches=(pitch,),
                        rms_db=rms_db,
                        is_onset=is_onset,
                        is_silent=False,
                        dominant_cents=y_cents,
                        spectrum_bands=spec_bands,
                    )

        # Polyphonic output from filtered NNLS weights
        norm_denom = max(1e-6, max_w)
        out_pitches: list[DetectedPitch] = []
        for midi, w_val in filtered:
            conf = float(min(0.99, max(0.15, w_val / norm_denom)))
            out_pitches.append(
                DetectedPitch(
                    midi=midi,
                    freq_hz=midi_to_freq(float(midi)),
                    cents_offset=0.0,
                    confidence=conf,
                    is_monophonic=len(filtered) == 1,
                )
            )

        out_pitches.sort(key=lambda p: p.midi)
        dom_cents = 0.0
        if yin_freq is not None and yin_conf >= 0.65 and out_pitches:
            _, c = freq_to_midi_and_cents(yin_freq)
            dom_cents = c

        return AudioAnalysisResult(
            pitches=tuple(out_pitches),
            rms_db=rms_db,
            is_onset=is_onset,
            is_silent=len(out_pitches) == 0,
            dominant_cents=dom_cents,
            spectrum_bands=spec_bands,
        )

    @staticmethod
    def _downsample_spectrum_bands(spec: np.ndarray, n_bands: int = 24) -> tuple[float, ...]:
        """Pool log-frequency spectrum into n_bands normalized [0..1] UI bars."""
        usable = spec[: min(len(spec), 180)]  # span guitar fundamentals & primary harmonics
        if len(usable) < n_bands:
            return tuple([0.0] * n_bands)
        chunks = np.array_split(usable, n_bands)
        raw = np.array([float(np.max(c)) for c in chunks], dtype=np.float64)
        compressed = np.log1p(18.0 * raw) / math.log1p(18.0 * 0.45)
        clipped = np.clip(compressed, 0.0, 1.0)
        return tuple(round(float(v), 3) for v in clipped)
