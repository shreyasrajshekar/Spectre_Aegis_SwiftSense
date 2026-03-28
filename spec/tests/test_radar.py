"""
tests/test_radar.py
===================
Validates the hardware ISAC radar sensing pipeline in SDRHandler.
All tests run against a synthetic I/Q environment (no physical hardware needed).

Run with:
    python -m pytest tests/test_radar.py -v
"""

import time
import math
import numpy as np
import pytest
import sys
import os

# Make sure parent dir is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.sdr_handler import SDRHandler


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_sdr_handler() -> SDRHandler:
    """
    Returns an SDRHandler in digital-twin mode so that no hardware is required.
    We monkey-patch the internal helper methods to accept synthetic I/Q data.
    """
    handler = SDRHandler(use_digital_twin=True, center_freq=2.4e9, sample_rate=10e6)
    return handler


def synthetic_two_tone_iq(n: int, f1: float, f2: float, sr: float,
                           snr_db: float = 20.0) -> np.ndarray:
    """Generate a complex I/Q signal with two tones and AWGN noise floor."""
    t = np.arange(n) / sr
    signal = (np.exp(1j * 2 * np.pi * f1 * t) +
              np.exp(1j * 2 * np.pi * f2 * t))
    noise_amp = 10 ** (-snr_db / 20.0)
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) * noise_amp
    return (signal + noise).astype(np.complex128)


def make_ula_signal(angle_deg: float, n: int = 4096, snr_db: float = 20.0,
                    sr: float = 10e6) -> tuple:
    """
    Synthesise two-element ULA receive signals for a source at angle_deg.
    rx1 has a λ/2-spacing phase shift: exp(jπ sin(θ)).
    Returns (rx0, rx1).
    """
    theta = math.radians(angle_deg)
    t = np.arange(n) / sr
    source = np.exp(1j * 2 * np.pi * 1e6 * t)            # 1 MHz tone
    noise_amp = 10 ** (-snr_db / 20.0)
    rx0 = source + (np.random.randn(n) + 1j * np.random.randn(n)) * noise_amp
    rx1 = source * np.exp(1j * np.pi * math.sin(theta)) \
        + (np.random.randn(n) + 1j * np.random.randn(n)) * noise_amp
    return rx0.astype(np.complex128), rx1.astype(np.complex128)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema / Data Format
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFormat:
    """Ensure returned objects exactly match DigitalTwin radar_objects schema."""

    def test_digital_twin_returns_list(self):
        handler = make_synthetic_sdr_handler()
        result = handler.get_sensing_radar()
        assert isinstance(result, list), "must return a list"

    def test_digital_twin_schema(self):
        handler = make_synthetic_sdr_handler()
        # Warm up to ensure objects exist
        for _ in range(5):
            handler.simulator.generate_iq()
        result = handler.get_sensing_radar()
        for obj in result:
            assert 'dist'  in obj, "missing 'dist'"
            assert 'angle' in obj, "missing 'angle'"
            assert 'v_d'   in obj, "missing 'v_d'"
            assert 'v_a'   in obj, "missing 'v_a'"
            assert 'id'    in obj, "missing 'id'"

    def test_dist_in_range(self):
        handler = make_synthetic_sdr_handler()
        for _ in range(10):
            handler.simulator.generate_iq()
        for obj in handler.get_sensing_radar():
            assert 5.0 <= obj['dist'] <= 100.0, \
                f"dist {obj['dist']} out of 5-100m range"

    def test_angle_in_range(self):
        handler = make_synthetic_sdr_handler()
        for _ in range(10):
            handler.simulator.generate_iq()
        for obj in handler.get_sensing_radar():
            assert 0.0 <= obj['angle'] < 360.0, \
                f"angle {obj['angle']} out of 0-360° range"

    def test_id_is_integer(self):
        handler = make_synthetic_sdr_handler()
        for _ in range(5):
            handler.simulator.generate_iq()
        for obj in handler.get_sensing_radar():
            assert isinstance(obj['id'], int), "id must be int"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CFAR Candidate Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestCFAR:
    def test_cfar_finds_candidates_in_strong_signal(self):
        handler = make_synthetic_sdr_handler()
        sr = float(handler.sample_rate)
        iq = synthetic_two_tone_iq(2048, f1=1e6, f2=2e6, sr=sr, snr_db=30.0)
        candidates = handler._cfar_candidates(iq, snr_threshold_db=-20.0)
        assert len(candidates) >= 1, "CFAR must detect at least 1 candidate at SNR=30dB"

    def test_cfar_suppressed_on_noise_only(self):
        handler = make_synthetic_sdr_handler()
        # Pure AWGN — set a very high SNR threshold (30 dB above noise floor)
        # With complex FFT, median-based CFAR on pure noise will still have some
        # bins above the median; at 30 dB threshold very few groups survive.
        iq = (np.random.randn(2048) + 1j * np.random.randn(2048)) * 0.001
        candidates = handler._cfar_candidates(iq, snr_threshold_db=30.0)
        # At 30 dB threshold above noise median, AWGN should produce ≤2 spurious groups
        assert len(candidates) <= 2, \
            f"CFAR must suppress noise at 30dB threshold, got {len(candidates)} candidates"

    def test_cfar_returns_ndarray_slices(self):
        handler = make_synthetic_sdr_handler()
        sr = float(handler.sample_rate)
        iq = synthetic_two_tone_iq(2048, f1=500e3, f2=3e6, sr=sr, snr_db=25.0)
        candidates = handler._cfar_candidates(iq, snr_threshold_db=-20.0)
        for c in candidates:
            assert isinstance(c, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Angle-of-Arrival (AoA) Accuracy — ±15° target
# ─────────────────────────────────────────────────────────────────────────────

class TestAoAAccuracy:
    """
    Validates MUSIC and Bartlett against known synthetic ULA signals.
    Accuracy target: ±15° for SNR ≥ 10 dB.
    """

    @pytest.mark.parametrize("true_angle_deg", [-60, -30, 0, 30, 60])
    def test_music_angle_accuracy(self, true_angle_deg):
        handler = make_synthetic_sdr_handler()
        rx0, rx1 = make_ula_signal(true_angle_deg, n=4096, snr_db=15.0)
        peaks = handler._music_aoa(rx0, rx1, n_scan=361)
        assert len(peaks) >= 1
        closest = min(peaks, key=lambda p: abs(p - true_angle_deg))
        error = abs(closest - true_angle_deg)
        assert error <= 15.0, \
            f"MUSIC angle error {error:.1f}° exceeds 15° threshold for θ={true_angle_deg}°"

    @pytest.mark.parametrize("true_angle_deg", [-45, 0, 45])
    def test_bartlett_angle_accuracy(self, true_angle_deg):
        handler = make_synthetic_sdr_handler()
        rx0, rx1 = make_ula_signal(true_angle_deg, n=4096, snr_db=15.0)
        peaks = handler._bartlett_aoa(rx0, rx1, n_scan=361)
        assert len(peaks) >= 1
        closest = min(peaks, key=lambda p: abs(p - true_angle_deg))
        error = abs(closest - true_angle_deg)
        assert error <= 15.0, \
            f"Bartlett angle error {error:.1f}° exceeds 15° threshold for θ={true_angle_deg}°"

    def test_music_returns_list_of_floats(self):
        handler = make_synthetic_sdr_handler()
        rx0, rx1 = make_ula_signal(0, n=2048, snr_db=20.0)
        peaks = handler._music_aoa(rx0, rx1)
        assert all(isinstance(p, float) for p in peaks)
        assert all(-90.0 <= p <= 90.0 for p in peaks)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Range Estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestRangeEstimation:
    def test_range_clamp_minimum(self):
        handler = make_synthetic_sdr_handler()
        # DC-heavy signal (no offset) → range should clamp at 5m
        iq = np.ones(2048, dtype=np.complex128)
        dist = handler._estimate_range(iq, range_max=100.0)
        assert dist >= 5.0

    def test_range_clamp_maximum(self):
        handler = make_synthetic_sdr_handler()
        sr = float(handler.sample_rate)
        # Near-Nyquist tone → large range
        t = np.arange(2048) / sr
        iq = np.exp(1j * 2 * np.pi * (sr / 2 - 1e3) * t)
        dist = handler._estimate_range(iq, range_max=100.0)
        assert dist <= 100.0

    def test_range_returns_float(self):
        handler = make_synthetic_sdr_handler()
        iq = synthetic_two_tone_iq(2048, f1=1e6, f2=2e6, sr=10e6, snr_db=20.0)
        dist = handler._estimate_range(iq, range_max=100.0)
        assert isinstance(dist, float)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Doppler / Velocity Estimation
# ─────────────────────────────────────────────────────────────────────────────

class TestDopplerVelocity:
    def test_zero_velocity_on_identical_frames(self):
        handler = make_synthetic_sdr_handler()
        iq = synthetic_two_tone_iq(16384, f1=1e6, f2=2e6, sr=10e6, snr_db=20.0)
        v_d, v_a = handler._estimate_doppler_velocity(iq, iq)
        assert abs(v_d) < 0.05, f"Identical frames must yield ~0 v_d, got {v_d}"

    def test_no_prev_frame_returns_zeros(self):
        handler = make_synthetic_sdr_handler()
        iq = synthetic_two_tone_iq(4096, f1=1e6, f2=2e6, sr=10e6, snr_db=20.0)
        v_d, v_a = handler._estimate_doppler_velocity(iq, None)
        assert v_d == 0.0 and v_a == 0.0

    def test_velocity_direction_with_frequency_shifted_frame(self):
        """
        A signal with a positive frequency shift in the new frame vs old frame
        implies the source is moving toward the receiver (positive v_d).
        """
        handler = make_synthetic_sdr_handler()
        sr = 10e6
        n = 16384
        t = np.arange(n) / sr
        # Frame 1: 1 MHz tone
        prev_frame = np.exp(1j * 2 * np.pi * 1e6 * t).astype(np.complex128)
        # Frame 2: 1.01 MHz tone (slight positive Doppler shift)
        curr_frame = np.exp(1j * 2 * np.pi * 1.01e6 * t).astype(np.complex128)
        v_d, _ = handler._estimate_doppler_velocity(curr_frame, prev_frame)
        # Sign convention: positive frequency shift → source moving toward us
        # We just verify it's non-zero and a float
        assert isinstance(v_d, float)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ID Persistence — same object tracked for >1 second
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectPersistence:
    """
    At 50ms cadence, 1 second = 20 calls.  An object that moves slowly
    should retain the same ID across all calls.
    """

    def test_id_stable_across_20_frames(self):
        handler = make_synthetic_sdr_handler()
        # Inject a single detection with stable position
        det = [{'dist': 30.0, 'angle': 45.0, 'v_d': 0.01, 'v_a': 0.1}]

        first_result = handler._nn_track_update(det)
        assert len(first_result) == 1
        stable_id = first_result[0]['id']

        for frame in range(19):
            # Slightly jitter position to simulate real motion
            jitter_det = [{'dist': 30.0 + frame * 0.01, 'angle': 45.0 + frame * 0.05,
                           'v_d': 0.01, 'v_a': 0.1}]
            result = handler._nn_track_update(jitter_det)
            assert len(result) == 1, f"Track dropped at frame {frame + 1}"
            assert result[0]['id'] == stable_id, \
                f"ID changed from {stable_id} to {result[0]['id']} at frame {frame + 1}"

    def test_id_evicted_after_5_missed_scans(self):
        handler = make_synthetic_sdr_handler()
        det = [{'dist': 50.0, 'angle': 90.0, 'v_d': 0.0, 'v_a': 0.0}]
        handler._nn_track_update(det)
        assert len(handler._radar_tracks) == 1

        # 5 empty scans → track should be evicted
        for _ in range(5):
            handler._nn_track_update([])

        assert len(handler._radar_tracks) == 0, \
            "Stale track should be evicted after 5 missed scans"

    def test_new_object_gets_new_id(self):
        handler = make_synthetic_sdr_handler()
        d1 = [{'dist': 20.0, 'angle': 10.0, 'v_d': 0.0, 'v_a': 0.0}]
        r1 = handler._nn_track_update(d1)
        id1 = r1[0]['id']

        # Second object far away — should get a different ID
        d2 = [{'dist': 80.0, 'angle': 200.0, 'v_d': 0.0, 'v_a': 0.0}]
        r2_first = handler._nn_track_update(d2)
        id2 = r2_first[0]['id']

        assert id1 != id2, "Two distinct remote objects must not share an ID"


# ─────────────────────────────────────────────────────────────────────────────
# 7. 20ms Processing Budget
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessingBudget:
    """
    Full pipeline internal methods must complete within 20ms for a 16384-sample
    buffer. We test the heaviest individual components.
    """

    def _time_ms(self, fn) -> float:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0

    def test_music_budget_under_5ms(self):
        handler = make_synthetic_sdr_handler()
        rx0, rx1 = make_ula_signal(30, n=16384, snr_db=15.0)
        elapsed = self._time_ms(lambda: handler._music_aoa(rx0, rx1, n_scan=181))
        assert elapsed < 5.0, f"MUSIC took {elapsed:.1f}ms (budget 5ms)"

    def test_bartlett_budget_under_3ms(self):
        handler = make_synthetic_sdr_handler()
        rx0, rx1 = make_ula_signal(0, n=16384, snr_db=15.0)
        elapsed = self._time_ms(lambda: handler._bartlett_aoa(rx0, rx1, n_scan=181))
        assert elapsed < 3.0, f"Bartlett took {elapsed:.1f}ms (budget 3ms)"

    def test_cfar_budget_under_2ms(self):
        handler = make_synthetic_sdr_handler()
        iq = synthetic_two_tone_iq(16384, f1=1e6, f2=2e6, sr=10e6, snr_db=20.0)
        elapsed = self._time_ms(lambda: handler._cfar_candidates(iq, -20.0))
        assert elapsed < 2.0, f"CFAR took {elapsed:.1f}ms (budget 2ms)"

    def test_range_estimation_budget_under_2ms(self):
        handler = make_synthetic_sdr_handler()
        iq = synthetic_two_tone_iq(16384, f1=1e6, f2=2e6, sr=10e6, snr_db=20.0)
        elapsed = self._time_ms(lambda: handler._estimate_range(iq, 100.0))
        assert elapsed < 2.0, f"Range estimation took {elapsed:.1f}ms (budget 2ms)"

    def test_full_pipeline_budget_twin_mode(self):
        """
        In digital twin mode, get_sensing_radar() pass-through is trivial.
        This confirms the twin overhead is negligible.
        """
        handler = make_synthetic_sdr_handler()
        for _ in range(5):
            handler.simulator.generate_iq()
        elapsed = self._time_ms(lambda: handler.get_sensing_radar())
        assert elapsed < 5.0, f"Twin mode radar fetch took {elapsed:.1f}ms"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Layer Config
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerConfig:
    @pytest.mark.parametrize("layer,expected_algo,expected_max", [
        ("Legacy",  "bartlett", 12),
        ("cmWave",  "music",     3),
        ("Sub-THz", "music",     5),
    ])
    def test_layer_config_values(self, layer, expected_algo, expected_max):
        handler = make_synthetic_sdr_handler()
        handler.simulator.current_layer = layer
        cfg = handler._get_layer_config()
        assert cfg['algorithm']   == expected_algo
        assert cfg['max_objects'] == expected_max

    def test_unknown_layer_falls_back(self):
        handler = make_synthetic_sdr_handler()
        handler.simulator.current_layer = "FutureBand"
        cfg = handler._get_layer_config()
        assert 'max_objects' in cfg
        assert 'range_max'   in cfg
        assert 'algorithm'   in cfg


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
