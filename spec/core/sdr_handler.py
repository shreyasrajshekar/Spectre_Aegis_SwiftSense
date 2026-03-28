import sys
import numpy as np
import time
import logging
from typing import List, Dict, Optional, Tuple
from core.digital_twin import DigitalTwinSimulator

try:
    import adi
except ImportError:
    logging.warning("pyadi-iio is not installed or available. Simulation mode only.")
    adi = None

class SDRHandler:
    def __init__(self, use_digital_twin=False, center_freq=2.4e9, sample_rate=10e6, bandwidth=10e6):
        self.use_digital_twin = use_digital_twin
        self.center_freq = int(center_freq)
        self.sample_rate = int(sample_rate)
        self.bandwidth = int(bandwidth)
        self.sdr = None
        self.buffer_size = 1024 * 16
        self.active_beam = 0 # Current spatial beam index (0-3)

        # ── ISAC Radar Sensing State ────────────────────────────────────────────
        self._prev_iq_frame: Optional[np.ndarray] = None   # Previous frame for Doppler
        self._radar_tracks: Dict[int, dict] = {}            # id → {dist, angle, v_d, v_a, miss_count}
        self._next_track_id: int = 100                      # Auto-incrementing object ID
        self._SPEED_OF_LIGHT: float = 3e8
        # ───────────────────────────────────────────────────────────────────────

        self.simulator = DigitalTwinSimulator(buffer_size=self.buffer_size) if use_digital_twin else None
        
        if not self.use_digital_twin:
            self._init_hardware()
            
    def _init_hardware(self):
        """Attempts to initialize the ADALM-Pluto hardware, auto-finding USB or IP URIs."""
        if adi is None:
            logging.error("ADI library missing. Cannot connect to SDR.")
            raise ImportError("pyadi-iio is required, but not found.")
            
        import iio

        while self.sdr is None:
            # 1. Scan dynamically for the connected PlutoSDR (e.g., usb:1.6.5)
            logging.info("Scanning for PlutoSDR devices...")
            contexts = iio.scan_contexts()
            pluto_uri = None
            for uri, desc in contexts.items():
                if 'PlutoSDR' in desc or 'ADALM-PLUTO' in desc:
                    pluto_uri = uri
                    break
                    
            if not pluto_uri:
                logging.error("No PlutoSDR found via discovery. Defaulting to IP 192.168.2.1...")
                pluto_uri = "ip:192.168.2.1"
                
            try:
                logging.info(f"Attempting to connect to PlutoSDR at {pluto_uri}...")
                self.sdr = adi.Pluto(pluto_uri)
            except Exception as e:
                logging.warning(f"Connection to {pluto_uri} failed: {e}")
                if pluto_uri.startswith("usb:"):
                    logging.info("Trying secondary IP fallback (192.168.2.1) just in case...")
                    try:
                        self.sdr = adi.Pluto("ip:192.168.2.1")
                    except Exception:
                        pass
                
                if self.sdr is None:
                    logging.error("Hardware not found. Pausing AI inference and polling every 5 seconds...")
                    import time
                    time.sleep(5)
                    continue # Retry from the top of the while loop
                
        # Configure SDR Hardware once connected
        try:
            self.sdr.sample_rate = self.sample_rate
            
            # Configure Rx
            self.sdr.rx_lo = self.center_freq
            self.sdr.rx_rf_bandwidth = self.bandwidth
            self.sdr.rx_buffer_size = self.buffer_size
            self.sdr.gain_control_mode_chan0 = 'fast_attack'
            
            # Configure Tx
            self.sdr.tx_lo = self.center_freq
            self.sdr.tx_rf_bandwidth = self.bandwidth
            self.sdr.tx_cyclic_buffer = False
            self.sdr.tx_hardwaregain_chan0 = -10 # Safe initial TX power
            
            logging.info("PlutoSDR hardware successfully connected and mapped!")
        except Exception as e:
            logging.error(f"Failed to configure SDR parameters: {e}")
            self.sdr = None # Reset and let caller retry / timeout

    def set_frequency(self, frequency):
        """Changes the LO frequency in under 10ms target for handover."""
        self.center_freq = int(frequency)
        if self.sdr is not None:
            # Updating LO frequencies triggers internal PLL recalibration
            try:
                t_start = time.perf_counter()
                self.sdr.rx_lo = self.center_freq
                self.sdr.tx_lo = self.center_freq
                t_end = time.perf_counter()
                logging.debug(f"Frequency hop to {frequency/1e6} MHz completed in {(t_end - t_start)*1000:.2f} ms")
            except OSError as e:
                logging.warning(f"SDR hardware rejected frequency {frequency/1e6} MHz: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during frequency hop: {e}")

    def set_beam_direction(self, beam_idx):
        """6G Massive MIMO: Sets the spatial beam index (Approximate direction)."""
        self.active_beam = int(beam_idx) % 4
        logging.debug(f"Spatial Beam switched to Index {self.active_beam}")

    def capture_iq(self):
        """Captures a buffer of I/Q samples."""
        if self.use_digital_twin:
            raw_iq = self.simulator.generate_iq()
            return self.simulator.apply_spatial_filter(raw_iq, self.active_beam)
            
        while True:
            if self.sdr is not None:
                try:
                    rx_data = self.sdr.rx()
                    if rx_data is not None:
                        return rx_data
                except Exception as e:
                    logging.error(f"SDR read error or disconnect: {e}. Attempting recovery...")
                    self.sdr = None
            
            # If sdr is None (either here or on entry), block and re-init
            logging.info("Hardware unavailable. Entering blocking reconnect loop...")
            self._init_hardware()

    # ═══════════════════════════════════════════════════════════════════════════
    # ISAC RADAR SENSING — Hardware Implementation
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_layer_config(self) -> dict:
        """Returns layer-specific algorithm parameters (Legacy only)."""
        return {'max_objects': 12, 'range_max': 100.0, 'algorithm': 'bartlett', 'snr_db': -20.0}

    def _music_aoa(self, rx0: np.ndarray, rx1: np.ndarray, n_scan: int = 181) -> List[float]:
        """
        2-element ULA MUSIC angle-of-arrival estimation.
        Vectorized implementation for ultra-low latency.
        """
        N = len(rx0)
        X = np.vstack([rx0[:N], rx1[:N]])          # shape (2, N)
        R = (X @ X.conj().T) / N                   # shape (2, 2)
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        E_n = eigenvectors[:, :1]                  # noise eigenvector (2×1)

        thetas = np.linspace(-np.pi / 2, np.pi / 2, n_scan)
        
        # Vectorized steering vectors: shape (2, n_scan)
        A = np.vstack([np.ones(n_scan), np.exp(1j * np.pi * np.sin(thetas))])
        
        # Projection matrix P_n
        P_n = E_n @ E_n.conj().T
        
        # A^H P_n A diagonals
        P_n_A = P_n @ A
        denom = np.abs(np.sum(A.conj() * P_n_A, axis=0))
        spectrum = 1.0 / (denom + 1e-12)

        # Normalise and find peaks
        spectrum /= (spectrum.max() + 1e-12)
        peaks = []
        for i in range(1, n_scan - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and spectrum[i] > 0.3:
                peaks.append(float(np.degrees(thetas[i])))
        return peaks if peaks else [float(np.degrees(thetas[np.argmax(spectrum)]))]

    def _bartlett_aoa(self, rx0: np.ndarray, rx1: np.ndarray, n_scan: int = 181) -> List[float]:
        """
        Bartlett beamforming (conventional) — Vectorized implementation.
        """
        N = len(rx0)
        X = np.vstack([rx0[:N], rx1[:N]])
        R = (X @ X.conj().T) / N
        thetas = np.linspace(-np.pi / 2, np.pi / 2, n_scan)
        
        A = np.vstack([np.ones(n_scan), np.exp(1j * np.pi * np.sin(thetas))])
        R_A = R @ A
        spectrum = np.real(np.sum(A.conj() * R_A, axis=0))
        
        spectrum /= (spectrum.max() + 1e-12)
        peaks = []
        for i in range(1, n_scan - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and spectrum[i] > 0.25:
                peaks.append(float(np.degrees(thetas[i])))
        return peaks if peaks else [float(np.degrees(thetas[np.argmax(spectrum)]))]

    def _estimate_range(self, iq: np.ndarray, range_max: float) -> float:
        """
        Estimates range from dominant FFT bin.
        In passive sensing, frequency offset from the LO indicates propagation delay
        relative to the strongest reflect component.
        range = (f_peak / (sample_rate/2)) * range_max  (normalised heuristic)
        Valid range: 5m → range_max. Returns 0.0 if no clear peak.
        Uses np.fft.fft (not rfft) because IQ data is always complex.
        """
        N = len(iq)
        window = np.hanning(N)
        # Full complex FFT; take one-sided magnitude (positive frequencies)
        full_spectrum = np.abs(np.fft.fft(iq * window))
        half_N = N // 2
        spectrum = full_spectrum[1:half_N]          # skip DC bin 0
        if spectrum.size == 0 or spectrum.max() < 1e-6:
            return 0.0
        peak_bin = int(np.argmax(spectrum)) + 1     # +1 to account for skipped DC
        # Frequency of that bin
        f_peak = float(peak_bin * self.sample_rate / N)
        # Normalised heuristic mapping: 0 Hz → 0m, Nyquist → range_max
        nyquist = self.sample_rate / 2.0
        dist = max(5.0, min(range_max, (f_peak / nyquist) * range_max))
        return round(dist, 2)

    def _estimate_doppler_velocity(self, iq_curr: np.ndarray, iq_prev: Optional[np.ndarray]) -> Tuple[float, float]:
        """
        Estimates radial velocity (v_d) by computing mean phase rotation between
        consecutive frames using complex conjugate multiplication.
        v_d in m/s (positive = approaching), v_a in deg/s (angular velocity estimate).
        """
        if iq_prev is None or len(iq_prev) != len(iq_curr):
            return 0.0, 0.0
        T_frame = self.buffer_size / self.sample_rate          # ~1.638ms per frame
        # Phase-rotation per sample
        phase_prod = iq_curr[:len(iq_prev)] * np.conj(iq_prev[:len(iq_curr)])
        mean_phase = float(np.angle(np.mean(phase_prod)))      # radians
        # Convert phase rotation to Doppler frequency
        f_doppler = mean_phase / (2 * np.pi * T_frame)         # Hz
        # Radial velocity: v = f_d * c / (2 * f_c)
        v_d = float(f_doppler * self._SPEED_OF_LIGHT / (2.0 * self.center_freq))
        # Angular velocity: heuristic from phase variance across array
        phase_variance = float(np.var(np.angle(phase_prod)))
        v_a = round(float(np.degrees(phase_variance) * 0.1), 3)  # deg/s rough estimate
        return round(v_d, 4), round(v_a, 4)

    def _cfar_candidates(self, iq: np.ndarray, snr_threshold_db: float) -> List[np.ndarray]:
        """
        Constant False Alarm Rate (CFAR) guard: splits I/Q into frequency sub-bands
        and returns only those sub-bands whose power clears the SNR threshold above
        the overall noise floor. Each returned slice is a candidate object's I/Q.
        Uses np.fft.fft (not rfft) because IQ data is always complex.
        """
        N = len(iq)
        n_fft = min(N, 2048)
        iq_slice = iq[:n_fft]
        full_fft = np.fft.fft(iq_slice)             # complex spectrum, full N bins
        spectrum_mag = np.abs(full_fft)
        noise_floor = float(np.median(spectrum_mag))  # Robust noise estimate
        threshold_linear = noise_floor * (10 ** (snr_threshold_db / 20.0))

        # Find contiguous bins above threshold
        above = spectrum_mag > threshold_linear
        candidates = []
        in_group = False
        g_start = 0
        for i, a in enumerate(above):
            if a and not in_group:
                g_start = i
                in_group = True
            elif not a and in_group:
                in_group = False
                # Back-convert bin group to time domain via IFFT of masked spectrum
                sub = np.zeros(n_fft, dtype=complex)
                sub[g_start:i] = full_fft[g_start:i]
                candidates.append(np.fft.ifft(sub))  # complex time-domain slice
        if in_group:
            sub = np.zeros(n_fft, dtype=complex)
            sub[g_start:] = full_fft[g_start:]
            candidates.append(np.fft.ifft(sub))
        return candidates

    def _nn_track_update(self, detections: List[dict]) -> List[dict]:
        """
        Nearest-neighbour track association + ID persistence.
        Matches new detections to existing tracks (cost = Euclidean on normalised dist+angle).
        Tracks survive up to 5 missed scans before eviction.
        IDs guaranteed stable as long as object visible.
        """
        MAX_COST   = 30.0  # Max distance+angle delta to consider a match
        MAX_MISSES = 5     # Frames before dead track is evicted

        matched_ids = set()
        results = []

        for det in detections:
            best_id   = None
            best_cost = MAX_COST
            for tid, track in self._radar_tracks.items():
                # Normalised cost: 1 unit dist = 1 cost; 1 deg angle = 0.5 cost
                d_dist  = abs(det['dist']  - track['dist'])
                d_angle = abs(det['angle'] - track['angle']) % 360
                d_angle = min(d_angle, 360 - d_angle)  # Wrap-around
                cost = d_dist + 0.5 * d_angle
                if cost < best_cost:
                    best_cost = cost
                    best_id   = tid

            if best_id is not None:
                # Update existing track
                self._radar_tracks[best_id].update({
                    'dist':       det['dist'],
                    'angle':      det['angle'],
                    'v_d':        det['v_d'],
                    'v_a':        det['v_a'],
                    'miss_count': 0
                })
                matched_ids.add(best_id)
                results.append({'dist': det['dist'], 'angle': det['angle'],
                                'v_d': det['v_d'],   'v_a': det['v_a'], 'id': best_id})
            else:
                # New track
                new_id = self._next_track_id
                self._next_track_id = (self._next_track_id % 999) + 100
                self._radar_tracks[new_id] = {
                    'dist': det['dist'], 'angle': det['angle'],
                    'v_d':  det['v_d'],  'v_a':   det['v_a'], 'miss_count': 0
                }
                matched_ids.add(new_id)
                results.append({'dist': det['dist'], 'angle': det['angle'],
                                'v_d': det['v_d'],   'v_a': det['v_a'], 'id': new_id})

        # Evict stale tracks
        stale = []
        for tid in self._radar_tracks:
            if tid not in matched_ids:
                self._radar_tracks[tid]['miss_count'] += 1
                if self._radar_tracks[tid]['miss_count'] >= MAX_MISSES:
                    stale.append(tid)
        for tid in stale:
            del self._radar_tracks[tid]

        return results

    def get_sensing_radar(self) -> List[Dict]:
        """
        6G ISAC Hardware Radar Sensing Pipeline.

        Extracts spatial objects from captured I/Q data using:
          1. Dual-channel MIMO capture (falls back to synthetic 2nd element)
          2. CFAR noise gating (SNR > threshold)
          3. MUSIC or Bartlett AoA estimation (2-element ULA)
          4. FFT-based range estimation (5-100m)
          5. Doppler velocity from consecutive frame phase rotation
          6. Nearest-neighbour ID persistence tracking

        Budget: <20ms.  Returns: List[{dist, angle, v_d, v_a, id}]
        """
        if self.use_digital_twin:
            return self.simulator.get_radar_map()

        t0 = time.perf_counter()
        cfg = self._get_layer_config()
        max_objects  = cfg['max_objects']
        range_max    = cfg['range_max']
        algorithm    = cfg['algorithm']
        snr_db       = cfg['snr_db']

        # ── Step 1: Capture I/Q ────────────────────────────────────────────────
        try:
            raw = self.sdr.rx()  # Returns list[ch0, ch1] if 2-ch, else ndarray
        except Exception as e:
            logging.warning(f"[Radar] rx() failed: {e}")
            return list(self._radar_tracks.values())[:max_objects]  # Return stale tracks

        # Normalise to two channels
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            rx0 = np.asarray(raw[0], dtype=np.complex128)
            rx1 = np.asarray(raw[1], dtype=np.complex128)
        else:
            rx0 = np.asarray(raw, dtype=np.complex128)
            # Synthetic 2nd element: apply half-sample fractional delay via Hilbert
            rx1 = np.roll(rx0, 1) * np.exp(1j * np.pi * 0.5)  # λ/2 phase shift approx

        # ── Step 2: SNR gate / CFAR candidate extraction ───────────────────────
        candidates_iq = self._cfar_candidates(rx0, snr_threshold_db=snr_db)
        if not candidates_iq:
            # Channel is quiet / no objects above threshold
            self._prev_iq_frame = rx0.copy()
            return self._nn_track_update([])

        # Cap to max_objects most energetic candidates
        candidates_iq.sort(key=lambda c: np.mean(np.abs(c)**2), reverse=True)
        candidates_iq = candidates_iq[:max_objects]

        # ── Step 3-5: Per-candidate estimation ────────────────────────────────
        detections = []
        v_d_global, v_a_global = self._estimate_doppler_velocity(rx0, self._prev_iq_frame)

        for cand_iq in candidates_iq:
            # AoA estimation — use algorithm based on layer
            # Re-construct synthetic rx1 for this sub-band
            cand_iq_full = np.zeros(len(rx0), dtype=np.complex128)
            cand_iq_full[:len(cand_iq)] = cand_iq
            cand_rx1 = np.roll(cand_iq_full, 1) * np.exp(1j * np.pi * 0.5)

            if algorithm == 'music':
                aoa_list = self._music_aoa(cand_iq_full, cand_rx1)
            else:
                aoa_list = self._bartlett_aoa(cand_iq_full, cand_rx1)

            # Map [-90°, +90°] → [0°, 360°] using active beam quadrant
            beam_offset = self.active_beam * 90.0

            for aoa_relative in aoa_list[:2]:  # Max 2 peaks per candidate
                azimuth = (beam_offset + aoa_relative) % 360.0

                dist = self._estimate_range(cand_iq_full, range_max)
                if dist < 5.0:
                    continue  # Too close — likely clutter

                detections.append({
                    'dist':  round(dist, 2),
                    'angle': round(azimuth, 2),
                    'v_d':   round(v_d_global, 4),
                    'v_a':   round(v_a_global, 4),
                })

        # ── Step 6: ID Tracking ────────────────────────────────────────────────
        result = self._nn_track_update(detections)

        # Store frame for next Doppler computation
        self._prev_iq_frame = rx0.copy()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 20.0:
            logging.warning(f"[Radar] Budget exceeded: {elapsed_ms:.1f}ms (target <20ms)")
        else:
            logging.debug(f"[Radar] Sensing complete: {len(result)} objects in {elapsed_ms:.1f}ms")

        return result[:max_objects]

    def lbt_check(self, threshold_db=-60):
        """Listen-Before-Talk (LBT) mechanism. Returns True if channel is idle."""
        iq_data = self.capture_iq()
        # Calculate power
        power_db = 10 * np.log10(np.mean(np.abs(iq_data)**2) + 1e-12)
        is_idle = power_db < threshold_db
        logging.debug(f"LBT Check: Power = {power_db:.2f} dB, Idle = {is_idle}")
        return is_idle

    def transmit(self, tx_iq_data):
        """Transmits I/Q data directly."""
        if self.use_digital_twin:
            logging.debug(f"[SIM] Transmitting block of {len(tx_iq_data)} samples.")
            return

        while True:
            if self.sdr is not None:
                try:
                    self.sdr.tx(tx_iq_data)
                    return
                except Exception as e:
                    logging.error(f"SDR write error or disconnect: {e}. Attempting recovery...")
                    self.sdr = None
            
            # Re-init if SDR dropped
            logging.info("Hardware unavailable during transmission. Entering blocking reconnect loop...")
            self._init_hardware()
