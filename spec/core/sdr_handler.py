import sys
import numpy as np
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple
from core.digital_twin import DigitalTwinSimulator

import torch
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

        # ── Device Selection ────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"SDRHandler sensing module initialized on: {self.device}")

        # ── ISAC Radar Sensing State ────────────────────────────────────────────
        self._prev_iq_frame: Optional[np.ndarray] = None   # Previous frame for Doppler
        self._radar_tracks: Dict[int, dict] = {}            # id → {dist, angle, v_d, v_a, miss_count}
        self._next_track_id: int = 100                      # Auto-incrementing object ID
        self._SPEED_OF_LIGHT: float = 3e8
        
        # ── Async Sensing ───────────────────────────────────────────────────────
        self._latest_rx = None
        self._rx_running = False
        self._rx_thread = None
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
            
            # Start Async I/O Thread
            if not self._rx_running:
                self._rx_running = True
                self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
                self._rx_thread.start()
                logging.info("Background Hardware Rx thread started.")

        except Exception as e:
            logging.error(f"Failed to configure SDR parameters: {e}")
            self.sdr = None # Reset and let caller retry / timeout

    def _rx_worker(self):
        """Asynchronously pulls data from the ADALM-Pluto to ensure main loop hits <50ms target."""
        while self._rx_running:
            if self.sdr is not None:
                try:
                    # Blocking hardware call runs ONLY in this thread
                    self._latest_rx = self.sdr.rx()
                except Exception as e:
                    logging.error(f"SDR async read error: {e}")
                    self.sdr = None
            else:
                time.sleep(0.5)

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
        """Instantly returns the latest buffer without blocking for hardware transfer."""
        if self.use_digital_twin:
            raw_iq = self.simulator.generate_iq()
            return self.simulator.apply_spatial_filter(raw_iq, self.active_beam)
            
        if self.sdr is None:
            logging.info("Hardware unavailable. Entering blocking reconnect loop...")
            self._init_hardware()
            return None
                
        # Return instantly. If not populated yet, wait slightly or return None.
        if self._latest_rx is not None:
            return self._latest_rx
            
        # First-time boot sync
        time.sleep(0.01)
        return self._latest_rx

    def close(self):
        self._rx_running = False
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # ISAC RADAR SENSING — Hardware Implementation
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_layer_config(self) -> dict:
        """Returns layer-specific algorithm parameters (Legacy only)."""
        return {'max_objects': 12, 'range_max': 100.0, 'algorithm': 'bartlett', 'snr_db': -20.0}

    def _music_aoa(self, rx0: torch.Tensor, rx1: torch.Tensor, n_scan: int = 181) -> List[float]:
        """
        2-element ULA MUSIC angle-of-arrival estimation.
        GPU-accelerated via PyTorch.
        """
        N = rx0.shape[0]
        X = torch.stack([rx0, rx1], dim=0)          # shape (2, N)
        # Covariance matrix R
        R = (X @ X.conj().T) / N                   # shape (2, 2)
        
        # Eigendecomposition on GPU
        eigenvalues, eigenvectors = torch.linalg.eigh(R)
        E_n = eigenvectors[:, :1]                  # noise eigenvector (2×1)

        thetas = torch.linspace(-np.pi / 2, np.pi / 2, n_scan, device=self.device)
        
        # Vectorized steering vectors: shape (2, n_scan)
        A = torch.stack([
            torch.ones(n_scan, device=self.device, dtype=torch.complex64),
            torch.exp(1j * np.pi * torch.sin(thetas))
        ], dim=0)
        
        # Projection matrix P_n
        P_n = E_n @ E_n.conj().T
        
        # A^H P_n A diagonals
        P_n_A = P_n @ A
        denom = torch.abs(torch.sum(A.conj() * P_n_A, dim=0))
        spectrum = 1.0 / (denom + 1e-12)

        # Normalise
        spectrum /= (spectrum.max() + 1e-12)
        
        # Find peaks on GPU (simplified)
        spectrum_cpu = spectrum.cpu().numpy()
        thetas_cpu = thetas.cpu().numpy()
        peaks = []
        for i in range(1, n_scan - 1):
            if spectrum_cpu[i] > spectrum_cpu[i-1] and spectrum_cpu[i] > spectrum_cpu[i+1] and spectrum_cpu[i] > 0.3:
                peaks.append(float(np.degrees(thetas_cpu[i])))
        return peaks if peaks else [float(np.degrees(thetas_cpu[np.argmax(spectrum_cpu)]))]

    def _bartlett_aoa(self, rx0: torch.Tensor, rx1: torch.Tensor, n_scan: int = 181) -> List[float]:
        """
        Bartlett beamforming — GPU-accelerated via PyTorch.
        """
        N = rx0.shape[0]
        X = torch.stack([rx0, rx1], dim=0)
        R = (X @ X.conj().T) / N
        thetas = torch.linspace(-np.pi / 2, np.pi / 2, n_scan, device=self.device)
        
        A = torch.stack([
            torch.ones(n_scan, device=self.device, dtype=torch.complex64),
            torch.exp(1j * np.pi * torch.sin(thetas))
        ], dim=0)
        
        R_A = R @ A
        spectrum = torch.real(torch.sum(A.conj() * R_A, dim=0))
        
        spectrum /= (spectrum.max() + 1e-12)
        spectrum_cpu = spectrum.cpu().numpy()
        thetas_cpu = thetas.cpu().numpy()
        peaks = []
        for i in range(1, n_scan - 1):
            if spectrum_cpu[i] > spectrum_cpu[i-1] and spectrum_cpu[i] > spectrum_cpu[i+1] and spectrum_cpu[i] > 0.25:
                peaks.append(float(np.degrees(thetas_cpu[i])))
        return peaks if peaks else [float(np.degrees(thetas_cpu[np.argmax(spectrum_cpu)]))]

    def _estimate_range(self, iq: torch.Tensor, range_max: float) -> float:
        """
        Estimates range from dominant FFT bin using PyTorch GPU FFT.
        """
        N = iq.shape[0]
        window = torch.hann_window(N, device=self.device)
        # GPU FFT
        full_spectrum = torch.abs(torch.fft.fft(iq * window))
        half_N = N // 2
        spectrum = full_spectrum[1:half_N]
        
        if spectrum.numel() == 0 or torch.max(spectrum) < 1e-6:
            return 0.0
            
        peak_bin = torch.argmax(spectrum).item() + 1
        f_peak = float(peak_bin * self.sample_rate / N)
        nyquist = self.sample_rate / 2.0
        dist = max(5.0, min(range_max, (f_peak / nyquist) * range_max))
        return round(dist, 2)

    def _estimate_doppler_velocity(self, iq_curr: torch.Tensor, iq_prev: Optional[torch.Tensor]) -> Tuple[float, float]:
        """
        Estimates radial velocity (v_d) and angular velocity variance (v_a) using GPU tensors.
        """
        if iq_prev is None or iq_prev.shape[0] != iq_curr.shape[0]:
            return 0.0, 0.0
            
        T_frame = self.buffer_size / self.sample_rate
        # Phase-rotation via complex conjugate multiplication
        phase_prod = iq_curr * torch.conj(iq_prev)
        mean_phase = torch.angle(torch.mean(phase_prod)).item()
        
        f_doppler = mean_phase / (2 * np.pi * T_frame)
        v_d = float(f_doppler * self._SPEED_OF_LIGHT / (2.0 * self.center_freq))
        
        # Angular velocity variance
        phase_variance = torch.var(torch.angle(phase_prod)).item()
        v_a = round(float(np.degrees(phase_variance) * 0.1), 3)
        return round(v_d, 4), round(v_a, 4)

    def _cfar_candidates(self, iq: torch.Tensor, snr_threshold_db: float) -> List[torch.Tensor]:
        """
        GPU-accelerated CFAR candidate extraction using PyTorch.
        """
        N = iq.shape[0]
        n_fft = min(N, 2048)
        iq_slice = iq[:n_fft]
        full_fft = torch.fft.fft(iq_slice)
        spectrum_mag = torch.abs(full_fft)
        
        # Use median for noise floor
        noise_floor = torch.median(spectrum_mag).item()
        threshold_linear = noise_floor * (10 ** (snr_threshold_db / 20.0))

        # Vectorized thresholding
        above = (spectrum_mag > threshold_linear).cpu().numpy()
        candidates = []
        in_group = False
        g_start = 0
        for i, a in enumerate(above):
            if a and not in_group:
                g_start = i
                in_group = True
            elif not a and in_group:
                in_group = False
                sub = torch.zeros(n_fft, dtype=torch.complex64, device=self.device)
                sub[g_start:i] = full_fft[g_start:i]
                candidates.append(torch.fft.ifft(sub))
        if in_group:
            sub = torch.zeros(n_fft, dtype=torch.complex64, device=self.device)
            sub[g_start:] = full_fft[g_start:]
            candidates.append(torch.fft.ifft(sub))
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

    def get_sensing_radar(self, iq_buffer: Optional[torch.Tensor] = None) -> List[Dict]:
        """
        6G ISAC Hardware Radar Sensing Pipeline (GPU Accelerated).
        """
        if self.use_digital_twin:
            return self.simulator.get_radar_map()

        t0 = time.perf_counter()
        cfg = self._get_layer_config()
        max_objects  = cfg['max_objects']
        range_max    = cfg['range_max']
        algorithm    = cfg['algorithm']
        snr_db       = cfg['snr_db']

        # ── Step 1: Capture or Reuse I/Q ───────────────────────────────────────
        if iq_buffer is not None:
            # Reusing buffer from main loop (assumed already a GPU tensor)
            rx0 = iq_buffer
        else:
            try:
                raw = self.sdr.rx()  # Returns list[ch0, ch1] if 2-ch, else ndarray
            except Exception as e:
                logging.warning(f"[Radar] rx() failed: {e}")
                return list(self._radar_tracks.values())[:max_objects]

            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                rx0 = torch.from_numpy(np.asarray(raw[0])).to(device=self.device, dtype=torch.complex64)
            else:
                rx0 = torch.from_numpy(np.asarray(raw)).to(device=self.device, dtype=torch.complex64)

        # ── Step 2: SNR gate / CFAR candidate extraction ───────────────────────
        candidates_iq = self._cfar_candidates(rx0, snr_threshold_db=snr_db)
        if not candidates_iq:
            self._prev_iq_frame = rx0.detach().clone()
            return self._nn_track_update([])

        # Cap to max_objects most energetic candidates
        candidates_iq.sort(key=lambda c: torch.mean(torch.abs(c)**2).item(), reverse=True)
        candidates_iq = candidates_iq[:max_objects]

        # ── Step 3-5: Per-candidate estimation ────────────────────────────────
        detections = []
        v_d_global, v_a_global = self._estimate_doppler_velocity(rx0, self._prev_iq_frame)

        for cand_iq in candidates_iq:
            # Re-construct synthetic rx1 for this sub-band on GPU
            cand_iq_full = torch.zeros(rx0.shape[0], dtype=torch.complex64, device=self.device)
            n_c = cand_iq.shape[0]
            cand_iq_full[:n_c] = cand_iq
            cand_rx1 = torch.roll(cand_iq_full, 1) * np.exp(1j * np.pi * 0.5)

            if algorithm == 'music':
                aoa_list = self._music_aoa(cand_iq_full, cand_rx1)
            else:
                aoa_list = self._bartlett_aoa(cand_iq_full, cand_rx1)

            beam_offset = self.active_beam * 90.0
            for aoa_relative in aoa_list[:2]:
                azimuth = (beam_offset + aoa_relative) % 360.0
                dist = self._estimate_range(cand_iq_full, range_max)
                if dist < 5.0:
                    continue

                detections.append({
                    'dist':  round(dist, 2),
                    'angle': round(azimuth, 2),
                    'v_d':   round(v_d_global, 4),
                    'v_a':   round(v_a_global, 4),
                })

        # ── Step 6: ID Tracking ────────────────────────────────────────────────
        result = self._nn_track_update(detections)
        self._prev_iq_frame = rx0.detach().clone()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 35.0:
            logging.warning(f"[Radar] Budget exceeded: {elapsed_ms:.1f}ms (target <35ms)")
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
