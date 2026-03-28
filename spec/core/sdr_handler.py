import sys
import numpy as np
import time
import logging
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
            # Target is to keep this process < 10ms
            t_start = time.perf_counter()
            self.sdr.rx_lo = self.center_freq
            self.sdr.tx_lo = self.center_freq
            t_end = time.perf_counter()
            logging.debug(f"Frequency hop to {frequency/1e6} MHz completed in {(t_end - t_start)*1000:.2f} ms")

    def capture_iq(self):
        """Captures a buffer of I/Q samples."""
        if self.use_digital_twin:
            return self.simulator.generate_iq()
            
        while True:
            if self.sdr is not None:
                try:
                    rx_data = self.sdr.rx()
                    return rx_data
                except Exception as e:
                    logging.error(f"SDR read error or disconnect: {e}. Attempting recovery...")
                    self.sdr = None
            
            # If sdr is None (either here or on entry), block and re-init
            logging.info("Hardware unavailable. Entering blocking reconnect loop...")
            self._init_hardware()

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
