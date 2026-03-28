import numpy as np
import logging

class DigitalTwinSimulator:
    def __init__(self, buffer_size=16384):
        self.buffer_size = buffer_size
        self.active_signal_type = "Idle"
        self.signal_lifetime = 0
        self.active_beam = 0  # 0 to 3
        self.radar_objects = [] # List of {dist, angle, id, velocity_dist, velocity_angle}
        self.current_layer = "Legacy"
        np.random.seed(42)
        logging.info("Digital Twin Simulator Initialized. Synthesizing 6G ISAC & Eco-Sensing traffic.")

    def generate_iq(self):
        """Generates mock complex I/Q data representing physical layer signatures."""
        # Base noise floor
        iq = (np.random.randn(self.buffer_size) + 1j * np.random.randn(self.buffer_size)) * 0.1
        
        # Scaling physics based on frequency layer
        noise_floor = -80 if self.current_layer == "cmWave" else -70 if self.current_layer == "Sub-THz" else -60
        max_objs = 3 if self.current_layer == "cmWave" else 5 if self.current_layer == "Sub-THz" else 12

        # Change state randomly to simulate active environment
        if self.signal_lifetime <= 0:
            self.active_signal_type = np.random.choice(["Idle", "6G-URLLC", "6G-eMBB", "Legacy IoT"], p=[0.4, 0.1, 0.3, 0.2])
            self.active_beam = np.random.randint(0, 4)
            self.signal_lifetime = np.random.randint(20, 100) 
            
        # Simulate ISAC Spatial Objects Tracking - Persistence
        # In THz mode, we simulate holographic sensing with much higher object density
        if len(self.radar_objects) < 2 or (len(self.radar_objects) < max_objs and np.random.random() < 0.15):
            dist = 20 if self.active_signal_type == "6G-URLLC" else np.random.randint(15, 90)
            base_angle = self.active_beam * 90 if self.active_signal_type != "Idle" else np.random.randint(0, 360)
            angle = base_angle + np.random.randint(-40, 40)
            self.radar_objects.append({
                "dist": float(dist),
                "angle": float(angle % 360),
                "v_d": np.random.uniform(-0.5, 0.5),
                "v_a": np.random.uniform(-1.5, 1.5),
                "id": np.random.randint(100, 999)
            })
            
        # Add background noise scaled by frequency layer (applied to full buffer)
        noise_full = (np.random.randn(self.buffer_size) + 1j * np.random.randn(self.buffer_size)) * 10**(noise_floor/20)
        iq += noise_full
            
        # Update kinematic state for all objects
        for obj in self.radar_objects:
            obj["dist"] += obj["v_d"]
            obj["angle"] = (obj["angle"] + obj["v_a"]) % 360
            
            # Boundary check
            if obj["dist"] < 5 or obj["dist"] > 95:
                obj["v_d"] *= -1
                
        # Occasionally swap out old objects to keep it dynamic
        if len(self.radar_objects) > 4:
            self.radar_objects.pop(0)

        self.signal_lifetime -= 1
        
        if self.active_signal_type == "Idle":
            return iq
            
        t = np.arange(self.buffer_size)
        
        if self.active_signal_type == "6G-URLLC":
            # High-priority, periodic ultra-reliable bursts
            pulse_train = np.zeros(self.buffer_size, dtype=complex)
            for i in range(0, self.buffer_size, 1000):
                pulse_train[i:i+100] = 5.0 + 5.0j
            iq += pulse_train
            
        elif self.active_signal_type == "6G-eMBB":
            # Broadband high-throughput OFDM
            subcarriers = np.sum([np.exp(1j * 2 * np.pi * f * t / self.buffer_size) 
                                  for f in np.random.randint(10, 800, 30)], axis=0)
            iq += subcarriers * 0.6
            
        elif self.active_signal_type == "Legacy IoT":
            # Narrowband chirps
            f_center = 200
            chirp = np.exp(1j * 2 * np.pi * (f_center * t + 0.5 * 2 * t**2 / self.buffer_size) / self.buffer_size)
            iq += chirp * 1.2
            
        # Simulate Spatial Multiplexing (Beamforming)
        # If the receiver beam is NOT aligned with the transmitter beam, attenuate the signal
        # This allows sharing the same freq if beams are orthogonal.
        # We'll use a simple 'beam_index' from the SDRHandler to check alignment
        return iq

    def apply_spatial_filter(self, iq, rx_beam_idx):
        """Attenuates the signal if external TX beam vs local RX beam are mismatched"""
        if rx_beam_idx != self.active_beam:
            # orthogonal/misaligned beam -> 20dB attenuation
            return iq * 0.1
        return iq

    def get_radar_map(self):
        """Returns the simulated ISAC radar tracking list."""
        return self.radar_objects
