import numpy as np
import logging

class DigitalTwinSimulator:
    def __init__(self, buffer_size=16384):
        self.buffer_size = buffer_size
        self.active_signal_type = "Idle"
        self.signal_lifetime = 0
        logging.info("Digital Twin Simulator Initialized. Synthesizing zero-latency Multi-Class I/Q traffic.")

    def generate_iq(self):
        """Generates mock complex I/Q data representing physical layer signatures."""
        # Base noise floor
        iq = (np.random.randn(self.buffer_size) + 1j * np.random.randn(self.buffer_size)) * 0.1
        
        # Change state randomly to simulate active environment
        if self.signal_lifetime <= 0:
            self.active_signal_type = np.random.choice(["Idle", "5G", "IoT", "Radar"], p=[0.4, 0.2, 0.3, 0.1])
            self.signal_lifetime = np.random.randint(10, 50) # Maintain signal for 10-50 cycles
        self.signal_lifetime -= 1
        
        if self.active_signal_type == "Idle":
            return iq
            
        t = np.arange(self.buffer_size)
        
        if self.active_signal_type == "5G":
            # Broadband OFDM-like
            subcarriers = np.sum([np.exp(1j * 2 * np.pi * f * t / self.buffer_size) 
                                  for f in np.random.randint(10, 500, 20)], axis=0)
            iq += subcarriers * 0.8
            
        elif self.active_signal_type == "IoT":
            # Narrowband LoRa/NB-IoT like (Chirps or steady tones)
            f_center = 200
            chirp = np.exp(1j * 2 * np.pi * (f_center * t + 0.5 * 2 * t**2 / self.buffer_size) / self.buffer_size)
            iq += chirp * 1.5
            
        elif self.active_signal_type == "Radar":
            # Periodic high-power pulses
            pulse_train = np.zeros(self.buffer_size, dtype=complex)
            for i in range(0, self.buffer_size, 2000):
                pulse_train[i:i+200] = 5.0 + 5.0j
            iq += pulse_train
            
        return iq
