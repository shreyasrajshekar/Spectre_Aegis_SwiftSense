import numpy as np
import torch
import logging

class FastSpectrogramProcessor:
    def __init__(self, device=None, n_fft=256, hop_length=64, input_size=(1, 64, 64)):
        """
        PyTorch-accelerated spectrogram generation to feed the CNN.
        Utilizes RTX 4050 if available for <50ms end-to-end latency.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logging.info(f"Initialized DSP Module on device: {self.device}")
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_shape = input_size # Default (Channels, Height, Width) -> (1, 64, 64)

    def process_iq(self, iq_data) -> torch.Tensor:
        """
        Converts array or tensor of I/Q samples into a 2D magnitude spectrogram tensor
        ready for the CNN.
        """
        # 1. Transfer to Tensor and Device
        if isinstance(iq_data, torch.Tensor):
            iq_tensor = iq_data.to(dtype=torch.complex64, device=self.device)
        else:
            iq_tensor = torch.from_numpy(iq_data).to(dtype=torch.complex64, device=self.device)
        
        # 2. Compute STFT
        # stft expects real inputs for audio, but complex works via proper windowing or we process real/imag separately.
        # Alternatively, for RF, we take STFT of complex signal.
        # PyTorch stft supports complex inputs as of recent versions.
        
        window = torch.hann_window(self.n_fft, device=self.device)
        
        stft_res = torch.stft(
            iq_tensor, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window, 
            return_complex=True,
            center=True
        )
        
        # 3. Magnitude and Log Scale
        mag = torch.abs(stft_res)
        # Add small epsilon to avoid log(0)
        mag_db = 10 * torch.log10(mag + 1e-12)
        
        # 4. Normalize and Resize
        # Normalize to [0, 1] for CNN input
        mag_db = (mag_db - mag_db.min()) / (mag_db.max() - mag_db.min() + 1e-8)
        
        # Add batch and channel dimensions for interpolation: (1, 1, H, W)
        mag_db = mag_db.unsqueeze(0).unsqueeze(0)
        
        # Resize to Target Shape (e.g., 64x64)
        target_h, target_w = self.target_shape[1], self.target_shape[2]
        resized = torch.nn.functional.interpolate(
            mag_db, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Return as (C, H, W) format without batch dim (Controller will handle batching)
        return resized.squeeze(0)
