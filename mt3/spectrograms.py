import numpy as np
import torch
DEFAULT_HOP_WIDTH = 128
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_MEL_BINS = 512

class SpectogramConfig:
    hop_width: int = DEFAULT_HOP_WIDTH
    sample_rate: int = DEFAULT_SAMPLE_RATE

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width

def split_audio(samples, spectrogram_config):
    """Split audio into frames using PyTorch and NumPy."""
    frame_length = spectrogram_config.hop_width
    frame_step = spectrogram_config.hop_width
    pad_end = True
    
    if pad_end:
        pad_value = (len(samples) + frame_step - 1) // frame_step * frame_step - len(samples)
        samples = np.pad(samples, (0, pad_value), mode='constant')
    
    num_frames = (len(samples) - frame_length) // frame_step + 1
    
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(num_frames, frame_length),
        strides=(samples.strides[0] * frame_step, samples.strides[0])
    )
    
    return torch.tensor(frames)
