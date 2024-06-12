import numpy as np
import torch
import torchaudio.transforms as transforms
from librosa import filters

DEFAULT_HOP_WIDTH = 128
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_MEL_BINS = 512

# Constants for spectrogram computation
FFT_SIZE = 2048
MEL_LO_HZ = 20.0

class SpectogramConfig:
    hop_width: int = DEFAULT_HOP_WIDTH
    sample_rate: int = DEFAULT_SAMPLE_RATE
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS
    fft_size: int = FFT_SIZE
    mel_lo_hz: float = MEL_LO_HZ

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width
    
    @property
    def abbrev_str(self):
        s = ''
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += 'sr%d' % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += 'hw%d' % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += 'mb%d' % self.num_mel_bins
        return s

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


def compute_spectrogram(samples, spectrogram_config):
    """
    This function computes a mel spectrogram from a given audio sample.

    Args:
        samples: A NumPy array of audio samples.
        sample_rate: The sample rate of the audio in Hz (default: 16000).
        frame_size: The hop length or frame size in samples (default: 128).
        n_mels: The number of mel filterbanks (default: 512).

    Returns:
        A PyTorch tensor of shape (time_steps, n_mels) representing the mel spectrogram.
    """
    sample_rate = spectrogram_config.sample_rate
    frame_size = spectrogram_config.hop_width
    n_mels = spectrogram_config.num_mel_bins

    # Convert samples to a PyTorch tensor
    samples_tensor = torch.from_numpy(samples).float()

    # Compute spectrogram using torch.stft with return_complex=True
    spectrogram = torch.stft(samples_tensor, n_fft=frame_size * 2, hop_length=frame_size, win_length=frame_size, window=torch.hann_window(frame_size), return_complex=True)

    # Magnitude spectrogram
    magnitude_spectrogram = torch.abs(spectrogram)

    # Create mel filters
    mel_filters = filters.mel(sr=sample_rate, n_fft=frame_size * 2, n_mels=n_mels)

    # Convert mel_filters to PyTorch tensor (ensure correct order)
    mel_filters_tensor = torch.from_numpy(mel_filters.T).float()  # Transpose mel_filters

    # Apply mel filters to magnitude spectrogram
    mel_spectrogram = torch.matmul(magnitude_spectrogram.T, mel_filters_tensor)

    # Ensure resulting spectrogram has desired shape
    mel_spectrogram = mel_spectrogram.squeeze(0)  # Remove batch dimension if present

    return mel_spectrogram

