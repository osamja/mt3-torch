import numpy as np
from mt3 import spectrograms
import torch

def split_tokens_to_inputs_length(ds, sequence_length, additional_feature_keys=None):
    tokens = ds['inputs']
    max_tokens = sequence_length['inputs']
    num_tokens = tokens.shape[0]    # number of frames
    frame_size = tokens.shape[1]

    num_segments = (num_tokens + max_tokens - 1) // max_tokens
    padding = num_segments * max_tokens - num_tokens

    padded_tokens = np.pad(tokens, ((0, padding), (0, 0)), mode='constant')
    segments = padded_tokens.reshape(num_segments, max_tokens, frame_size)
    ds['inputs'] = segments

    return ds

def flatten_frames(frames):
    """Convert frames back into a flat array of samples."""
    return frames.reshape(-1)

def compute_spectrograms(ex, spectrogram_config):
    for segment in ex['inputs']:
        samples = flatten_frames(segment)
        segment = spectrograms.compute_spectrogram(samples, spectrogram_config).numpy()
        ex['raw_inputs'] = samples
    return ex
