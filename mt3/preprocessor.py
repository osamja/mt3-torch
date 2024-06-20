import numpy as np
from mt3 import spectrograms
import torch
import scipy.io.wavfile as wav

def split_tokens_to_inputs_length(ds, sequence_length, additional_feature_keys=None):
    tokens = ds['inputs']
    input_times = ds['input_times']
    max_tokens = sequence_length['inputs']
    num_tokens = tokens.shape[0]    # number of frames
    frame_size = tokens.shape[1]

    num_segments = (num_tokens + max_tokens - 1) // max_tokens
    padding = num_segments * max_tokens - num_tokens

    # Pad tokens and input_times to ensure they are multiples of max_tokens
    padded_tokens = np.pad(tokens, ((0, padding), (0, 0)), mode='constant')
    padded_times = np.pad(input_times, (0, padding), mode='constant')

    # Reshape into segments
    segments_tokens = padded_tokens.reshape(num_segments, max_tokens, frame_size)
    segments_times = padded_times.reshape(num_segments, max_tokens)

    # Create list of dictionaries for each segment
    segments = []
    for i in range(num_segments):
        segment = {
            'inputs': segments_tokens[i],
            'input_times': segments_times[i]
        }
        segments.append(segment)

    return segments

def flatten_frames(frames):
    """Convert frames back into a flat array of samples."""
    return frames.reshape(-1)

def compute_spectrograms(ds, spectrogram_config):
    for i, segment in enumerate(ds):
        samples = flatten_frames(segment['inputs'])
        spectrogram = spectrograms.compute_spectrogram(samples, spectrogram_config).numpy()
        segment['raw_inputs'] = samples
        segment['inputs'] = spectrogram
    
    return ds

def split_audio_segments(audio, sample_rate, chunk_length_ms=2000, num_chunks=5):
    """Split the audio file into segments of chunk_length_ms and save them as AudioChunk objects."""

    split_audio = []

    # Length of the audio in milliseconds
    length_ms = len(audio) / sample_rate * 1000

    # Start and end points for slicing
    start_ms = 0
    end_ms = chunk_length_ms

    file_counter = 0

    while start_ms < length_ms and file_counter < num_chunks:
        # Convert start and end times from milliseconds to samples
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)

        # Extract the chunk
        chunk = audio[start_sample:end_sample]
        split_audio.append(chunk)

        # Move to the next chunk
        start_ms = end_ms
        end_ms += chunk_length_ms
        file_counter += 1

    return split_audio