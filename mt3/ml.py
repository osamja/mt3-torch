import torch
# import spectograms module
from mt3 import spectrograms
import numpy as np

SAMPLE_RATE = 16000

class InferenceModel(object):
    "Pytorch wrapper of the MT3 architecture for music transcription inference."

    def __init__(self, model_path=None):
        self.spectrogram_config = spectrograms.SpectogramConfig()
        # self.model = torch.jit.load(model_path)
        # self.model.eval()

    def __call__(self, audio):
        ds = self.audio_to_dataset(audio)

    def audio_to_dataset(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        print(frames.shape)
        
    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]   # we'll always pad to the next frame. [0, pad_width] means pad nothing at beginning, but pad pad_width to end of array with default value 0
        # print('Pad width:', padding)
        audio = np.pad(audio, padding, mode='constant')
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        frame_times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, frame_times
