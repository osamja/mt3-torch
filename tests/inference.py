import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from mt3 import spectrograms
from mt3.ml import InferenceModel

class TestAudioToFrames(unittest.TestCase):
    def setUp(self):
        self.spectrogram_config = spectrograms.SpectogramConfig()
        self.model = InferenceModel()

    def test_audio_length_multiple_of_hop_width(self):
        audio = np.random.rand(1024)
        frames, frame_times = self.model._audio_to_frames(audio)
        expected_num_frames = len(audio) // self.spectrogram_config.hop_width + 1   # +1 because we pad the audio to the next frame
        self.assertEqual(frames.shape[0], expected_num_frames)
        self.assertEqual(frames.shape[1], self.spectrogram_config.hop_width)
        self.assertEqual(len(frame_times), expected_num_frames)

    def test_audio_length_5sec(self):
        audio = np.random.rand(80000)   # 80000 is 5 seconds of audio at 16kHz
        frames, frame_times = self.model._audio_to_frames(audio)
        expected_num_frames = len(audio) // self.spectrogram_config.hop_width + 1   # +1 because we pad the audio to the next frame
        self.assertEqual(frames.shape[0], expected_num_frames)
        self.assertEqual(frames.shape[1], self.spectrogram_config.hop_width)
        self.assertEqual(len(frame_times), expected_num_frames)
        self.assertEqual(min(frame_times), 0)
        self.assertEqual(max(frame_times), 5.0)
        print('frame shape, len frame times:', frames.shape, len(frame_times))

if __name__ == '__main__':
    unittest.main()
