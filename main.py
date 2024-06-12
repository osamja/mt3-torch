import librosa
from mt3 import ml

piano_song = 'datasets/relaxing-piano-201831.mp3'
audio, sr = librosa.load(piano_song, sr=16000)
print(audio.shape, sr)
model = ml.InferenceModel()
# dataset = model.audio_to_dataset(audio)
inferenced = model(audio)
