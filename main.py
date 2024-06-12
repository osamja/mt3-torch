import librosa
from mt3 import ml

songs = [
    'datasets/in-the-morning-jcole.mp3',
    'datasets/relaxing-piano-201831.mp3',
]

piano_song = songs[0]
audio, sr = librosa.load(piano_song, sr=16000)
print(audio.shape, sr)
model = ml.InferenceModel()
# dataset = model.audio_to_dataset(audio)
inferenced = model(audio)
