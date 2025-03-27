import librosa

def load_audio(path, sr=16000):
    audio, sample_rate = librosa.load(path, sr=sr)
    return audio, sample_rate
