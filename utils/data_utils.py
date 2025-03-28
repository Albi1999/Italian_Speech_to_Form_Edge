import librosa
import numpy as np

def load_audio(path, sr=16000):
    # Load audio
    audio, sample_rate = librosa.load(path, sr=sr)
    
    # Normalize audio (important for Whisper models)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # Convert stereo to mono if needed
    
    audio = audio / np.max(np.abs(audio))  # Normalize
    
    return audio, sample_rate
