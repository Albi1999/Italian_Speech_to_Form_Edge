# data_utils.py
import numpy as np

def load_audio(audio_input, sr=16000):
    """Loads and preprocesses audio, handling both file paths and numpy arrays."""
    if isinstance(audio_input, str):
        raise ValueError("Invalid parameter type. Parameter type must be a numpy array")
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
        sample_rate = sr  # Assume the desired sample rate if not provided
    else:
        raise ValueError("audio_input must be a file path (str) or a numpy array.")

    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    audio = audio / np.max(np.abs(audio))

    return audio, sample_rate