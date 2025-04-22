from datasets import load_from_disk
import random
import torch
import torchaudio
import os
import json
import sys
import numpy as np
import librosa

from .data_utils import load_audio_array

from .load_commonvoice import load_commonvoice_subset as load_cv

class DatasetLoader:
    def __init__(self, sr=16000, seed=42):
        """
        Initializes the DatasetLoader with default parameters.
        Args:
            sr (int): Default sample rate for audio.
            seed (int): Random seed for shuffling.
        """
        self.sr = sr
        self.seed = seed
        random.seed(self.seed)
        self.rng = np.random.default_rng(seed)

    def _load_json_metadata(self, metadata_path, data_dir, audio_path_key, text_key, max_samples):
        """Loads metadata from a JSON file and returns samples in a consistent format."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
        except FileNotFoundError:
            print(f"Error: Metadata not found at {metadata_path}.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {metadata_path}: {e}")
            sys.exit(1)

        random.seed(self.seed)
        samples = random.sample(samples, min(max_samples, len(samples)))

        formatted_samples = []
        for sample in samples:
            try:
                sentence = sample[text_key]
                if audio_path_key == "google_clean":
                    sample_id = sample.get("id")
                    if sample_id is None:
                        print(f"Warning: Skipping sample in {metadata_path} due to missing 'id' field.")
                        continue
                    audio_relative_path = os.path.join("audio", "clean", f"clean_{sample_id}.wav")
                    full_path = os.path.join(data_dir, audio_relative_path)
                elif audio_path_key in sample:
                    full_path = os.path.join(data_dir, sample[audio_path_key])
                else:
                     print(f"Warning: Skipping sample with id {sample.get('id', 'N/A')} in {metadata_path} due to missing key '{audio_path_key}'.")
                     continue
                formatted_samples.append({
                    "path": full_path.replace("\\", "/"),
                    "sentence": sentence
                })
            except KeyError as e:
                 print(f"Warning: Skipping sample in {metadata_path} due to missing key: {e}. Sample data: {sample}")
                 continue
            except TypeError as e:
                 print(f"Warning: Skipping sample in {metadata_path} due to unexpected data type: {e}. Sample data: {sample}")
                 continue
        return formatted_samples

    def load_coqui_dataset(self, max_samples=100):
        """Loads the Coqui dataset from a JSON file."""
        DATA_DIR = "data/coqui_output"
        METADATA_FILE = os.path.join(DATA_DIR, "all_samples_coqui.json")
        return self._load_json_metadata(METADATA_FILE, DATA_DIR, "coqui_audio_path", "text", max_samples)

    def load_google_synthetic_dataset(self, max_samples=100, use_clean=False):
        """Loads the Google synthetic dataset from a JSON file."""
        DATA_DIR = os.path.join("data", "synthetic_datasets", "GoogleTTS")
        METADATA_FILE = os.path.join(DATA_DIR, "meta", "samples.json")
        if use_clean:
             audio_key = "google_clean"
        else:
             audio_key = "audio_path"
        return self._load_json_metadata(METADATA_FILE, DATA_DIR, audio_key, "text", max_samples)

    def load_ITALIC_dataset(self, max_samples=100):
        """Loads a subset of the ITALIC dataset from disk, extracts audio, resamples, and saves to disk."""
        DATASET_DIR = "data/datasets/ITALIC_massive"
        AUDIO_DIR = os.path.join(DATASET_DIR, "audio_samples")
        os.makedirs(AUDIO_DIR, exist_ok=True)

        try:
            dataset = load_from_disk(DATASET_DIR)
        except FileNotFoundError:
            print(f"Error: ITALIC dataset not found at {DATASET_DIR}.")
            sys.exit(1)

        sampled_dataset = dataset.shuffle(seed=self.seed).select(range(min(max_samples, len(dataset))))

        exported_samples = []
        for i, sample in enumerate(sampled_dataset):
            audio_array = sample["audio"]["array"]
            audio_path = os.path.join(AUDIO_DIR, f"audio_{i}.wav")
            
            audio, _ = load_audio_array(audio_array, sr=self.sr)
            audio = torch.tensor(audio).unsqueeze(0)
            torchaudio.save(audio_path, audio, self.sr)
            exported_samples.append({
                "path": audio_path,
                "sentence": sample["utt"]
            })
        return exported_samples

    def load_commonvoice_subset(self, max_samples=100):
        """Loads a subset of the Common Voice dataset."""

        samples = load_cv(max_samples=max_samples)
        formatted_samples = []
        for path, sentence in samples:
            formatted_samples.append({
                "path": path,
                "sentence": sentence
            })
        return formatted_samples
    
    def load_azure_synthetic_dataset(self, max_samples=100, use_clean=False):
        """Loads the Azure synthetic dataset from a JSON file."""
        DATA_DIR = os.path.join("data", "synthetic_datasets", "AzureTTS")
        METADATA_FILE = os.path.join(DATA_DIR, "meta", "samples.json")
        if use_clean:
            audio_key = "clean_audio_path"
        else:
            audio_key = "audio_path"
        return self._load_json_metadata(METADATA_FILE, DATA_DIR, audio_key, "text", max_samples)

    def load_dataset(self, name, samples_per_dataset, use_clean=False):
        if name == "common_voice":
            return self.load_commonvoice_subset(samples_per_dataset)
        elif name == "coqui":
            return self.load_coqui_dataset(samples_per_dataset)
        elif name == "google_synthetic":
            return self.load_google_synthetic_dataset(samples_per_dataset, use_clean=use_clean)
        elif name == "ITALIC":
            return self.load_ITALIC_dataset(samples_per_dataset)
        elif name == "azure_synthetic":
            return self.load_azure_synthetic_dataset(samples_per_dataset, use_clean=use_clean)
        else:
            raise ValueError(f"Unknown dataset: {name}")