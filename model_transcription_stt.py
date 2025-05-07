import os
import json
from pyexpat import model
from tqdm import tqdm

from models import Vosk, Wav2Vec2Grosman
from utils import DatasetLoader

#OUTPUT_DIR = "output/stt"
OUTPUT_DIR = "output/stt/train_transcription"
os.makedirs(OUTPUT_DIR, exist_ok=True)
#NUM_SAMPLES = 100
NUM_SAMPLES = 500
#dataset_names = ["google_synthetic", "azure_synthetic", "coqui"]
dataset_names = ["google_synthetic", "azure_synthetic"]

def transcribe_and_save_results(model, dataset_name, samples):
    """
    Transcribes audio samples using the Vosk model and returns the results.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
            original sample data along with the transcribed text.  Returns an
            empty list if an error occurs during transcription.
    """
    results = []
    for s in tqdm(samples, desc=dataset_name):
        try:
            transcription = model.transcribe(s['path'], s['sentence'])
            results.append({
                'dataset': dataset_name,
                'file_path': s['path'],
                'original_sentence': s['sentence'],
                'transcribed_sentence': transcription
            })
        except Exception as e:
            print(f"Error transcribing {s['path']} from {dataset_name}: {e}")
            results.append({
                'dataset': dataset_name,
                'file_path': s['path'],
                'original_sentence': s['sentence'],
                'transcribed_sentence': "Transcription Failed"
            })
    return results

if __name__ == "__main__":
    
    # Initialize the transcription model
    model_name = "vosk"
    #model_name = "wav2vec2"
    print(f"Loading {model_name} Model...")
    
    #model = Wav2Vec2Grosman()
    model = Vosk()

    # Initialize the dataset loader
    data_loader = DatasetLoader()

    all_transcribed_sentences = []

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        # Load the dataset
        samples = data_loader.load_dataset(dataset_name, NUM_SAMPLES, use_clean=True, use_train=True)
        # Transcribe and save the results
        results = transcribe_and_save_results(model, dataset_name, samples)
        all_transcribed_sentences.extend(results)  # Add to the cumulative list

    # Save all transcribed sentences to a single JSON file
    #output_file_path = os.path.join(OUTPUT_DIR, f"{model_name}_transcription", "transcriptions.json")
    output_file_path = os.path.join(OUTPUT_DIR, "train_transcriptions.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_transcribed_sentences, f, indent=4, ensure_ascii=False)
        print(f"All transcriptions saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving to JSON: {e}")