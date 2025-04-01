from datasets import load_dataset, load_from_disk
import os
import json

DATASET_NAME = "ITALIC"
DATASET_CONFIG = "massive"
DATASET_DIR = f"data/datasets/{DATASET_NAME}_{DATASET_CONFIG}"
SAMPLE_EXPORT_PATH = f"{DATASET_DIR}/test_samples.json"
NUM_SAMPLES = 20

def prepare_dataset():
    if not os.path.exists(DATASET_DIR):
        print("Dataset not found in local storage. Downloading...")

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Enable transfer to avoid download issues

        dataset = load_dataset(
            "RiTA-nlp/ITALIC",
            name=DATASET_CONFIG,
            split="test",
            trust_remote_code=True
        )
        print("Dataset downloaded. Saving to disk...")
        dataset.save_to_disk(DATASET_DIR)
    else:
        print("Dataset found in local storage.")

    print("Loading dataset...")
    dataset = load_from_disk(DATASET_DIR)

    print(f"Selecting {NUM_SAMPLES} sample...")
    sampled_dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

    # Save subset sample to disk
    sample_dicts = [
        {
            "id": i,
            "text": sample["utt"],
            "audio_path": f"audio_{i}.wav"
        }
        for i, sample in enumerate(sampled_dataset)
    ]
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(SAMPLE_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(sample_dicts, f, ensure_ascii=False, indent=2)

    print(f"Sample saved in: {SAMPLE_EXPORT_PATH}")

    return sampled_dataset

if __name__ == "__main__":
    prepare_dataset()
