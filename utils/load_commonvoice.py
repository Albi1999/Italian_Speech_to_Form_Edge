import os
import pandas as pd

def load_commonvoice_subset(
    base_path="data/datasets/cv-corpus-21.0-delta-2025-03-14/it",
    max_samples=10
):

    validated_path = os.path.join(base_path, "validated.tsv")
    clips_dir = os.path.join(base_path, "clips")

    df = pd.read_csv(validated_path, sep="\t")
    df = df.dropna(subset=["path", "sentence"])
    df = df.sample(n=max_samples, random_state=42)

    samples = []
    for _, row in df.iterrows():
        audio_path = os.path.join(clips_dir, row["path"])
        transcript = row["sentence"].lower().strip()
        samples.append((audio_path, transcript))

    return samples


