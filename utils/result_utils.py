import pandas as pd
import os
import matplotlib.pyplot as plt

def save_results(all_results, output_dir="output/stt/csv"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/stt_results.csv", index=False)
    print(f"\nResults saved to {output_dir}/stt_results.csv")
