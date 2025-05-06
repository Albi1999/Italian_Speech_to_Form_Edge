# Pricing Information (USD per 1,000,000 tokens)
PRICING = {
    "gemini-2.0-flash": {"input": 0.70, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30}, #Assuming price <= 128K context
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00} #Assuming price <= 128K context
}

# Estimated Token Counts (Obtained from the previous script)
ESTIMATED_TOKENS = {
    "gemini-2.0-flash": {"input": 489, "output": 502},
    "gemini-2.0-flash-lite": {"input": 489, "output": 491},
    "gemini-1.5-flash": {"input": 489, "output": 501},
    "gemini-1.5-pro": {"input": 489, "output": 502}
}

SAMPLE_SIZES = [80, 200, 400]  # Number of audio files in each sample

def estimate_cost(model_name, num_samples):
    """Estimates the cost of processing a given number of audio samples.

    Args:
        model_name (str): The name of the Gemini model.
        num_samples (int): The number of audio files to process.

    Returns:
        float: The estimated cost in USD.
    """
    if model_name not in ESTIMATED_TOKENS:
        print(f"Error: No token estimates found for model '{model_name}'.")
        return None

    if model_name not in PRICING:
        print(f"Error: No pricing information found for model '{model_name}'.")
        return None

    input_tokens = ESTIMATED_TOKENS[model_name]["input"]
    output_tokens = ESTIMATED_TOKENS[model_name]["output"]

    input_cost_per_sample = (input_tokens / 1_000_000) * PRICING[model_name]["input"]
    output_cost_per_sample = (output_tokens / 1_000_000) * PRICING[model_name]["output"]

    total_cost = (input_cost_per_sample + output_cost_per_sample) * num_samples
    return total_cost

# Run cost estimation for all models and sample sizes
print("Cost Estimation (USD):")
for model_name in ESTIMATED_TOKENS:
    print(f"\nModel: {model_name}")
    for num_samples in SAMPLE_SIZES:
        cost = estimate_cost(model_name, num_samples)
        if cost is not None:
            print(f"  {num_samples} samples: ${cost:.4f}")
        else:
            print(f"  {num_samples} samples: Cost estimation failed.")

print("Finished.")