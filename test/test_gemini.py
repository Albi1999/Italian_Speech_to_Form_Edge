import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from utils import AudioEncoder

# Constants
AUDIO_PATH = "data\datasets\cv-corpus-21.0-delta-2025-03-14\it\clipscommon_voice_it_42491529.mp3"
AUDIO_DURATION = 15  # Average audio duration in seconds
AUDIO_TOKENS_PER_SECOND = 32  # From the Google AI Documentation

MODELS_TO_TEST = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# Pricing Information (USD per 1,000,000 tokens)
PRICING = {
    "gemini-2.0-flash": {"input": 0.70, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},  # Assuming price <= 128K context
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00}  # Assuming price <= 128K context
}

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")


def count_tokens_split(text):
    """Counts tokens using simple text.split()."""
    return len(text.split())


def test_model_and_count_tokens(model_name):
    """Tests a model, returns the response, and estimates total tokens."""
    input_file = {
        "type": "input_audio",
        "input_audio": {"data": AudioEncoder(AUDIO_PATH).encode(), "format": "mp3"},
    }
    input_text = {
        "type": "text",
        "text": "Trascrivi l'audio allegato parola per parola. Fallo in italiano.",
    }

    model = ChatOpenAI(
        model=model_name,
        temperature=0
    )

    print(f"Sending audio to {model_name}\n")

    try:
        response = model.invoke([
            HumanMessage(content=[input_file, input_text])
        ])
        transcription = response.content
    except Exception as e:
        print(f"API invocation failed for {model_name}: {e}")
        return None  # Signal failure

    print("Transcription:\n")
    print(transcription)

    # Calculate audio tokens
    audio_tokens = AUDIO_DURATION * AUDIO_TOKENS_PER_SECOND

    # Estimate tokens for text using split
    input_tokens_split = count_tokens_split(input_text["text"])
    output_tokens_split = count_tokens_split(transcription)

    # Calculate Total tokens
    total_input_tokens_split = audio_tokens + input_tokens_split
    total_output_tokens_split = audio_tokens + output_tokens_split

    tokens_per_second_audio_split = input_tokens_split / AUDIO_DURATION
    tokens_per_second_output_split = output_tokens_split / AUDIO_DURATION

    # Calculate Cost
    input_cost = (total_input_tokens_split / 1_000_000) * PRICING[model_name]["input"]
    output_cost = (total_output_tokens_split / 1_000_000) * PRICING[model_name]["output"]
    total_cost = input_cost + output_cost

    print(f"\nToken Usage and Cost for {model_name}:")
    print(f"  Audio Tokens (fixed): {audio_tokens}")
    print(f"  Input Tokens (ESTIMATED split): {input_tokens_split}")
    print(f"  Output Tokens (ESTIMATED split): {output_tokens_split}")
    print(f"  Total Input Tokens (ESTIMATED split): {total_input_tokens_split}")
    print(f"  Total Output Tokens (ESTIMATED split): {total_output_tokens_split}")
    print(f"  Tokens per second of audio: {AUDIO_TOKENS_PER_SECOND}")
    print(f"  Tokens per second of output (ESTIMATED split): {tokens_per_second_output_split:.2f}")
    print(f"  Estimated Cost: ${total_cost:.6f}")

    return (total_input_tokens_split, total_output_tokens_split, transcription,
            tokens_per_second_audio_split, tokens_per_second_output_split, total_cost)


# Test all models
model_results = {}
total_operation_cost = 0
for model_name in MODELS_TO_TEST:
    result = test_model_and_count_tokens(model_name)
    if result:
        (total_input_tokens_split, total_output_tokens_split, transcription,
         tokens_per_second_audio_split, tokens_per_second_output_split, total_cost) = result
        model_results[model_name] = {
            "input_split": total_input_tokens_split,
            "output_split": total_output_tokens_split,
            "transcription": transcription,
            "tokens_per_second_audio_split": tokens_per_second_audio_split,
            "tokens_per_second_output_split": tokens_per_second_output_split,
            "cost": total_cost
        }
        total_operation_cost += total_cost
    else:
        model_results[model_name] = None

# Print Summary
print("\n--- Summary ---")

for model_name in MODELS_TO_TEST:
    if model_results[model_name]:
        print(f"\nResults for {model_name}:")
        print(f"  Audio Tokens (fixed): {AUDIO_DURATION * AUDIO_TOKENS_PER_SECOND}")
        print(f"  Input Tokens (ESTIMATED split): {model_results[model_name]['input_split']}")
        print(f"  Output Tokens (ESTIMATED split): {model_results[model_name]['output_split']}")
        print(f"  Tokens per second of audio: {AUDIO_TOKENS_PER_SECOND}")
        print(f"  Tokens per second of output (ESTIMATED split): {model_results[model_name]['tokens_per_second_output_split']:.2f}")
        print(f"  Estimated Cost: ${model_results[model_name]['cost']:.6f}")
        print(f"  Transcription: {model_results[model_name]['transcription']}")
    else:
        print(f"\nTest failed for {model_name}.")

print(f"\nTotal estimated costs of the operation: ${total_operation_cost:.6f}")