import os
import json
import dotenv
import requests
import sys

dotenv.load_dotenv()
API_KEY_ENV_VAR = "APIM_AI_DEV_KEY"
BASE_URL_ENV_VAR = "AI_DEV_BASE_URL"
TRANSCRIPTIONS_FILE_PATH = "output/stt/vosk_transcription/transcriptions.json"
MACRO_JSON_DIR = "json"
MACRO_JSON_FILENAME = "macro_complete.json"
OUTPUT_DIR = "output/ner/apim_scraper_out/"

class InformationScraper:
    """Handles interaction with the document analysis API."""
    def __init__(self, api_key_env_var: str, base_url_env_var: str, json_dir: str, json_filename: str, output_dir: str):
        self.authorization = os.getenv(api_key_env_var)
        self.base_url = os.getenv(base_url_env_var)
        self.url = self.base_url.rstrip('/') + "/document-analysis-api/summary_with_report"
        self.json_dir = json_dir
        self.json_filename = json_filename
        self.output_dir = output_dir
        self.base_payload_data = self._load_base_payload()

    def _load_base_payload(self):
        """Loads the base structure of the API payload from the macro file."""
        json_path = os.path.join(self.json_dir, self.json_filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload_data = json.load(f)
                if "document" not in payload_data:
                     payload_data["document"] = ""
                return payload_data
        except FileNotFoundError:
            print(f"Error: Macro file '{self.json_filename}' not found in directory '{self.json_dir}'.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from macro file '{json_path}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred loading the macro file: {e}")
            sys.exit(1)


    def extract_information(self, document_text: str, output_filename: str, original_transcribed_sentence: str, original_sentence: str):
        """
        Sends the document_text to the API and saves the combined result.

        Args:
            document_text (str): The text to send to the API (transcribed sentence).
            output_filename (str): The name for the output JSON file.
            original_transcribed_sentence (str): The original transcribed sentence.
            original_sentence (str): The original sentence.
        """
        headers = {
            'X-Client-Application': 'speech-to-text-benchmark',
            'X-Client-Tenant': 'ai-team',
            'Content-Type': 'application/json',
            'Authorization': self.authorization
        }

        # Update the payload with the current document text
        current_payload_data = self.base_payload_data.copy()
        current_payload_data["document"] = document_text
        payload_json = json.dumps(current_payload_data)

        if not payload_json:
            print("Error: Failed to create JSON payload.")
            return # Continue to next sentence if payload fails

        try:
            response = requests.post(self.url, headers=headers, data=payload_json, timeout=60) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            scraped_data = response.json()

            # Combine scraped data with original sentences
            output_data = {
                "scraped_data": scraped_data,
                "original_transcribed_sentence": original_transcribed_sentence,
                "original_sentence": original_sentence
            }

            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"Output successfully saved to {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error during API request for document '{document_text[:30]}...': {e}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON response from API for document '{document_text[:30]}...'. Response text: {response.text}")
        except Exception as e:
            print(f"An unexpected error occurred during extraction or saving for document '{document_text[:30]}...': {e}")


def load_transcription_data(file_path: str):
    """
    Loads transcription data, extracting index, transcribed, and original sentences,
    filtering out entries where dataset is 'coqui'.

    Args:
        file_path (str): Path to the input JSON file (e.g., transcriptions.json).

    Returns:
        list: A list of tuples, where each tuple contains
              (original_index, transcribed_text, original_text)
              for entries not from the 'coqui' dataset.
              Returns an empty list if the file cannot be processed or no valid entries remain.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input transcription file not found at '{file_path}'.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{file_path}'.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred loading '{file_path}': {e}")
        return []

    processed_data = []
    skipped_coqui_count = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Warning: Skipping item at index {i} because it's not a dictionary.")
            continue

        dataset_name = item.get("dataset")
        if dataset_name == "coqui":
            skipped_coqui_count += 1
            continue # Skip the rest of the loop for this item

        transcribed_sentence_info = item.get('transcribed_sentence', {})
        if not isinstance(transcribed_sentence_info, dict):
            transcribed_text = ''
            print(f"Warning: 'transcribed_sentence' at index {i} (dataset: {dataset_name}) is not a dictionary. Setting text to empty.")
        else:
            transcribed_text = transcribed_sentence_info.get('text', '')

        original_text = item.get('original_sentence', '')

        if not transcribed_text:
            print(f"Warning: Skipping item at index {i} (dataset: {dataset_name}) due to missing or empty transcribed text.")
            continue

        processed_data.append((i, transcribed_text, original_text))

    if skipped_coqui_count > 0:
        print(f"Info: Skipped {skipped_coqui_count} entries because their dataset was 'coqui'.")

    if not processed_data:
         print("No valid transcription entries (excluding 'coqui' dataset) found in the input file.")

    return processed_data


def main():
    """
    Main function to orchestrate the processing of transcribed sentences.
    """
    print("Starting transcription processing...")
    print(f"Loading transcriptions from: {TRANSCRIPTIONS_FILE_PATH}")
    print(f"Using macro: {os.path.join(MACRO_JSON_DIR, MACRO_JSON_FILENAME)}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load all transcription data entries
    transcription_entries = load_transcription_data(TRANSCRIPTIONS_FILE_PATH)

    if not transcription_entries:
        print("No data to process. Exiting.")
        return

    # Initialize the scraper
    scraper = InformationScraper(
        api_key_env_var=API_KEY_ENV_VAR,
        base_url_env_var=BASE_URL_ENV_VAR,
        json_dir=MACRO_JSON_DIR,
        json_filename=MACRO_JSON_FILENAME,
        output_dir=OUTPUT_DIR
    )

    print(f"\nFound {len(transcription_entries)} entries to process.")

    # Process each entry
    for index, transcribed_text, original_text in transcription_entries:
        print(f"\nProcessing sentence index: {index}")
        print(f"  Transcribed: '{transcribed_text[:50]}...'")

        output_filename = f"scraped_sentence_{index}.json"

        scraper.extract_information(
            document_text=transcribed_text,
            output_filename=output_filename,
            original_transcribed_sentence=transcribed_text,
            original_sentence=original_text
        )

    print("\nProcessing finished.")

if __name__ == "__main__":
    main()