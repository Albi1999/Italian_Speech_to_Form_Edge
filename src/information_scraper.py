import os
import json
import dotenv
import requests
import random

class InformationScraper:
    def __init__(self, api_key_env_var: str, base_url_env_var: str, json_dir: str, json_filename: str, document_text: str, output_dir: str):
        dotenv.load_dotenv()
        self.authorization = os.getenv(api_key_env_var)
        self.base_url = os.getenv(base_url_env_var)
        self.url = self.base_url + "/document-analysis-api/summary_with_report"
        self.json_dir = json_dir
        self.json_filename = json_filename
        self.document_text = document_text
        self.output_dir = output_dir

    def load_json_payload(self):
        json_path = os.path.join(self.json_dir, self.json_filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload_data = json.load(f)
                payload_data["document"] = self.document_text
                return json.dumps(payload_data)
        except FileNotFoundError:
            print(f"File {self.json_filename} not found in directory {self.json_dir}.")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON from the file.")
            return None

    def extract_information(self, output_filename: str):
        """
        Extracts information and saves it to a JSON file.

        Args:
            output_filename (str): The name of the output JSON file.
        """
        headers = {
            'X-Client-Application': 'speech-to-text-benchmark',
            'X-Client-Tenant': 'ai-team',
            'Content-Type': 'application/json',
            'Authorization': self.authorization
        }
        payload = self.load_json_payload()
        if not payload:
            print("Payload is empty or invalid.")
            return None

        response = requests.post(self.url, headers=headers, data=payload)
        if response.ok:
            try:
                with open(os.path.join(self.output_dir, output_filename), 'w', encoding='utf-8') as f:
                    json.dump(response.json(), f, indent=4, ensure_ascii=False)
                print(f"Output saved to {os.path.join(self.output_dir, output_filename)}")
            except Exception as e:
                print(f"Error saving to JSON: {e}")
        else:
            print(f"Error in the request {response.status_code}, {response.text}")


def load_texts(file_path: str, num_sentences: int = None):
    """Loads texts from a JSON file, with an optional limit.

    Args:
        file_path (str): Path to the JSON file.
        num_sentences (int, optional): Maximum number of sentences to load.
            Defaults to None (load all).

    Returns:
        list: A list of text strings, or an empty list on error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_texts = [(i, item['text']) for i, item in enumerate(data)]
            if num_sentences is not None and 0 < num_sentences <= len(all_texts):
                return random.sample(all_texts, num_sentences)
            return all_texts
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
        return []
    except KeyError:
        print("Error: 'text' key not found in JSON data.")
        return []


def helper_load_texts(file_path: str):
    """Loads texts from a JSON file.  Helper function to avoid code duplication.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [(i, item) for i, item in enumerate(data)]  # Return list of (index, item)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
        return []

def load_transcribed_texts(file_path: str, selected_indices: list = None):
    """Loads transcribed texts from a JSON file, optionally filtered by index.

    Args:
        file_path (str): Path to the JSON file.
        selected_indices (list, optional): A list of indices to load. If None, load all.

    Returns:
        list: A list of tuples (original_index, text), or an empty list on error.
    """
    data = helper_load_texts(file_path)
    if not data:
        return []

    texts = [(i, item.get('transcribed_sentence', {}).get('text', '')) for i, item in data]
    valid_texts = [item for item in texts if item[1]]

    if selected_indices:
        return [item for item in valid_texts if item[0] in selected_indices]
    return valid_texts


def load_original_texts(file_path: str, selected_indices: list = None):
    """Loads original texts from a JSON file, optionally filtered by index.

    Args:
        file_path (str): Path to the JSON file.
        selected_indices (list, optional): A list of indices to load. If None, load all.

    Returns:
        list: A list of tuples (original_index, text), or an empty list on error.
    """
    data = helper_load_texts(file_path)
    if not data:
        return []

    texts = [(i, item.get('original_sentence', '')) for i, item in data]
    valid_texts = [item for item in texts if item[1]]

    if selected_indices:
        return [item for item in valid_texts if item[0] in selected_indices]
    return valid_texts


def main(num_sentences=None, all_macros=False, transcribed_texts=False):
    """
    Main function to run the information scraping process.

    Args:
        num_sentences (int, optional): Number of sentences to process.
            If None, process all sentences.
        all_macros (bool, optional): Flag to indicate whether to process all macros.
            Defaults to False.
        transcribed_texts (bool, optional): Flag to indicate whether to use transcribed texts.
            Defaults to False.
    """
    # Load macros
    if all_macros:
        # Load all macros to compare the performance between a complete and a partial macro
        macros = ["macro_complete.json", "macro_general_info.json", "macro_vehicle_info.json", "macro_violation_info.json"]
    else:
        # Load only the complete macro for the benchmark
        macros = ["macro_complete.json"]

    if transcribed_texts:
        # Load transcribed texts
        transcribed_texts_path = "output/stt/vosk_transcription/transcriptions.json"
        all_transcribed_data = load_transcribed_texts(transcribed_texts_path)  # Load all data first
        all_original_data = load_original_texts(transcribed_texts_path) # Load all original texts
        
        if not all_transcribed_data or not all_original_data or len(all_transcribed_data) != len(all_original_data):
            print("Error: Could not load transcribed or original texts, or lists have different lengths. Exiting.")
            return

        if num_sentences is not None and 0 < num_sentences <= len(all_transcribed_data):
            selected_indices = random.sample(range(len(all_transcribed_data)), num_sentences)
            transcribed_texts_list = [all_transcribed_data[i] for i in selected_indices]
            original_texts_list = [all_original_data[i] for i in selected_indices]
        else:
            transcribed_texts_list = all_transcribed_data
            original_texts_list = all_original_data

        # Process both transcribed and original texts
        for (transcribed_index, transcribed_text), (original_index, original_text) in zip(transcribed_texts_list, original_texts_list):
            output_dir = os.path.join("output/ner/apim_scraper_out/", f"sentence_{str(original_index)}")
            os.makedirs(output_dir, exist_ok=True)

            for macro in macros:
                # Scrape with transcribed text
                scraper_transcribed = InformationScraper(
                    api_key_env_var="APIM_AI_DEV_KEY",
                    base_url_env_var="AI_DEV_BASE_URL",
                    json_dir="json",
                    json_filename=macro,
                    document_text=transcribed_text,
                    output_dir=output_dir
                )
                print(f"Processing {macro} with transcribed text: '{transcribed_text[:30]}...'")
                scraper_transcribed.extract_information(output_filename=f"transcript_scraped_{str(original_index)}.json")

                # Scrape with original text
                scraper_original = InformationScraper(
                    api_key_env_var="APIM_AI_DEV_KEY",
                    base_url_env_var="AI_DEV_BASE_URL",
                    json_dir="json",
                    json_filename=macro,
                    document_text=original_text,
                    output_dir=output_dir
                )
                print(f"Processing {macro} with original text: '{original_text[:30]}...'")
                scraper_original.extract_information(output_filename=f"original_scraped_{str(original_index)}.json")
    else:
        # Load regular texts
        texts_file_path = "data/synthetic_datasets/metadata/samples.json"
        texts = load_texts(texts_file_path, num_sentences)
        for original_index, document_text in texts:
            output_dir = os.path.join("output/ner/apim_scraper_out/", f"sentence_{str(original_index)}")
            os.makedirs(output_dir, exist_ok=True)

            for macro in macros:
                scraper = InformationScraper(
                    api_key_env_var="APIM_AI_DEV_KEY",
                    base_url_env_var="AI_DEV_BASE_URL",
                    json_dir="json",
                    json_filename=macro,
                    document_text=document_text,
                    output_dir=output_dir
                )
                scraper.extract_information(output_filename=f"original_scraped_{str(original_index)}.json")
                print(f"Processed {macro} with document text: '{document_text[:30]}...'")

if __name__ == "__main__":
    main(num_sentences=10, transcribed_texts=True, all_macros=False)