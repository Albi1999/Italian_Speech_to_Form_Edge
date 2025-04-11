import os
import json
import dotenv
import requests
from datetime import datetime

class InformationScraper:
    def __init__(self, api_key_env_var: str, base_url_env_var: str, json_dir: str, json_filename: str, document_text: str):
        dotenv.load_dotenv()
        self.authorization = os.getenv(api_key_env_var)
        self.base_url = os.getenv(base_url_env_var)
        self.url = self.base_url + "/document-analysis-api/summary_with_report"
        self.json_dir = json_dir
        self.json_filename = json_filename
        self.document_text = document_text

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

    def extract_information(self):
        headers = {
            'X-Client-Application': 'speech-to-text-benchmark',
            'X-Client-Tenant': 'ai-team',
            'Content-Type': 'application/json',
            'Authorization': self.authorization
        }
        payload = self.load_json_payload()
        if not payload:
            print("Payload is empty or invalid.")
            return

        response = requests.post(self.url, headers=headers, data=payload)
        if response.ok:
            print(response.text)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.json_filename == "macro_complete.json":
                output_filename = f"complete_results_{timestamp}.json"
            elif self.json_filename == "macro_general_info.json":
                output_filename = f"general_info_results_{timestamp}.json"
            elif self.json_filename == "macro_vehicle_info.json":
                output_filename = f"vehicle_info_results_{timestamp}.json"
            elif self.json_filename == "macro_violation_info.json":
                output_filename = f"violation_info_results_{timestamp}.json"
            else:
                output_filename = f"result_{timestamp}.json"

            with open(os.path.join("json/out", output_filename), 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=4)
            print(f"Output saved to {output_filename}")
        else:
            print(f"Error in the request {response.status_code}, {response.text}")

def load_texts(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [item['text'] for item in data]
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
        return []

texts_file_path = "data/synthetic_datasets/metadata/samples.json"
texts = load_texts(texts_file_path)

macros = ["macro_complete.json", "macro_general_info.json", "macro_vehicle_info.json", "macro_violation_info.json"]

for document_text in texts:
    for macro in macros:
        scraper = InformationScraper(
            api_key_env_var="APIM_AI_DEV_KEY",
            base_url_env_var="AI_DEV_BASE_URL",
            json_dir="json",
            json_filename=macro,
            document_text=document_text
        )
        scraper.extract_information()
        print(f"Processed {macro} with document '{document_text[:30]}...' successfully.")