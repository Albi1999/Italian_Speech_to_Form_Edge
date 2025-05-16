
#!pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#!pip install -U transformers accelerate bitsandbytes sentencepiece huggingface_hub Pillow flash-attn

import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, BitsAndBytesConfig
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
drive_base_path = "/content/drive/MyDrive/data/slm/json/"
print(f"Google Drive base path: {drive_base_path}")
from google.colab import userdata
userdata.get('HF_TOKEN')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class BaseLLMModel:
    def __init__(self, model_id: str, device_name: str, quantization_config=None, torch_dtype_str: str = "auto", trust_remote: bool = False, attn_implementation_str = None):
        self.model_id = model_id
        self.device_name = device_name
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote
        self.attn_implementation = attn_implementation_str

        if self.device_name == "cuda":
            if torch_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            elif torch_dtype_str == "float16":
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float16
        else: # CPU
            self.torch_dtype = torch.float32

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_id}...")
        try:
            load_args = {
                "torch_dtype": self.torch_dtype if not self.quantization_config else None,
                "quantization_config": self.quantization_config,
                "device_map": self.device_name if not self.quantization_config else None,
                "trust_remote_code": self.trust_remote_code
            }
            if self.attn_implementation:
                load_args["attn_implementation"] = self.attn_implementation

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_args
            )

            if not self.quantization_config and hasattr(self.model, "to") and self.device_name not in str(self.model.device):
                 self.model.to(self.device_name)
            elif self.quantization_config and self.device_name not in str(self.model.device) and hasattr(self.model, "hf_device_map"):
                pass

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model {self.model_id}: {e}")
            raise

    def generate(self, prompt_text: str, max_new_tokens: int = 1536, temperature: float = 0.1, top_p: float = 0.9):
        if not self.model or not self.tokenizer: return None
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True).to(self.model.device)
        start_time = time.time()
        try:
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True if temperature > 0.0 else False,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )
            end_time = time.time()
            print(f"Gen time for {self.model_id}: {end_time - start_time:.2f}s")
            full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            json_output_str = ""
            if full_generated_text.startswith(prompt_text):
                 cleaned_output = full_generated_text[len(prompt_text):].strip()
            else:
                cleaned_output = full_generated_text.strip()

            first_brace = cleaned_output.find('{')
            last_brace = cleaned_output.rfind('}')

            final_json_str = ""
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                final_json_str = cleaned_output[first_brace : last_brace+1]
            elif first_brace != -1:
                print(f"Warning: JSON structure seems incomplete or malformed in output for {self.model_id}. Attempting to extract from first brace.")
                final_json_str = cleaned_output[first_brace:]
            else:
                print(f"Warning: No JSON object clearly identified in the output for {self.model_id}. Full cleaned output (first 500 chars): {cleaned_output[:500]}")
                final_json_str = cleaned_output
            return final_json_str
        except Exception as e:
            print(f"Error during generation with {self.model_id}: {e}")
            return None

    def release(self):
        print(f"Releasing resources for model {self.model_id}...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Resources for {self.model_id} released.")

class GemmaModel(BaseLLMModel):
    def __init__(self, model_id: str = "google/gemma-2b-it", device_name: str = device, quantization_config=None, torch_dtype_str: str = "bfloat16"):
        super().__init__(model_id, device_name, quantization_config, torch_dtype_str, trust_remote=False)

class PhiModel(BaseLLMModel):
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct", device_name: str = device, quantization_config=None, torch_dtype_str: str = "bfloat16"):
        super().__init__(model_id, device_name, quantization_config, torch_dtype_str, trust_remote=True, attn_implementation_str="eager")

class QwenModel(BaseLLMModel):
    def __init__(self, model_id: str = "Qwen/Qwen2-1.5B-Instruct", device_name: str = device, quantization_config=None, torch_dtype_str: str = "bfloat16"):
        super().__init__(model_id, device_name, quantization_config, torch_dtype_str, trust_remote=True)

schema_descriptive_file_name = "slm_schema.json"
sentences_file_name = "sentences.json"
example_json_file_name = "example_output.json"
blank_json_template_file_name = "blank_output.json"

path_to_schema_description = os.path.join(drive_base_path, schema_descriptive_file_name)
path_to_sentences = os.path.join(drive_base_path, sentences_file_name)
path_to_example_json = os.path.join(drive_base_path, example_json_file_name)
path_to_blank_json_template = os.path.join(drive_base_path, blank_json_template_file_name)

schema_description_str = None
input_sentences = []
example_json_str_for_prompt = None
blank_json_template_str_for_prompt = None

try:
    with open(path_to_schema_description, "r", encoding="utf-8") as f_schema:
        schema_description_str = f_schema.read()
except FileNotFoundError:
    print(f"ERROR: Schema description file '{path_to_schema_description}' not found in Google Drive.")
except Exception as e:
    print(f"ERROR while reading the schema description file from Drive: {e}")

try:
    with open(path_to_sentences, "r", encoding="utf-8") as f_sentences:
        json_data_sentences = json.load(f_sentences)

    if isinstance(json_data_sentences, list):
        input_sentences = [
            item.strip() for i, item in enumerate(json_data_sentences)
            if isinstance(item, str) and item.strip()
        ]
        input_sentences = input_sentences[:3]
    else:
        print(f"WARNING: File '{path_to_sentences}' does not contain a JSON list as expected.")

    if not input_sentences:
        print(f"WARNING: No input sentences loaded from '{path_to_sentences}'. Check the file and loading logic.")
    else:
        print(f"Input sentences loaded successfully from: {path_to_sentences}")
        print(f"Number of sentences to process: {len(input_sentences)}")

except FileNotFoundError:
    print(f"ERROR: Sentences file '{path_to_sentences}' not found in Google Drive.")
except json.JSONDecodeError:
    print(f"ERROR: Sentences file '{path_to_sentences}' is not a valid JSON.")
except Exception as e:
    print(f"ERROR while reading the sentences file from Drive: {e}")

try:
    with open(path_to_example_json, "r", encoding="utf-8") as f_example:
        loaded_example_json_object = json.load(f_example)
    example_json_str_for_prompt = json.dumps(loaded_example_json_object, indent=4, ensure_ascii=False)
    print(f"Example JSON for prompt loaded and processed from: {path_to_example_json}")
except FileNotFoundError:
    print(f"ERROR: Example JSON file '{path_to_example_json}' not found in Google Drive.")
except json.JSONDecodeError:
    print(f"ERROR: Example JSON file '{path_to_example_json}' is not a valid JSON.")
except Exception as e:
    print(f"ERROR while reading/processing the example JSON from Drive: {e}")

try:
    with open(path_to_blank_json_template, "r", encoding="utf-8") as f_blank:

        loaded_blank_json_object = json.load(f_blank)
    blank_json_template_str_for_prompt = json.dumps(loaded_blank_json_object, indent=4, ensure_ascii=False)
    print(f"Blank JSON template for prompt loaded and processed from: {path_to_blank_json_template}")
except FileNotFoundError:
    print(f"ERROR: Blank JSON template file '{path_to_blank_json_template}' not found in Google Drive.")
except json.JSONDecodeError:
    print(f"ERROR: Blank JSON template file '{path_to_blank_json_template}' is not a valid JSON.")
except Exception as e:
    print(f"ERROR while reading/processing the blank JSON template from Drive: {e}")

if not all([schema_description_str, input_sentences, example_json_str_for_prompt, blank_json_template_str_for_prompt]):
    print(("\nCRITICAL ERROR: One or more files required for the prompt "
           "(schema description, sentences, example JSON, blank JSON template) "
           "were not loaded correctly. Subsequent cells may fail."))
else:
    print("\nAll data for the prompt (schema description, sentences, example JSON, blank JSON template) are ready.")

def prompt_definition(user_sentence_input, example_json_output_str, blank_json_template_str):
    return f"""
Sei un assistente AI specializzato nell'estrazione di informazioni da verbali di polizia stradale in italiano.
Analizza il testo trascritto fornito di seguito ("Testo Trascritto Effettivo") e popola un oggetto JSON.
L'output DEVE essere ESCLUSIVAMENTE un oggetto JSON valido, senza testo aggiuntivo prima o dopo.

Utilizza la seguente descrizione dei campi come guida per l'estrazione. Presta attenzione ai tipi di dato e ai valori possibili indicati:
- verbale_preavviso: (testo) Indica se si vuole redigere un verbale o preavviso. Valori possibili: ["verbale", "preavviso"].
- data_violazione: (data) Data della violazione. Formato: "gg/mm/aaaa".
- ora_violazione: (testo) Ora della violazione. Formato: "hh:mm".
- Strada_1: (indirizzo) Indirizzo della violazione (solo la via, non anche il civico).
- Civico_1: (indirizzo) Civico della violazione.
- veicolo: Dettagli del veicolo.
    - tipologia: (testo) Tipo di veicolo. Valori possibili: ["ciclomotore", "motoveicolo", "autovettura", "rimorchio", "macchina agricola", "macchina operatrice"].
    - nazione: (testo) Nazione del veicolo.
    - targa: (testo) Targa del veicolo.
    - tipologia_targa: (testo) Tipologia della targa del veicolo. Valori possibili: ["ufficiale", "speciale", "prova", "copertura"].
    - marca_modello: (testo) Marca e modello del veicolo.
    - colore: (testo) Colore del veicolo.
    - telaio: (testo) Numero di telaio del veicolo.
    - massa: (testo) Massa del veicolo.
    - note_veicolo: (testo) Note aggiuntive del veicolo. Esempio: "veicolo in divieto di sosta".
- contestazione_immediata: (testo) Indica se la contestazione è stata immediata oppure no. Valori possibili: ["contestazione immediata", "contestazione non immediata"].
- motivo_mancata_contestazione: (testo) Le motivazioni per cui la contestazione non è stata immediata.
- violazione: Dettagli della violazione.
    - codice: (testo) codice di legge infranto. Valori possibili: ["civile", "penale", "stradale"].
    - articolo: (testo) articolo di leggge violato.
    - comma: (testo) comma dell'articolo violato.
    - sanzione_accessoria: (testo) presenza o meno di una sanzione accessoria. Valori possibili: ["con sanzione accessoria", "senza sanzione accessoria"].
- punti: (testo) punti decurtati.
- tipo_stampa: (testo) come vuole essere stampato il documento. Valori possibili: ["bluetooth", "wifi"].
- lingua_stampa: (testo) lingua da utilizzare per stampare il documento.
- stampa_anche_comunicazione: (testo) stampa anche comunicazione. Valori possibili: ["stampa anche la comunicazione", "non stampare anche la comunicazione"].

I valori nell'output JSON devono provenire ESCLUSIVAMENTE dal "Testo Trascritto Effettivo", devono rispettare le regole sui valori possibili presenti nello schema precedente quindi se un dato risulta in una categoria con la sezione "Valori possibili" il valore che vai a prendere e mettere nel JSON finale deve essere ESCLUSIVAMENTE uno di quelli nei "Valori possibili"
Se un'informazione per un campo non è presente nel "Testo Trascritto Effettivo", utilizza il valore "null" per quel campo nel JSON. Non inventare o aggiungere informazioni.

Per costruire l'output utilizza questo JSON come modello di base, popola i campi utilizzando le informazioni che trovi SOLO nel "Testo Trascritto Effettivo":
{blank_json_template_str}


{'-'*20}
Testo Trascritto Effettivo (da cui estrarre le informazioni):
"{user_sentence_input}"
{'-'*20}

Output JSON Estratto (basato ESCLUSIVAMENTE sul "Testo Trascritto Effettivo" qui sopra):
"""

active_quant_config = None # Set to bnb_config_4bit to use quantization
gpu_torch_dtype_selection = "bfloat16"

models_to_test_config = [
    {
        "display_name": "Gemma-2B-IT",
        "model_class_ref": GemmaModel,
        "model_id_hf": "google/gemma-2b-it",
        "quant_config": active_quant_config,
        "torch_dtype_str_arg": gpu_torch_dtype_selection
    },
    {
        "display_name": "Gemma-3-1B-IT",
        "model_class_ref": GemmaModel,
        "model_id_hf": "google/gemma-3-1b-it",
        "quant_config": active_quant_config,
        "torch_dtype_str_arg": gpu_torch_dtype_selection
    },
    {
        "display_name": "Phi-4-mini-Instruct",
        "model_class_ref": PhiModel,
        "model_id_hf": "microsoft/Phi-4-mini-instruct",
        "quant_config": active_quant_config,
        "torch_dtype_str_arg": gpu_torch_dtype_selection
    },
    {
        "display_name": "Qwen2-1.5B-Instruct",
        "model_class_ref": QwenModel,
        "model_id_hf": "Qwen/Qwen2-1.5B-Instruct",
        "quant_config": active_quant_config,
        "torch_dtype_str_arg": gpu_torch_dtype_selection
    },
    {
        "display_name": "Qwen3-0.6B",
        "model_class_ref": QwenModel,
        "model_id_hf": "Qwen/Qwen3-0.6B",
        "quant_config": active_quant_config,
        "torch_dtype_str_arg": gpu_torch_dtype_selection
    },
    {
        "display_name": "Qwen3-1.7B",
        "model_class_ref": QwenModel,
        "model_id_hf": "Qwen/Qwen3-1.7B",
        "quant_config": active_quant_config,
        "torch_dtype_str_arg": gpu_torch_dtype_selection
    }
]

all_models_results = {}

if not input_sentences:
    print("\nNo input sentences available. Stopping the model comparison.")
elif not all([schema_description_str, example_json_str_for_prompt, blank_json_template_str_for_prompt]):
    print("\nEssential prompt components (schema description, example JSON, or blank template) are missing. Stopping model comparison.")
else:
    for model_config in models_to_test_config:
        model_display_name = model_config['display_name']
        print(f"\n{'='*40}\nTesting Model: {model_display_name} ({model_config['model_id_hf']})\n{'='*40}")

        current_model_instance = None
        try:
            current_model_instance = model_config["model_class_ref"](
                model_id=model_config["model_id_hf"],
                device_name=device,
                quantization_config=model_config["quant_config"],
                torch_dtype_str=model_config["torch_dtype_str_arg"]
            )

            current_model_individual_results = []
            for i, sentence_text in enumerate(input_sentences):
                print(f"\n--- Processing sentence {i+1}/{len(input_sentences)} ---")
                print(f"Input: {sentence_text}")

                current_prompt = prompt_definition(
                    sentence_text,
                    example_json_str_for_prompt,
                    blank_json_template_str_for_prompt
                )

                generated_json_string = current_model_instance.generate(current_prompt)
                print(f"Full Raw JSON output:\n{str(generated_json_string)}") # For debugghing

                result_analysis = {
                    "input_sentence_text": sentence_text,
                    "raw_model_output": str(generated_json_string),
                    "parsed_json_object": None,
                    "is_valid_json_output": False,
                    "json_parsing_error_message": None
                }

                if generated_json_string and generated_json_string.strip():
                    try:
                        clean_json_string = generated_json_string
                        if generated_json_string.startswith("```json"):
                            clean_json_string = generated_json_string.split("```json", 1)[1].strip()
                        if clean_json_string.endswith("```"):
                            clean_json_string = clean_json_string[:-3].strip()

                        first_brace_idx = clean_json_string.find('{')
                        last_brace_idx = clean_json_string.rfind('}')
                        if first_brace_idx != -1 and last_brace_idx != -1 and last_brace_idx > first_brace_idx:
                             string_to_parse = clean_json_string[first_brace_idx : last_brace_idx+1]
                        else:
                             string_to_parse = clean_json_string

                        parsed_object = json.loads(string_to_parse)
                        result_analysis["parsed_json_object"] = parsed_object
                        result_analysis["is_valid_json_output"] = True
                        print("Valid JSON: Yes")

                    except json.JSONDecodeError as e_json_decode:
                        result_analysis["json_parsing_error_message"] = str(e_json_decode)
                        print(f"Valid JSON: No - JSONDecodeError: {e_json_decode}")
                    except Exception as e_general_parsing:
                        result_analysis["json_parsing_error_message"] = f"Unexpected error during JSON parsing: {str(e_general_parsing)}"
                        print(f"Valid JSON: No - Unexpected Parsing Error: {e_general_parsing}")
                else:
                    print("Valid JSON: No - Empty or whitespace output from the model.")
                    result_analysis["json_parsing_error_message"] = "Empty or whitespace output from the model."

                current_model_individual_results.append(result_analysis)

            all_models_results[model_display_name] = current_model_individual_results

        except Exception as e_model_suite_error:
            print(f"Irrecoverable error while testing model {model_display_name}: {e_model_suite_error}")
            import traceback
            traceback.print_exc()
            all_models_results[model_display_name] = [{"error_message": str(e_model_suite_error), "traceback_info": traceback.format_exc()}]

        finally:
            if current_model_instance:
                print(f"Attempting to release resources for {model_display_name}...")
                current_model_instance.release()
            if device == "cuda": # Explicitly clear cache after each model in Colab
                torch.cuda.empty_cache()
                print("CUDA cache cleared after model test run.")

    print(f"\n\n{'='*30}\nFINAL COMPARISON SUMMARY\n{'='*30}")
    for model_name_report, model_test_results_list in all_models_results.items():
        print(f"\n--- Model: {model_name_report} ---")
        if isinstance(model_test_results_list, list) and len(model_test_results_list) > 0 and "error_message" in model_test_results_list[0]:
            print(f"  ERROR DURING MODEL TEST SUITE: {model_test_results_list[0]['error_message']}")
            if "traceback_info" in model_test_results_list[0]:
                 print(f"    Traceback (partial): {model_test_results_list[0]['traceback_info'][:500]}...")
            continue

        num_valid_json = 0

        if not isinstance(model_test_results_list, list):
            print(f"  Unexpected result format for model {model_name_report}")
            continue

        num_total_tests = len(model_test_results_list)
        if num_total_tests == 0:
            print("  No test results recorded for this model.")
            continue

        for i, res_item in enumerate(model_test_results_list):
            print(f"  Sentence Test #{i+1}:")
            print(f"    Input: \"{res_item['input_sentence_text'][:70]}...\"")
            print(f"    Valid JSON Generated: {'Yes' if res_item['is_valid_json_output'] else 'No'}")
            if res_item['is_valid_json_output']:
                num_valid_json += 1
            else:
                print(f"    JSON Parsing Error: {res_item['json_parsing_error_message']}")
                problematic_raw_output = str(res_item.get('raw_model_output', ''))[:300]
                print(f"    Problematic Raw Output (first 300 chars): {problematic_raw_output}")

        print(f"\n  Statistics for {model_name_report}:")
        print(f"    Valid JSON outputs: {num_valid_json}/{num_total_tests}")

print("\nComparison script completed.")


