import json
import time
from models import GemmaModel, PhiModel, QwenModel

# Schema JSON
with open("json/macro_complete.json", "r", encoding="utf-8") as f_schema:
    schema_json_str = f_schema.read()

with open("data/synthetic_datasets/new_data/AzureTTS/tx/sentences.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    example_sentences = [item.strip() for i, item in enumerate(json_data) if i < 3 and isinstance(item, str) and item.strip()] # Takes the first 3 sentences

def prompt_definition(sentence_input, schema_json_str_prompt):
    return f"""
Sei un assistente AI specializzato nell'estrazione di informazioni da verbali di polizia stradale in italiano.
Analizza il seguente sentence trascritto e popola un oggetto JSON con le informazioni estratte.
L'output DEVE essere ESCLUSIVAMENTE un oggetto JSON valido, senza sentence aggiuntivo prima o dopo.
Utilizza la seguente struttura e descrizioni dei campi come guida per l'estrazione:

{schema_json_str_prompt}

Testo Trascritto:
"{sentence_input}"

Output JSON Estratto:
"""

quant_config = None 

gpu_torch_dtype = "bfloat16" 

models_config = [
    {
        "name": "Gemma-2B-IT",
        "model_class": GemmaModel,
        "model_id_hf": "google/gemma-2b-it",
        "quant_config": quant_config,
        "torch_dtype": gpu_torch_dtype
    },
    {
        "name": "Phi-3-mini-4k-Instruct",
        "model_class": PhiModel,
        "model_id_hf": "microsoft/Phi-3-mini-4k-instruct",
        "quant_config": quant_config,
        "torch_dtype": gpu_torch_dtype
    },
    {
        "name": "Qwen2-1.5B-Instruct",
        "model_class": QwenModel,
        "model_id_hf": "Qwen/Qwen2-1.5B-Instruct",
        "quant_config": quant_config,
        "torch_dtype": gpu_torch_dtype
    }
]

global_results = {}

for config in models_config:
    print(f"\n{'='*30}\nTesting Model: {config['name']}\n{'='*30}")
    
    current_model_instance = None
    try:
        current_model_instance = config["model_class"](
            model_id=config["model_id_hf"],
            quantization_config=config["quant_config"],
            torch_dtype_str=config["torch_dtype"]
        )
        
        model_results = []
        for i, sentence in enumerate(example_sentences):
            print(f"\n--- Processing sentence {i+1}/{len(example_sentences)} ---")
            print(f"Input: {sentence[:100]}...")

            prompt = prompt_definition(sentence, schema_json_str)
            
            start_inference_time = time.time()
            generated_json_str = current_model_instance.generate(prompt)
            inference_time = time.time() - start_inference_time
            
            print(f"Inference time (only generate()): {inference_time:.2f}s")
            print(f"Output raw JSON:\n{generated_json_str}")

            analysis = {"input_text": sentence, "raw_output": generated_json_str, "parsed_json": None, "is_valid_json": False, "parsing_error": None, "inference_time": inference_time}
            
            if generated_json_str:
                try:
                    parsed = json.loads(generated_json_str)
                    analysis["parsed_json"] = parsed
                    analysis["is_valid_json"] = True
                    print("Valid JSON: yes")

                except json.JSONDecodeError as e:
                    analysis["parsing_error"] = str(e)
                    print(f"Valid JSON: No - Error: {e}")
            else:
                print("Valid JSON: No - Empty output from the model.")
                analysis["parsing_error"] = "Empty output from the model."

            model_results.append(analysis)
        
        global_results[config['name']] = model_results

    except Exception as e:
        print(f"Irreversible erorr while testing model {config['name']}: {e}")
        global_results[config['name']] = [{"error": str(e)}]
        
    finally:
        if current_model_instance:
            current_model_instance.release()

print(f"\n\n{'='*30}\nFINAL SUMMARY COMPARISON\n{'='*30}")

for model_name, model_test_result in global_results.items():
    print(f"\n--- Model: {model_name} ---")
    if isinstance(model_test_result, list) and len(model_test_result) > 0 and "error" in model_test_result[0]:
        print(f"  ERROR DURING TEST: {model_test_result[0]['error']}")
        continue

    valid_json_count = 0
    total_tests = len(model_test_result)
    total_inference_time = 0

    for i, res in enumerate(model_test_result):
        print(f"  Sentence n.{i+1}:")
        print(f"    Input: \"{res['input_text'][:70]}...\"")
        print(f"    Valid JSON: {'yes' if res['is_valid_json'] else 'no'}")
        total_inference_time += res.get('inference_time', 0)
        if res['is_valid_json']:
            valid_json_count += 1
        else:
            print(f"    Parsing Error: {res['parsing_error']}")
            print(f"    Problematic Raw Error (first 300 char): {str(res['raw_output'])[:300]}")
        print(f"    Inference Time: {res.get('inference_time', 0):.2f}s")


    print(f"\n  Stats for {model_name}:")
    print(f"    Valid JSON: {valid_json_count}/{total_tests}")
    if total_tests > 0:
        print(f"    Mean Inference Time: {total_inference_time/total_tests:.2f}s")

print("\nCompleted.")