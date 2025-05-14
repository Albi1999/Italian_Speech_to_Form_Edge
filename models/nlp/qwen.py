# models/nlp/qwen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class QwenModel:
    def __init__(self, model_id: str = "Qwen/Qwen2-1.5B-Instruct", 
                 device: str = None, 
                 quantization_config=None,
                 torch_dtype_str: str = "auto"):
        
        self.model_id = model_id
        self.device_name = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization_config = quantization_config

        if self.device_name == "cuda":
            if torch_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
            elif torch_dtype_str == "float16":
                self.torch_dtype = torch.float16
            else: # "auto" o altro
                self.torch_dtype = torch.bfloat16
        else: # CPU
            self.torch_dtype = torch.float32
            if quantization_config is not None:
                 print("Warning: BitsAndBytes quantization is typically for CUDA.")
        
        print(f"Initializing QwenModel ({self.model_id}) on {self.device_name} with dtype {self.torch_dtype}")

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_id}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype if not self.quantization_config else None,
                quantization_config=self.quantization_config,
                device_map=self.device_name if not self.quantization_config else None,
                trust_remote_code=True
            )
            if not self.quantization_config and hasattr(self.model, "to") and self.device_name not in str(self.model.device):
                 self.model.to(self.device_name)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Tokenizer pad_token set to eos_token: {self.tokenizer.eos_token}")

            print(f"Model {self.model_id} and tokenizer loaded successfully.")
            if self.model:
                 print(f"Model is on device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model {self.model_id}: {e}")
            raise

    def generate(self, prompt_text: str, max_new_tokens: int = 1536, temperature: float = 0.1, top_p: float = 0.9):
        if not self.model or not self.tokenizer:
            print("Model or tokenizer not loaded.")
            return None
        print(f"Generating text on device: {self.model.device} for prompt (first 100 chars): {prompt_text[:100]}...")

        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True).to(self.model.device)
        
        start_time = time.time()
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0.0 else False,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )
            end_time = time.time()
            print(f"Text generation time for {self.model_id}: {end_time - start_time:.2f} seconds")

            full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if full_generated_text.startswith(prompt_text):
                 json_output_str = full_generated_text[len(prompt_text):].strip()
            else:
                potential_json_start = full_generated_text.rfind(prompt_text[-50:])
                if potential_json_start != -1 and len(prompt_text) > 50:
                    json_output_str = full_generated_text[potential_json_start + 50:].strip()
                else: 
                    json_output_str = full_generated_text.strip()

            first_brace = json_output_str.find('{')
            last_brace = json_output_str.rfind('}')
            
            final_json_str = ""
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                final_json_str = json_output_str[first_brace : last_brace+1]
            elif first_brace != -1: 
                print("Warning: JSON closing brace not found, attempting to extract from first opening brace.")
                final_json_str = json_output_str[first_brace:]
            else:
                print(f"Warning: No JSON object clearly found in the generated text for {self.model_id}.")
                final_json_str = json_output_str
            return final_json_str
        except Exception as e:
            print(f"Error during text generation with {self.model_id}: {e}")
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