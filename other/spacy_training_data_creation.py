import json
import os
import re
import string

from thefuzz import process, fuzz

TRANSCRIPTIONS_FILE = 'output/stt/vosk_transcription/transcriptions.json'
SCRAPED_TRANSCRIPTS_BASE_DIR = 'output/ner/apim_scraper_out/'
OUTPUT_TRAINING_DATA_PATH = 'data/ner/spacy_training'
os.makedirs(OUTPUT_TRAINING_DATA_PATH, exist_ok=True)
OUTPUT_TRAINING_DATA_FILE = os.path.join(OUTPUT_TRAINING_DATA_PATH, 'training_data.jsonl')

FIELD_TO_LABEL_MAP = {
    "verbale_preavviso": "TIPO_DOCUMENTO",
    "data_violazione": "DATA_VIOLAZIONE",
    "ora_violazione": "ORA_VIOLAZIONE",
    "data_verbalizzazione": "DATA_VERBALIZZAZIONE",
    "ora_verbalizzazione": "ORA_VERBALIZZAZIONE",
    "luogo_verbalizzazione": "LUOGO_VERBALIZZAZIONE",
    "Strada_1": "STRADA_VIOLAZIONE",
    "Civico_1": "CIVICO_VIOLAZIONE",
    # --- Vehicle specific fields ---
    "tipologia": "VEICOLO_TIPOLOGIA",
    "nazione": "VEICOLO_NAZIONE",
    "targa": "VEICOLO_TARGA",
    "tipologia_targa": "VEICOLO_TIPOLOGIA_TARGA",
    "marca_modello": "VEICOLO_MARCA_MODELLO",
    "colore": "VEICOLO_COLORE",
    "telaio": "VEICOLO_TELAIO",
    "massa": "VEICOLO_MASSA",
    "note_veicolo": "VEICOLO_NOTE",
    # --- Other fields ---
    "contestazione_immediata": "CONTESTAZIONE",
    "motivo_mancata_contestazione": "MOTIVO_MANCATA_CONTESTAZIONE",
    # --- Violation specific fields ---
    "codice": "VIOLAZIONE_CODICE",
    "articolo": "VIOLAZIONE_ARTICOLO",
    "comma": "VIOLAZIONE_COMMA",
    "sanzione_accessoria": "VIOLAZIONE_SANZIONE_ACCESSORIA",
    # --- Remaining fields ---
    "punti": "PUNTI_DECURTATI",
    "dichiarazioni": "DICHIARAZIONI",
    "tipo_stampa": "TIPO_STAMPA",
    "lingua_stampa": "LINGUA_STAMPA",
    "stampa_anche_comunicazione": "STAMPA_COMUNICAZIONE",
}

def get_scraped_filepath(original_filepath, base_scraped_dir):
    """
    Constructs the path to the scraped JSON file based on the original audio path
    and the new directory structure `(output/ner/apim_scraper_out/sentence_X/)`.
    """
    if not original_filepath:
        return None
    base_name = os.path.basename(original_filepath)
    match = re.search(r'[_\-](\d+)[_\.]', base_name)
    if not match:
         match = re.search(r'(\d+)\.(wav|json)', base_name, re.IGNORECASE)
         if not match:
              print(f"Warning: Could not extract sentence number from '{base_name}' using known patterns.")
              return None

    sentence_number = match.group(1) # Extract the number
    scraped_filename = f"transcript_scraped_{sentence_number}.json"
    sentence_folder = f"sentence_{sentence_number}"
    full_path = os.path.normpath(os.path.join(base_scraped_dir, sentence_folder, scraped_filename))
    return full_path

def normalize_text_basic(text):
    """Basic normalization: lowercase and remove punctuation for matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep spaces and digits, remove other punctuation
    # Keep hyphens as they might be relevant in some values like plate types or models
    translator = str.maketrans('', '', string.punctuation.replace('-', ''))
    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

def find_span_fuzzy(text_to_search, value_to_find, score_threshold=85, scorer=fuzz.WRatio):
    """
    Finds the best fuzzy match span using iterative substring comparison.

    Args:
        text_to_search (str): The text containing potential entities.
        value_to_find (str): The entity value string to search for.
        score_threshold (int): Minimum fuzz ratio (0-100) to accept a match.
        scorer (function): The scoring function from thefuzz.fuzz to use.

    Returns:
        tuple: (start_char, end_char) if a good match is found, otherwise None.
    """
    if not isinstance(value_to_find, str) or not value_to_find or not isinstance(text_to_search, str) or not text_to_search:
        return None

    normalized_value = normalize_text_basic(value_to_find)
    if not normalized_value:
        return None

    best_score = -1
    best_span = None
    value_len = len(normalized_value)

    # Define window size tolerance (e.g., +/- 30% length, minimum 3 chars)
    len_tolerance = max(3, int(value_len * 0.3))
    min_len = max(1, value_len - len_tolerance)
    max_len = value_len + len_tolerance

    # Iterate through possible start positions
    for i in range(len(text_to_search)):
        # Iterate through possible end positions within the window size limits
        for j in range(i + min_len, min(i + max_len + 1, len(text_to_search) + 1)):
            substring = text_to_search[i:j]
            normalized_substring = normalize_text_basic(substring)

            if not normalized_substring:
                continue

            current_score = scorer(normalized_value, normalized_substring)

            if current_score > best_score:
                best_score = current_score
                # If this is the best score so far, store its span
                if current_score >= score_threshold:
                    best_span = (i, j) # Store the span from the original text

    # Return the span only if the best score met the threshold
    if best_span and best_score >= score_threshold:
         # print(f"Found fuzzy match: '{value_to_find}' matched '{text_to_search[best_span[0]:best_span[1]]}' with score {best_score}")
         return best_span
    else:
        # print(f"Debug: No fuzzy match found for '{value_to_find}' above threshold {score_threshold}. Best score: {best_score}")
        return None

def process_report_dict(report_dict, text_to_search, current_entities_set):
    """
    Recursively processes the report dictionary.
    Uses a set `current_entities_set` to store (start, end, label) tuples
    and naturally handle duplicate additions.

    Returns:
        list of spans found *at this level* to help avoid re-matching.
    """
    processed_spans_in_level = []

    if not isinstance(report_dict, dict):
        return processed_spans_in_level

    for key, value in report_dict.items():
        if key == "lista_veicoli" and isinstance(value, list):
            for item_dict in value:
                processed_spans_in_level.extend(
                    process_report_dict(item_dict, text_to_search, current_entities_set)
                )
        elif key == "violazioni" and isinstance(value, list):
             for item_dict in value:
                 processed_spans_in_level.extend(
                     process_report_dict(item_dict, text_to_search, current_entities_set)
                 )
        elif key in FIELD_TO_LABEL_MAP and value is not None:
            label = FIELD_TO_LABEL_MAP[key]
            value_str = str(value)
            if value_str:
                # Avoid searching again if this value was processed in a nested call
                already_found_in_nested = any(
                    s for s in processed_spans_in_level if s[2] == label and text_to_search[s[0]:s[1]].strip() == value_str.strip()
                    )
                if already_found_in_nested:
                    continue

                span = find_span_fuzzy(text_to_search, value_str, score_threshold=80) # Lowered threshold slightly

                if span:
                    start, end = span
                    entity_tuple = (start, end, label)

                    # Check for overlap before adding
                    is_overlapping = False
                    for ex_start, ex_end, ex_label in current_entities_set:
                        if max(start, ex_start) < min(end, ex_end): # Any overlap
                             # Allow identical spans only if labels are different
                             if not (start == ex_start and end == ex_end and label != ex_label):
                                 # Check if one contains the other - allow if labels differ (e.g., address vs street)
                                 if not ((start >= ex_start and end <= ex_end and label != ex_label) or \
                                         (ex_start >= start and ex_end <= end and label != ex_label)):
                                    print(f"Warning: Span overlap detected for '{value_str}' [{start}:{end}/{label}] with existing entity [{ex_start}:{ex_end}/{ex_label}]. Skipping.")
                                    is_overlapping = True
                                    break
                    if not is_overlapping:
                         current_entities_set.add(entity_tuple)
                         # Track span found at this level (even if already in main set)
                         processed_spans_in_level.append(entity_tuple)
                else:
                     print(f"Warning: Could not find fuzzy span for Label '{label}', Value '{value_str}' in text.")

        elif isinstance(value, dict): # Recursively handle other nested dictionaries
             processed_spans_in_level.extend(
                 process_report_dict(value, text_to_search, current_entities_set)
             )

    return processed_spans_in_level

spacy_training_data = []

print(f"Loading transcriptions from: {TRANSCRIPTIONS_FILE}")
# Load transcriptions
try:
    with open(TRANSCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)
except FileNotFoundError:
    print(f"Error: Transcription file not found at {TRANSCRIPTIONS_FILE}")
    exit()
except json.JSONDecodeError:
     print(f"Error: Could not decode JSON from {TRANSCRIPTIONS_FILE}")
     exit()

print(f"Found {len(transcriptions)} transcriptions.")

# Ensure base scraped directory exists
if not os.path.isdir(SCRAPED_TRANSCRIPTS_BASE_DIR):
     print(f"Error: Base scraped transcripts directory not found at {SCRAPED_TRANSCRIPTS_BASE_DIR}")
     exit()

processed_count = 0
skipped_count = 0

for i, record in enumerate(transcriptions):
    original_filepath = record.get("file_path")
    transcribed_sentence_data = record.get("transcribed_sentence")

    if not original_filepath or not transcribed_sentence_data:
        skipped_count += 1
        continue

    transcribed_text = transcribed_sentence_data.get("text")
    if not transcribed_text:
         skipped_count += 1
         continue

    scraped_filepath = get_scraped_filepath(original_filepath, SCRAPED_TRANSCRIPTS_BASE_DIR)

    if scraped_filepath is None: # Handle case where number extraction failed
        skipped_count += 1
        continue

    if not os.path.exists(scraped_filepath):
        skipped_count += 1
        continue

    try:
        with open(scraped_filepath, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        report_data = scraped_data.get("report")
        if not report_data:
            skipped_count += 1
            continue
    except Exception as e:
        skipped_count += 1
        continue

    entities_set = set() # Use a set to automatically handle duplicates
    process_report_dict(report_data, transcribed_text, entities_set)

    # Convert set of tuples to list of tuples and sort
    entities_list = sorted(list(entities_set), key=lambda x: x[0])

    # Filter out invalid spans again just in case
    valid_entities = [ent for ent in entities_list if isinstance(ent[0], int) and isinstance(ent[1], int)]

    # Format for spaCy
    # Ensure the second element is a dictionary with the key "entities"
    spacy_entry = (transcribed_text, {"entities": valid_entities})
    spacy_training_data.append(spacy_entry)
    processed_count += 1

print(f"\n--- Processing Complete ---")
print(f"Successfully processed: {processed_count}")
print(f"Skipped records: {skipped_count}")

print(f"\nSaving spaCy training data to: {os.path.join(OUTPUT_TRAINING_DATA_PATH, 'training_data.jsonl')}")
try:
    with open(os.path.join(OUTPUT_TRAINING_DATA_PATH, 'training_data.jsonl'), 'w', encoding='utf-8') as f:
        # Save as JSON Lines (each line is a valid JSON object)
        for entry in spacy_training_data:
             json.dump(entry, f, ensure_ascii=False)
             f.write('\n')
    print("Data saved successfully.")
except Exception as e:
    print(f"Error saving data: {e}")