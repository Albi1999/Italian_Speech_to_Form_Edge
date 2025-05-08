import json
import os
import re
from tqdm import tqdm
from collections import Counter

processing_issues = []

def find_entity_positions(text, entity_text, label, before_word=None, after_word=None):
    """
    Finds start and end positions of an entity in the text.
    Uses before and after words for more precise matching if available.

    Args:
        text (str): The text to search in.
        entity_text (str): The entity text to find.
        label (str): The label of the entity.
        before_word (str, optional): A word that should precede the entity.
        after_word (str, optional): A word that should follow the entity.

    Returns:
        tuple: A tuple containing the start and end positions of the entity in the text.
        None if not found, along with a reason for not finding it.
        str: Reason for not finding the entity (e.g., "not_found", "context_mismatch").
    """
    positions = None
    reason_not_found = None

    normalized_text = text.lower()
    normalized_entity_text = entity_text.lower()

    if before_word and after_word:
        norm_before = before_word.lower()
        norm_after = after_word.lower()
        for match in re.finditer(re.escape(normalized_entity_text), normalized_text):
            s, e = match.start(), match.end()
            prefix_end = s
            prefix_start = max(0, s - (len(norm_before) + 30))
            prefix_text = normalized_text[prefix_start:prefix_end]
            suffix_start = e
            suffix_end = min(len(normalized_text), e + (len(norm_after) + 30))
            suffix_text = normalized_text[suffix_start:suffix_end]
            if norm_before in prefix_text and norm_after in suffix_text:
                idx_before_in_prefix = prefix_text.rfind(norm_before)
                idx_after_in_suffix = suffix_text.find(norm_after)
                if idx_before_in_prefix != -1 and idx_after_in_suffix != -1:
                    positions = (s, e)
                    break
        if not positions:
            reason_not_found = "context_mismatch"
    else:
        try:
            for match in re.finditer(r'\b' + re.escape(normalized_entity_text) + r'\b', normalized_text):
                positions = (match.start(), match.end())
                break
            if not positions:
                start_pos = normalized_text.find(normalized_entity_text)
                if start_pos != -1:
                    positions = (start_pos, start_pos + len(normalized_entity_text))
        except re.error:
            start_pos = normalized_text.find(normalized_entity_text)
            if start_pos != -1:
                positions = (start_pos, start_pos + len(normalized_entity_text))
        
        if not positions:
            reason_not_found = "not_found"
            
    if positions:
        return positions, None
    else:
        return None, reason_not_found


def process_report_item(transcribed_sentence, key, value, entities_list, filepath_for_logging):
    """
    Processes a single item from the report, finds its position, and adds to entities_list.

    Args:
        transcribed_sentence (str): The transcribed sentence to search in.
        key (str): The key/label of the item.
        value (str): The value of the item to find in the transcribed sentence.
        entities_list (list): The list to append found entities to.
        filepath_for_logging (str): The file path for logging issues.
    """
    global processing_issues
    if value is None or not str(value).strip():
        return

    entity_text_to_find = None
    before_word = None
    after_word = None
    current_value_str = str(value).strip()
    current_label = key

    if isinstance(current_value_str, str) and ', before:' in current_value_str and ', after:' in current_value_str:
        match = re.match(r"^(.*?),\s*before:\s*(.*?),\s*after:\s*(.*?)$", current_value_str, re.IGNORECASE)
        if match:
            entity_text_to_find = match.group(1).strip()
            before_word = match.group(2).strip()
            after_word = match.group(3).strip()
        else:
            entity_text_to_find = current_value_str.split(',', 1)[0].strip()
            
    else:
        entity_text_to_find = current_value_str

    if not entity_text_to_find:
        return

    positions, reason_not_found = find_entity_positions(transcribed_sentence, entity_text_to_find, current_label, before_word, after_word)
    
    if positions:
        entity_tuple = (positions[0], positions[1], current_label)
        if entity_tuple not in entities_list:
            entities_list.append(entity_tuple)
    elif reason_not_found:
        processing_issues.append({
            "file": os.path.basename(filepath_for_logging),
            "label": current_label,
            "entity_text": entity_text_to_find,
            "reason": reason_not_found,
            "context": f"before: '{before_word}', after: '{after_word}'" if before_word else "N/A"
        })


def process_json_file(filepath):
    """ Processes a single JSON file and returns spaCy formatted data. """
    global processing_issues
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"CRITICAL: Error decoding JSON from file {filepath}: {e}")
        processing_issues.append({"file": os.path.basename(filepath), "label": "FILE_IO_ERROR", "entity_text": "JSONDecodeError", "reason": str(e), "context": "File Reading"})
        return None
    except Exception as e:
        print(f"CRITICAL: Error reading file {filepath}: {e}")
        processing_issues.append({"file": os.path.basename(filepath), "label": "FILE_IO_ERROR", "entity_text": "FileReadError", "reason": str(e), "context": "File Reading"})
        return None

    transcribed_sentence = data.get("original_transcribed_sentence")
    report = data.get("scraped_data", {}).get("report", {})

    if not transcribed_sentence:
        print(f"CRITICAL: Missing 'original_transcribed_sentence' in {filepath}. Skipping.")
        processing_issues.append({"file": os.path.basename(filepath), "label": "DATA_MISSING", "entity_text": "original_transcribed_sentence", "reason": "Field not found", "context": "File Structure"})
        return None
    if not report:
        print(f"CRITICAL: Missing 'report' data in {filepath}. Processing with empty report, but this might indicate a problem.")
        processing_issues.append({"file": os.path.basename(filepath), "label": "DATA_MISSING", "entity_text": "report", "reason": "Field not found", "context": "File Structure"})
        return (transcribed_sentence, {'entities': []})

    entities = []
 
    desired_labels = {
        "verbale_preavviso": "VERBALE_PREAVVISO",
        "data_violazione": "DATA_VIOLAZIONE",
        "ora_violazione": "ORA_VIOLAZIONE",
        "Strada_1": "STRADA_1",
        "Civico_1": "CIVICO_1",
        "contestazione_immediata": "CONTESTAZIONE_IMMEDIATA",
        "motivo_mancata_contestazione": "MOTIVO_MANCATA_CONTESTAZIONE",
        "punti": "PUNTI",
        "tipo_stampa": "TIPO_STAMPA",
        "lingua_stampa": "LINGUA_STAMPA",
        "stampa_anche_comunicazione": "STAMPA_ANCHE_COMUNICAZIONE",
        "tipologia": "TIPOLOGIA",
        "nazione": "NAZIONE",
        "targa": "TARGA",
        "tipologia_targa": "TIPOLOGIA_TARGA",
        "marca_modello": "MARCA_MODELLO",
        "colore": "COLORE",
        "codice": "CODICE",
        "articolo": "ARTICOLO",
        "comma": "COMMA",
        "sanzione_accessoria": "SANZIONE_ACCESSORIA"
    }

    for key, value in report.items():
        if value is None:
            continue
        if key == "lista_veicoli" and isinstance(value, list):
            for item_dict in value:
                if isinstance(item_dict, dict):
                    for sub_key, sub_value in item_dict.items():
                        final_label = desired_labels.get(sub_key, sub_key.upper())
                        process_report_item(transcribed_sentence, final_label, sub_value, entities, filepath)
        elif key == "violazioni" and isinstance(value, list):
            for item_dict in value:
                if isinstance(item_dict, dict):
                    for sub_key, sub_value in item_dict.items():
                        final_label = desired_labels.get(sub_key, sub_key.upper())
                        process_report_item(transcribed_sentence, final_label, sub_value, entities, filepath)
        else:
            final_label = desired_labels.get(key, key.upper())
            process_report_item(transcribed_sentence, final_label, value, entities, filepath)

    if entities:
        entities.sort(key=lambda x: (x[0], -x[1]))
        final_entities = []
        for current_ent in entities:
            is_subsumed = False
            for existing_ent in final_entities:
                if current_ent[2] == existing_ent[2] and \
                   existing_ent[0] <= current_ent[0] and existing_ent[1] >= current_ent[1] and \
                   current_ent != existing_ent:
                    is_subsumed = True
                    break
            if not is_subsumed:
                final_entities = [e for e in final_entities if not (
                    e[2] == current_ent[2] and
                    current_ent[0] <= e[0] and current_ent[1] >= e[1] and
                    e != current_ent
                )]
                final_entities.append(current_ent)
        entities = sorted(final_entities, key=lambda x: x[0])
    return (transcribed_sentence, {'entities': entities})

def save_spacy_data(spacy_data, output_filepath):
    valid_data = [item for item in spacy_data if item is not None and item[0] is not None]
    if not valid_data:
        print("CRITICAL: No valid spaCy data was generated to save.")
        return
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(valid_data, file, ensure_ascii=False, indent=4)
    print(f"spaCy training data saved to {output_filepath}")


def main(input_directory, output_filepath):
    global processing_issues
    processing_issues = []
    all_spacy_data = []
    
    if not os.path.isdir(input_directory):
        print(f"CRITICAL: Input directory '{input_directory}' not found.")
        return

    filenames = [f for f in os.listdir(input_directory) if f.endswith(".json")]
    if not filenames:
        print(f"CRITICAL: No JSON files found in '{input_directory}'.")
        return
        
    print(f"Found {len(filenames)} JSON files to process.")

    for filename in tqdm(filenames, desc="Processing files"):
        filepath = os.path.join(input_directory, filename)
        processed_data = process_json_file(filepath)
        if processed_data:
            all_spacy_data.append(processed_data)
    
    if all_spacy_data:
        save_spacy_data(all_spacy_data, output_filepath)
    else:
        print("CRITICAL: No data was successfully processed. Output file will not be created.")

    print("\n--- Processing Summary ---")
    if not processing_issues:
        print("All entities in all files processed successfully!")
    else:
        total_issues = len(processing_issues)
        print(f"Encountered {total_issues} issues during processing:")

        # Count issues by label
        issues_by_label = Counter(issue['label'] for issue in processing_issues)
        print("\nIssues by Label:")
        for label, count in issues_by_label.items():
            print(f"  - {label}: {count} issue(s)")

        # Count issues by reason (not_found, context_mismatch, etc.)
        issues_by_reason = Counter(issue['reason'] for issue in processing_issues)
        print("\nIssues by Reason:")
        for reason, count in issues_by_reason.items():
            print(f"  - {reason}: {count} issue(s)")
        
        # Count issues by file
        issues_by_file = Counter(issue['file'] for issue in processing_issues)
        print("\nFiles with Issues:")
        for file_name, count in issues_by_file.items():
            print(f"  - {file_name}: {count} issue(s)")

if __name__ == '__main__':
    input_directory = 'output/ner/apim_scraper_out/train'
    output_filepath = 'output/ner/spacy_data/spacy_training_data.json'
    main(input_directory, output_filepath)