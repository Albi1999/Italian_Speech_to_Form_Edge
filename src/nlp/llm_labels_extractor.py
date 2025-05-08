import json
import os
import re

def load_json_files(directory):
    """Load all JSON files from the given directory."""
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    data = []
    for file in files:
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    for item in loaded_data:
                        if isinstance(item, dict):
                            data.append(item)
                        else:
                            print(f"Warning: Unexpected item type within list in {file_path}: {type(item)}. Skipping.")
                elif isinstance(loaded_data, dict):
                    data.append(loaded_data)
                else:
                    print(f"Warning: Unexpected data type in {file_path}: {type(loaded_data)}. Skipping file.")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {file_path}. Skipping file.")
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}. Skipping file.")
    return data

def find_all_entity_spans(text, entity_value, entity_label):
    """Helper function to find *all* entity spans in the text (case-insensitive, whitespace tolerant)."""
    spans = []
    if entity_value and isinstance(entity_value, str):
        normalized_text = re.sub(r'\s+', ' ', text.lower())
        normalized_value = re.sub(r'\s+', ' ', entity_value.lower())
        
        start_index = 0
        while True:
            start_index = normalized_text.find(normalized_value, start_index)
            if start_index == -1:
                break
            
            # Map normalized index back to original text index
            original_start = -1
            temp_index = 0
            normalized_index = 0
            for i, char in enumerate(text.lower()):
                if not char.isspace():
                    if normalized_index == start_index:
                        original_start = i
                        break
                    normalized_index += 1
            
            if original_start != -1:
                original_end = original_start + len(entity_value)
                spans.append((original_start, original_end, entity_label))
                start_index += len(normalized_value) # Move past the found occurrence
            else:
                break # Should not happen, but safety check
    return spans

def is_overlapping(new_start, new_end, existing_spans):
    """Checks if a span (start, end) overlaps with any existing spans."""
    for existing_start, existing_end, _ in existing_spans:
        if new_start < existing_end and new_end > existing_start:
            return True
    return False

def extract_entities(scraped_data, transcribed_sentence):
    """Extract entities from 'scraped_data' and find their positions in 'transcribed_sentence'."""
    entities = []
    report = scraped_data.get('report', {})
    
    # Define entities to extract from the report with corresponding labels
    entity_mappings = {
        'verbale_preavviso': 'REPORT_TYPE',
        'data_violazione': 'DATE',
        'ora_violazione': 'TIME',
        'Strada_1': 'STREET',
        'Civico_1': 'HOUSE_NUMBER',
        'contestazione_immediata': 'CONTESTATION_STATUS',
        'motivo_mancata_contestazione': 'NO_CONTESTATION_REASON',
        'punti': 'POINTS',
        'tipo_stampa': 'PRINT_MODE',
        'lingua_stampa': 'LANGUAGE',
        'stampa_anche_comunicazione': 'PRINT_REQUEST'
    }
    
    # Create a "working copy" of the transcribed sentence
    working_sentence = transcribed_sentence
    offset_mapping = []  # Keep track of how the indices shift as we "delete" parts of the sentence

    def map_to_original_indices(start, end):
        """Maps indices from the working sentence to the original transcribed sentence."""
        original_indices = []
        current_offset = 0
        for original_start, deleted_length in offset_mapping:
            if start >= original_start:
                start += deleted_length
                end += deleted_length
            else:
                break
        return start, end

    # Extract simple entities from the report
    for key, label in entity_mappings.items():
        entity_value = report.get(key)
        if entity_value:
            search_value = str(entity_value).lower()
            spans = find_all_entity_spans(working_sentence, search_value, label)
            if spans:
                start, end, _ = spans[0]  # Take only the first span
                original_start, original_end = map_to_original_indices(start, end)
                entities.append((original_start, original_end, label))

                # "Delete" the found entity from the working sentence
                working_sentence = working_sentence[:start] + " " * (end - start) + working_sentence[end:]
                offset_mapping.append((start, end - start))

    # Extract entities from 'lista_veicoli'
    for vehicle in report.get('lista_veicoli', []):
        vehicle_entity_map = {
            'tipologia': 'VEHICLE',
            'nazione': 'COUNTRY',
            'targa': 'LICENSE_PLATE',
            'tipologia_targa': 'LICENSE_PLATE_TYPE',
            'marca_modello': 'VEHICLE_MODEL',
            'colore': 'COLOR'
        }
        for key, label in vehicle_entity_map.items():
            value = vehicle.get(key)
            if value:
                search_value = str(value).lower()
                if label == 'LICENSE_PLATE':
                    cleaned_targa = search_value.replace(' ', '')
                    spans = find_all_entity_spans(working_sentence, cleaned_targa, label)
                    if not spans:
                        spans = find_all_entity_spans(working_sentence, search_value, label)
                else:
                    spans = find_all_entity_spans(working_sentence, search_value, label)
                if spans:
                    start, end, _ = spans[0]
                    original_start, original_end = map_to_original_indices(start, end)
                    entities.append((original_start, original_end, label))

                    working_sentence = working_sentence[:start] + " " * (end - start) + working_sentence[end:]
                    offset_mapping.append((start, end - start))

    # Extract entities from 'violazioni'
    for violation in report.get('violazioni', []):
        violation_mappings = {
            'codice': 'LAW_CODE',
            'articolo': 'LAW_ARTICLE',
            'comma': 'LAW_COMMA',
            'sanzione_accessoria': 'PENALTY'
        }
        for key, label in violation_mappings.items():
            value = violation.get(key)
            if value:
                search_value = str(value).lower()
                if key in ['articolo', 'comma', 'codice']:
                    search_value = f'{key} {search_value}'
                else:
                    spans = find_all_entity_spans(working_sentence, search_value, label)
                if spans:
                    start, end, _ = spans[0]
                    original_start, original_end = map_to_original_indices(start, end)
                    entities.append((original_start, original_end, label))

                    working_sentence = working_sentence[:start] + " " * (end - start) + working_sentence[end:]
                    offset_mapping.append((start, end - start))
    return entities

def create_spacy_train_data(directory, output_directory):
    """Create spaCy training data from JSON files and save to a new file."""
    data = load_json_files(directory)
    training_data = []

    for item in data:
        if isinstance(item, dict):
            scraped_data = item.get('scraped_data', {})
            transcribed_sentence = item.get('original_transcribed_sentence')
            if not transcribed_sentence:
                continue

            entities = extract_entities(scraped_data, transcribed_sentence)
            training_data.append((transcribed_sentence, {'entities': entities}))
        else:
            print(f"Warning: Unexpected data type in training data: {type(item)}. Skipping.")
            continue
    
    # Save the training data to a file in the specified output directory
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, 'spacy_train_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)
    print(f"Saved spaCy training data to: {output_file}")

if __name__ == "__main__":
    directory_path = 'output/ner/apim_scraper_out/train'  # Your input directory
    output_directory = 'output/ner/spacy_data'  # Your desired output directory
    create_spacy_train_data(directory_path, output_directory)