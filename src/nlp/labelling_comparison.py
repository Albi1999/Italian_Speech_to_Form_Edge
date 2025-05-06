import os
import json
import glob
from collections import defaultdict
import re

LLM_OUTPUT_DIR = "output/ner/apim_scraper_out/"
LABEL_STUDIO_FILE = "other/label_studio_output_refined.json"

LABELS_TO_IGNORE = {
    'DATA_VERBALIZZAZIONE',
    'LUOGO_VERBALIZZAZIONE',
    'NOTE_VEICOLO',
    'ORA_VERBALIZZAZIONE',
    "TELAIO",
    "MASSA",
    "DATA_VIOLAZIONE",
    "ORA_VIOLAZIONE",
    "TARGA",
    "SANZIONE_ACCESSORIA",
    #"CONTESTAZIONE_IMMEDIATA",
}


LABEL_STUDIO_TEXT_TO_CANONICAL = {
    ('PUNTI', 'uno'): '1', ('PUNTI', 'un'): '1', 
    ('PUNTI', 'due'): '2',
    ('PUNTI', 'tre'): '3',
    ('PUNTI', 'quattro'): '4',
    ('PUNTI', 'cinque'): '5',
    ('PUNTI', 'sei'): '6',
    ('PUNTI', 'sette'): '7',
    ('PUNTI', 'otto'): '8',
    ('PUNTI', 'nove'): '9',
    ('PUNTI', 'dieci'): '10',

    ('ARTICOLO', 'uno'): '1', ('ARTICOLO', 'un'): '1',
    ('ARTICOLO', 'due'): '2',
    ('ARTICOLO', 'tre'): '3',
    ('ARTICOLO', 'quattro'): '4',
    ('ARTICOLO', 'cinque'): '5',
    ('ARTICOLO', 'sei'): '6',
    ('ARTICOLO', 'sette'): '7',
    ('ARTICOLO', 'otto'): '8',
    ('ARTICOLO', 'undici'): '11',
    ('ARTICOLO', 'dodici'): '12',
    ('ARTICOLO', 'tredici'): '13',
    ('ARTICOLO', 'quattordici'): '14',
    ('ARTICOLO', 'quindici'): '15',
    ('ARTICOLO', 'sedici'): '16',
    ('ARTICOLO', 'diciassette'): '17',
    ('ARTICOLO', 'diciotto'): '18',
    ('ARTICOLO', 'diciannove'): '19',
    ('ARTICOLO', 'venti'): '20',
    ('ARTICOLO', 'centottantacinque'): '185',
    ('ARTICOLO', 'centocinquantotto'): '158',
    ('ARTICOLO', 'centonovantotto'): '198',

    ('COMMA', 'uno'): '1', ('COMMA', 'un'): '1',
    ('COMMA', 'due'): '2',
    ('COMMA', 'tre'): '3',
    ('COMMA', 'quattro'): '4',
    ('COMMA', 'cinque'): '5',
    ('COMMA', 'sei'): '6',
    ('COMMA', 'sette'): '7',
    ('COMMA', 'otto'): '8',
    ('COMMA', 'nove'): '9',
    ('COMMA', 'dieci'): '10',
    ('COMMA', 'uno a uno dodici'): '1a(1-12)', ('COMMA', 'prima uno dodici'): '1a(1-12)', ('COMMA', 'uno a è uno a dodici'): '1a(1-12)',

    ('CIVICO_1', 'uno'): '1', ('CIVICO_1', 'un'): '1',
    ('CIVICO_1', 'due'): '2',
    ('CIVICO_1', 'tre'): '3',
    ('CIVICO_1', 'cinque'): '5',
    ('CIVICO_1', 'otto'): '8',
    ('CIVICO_1', 'undici'): '11',
    ('CIVICO_1', 'dodici'): '12',
    ('CIVICO_1', 'quindici'): '15',
    ('CIVICO_1', 'venti'): '20',
    ('CIVICO_1', 'ventiquattro'): '24',
    ('CIVICO_1', 'quarantacinque'): '45',
    ('CIVICO_1', 'settantasette'): '77',
    ('CIVICO_1', 'novantotto'): '98',

    ('CONTESTAZIONE_IMMEDIATA', 'non è stato possibile contestare sul posto'): 'mancata contestazione',
    ('CONTESTAZIONE_IMMEDIATA', 'non è stata possibile una contestazione immediata'): 'mancata contestazione',
    ('CONTESTAZIONE_IMMEDIATA', 'non è stato possibile contestare immediatamente'): 'mancata contestazione',
    ('CONTESTAZIONE_IMMEDIATA', 'impossibile contestare immediatamente'): 'mancata contestazione',
    ('CONTESTAZIONE_IMMEDIATA', 'impossibile effettuare la contestazione immediata'): 'mancata contestazione',
    ('CONTESTAZIONE_IMMEDIATA', 'non è stato possibile contestare'): 'mancata contestazione',
    ('CONTESTAZIONE_IMMEDIATA', 'non è stato possibile effettuare la contestazione immediata'): 'mancata contestazione',

    ('CONTESTAZIONE_IMMEDIATA', 'il verbale è stato contestato immediatamente'): 'contestazione immediata',
    ('CONTESTAZIONE_IMMEDIATA', 'contestazione immediata eseguita'): 'contestazione immediata',
    ('CONTESTAZIONE_IMMEDIATA', 'contestazione stata immediata'): 'contestazione immediata',
    ('CONTESTAZIONE_IMMEDIATA', 'contestazioni immediata'): 'contestazione immediata',
    ('CONTESTAZIONE_IMMEDIATA', 'contestazione avvenuta immediatamente'): 'contestazione immediata',

    ('SANZIONE_ACCESSORIA', 'non è stata applicata alcuna sanzione accessoria'): 'senza sanzione accessoria',
    ('SANZIONE_ACCESSORIA', 'non è stato applicata una sanzione accessoria'): 'senza sanzione accessoria',
    ('SANZIONE_ACCESSORIA', 'applico una sanzione accessoria'): 'con sanzione accessoria',

    ('MOTIVO_MANCATA_CONTESTAZIONE', 'assenza del trasgressori'): 'assenza del trasgressore',
    ('MOTIVO_MANCATA_CONTESTAZIONE', 'assenza del conducente'): 'assenza del trasgressore',
    ('MOTIVO_MANCATA_CONTESTAZIONE', 'non essendo stato possibile identificare il conducente sul posto'): 'assenza del trasgressore',

    ('TIPO_STAMPA', 'blu tutte'): 'bluetooth',
    ('TIPO_STAMPA', 'blu tutti'): 'bluetooth',
    ('TIPO_STAMPA', 'blu tut'): 'bluetooth',
    ('TIPO_STAMPA', 'blu tu'): 'bluetooth',
    ('TIPO_STAMPA', 'blu'): 'bluetooth',
    ('TIPO_STAMPA', 'blood'): 'bluetooth', 
    ('TIPO_STAMPA', 'blu'): 'bluetooth',
    ('TIPO_STAMPA', 'blu blood'): 'bluetooth',

    ('TIPO_STAMPA', 'wifi'): 'wifi',
    ('TIPO_STAMPA', 'vuoi fai'): 'wifi',
    ('TIPO_STAMPA', 'hawaii fai'): 'wifi',
    ('TIPO_STAMPA', 'wife'): 'wifi',
    ('TIPO_STAMPA', 'weiwei'): 'wifi',
    ('TIPO_STAMPA', 'guai fai'): 'wifi', 

    ('TIPOLOGIA', 'molto veicolo'): 'motoveicolo',
    ('TIPOLOGIA', 'ciclomotori'): 'ciclomotore',

    ('STAMPA_ANCHE_COMUNICAZIONE', 'stampa anche la comunicativa'): 'stampa anche la comunicazione',
}


def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def parse_llm_report_entities(llm_data):
    """ Extracts entities, IGNORING labels in LABELS_TO_IGNORE. """
    entities = set()
    try:
        report = llm_data.get('scraped_data', {}).get('report', {})
        if not isinstance(report, dict): return set()
        def extract_recursive(data, current_entities):
            if isinstance(data, dict):
                for key, value in data.items():
                    label = key.upper()
                    if label in LABELS_TO_IGNORE:
                        continue # Skip this key entirely
                    if isinstance(value, (dict, list)): extract_recursive(value, current_entities)
                    elif value is not None:
                        entity_text = normalize_text(str(value))
                        if entity_text: current_entities.add((label, entity_text))
            elif isinstance(data, list):
                for item in data: extract_recursive(item, current_entities)
        extract_recursive(report, entities)
    except Exception as e: print(f"Error parsing LLM: {e}"); return set()
    return entities

def parse_label_studio_data(filepath):
    """ Loads LS data, applies mapping, uses normalized key, IGNORING labels in LABELS_TO_IGNORE. """
    ls_annotations_by_text = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f: ls_data = json.load(f)
    except Exception as e: print(f"Error loading LS file: {e}"); return None
    print(f"Loaded {len(ls_data)} tasks from LS file.")
    processed_count, skipped_annotations, map_applied_count, duplicate_text_keys = 0, 0, 0, 0
    for task in ls_data:
        try:
            sentence_text_original = task.get('data', {}).get('text')
            if not sentence_text_original: continue
            sentence_text_normalized_key = normalize_text(sentence_text_original)
            if not sentence_text_normalized_key: continue
            if sentence_text_normalized_key in ls_annotations_by_text: duplicate_text_keys += 1; continue

            annotations = task.get('annotations', [])
            current_task_entities = set()
            if annotations:
                results = annotations[0].get('result', [])
                for annotation in results:
                    if annotation.get('type') == 'labels' and 'value' in annotation:
                        value = annotation['value']
                        label = value.get('labels', [None])[0]
                        entity_text_original_ls = value.get('text')
                        if label and entity_text_original_ls is not None:
                            label_upper = label.upper()
                            if label_upper in LABELS_TO_IGNORE:
                                continue # Skip this annotation entirely
                            normalized_ls_text = normalize_text(entity_text_original_ls)
                            canonical_value = LABEL_STUDIO_TEXT_TO_CANONICAL.get((label_upper, normalized_ls_text))
                            if canonical_value is not None: final_entity_text = canonical_value; map_applied_count += 1
                            else: final_entity_text = normalized_ls_text
                            current_task_entities.add((label_upper, final_entity_text))
                        else: skipped_annotations += 1
            ls_annotations_by_text[sentence_text_normalized_key] = current_task_entities
            processed_count += 1
        except Exception as e: print(f"Error processing LS task ID {task.get('id', 'N/A')}: {e}")

    print(f"Successfully processed {processed_count} unique normalized tasks from Label Studio.")
    if skipped_annotations > 0: print(f"Warning: Skipped {skipped_annotations} LS annotations (missing label/text or ignored).")
    if map_applied_count > 0: print(f"Applied canonical mapping to {map_applied_count} Label Studio annotations.")
    if duplicate_text_keys > 0: print(f"Warning: Skipped {duplicate_text_keys} tasks due to duplicate keys after text normalization.")
    return ls_annotations_by_text

if __name__ == "__main__":
    print("Starting NER Comparison...")
    print(f"Ignoring labels: {LABELS_TO_IGNORE}")

    print(f"Loading Label Studio annotations from: {LABEL_STUDIO_FILE}")
    ground_truth_data = parse_label_studio_data(LABEL_STUDIO_FILE)
    if ground_truth_data is None: exit()
    if not ground_truth_data: print("Warning: No annotations loaded.")

    total_tp, total_fp, total_fn = 0, 0, 0
    per_label_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    processed_files, match_failures, files_with_data_compared = 0, 0, 0

    llm_files = glob.glob(os.path.join(LLM_OUTPUT_DIR, "scraped_sentence_*.json"))
    print(f"Found {len(llm_files)} LLM output files in '{LLM_OUTPUT_DIR}'.")
    if not llm_files: print("Error: No LLM files found."); exit()

    for filepath in llm_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f: llm_output_data = json.load(f)
        except Exception as e: print(f"Error reading {filepath}: {e}"); continue

        llm_transcribed_text_original = llm_output_data.get('original_transcribed_sentence')
        if not llm_transcribed_text_original: continue
        llm_transcribed_text_normalized_key = normalize_text(llm_transcribed_text_original)

        if llm_transcribed_text_normalized_key in ground_truth_data:
            ground_truth_entities = ground_truth_data[llm_transcribed_text_normalized_key]
            processed_files += 1
            llm_entities = parse_llm_report_entities(llm_output_data)

            if llm_entities or ground_truth_entities:
                files_with_data_compared += 1
                sentence_tp_set = llm_entities.intersection(ground_truth_entities)
                sentence_fp_set = llm_entities.difference(ground_truth_entities)
                sentence_fn_set = ground_truth_entities.difference(llm_entities)

                total_tp += len(sentence_tp_set)
                total_fp += len(sentence_fp_set)
                total_fn += len(sentence_fn_set)
                for label, _ in sentence_tp_set: per_label_stats[label]['tp'] += 1
                for label, _ in sentence_fp_set: per_label_stats[label]['fp'] += 1
                for label, _ in sentence_fn_set: per_label_stats[label]['fn'] += 1
        else:
            match_failures += 1

    print(f"\nComparison Complete. Processed {processed_files} matching entries.")
    if files_with_data_compared < processed_files: print(f"Note: Comparison yielded non-zero TP/FP/FN counts for {files_with_data_compared} of {processed_files} matched files.")
    if match_failures > 0: print(f"Warning: Could not find matching LS entries for {match_failures} LLM files.")

    if total_tp == 0 and total_fp == 0 and total_fn == 0:
        if files_with_data_compared > 0:
             print("\nAll compared files had zero matching or mismatching entities after processing.")
        else:
             print("\nNo True Positives, False Positives, or False Negatives found.")
    else:
        print("\n--- Overall Performance ---")
        print(f"Total True Positives (TP): {total_tp}")
        print(f"Total False Positives (FP): {total_fp}")
        print(f"Total False Negatives (FN): {total_fn}")
        precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_fn)
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\n--- Performance by Label ---")
        all_reported_labels = set(per_label_stats.keys())
        report_labels = sorted([lbl for lbl in all_reported_labels if lbl not in LABELS_TO_IGNORE])

        if not report_labels:
            print("No per-label statistics generated for non-ignored labels.")
        else:
            print(f"{'Label':<30} {'TP':<5} {'FP':<5} {'FN':<5} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 80)
            for label in report_labels:
                stats = per_label_stats[label]
                tp = stats['tp']; fp = stats['fp']; fn = stats['fn']
                p_label, r_label, f1_label = calculate_metrics(tp, fp, fn)
                print(f"{label:<30} {tp:<5} {fp:<5} {fn:<5} {p_label:<10.4f} {r_label:<10.4f} {f1_label:<10.4f}")

