import json
import spacy
from spacy.tokens import DocBin, Span
import warnings # To handle warnings from char_span

def convert_label_studio_to_spacy(input_path, output_path, lang_code="it"):
    """
    Converts Label Studio JSON export (standard format) to spaCy's .spacy binary format.

    Args:
        input_path (str): Path to the Label Studio JSON export file.
        output_path (str): Path where the output .spacy file will be saved.
        lang_code (str): spaCy language code (e.g., "it" for Italian, "en" for English).
                         Used for tokenization.
    """
    print(f"Loading Label Studio JSON from: {input_path}")
    try:
        # Use forward slashes or raw strings for paths to avoid escape sequence issues
        with open(input_path.replace("\\", "/"), 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred reading the file: {e}")
        return

    print(f"Using spaCy language code: {lang_code}")
    # Use a blank spaCy pipeline for tokenization only
    # You can replace "it" with your specific language code if needed
    nlp = spacy.blank(lang_code)
    db = DocBin() # Create a DocBin object

    skipped_span_count = 0
    processed_tasks = 0
    total_entities = 0
    successful_entity_spans = 0 # Counter for successful spans

    print("Starting conversion process...")
    # Iterate through each task (text item) in the Label Studio export
    for task in data:
        # Basic check for required keys
        if 'annotations' not in task or 'data' not in task or 'text' not in task['data']:
            print(f"Warning: Skipping task due to missing keys (annotations/data/text): {task.get('id', 'Unknown ID')}")
            continue

        text = task['data']['text']
        annotations = task['annotations']

        # Create a spaCy Doc object from the text
        doc = nlp.make_doc(text)
        ents = [] # List to store Span objects for the current Doc
        current_doc_successful_spans = 0

        # Process annotations for this task
        # Often there's only one set of annotations, but we loop just in case
        for annot_set in annotations:
            # Check if the annotation set was cancelled
            if annot_set.get("was_cancelled", False):
                continue
            # Check if 'result' key exists
            if 'result' not in annot_set:
                continue

            # Extract entity annotations from the 'result' list
            for entity_annot in annot_set['result']:
                # Ensure it's a label annotation and has the necessary structure
                if entity_annot.get('type') == 'labels' and 'value' in entity_annot:
                    value = entity_annot['value']
                    if 'start' in value and 'end' in value and 'labels' in value and value['labels']:
                        start = value['start']
                        end = value['end']
                        # Assuming the first label is the one we want
                        label = value['labels'][0]
                        total_entities += 1

                        # Create a spaCy Span
                        # Use try-except for potential alignment issues
                        try:
                            # Use alignment_mode="expand" or "contract" based on preference
                            # "contract" is generally safer if boundaries might be slightly off
                            span = doc.char_span(start, end, label=label, alignment_mode="contract")
                            if span is None:
                                # Warning if span boundaries don't align with tokens
                                warnings.warn(
                                    f"Skipping entity: Span ({start}, {end}, '{label}') in task ID {task.get('id', 'Unknown')} "
                                    f"does not align with token boundaries. Text: '{text[start:end]}'"
                                )
                                skipped_span_count += 1
                            else:
                                ents.append(span)
                                current_doc_successful_spans += 1
                        except Exception as e:
                            # Catch other potential errors during span creation
                             warnings.warn(
                                f"Error creating span ({start}, {end}, '{label}') in task ID {task.get('id', 'Unknown')}: {e}"
                             )
                             skipped_span_count += 1
                    else:
                         warnings.warn(f"Skipping malformed entity annotation value in task ID {task.get('id', 'Unknown')}: {value}")

        # Assign the collected entities to the Doc
        try:
            doc.ents = ents
            successful_entity_spans += current_doc_successful_spans # Add count for this doc
        except ValueError as e:
            # Handle overlapping spans if spaCy raises an error
            warnings.warn(f"ValueError setting ents for task ID {task.get('id', 'Unknown')}: {e}. Spans might overlap.")
            # You might lose spans here if overlaps cause the assignment to fail completely,
            # or spaCy might resolve it depending on the version.
            # Consider adding filtering logic if this warning appears frequently.
            # from spacy.util import filter_spans
            # try:
            #    doc.ents = filter_spans(ents)
            #    successful_entity_spans += len(doc.ents) # Count only filtered spans
            # except ValueError as e2:
            #    warnings.warn(f"Still ValueError after filtering spans for task ID {task.get('id', 'Unknown')}: {e2}")


        # Add the processed Doc to the DocBin
        db.add(doc)
        processed_tasks += 1

    print("\n--- Conversion Summary ---")
    print(f"Processed tasks: {processed_tasks}")
    print(f"Total entities found in annotations: {total_entities}")
    # FIX: Convert generator to list before getting length
    # print(f"Entities successfully converted to spans: {len(list(db.get_docs(nlp.vocab)))}") # This is inefficient as it re-creates docs
    print(f"Entities successfully converted to spans: {successful_entity_spans}") # Use the counter
    print(f"Entities skipped due to alignment or errors: {skipped_span_count}")
    print(f"Saving DocBin to: {output_path}")

    # Save the DocBin to the output file
    try:
        db.to_disk(output_path)
        print("Conversion complete!")
    except Exception as e:
        print(f"An error occurred saving the DocBin file: {e}")


label_studio_json_path = 'other/label_studio_output.json'

spacy_output_path = 'data/ner/spacy_training/training_data.spacy'

language = "it"

convert_label_studio_to_spacy(label_studio_json_path, spacy_output_path, lang_code=language)
