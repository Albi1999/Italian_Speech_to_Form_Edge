import json

def extract_transcriptions(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_file}: {e}")
        return
    
    transcriptions = []
    for item in data:
        if item['dataset'] != 'coqui':
            transcriptions.append(item['transcribed_sentence']['text'])

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(transcriptions, outfile, ensure_ascii=False, indent=4)
    
    print(f"Transcriptions extracted and saved to {output_file}")

input_file = 'output/stt/train_transcription/train_transcriptions.json'
output_file = 'output/stt/train_transcription/transcription_plain.json'

extract_transcriptions(input_file, output_file)


def extract_transcriptions_to_txt(input_file, output_file_txt):
    """
    Reads a JSON file containing transcription data, extracts specific transcriptions,
    and saves them line by line into a plain text file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file_txt (str): Path to the output TXT file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_file}: {e}")
        return

    transcriptions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and \
               'dataset' in item and \
               'transcribed_sentence' in item and \
               isinstance(item['transcribed_sentence'], dict) and \
               'text' in item['transcribed_sentence']:

                if item['dataset'] != 'coqui':
                    transcriptions.append(item['transcribed_sentence']['text'])
            else:
                print(f"Warning: Skipping malformed item in input file: {item}")
    else:
        print(f"Error: Expected a list of items in {input_file}, but got {type(data)}")
        return

    try:
        with open(output_file_txt, 'w', encoding='utf-8') as outfile:
            for transcription in transcriptions:
                outfile.write(transcription + '\n')

        print(f"Transcriptions extracted and saved to {output_file_txt}")

    except Exception as e:
        print(f"An unexpected error occurred while writing to {output_file_txt}: {e}")

input_file = 'output/stt/train_transcription/train_transcriptions.json'
output_file = 'output/stt/train_transcription/transcription_plain.txt'

extract_transcriptions_to_txt(input_file, output_file)


def create_label_studio_json(input_file, output_file):
    """
    Reads a JSON file containing transcription data, extracts specific transcriptions,
    and saves them in a JSON format suitable for Label Studio import based on
    an XML configuration using <Text name="text" value="$text"/>.

    The output format will be a list of objects, like:
    [
        {"text": "transcription sentence 1"},
        {"text": "transcription sentence 2"},
        ...
    ]

    Args:
        input_file (str): Path to the input JSON file (e.g., transcriptions.json).
                          Expected format: A list of objects, where each object has
                          'dataset' and 'transcribed_sentence': {'text': ...}.
        output_file (str): Path to the output JSON file for Label Studio.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_file}: {e}")
        return

    label_studio_data = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and \
               'dataset' in item and \
               'transcribed_sentence' in item and \
               isinstance(item['transcribed_sentence'], dict) and \
               'text' in item['transcribed_sentence']:

                if item['dataset'] != 'coqui':
                    transcription_text = item['transcribed_sentence']['text']
                    label_studio_data.append({"text": transcription_text})
            else:
                print(f"Warning: Skipping malformed item in input file: {item}")
    else:
        print(f"Error: Expected a list of items in {input_file}, but got {type(data)}")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(label_studio_data, outfile, ensure_ascii=False, indent=4)

        print(f"Label Studio compatible JSON data saved to {output_file}")

    except Exception as e:
        print(f"An unexpected error occurred while writing to {output_file}: {e}")

input_filename = 'output/stt/vosk_transcription/transcriptions.json'
output_filename_label_studio = 'output/stt/vosk_transcription/label_studio_import.json'

create_label_studio_json(input_filename, output_filename_label_studio)