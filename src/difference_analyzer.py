import os
import json

def json_diff(file1_path, file2_path):
    """
    Computes the difference between two JSON files, focusing on detailed value comparison
    and excluding the 'summary' field. Handles nested structures.

    Args:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.

    Returns:
        dict: A dictionary representing the differences. Returns an empty dict if the files are identical or if there's an error.
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return {}

    return _compare_dicts(data1, data2, path="")

def _compare_dicts(dict1, dict2, path=""):
    """
    Recursively compares two dictionaries and returns the differences, excluding 'summary'
    and drilling down to find granular differences.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        path (str, optional): The path to the current dictionary level (used for nested structures). Defaults to "".

    Returns:
        dict: A dictionary of differences, where keys are paths and values represent the specific differences.
    """
    diff = {}
    for key in dict1.keys() | dict2.keys():
        new_path = f"{path}.{key}" if path else key
        if key == "summary":
            continue
        if key in dict1 and key in dict2:
            val1 = dict1[key]
            val2 = dict2[key]
            if isinstance(val1, dict) and isinstance(val2, dict):
                nested_diff = _compare_dicts(val1, val2, new_path)
                if nested_diff:
                    diff.update(nested_diff)
            elif val1 != val2:
                if isinstance(val1, str) and isinstance(val2, str):
                    # Find the actual difference within the strings
                    i = 0
                    while i < len(val1) and i < len(val2) and val1[i] == val2[i]:
                        i += 1
                    if i < len(val1) or i < len(val2):
                        diff[new_path] = f"Difference starts at position {i}. Original: '{val1[i:]}', Transcript: '{val2[i:]}'"
                elif isinstance(val1, list) and isinstance(val2, list):
                    list_diff = _compare_lists(val1, val2, new_path)
                    if list_diff:
                        diff.update(list_diff)
                else:
                    diff[new_path] = {"original": val1, "transcript": val2}
        elif key in dict1:
            diff[new_path] = {"original": dict1[key], "transcript": None}
        else:
            diff[new_path] = {"original": None, "transcript": dict2[key]}
    return diff

def _compare_lists(list1, list2, path):
    """
    Compares two lists and returns the differences, handling lists of dictionaries.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.
        path (str): The path

    Returns:
        dict: A dictionary of differences.
    """
    diff = {}
    if len(list1) != len(list2):
        diff[path] = {"original_length": len(list1), "transcript_length": len(list2)}
        return diff # Stop if lengths differ

    for i in range(len(list1)):
        new_path = f"{path}[{i}]"
        val1 = list1[i]
        val2 = list2[i]

        if isinstance(val1, dict) and isinstance(val2, dict):
            nested_diff = _compare_dicts(val1, val2, new_path)
            if nested_diff:
                diff[new_path] = nested_diff
        elif val1 != val2:
            diff[new_path] = {"original": val1, "transcript": val2}
    return diff


def _print_dict_diff(value):
    """
    Helper function to print the difference.  Handles both direct value comparison
    and nested dictionary comparison (e.g., for lists of dicts).
    """
    if isinstance(value, dict):
        if "original" in value and "transcript" in value:
            print(f"    - Original: {value['original']}")
            print(f"    - Transcript: {value['transcript']}")
        else:
            for k, v in value.items():
                print(f"      - {k}: {v}")
    else:
        print(f"    - {value}")



if __name__ == "__main__":
    base_dir = "output/ner/apim_scraper_out/"
    results = []
    sentence_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("sentence_")]

    for sentence_dir in sentence_dirs:
        sentence_path = os.path.join(base_dir, sentence_dir)
        original_file = os.path.join(sentence_path, f"original_scraped_{sentence_dir.split('_')[-1]}.json")
        transcript_file = os.path.join(sentence_path, f"transcript_scraped_{sentence_dir.split('_')[-1]}.json")

        if os.path.exists(original_file) and os.path.exists(transcript_file):
            print(f"Comparing {original_file} and {transcript_file}")
            differences = json_diff(original_file, transcript_file)
            results.append({
                "original_file": original_file,
                "transcript_file": transcript_file,
                "differences": differences
            })
            if differences:
                print("Differences found:")
                for path, value in differences.items():
                    print(f"  {path}:")
                    _print_dict_diff(value)
            else:
                print("No differences found between the files.")
        else:
            print(f"Warning: Could not find matching files in {sentence_path}")
            results.append({
                "original_file": original_file,
                "transcript_file": transcript_file,
                "error": f"Could not find matching files in {sentence_path}"
            })

    # Save the results to a JSON file
    output_file = "comparison_results.json"
    output_path = os.path.join("output/ner", output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")
