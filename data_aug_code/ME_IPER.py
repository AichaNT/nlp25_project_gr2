import pandas as pd

def extract_last_names(name_path):
    with open(name_path, "r", encoding="utf-8") as f:
        last_names = f.read()
    lines = last_names.split("\n")
    names = []
    for line in lines:
        split = line.split(" (")
        name = split[0]
        if len(name) <= 1 or any(char in name for char in "+-*"):
            continue
        names.append(name)
    return names

def get_combined_names(txt_path, tsv_path, column_name="Roman", skiprows=82, sep='\t'):
    text_names = extract_last_names(txt_path)
    df = pd.read_csv(tsv_path, sep=sep, skiprows=skiprows)
    csv_names = df[column_name].dropna().astype(str).tolist()
    
    combined = list(set(text_names + csv_names))
    return combined