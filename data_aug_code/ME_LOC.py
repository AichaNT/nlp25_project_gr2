import pandas as pd

def add_location(location):
    tokens = location.split()  
    tags = ['B-LOC'] + ['I-LOC'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}

def load_location(csv_path, sep=';', skiprows=1):
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows)
    unique_locations = df["city_da"].drop_duplicates()
    ME_LOC = [add_location(loc) for loc in unique_locations]
    return ME_LOC