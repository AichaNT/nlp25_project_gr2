import pandas as pd

def add_location(location):
    tokens = location.split()  
    tags = ['B-LOC'] + ['I-LOC'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}

df = pd.read_csv('../data_aug_sources/KDBGIVE.tsv', sep='\t', skiprows=82)
unique_city_da = df["city_da"].drop_duplicates()
ME_LOC = [add_location(loc) for loc in unique_city_da]
