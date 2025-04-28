import pandas as pd

def add_organisation(organisation):
    tokens = organisation.split()  
    tags = ['B-ORG'] + ['I-ORG'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}

def load_organisation(csv_path, sep=';', skiprows=1):
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows)
    unique_organisations = df["org"].drop_duplicates()
    ME_LOC = [add_organisation(loc) for loc in unique_organisations]
    return ME_LOC