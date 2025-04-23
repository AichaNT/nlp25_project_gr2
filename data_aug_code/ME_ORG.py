import pandas as pd

df = pd.read_csv("../data_aug_sources/middle_eastern_organisations.csv", sep = ";", skiprows = 1)

def add_organisation(organisation):
    tokens = organisation.split()  
    tags = ['B-ORG'] + ['I-ORG'] * (len(tokens) - 1)
    return {"tokens": tokens, "tags": tags}

unique_orgs = df["org"].drop_duplicates()
ME_ORG = [add_organisation(org) for org in unique_orgs]

