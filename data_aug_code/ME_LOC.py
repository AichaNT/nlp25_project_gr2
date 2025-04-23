import pandas as pd

df = pd.read_csv("../data_aug_sources/the-middle-east-cities.csv", sep = ";", skiprows = 1)

def add_location(location):
    tokens = location.split()  
    tags = ['B-LOC'] + ['I-LOC'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}

unique_city_da = df["city_da"].drop_duplicates()
ME_LOC = [add_location(loc) for loc in unique_city_da]
