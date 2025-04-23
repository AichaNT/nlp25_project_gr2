import pandas as pd

#df = pd.read_csv('data_aug_sources/KDBGIVE.tsv', sep='\t', skiprows = 82 )
#I_PER = df["Roman"].tolist()

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

#UNFINISHED!!