import pandas as pd

df = pd.read_csv('data_aug_sources/KDBGIVE.tsv', sep='\t', skiprows = 82 )
I_PER = df["Roman"].tolist()

def extract_last_names(name_str):
    lines = name_str.split("\n")
    names = []
    for line in lines:
        split = line.split(" (")
        name = split[0]
        names.append(name)
    return names

#UNFINISHED!!