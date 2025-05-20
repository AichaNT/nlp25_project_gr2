# imports
import fitz
import pandas as pd




###################### MIDDLE EASTERN FIRST NAMES (B-PER) ######################

def extract_first_names(name_path):
    '''
    This function extracts first names from a PDF file and cleans unwanted characters
    '''
    first_names = []
    with fitz.open(name_path) as doc:
        for page in doc: # Iterating through all pages
            blocks = page.get_text("dict")["blocks"] # Extracting text on each page
            for block in blocks: 
                for line in block["lines"]:  # Iterating through the text lines
                    for span in line["spans"]:  # Iterating through the text spans
                        if span["flags"] & 16:  # Targetting the bold text 
                            name = span["text"].strip()
                            if name:
                                if len(name) <= 1 or any(char in name for char in "+-*"):
                                    continue
                                first_names.append(name)

    # Data cleaning
    if first_names:
        first_names.pop(0)
    ME_BPER = [name.replace("*", "") for name in first_names]
    ME_BPER = list(set(ME_BPER)) # Removing duplicates
    return ME_BPER




###################### MIDDLE EASTERN LAST NAMES (I-PER) ######################

def extract_last_names(name_path):
    '''
    This function extracts last names from a text file, cleaning unwanted characters
    '''
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

def get_last_names(txt_path, tsv_path, column_name="Roman", skiprows=82, sep='\t'):
    '''
    This function combines last names from a text file and a TSV file into one unique list
    '''
    text_names = extract_last_names(txt_path)
    df = pd.read_csv(tsv_path, sep=sep, skiprows=skiprows)
    csv_names = df[column_name].dropna().astype(str).tolist()

    combined = list(set(text_names + csv_names))
    return combined




###################### MIDDLE EASTERN LOCATIONS (-LOC) ######################

def add_location(location):
    '''
    This function converts a string with a location name into a list of dictionaries of tokens with B-LOC and I-LOC tags
    '''
    tokens = location.split()  
    tags = ['B-LOC'] + ['I-LOC'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}

def load_location(csv_path, sep=';', skiprows=1):
    '''
    This function loads location names from a CSV file and assigns NER tagging with the help of the add_location function
    '''
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows)
    unique_locations = df["city_da"].drop_duplicates()
    ME_LOC = [add_location(loc) for loc in unique_locations]
    return ME_LOC




###################### MIDDLE EASTERN ORGANISATIONS (-ORG) ######################

def add_organisation(organisation):
    '''
    This function converts a string with an organisation name into a list of dictionaries of tokens with B-ORG and I-ORG tags
    '''
    tokens = organisation.split()  
    tags = ['B-ORG'] + ['I-ORG'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}

def load_organisation(csv_path, sep=';', skiprows=1):
    '''
    This function loads organisation names from a CSV file and assigns NER tagging with the help of the add_organisation function
    '''
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows)
    unique_organisations = df["org"].drop_duplicates()
    ME_LOC = [add_organisation(loc) for loc in unique_organisations]
    return ME_LOC