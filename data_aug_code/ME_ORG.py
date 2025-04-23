import pandas as pd

def add_organisation(organisation):
    tokens = organisation.split()  
    tags = ['B-ORG'] + ['I-ORG'] * (len(tokens) - 1)
    return {"tokens": tokens, "ner_tags": tags}


