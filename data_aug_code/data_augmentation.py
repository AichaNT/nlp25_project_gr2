import random
import pandas as pd

def entities_by_label(data, target_label):
    """
    Collects full entity name of locations and organisation from labeled the dataset. 
    If an entity is made up of multiple words, it joins them into a single string.
    
    Args:
        data (List[Dict]): Dataset containing 'tokens' and 'tags' for each sentence.
        target_label (str): Label prefix to filter for (e.g., 'B-LOC', 'B-ORG').
        
    Returns:
        Set[str]: A set of labeled token strings (e.g., {'Beirut', 'Al Mawsil al Jadidah'})
    """
    grouped_strings = set()

    for sent in data:
        tokens = sent['tokens']
        tags = sent['ner_tags']

        i = 0
        while i < len(tokens):
            tag = tags[i]

            if tag.startswith(target_label):
                span_tokens = [tokens[i]]
                i += 1
                while i < len(tokens) and tags[i].startswith('I'):
                    span_tokens.append(tokens[i])
                    i += 1

                # Join tokens into a single string and add to the set
                entity_string = ' '.join(span_tokens)
                grouped_strings.add(entity_string)
            else:
                i += 1

    return grouped_strings


def get_all_entities(data, exclude_label="O"):
    """
    Collects full entity name of locations and organisation from labeled the dataset. 
    If an entity is made up of multiple words, it joins them into a single string.

    Args:
        data (List[Dict]): Dataset with 'tokens' and 'tags' per sentence.
        exclude_label (str): Label to ignore (default is 'O').

    Returns:
        Set[str]: Set of labeled entity strings (e.g., {'Beirut', 'Al Mawsil al Jadidah'})
    """
    grouped_strings = set()

    for sent in data:
        tokens = sent['tokens']
        tags = sent['ner_tags']
        i = 0
        while i < len(tokens):
            tag = tags[i]

            if tag != exclude_label and tag.startswith('B-'):
                span_tokens = [tokens[i]]
                i += 1
                # Collect I- continuation tags
                while i < len(tokens) and tags[i].startswith('I-'):
                    span_tokens.append(tokens[i])
                    i += 1

                entity_string = ' '.join(span_tokens)
                grouped_strings.add(entity_string)
            else:
                i += 1

    return grouped_strings

ME_LOC_tokens = entities_by_label(ME_LOC, target_label = "B-LOC")
ME_ORG_tokens = entities_by_label(ME_ORG, target_label = "B-ORG")
ME_BPER_tokens = set(ME_BPER)
ME_IPER_tokens = set(ME_IPER)

train_tokens = get_all_entities(train_data)
dev_tokens = get_all_entities(dev_data)
test_tokens = get_all_entities(test_data)



updated_ME_BPER = list(ME_BPER_tokens - (train_tokens & ME_BPER_tokens) - (dev_tokens & ME_BPER_tokens) - (test_tokens & ME_BPER_tokens))
updated_ME_IPER = list(ME_IPER_tokens - (train_tokens & ME_IPER_tokens) - (dev_tokens & ME_IPER_tokens) - (test_tokens & ME_IPER_tokens))
updated_ME_LOC = list(ME_LOC_tokens - (train_tokens & ME_LOC_tokens) - (dev_tokens & ME_LOC_tokens) - (test_tokens & ME_LOC_tokens))
updated_ME_ORG = list(ME_ORG_tokens - (train_tokens & ME_ORG_tokens) - (dev_tokens & ME_ORG_tokens) - (test_tokens & ME_ORG_tokens))

