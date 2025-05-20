# imports
import random
from collections import defaultdict, Counter
from scripts.load_data import extract_labeled_tokens




###################### REMOVING ENTITY OVERLAP ######################

def fix_overlap(train_data, dev_data, test_data):
    '''
    This function removes overlap of labeled entities across the train, dev, and test splits.

    Parameters:
        train_data (List[dict]): Training data.
        dev_data (List[dict]): Development data.
        test_data (List[dict]): Test data.

    Returns:
        Tuple[List[dict], List[dict], List[dict]]: A tuple of train, dev, and test splits after removing entity overlaps. 
    '''

    # concatenate data
    total_data = concatenate_data(train_data, dev_data, test_data)

    # extract all unique non-"O" entities
    total_entities = extract_labeled_tokens(total_data)

    # create entity sentence mapping
    entity_to_sents, sent_to_entities = enitity_sentence_mapping(total_data, total_entities)

    # group sentences by shared entities
    sentence_groups = group_sentences(entity_to_sents, sent_to_entities)    

    # shuffle and split groups by total sentence count
    train_group, dev_group, test_group = split_sentence_groups(sentence_groups) 

    # get final slipts (with all "O" sentences)
    train_data, dev_data, test_data = finalize_split_with_o_sentences(total_data, train_group, dev_group, test_group)

    return train_data, dev_data, test_data




###################### HELPER FUNCTIONS ######################

# concatenate data
def concatenate_data(train_data, dev_data, test_data):
    '''
    This function concatenates the train, dev and test data into a single list.

    Parameters:
        train_data (List[dict]): Training data.
        dev_data (List[dict]): Development data.
        test_data (List[dict]): Test data.

    Returns:
        List[dict]: Combined dataset as a single list of sentences. 
    '''

    total_data = train_data + dev_data + test_data
    return total_data


# Build mapping from entity to sentence and sentence to entity
def enitity_sentence_mapping(data, entities):
    '''
    This function builds mappings between entities and sentence indices. 

    Parameters:

    Returns:
    '''
    
    # dict with entities as keys and lists of sentence IDs as values
    entity_to_sents = defaultdict(set)
    sent_to_entities = defaultdict(set) # also creating mapping from sentence ID to entity

    for sent_id, sent in enumerate(data):

        for tok_id, ent in enumerate(sent["tokens"]):

            if ent in entities and sent['ner_tags'][tok_id] != 'O':

                entity_to_sents[ent].add(sent_id)

                sent_to_entities[sent_id].add(ent)

    return entity_to_sents, sent_to_entities


# Group sentences by overlapping entities
def group_sentences(entity_to_sents, sent_to_entities):
    '''
    This function 

    Parameters:

    Returns:
    '''

    visited = set()
    sentence_groups = []

    for sent_id in sent_to_entities:

        if sent_id in visited:
            continue

        group, queue = set(), [sent_id]

        while queue:

            current = queue.pop()

            if current in visited:
                continue

            visited.add(current)
            group.add(current)

            for entity in sent_to_entities[current]:

                queue.extend(entity_to_sents[entity])

        sentence_groups.append(group)

    return sentence_groups
    

# shuffle and split groups by total sentence count
def split_sentence_groups(sentence_groups):
    '''
    This function 

    Parameters:

    Returns:
    '''

    random.shuffle(sentence_groups)

    train_group, dev_group, test_group, count = [], [], [], 0
    total = sum(len(g) for g in sentence_groups)
    train_cutoff, dev_cutoff = int(total * 0.8), int(total * 0.9)

    for group in sentence_groups:
        if count < train_cutoff:
            train_group += group
        elif count < dev_cutoff:
            dev_group += group
        else:
            test_group += group
        count += len(group)

    return train_group, dev_group, test_group


# add sentences with only 'O' tags
def finalize_split_with_o_sentences(data, train_group, dev_group, test_group):
    '''
    This function 

    Parameters:

    Returns:
    '''

    used = set(train_group + dev_group + test_group)
    o_tagged = []

    for idx, sent in enumerate(data):
        if idx not in used and all(tag == "O" for tag in sent["ner_tags"]):
            o_tagged.append(idx)


    random.shuffle(o_tagged)
    cut1, cut2 = int(len(o_tagged) * 0.8), int(len(o_tagged) * 0.9)
    train_group += o_tagged[:cut1]
    dev_group += o_tagged[cut1:cut2]
    test_group += o_tagged[cut2:]

    train_data = [data[i] for i in sorted(train_group)]
    dev_data = [data[i] for i in sorted(dev_group)]
    test_data = [data[i] for i in sorted(test_group)]

    return train_data, dev_data, test_data
