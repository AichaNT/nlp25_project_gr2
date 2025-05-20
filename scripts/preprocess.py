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
    entity_to_sents, sent_to_entities = entity_sentence_mapping(total_data, total_entities)

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
def entity_sentence_mapping(data, entities):
    '''
    This function builds mappings between entities and sentence indices. 
    For each sentence, it records which labeled entities (excluding "O") are present, 
    and for each labeled entity, it tracks which sentences it occurs in.

    Parameters:
        data (List[dict]): Dataset containing sentences with NER tags.
        entities (Set[str]): Set of labeled tokens.

    Returns:
        Tuple[Dict[str, Set[int]], Dict[int, Set[str]]]: 
            - A mapping from entity to the set of sentence indices it appears in.
            - A mapping from sentence index to the set of labeled entities it contains.
    '''
    
    # dict with entities as keys and lists of sentence IDs as values
    entity_to_sents = defaultdict(set)
    sent_to_entities = defaultdict(set) # also creating mapping from sentence ID to entity

    # iterating through sentences
    for sent_id, sent in enumerate(data):

        # iterating through entities
        for tok_id, ent in enumerate(sent["tokens"]):
            
            # check if token is a labeled entity and not 'O'
            if ent in entities and sent['ner_tags'][tok_id] != "O":

                entity_to_sents[ent].add(sent_id)

                sent_to_entities[sent_id].add(ent)

    return entity_to_sents, sent_to_entities



# Group sentences by overlapping entities
def group_sentences(entity_to_sents, sent_to_entities):
    '''
    This function groups sentences that share any overlapping entities.

    Parameters:
        entity_to_sents (Dict[str, Set[int]]): Mapping from entity to sentences.
        sent_to_entities (Dict[int, Set[str]]): Mapping from sentences to entities.

    Returns:
        List[Set[int]]: A list where each element is a set of sentence indices forming a connected group. 
    '''

    visited = set()         # to track already visited sentence IDs
    sentence_groups = []    # final list for the sentence groups

    # iterate over each sentence that has labeled (non-"O") entities 
    for sent_id in sent_to_entities:

        # skip if sentence has already been assigned to a group
        if sent_id in visited: 
            continue

        # store sentence ID in current group and initialize queue with current sentence
        group, queue = set(), [sent_id]

        # perform BFS to find all sentences with shared entites
        while queue:

            current = queue.pop()

            if current in visited:
                continue

            visited.add(current)
            group.add(current)

            # for each entity in the current sentence, add all linked sentence IDs to the queue 
            for entity in sent_to_entities[current]:

                queue.extend(entity_to_sents[entity])

        # add the group to the list of groups
        sentence_groups.append(group)

    return sentence_groups
    


# shuffle and split groups by total sentence count
def split_sentence_groups(sentence_groups):
    '''
    This function splits the sentence groups intro train, dev and test sets.
    The splitting is done by total sentence count, aiming for an 80-10-10 distribution,
    while keeping entire groups together to avoid entity leakage.

    Parameters:
        sentence_groups (List[Set[int]]): Groups of sentence indices with shared entities. 

    Returns:
        Tuple[List[int], List[int], List[int]]: Sentence indices for train, dev, and test sets. 
    '''

    # randomly shuffle groups before splitting 
    random.shuffle(sentence_groups)

    # intialize lists for the splits and a count
    train_group, dev_group, test_group, count = [], [], [], 0

    # total number of sentences
    total = sum(len(g) for g in sentence_groups)

    # defining the 80-10-10 split threshold
    train_cutoff, dev_cutoff = int(total * 0.8), int(total * 0.9)

    # iterate through the groups and assign based on the cumulative count
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
    This function adds sentences that contain only 'O' labels into the splits. 
    These sentences are distributed randomly among the train, dev and test sets.

    Parameters:
        data (List[dict]): The full dataset.
        train_group (List[int]): Sentence indices assigned to the training set.
        dev_group (List[int]): Sentence indices assigned to the dev set.
        test_group (List[int]): Sentence indices assigned to the test set.

    Returns:
        Tuple[List[dict], List[dict], List[dict]]: The final non-overlapping train, dev, and test datasets. 
    '''
    
    # get all sentences IDs already assigned to a split
    used = set(train_group + dev_group + test_group)

    # initialize empty list to store IDs of sentences with only "O" labels.
    o_tagged = []

    # find sentences not used yet and with only "O" tags
    for idx, sent in enumerate(data):
        if idx not in used and all(tag == "O" for tag in sent["ner_tags"]):
            o_tagged.append(idx)

    # shuffle to get a random distribution
    random.shuffle(o_tagged)

    # split the "O" sentences into the 80-10-10 datas plit ratio
    cut1, cut2 = int(len(o_tagged) * 0.8), int(len(o_tagged) * 0.9)
    train_group += o_tagged[:cut1]
    dev_group += o_tagged[cut1:cut2]
    test_group += o_tagged[cut2:]

    # build the final splits
    train_data = [data[i] for i in sorted(train_group)]
    dev_data = [data[i] for i in sorted(dev_group)]
    test_data = [data[i] for i in sorted(test_group)]

    return train_data, dev_data, test_data
