# imports
import random
from collections import defaultdict, Counter

from scripts.load_data import (
    label_mapping, extract_labeled_tokens,
    read_tsv_file, write_tsv_file,
    write_iob2_file, modified_readNlu,
    read_iob2_file
)

# set local seed for this script
rnd = random.Random(20)

def concatenate_data(train_data, dev_data, test_data):
    '''
    
    '''
    total_data = train_data + dev_data + test_data
    return total_data


# Build mapping from entity to sentence and sentence to entity
def enitity_sentence_mapping(data, entities):
    '''
    
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
    
    '''
    rnd.shuffle(sentence_groups)

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
    used = set(train_group + dev_group + test_group)
    o_tagged = []

    for idx, sent in enumerate(data):
        if idx not in used and all(tag == "O" for tag in sent["ner_tags"]):
            o_tagged.append(idx)


    rnd.shuffle(o_tagged)
    cut1, cut2 = int(len(o_tagged) * 0.8), int(len(o_tagged) * 0.9)
    train_group += o_tagged[:cut1]
    dev_group += o_tagged[cut1:cut2]
    test_group += o_tagged[cut2:]

    train_data = [data[i] for i in sorted(train_group)]
    dev_data = [data[i] for i in sorted(dev_group)]
    test_data = [data[i] for i in sorted(test_group)]

    return train_data, dev_data, test_data

# dataset sizes
def check_dataset_sizes(train_data, dev_data, test_data):
    total = len(train_data) + len(dev_data) + len(test_data)
    print("train size:", len(train_data))
    print("dev size:", len(dev_data))
    print("test size:", len(test_data))
    print("total dataset size:", total)

# token overlap
def check_token_overlap(train_data, dev_data, test_data):
    train_tokens = extract_labeled_tokens(train_data)
    dev_tokens = extract_labeled_tokens(dev_data)
    test_tokens = extract_labeled_tokens(test_data)

    print('overlap between train and dev:', len(train_tokens & dev_tokens))
    print('overlap between dev and test:', len(dev_tokens & test_tokens))
    print('overlap between train and test:', len(train_tokens & test_tokens))

