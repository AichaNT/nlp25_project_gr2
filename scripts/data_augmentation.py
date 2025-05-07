import random
import copy

from load_data import extract_labeled_tokens, read_tsv_file, label_mapping
from scripts.extract_ME_entities import extract_first_names, get_last_names,  load_location, load_organisation

path_train = "../data/no_overlap_da_news/da_news_train.tsv"

label2id, id2label = label_mapping(path_train)

train_data = read_tsv_file(path_train, label2id)

# extracting all tokens in train data - to ensure no overlap later
train_tokens = extract_labeled_tokens(train_data)

ME_BPER = extract_first_names("data_aug_sources/Ordbog_over_muslimske_fornavne_i_DK.pdf")
ME_IPER = get_last_names("data_aug_sources/middle_eastern_last_names.txt", "data_aug_sources/KDBGIVE.tsv")
ME_LOC = load_location("data_aug_sources/the-middle-east-cities.csv")
ME_ORG = load_organisation("data_aug_sources/middle_eastern_organisations.csv")


def data_aug_replace(dataset, sentence_amount, ME_LOC=ME_LOC, ME_ORG=ME_ORG,
                     ME_BPER=ME_BPER, ME_IPER=ME_IPER, used_entities = None, train_tokens=train_tokens):
    """
    Replaces named entities in a subset of the dataset with new MENAPT ones, ensuring:
    - No reused tokens across datasets
    - No tokens from train set
    - Deterministic behavior
    - Returns updated used_entities (flat set of tokens)
    """
    local_used = set(used_entities)

    # extract sentences with containing relevant tags
    eligible_sentences = [sent for sent in dataset if any(tag not in ["O", "B-MISC", "I-MISC"] for tag in sent["ner_tags"])]
    # select random sentences
    selected_sentences = random.sample(eligible_sentences, min(sentence_amount, len(eligible_sentences)))
    # create copy to not modify original dataset 
    modified_dataset = copy.deepcopy(dataset)

    for sent in modified_dataset:
        if sent not in selected_sentences:
            continue

        i = 0
        while i < len(sent["tokens"]):
            tag = sent["ner_tags"][i]

            if tag == 'B-PER':
                available = sorted([p for p in ME_BPER if p not in local_used and p not in train_tokens])
                if available:
                    replace = random.choice(available)
                    sent["tokens"][i] = replace
                    local_used.add(replace)
                i += 1

            elif tag == 'I-PER':
                available = sorted([p for p in ME_IPER if p not in local_used and p not in train_tokens])
                if available:
                    replace = random.choice(available)
                    sent["tokens"][i] = replace
                    local_used.add(replace)
                i += 1

            elif tag == 'B-LOC':
                span_start = i
                span_len = 1
                i += 1
                while i < len(sent["ner_tags"]) and sent["ner_tags"][i] == "I-LOC":
                    span_len += 1
                    i += 1

                available = [
                    loc for loc in ME_LOC
                    if len(loc["tokens"]) == span_len and
                    tuple(loc["tokens"]) not in local_used and
                    tuple(loc["tokens"]) not in train_tokens
                ]
                
                if available:
                    replace = random.choice(available)
                    sent["tokens"][span_start:span_start + span_len] = replace["tokens"]
                    local_used.add(tuple(replace["tokens"]))

            elif tag == 'B-ORG':
                span_start = i
                span_len = 1
                i += 1
                while i < len(sent["ner_tags"]) and sent["ner_tags"][i] == "I-ORG":
                    span_len += 1
                    i += 1

                available = [
                    org for org in ME_ORG
                    if len(org["tokens"]) == span_len and
                    tuple(org["tokens"]) not in local_used and
                    tuple(org["tokens"]) not in train_tokens
                ]

                if available:
                    replace = random.choice(available)
                    sent["tokens"][span_start:span_start + span_len] = replace["tokens"]
                    local_used.add(tuple(replace["tokens"]))

            else:
                i += 1

    return modified_dataset, local_used
