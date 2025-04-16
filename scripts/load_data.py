# reading label data from a given column
# this is the readNlu function from the provided span_f1 file
# minor modifications were made to make it usable with our data. 
def readNlu(path, target_column = 1): # default to index 1 (thats where DaN+ labels are)
    '''
    This function reads labeled annotations from a CoNLL-like file.

    It parses a file where each line typically represents a single token and its annotations,
    separated by tabs. Empty lines denote sentence boundaries. It extracts labels from a specified column
    (by default, column index 1), collecting them as a list of label sequences, one per sentence.

    Parameters:
        path (str): Path to the input file.
        target_column (int, optional): Index of the column to extract labels from. Defaults to 1.

    Returns:
        List[List[str]]: A list where each element is a list of labels (strings) corresponding
                         to tokens in a sentence.
    '''

    annotations = []    # list for storing all the label sequences (one per sentence)
    cur_annotation = [] # temp list for labels of the current sentence

    # reading through the file line by line
    for line in open(path, encoding='utf-8'):
        line = line.strip()                     # remove leading/trailing whitespaces

        # empty lines denotes end of sentence
        if line == '':
            annotations.append(cur_annotation)  # add current annotations to annotations list
            cur_annotation = []                 # reset for the next sentence
        
        # skipping comments (start with "#" and no tokens columns)
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        
        else:
            # extract the label from the specified column and add to current sentence
            cur_annotation.append(line.split('\t')[target_column])

    return annotations


# mapping funciton 
def mapping(path):
    '''
    This function generates mappings between labels and their corresponding integer IDs from a labeled dataset.

    It reads annotations from a CoNLL-like file using the `readNlu` function,
    filters out labels containing substrings like "part" or "deriv" (case-insensitive),
    and creates a bidirectional mapping between the remaining unique labels and integer IDs.

    Parameters:
        path (str): Path to the labeled data file.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]:
            - label2id: A dictionary mapping each label to a unique integer ID.
            - id2label: A reverse dictionary mapping each integer ID back to its label.
    '''

    # get the data labels
    data_labels = readNlu(path) 

    # create empty set to store unique labels
    label_set = set()

    for labels in data_labels:
        #  filter out any labels that contain 'part' or 'deriv' (case-insensitive)
        filtered = [label for label in labels if 'part' not in label.lower() and 'deriv' not in label.lower()]
        label_set.update(filtered)

    # count of unique filtered labels
    num_labels = len(label_set)

    # create a dictionary mapping each label to a unique integer ID
    label2id = {label: id for id, label in enumerate(label_set)}

    # create a dictionary mapping each unique integer ID to a label
    id2label = {id: label for label, id in label2id.items()}

    return label2id, id2label


# load data function
# heavily inspired by the solution from assignment 5
def read_tsv_file(path, label2id):
    '''
    This function reads a TSV file containing tokens and NER labels and converts it into structured data.
    It collects the tokens, their original labels, and their corresponding integer IDs (based on the provided `label2id` mapping) for each sentence.
    Sentences are separated by empty lines. 

    Each non-empty line in the file is expected to have at least two tab-separated columns:
    - The first column is the token.
    - The second column is the corresponding NER label.

    Parameters:
        path (str): Path to the TSV file to read.
        label2id (dict): A dictionary mapping NER label strings to their corresponding integer IDs.

    Returns:
        List[dict]: A list of dictionaries, one per sentence, with keys:
            - 'tokens': list of tokens.
            - 'ner_tags': list of original NER label strings.
            - 'tag_ids': list of integer tag IDs corresponding to the NER labels.
    '''

    data = []               # final list to hold all sentences as dictionaries
    current_words = []      # tokens for the current sentence
    current_tags = []       # NER tags for the current sentence
    current_tag_ids = []    # corresponding tag IDs for the current sentence

    for line in open(path, encoding='utf-8'):
        line = line.strip() # removes any leading and trailing whitespaces from the line

        if line:
            if line[0] == '#': 
                continue # skip comments

            # splitting at 'tab', as the data is tab separated 
            tok = line.split('\t')

            # append the word (first column) to current_words
            current_words.append(tok[0]) 

            # skip labels that are not in the mapping (part and deriv)
            if tok[1] not in label2id:
                continue

            # add the current tag 
            current_tags.append(tok[1]) 

            # add the current tag mapped to the corresponding ID (int)
            current_tag_ids.append(label2id[tok[1]]) 
        
        else: # skip empty lines
            if current_words: # if current_words is not empty

                # add entry to dict where tokens and ner_tags are keys and the values are lists
                data.append({"tokens": current_words, "ner_tags": current_tags, "tag_ids": current_tag_ids})

            # start over  
            current_words = []
            current_tags = []
            current_tag_ids = []

    # check for last one
    if current_tags != []:
        data.append({"tokens": current_words, "ner_tags": current_tags, "tag_ids": current_tag_ids})
  
    return data

# extracting tokens to check for overlap in train, dev and test sets
def extract_labeled_tokens(dataset, exclude_label_id = "0"):
    '''
    This functionxtracts tokens from a dataset that have a tag ID not equal to `exclude_label_id`.
    
    Parameters:
        dataset (List[dict]): The token-tagged dataset.
        exclude_label_id (int): The tag ID to exclude (default is 0).

    Returns:
        Set[str]: A set of tokens with tag IDs different from `exclude_label_id`.
    '''

    # create empty set to store the unique tokens
    labeled_tokens = set()
    
    for sentence in dataset:
        # iterate over each token and its corresponding tag ID
        for token, tag_id in zip(sentence["tokens"], sentence["tag_ids"]):
            if tag_id != exclude_label_id:  # check if the tag is not the excluded one
                labeled_tokens.add(token)    # add the token to the set (duplicates automatically removed)
    
    return labeled_tokens
