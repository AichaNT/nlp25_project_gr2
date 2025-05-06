# imports
import random
from collections import defaultdict, Counter

from scripts.load_data import (
    mapping, extract_labeled_tokens,
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