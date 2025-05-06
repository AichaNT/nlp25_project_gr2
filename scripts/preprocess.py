
def fix_overlap(train_data, dev_data, test_data, seed = 20):
    '''
    
    '''
    rng = random.Random(seed)  # Create a local RNG instance

    # concatenate datasets
    total_data = train_data + dev_data + test_data    

    # extraxt unique entities
    total_entities = extract_labeled_tokens(total_data)
