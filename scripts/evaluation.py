
###################### EXTRACTING TRUE AND PREDICTED LABELS ######################

def extract_true_and_pred_labels(data, pred):
    '''
    This function extracts the true and predicted NER label sequences from two datasets. 

    Parameter:
        data (List[dict]): The ground truth dataset, where each dictionary contains a "ner_tags" list.
        pred (List[dict]): The predicted dataset, where each dictionary contains a "ner_tags" list.

    Returns:
        Tuple[List[List[str]], List[List[str]]]:
            - true_labels: A list of label sequences from the true data.
            - pred_labels: A list of label sequences from the predicted data.
    '''
    
    true_labels = []

    for sent in data:
        true_labels.append(sent['ner_tags'])

    # saving all predicted labels
    pred_labels = []

    for sent in pred:
        pred_labels.append(sent['ner_tags'])

    return true_labels, pred_labels