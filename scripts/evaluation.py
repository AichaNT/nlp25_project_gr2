
###################### EXTRACTING TRUE AND PREDICTED LABELS ######################

def extract_true_and_pred_labels(data, pred):
    true_labels = []

    for sent in data:
        true_labels.append(sent['ner_tags'])

    # saving all predicted labels
    pred_labels = []

    for sent in pred:
        pred_labels.append(sent['ner_tags'])

    return true_labels, pred_labels