import numpy as np
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")

def tokenize_and_align_labels(data):
    '''
    This function tokentizes the tokens and align the labels to the newly created subwords.
    The tokens can be split into multiple subwords, which are marked with -100, so they are ignored
    in the model *********

    Parameters:
        - data : the data we wish to tokenize and align. Must be a Huggingface dataset.

    Returns: 
        - the tokenized input with aligned labels.
    '''

    # tokenize the input
    tokenized_inputs = tokenizer(
        data["tokens"],             # tokenize the tokens (words)
        is_split_into_words = True, # tells the tokenizer each item in the list is already a separate word/token
        truncation = True,          # if a sentence is longer than the max_length it will be truncated / cut off 
        max_length = 128,           # a sentence can't be longer than 128
        padding = False             # no padding to save memory
    )

    
    # create empty list for aligned labels (to the subwords)
    all_labels = []

    # loop through each sentence
    for batch_index, labels in enumerate(data["tag_ids"]): 
        
        # 'word_ids()' returns a list the same length as the subword-tokens,
        # each entry telling you which 'word' or token it came from
        word_ids = tokenized_inputs.word_ids(batch_index = batch_index)  
        
        label_ids = []
        prev_word_id = None  

        # loop through the ids of the subword-tokens 
        for word_id in word_ids:

            if word_id is None:
                # e.g. special tokens or padding => ignore
                label_ids.append(-100)

            elif word_id == prev_word_id:
                # subword token of the same word => ignore
                label_ids.append(-100)
            
            else:
                # new subword, so use the label for the original token
                label_ids.append(labels[word_id])
            
            # move on to the next word
            prev_word_id = word_id
        
        all_labels.append(label_ids)

    # add the new algined labels to the tokenized inputs
    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


def pred2label(predictions, id2label):
    '''
    
    '''
    logits, labels = predictions # unpack predictons into logits (probabilities) and labels

    preds = np.argmax(logits, axis = -1) # choose the highest probability as the prediciton

    true_labels = [] # list to hold true labels
    pred_labels = []  # list to hold predicted labels

    # convert true labels and predictions to string
    for pred_seq, label_seq in zip(preds, labels):

        true_labels.append([id2label[label] for label in label_seq if label != -100])
        
        pred_labels.append([id2label[pred] for pred, label in zip(pred_seq, label_seq) if label != -100])

    return true_labels, pred_labels