# imports
import numpy as np
from transformers import AutoTokenizer

# load the tokenizer 
tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")

# tokenizing and aligning
def tokenize_and_align_labels(data):
    '''
    This function tokentizes the input sentences and aligns the original NER labels with the resulting subword tokens.
    When a token is split into multiple subwords, only the first subword retains the original label.
    All subsequent subwords and special tokens are marked with -100 so they are ignored during model training.

    Parameters:
        data (Huggingface dataset): The data we wish to tokenize and align. Must be a Huggingface dataset and contain:
            - "tokens": a list of tokens per sentence.
            - "tag_ids": a list of corresponding label IDs per sentence.
        
    Returns:
        dict: A dictionary containing tokenized inputs with an added "labels" key that holds the aligned label IDs.
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
                # first subword of a token, so use the label for the original token
                label_ids.append(labels[word_id])
            
            # move on to the next word
            prev_word_id = word_id # track the previous word ID to catch subwords
        
        all_labels.append(label_ids)

    # add the new algined labels to the tokenized inputs
    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


# converting predictions to NER labels
def pred2label(predictions, id2label):
    '''
    This function converts model output (logits and true labels) into NER label strings.
    It ignores special tokens and subwords that were marked with -100 during training.

    Parameters:
        predictions (Tuple[np.ndarray, np.ndarray]): A tuple of (logits, true_labels) obtained from Huggingface Trainer's ".predict()" method. 
            - logits: Model output of shape [batch_size, seq_len, num_labels].
            - true_labels: Ground truth label IDs, with -100 marking ignored positions.

        id2label (dict): A dictionary mapping label IDs (int) to label strings.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Two lists:
            - true_labels: list of label sequences from the gold data.
            - pred_labels: list of label sequences from the model predictions.
    '''

    # unpack predictons into logits (probabilities) and labels
    logits, labels = predictions 

    # convert logits to predicted class IDs (take the highest scoring label for each token)
    preds = np.argmax(logits, axis = -1) 

    true_labels = [] # list to hold true labels
    pred_labels = [] # list to hold predicted labels

    # convert true labels and predictions to string
    for pred_seq, label_seq in zip(preds, labels):

        # only include labels for non-ignored positions (-100)
        true_labels.append([id2label[label] for label in label_seq if label != -100])
        
        pred_labels.append([id2label[pred] for pred, label in zip(pred_seq, label_seq) if label != -100])

    return true_labels, pred_labels