# imports
import random
import sys
sys.path.append("../")

from scripts.load_data import write_tsv_file, extract_labeled_tokens, label_mapping, read_tsv_file, write_iob2_file
from scripts.data_aug import data_aug_replace
from middle_eastern_ne import extract_first_names, get_last_names,  load_location, load_organisation

random.seed(42)

ME_BPER = extract_first_names("../data_aug_sources/Ordbog_over_muslimske_fornavne_i_DK.pdf")
ME_IPER = get_last_names("../data_aug_sources/middle_eastern_last_names.txt", "../data_aug_sources/KDBGIVE.tsv")
ME_LOC = load_location("../data_aug_sources/the-middle-east-cities.csv")
ME_ORG = load_organisation("../data_aug_sources/middle_eastern_organisations.csv")

# path to the data files
path_train = "../data/no_overlap_da_news/da_news_train.tsv"
path_dev = "../data/no_overlap_da_news/da_news_dev.tsv"
path_test = "../data/no_overlap_da_news/da_news_test.tsv"

# create mapping
label2id, id2label = label_mapping(path_train)

# read in the DaN+ data
train_data = read_tsv_file(path_train, label2id)
dev_data = read_tsv_file(path_dev, label2id)
test_data = read_tsv_file(path_test, label2id)

# extracting all tokens in train data - to ensure no overlap later
train_tokens = extract_labeled_tokens(train_data)

# for saving all used entities
used_entities = set()

ME_dev, used_entities = data_aug_replace(dev_data, sentence_amount=1000,
                                         ME_LOC = ME_LOC, ME_ORG = ME_ORG, ME_BPER = ME_BPER, ME_IPER = ME_IPER, 
                                         used_entities = used_entities, train_tokens=train_tokens)

ME_test, used_entities = data_aug_replace(test_data, sentence_amount=1000,
                                         ME_LOC = ME_LOC, ME_ORG = ME_ORG, ME_BPER = ME_BPER, ME_IPER = ME_IPER, 
                                         used_entities = used_entities, train_tokens=train_tokens)
final_used = used_entities

sentence_values = [100, 250, 500, 864, 1000, 1500, 1729]
augmented_datasets = []

for amount in sentence_values:
    aug_set, _ = data_aug_replace(train_data, sentence_amount=amount,
                                         ME_LOC = ME_LOC, ME_ORG = ME_ORG, ME_BPER = ME_BPER, ME_IPER = ME_IPER, 
                                         used_entities = final_used, train_tokens=train_tokens)
    augmented_datasets.append(aug_set)

# save as tsv files
write_tsv_file(ME_dev, "../data/me_data/middle_eastern_dev.tsv")
write_tsv_file(ME_test, "../data/me_data/middle_eastern_test.tsv")


################################################# Baseline code ######################################################

from scripts.train_and_pred import tokenize_and_align_labels, pred2label

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset
import torch
from evaluate import load 

# path to the data files
path_me_dev = "data/me_data/middle_eastern_dev.tsv"  

# saving model name
model_name = "vesteinn/DanskBERT"

# creating the label to id mapping 
label2id, id2label = label_mapping(path_train)

# number of labels
num_labels = len(label2id)

# reading in the data
dev_data = read_tsv_file(path_dev, label2id=label2id)
me_dev_data = read_tsv_file(path_me_dev, label2id=label2id)

# convert to huggingface format
dev_dataset = Dataset.from_list(dev_data)
me_dev_dataset = Dataset.from_list(me_dev_data)

# tokenize and align dev dataset
tokenized_dev_dataset = dev_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dev_dataset.column_names
)

# tokenize and align ME dev dataset
tokenized_me_dev_dataset = me_dev_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=me_dev_dataset.column_names
)

# load model configuration
config = AutoConfig.from_pretrained(
    model_name, 
    num_labels = num_labels, 
    id2label = id2label, 
    label2id = label2id
)

# load pretrained model
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    torch_dtype = 'auto', 
    config = config
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")

# define data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# set the device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device) # move model to device

# define the training arguments
args = TrainingArguments(
    output_dir = "output_trainer", 
    eval_strategy = 'no', 
    save_strategy = "no",
    learning_rate = 2e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    weight_decay = 0.01,
    remove_unused_columns=False
)

# loop for training sets
for idx, train_data in enumerate(augmented_datasets):
    train_dataset = Dataset.from_list(train_data) # convert to huggingface format

    tokenized_train_dataset = train_dataset.map( # tokenize train dataset and align labels with subword tokens
        tokenize_and_align_labels,
        batched = True,
        remove_columns=train_dataset.column_names
    )

    # define parameters for trainer
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = tokenized_train_dataset,
        data_collator = data_collator
    )

    # train the model
    trainer.train()

    # predicting on non-augmented dev set
    dev_preds, dev_labels, _ = trainer.predict(tokenized_dev_dataset)

    # predicting on augmented dev set
    me_dev_preds, me_dev_labels, _ = trainer.predict(tokenized_me_dev_dataset)

    # predict max logit and convert to strings for non-augmented
    _, dev_predictions = pred2label((dev_preds, dev_labels), id2label)

    # predict max logit and convert to strings for augmented
    _, me_dev_predictions = pred2label((me_dev_preds, me_dev_labels), id2label)

    # write output file for predictions on dev sets
    write_iob2_file(dev_data, predictions = dev_predictions, path = f"{idx}dev_pred.iob2")
    write_iob2_file(me_dev_data, predictions = me_dev_predictions, path = f"{idx}me_dev_pred.iob2")