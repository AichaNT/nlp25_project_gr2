# imports
import sys
sys.path.append("../")

from scripts.load_data import mapping, read_tsv_file, tokenize_and_align_labels, pred2label, compute_metrics, write_iob2_file
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset
import torch
import numpy as np
from evaluate import load 

# path to the data files
path_train = "../new_data/new_da_news_train.tsv"
path_dev = "../new_data/new_da_news_dev.tsv"
path_test = "../new_data/new_da_news_test.tsv"

# saving model name
model_name = "vesteinn/DanskBERT"

# creating the label to id mapping 
label2id, id2label = mapping(path_train)

# number of labels
num_labels = len(label2id)

# reading in the data
train_data = read_tsv_file(path_train, label2id=label2id)
dev_data = read_tsv_file(path_dev, label2id=label2id)
test_data = read_tsv_file(path_test, label2id=label2id)

# convert to huggingface format
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)
test_dataset = Dataset.from_list(test_data)

tokenized_train_dataset = train_dataset.map(
    tokenize_and_align_labels,
    batched = True,
    remove_columns=train_dataset.column_names
)

tokenized_dev_dataset = dev_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dev_dataset.column_names
)

tokenized_test_dataset = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset.column_names
)

# defining the model and config
config = AutoConfig.from_pretrained(
    model_name, 
    num_labels = num_labels, 
    id2label = id2label, 
    label2id = label2id
)

model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    torch_dtype = 'auto', 
    config = config
)

tokenizer = AutoTokenizer.from_pretrained("vesteinn/DanskBERT")

data_collator = DataCollatorForTokenClassification(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# defining the training arguments
args = TrainingArguments(
    output_dir = "output_trainer", 
    eval_strategy = 'epoch', 
    save_strategy = "no",
    learning_rate = 2e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    weight_decay = 0.01
)

# define parameters for trainer
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_train_dataset,
    eval_dataset = tokenized_dev_dataset,
    compute_metrics = compute_metrics,
    data_collator = data_collator
)

# train the model
trainer.train()

# save the model
model.save_pretrained("output_trainer")
tokenizer.save_pretrained("output_trainer")

# predicting
test_preds, test_labels, _ = trainer.predict(tokenized_test_dataset)

# predict max logit and convert to strings
_, test_predictions = pred2label((test_preds, test_labels))

# write output file for predictions on test data
write_iob2_file(test_data, predictions = test_predictions, path = "test_predictions.iob2")