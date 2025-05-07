from scripts.train_pred import tokenize_and_align_labels, pred2label
from scripts.load_data import label_mapping, read_tsv_file, write_iob2_file
from scripts.data_augmentation import data_aug_replace

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset
import torch
import pickle
import random
random.seed(20)

# path to the data files
path_train = "data/no_overlap_da_news/da_news_train.tsv"
path_test = "data/no_overlap_da_news/da_news_test.tsv"
path_me_test = "data/me_data/middle_eastern_test.tsv" 

# create mapping
label2id, id2label = label_mapping(path_train)

# number of labels
num_labels = len(label2id)

# read in the data
train_data = read_tsv_file(path_train, label2id)
test_data = read_tsv_file(path_test, label2id)
me_test_data = read_tsv_file(path_me_test, label2id)

with open('used_entities.pkl', 'rb') as f:
    final_used = pickle.load(f)

# create augmented train set
aug_data, _ = data_aug_replace(train_data, sentence_amount=300, used_entities = final_used)

# saving model name
model_name = "vesteinn/DanskBERT"

# convert to huggingface format
train_dataset = Dataset.from_list(aug_data)
test_dataset = Dataset.from_list(test_data)
me_test_dataset = Dataset.from_list(me_test_data)

# tokenize and align train dataset
tokenized_train_dataset = train_dataset.map( # tokenize train dataset and align labels with subword tokens
        tokenize_and_align_labels,
        batched = True,
        remove_columns=train_dataset.column_names
    )

# tokenize and align dev dataset
tokenized_test_dataset = test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=test_dataset.column_names
)

# tokenize and align ME dev dataset
tokenized_me_test_dataset = me_test_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=me_test_dataset.column_names
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
    num_train_epochs = 1,
    weight_decay = 0.01,
    remove_unused_columns=False
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
test_preds, test_labels, _ = trainer.predict(tokenized_test_dataset)

# predicting on augmented dev set
me_test_preds, me_test_labels, _ = trainer.predict(tokenized_me_test_dataset)

# predict max logit and convert to strings for non-augmented
_, test_predictions = pred2label((test_preds, test_labels), id2label)

# predict max logit and convert to strings for augmented
_, me_test_predictions = pred2label((me_test_preds, me_test_labels), id2label)

# write output file for predictions on dev sets
write_iob2_file(test_data, predictions = test_predictions, path = "final_test_pred.iob2")
write_iob2_file(me_test_data, predictions = me_test_predictions, path = "final_me_test_pred.iob2")