from scripts.train_pred import tokenize_and_align_labels, pred2label, label_mapping

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig, AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset
import torch

# path to test sets
path_test = "data/no_overlap_da_news/da_news_test.tsv"
path_me_test = "data/me_data/middle_eastern_dev.tsv"  

# saving model name
model_name = "vesteinn/DanskBERT"

# creating the label to id mapping 
label2id, id2label = label_mapping(path_test)

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