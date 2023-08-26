#%%
import csv
import math
import os
import re
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from paths import DATA_DIR, OUTPUT_DIR
from preprocess import clean_text

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
categories = [
    "Enquiry Product",
    "Enquiry Pricing",
    "Enquiry Promotion",
    "Enquiry Delivery Method",
    "Enquiry Order Method",
    "Enquiry Order Status",
    "Enquiry Return/Exchange",
    "Others",
]
n_classes = len(categories)
MAX_SEQ_LEN = 512
os.makedirs(DATA_DIR / "train", exist_ok=True)
os.makedirs(DATA_DIR / "test", exist_ok=True)


def load_data():
    data_file = DATA_DIR / "chatlog_label.csv"
    df = pd.read_csv(data_file)
    df = df.dropna(subset=["label 1", "label 2", "label 3", "label 4"], how="all")
    df = df[["text", "label 1", "label 2", "label 3", "label 4"]]

    df = df.reindex(columns=df.columns.tolist() + categories)
    # df.head()
    for category in categories:
        df[category] = df.apply(
            lambda row: 1
            if row["label 1"] == category
            or row["label 2"] == category
            or row["label 3"] == category
            or row["label 4"] == category
            else 0,
            axis=1,
        )
    df["text"] = df.apply(lambda row: clean_text(row["text"]), axis=1)
    df.to_csv(DATA_DIR / "train" / "classification_label.csv", index=False)
    return df


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_datasets(model_name="bert-base-multilingual-uncased"):
    train_data = DATA_DIR / "train" / "classification_label.csv"
    if not os.path.exists(train_data):
        df = load_data()
    else:
        df = pd.read_csv(train_data)

    df["one_hot_label"] = list(df[categories].values)
    df = df[["text", "one_hot_label"]]
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    df.head()
    train_texts = list(df["text"].values)
    train_labels = list(df["one_hot_label"].values)
    (
        train_texts,
        val_texts,
        train_labels,
        val_labels,
    ) = train_test_split(train_texts, train_labels, test_size=0.2, random_state=0)

    print(f"Train size: {len(train_texts)}")
    print(f"Val size: {len(val_texts)}")

    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN
    )

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    return train_dataset, val_dataset


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


def hamming_score(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(
            np.logical_or(y_true[i], y_pred[i])
        )
    return temp / y_true.shape[0]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sigmoid_v = np.vectorize(sigmoid)


def compute_metrics(pred, threshold=0.5):
    labels = pred.label_ids
    logits = pred.predictions
    probabilities = sigmoid_v(logits)

    for i in range(n_classes):
        precision = metrics.precision_score(
            labels[:, i], (probabilities[:, i] >= threshold).astype("int")
        )
        print(categories[i], precision)

    predictions = np.array(probabilities) > threshold

    _f1_score_micro = metrics.f1_score(labels, predictions, average="micro")
    _f1_score_macro = metrics.f1_score(labels, predictions, average="macro")
    _f1_score_weighted = metrics.f1_score(labels, predictions, average="weighted")

    _hamming_loss = metrics.hamming_loss(labels, predictions)
    _hamming_score = hamming_score(np.array(labels), np.array(predictions))
    _recall_micro = metrics.recall_score(labels, predictions, average="micro")
    _recall_macro = metrics.recall_score(labels, predictions, average="macro")
    _precision_micro = metrics.precision_score(labels, predictions, average="micro")
    _precision_macro = metrics.precision_score(labels, predictions, average="macro")

    return {
        "f1_score_micro": _f1_score_micro,
        "f1_score_macro": _f1_score_macro,
        "f1_score_weighted": _f1_score_weighted,
        "hamming_loss": _hamming_loss,
        "hamming_score": _hamming_score,
        "recall_micro": _recall_micro,
        "recall_macro": _recall_macro,
        "precision_micro": _precision_micro,
        "precision_macro": _precision_macro,
    }


#%%
def train(
    output_model: str,
    model_name="bert-base-multilingual-uncased",
    epochs=30,
    batch_size=8,
):
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes
    )

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        # learning_rate=5e-5,  # initial learning rate for AdamW optimizer
        warmup_steps=75,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,
        evaluation_strategy="steps",  # no, steps, epoch
        # evaluate every n steps, use with "steps" strategy
        eval_steps=800 / batch_size,
        output_dir=str(
            OUTPUT_DIR / "checkpoints" / output_model
        ),  # checkpoint directory
        # save_steps=100,  # checkpoint every n steps, ignored if load best at end
        load_best_model_at_end=True,  # True will make save_steps=eval_steps
        metric_for_best_model="eval_f1_score_macro",  # or eval_accuracy
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
    )
    # %% Train
    train_dataset, val_dataset = load_datasets(model_name=model_name)
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset,
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR / "checkpoints" / output_model / "best"))


#%%
if __name__ == "__main__":
    model_name = "bert-base-multilingual-uncased"
    output_model = f"{model_name}_css_livechat"
    train(output_model=output_model, model_name=model_name, epochs=20)

#%%
