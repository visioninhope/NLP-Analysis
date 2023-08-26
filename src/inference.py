#%%
import csv
import os

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from classifier import (MAX_SEQ_LEN, categories, clean_text, n_classes,
                        sigmoid_v)
from paths import DATA_DIR, OUTPUT_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
base_model = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(base_model)
classifier = str(OUTPUT_DIR / "checkpoints" / f"{base_model}_css_livechat" / "best")
model = BertForSequenceClassification.from_pretrained(
    classifier, num_labels=n_classes
).to(device)
#%%
def inference(model, tokenizer, text, threshold=0.5):
    # print(text)
    text = clean_text(text)
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    inputs.to(device)
    ids = inputs["input_ids"]
    logits = model(ids)[0][0]
    probabilities = sigmoid_v(logits.to("cpu").detach())

    # print(probabilities)
    pred = np.array(probabilities) >= threshold
    pred = [1 if x else 0 for i, x in enumerate(pred)]
    pred = list(np.argwhere(pred).flatten())
    pred_categories = [categories[i] for i in pred]
    return ",".join(pred_categories)


#%%
def evaluate(size=100):
    eval_data = DATA_DIR / "test" / "eval.csv"
    if not os.path.exists(eval_data):
        data_file = DATA_DIR / "chatlog_label.csv"
        df = pd.read_csv(data_file)

        eval_df = df[df["label 1"].isna()]
        eval_df = eval_df[["id", "text"]]
        eval_df.to_csv(
            DATA_DIR / "test" / "eval.csv", index=False, quoting=csv.QUOTE_ALL
        )
    else:
        eval_df = pd.read_csv(eval_data)

    eval_df = eval_df.dropna(subset=["text"], how="all")
    eval_df = eval_df[: size - 1]
    # print(eval_df.head())

    tests = sum(eval_df.loc[:, ["text"]].values.tolist(), [])
    test_results = []
    for test in tests:
        _result = inference(model, tokenizer, test)
        test_results.append(_result)
    eval_df["prediction"] = np.array(test_results)
    print(eval_df.head())
    eval_df.to_csv(
        DATA_DIR / "test" / f"eval_results_{'all' if size == 0 else size}.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
    )


if __name__ == "__main__":
    evaluate()
#%%
