#%%
import csv
import re

import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import DATA_DIR
from preprocess import clean_text

#%%
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
labels = ["negative", "neutral", "positive"]
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#%%
MAX_SEQ_LEN = 512


def inference(text):
    encoded_input = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQ_LEN
    )
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return labels[np.argmax(scores)]


# inference("good night")
#%%
data_file = DATA_DIR / "chatlogs_qna.csv"
df = pd.read_csv(data_file)

query_df = df.dropna(subset=["is_customer"], how="all")
query_df.head()
#%%
queries = list(set(sum(query_df.loc[:, ["text"]].values.tolist(), [])))
queries = [clean_text(q) for q in queries if len(q) > 1 and len(q) < MAX_SEQ_LEN]

REGEX_ZH = re.compile(r"[\u4e00-\u9fff]+", re.UNICODE)
en_queries = [q for q in queries if not re.match(REGEX_ZH, q)]

en_queries = sorted(en_queries, key=lambda q: len(q), reverse=True)
size = 1000
en_queries = en_queries[:size]
print(len(en_queries))

results = []
for query in en_queries:
    _sentiment = inference(query)
    results.append([query, _sentiment])
result_df = pd.DataFrame(results, columns=["query", "sentiment"])
result_df.to_csv(
    DATA_DIR / f"sentiment_results_{size}.csv", index=False, quoting=csv.QUOTE_ALL
)
#%%
