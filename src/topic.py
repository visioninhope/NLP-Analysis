# %%
import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from pprint import pprint
from time import time
from typing import List

import jieba
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from stopwordsiso import stopwords
from umap import UMAP

from paths import DATA_DIR
from preprocess import clean_text

# %%
UMAP_MODEL = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    low_memory=False,
    random_state=42,
)
zh_stopwords = list(stopwords("zh"))
custom_stopwords = []
all_stopwords = text.ENGLISH_STOP_WORDS.union(zh_stopwords + custom_stopwords)

#%%
model_name = "paraphrase-multilingual-mpnet-base-v2"
num_topic = "auto"
ngram = 2
cv = CountVectorizer(
    ngram_range=(1, ngram), stop_words=all_stopwords, tokenizer=jieba.lcut
)
embedding_model = model_name
model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=cv,
    nr_topics=num_topic,
    umap_model=UMAP_MODEL,
    verbose=False,
    min_topic_size=3,  # >=3
)

# %% Topic extraction function
def extract_topics(docs, model):
    size = len(docs)
    doc_topics_id, probabilities = model.fit_transform(docs)
    df = model.get_topic_info()

    df["keywords"] = df["Name"].str.split("_", 1)
    df[["topic", "keywords"]] = pd.DataFrame(df["keywords"].tolist(), index=df.index)
    df["keywords"] = df["keywords"].str.split("_")
    df.drop("Name", axis=1, inplace=True)
    # df.set_index("Topic", inplace=True)
    df = df.loc[df["topic"] != "-1"]
    df = df[["topic", "keywords", "Count"]]
    print(df.head())
    df.to_csv(
        DATA_DIR / f"topic_results_{size}.csv", index=False, quoting=csv.QUOTE_ALL
    )
    return df


def clean_result():
    raw_result = DATA_DIR / "topic_results_57128.csv"
    result_df = pd.read_csv(raw_result)
    result_df["Keywords"] = result_df.apply(
        lambda row: list(
            set([k for k in re.split(r"[_ ]", row["Name"])[1:] if len(k) > 1])
        ),
        axis=1,
    )
    result_df = result_df[["Keywords", "Count"]]
    result_df.head()
    result_df.to_csv(
        DATA_DIR / f"topic_results_clean.csv", index=False, quoting=csv.QUOTE_ALL
    )


#%%
def main(size=1000):
    data_file = DATA_DIR / "chatlogs_qna.csv"
    df = pd.read_csv(data_file)

    query_df = df.dropna(subset=["is_customer"], how="all")
    query_df.head()
    #%%
    queries = list(set(sum(query_df.loc[:, ["text"]].values.tolist(), [])))
    queries = [clean_text(q) for q in queries if len(q) > 1]
    print(len(queries))
    #%%
    extract_topics(docs=queries[:size], model=model)


# %%
if __name__ == '__main__':
    main()