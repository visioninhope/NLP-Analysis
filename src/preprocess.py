#%%
import csv
import json
import re
from pprint import pprint

import opencc
import pandas as pd

from paths import DATA_DIR
from utility import REGEX_URL

CHAT_DIR = DATA_DIR / "chat_logs"
TICKET_DIR = DATA_DIR / "tickets"

#%%
REGEX_EMOJI = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)

tc = opencc.OpenCC("s2t.json")


def clean_text(text):
    text = re.sub(r"\[.+\]", "", text)
    text = re.sub(REGEX_EMOJI, "", text)
    text = tc.convert(text)
    return text


def process_chat_log(chat_log_file):
    with open(chat_log_file, "r", encoding="utf-8") as f:
        chat_log = json.loads(f.read())
    #%%
    data = []
    for chat in chat_log["chats"]:
        data.append(
            {
                "id": chat["id"],
                "text": [
                    (
                        (
                            "[cs]"
                            if event["author_id"] == "agent@chowsangsang.com"
                            else ""
                        )
                        + re.sub(r"\s+", " ", re.sub(REGEX_URL, "[url]", event["text"]))
                    ).strip()
                    for event in chat["thread"]["events"]
                    if event["type"] == "message"
                    and (
                        "properties" in event
                        and (
                            "lc2" not in event["properties"]
                            or "welcome_message" not in event["properties"]["lc2"]
                        )
                    )
                ],
            }
        )

    data = [
        {"id": d["id"], "text": ("\n".join(d["text"]).strip())}
        for d in data
        if len(d["text"]) > 1
    ]
    # pprint(data)
    # print(len(data))
    return data


#%%
def classification_data():
    chat_logs = list(CHAT_DIR.glob("*.json"))
    data = sum([process_chat_log(log) for log in chat_logs], [])
    print(len(data))
    #%%
    df = pd.DataFrame.from_dict(data)
    df.head()
    df.to_csv(DATA_DIR / "chatlogs.csv", index=False, quoting=csv.QUOTE_ALL)


#%%
def process_chat_log_qna(chat_log_file):
    with open(chat_log_file, "r", encoding="utf-8") as f:
        chat_log = json.loads(f.read())
    #%%
    data = []
    for chat in chat_log["chats"]:
        chat_id = chat["id"]
        events = [
            event
            for event in chat["thread"]["events"]
            if event["type"] == "message"
            and (
                "properties" in event
                and (
                    "lc2" not in event["properties"]
                    or "welcome_message" not in event["properties"]["lc2"]
                )
            )
        ]
        for event in events:
            entry = {
                "chat_id": chat_id,
                "text": re.sub(r"\s+", " ", re.sub(REGEX_URL, "[url]", event["text"])),
            }
            if event["author_id"] != "agent@chowsangsang.com":
                entry["is_customer"] = 1
            if len(entry["text"]) > 1:
                data.append(entry)

    return data


def qna_data():
    chat_logs = list(CHAT_DIR.glob("*.json"))
    data = sum([process_chat_log_qna(log) for log in chat_logs], [])
    print(len(data))
    #%%
    df = pd.DataFrame.from_dict(data)
    df.head()
    df.to_csv(DATA_DIR / "chatlogs_qna.csv", index=False, quoting=csv.QUOTE_ALL)

#%%

if __name__ == '__main__':
    classification_data()
    qna_data()
