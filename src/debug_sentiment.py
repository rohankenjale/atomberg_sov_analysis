import pandas as pd
from utils import sentiment_label_and_score

PROCESSED_DIR = "data/processed"

df = pd.read_parquet(f"{PROCESSED_DIR}/combined_processed.parquet")

youtube_df = df[df["platform"] == "youtube"].copy()

youtube_df["text"] = youtube_df["title"].fillna("") + " " + youtube_df["description"].fillna("")

youtube_df["sentiment"] = youtube_df["text"].apply(sentiment_label_and_score)

for index, row in youtube_df.head(10).iterrows():
    print(f"Text: {row['text']}")
    print(f"Sentiment: {row['sentiment']}")
    print("-----")
