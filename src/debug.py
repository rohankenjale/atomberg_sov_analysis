import pandas as pd
from utils import brand_flags

PROCESSED_DIR = "data/processed"

df = pd.read_parquet(f"{PROCESSED_DIR}/combined_processed.parquet")

youtube_df = df[df["platform"] == "youtube"].copy()

youtube_df["text"] = youtube_df["title"].fillna("") + " " + youtube_df["description"].fillna("")

flags_df = youtube_df["text"].apply(brand_flags).apply(pd.Series)

print("YouTube Brand Flags:")
print(flags_df.head())

print("Number of mentions per brand:")
print(flags_df.sum())
