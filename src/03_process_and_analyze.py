import os
from pathlib import Path
import pandas as pd

from utils import brand_flags, sentiment_label_and_score

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = ["smart fan", "energy efficient fan", "BLDC fan"]
CANONICAL_BRANDS = ["Atomberg","Orient","Havells","Crompton","Polycab"]


def compute_engagement(row):

    if "engagement_score" in row and pd.notnull(row["engagement_score"]):
        return float(row["engagement_score"])

    return 0.0

def load_google():

    paths = [
        RAW_DIR / "google_sov_india.parquet",
        RAW_DIR / "google_sov_india.csv",
    ]
    p = next((x for x in paths if x.exists()), None)
    if not p:
        return pd.DataFrame()
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    keep = {
        "query":"query",
        "title":"title",
        "description":"description",
        "url":"url",
        "platform":"platform",
    }
    out = df.rename(columns={k:v for k,v in keep.items() if k in df.columns})
    out["platform"] = "google"
    out["engagement_score"] = 0.0
    return out

def load_youtube():
    paths = [
        RAW_DIR / "youtube_sov_india.parquet",
        RAW_DIR / "youtube_sov_india.csv",
    ]
    p = next((x for x in paths if x.exists()), None)
    if not p:
        return pd.DataFrame()
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    df["platform"] = "youtube"
    df["url"] = df.get("video_id", "")
    return df[["query","title","description","url","platform","engagement_score"]].copy()

def enrich_flags_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    txt = (df.get("title","").fillna("") + " " + df.get("description","").fillna("")).astype(str)

    sent = txt.apply(sentiment_label_and_score)
    df["sentiment_label"] = sent.apply(lambda x: x[0])
    df["sentiment_score"] = sent.apply(lambda x: x[1])


    for b in CANONICAL_BRANDS:
        col = f"mention_{b.lower()}"
        if col in df.columns:
            df = df.drop(columns=[col])

    flags_df = txt.apply(brand_flags).apply(pd.Series)
    df = pd.concat([df, flags_df], axis=1)

    return df


def pivot_sov(df: pd.DataFrame, level_cols):
    """
    Computes SoV by Mentions, SoV by Engagement, and SoPV (Share of Positive Voice)
    grouped by `level_cols` (e.g., ['query'] or ['query','platform']).
    """
    rows = []
    for keys, grp in df.groupby(level_cols):

        m = {}
        for b in CANONICAL_BRANDS:
            col = f"mention_{b.lower()}"
            if col in grp.columns:
                m[b] = int(grp[col].sum())
            else:
                m[b] = 0
        total_mentions = sum(m.values()) or 1


        e = {}

        grp_e = grp[grp["engagement_score"] > 0]
        for b in CANONICAL_BRANDS:
            col = f"mention_{b.lower()}"
            e[b] = float(grp_e.loc[grp_e[col] == True, "engagement_score"].sum())
        total_eng = sum(e.values()) or 1.0

        p = {}
        grp_pos = grp[grp["sentiment_label"] == "positive"]
        for b in CANONICAL_BRANDS:
            col = f"mention_{b.lower()}"
            p[b] = int(grp_pos[col].sum()) if col in grp_pos.columns else 0
        total_pos = sum(p.values()) or 1

        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {level_cols[i]: keys[i] for i in range(len(level_cols))}
        for b in CANONICAL_BRANDS:
            rows.append({
                **base,
                "brand": b,
                "mentions": m[b],
                "sov_mentions_pct": 100.0 * m[b] / total_mentions,
                "engagement": e[b],
                "sov_engagement_pct": 100.0 * (e[b] / total_eng) if total_eng > 0 else 0.0,
                "positive_mentions": p[b],
                "sopv_pct": 100.0 * p[b] / total_pos
            })
    return pd.DataFrame(rows)

def process_and_analyze():
    print("Starting data processing and analysis...")


    google_df = load_google()
    youtube_df = load_youtube()
    if google_df.empty and youtube_df.empty:
        print("Error: No raw data found. Run collectors first.")
        return
    print(f"Loaded: google={len(google_df)}, youtube={len(youtube_df)}")

    combined = pd.concat([google_df, youtube_df], ignore_index=True, sort=False)
    if "query" not in combined.columns:

        combined["query"] = "smart fan"
    
    combined = combined[combined["query"].isin(QUERIES)] if not combined["query"].isna().all() else combined

    if "engagement" in combined.columns and "engagement_score" not in combined.columns:
        combined = combined.rename(columns={"engagement": "engagement_score"})
    elif "engagement" in combined.columns and "engagement_score" in combined.columns:
        combined = combined.drop(columns=["engagement"])


    print("Enriching brand flags and sentiment...")
    combined = enrich_flags_and_sentiment(combined)

    combined = combined.loc[:, ~combined.columns.duplicated()]
    processed_path = PROCESSED_DIR / "combined_processed.parquet"
    combined.to_parquet(processed_path, index=False)
    combined.to_csv(PROCESSED_DIR / "combined_processed.csv", index=False)

    print(f"Processed data saved to {processed_path}")


    print("Computing SoV tables...")
    
    sov_by_query = pivot_sov(combined, ["query"]).sort_values(["query","brand"])
    
    sov_by_query_platform = pivot_sov(combined, ["query","platform"]).sort_values(["query","platform","brand"])

    sov_by_query.to_csv(RESULTS_DIR / "sov_by_query.csv", index=False)
    sov_by_query_platform.to_csv(RESULTS_DIR / "sov_by_query_platform.csv", index=False)

    sent_dist = (
        combined.assign(any_brand=combined[[f"mention_{b.lower()}" for b in CANONICAL_BRANDS]].any(axis=1))
                .loc[lambda d: d["any_brand"]]
                .groupby(["query","sentiment_label"])
                .size()
                .reset_index(name="count")
    )
    sent_dist.to_csv(RESULTS_DIR / "sentiment_distribution.csv", index=False)

    print("Data processing and analysis complete.")
    print(f"Saved: \n  {RESULTS_DIR/'sov_by_query.csv'}\n  {RESULTS_DIR/'sov_by_query_platform.csv'}\n  {RESULTS_DIR/'sentiment_distribution.csv'}")

if __name__ == "__main__":
    process_and_analyze()
