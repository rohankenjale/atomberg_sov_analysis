import os, time, logging, math
from pathlib import Path
from typing import List, Dict

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

BRANDS = ["atomberg","orient","havells","crompton","polycab"]

def mention_flags(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    return {f"mention_{b}": (b in t) for b in BRANDS}

def to_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def collect_youtube_multi(
    queries: List[str],
    api_key: str,
    n_per_query: int = 30,
    out_dir: str = "data/raw",
    out_name: str = "youtube_sov_india",
    sleep_between_pages: float = 0.8
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    yt = build("youtube", "v3", developerKey=api_key)
    all_rows = []

    for q in queries:
        logging.info(f"[YouTube] Query: {q}")
        got = 0
        page_token = None
        seen_ids = set()

        while got < n_per_query:
            try:
                srch = yt.search().list(
                    q=q,
                    part="snippet",
                    type="video",
                    maxResults=min(50, n_per_query - got),
                    regionCode="IN",
                    relevanceLanguage="en",
                    order="relevance",
                    pageToken=page_token
                ).execute()
            except HttpError as e:
                logging.error(f"Search error for '{q}': {e}")
                break

            items = srch.get("items", [])
            if not items:
                break

            ids = []
            meta = {}
            for it in items:
                if it.get("id", {}).get("kind") != "youtube#video":
                    continue
                vid = it["id"]["videoId"]
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                sn = it.get("snippet", {})
                meta[vid] = {
                    "query": q,
                    "platform": "youtube",
                    "video_id": vid,
                    "title": sn.get("title",""),
                    "description": sn.get("description",""),
                    "channel_title": sn.get("channelTitle",""),
                    "published_at": sn.get("publishedAt",""),
                }
                ids.append(vid)

            if not ids:
                break

            try:
                vresp = yt.videos().list(
                    id=",".join(ids),
                    part="snippet,statistics"
                ).execute()
            except HttpError as e:
                logging.error(f"Videos error for '{q}': {e}")
                break

            for v in vresp.get("items", []):
                vid = v["id"]
                sn = v.get("snippet", {}) or {}
                st = v.get("statistics", {}) or {}
                tags = sn.get("tags", [])
                views = to_int(st.get("viewCount"))
                likes = to_int(st.get("likeCount"))
                comments = to_int(st.get("commentCount"))
                txt = f"{meta[vid]['title']} {meta[vid]['description']} {' '.join(tags)}"
                flags = mention_flags(txt)
                engagement_score = views + 5*likes + 10*comments

                row = {
                    **meta[vid],
                    "tags": "|".join(tags) if isinstance(tags, list) else "",
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "engagement_score": engagement_score,
                    **flags
                }
                all_rows.append(row)
                got += 1
                if got >= n_per_query:
                    break

            page_token = srch.get("nextPageToken")
            if not page_token:
                break
            time.sleep(sleep_between_pages)

    if not all_rows:
        logging.warning("No YouTube rows collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["video_id","query"])
    csv_path = os.path.join(out_dir, f"{out_name}.csv")
    parquet_path = os.path.join(out_dir, f"{out_name}.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logging.info(f"Saved {len(df)} YouTube rows to:\n  {csv_path}\n  {parquet_path}")
    return df

if __name__ == "__main__":
    load_dotenv()
    YT_KEY = os.getenv("YOUTUBE_API_KEY")

    QUERIES = ["smart fan", "energy efficient fan", "BLDC fan"]
    collect_youtube_multi(QUERIES, api_key=YT_KEY, n_per_query=30,
                          out_dir="data/raw", out_name="youtube_sov_india")
