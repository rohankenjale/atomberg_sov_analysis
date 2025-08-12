import re
from typing import Dict, List, Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

try:
    _ = nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

BRAND_ALIASES: Dict[str, List[str]] = {
    "Atomberg":   [r"atomberg"],
    "Orient":     [r"orient electric", r"\borient\b"],
    "Havells":    [r"havells", r"havell'?s"],
    "Crompton":   [r"crompton"],
    "Polycab":    [r"polycab"],
}

BRAND_PATTERNS: Dict[str, re.Pattern] = {
    b: re.compile(r"(?i)\b(" + r"|".join(aliases) + r")\b") for b, aliases in BRAND_ALIASES.items()
}

def extract_brands(text: str) -> List[str]:

    if not isinstance(text, str) or not text:
        return []
    t = text.strip().lower()
    hits = []
    for brand, pat in BRAND_PATTERNS.items():
        if pat.search(t):
            hits.append(brand)
    return hits

def brand_flags(text: str) -> Dict[str, bool]:

    mentions = set(extract_brands(text))
    return {f"mention_{b.lower()}": (b in mentions) for b in BRAND_ALIASES.keys()}

def sentiment_label_and_score(text: str) -> Tuple[str, float]:

    if not isinstance(text, str) or not text.strip():
        return ("neutral", 0.0)
    s = sia.polarity_scores(text)
    c = s.get("compound", 0.0)
    if c > 0.05:
        return ("positive", c)
    elif c < -0.05:
        return ("negative", c)
    else:
        return ("neutral", c)
