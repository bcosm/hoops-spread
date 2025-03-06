# Merges Reddit sentiment analysis data with game features for both teams
import pandas as pd
from pathlib import Path

SENT_FILE = Path("school_sentiment_timeseries_v3_high_throughput.csv")
FEAT_FILE = Path("data/features/ncaab_cumulative_features_v12_team_names_cleaned.csv")
OUT_FILE = Path("data/features/ncaab_cumulative_features_v13_sentiment_enhanced.csv")

sent = (
    pd.read_csv(SENT_FILE, parse_dates=["date"])
      .loc[:, ["date", "school_name",
               "sentiment_ema", "confidence_ema",
               "sentiment_normalized", "confidence_normalized"]]
      .sort_values(["school_name", "date"])
)

feat = pd.read_csv(FEAT_FILE, parse_dates=["date"])

# Standardizes team names by removing punctuation and normalizing case
def clean_names(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"[^\w\s]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.lower()
    )

sent["school"] = clean_names(sent["school_name"])
feat["visitor_school"] = clean_names(feat["visitor_team"])
feat["home_school"] = clean_names(feat["home_team"])

fill_cols = ["sentiment_ema", "confidence_ema",
             "sentiment_normalized", "confidence_normalized"]

sent[fill_cols] = (
    sent.sort_values(["school", "date"])
        .groupby("school")[fill_cols]
        .transform(lambda g: g.ffill())
        .fillna(0.0)
)

sent_prev = sent.copy()
sent_prev["date"] += pd.Timedelta(days=1)

visitor_cols = {
    "sentiment_ema": "visitor_sentiment_ema",
    "confidence_ema": "visitor_confidence_ema",
    "sentiment_normalized": "visitor_sentiment_norm",
    "confidence_normalized": "visitor_confidence_norm",
}

feat = feat.merge(
    sent_prev.rename(columns=visitor_cols)
             .rename(columns={"school": "visitor_school"}),
    how="left",
    left_on=["date", "visitor_school"],
    right_on=["date", "visitor_school"]
)

home_cols = {
    "sentiment_ema": "home_sentiment_ema",
    "confidence_ema": "home_confidence_ema", 
    "sentiment_normalized": "home_sentiment_norm",
    "confidence_normalized": "home_confidence_norm",
}

feat = feat.merge(
    sent_prev.rename(columns=home_cols)
             .rename(columns={"school": "home_school"}),
    how="left",
    left_on=["date", "home_school"],
    right_on=["date", "home_school"]
)

for col in list(visitor_cols.values()) + list(home_cols.values()):
    feat[col] = feat[col].fillna(0.0)

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
feat.to_csv(OUT_FILE, index=False)
