"""
sentiment.py — Per-Turn Sentiment Classification
==================================================
Reads turns.csv from data/processed/, runs VADER sentiment scoring
on every turn, computes transcript-level sentiment aggregates, and
saves two output files:

  turns_sentiment.csv      — turns.csv + per-turn VADER scores
  transcripts_sentiment.csv — transcript-level sentiment aggregates
                              joined onto transcripts.csv metadata

WHY VADER:
  VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
  rule-based sentiment analyser built specifically for short,
  conversational text. It requires no GPU, no model download beyond
  a small lexicon, and runs the full corpus in seconds. It returns
  four scores per text:
    neg  — proportion of negative sentiment (0.0–1.0)
    neu  — proportion of neutral sentiment  (0.0–1.0)
    pos  — proportion of positive sentiment (0.0–1.0)
    compound — normalised aggregate score   (-1.0 to +1.0)
  The compound score is the primary signal we use downstream.
  Negative compound = negative sentiment, positive = positive,
  near zero = neutral.

WHAT THIS MODULE COMPUTES:
  Per turn:
    vader_neg, vader_neu, vader_pos, vader_compound

  Per transcript (used by scoring.py):
    customer_start_sentiment  — avg compound of first 2 customer turns
    customer_end_sentiment    — avg compound of last 2 customer turns
    customer_sentiment_delta  — end minus start (positive = improved)
    agent_mean_sentiment      — avg compound across all agent turns
    agent_min_sentiment       — worst agent turn (flags consistency dips)
    customer_mean_sentiment   — avg compound across all customer turns

  These six aggregates feed directly into the scoring model:
    customer_sentiment_delta → Customer sentiment trajectory (30%)
    agent_mean_sentiment     → Agent sentiment consistency (20%)

USAGE:
  python src/sentiment.py
  Run from repo root. Reads data/processed/turns.csv and
  data/processed/transcripts.csv. Outputs to data/processed/.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
TURNS_PATH = PROCESSED_DIR / "turns.csv"
TRANSCRIPTS_PATH = PROCESSED_DIR / "transcripts.csv"
TURNS_OUT = PROCESSED_DIR / "turns_sentiment.csv"
TRANSCRIPTS_OUT = PROCESSED_DIR / "transcripts_sentiment.csv"

# ---------------------------------------------------------------------------
# VADER setup
# ---------------------------------------------------------------------------

def load_vader() -> SentimentIntensityAnalyzer:
    """
    Load VADER. Downloads the lexicon if not already present.
    The lexicon is ~1MB and only downloads once — subsequent runs
    use the cached version.
    """
    try:
        analyser = SentimentIntensityAnalyzer()
        # Trigger a test call to confirm lexicon loaded correctly
        analyser.polarity_scores("test")
        return analyser
    except LookupError:
        print("  VADER lexicon not found — downloading now...")
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Per-turn scoring
# ---------------------------------------------------------------------------

def score_turns(turns_df: pd.DataFrame, analyser: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Add VADER scores to every row in turns_df.
    Returns turns_df with four new columns:
      vader_neg, vader_neu, vader_pos, vader_compound
    """
    print(f"  Scoring {len(turns_df):,} turns...")

    scores = turns_df["text"].apply(
        lambda text: analyser.polarity_scores(str(text))
    )

    scores_df = pd.DataFrame(scores.tolist(), index=turns_df.index)
    scores_df = scores_df.rename(columns={
        "neg": "vader_neg",
        "neu": "vader_neu",
        "pos": "vader_pos",
        "compound": "vader_compound",
    })

    return pd.concat([turns_df, scores_df], axis=1)


# ---------------------------------------------------------------------------
# Transcript-level aggregation
# ---------------------------------------------------------------------------

def aggregate_sentiment(turns_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transcript-level sentiment aggregates from scored turns.

    WHY START/END WINDOWS OF 2 TURNS:
      A single turn can be an outlier — a customer saying "okay"
      mid-call reads as neutral even in a hostile call. Averaging
      the first 2 and last 2 customer turns gives a more stable
      signal of where the call started and ended emotionally.
      With calls averaging 19 turns, 2-turn windows are small
      enough to be genuinely "start" and "end" without overlap
      in most transcripts.

    Returns a DataFrame indexed by transcript_id with six columns.
    """
    records = []

    for tid, group in turns_sentiment_df.groupby("transcript_id"):
        agent_turns = group[group["speaker"] == "AGENT"]["vader_compound"]
        customer_turns = group[group["speaker"] == "CUSTOMER"]["vader_compound"]

        # Start = first 2 customer turns, end = last 2
        customer_start = customer_turns.iloc[:2].mean() if len(customer_turns) >= 2 else customer_turns.mean()
        customer_end = customer_turns.iloc[-2:].mean() if len(customer_turns) >= 2 else customer_turns.mean()

        records.append({
            "transcript_id":           tid,
            "customer_start_sentiment": round(customer_start, 4),
            "customer_end_sentiment":   round(customer_end, 4),
            "customer_sentiment_delta": round(customer_end - customer_start, 4),
            "agent_mean_sentiment":     round(agent_turns.mean(), 4),
            "agent_min_sentiment":      round(agent_turns.min(), 4),
            "customer_mean_sentiment":  round(customer_turns.mean(), 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(turns_sentiment_df: pd.DataFrame, agg_df: pd.DataFrame) -> bool:
    print("\n  --- Validation ---")
    passed = True

    # All compound scores in valid range
    out_of_range = (
        (turns_sentiment_df["vader_compound"] < -1.0) |
        (turns_sentiment_df["vader_compound"] > 1.0)
    ).sum()
    if out_of_range:
        print(f"  FAIL: {out_of_range} compound scores outside [-1, 1]")
        passed = False
    else:
        print(f"  OK  : all compound scores in valid range")

    # No nulls in scored columns
    score_cols = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
    nulls = turns_sentiment_df[score_cols].isna().sum().sum()
    if nulls:
        print(f"  FAIL: {nulls} null values in VADER score columns")
        passed = False
    else:
        print(f"  OK  : no nulls in VADER score columns")

    # Aggregate row count matches transcript count
    unique_tids = turns_sentiment_df["transcript_id"].nunique()
    if len(agg_df) != unique_tids:
        print(f"  FAIL: {len(agg_df)} aggregate rows for {unique_tids} transcripts")
        passed = False
    else:
        print(f"  OK  : aggregate row count matches transcript count ({len(agg_df):,})")

    return passed


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(turns_sentiment_df: pd.DataFrame, agg_df: pd.DataFrame):
    print("\n  --- Summary ---")

    agent_turns = turns_sentiment_df[turns_sentiment_df["speaker"] == "AGENT"]
    customer_turns = turns_sentiment_df[turns_sentiment_df["speaker"] == "CUSTOMER"]

    print(f"  Agent turns scored    : {len(agent_turns):,}")
    print(f"  Customer turns scored : {len(customer_turns):,}")

    print(f"\n  Agent compound    — mean: {agent_turns['vader_compound'].mean():.3f}  "
          f"min: {agent_turns['vader_compound'].min():.3f}  "
          f"max: {agent_turns['vader_compound'].max():.3f}")
    print(f"  Customer compound — mean: {customer_turns['vader_compound'].mean():.3f}  "
          f"min: {customer_turns['vader_compound'].min():.3f}  "
          f"max: {customer_turns['vader_compound'].max():.3f}")

    print(f"\n  Customer sentiment delta (end - start):")
    print(f"    mean  : {agg_df['customer_sentiment_delta'].mean():.3f}")
    print(f"    positive (improved) : "
          f"{(agg_df['customer_sentiment_delta'] > 0).sum():,} transcripts")
    print(f"    negative (worsened) : "
          f"{(agg_df['customer_sentiment_delta'] < 0).sum():,} transcripts")
    print(f"    flat (< 0.05 change): "
          f"{(agg_df['customer_sentiment_delta'].abs() < 0.05).sum():,} transcripts")

    # Sanity check: resolved calls should skew positive delta
    if "resolution" in turns_sentiment_df.columns:
        resolved_ids = turns_sentiment_df[
            turns_sentiment_df["resolution"] == "resolved"
        ]["transcript_id"].unique()
        unresolved_ids = turns_sentiment_df[
            turns_sentiment_df["resolution"] == "unresolved"
        ]["transcript_id"].unique()

        resolved_delta = agg_df[
            agg_df["transcript_id"].isin(resolved_ids)
        ]["customer_sentiment_delta"].mean()
        unresolved_delta = agg_df[
            agg_df["transcript_id"].isin(unresolved_ids)
        ]["customer_sentiment_delta"].mean()

        print(f"\n  Sanity check — avg sentiment delta by resolution:")
        print(f"    resolved   : {resolved_delta:.3f}  (expect positive)")
        print(f"    unresolved : {unresolved_delta:.3f}  (expect negative or flat)")
        if resolved_delta > unresolved_delta:
            print(f"    OK  : resolved calls show better sentiment trajectory")
        else:
            print(f"    NOTE: resolved delta not higher than unresolved — "
                  f"VADER may need tuning for this domain")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("sentiment.py — VADER sentiment scoring\n")

    # Load inputs
    if not TURNS_PATH.exists():
        raise FileNotFoundError(
            f"{TURNS_PATH} not found. Run preprocessing.py first."
        )
    if not TRANSCRIPTS_PATH.exists():
        raise FileNotFoundError(
            f"{TRANSCRIPTS_PATH} not found. Run preprocessing.py first."
        )

    turns_df = pd.read_csv(TURNS_PATH)
    transcripts_df = pd.read_csv(TRANSCRIPTS_PATH)
    print(f"  Loaded {len(turns_df):,} turns from {TURNS_PATH}")

    # Load VADER
    analyser = load_vader()
    print(f"  VADER loaded")

    # Score turns
    turns_sentiment_df = score_turns(turns_df, analyser)

    # Aggregate to transcript level
    print(f"  Aggregating to transcript level...")
    agg_df = aggregate_sentiment(turns_sentiment_df)

    # Join aggregates onto transcripts metadata
    transcripts_sentiment_df = transcripts_df.merge(agg_df, on="transcript_id", how="left")

    # Validate
    ok = validate(turns_sentiment_df, agg_df)

    # Summary
    print_summary(turns_sentiment_df, agg_df)

    if not ok:
        print("\n  WARNING: validation failures — inspect before proceeding.")
    else:
        print("\n  Validation passed.")

    # Save main outputs
    turns_sentiment_df.to_csv(TURNS_OUT, index=False)
    transcripts_sentiment_df.to_csv(TRANSCRIPTS_OUT, index=False)

    print(f"\n  Saved: {TURNS_OUT}  ({len(turns_sentiment_df):,} rows)")
    print(f"  Saved: {TRANSCRIPTS_OUT}  ({len(transcripts_sentiment_df):,} rows)")

    # Save golden-set-only slice for HuggingFace comparison.
    # WHY GOLDEN ONLY: the comparison script needs ground truth labels
    # to evaluate which model aligns better with verified outcomes.
    # The volume set has programmatic labels — not reliable enough for
    # model evaluation. This slice is consumed by sentiment_hf_comparison.py
    # and is not used anywhere in the main pipeline.
    golden_turns = turns_sentiment_df[turns_sentiment_df["set"] == "golden"]
    golden_out = PROCESSED_DIR / "turns_sentiment_golden.csv"
    golden_turns.to_csv(golden_out, index=False)
    print(f"  Saved: {golden_out}  ({len(golden_turns):,} rows, golden set only)")
    print("\n  Phase 2 complete. Next: sentiment_hf_comparison.py, then intent.py")

    return turns_sentiment_df, transcripts_sentiment_df


if __name__ == "__main__":
    run()