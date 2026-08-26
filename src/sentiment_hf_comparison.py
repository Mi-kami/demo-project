"""
sentiment_hf_comparison.py — VADER vs HuggingFace Comparison
=============================================================
Runs DistilBERT sentiment classification on the golden set turns
and produces a side-by-side comparison with VADER scores.

WHY THIS IS A SEPARATE SCRIPT (not part of sentiment.py):
  The main pipeline uses VADER — fast, no GPU required, runs the full
  corpus in seconds. This script is an analytical artefact: it exists
  to evaluate whether a transformer-based model produces meaningfully
  different (or better) sentiment signals on our specific domain.
  Keeping it separate means:
    1. The pipeline has no HuggingFace dependency — it runs cleanly
       without a 250MB model download.
    2. The comparison is explicitly labelled as evaluation, not
       production scoring.
    3. On interview day, the demo works regardless of whether
       HuggingFace loads successfully.

WHY GOLDEN SET ONLY:
  The golden set has hand-verified ground truth labels (resolution,
  sentiment_arc). This is the only place where we can ask "which
  model's sentiment scores align better with what we know to be true?"
  Running the comparison on the volume set would test throughput, not
  accuracy — that's a different question.

MODEL: distilbert-base-uncased-finetuned-sst-2-english
  A distilled version of BERT fine-tuned on SST-2 (Stanford Sentiment
  Treebank). Returns POSITIVE/NEGATIVE with a confidence score.
  Faster than full BERT, still meaningful for comparison.
  Limitation: fine-tuned on English movie reviews — our domain
  (African/Gulf fintech calls with code-switching) is out of
  distribution. This limitation is itself useful interview material.

OUTPUTS:
  data/processed/hf_comparison.csv  — turn-level comparison table
  Printed summary comparing model agreement rates by archetype
  and resolution status.

USAGE:
  pip install transformers torch
  python src/sentiment_hf_comparison.py
  Run from repo root AFTER sentiment.py has been run.
  Expect 3-5 minutes on CPU — ~3,000 turns through DistilBERT.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
GOLDEN_TURNS_PATH = PROCESSED_DIR / "turns_sentiment_golden.csv"
TRANSCRIPTS_SENTIMENT_PATH = PROCESSED_DIR / "transcripts_sentiment.csv"
HF_OUT = PROCESSED_DIR / "hf_comparison.csv"


# ---------------------------------------------------------------------------
# HuggingFace model loader
# ---------------------------------------------------------------------------

def load_hf_model():
    """
    Load DistilBERT sentiment pipeline.
    Imports are inside this function so the rest of the script
    remains importable even if transformers is not installed.
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers not installed. Run:\n"
            "  pip install transformers torch\n"
            "then re-run this script."
        )

    print("  Loading distilbert-base-uncased-finetuned-sst-2-english...")
    print("  (first run downloads ~250MB — subsequent runs use cache)")

    classifier = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,      # truncate texts longer than 512 tokens
        max_length=512,
    )
    print("  Model loaded.")
    return classifier


# ---------------------------------------------------------------------------
# HuggingFace scoring
# ---------------------------------------------------------------------------

def score_with_hf(turns_df: pd.DataFrame, classifier) -> pd.DataFrame:
    """
    Run DistilBERT on every turn in turns_df.

    HuggingFace returns: [{"label": "POSITIVE", "score": 0.998}, ...]
    We convert this to a compound-style score on [-1, +1] so it's
    directly comparable to VADER's compound score:
      POSITIVE confidence  ->  +score
      NEGATIVE confidence  ->  -score

    WHY THIS NORMALISATION:
      VADER compound is on [-1, +1]. DistilBERT returns a 0-1
      confidence score for a binary label. Mapping POSITIVE->+score
      and NEGATIVE->-score puts both on the same scale so the
      comparison table is readable without mental conversion.

    Processes in batches of 32 for memory efficiency.
    """
    texts = turns_df["text"].astype(str).tolist()
    batch_size = 32
    results = []

    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)
        if (i // batch_size) % 10 == 0:
            print(f"    {min(i + batch_size, total):,} / {total:,} turns scored")

    hf_labels = [r["label"] for r in results]
    hf_scores = [r["score"] for r in results]

    # Normalise to [-1, +1]
    hf_compound = [
        score if label == "POSITIVE" else -score
        for label, score in zip(hf_labels, hf_scores)
    ]

    out = turns_df.copy()
    out["hf_label"] = hf_labels
    out["hf_confidence"] = [round(s, 4) for s in hf_scores]
    out["hf_compound"] = [round(c, 4) for c in hf_compound]

    return out


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------

def vader_label(compound: float) -> str:
    """Convert VADER compound score to a POSITIVE/NEGATIVE/NEUTRAL label."""
    if compound >= 0.05:
        return "POSITIVE"
    if compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def compare_models(comparison_df: pd.DataFrame) -> None:
    """
    Print a structured comparison between VADER and HuggingFace scores.

    Key questions we answer:
      1. Overall agreement rate — how often do the two models agree
         on the sentiment direction of a turn?
      2. Agreement by speaker — do they diverge more on agent or
         customer turns?
      3. Agreement by archetype — which call types show the most
         disagreement? (useful for understanding domain mismatch)
      4. Compound score correlation — Pearson r between vader_compound
         and hf_compound. Higher = more consistent signal.
      5. Disagreement on resolved vs unresolved — if VADER disagrees
         with HuggingFace more on one resolution type, that tells us
         something about which model handles the domain better.
    """
    print("\n  --- VADER vs HuggingFace Comparison ---")

    # Add VADER label column for agreement check
    df = comparison_df.copy()
    df["vader_label"] = df["vader_compound"].apply(vader_label)

    # HuggingFace doesn't produce NEUTRAL — map VADER NEUTRAL to
    # whichever HF label it's closest to for agreement purposes
    def agreement(row):
        v = row["vader_label"]
        h = row["hf_label"]
        if v == "NEUTRAL":
            # Neutral is ambiguous — count as partial agreement
            return "neutral_ambiguous"
        return "agree" if v == h else "disagree"

    df["agreement"] = df.apply(agreement, axis=1)

    # 1. Overall agreement
    agree_count = (df["agreement"] == "agree").sum()
    disagree_count = (df["agreement"] == "disagree").sum()
    neutral_count = (df["agreement"] == "neutral_ambiguous").sum()
    total = len(df)

    print(f"\n  Overall (excluding VADER NEUTRAL turns):")
    non_neutral = total - neutral_count
    print(f"    Agree    : {agree_count:,} / {non_neutral:,}  "
          f"({agree_count/non_neutral*100:.1f}%)")
    print(f"    Disagree : {disagree_count:,} / {non_neutral:,}  "
          f"({disagree_count/non_neutral*100:.1f}%)")
    print(f"    VADER NEUTRAL (excluded): {neutral_count:,} turns")

    # 2. Agreement by speaker
    print(f"\n  Agreement rate by speaker:")
    for speaker in ["AGENT", "CUSTOMER"]:
        sp = df[(df["speaker"] == speaker) & (df["agreement"] != "neutral_ambiguous")]
        rate = (sp["agreement"] == "agree").mean() * 100
        print(f"    {speaker:<10} {rate:.1f}%")

    # 3. Compound correlation
    corr = df["vader_compound"].corr(df["hf_compound"])
    print(f"\n  Compound score correlation (Pearson r): {corr:.3f}")
    if corr >= 0.7:
        print(f"    Strong agreement on sentiment direction")
    elif corr >= 0.4:
        print(f"    Moderate agreement — models diverge on ambiguous turns")
    else:
        print(f"    Weak agreement — domain mismatch likely a factor")

    # 4. Agreement by archetype
    print(f"\n  Agreement rate by archetype:")
    for arch, grp in df[df["agreement"] != "neutral_ambiguous"].groupby("archetype"):
        rate = (grp["agreement"] == "agree").mean() * 100
        print(f"    {arch:<35} {rate:.1f}%")

    # 5. Agreement by resolution
    print(f"\n  Agreement rate by resolution:")
    for res, grp in df[df["agreement"] != "neutral_ambiguous"].groupby("resolution"):
        rate = (grp["agreement"] == "agree").mean() * 100
        print(f"    {res:<12} {rate:.1f}%")

    # 6. Biggest disagreements — useful for interview discussion
    print(f"\n  Largest compound score gaps (top 5 disagreements):")
    df["compound_gap"] = (df["vader_compound"] - df["hf_compound"]).abs()
    top_gaps = df.nlargest(5, "compound_gap")[
        ["transcript_id", "speaker", "vader_compound", "hf_compound",
         "compound_gap", "text"]
    ]
    for _, row in top_gaps.iterrows():
        print(f"    {row['transcript_id']} | {row['speaker']} | "
              f"VADER {row['vader_compound']:+.3f} vs HF {row['hf_compound']:+.3f} | "
              f"{str(row['text'])[:60]}...")

    # 7. Recommendation
    print(f"\n  --- Recommendation ---")
    if corr >= 0.6 and agree_count / max(non_neutral, 1) >= 0.65:
        print(f"  Both models produce consistent signals on this domain.")
        print(f"  VADER is preferred for the pipeline: faster, no GPU,")
        print(f"  negligible accuracy trade-off for this use case.")
    else:
        print(f"  Models diverge meaningfully — likely due to domain mismatch.")
        print(f"  DistilBERT was fine-tuned on English movie reviews (SST-2),")
        print(f"  not African/Gulf fintech calls with code-switching.")
        print(f"  VADER's rule-based approach handles short conversational")
        print(f"  text without domain fine-tuning — better fit here.")
        print(f"  With real Aethex data, fine-tuning a transformer on")
        print(f"  in-domain transcripts would be the next step.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("sentiment_hf_comparison.py — VADER vs HuggingFace\n")

    if not GOLDEN_TURNS_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_TURNS_PATH} not found. Run sentiment.py first."
        )

    turns_df = pd.read_csv(GOLDEN_TURNS_PATH)
    print(f"  Loaded {len(turns_df):,} golden set turns from {GOLDEN_TURNS_PATH}")
    print(f"  Transcripts: {turns_df['transcript_id'].nunique()}")

    # Load and run HuggingFace
    classifier = load_hf_model()
    print(f"\n  Scoring {len(turns_df):,} turns with DistilBERT...")
    comparison_df = score_with_hf(turns_df, classifier)

    # Analysis
    compare_models(comparison_df)

    # Save
    comparison_df.to_csv(HF_OUT, index=False)
    print(f"\n  Saved: {HF_OUT}  ({len(comparison_df):,} rows)")
    print("\n  Comparison complete. Pipeline uses VADER — see recommendation above.")

    return comparison_df


if __name__ == "__main__":
    run()