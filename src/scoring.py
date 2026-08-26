"""
scoring.py — Agent Performance Scorer
=======================================
Reads transcripts_resolution.csv (which contains all upstream fields from
preprocessing, sentiment, intent, and resolution phases) and computes a
final agent performance score (0–100) from four weighted components.

SCORING MODEL (weights visible — not a black box):
  Resolution achieved          40%  — binary: resolved=100, unresolved=0
  Customer sentiment trajectory 30%  — did customer sentiment improve?
  Agent sentiment consistency   20%  — did agent maintain positive tone?
  Call efficiency               10%  — fewer turns = faster = better (inverted)

ALL COMPONENTS NORMALISED TO [0, 100] BEFORE WEIGHTING.

NORMALISATION STRATEGY:
  All continuous components (sentiment delta, agent mean sentiment, turn
  count) are normalised against the empirical min/max observed in the
  corpus — not the theoretical VADER range of [-1, +1]. VADER's realistic
  range on short conversational text is much narrower than ±1; a raw
  linear scale against theoretical bounds would compress the entire corpus
  into a narrow band, losing discriminative power on 60% of the score.

  Bounds are computed once from the full corpus and saved to
  data/processed/score_bounds.json so app.py can score single new
  transcripts against the same reference bounds.

NARRATIVE SUMMARY:
  Generated from component scores via conditional logic — not from an LLM.
  Consistent with the pipeline's "math scores, math narrates" principle.

OUTPUTS:
  data/processed/transcripts_scored.csv
  data/processed/score_bounds.json

USAGE:
  python src/scoring.py
  Run from repo root AFTER resolution.py has been run.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR  = Path("data/processed")
TRANSCRIPTS_IN = PROCESSED_DIR / "transcripts_resolution.csv"
SCORED_OUT     = PROCESSED_DIR / "transcripts_scored.csv"
BOUNDS_OUT     = PROCESSED_DIR / "score_bounds.json"

# Scoring weights — visible here and mirrored in the UI
WEIGHTS = {
    "resolution":  0.40,
    "sentiment":   0.30,
    "consistency": 0.20,
    "efficiency":  0.10,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(value: float, min_val: float, max_val: float, invert: bool = False) -> float:
    """
    Linear normalisation to [0, 100], clipped to that range.

    If invert=True, lower input values produce higher scores — used for
    turn count: fewer turns = faster resolution = better performance.

    Degenerate case: if min == max (all values identical), return 50.0
    rather than divide-by-zero. This can't happen in practice with our
    corpus but is correct to handle.
    """
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100.0
    if invert:
        score = 100.0 - score
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Bounds computation
# ---------------------------------------------------------------------------

def compute_bounds(df: pd.DataFrame) -> dict:
    """
    Compute empirical min/max for each continuous scoring component from the
    full corpus. Saved to JSON so app.py can normalise single new transcripts
    against the same reference bounds — ensuring scores are stable and
    reproducible regardless of how many transcripts are in any given batch.
    """
    return {
        "customer_sentiment_delta": {
            "min": float(df["customer_sentiment_delta"].min()),
            "max": float(df["customer_sentiment_delta"].max()),
        },
        "agent_mean_sentiment": {
            "min": float(df["agent_mean_sentiment"].min()),
            "max": float(df["agent_mean_sentiment"].max()),
        },
        "turns": {
            "min": float(df["turns"].min()),
            "max": float(df["turns"].max()),
        },
    }


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def score_resolution(predicted_resolution: str) -> float:
    """Binary: resolved = 100, unresolved = 0."""
    return 100.0 if predicted_resolution == "resolved" else 0.0


def score_sentiment(delta: float, bounds: dict) -> float:
    """
    Normalise customer_sentiment_delta against corpus range.
    Higher delta (sentiment improved more) → higher score.
    """
    b = bounds["customer_sentiment_delta"]
    return normalise(delta, b["min"], b["max"])


def score_consistency(agent_mean: float, bounds: dict) -> float:
    """
    Normalise agent_mean_sentiment against corpus range.
    Higher agent compound (more positive tone) → higher score.
    """
    b = bounds["agent_mean_sentiment"]
    return normalise(agent_mean, b["min"], b["max"])


def score_efficiency(turns: int, bounds: dict) -> float:
    """
    Normalise turn count against corpus range, inverted.
    Fewer turns (faster resolution) → higher score.
    """
    b = bounds["turns"]
    return normalise(turns, b["min"], b["max"], invert=True)


def compute_final_score(
    resolution_score: float,
    sentiment_score: float,
    consistency_score: float,
    efficiency_score: float,
) -> float:
    return round(
        resolution_score  * WEIGHTS["resolution"]  +
        sentiment_score   * WEIGHTS["sentiment"]   +
        consistency_score * WEIGHTS["consistency"] +
        efficiency_score  * WEIGHTS["efficiency"],
        2,
    )


# ---------------------------------------------------------------------------
# Narrative summary — logic-driven, not LLM-generated
# ---------------------------------------------------------------------------

def generate_narrative(
    predicted_resolution: str,
    resolution_score: float,
    sentiment_score: float,
    consistency_score: float,
    efficiency_score: float,
    final_score: float,
) -> str:
    """
    Plain-language summary built from component score thresholds.
    Math scores, math narrates — no LLM in this path.

    Thresholds (70 / 45) divide the normalised [0,100] range into
    top third / middle / bottom third relative to the observed corpus.
    """
    parts = []

    # Resolution
    if predicted_resolution == "resolved":
        parts.append("Call ended in resolution.")
    else:
        parts.append("Call did not reach resolution.")

    # Customer sentiment trajectory
    if sentiment_score >= 70:
        parts.append("Customer sentiment improved notably over the course of the call.")
    elif sentiment_score >= 45:
        parts.append("Customer sentiment was broadly stable through the call.")
    else:
        parts.append("Customer sentiment declined or remained negative through the call.")

    # Agent consistency
    if consistency_score >= 70:
        parts.append("Agent maintained a consistently positive and professional tone.")
    elif consistency_score >= 35:
        parts.append("Agent tone was adequate but showed some variation.")
    else:
        parts.append("Agent tone was inconsistent or leaned negative at points.")

    # Call efficiency
    if efficiency_score >= 70:
        parts.append("Call was handled efficiently with a low turn count.")
    elif efficiency_score >= 45:
        parts.append("Call length was within a normal range for this call type.")
    else:
        parts.append("Call required a high number of turns to reach its outcome.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Score all transcripts
# ---------------------------------------------------------------------------

def score_all(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Apply all four scoring components to every transcript.
    Returns df with five new columns:
      resolution_score, sentiment_score, consistency_score,
      efficiency_score, final_score, narrative
    """
    resolution_scores  = []
    sentiment_scores   = []
    consistency_scores = []
    efficiency_scores  = []
    final_scores       = []
    narratives         = []

    for _, row in df.iterrows():
        r = score_resolution(row["predicted_resolution"])
        s = score_sentiment(row["customer_sentiment_delta"], bounds)
        c = score_consistency(row["agent_mean_sentiment"], bounds)
        e = score_efficiency(row["turns"], bounds)
        f = compute_final_score(r, s, c, e)
        n = generate_narrative(row["predicted_resolution"], r, s, c, e, f)

        resolution_scores.append(round(r, 2))
        sentiment_scores.append(round(s, 2))
        consistency_scores.append(round(c, 2))
        efficiency_scores.append(round(e, 2))
        final_scores.append(f)
        narratives.append(n)

    result = df.copy()
    result["resolution_score"]  = resolution_scores
    result["sentiment_score"]   = sentiment_scores
    result["consistency_score"] = consistency_scores
    result["efficiency_score"]  = efficiency_scores
    result["final_score"]       = final_scores
    result["narrative"]         = narratives

    return result


# ---------------------------------------------------------------------------
# Evaluation — golden set only
# ---------------------------------------------------------------------------

def evaluate(scored_df: pd.DataFrame) -> None:
    """
    Evaluate score distributions and directional sanity on the golden set.

    We do not have a ground-truth "correct score" to compare against —
    final_score is a composite, not a labelled target. What we can verify:
      1. Score distributions are spread across [0, 100] (not compressed)
      2. Resolved calls score higher than unresolved (directional sanity)
      3. Per-archetype means are plausible given what we know about the data
    """
    golden = scored_df[scored_df["set"] == "golden"].copy()

    if len(golden) == 0:
        print("  WARNING: no golden set transcripts found — skipping evaluation")
        return

    print(f"\n  --- Scoring Evaluation (golden set, n={len(golden)}) ---")

    print(f"\n  Final score distribution:")
    print(f"    mean   : {golden['final_score'].mean():.1f}")
    print(f"    median : {golden['final_score'].median():.1f}")
    print(f"    min    : {golden['final_score'].min():.1f}")
    print(f"    max    : {golden['final_score'].max():.1f}")
    print(f"    std    : {golden['final_score'].std():.1f}")

    print(f"\n  Mean scores by ground-truth resolution:")
    print(f"  {'Resolution':<14} {'Final':>6} {'Sentiment':>10} {'Consistency':>12} {'Efficiency':>11}")
    print(f"  {'-'*56}")
    for res, grp in golden.groupby("resolution"):
        print(
            f"  {res:<14} {grp['final_score'].mean():>6.1f}"
            f" {grp['sentiment_score'].mean():>10.1f}"
            f" {grp['consistency_score'].mean():>12.1f}"
            f" {grp['efficiency_score'].mean():>11.1f}"
        )

    # Directional sanity check
    resolved_mean   = golden[golden["resolution"] == "resolved"]["final_score"].mean()
    unresolved_mean = golden[golden["resolution"] == "unresolved"]["final_score"].mean()

    if resolved_mean > unresolved_mean:
        print(f"\n  OK  : resolved calls score higher ({resolved_mean:.1f}) "
              f"than unresolved ({unresolved_mean:.1f})")
    else:
        print(f"\n  WARN: resolved calls not scoring higher than unresolved — "
              f"check normalisation bounds or weight assumptions")

    print(f"\n  Mean final score by archetype:")
    for arch, grp in golden.groupby("archetype"):
        print(f"    {arch:<35}  {grp['final_score'].mean():.1f}")

    # Score spread check — are components actually using the range?
    print(f"\n  Component score spread (std — higher = more discriminative):")
    for col in ["resolution_score", "sentiment_score", "consistency_score", "efficiency_score"]:
        print(f"    {col:<22}  std={golden[col].std():.1f}  "
              f"min={golden[col].min():.1f}  max={golden[col].max():.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("scoring.py — agent performance scoring\n")

    if not TRANSCRIPTS_IN.exists():
        raise FileNotFoundError(
            f"{TRANSCRIPTS_IN} not found. Run resolution.py first."
        )

    df = pd.read_csv(TRANSCRIPTS_IN)
    print(f"  Loaded {len(df):,} transcripts")

    # Compute bounds from full corpus and save for app.py
    bounds = compute_bounds(df)
    BOUNDS_OUT.write_text(json.dumps(bounds, indent=2))
    print(f"  Bounds saved to {BOUNDS_OUT}")
    print(f"    customer_sentiment_delta : "
          f"[{bounds['customer_sentiment_delta']['min']:.4f}, "
          f"{bounds['customer_sentiment_delta']['max']:.4f}]")
    print(f"    agent_mean_sentiment     : "
          f"[{bounds['agent_mean_sentiment']['min']:.4f}, "
          f"{bounds['agent_mean_sentiment']['max']:.4f}]")
    print(f"    turns                    : "
          f"[{int(bounds['turns']['min'])}, {int(bounds['turns']['max'])}]")

    # Score all transcripts
    scored_df = score_all(df, bounds)

    # Evaluate on golden set
    evaluate(scored_df)

    # Save
    scored_df.to_csv(SCORED_OUT, index=False)
    print(f"\n  Saved: {SCORED_OUT}  ({len(scored_df):,} rows)")
    print("\n  Phase 5 complete. Next: app.py")

    return scored_df


if __name__ == "__main__":
    run()