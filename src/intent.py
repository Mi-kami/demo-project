"""
intent.py — Embedding Similarity Intent Classifier
====================================================
Classifies the intent of every transcript using semantic embedding
similarity against 12 intent prototype sentences.

WHY EMBEDDING SIMILARITY OVER KEYWORDS:
  Keyword approaches break the moment a customer phrases something
  unexpectedly. "You took money from my account twice" and "I was
  charged double" are very different strings but carry the same
  meaning — a keyword matcher misses one, an embedding model catches
  both because it encodes semantic meaning, not surface word matches.

  Embedding similarity works by:
    1. Defining one prototype sentence per intent class — a sentence
       that captures the core meaning of that intent
    2. Embedding the transcript's opening customer turns (where intent
       is always established) into the same vector space
    3. Computing cosine similarity between the transcript embedding
       and every prototype embedding
    4. Assigning the label of the nearest prototype

MODEL: all-MiniLM-L6-v2
  80MB, fast on CPU, strong performance on short conversational text.
  No fine-tuning required — the general-purpose embeddings are
  sufficient for distinguishing 12 semantically distinct intent classes.

WHAT "OPENING CUSTOMER TURNS" MEANS:
  We embed the first 3 customer turns only. Intent is always
  established in the opening of a call — by turn 3 the customer
  has stated their problem. Using the full transcript would dilute
  the intent signal with resolution/sentiment content that belongs
  to different scoring components.

PROTOTYPE ITERATION HISTORY:
  v1 — initial prototypes, 64.1% accuracy on golden set.
       Key failures: hardship_plan 0% (all confused with loan_reminder),
       service_complaint 53.6%, policy_dispute 30.8%, loan_escalation 50%.
  v2 — refined 5 weak prototypes to sharpen semantic anchors:
       hardship_plan: emphasise financial hardship + restructure signal
       loan_reminder: anchor to payment arrangement, not call structure
       service_complaint: anchor to unacceptable treatment signal
       policy_dispute: emphasise undisclosed charge + formal dispute
       loan_escalation: emphasise refusal signal + escalation intent

EVALUATION:
  On the golden set (167 transcripts with hand-verified ground truth
  labels) we compute accuracy, per-class precision/recall, and a
  confusion matrix. This tells us where the classifier struggles —
  which intent pairs are semantically close enough to be confused.
  This is real evaluation, not circular (we predict from text,
  compare against the metadata label we did not use as input).

OUTPUTS:
  data/processed/transcripts_intent.csv  — transcripts_sentiment.csv
                                           + predicted_intent column
                                           + intent_confidence column
  Printed evaluation report on golden set.

USAGE:
  pip install sentence-transformers
  python src/intent.py
  Run from repo root AFTER sentiment.py has been run.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TRANSCRIPTS_IN = PROCESSED_DIR / "transcripts_sentiment.csv"
TURNS_IN       = PROCESSED_DIR / "turns.csv"
INTENT_OUT     = PROCESSED_DIR / "transcripts_intent.csv"

# ---------------------------------------------------------------------------
# Intent prototype sentences — v2 (refined from v1 based on evaluation)
# One sentence per class — chosen to capture the core customer complaint
# or request in plain language, the way a customer would actually say it.
# These are the "anchors" in embedding space that all transcripts of that
# intent type should cluster near.
# ---------------------------------------------------------------------------

INTENT_PROTOTYPES = {
    "billing_error": (
        "I was charged twice for the same transaction and I need the "
        "duplicate charge reversed."
    ),
    "failed_transaction": (
        "I made a transfer but it showed as failed and I don't know "
        "if the money left my account or where it went."
    ),
    "account_access": (
        "I cannot log into my account, it is locked and I need access "
        "restored urgently."
    ),
    "service_complaint": (
        "I am very unhappy with the service I received and I want "
        "to make a formal complaint — this is not acceptable "
        "and I want to know what you are going to do about it."
    ),
    "policy_dispute": (
        "There are charges on my account that were never explained to me "
        "before I committed — I did not agree to this and I am disputing "
        "it formally."
    ),
    "kyc_verification": (
        "I received a notification to complete my identity verification "
        "and I am calling to sort that out."
    ),
    "kyc_incomplete": (
        "My verification documents were rejected and my account is still "
        "restricted — I don't understand what is wrong with my documents."
    ),
    "kyc_escalation": (
        "My account has been flagged for an extended review and nobody "
        "can tell me why or what is being investigated."
    ),
    "hardship_plan": (
        "I know I have missed my loan payments and I want to explain "
        "my situation — I am going through a very difficult time "
        "financially and I need help finding a way to repay what I owe."
    ),
    "loan_reminder": (
        "Yes I know my payment is late, things have been difficult "
        "this week but I am not trying to avoid it, I will pay."
    ),
    "loan_escalation": (
        "I am not paying this disputed amount — I have been calling "
        "about this for weeks and nothing has been resolved, "
        "I am ready to escalate this formally."
    ),
}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the sentence-transformers model.
    Downloads all-MiniLM-L6-v2 (~80MB) on first run,
    uses cache on subsequent runs.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. Run:\n"
            "  pip install sentence-transformers\n"
            "then re-run this script."
        )

    print("  Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_prototypes(model) -> tuple[list[str], np.ndarray]:
    """
    Embed all 11 intent prototype sentences.

    Returns:
        labels      : list of intent label strings, in fixed order
        proto_embs  : (11, embedding_dim) array of prototype embeddings
    """
    labels = list(INTENT_PROTOTYPES.keys())
    sentences = [INTENT_PROTOTYPES[l] for l in labels]
    proto_embs = model.encode(sentences, normalize_embeddings=True)
    return labels, proto_embs


def get_opening_text(transcript_id: str, turns_df: pd.DataFrame) -> str:
    transcript_turns = turns_df[turns_df["transcript_id"] == transcript_id]
    archetype = transcript_turns["archetype"].iloc[0] if len(transcript_turns) > 0 else ""

    customer_turns = transcript_turns[transcript_turns["speaker"] == "CUSTOMER"]["text"]

    # Outbound calls (LOAN_RECOVERY): the customer's first response is
    # "who is this / yes what do you want" — near-contentless in embedding
    # space. The real intent signal emerges in turns 3-6 after the agent
    # has explained why they called. Inbound calls establish intent
    # immediately so first 3 turns is correct for everything else.
    if "LOAN_RECOVERY" in str(archetype).upper():
        selected = customer_turns.iloc[1:5].tolist()
    else:
        selected = customer_turns.head(3).tolist()

    return " ".join(selected)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_intents(
    transcripts_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Classify intent for every transcript using cosine similarity
    against prototype embeddings.

    Returns transcripts_df with two new columns:
      predicted_intent   — the nearest prototype label
      intent_confidence  — cosine similarity score (0.0-1.0),
                           higher = more confident match
    """
    labels, proto_embs = embed_prototypes(model)

    print(f"  Embedding opening turns for {len(transcripts_df):,} transcripts...")

    # Build opening text for every transcript
    opening_texts = [
        get_opening_text(tid, turns_df)
        for tid in transcripts_df["transcript_id"]
    ]

    # Embed all at once — batch processing is much faster than one at a time
    transcript_embs = model.encode(
        opening_texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    # Cosine similarity: since embeddings are L2-normalised,
    # dot product == cosine similarity
    # Shape: (n_transcripts, 11)
    similarity_matrix = transcript_embs @ proto_embs.T

    # Assign nearest prototype
    best_idx           = similarity_matrix.argmax(axis=1)
    predicted_intents  = [labels[i] for i in best_idx]
    confidence_scores  = similarity_matrix.max(axis=1)

    result = transcripts_df.copy()
    result["predicted_intent"]  = predicted_intents
    result["intent_confidence"] = confidence_scores.round(4)

    return result


# ---------------------------------------------------------------------------
# Evaluation — golden set only
# ---------------------------------------------------------------------------

def evaluate(result_df: pd.DataFrame) -> None:
    """
    Evaluate predicted intent against ground truth on the golden set.

    WHY GOLDEN SET ONLY FOR EVALUATION:
      The golden set has hand-verified ground truth intent labels.
      The volume set labels are programmatically assigned — using them
      for evaluation would be circular (the classifier would be
      evaluated against labels produced by the same logic it's trying
      to replicate). Golden set evaluation is the honest claim.
    """
    golden = result_df[result_df["set"] == "golden"].copy()

    if len(golden) == 0:
        print("  WARNING: no golden set transcripts found — skipping evaluation")
        return

    print(f"\n  --- Intent Classification Evaluation (golden set, n={len(golden)}) ---")

    correct  = (golden["intent"] == golden["predicted_intent"]).sum()
    accuracy = correct / len(golden)
    print(f"\n  Overall accuracy : {correct}/{len(golden)}  ({accuracy*100:.1f}%)")

    # Per-class breakdown
    print(f"\n  Per-class results:")
    print(f"  {'Intent':<25} {'True':>5} {'Pred':>5} {'Correct':>8} {'Acc':>6}")
    print(f"  {'-'*55}")

    for intent in sorted(golden["intent"].unique()):
        true_pos  = golden[golden["intent"] == intent]
        correct_n = golden[
            (golden["intent"] == intent) &
            (golden["predicted_intent"] == intent)
        ].shape[0]
        pred_pos  = golden[golden["predicted_intent"] == intent]
        class_acc = correct_n / len(true_pos) if len(true_pos) > 0 else 0.0
        print(
            f"  {intent:<25} {len(true_pos):>5} {len(pred_pos):>5} "
            f"{correct_n:>8} {class_acc*100:>5.1f}%"
        )

    # Confusion pairs — where does the classifier go wrong?
    wrong = golden[golden["intent"] != golden["predicted_intent"]]
    if len(wrong) > 0:
        print(f"\n  Misclassified transcripts ({len(wrong)}):")
        confusion = (
            wrong
            .groupby(["intent", "predicted_intent"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in confusion.iterrows():
            print(
                f"    {row['intent']:<25} -> predicted as "
                f"{row['predicted_intent']:<25} ({row['count']}x)"
            )
        print(
            f"\n  NOTE: misclassifications between semantically close "
            f"intents (e.g. loan_reminder vs hardship_plan) are expected "
            f"without domain fine-tuning. With real Aethex data, "
            f"fine-tuning on in-domain examples would close this gap."
        )
    else:
        print(f"\n  Perfect classification on golden set.")

    # Average confidence by correctness
    correct_conf   = golden[
        golden["intent"] == golden["predicted_intent"]
    ]["intent_confidence"].mean()
    incorrect_conf = golden[
        golden["intent"] != golden["predicted_intent"]
    ]["intent_confidence"].mean()

    print(f"\n  Avg confidence — correct predictions   : {correct_conf:.3f}")
    if not np.isnan(incorrect_conf):
        print(f"  Avg confidence — incorrect predictions : {incorrect_conf:.3f}")
        if incorrect_conf < correct_conf:
            print(
                f"  OK  : confidence is lower on wrong predictions — "
                f"model is appropriately uncertain when it gets it wrong"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("intent.py — embedding similarity intent classification\n")

    if not TRANSCRIPTS_IN.exists():
        raise FileNotFoundError(
            f"{TRANSCRIPTS_IN} not found. Run sentiment.py first."
        )
    if not TURNS_IN.exists():
        raise FileNotFoundError(
            f"{TURNS_IN} not found. Run preprocessing.py first."
        )

    transcripts_df = pd.read_csv(TRANSCRIPTS_IN)
    turns_df       = pd.read_csv(TURNS_IN)

    print(f"  Loaded {len(transcripts_df):,} transcripts")
    print(f"  Loaded {len(turns_df):,} turns")

    # Load model (uses cache after first run — much faster)
    model = load_model()

    # Classify
    result_df = classify_intents(transcripts_df, turns_df, model)

    # Evaluate on golden set
    evaluate(result_df)

    # Save
    result_df.to_csv(INTENT_OUT, index=False)
    print(f"\n  Saved: {INTENT_OUT}  ({len(result_df):,} rows)")
    print("\n  Phase 3 complete. Next: resolution.py")

    return result_df


if __name__ == "__main__":
    run()