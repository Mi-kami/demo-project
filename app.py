"""
app.py — Streamlit Interface
=============================
Runs the full call analytics pipeline on a pasted or uploaded transcript
in real time. Built as a lightweight version of Aethex's call analytics
layer — sentiment tracking, intent classification, resolution detection,
and agent performance scoring.

Three sections:
  1. Input       — paste or upload a transcript
  2. Analysis    — turn-by-turn sentiment, detected intent, resolution signals
  3. Score Card  — final score (0–100) with visible component breakdown

Scoring weights are visible by design — not a black box.

USAGE:
  streamlit run app.py
  Run from repo root. Run scoring.py first to generate score_bounds.json.
"""

import json
import re
import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Add src/ to path so we can import pipeline modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentiment import load_vader, score_turns, aggregate_sentiment
from intent import load_model, embed_prototypes
from resolution import classify_resolution
from scoring import (
    score_resolution,
    score_sentiment,
    score_consistency,
    score_efficiency,
    compute_final_score,
    generate_narrative,
)

BOUNDS_PATH = Path("data/processed/score_bounds.json")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Call Analytics — Aethex Demo",
    page_icon="📞",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resource loaders — models load once per session, not per run
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading VADER...")
def get_vader():
    return load_vader()


@st.cache_resource(show_spinner="Loading intent model (all-MiniLM-L6-v2)...")
def get_intent_resources():
    """Load model and pre-compute prototype embeddings once."""
    model = load_model()
    labels, proto_embs = embed_prototypes(model)
    return model, labels, proto_embs


@st.cache_data
def get_bounds() -> dict:
    """
    Load normalisation bounds from score_bounds.json (produced by scoring.py).
    Falls back to conservative defaults if the file doesn't exist — allows the
    app to run on a fresh transcript even if the full corpus hasn't been scored.
    """
    if BOUNDS_PATH.exists():
        return json.loads(BOUNDS_PATH.read_text())
    # Fallback: theoretical VADER range — narrower spread but still functional
    return {
        "customer_sentiment_delta": {"min": -0.8,  "max": 0.8},
        "agent_mean_sentiment":     {"min": -0.5,  "max": 0.8},
        "turns":                    {"min": 15.0,  "max": 27.0},
    }


# ---------------------------------------------------------------------------
# Transcript parser — works on raw text string, no file I/O
# ---------------------------------------------------------------------------

def parse_transcript_text(text: str) -> tuple[dict, pd.DataFrame]:
    """
    Parse a transcript from a raw string into metadata + turns DataFrame.

    Handles both formats:
      - With header block (TRANSCRIPT_ID: ..., ---, AGENT: ...)
      - Body-only (AGENT: ... / CUSTOMER: ... with no header)

    Rejoins multi-line wrapped turns before parsing — same logic as
    preprocessing.py so single-transcript parsing is consistent with
    the batch pipeline.
    """
    metadata = {}
    HEADER_FIELDS = {
        "TRANSCRIPT_ID", "SET", "ARCHETYPE", "SUB_TYPE", "MARKET",
        "RESOLUTION", "SENTIMENT_ARC", "AGENT_CONSISTENCY", "INTENT", "TURNS",
    }

    if "---" in text:
        header_block, body_block = text.split("---", 1)
        for line in header_block.strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            if key.strip() in HEADER_FIELDS:
                metadata[key.strip()] = value.strip()
    else:
        body_block = text

    transcript_id = metadata.get("TRANSCRIPT_ID", "LIVE_INPUT")

    # Rejoin multi-line turns
    lines = body_block.splitlines()
    rejoined = []
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("AGENT:") or stripped.startswith("CUSTOMER:"):
            if current is not None:
                rejoined.append(current)
            current = stripped
        elif stripped and current is not None:
            current += " " + stripped
    if current is not None:
        rejoined.append(current)

    turn_pattern = re.compile(r"^(AGENT|CUSTOMER):\s*(.+)$")
    turns = []
    for i, line in enumerate(rejoined):
        m = turn_pattern.match(line)
        if not m:
            continue
        turns.append({
            "transcript_id": transcript_id,
            "turn_index":    i,
            "speaker":       m.group(1),
            "text":          m.group(2).strip(),
        })

    if not turns:
        return metadata, pd.DataFrame(columns=["transcript_id", "turn_index", "speaker", "text"])

    return metadata, pd.DataFrame(turns)


# ---------------------------------------------------------------------------
# Full pipeline — one transcript, in memory
# ---------------------------------------------------------------------------

def run_pipeline(text: str) -> tuple[dict | None, str | None]:
    """
    Run preprocessing → sentiment → intent → resolution → scoring on a single
    transcript string. Returns (results dict, error string) — one will be None.
    """
    # 1. Parse
    metadata, turns_df = parse_transcript_text(text)
    if turns_df.empty:
        return None, (
            "No speaker turns found. Check that turns are formatted as:\n"
            "AGENT: text\nCUSTOMER: text"
        )

    n_agents    = (turns_df["speaker"] == "AGENT").sum()
    n_customers = (turns_df["speaker"] == "CUSTOMER").sum()
    if n_agents == 0 or n_customers == 0:
        return None, "Both AGENT and CUSTOMER turns are required."

    # 2. Sentiment — VADER on every turn, then aggregate
    analyser = get_vader()
    turns_scored = score_turns(turns_df, analyser)
    agg_df   = aggregate_sentiment(turns_scored)
    agg      = agg_df.iloc[0].to_dict()

    # 3. Intent — embedding similarity
    model, labels, proto_embs = get_intent_resources()

    # Use archetype-aware window: outbound LOAN_RECOVERY calls have a
    # different opening structure (agent identifies institution first).
    archetype    = metadata.get("ARCHETYPE", "").upper()
    is_outbound  = "LOAN_RECOVERY" in archetype
    customer_turns = turns_scored[turns_scored["speaker"] == "CUSTOMER"]["text"].tolist()

    if is_outbound and len(customer_turns) >= 5:
        # Turns 1-5 (iloc[1:5]) — skip the contentless first "who is this"
        window = customer_turns[1:5]
    else:
        window = customer_turns[:3]

    opening_text  = " ".join(window) if window else " ".join(customer_turns[:3])
    transcript_emb = model.encode([opening_text], normalize_embeddings=True)
    similarities   = (transcript_emb @ proto_embs.T)[0]
    best_idx       = int(similarities.argmax())
    predicted_intent   = labels[best_idx]
    intent_confidence  = float(similarities.max())

    # 4. Resolution — closing pattern heuristics on last 5 turns
    closing_lines = [
        f"{row['speaker']}: {row['text']}"
        for _, row in turns_scored.tail(5).iterrows()
    ]
    closing_text = "\n".join(closing_lines)
    predicted_resolution, resolution_confidence, signals = classify_resolution(closing_text)

    # 5. Score
    bounds  = get_bounds()
    n_turns = len(turns_df)

    r_score = score_resolution(predicted_resolution)
    s_score = score_sentiment(agg["customer_sentiment_delta"], bounds)
    c_score = score_consistency(agg["agent_mean_sentiment"], bounds)
    e_score = score_efficiency(n_turns, bounds)
    final   = compute_final_score(r_score, s_score, c_score, e_score)
    narrative = generate_narrative(
        predicted_resolution, r_score, s_score, c_score, e_score, final
    )

    return {
        "metadata":             metadata,
        "turns_scored":         turns_scored,
        "agg":                  agg,
        "predicted_intent":     predicted_intent,
        "intent_confidence":    intent_confidence,
        "predicted_resolution": predicted_resolution,
        "resolution_confidence": resolution_confidence,
        "resolution_signals":   signals,
        "resolution_score":     round(r_score, 1),
        "sentiment_score":      round(s_score, 1),
        "consistency_score":    round(c_score, 1),
        "efficiency_score":     round(e_score, 1),
        "final_score":          final,
        "narrative":            narrative,
        "turns_count":          n_turns,
    }, None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "🟢 Positive"
    if compound <= -0.05:
        return "🔴 Negative"
    return "⚪ Neutral"


def score_colour(score: float) -> str:
    if score >= 70:
        return "🟢"
    if score >= 45:
        return "🟡"
    return "🔴"


def render_score_bar(score: float, label: str, weight: str):
    """Render a single component as a labelled progress bar."""
    col1, col2, col3 = st.columns([3, 1, 1])
    col1.progress(int(score) / 100, text=label)
    col2.markdown(f"**{score:.1f}** / 100")
    col3.markdown(f"weight: **{weight}**")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.title("📞 Call Analytics Demo")
    st.caption(
        "Lightweight version of a call quality scoring pipeline — "
        "built to understand Aethex's analytics layer."
    )

    # ------------------------------------------------------------------ #
    # Section 1: Input
    # ------------------------------------------------------------------ #
    st.header("1. Input")

    input_method = st.radio(
        "Input method", ["Paste transcript", "Upload .txt file"], horizontal=True
    )

    transcript_text = ""

    if input_method == "Paste transcript":
        transcript_text = st.text_area(
            "Paste transcript here",
            height=220,
            placeholder=(
                "AGENT: Good afternoon, thank you for calling...\n"
                "CUSTOMER: Hi, I need help with...\n"
                "AGENT: Of course, let me look into that...\n"
                "\n"
                "Header block (TRANSCRIPT_ID: etc.) is optional — "
                "the pipeline handles body-only transcripts too."
            ),
        )
    else:
        uploaded = st.file_uploader("Upload a .txt transcript file", type=["txt"])
        if uploaded:
            transcript_text = uploaded.read().decode("utf-8")
            st.success(f"Loaded: {uploaded.name}")
            with st.expander("Preview"):
                st.text(transcript_text[:800] + ("..." if len(transcript_text) > 800 else ""))

    analyse_btn = st.button(
        "▶  Analyse transcript",
        type="primary",
        disabled=not transcript_text.strip(),
    )

    if not analyse_btn or not transcript_text.strip():
        st.info("Paste or upload a transcript above, then click Analyse.")
        return

    # Run pipeline
    with st.spinner("Running pipeline — sentiment → intent → resolution → scoring..."):
        results, error = run_pipeline(transcript_text)

    if error:
        st.error(error)
        return

    # ------------------------------------------------------------------ #
    # Section 2: Analysis
    # ------------------------------------------------------------------ #
    st.divider()
    st.header("2. Analysis")

    # Top-line metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Detected Intent",
        results["predicted_intent"].replace("_", " ").title(),
    )
    col1.caption(f"confidence: {results['intent_confidence']:.2f}")

    col2.metric(
        "Resolution",
        results["predicted_resolution"].upper(),
    )
    col2.caption(f"confidence: {results['resolution_confidence']:.2f}")

    col3.metric("Turn Count", results["turns_count"])

    delta_val = results["agg"]["customer_sentiment_delta"]
    col4.metric(
        "Customer Sentiment Δ",
        f"{delta_val:+.3f}",
        delta=("improved" if delta_val > 0.01 else ("declined" if delta_val < -0.01 else "flat")),
        delta_color="normal",
    )

    # Turn-by-turn table
    st.subheader("Turn-by-turn sentiment")
    turns_display = results["turns_scored"][
        ["turn_index", "speaker", "text", "vader_compound"]
    ].copy()
    turns_display["Sentiment"] = turns_display["vader_compound"].apply(sentiment_label)
    turns_display = turns_display.rename(columns={
        "turn_index":    "#",
        "speaker":       "Speaker",
        "text":          "Turn",
        "vader_compound": "VADER score",
    })
    st.dataframe(
        turns_display[["#", "Speaker", "Turn", "VADER score", "Sentiment"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "VADER score": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    # Sentiment detail
    with st.expander("Sentiment detail — start / end / agent"):
        agg = results["agg"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Customer start (avg first 2 turns)", f"{agg['customer_start_sentiment']:.3f}")
        c2.metric("Customer end (avg last 2 turns)",    f"{agg['customer_end_sentiment']:.3f}")
        c3.metric("Customer delta (end − start)",        f"{agg['customer_sentiment_delta']:+.3f}")
        c1.metric("Agent mean sentiment",                f"{agg['agent_mean_sentiment']:.3f}")
        c2.metric("Agent min sentiment",                 f"{agg['agent_min_sentiment']:.3f}")
        c3.metric("Customer mean sentiment",             f"{agg['customer_mean_sentiment']:.3f}")

    # Resolution signals
    with st.expander("Resolution signals detected"):
        signals = results["resolution_signals"]
        if signals:
            for sig in signals:
                tag = "✅" if sig.startswith("RESOLVED") else "❌"
                st.text(f"{tag}  {sig}")
        else:
            st.text("No signals detected — defaulted to unresolved.")

    # ------------------------------------------------------------------ #
    # Section 3: Score Card
    # ------------------------------------------------------------------ #
    st.divider()
    st.header("3. Score Card")

    final = results["final_score"]
    colour = score_colour(final)
    st.markdown(f"## {colour} &nbsp; **{final:.1f} / 100**")
    st.progress(int(final))

    st.subheader("Component breakdown")
    st.caption(
        "Weights are visible by design. This score is not a black box — "
        "every number here is traceable back to a specific signal in the transcript."
    )

    # Component table
    comp_df = pd.DataFrame({
        "Component": [
            "Resolution achieved",
            "Customer sentiment trajectory",
            "Agent sentiment consistency",
            "Call efficiency",
        ],
        "Weight": ["40%", "30%", "20%", "10%"],
        "Score (0–100)": [
            results["resolution_score"],
            results["sentiment_score"],
            results["consistency_score"],
            results["efficiency_score"],
        ],
        "Weighted contribution": [
            round(results["resolution_score"]  * 0.40, 1),
            round(results["sentiment_score"]   * 0.30, 1),
            round(results["consistency_score"] * 0.20, 1),
            round(results["efficiency_score"]  * 0.10, 1),
        ],
        "What it measures": [
            "Did the call end in resolution?",
            "Did customer sentiment improve from start to end?",
            "Did the agent maintain a positive, consistent tone?",
            "Was the call resolved in fewer turns (faster)?",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Visual bars — component scores
    st.markdown("&nbsp;")
    render_score_bar(results["resolution_score"],  "Resolution achieved",              "40%")
    render_score_bar(results["sentiment_score"],   "Customer sentiment trajectory",    "30%")
    render_score_bar(results["consistency_score"], "Agent sentiment consistency",      "20%")
    render_score_bar(results["efficiency_score"],  "Call efficiency",                  "10%")

    # Narrative summary
    st.subheader("Summary")
    st.info(results["narrative"])

    # Metadata (if header was present in the transcript)
    meta = results["metadata"]
    display_fields = {k: v for k, v in meta.items() if v and k != "TURNS"}
    if display_fields:
        with st.expander("Transcript metadata"):
            for k, v in display_fields.items():
                st.text(f"{k}: {v}")


if __name__ == "__main__":
    main()