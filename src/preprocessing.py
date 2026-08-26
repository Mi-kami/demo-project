"""
preprocessing.py — Transcript Parser
=====================================
Reads all .txt transcript files from data/raw/golden/ and
data/raw/volume/, parses them into two structured DataFrames,
and saves both to data/processed/.

TWO OUTPUT FILES — WHY:
  turns.csv        — one row per speaker turn
                     consumed by: sentiment.py (scores each turn)
  transcripts.csv  — one row per transcript, metadata only
                     consumed by: intent.py, resolution.py, scoring.py

  Keeping them separate means downstream modules never have to
  collapse or expand the frame themselves — each gets exactly the
  shape it needs.

TRANSCRIPT FORMAT EXPECTED:
  TRANSCRIPT_ID: GS_001
  SET: golden          <- only present in volume set; golden files omit this
  ARCHETYPE: ...
  SUB_TYPE: ...
  MARKET: ...
  RESOLUTION: resolved / unresolved
  SENTIMENT_ARC: ...
  AGENT_CONSISTENCY: ...
  INTENT: ...
  TURNS: 17
  ---
  AGENT: text
  CUSTOMER: text
  ... (strictly alternating)

MULTI-LINE TURN HANDLING:
  Hand-authored seed transcripts wrap long dialogue lines across
  multiple lines. Generated transcripts use single lines per turn.
  The parser rejoins wrapped lines into single turns before counting,
  so both formats are handled correctly by the same logic.

USAGE:
  python src/preprocessing.py
  Run from repo root. Outputs land in data/processed/.
"""

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path("data/raw/golden")
VOLUME_DIR = Path("data/raw/volume")
PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Header field definitions
# These are the metadata fields we extract from every transcript header.
# ORDER MATTERS for the transcript-level DataFrame column ordering.
# ---------------------------------------------------------------------------

HEADER_FIELDS = [
    "TRANSCRIPT_ID",
    "SET",            # 'golden' or 'volume' — injected as 'golden' for GS_ files
    "ARCHETYPE",
    "SUB_TYPE",
    "MARKET",
    "RESOLUTION",
    "SENTIMENT_ARC",
    "AGENT_CONSISTENCY",
    "INTENT",
    "TURNS",
]


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_transcript(path: Path) -> tuple[dict, list[dict]]:
    """
    Parse a single transcript .txt file.

    Returns:
        metadata : dict         — one entry per header field
        turns    : list[dict]   — one entry per speaker turn,
                                  each with transcript_id, turn_index,
                                  speaker, and text

    Raises:
        ValueError if the file is malformed (missing --- separator,
        unknown speaker label, or zero turns parsed), or if the
        parsed turn count does not match the header declaration.
    """
    raw = path.read_text(encoding="utf-8")

    # Split on the --- separator that divides header from body
    if "---" not in raw:
        raise ValueError(f"{path.name}: missing '---' separator")

    header_block, body_block = raw.split("---", 1)

    # --- Parse header ---
    metadata = {}
    for line in header_block.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key in HEADER_FIELDS:
            metadata[key] = value

    # Inject SET field for golden set files (they predate the SET field)
    if "SET" not in metadata:
        metadata["SET"] = "golden"

    # TURNS is stored as string in the file — cast to int for validation
    if "TURNS" in metadata:
        metadata["TURNS"] = int(metadata["TURNS"])

    transcript_id = metadata.get("TRANSCRIPT_ID", path.stem)

    # --- Parse turns ---
    # Rejoin multi-line turns into single lines before parsing.
    # Hand-authored seeds wrap long lines; generated files do not.
    # Normalising both to single-line turns lets one regex handle both.
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

    # Extract speaker and text from each rejoined turn line
    turn_pattern = re.compile(r"^(AGENT|CUSTOMER):\s*(.+)$")
    turns = []
    turn_index = 0

    for line in rejoined:
        match = turn_pattern.match(line)
        if not match:
            continue
        speaker = match.group(1)
        text = match.group(2).strip()
        turns.append({
            "transcript_id": transcript_id,
            "turn_index":    turn_index,
            "speaker":       speaker,
            "text":          text,
        })
        turn_index += 1

    if not turns:
        raise ValueError(f"{path.name}: no turns parsed from body")

    # Validate turn count matches header declaration
    declared = metadata.get("TURNS", None)
    if declared is not None and len(turns) != declared:
        raise ValueError(
            f"{path.name}: header says TURNS={declared} "
            f"but {len(turns)} turns parsed"
        )

    return metadata, turns


# ---------------------------------------------------------------------------
# Loader — reads all files from both directories
# ---------------------------------------------------------------------------

def load_all_transcripts(
    golden_dir: Path = GOLDEN_DIR,
    volume_dir: Path = VOLUME_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and parse all transcript files from both directories.

    Returns:
        transcripts_df : DataFrame, one row per transcript
        turns_df       : DataFrame, one row per speaker turn
    """
    all_metadata = []
    all_turns = []
    errors = []

    sources = []
    if golden_dir.exists():
        sources.extend(sorted(golden_dir.glob("GS_*.txt")))
    else:
        print(f"  WARNING: golden directory not found: {golden_dir}")

    if volume_dir.exists():
        sources.extend(sorted(volume_dir.glob("VS_*.txt")))
    else:
        print(f"  WARNING: volume directory not found: {volume_dir}")

    print(f"  Found {len(sources)} transcript files")

    for path in sources:
        try:
            metadata, turns = parse_transcript(path)
            all_metadata.append(metadata)

            # Attach all metadata fields to every turn row so downstream
            # modules can filter or group without needing a join
            for turn in turns:
                turn.update({
                    "set":               metadata.get("SET",               "golden"),
                    "archetype":         metadata.get("ARCHETYPE",         ""),
                    "sub_type":          metadata.get("SUB_TYPE",          ""),
                    "market":            metadata.get("MARKET",            ""),
                    "resolution":        metadata.get("RESOLUTION",        ""),
                    "sentiment_arc":     metadata.get("SENTIMENT_ARC",     ""),
                    "agent_consistency": metadata.get("AGENT_CONSISTENCY", ""),
                    "intent":            metadata.get("INTENT",            ""),
                })
                all_turns.append(turn)

        except ValueError as e:
            errors.append(str(e))
            print(f"  PARSE ERROR: {e}")

    if errors:
        print(f"\n  {len(errors)} files failed to parse — check errors above")

    # --- Build transcript-level DataFrame ---
    transcripts_df = pd.DataFrame(all_metadata)

    # Normalise column names to lowercase for consistency with turns_df
    transcripts_df.columns = [c.lower() for c in transcripts_df.columns]

    # Enforce column order — transcript_id first, turns last
    col_order = [
        "transcript_id", "set", "archetype", "sub_type", "market",
        "resolution", "sentiment_arc", "agent_consistency", "intent", "turns",
    ]
    transcripts_df = transcripts_df.reindex(columns=col_order)

    # --- Build turns-level DataFrame ---
    turns_df = pd.DataFrame(all_turns)

    # Column order: identifiers first, metadata last
    turns_col_order = [
        "transcript_id", "turn_index", "speaker", "text",
        "set", "archetype", "sub_type", "market",
        "resolution", "sentiment_arc", "agent_consistency", "intent",
    ]
    turns_df = turns_df.reindex(columns=turns_col_order)

    return transcripts_df, turns_df


# ---------------------------------------------------------------------------
# Validation — runs before saving, catches systematic parse bugs early
# ---------------------------------------------------------------------------

def validate(transcripts_df: pd.DataFrame, turns_df: pd.DataFrame) -> bool:
    """
    Run basic sanity checks on the parsed DataFrames.
    Prints a report and returns True if all checks pass.

    WHY VALIDATE BEFORE SAVING: a systematic parse bug caught here
    costs seconds. The same bug caught silently in scoring.py costs
    an hour of debugging.
    """
    print("\n  --- Validation ---")
    passed = True

    # 1. No missing transcript_ids
    missing_ids = transcripts_df["transcript_id"].isna().sum()
    if missing_ids:
        print(f"  FAIL: {missing_ids} transcripts with missing transcript_id")
        passed = False
    else:
        print(f"  OK  : all transcript_ids present")

    # 2. Resolution field contains only expected values
    valid_resolutions = {"resolved", "unresolved"}
    bad_res = transcripts_df[~transcripts_df["resolution"].isin(valid_resolutions)]
    if len(bad_res):
        print(f"  FAIL: {len(bad_res)} unexpected resolution values: "
              f"{bad_res['resolution'].unique().tolist()}")
        passed = False
    else:
        print(f"  OK  : resolution values all valid")

    # 3. Speaker column contains only AGENT or CUSTOMER
    valid_speakers = {"AGENT", "CUSTOMER"}
    bad_speakers = turns_df[~turns_df["speaker"].isin(valid_speakers)]
    if len(bad_speakers):
        print(f"  FAIL: {len(bad_speakers)} turns with unexpected speaker label")
        passed = False
    else:
        print(f"  OK  : all speaker labels valid")

    # 4. No empty turn text
    empty_text = turns_df["text"].isna() | (turns_df["text"].str.strip() == "")
    if empty_text.sum():
        print(f"  FAIL: {empty_text.sum()} turns with empty text")
        passed = False
    else:
        print(f"  OK  : no empty turn text")

    # 5. Every transcript has at least one AGENT and one CUSTOMER turn
    speaker_counts = (
        turns_df
        .groupby(["transcript_id", "speaker"])
        .size()
        .unstack(fill_value=0)
    )
    if "AGENT" not in speaker_counts.columns or "CUSTOMER" not in speaker_counts.columns:
        print(f"  FAIL: speaker columns missing from groupby — check parse")
        passed = False
    else:
        missing_agent    = (speaker_counts["AGENT"] == 0).sum()
        missing_customer = (speaker_counts["CUSTOMER"] == 0).sum()
        if missing_agent or missing_customer:
            print(f"  FAIL: {missing_agent} transcripts missing AGENT turns, "
                  f"{missing_customer} missing CUSTOMER turns")
            passed = False
        else:
            print(f"  OK  : all transcripts have both AGENT and CUSTOMER turns")

    # 6. SET field contains only expected values
    valid_sets = {"golden", "volume"}
    bad_sets = transcripts_df[~transcripts_df["set"].isin(valid_sets)]
    if len(bad_sets):
        print(f"  FAIL: {len(bad_sets)} unexpected SET values")
        passed = False
    else:
        print(f"  OK  : SET values all valid")

    return passed


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(transcripts_df: pd.DataFrame, turns_df: pd.DataFrame):
    print("\n  --- Summary ---")
    print(f"  Transcripts          : {len(transcripts_df):,}")
    print(f"  Turns                : {len(turns_df):,}")
    print(f"  Avg turns/transcript : {len(turns_df) / len(transcripts_df):.1f}")

    print(f"\n  By set:")
    for s, grp in transcripts_df.groupby("set"):
        print(f"    {s:<10} {len(grp):>5} transcripts")

    print(f"\n  By resolution:")
    for r, grp in transcripts_df.groupby("resolution"):
        pct = len(grp) / len(transcripts_df) * 100
        print(f"    {r:<12} {len(grp):>5} transcripts  ({pct:.1f}%)")

    print(f"\n  By sub_type (transcript count):")
    for st, grp in transcripts_df.groupby("sub_type"):
        print(f"    {st:<35} {len(grp):>5}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("preprocessing.py — loading transcripts\n")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transcripts_df, turns_df = load_all_transcripts()

    ok = validate(transcripts_df, turns_df)

    print_summary(transcripts_df, turns_df)

    if not ok:
        print("\n  WARNING: validation failures detected. "
              "Inspect errors above before proceeding to sentiment.py.")
    else:
        print("\n  Validation passed.")

    # Save outputs
    transcripts_path = PROCESSED_DIR / "transcripts.csv"
    turns_path       = PROCESSED_DIR / "turns.csv"

    transcripts_df.to_csv(transcripts_path, index=False)
    turns_df.to_csv(turns_path, index=False)

    print(f"\n  Saved: {transcripts_path}  ({len(transcripts_df):,} rows)")
    print(f"\n  Saved: {turns_path}  ({len(turns_df):,} rows)")
    print("\n  Phase 1 complete. Next: sentiment.py")

    return transcripts_df, turns_df


if __name__ == "__main__":
    run()