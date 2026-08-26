"""
fix_seed_turns.py — One-off seed header fix
=============================================
Updates the TURNS: field in each of the 13 hand-authored seed
transcripts to match the actual parsed turn count.

WHY THIS EXISTS:
  The hand-authored seeds were written with manually set TURNS values
  that don't match the actual turn count in the file body — either
  because the count was estimated when the file was written, or because
  turns were added/removed during editing without updating the header.
  The preprocessor validates header vs body and rejects mismatches,
  so these need to be corrected before preprocessing can run cleanly.

CORRECT COUNTS (verified by direct turn counting from file body):
  GS_001: 17   GS_002: 15   GS_003: 23   GS_004: 17
  GS_005: 18   GS_006: 20   GS_007: 18   GS_008: 20 (after dedup fix)
  GS_009: 23   GS_010: 21   GS_011: 17   GS_012: 27
  GS_013: 19

USAGE:
  python fix_seed_turns.py
  Run once from repo root. Delete this file after running.
  GS_008 must be manually deduplicated first — open the file,
  delete everything from the second TRANSCRIPT_ID: GS_008 line
  to end of file, save, then run this script.
"""

import re
from pathlib import Path

GOLDEN_DIR = Path("data/raw/golden")

FIXES = {
    "GS_001": 17,
    "GS_002": 15,
    "GS_003": 23,
    "GS_004": 17,
    "GS_005": 18,
    "GS_006": 20,
    "GS_007": 18,
    "GS_008": 20,
    "GS_009": 23,
    "GS_010": 21,
    "GS_011": 17,
    "GS_012": 27,
    "GS_013": 19,
}

def fix_turns_header(path: Path, correct_turns: int) -> None:
    text = path.read_text(encoding="utf-8")
    updated = re.sub(
        r"^TURNS: \d+",
        f"TURNS: {correct_turns}",
        text,
        flags=re.MULTILINE,
    )
    path.write_text(updated, encoding="utf-8")

if not GOLDEN_DIR.exists():
    print(f"ERROR: {GOLDEN_DIR} not found. Run from repo root.")
else:
    for stem, correct_turns in FIXES.items():
        path = GOLDEN_DIR / f"{stem}.txt"
        if not path.exists():
            print(f"  MISSING: {path} — skipping")
            continue
        fix_turns_header(path, correct_turns)
        print(f"  {stem}: TURNS -> {correct_turns}")

    print("\nDone. Re-run preprocessing.py to confirm clean.")
    print("You can now delete fix_seed_turns.py.")