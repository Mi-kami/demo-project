"""
check_alternation.py
====================
Standalone speaker-alternation checker for the golden set generator.

WHY THIS EXISTS:
Every transcript must strictly alternate AGENT / CUSTOMER turns — two
consecutive turns from the same speaker is a structural bug that will
corrupt speaker-level sentiment scoring in sentiment.py and break the
preprocessing.py parser's turn-indexing logic. These bugs are invisible
to eyeballing but trivial to catch programmatically.

The most common cause is inserting optional turn-pairs at an absolute
index that no longer aligns with the spine after a template was edited.
This script catches that immediately.

USAGE:
  # From demo-project root (same place you run generate_golden_set.py):
  python scripts/check_alternation.py

  # To check a specific sub-type only:
  python scripts/check_alternation.py billing_error_resolved

  # To check generated .txt files on disk instead of in-memory:
  python scripts/check_alternation.py --files

OUTPUT:
  Lists every alternation bug found with transcript ID, sub-type, and
  position. Prints a clean summary at the end. Exit code 0 if clean,
  1 if any bugs found — so this can be used in a pre-commit hook.
"""

import sys
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Mode 1: Check generator output in-memory (fast, catches bugs before write)
# ---------------------------------------------------------------------------

def check_in_memory(target_subtype=None):
    """
    Imports the generator and runs each sub-type's generate function
    against a small sample, checking alternation in-memory before any
    files are written. This is the recommended check to run after any
    template edit.
    """
    # Generator must be importable — run from demo-project root,
    # or ensure scripts/ is in the Python path.
    try:
        import generate_golden_set as g
    except ImportError:
        # Try scripts/ subdirectory
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            import generate_golden_set as g
        except ImportError:
            print("ERROR: cannot import generate_golden_set.")
            print("Run this script from your demo-project root directory,")
            print("or from within the scripts/ folder.")
            sys.exit(1)

    generators = g.IMPLEMENTED_GENERATORS
    if target_subtype:
        if target_subtype not in generators:
            print(f"ERROR: '{target_subtype}' not found in IMPLEMENTED_GENERATORS.")
            print(f"Available: {sorted(generators.keys())}")
            sys.exit(1)
        generators = {target_subtype: generators[target_subtype]}

    total_bugs = 0
    total_checked = 0

    for sub_type, fn in sorted(generators.items()):
        # Generate 12 transcripts — enough to hit all three markets
        # at least 4 times each and exercise all insertion probabilities.
        specs = fn(count=12, start_id=9000)
        sub_bugs = 0
        for spec in specs:
            speakers = [turn[0] for turn in spec.turns]
            for i in range(1, len(speakers)):
                if speakers[i] == speakers[i - 1]:
                    sub_bugs += 1
                    total_bugs += 1
                    print(
                        f"  ALTERNATION BUG  |  {spec.transcript_id}"
                        f"  |  sub-type: {sub_type}"
                        f"  |  market: {spec.market}"
                        f"  |  consecutive {speakers[i]!r} at position {i}"
                        f"  |  turns {i-1} and {i}: "
                        f"{spec.turns[i-1][1][:40]!r} → {spec.turns[i][1][:40]!r}"
                    )
        total_checked += len(specs)
        status = "CLEAN" if sub_bugs == 0 else f"{sub_bugs} BUG(S)"
        print(f"  {sub_type:<35} {status}")

    print()
    print(f"Checked {total_checked} transcripts across {len(generators)} sub-type(s).")
    if total_bugs == 0:
        print("All clean — no alternation bugs found.")
    else:
        print(f"FOUND {total_bugs} ALTERNATION BUG(S) — fix before running the full generator.")
    return total_bugs


# ---------------------------------------------------------------------------
# Mode 2: Check .txt files already written to disk
# ---------------------------------------------------------------------------

def check_on_disk(data_dir="data/raw"):
    """
    Reads every GS_*.txt file in data/raw/ and checks speaker alternation
    in the actual written output. Use this after a full generation run to
    confirm the files on disk are clean.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: directory '{data_dir}' not found.")
        print("Run generate_golden_set.py first, or check your working directory.")
        sys.exit(1)

    files = sorted(data_path.glob("GS_*.txt"))
    if not files:
        print(f"No GS_*.txt files found in '{data_dir}'.")
        sys.exit(1)

    total_bugs = 0
    sub_bugs = {}

    for f in files:
        text = f.read_text(encoding="utf-8")
        # Split on the --- separator to get dialogue only
        parts = text.split("---\n", 1)
        if len(parts) < 2:
            print(f"  WARNING: {f.name} has no '---' separator — skipping.")
            continue

        # Extract sub-type from header for grouping
        sub_match = re.search(r"SUB_TYPE: (\S+)", parts[0])
        sub_type = sub_match.group(1) if sub_match else "unknown"

        dialogue_lines = [
            line for line in parts[1].split("\n")
            if line.startswith("AGENT:") or line.startswith("CUSTOMER:")
        ]
        speakers = [line.split(":")[0] for line in dialogue_lines]

        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i - 1]:
                total_bugs += 1
                sub_bugs[sub_type] = sub_bugs.get(sub_type, 0) + 1
                print(
                    f"  ALTERNATION BUG  |  {f.name}"
                    f"  |  sub-type: {sub_type}"
                    f"  |  consecutive {speakers[i]!r} at position {i}"
                )

    print()
    print(f"Checked {len(files)} files in '{data_dir}'.")
    if total_bugs == 0:
        print("All clean — no alternation bugs found.")
    else:
        print(f"FOUND {total_bugs} ALTERNATION BUG(S) across {len(sub_bugs)} sub-type(s):")
        for sub, count in sorted(sub_bugs.items()):
            print(f"  {sub}: {count} bug(s)")
    return total_bugs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--files" in args:
        # Check written files on disk
        args = [a for a in args if a != "--files"]
        bugs = check_on_disk()
    else:
        # Check in-memory (default)
        target = args[0] if args else None
        bugs = check_in_memory(target_subtype=target)

    # Exit code 1 if bugs found — useful for CI/pre-commit hooks
    sys.exit(1 if bugs > 0 else 0)