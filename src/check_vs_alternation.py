"""
Volume Set Alternation Checker
==============================
Verifies that no VS_ transcript contains two consecutive turns from
the same speaker. Run this after generate_volume_set.py.

Same principle as check_alternation.py for the golden set, but
operates on the written .txt files in data/raw/volume/ rather than
calling generator functions directly — this catches any render_volume()
bugs that wouldn't show up from inspecting the TranscriptSpec alone.

Usage (from repo root):
    python scripts/check_volume_alternation.py
"""

import re
from pathlib import Path
from collections import defaultdict

VOLUME_DIR = Path("data/raw/volume")


def check_volume_alternation():
    files = sorted(VOLUME_DIR.glob("VS_*.txt"))
    if not files:
        print(f"No VS_ files found in {VOLUME_DIR}. Run generate_volume_set.py first.")
        return

    bugs = 0
    subtype_bugs = defaultdict(int)
    subtype_counts = defaultdict(int)

    for f in files:
        text = f.read_text(encoding="utf-8")

        # Extract sub_type for grouping
        sub_match = re.search(r"SUB_TYPE: (\S+)", text)
        sub_type = sub_match.group(1) if sub_match else "unknown"
        subtype_counts[sub_type] += 1

        # Extract speaker turns from body (everything after ---)
        body = text.split("---", 1)[1] if "---" in text else text
        speakers = re.findall(r"^(AGENT|CUSTOMER):", body, re.MULTILINE)

        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i - 1]:
                bugs += 1
                subtype_bugs[sub_type] += 1
                print(f"  BUG: {f.name} ({sub_type}) — consecutive "
                      f"{speakers[i]} at position {i}")

    print(f"\n{'='*55}")
    print(f"Files checked : {len(files)}")
    print(f"Total bugs    : {bugs}")

    if bugs:
        print("\nBugs by sub-type:")
        for sub, n in sorted(subtype_bugs.items()):
            print(f"  {sub:<35} {n} bugs / {subtype_counts[sub]} files")
    else:
        print("All transcripts pass alternation check.")

    print(f"{'='*55}")


if __name__ == "__main__":
    check_volume_alternation()