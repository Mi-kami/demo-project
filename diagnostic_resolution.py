"""
diagnose_resolution.py — print actual closing turns of failing transcripts
Run from repo root: python diagnose_resolution.py
"""

import pandas as pd
from pathlib import Path

turns_df = pd.read_csv("data/processed/turns.csv")

# The transcripts we need to see
targets = [
    # False positives — predicted resolved, actually unresolved
    ("GS_009", "kyc_document_issue"),
    ("GS_010", "kyc_fraud_flag"),
    ("GS_103", "kyc_document_issue"),
    ("GS_116", "kyc_fraud_flag"),
    # False negatives — predicted unresolved, actually resolved
    ("GS_011", "payment_reminder_compliant"),
    ("GS_012", "hardship_negotiation"),
]

for tid, sub_type in targets:
    t = turns_df[turns_df["transcript_id"] == tid]
    last_5 = t.tail(5)
    print(f"\n{'='*60}")
    print(f"{tid}  |  {sub_type}")
    print(f"{'='*60}")
    for _, row in last_5.iterrows():
        print(f"[{row['speaker']}]: {row['text']}")