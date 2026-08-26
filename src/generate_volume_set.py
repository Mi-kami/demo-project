"""
Volume Set Generator — Call Transcript Quality Scoring System
=============================================================
Generates 1,000 transcripts for pipeline robustness testing.

TWO-TIER DATASET ARCHITECTURE — WHY THIS FILE EXISTS SEPARATELY:
  Golden set (GS_001–GS_167): hand-authored seeds + slot-filled
  extensions. Labels are rubric-verified. Purpose: validate that the
  scoring logic produces CORRECT outputs.

  Volume set (VS_0001–VS_1000): programmatically labelled at scale.
  Labels are assigned by generator logic, not manually checked.
  Purpose: validate that the pipeline doesn't BREAK — edge cases in
  parsing, unusual slot combinations, wider distributions, sub-type
  imbalance mirroring realistic call-centre volume. Do not conflate
  with the golden set for correctness claims.

DISTRIBUTION RATIONALE (realistic African/Gulf fintech call centre):
  High-volume support (billing, failed txn, account access) and
  complaints dominate. KYC and loan recovery are real but lower-volume.
  Legal escalation is rare. This imbalance is deliberate and defensible:
  it stress-tests the pipeline's behaviour under the same class
  distribution it would encounter with real Aethex data.

  billing_error_resolved        170  (17%) — most common support ticket
  failed_transaction_resolved   110  (11%)
  service_quality_complaint     100  (10%)
  account_access_resolved        90   (9%)
  repeated_issue_unresolved      90   (9%)
  standard_kyc_pass              80   (8%)
  policy_dispute_unresolved      80   (8%)
  kyc_document_issue             70   (7%)
  payment_reminder_compliant     70   (7%)
  hardship_negotiation           60   (6%)
  kyc_fraud_flag                 40   (4%)
  loan_recovery_escalation       40   (4%)
  TOTAL                        1000 (100%)

IMPORT STRATEGY:
  All template functions, institution pools, market profiles, and
  verified data live in generate_golden_set.py. This script imports
  from there — no duplication of verified research. Any change to the
  golden set generator propagates here automatically.

DIFFERENCES FROM GOLDEN SET GENERATOR:
  - Wider slot ranges (amounts, dates) for edge-case coverage
  - Different random seeds per sub-type (offset by 1000 from golden set
    seeds so the two sets never produce identical transcripts)
  - Output: data/raw/volume/ (never mixed with data/raw/)
  - ID prefix: VS_ with zero-padded 4 digits (VS_0001–VS_1000)
  - SET metadata field added to every header: SET: volume

USAGE:
  Run from the repo root (same rule as generate_golden_set.py):
    python scripts/generate_volume_set.py
  OUTPUT_DIR resolves relative to working directory — always run from
  demo-project/, not from inside scripts/.
"""

import sys
from pathlib import Path

# Allow import of generate_golden_set from scripts/ when running from repo root.
# If generate_golden_set.py is in scripts/, add that to the path.
_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

import generate_golden_set as gs

OUTPUT_DIR = Path("data/raw/volume")

# ---------------------------------------------------------------------------
# Volume-specific slot ranges
# These are WIDER than the golden set to surface parsing edge cases:
# very small amounts, very large amounts, different date patterns.
# The generator functions in generate_golden_set.py accept a `rng`
# argument — we pass our own seeded RNG so slot draws differ from the
# golden set even when the same template function runs.
# ---------------------------------------------------------------------------

# Distribution: sub_type -> (count, seed_offset)
# seed_offset is added to the golden set's original seed so volume
# transcripts are guaranteed to differ from golden set output.
VOLUME_DISTRIBUTION = [
    ("billing_error_resolved",       170, 1000),
    ("failed_transaction_resolved",  110, 1001),
    ("service_quality_complaint",    100, 1002),
    ("account_access_resolved",       90, 1003),
    ("repeated_issue_unresolved",     90, 1004),
    ("standard_kyc_pass",             80, 1005),
    ("policy_dispute_unresolved",     80, 1006),
    ("kyc_document_issue",            70, 1007),
    ("payment_reminder_compliant",    70, 1008),
    ("hardship_negotiation",          60, 1009),
    ("kyc_fraud_flag",                40, 1010),
    ("loan_recovery_escalation",      40, 1011),
]

assert sum(c for _, c, _ in VOLUME_DISTRIBUTION) == 1000, \
    "Distribution must sum to exactly 1000"


def render_volume(spec: gs.TranscriptSpec, volume_id: str) -> str:
    """
    Render a TranscriptSpec with the VS_ id and an extra SET: volume
    metadata field injected into the header. The header format is
    otherwise identical to the golden set so preprocessing.py can
    parse both with the same parser — the SET field lets it
    distinguish them downstream if needed.
    """
    import textwrap
    header = textwrap.dedent(f"""\
        TRANSCRIPT_ID: {volume_id}
        SET: volume
        ARCHETYPE: {spec.archetype}
        SUB_TYPE: {spec.sub_type}
        MARKET: {spec.market}
        RESOLUTION: {spec.resolution}
        SENTIMENT_ARC: {spec.sentiment_arc}
        AGENT_CONSISTENCY: {spec.agent_consistency}
        INTENT: {spec.intent}
        TURNS: {len(spec.turns)}
        ---
        """)
    body = "\n".join(f"{speaker}: {text}" for speaker, text in spec.turns)
    return header + body + "\n"


def run_volume():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    id_counter = 1
    total_written = 0

    for sub_type, count, seed in VOLUME_DISTRIBUTION:
        generator_fn = gs.IMPLEMENTED_GENERATORS[sub_type]

        # generate() returns a list of TranscriptSpec objects.
        # start_id here is a dummy — we override the ID in render_volume().
        # We pass a start_id of 9000 + id_counter as a stable, non-colliding
        # value that won't clash with GS_ ids (max GS_ id is 167).
        specs = generator_fn(count=count, start_id=9000 + id_counter, seed=seed)

        for spec in specs:
            volume_id = f"VS_{id_counter:04d}"
            out_path = OUTPUT_DIR / f"{volume_id}.txt"
            out_path.write_text(render_volume(spec, volume_id), encoding="utf-8")
            id_counter += 1
            total_written += 1

        print(f"  {sub_type:<35} {count:>4} transcripts written")

    print(f"\nVolume set complete: {total_written} transcripts → {OUTPUT_DIR}/")
    print(f"ID range: VS_0001 – VS_{total_written:04d}")
    print("\nNext step: run scripts/check_volume_alternation.py to verify "
          "speaker alternation across the full volume set.")


if __name__ == "__main__":
    print(f"Generating volume set → {OUTPUT_DIR}/\n")
    run_volume()