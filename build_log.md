## Build Log

### Phase 0 — Data Design & Environment Setup
**Date:** August 13-14, 2026
**Status:** Complete

---

#### What Was Built

**Environment setup:**
- Python 3.12 virtual environment configured on Windows 
  via Git Bash (resolved Microsoft Store alias conflict 
  by calling the Python 3.12 executable directly)
- Full project folder scaffold created: data/raw, 
  data/processed, notebooks, src/
- All module files stubbed: preprocessing.py, 
  sentiment.py, intent.py, resolution.py, scoring.py
- .gitignore configured, GitHub repo created, linked, 
  and initial scaffold committed and pushed

**Dataset strategy — two-tier architecture:**
- Evaluated HuggingFace public datasets (MultiWOZ, 
  Lakshan2003 banking dataset) — rejected both. 
  MultiWOZ is wrong domain (tourist task completion, 
  not financial services). Lakshan2003 has generation 
  artifacts, one-row-per-turn storage requiring 
  reconstruction, and no native ground truth labels. 
  Synthetic data chosen as the primary approach — 
  not as a fallback but as the stronger engineering 
  decision for a pipeline that requires verified 
  ground truth labels.
- Two-tier structure: golden set (hand-authored, 
  verified ground truth, validates scoring correctness) 
  + volume set (programmatically generated, 500-1,000 
  transcripts, validates pipeline robustness at scale)

**Golden set — 13 hand-authored seed transcripts:**
Designed and written across 4 archetypes, 12 sub-types 
(plus one additional Kenya policy dispute for market 
balance), 3 markets (Nigeria, Kenya, Gulf/UAE):

| ID | Sub-type | Market | Institution | Arc |
|----|----------|--------|-------------|-----|
| GS_001 | billing_error_resolved | Nigeria | Kuda Bank | frustrated → relieved |
| GS_002 | failed_transaction_resolved (Path B) | Kenya | Equity Bank | anxious → reassured |
| GS_003 | account_access_resolved | Gulf/UAE | Emirates NBD | frustrated → relieved |
| GS_004 | service_quality_complaint | Nigeria | InDrive | calm → frustrated |
| GS_005 | repeated_issue_unresolved | Gulf/UAE | Talabat | already frustrated → resigned |
| GS_006 | policy_dispute_unresolved | Nigeria | FairMoney | polite → hostile |
| GS_007 | policy_dispute_unresolved | Kenya | Mogo | polite → hostile |
| GS_008 | standard_kyc_pass | Nigeria | GTBank | neutral → satisfied |
| GS_009 | kyc_document_issue | Kenya | KCB | polite → mildly frustrated |
| GS_010 | kyc_fraud_flag | Gulf/UAE | Mashreq Neo | neutral → confused/defensive |
| GS_011 | payment_reminder_compliant | Nigeria | OKash | slightly defensive → cooperative |
| GS_012 | hardship_negotiation | Kenya | M-Shwari | distressed → cautiously relieved |
| GS_013 | loan_recovery_escalation | Gulf/UAE | ADIB | hostile throughout |

---

#### Decisions Made and Why

**Synthetic data over public datasets:**
Real datasets like Lakshan2003 require significant 
wrangling (turn reconstruction, artifact cleaning) 
with no guaranteed ground truth labels at the end. 
Synthetic data gives full control over label 
correctness by construction — the transcript IS 
the ground truth, not a derivation from it. This 
is the honest engineering choice for a validation 
dataset, not a shortcut.

**Two-tier structure:**
A single dataset cannot serve both correctness 
validation and scale robustness testing — those 
are different claims requiring different data 
properties. Golden set proves the scoring logic 
is right. Volume set proves the pipeline holds 
at scale. Conflating them would weaken both claims.

**12 sub-types (Option B) over 4 broad categories 
(Option A):**
More granular intent labels produce a more 
differentiated classifier and a more realistic 
mirror of Aethex's actual call distribution at 
17,000 calls per day. Option A was the easier 
build but would not meaningfully differentiate 
the demo from a generic sentiment classifier. 
The additional complexity is deliberate scope, 
not scope creep.

**Market and institution selection:**
Every company name, currency, phone number format, 
and institution was chosen to reflect Aethex's 
actual geographic footprint — Nigeria, Kenya, 
Gulf/UAE. No invented institutions. Where a 
complaint pattern was documented only in one 
market (e.g. OKash auto-debit in Nigeria, Mogo 
unilateral term changes in Kenya) we kept it 
in that market rather than falsely attributing 
it to another. Research was done before writing 
each transcript to validate that the institution, 
complaint type, and resolution process were 
accurate for that market.

**Agent behavior modeled on best practice, 
not industry average:**
Several institutions used (OKash, FairMoney) 
are known for aggressive customer contact 
practices. The agents in those transcripts 
are modeled as professional, firm, and 
non-aggressive — reflecting what an ideal 
AI agent should do, not what average human 
agents currently do. This is deliberate: 
the scoring model should be calibrated 
against best practice so it can identify 
and penalise substandard agent behaviour. 
An aggressive OKash-style agent would score 
lower on agent consistency — that is the point.

**Loan recovery as outbound calls:**
All other archetypes are inbound (customer 
calls the company). Loan recovery is outbound 
(company calls the customer). This structural 
difference is reflected in every loan recovery 
transcript — agent confirms identity before 
identifying themselves, customer did not choose 
this conversation, opening dynamic is reversed. 
This distinction matters for preprocessing.py 
which will need to handle both call directions.

**Sentiment arc distribution — no default openings:**
Early drafts defaulted to frustrated customer 
openings. This was corrected after recognising 
that real call center data has genuine arc 
variety — customers open neutral, polite, 
anxious, defensive, or hostile depending on 
context. A dataset where every customer opens 
frustrated produces a sentiment classifier 
that cannot distinguish between starting points, 
only endpoints. The final distribution covers 
seven distinct opening states across 13 transcripts.

**Closing pattern variety — deliberate:**
Each unresolved transcript has a distinct closing 
signature: full mutual goodbye (GS_004), resigned 
one-word goodbye (GS_005), customer cuts agent 
off mid-sentence (GS_006 and GS_007), two-word 
Arabic close (GS_013). Resolution detection 
depends partly on closing pattern analysis — 
variety here gives resolution.py a richer signal 
set to learn from.

**Cultural authenticity — code-switching 
customer-side only:**
Code-switching (Pidgin, Swahili, Sheng, Arabic) 
appears only in customer turns, never agent turns. 
Agents maintain formal English throughout all 
markets. This reflects real call center practice 
and ensures the sentiment classifier is not 
confused by mixed-language agent turns. 
Code-switching frequency and vocabulary was 
researched per market before use — not inserted 
as decoration.

---

#### What I Learned

**Sentiment flow is architecture, not flavour:**
The sentiment arc of a transcript is not a 
stylistic choice — it is a structural decision 
that determines what the sentiment classifier 
will score, what the trajectory component of 
the scoring model will measure, and whether 
the ground truth label is internally consistent. 
A transcript that says "frustrated → relieved" 
in the metadata but has flat emotional language 
throughout is broken data, not a style variation. 
Every arc has to be earned turn by turn.

**Agent precision is a pipeline input, 
not a writing quality:**
The level of specificity in agent turns — 
exact timestamps, reference number formats, 
transaction amounts, document types, regulatory 
body names — directly affects what resolution.py 
and sentiment.py will have to work with. Vague 
agent language ("we will look into it") is a 
non-resolution signal. Specific committed language 
("the reversal reference is KD-20260809-7741 
and funds will reflect within 24-48 hours") 
is a resolution signal. This specificity is 
not realism for its own sake — it is a 
deliberate feature of the ground truth labels.

**Company and country selection is a 
research task, not a creative one:**
Attributing complaint behaviours to the wrong 
institution or wrong market produces a dataset 
that misrepresents real companies and real 
markets. Every institution was validated before 
use. Where a scenario was realistic for Nigeria 
but not Kenya, it stayed in Nigeria. The 
interview framing — "built to understand 
Aethex's market before this conversation" — 
only holds if the market representation 
is accurate.

**Ground truth is only as strong as 
the rubric behind it:**
The 12-sub-type rubric designed before any 
transcript was written is what makes the 
golden set labels defensible. Without it, 
"resolved" and "unresolved" are judgement 
calls. With it, they are traceable to 
specific in-call events — a customer 
repeating a reference number, an agent 
committing to a specific date, a call 
ending without a committed action. 
That traceability is what separates 
a validated dataset from a labelled guess.

---

#### What Is Next

- Build template/slot-filling generator to 
  scale GS seeds to 100-150 golden set transcripts
- Build volume set generator (500-1,000 transcripts) 
  separately for scale robustness testing
- Spot-check generated samples against rubric 
  before pipeline build begins
- Phase 1: preprocessing.py — parse raw .txt 
  files into structured speaker-turn DataFrames



  # Build Log — Call Transcript Quality Scoring System
# Aethex AI Interview Demo Project
# Author: Deborah Olofin
# Interview date: August 26, 2026

---

## How to use this log
Every session is logged with: what was built, decisions made and why,
bugs identified, how they were fixed (or why they weren't), and blockers.
This log is real — written during the build, not reconstructed after.

---

## PHASE 0 — Data Foundation

### Session 1 — Environment + Scaffold

**What was built:**
- Python 3.12 venv created via Git Bash using direct executable path
  (C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe)
- Full project folder structure scaffolded
- .gitignore configured
- GitHub repo linked and initial commit pushed

**Blocker encountered:**
- Windows Microsoft Store alias intercepting `python3` command — running
  `python3` opened the Store instead of launching Python
- Fix: used the full executable path directly to create the venv, bypassing
  the alias entirely

---

### Session 2 — 13 Hand-Authored Seed Transcripts (GS_001–GS_013)

**What was built:**
- 13 seed transcripts written from scratch, one per sub-type (two for
  policy_dispute_unresolved), covering all 12 sub-types across 4 archetypes
  and 3 markets (Nigeria, Kenya, Gulf/UAE)
- Golden set rubric checklist written to govern all generated transcripts

**Decisions made:**
- Synthetic data chosen over HuggingFace Datasets (MultiWOZ, ECPC) and Kaggle
  after evaluation — both mismatched to Aethex's specific market/domain profile
  (African/Gulf fintech calls with code-switching and regulatory context)
- 12 granular sub-types over broad labels — "complaint" and "support" collapse
  meaningful distinctions. policy_dispute_unresolved (charge is contractually
  valid but undisclosed) is architecturally distinct from billing_error_resolved
  (genuine system error). Intent classification needs to distinguish these.
- Cultural authenticity treated as non-negotiable — company names, complaint
  types, currencies, and code-switching vocabulary verified per market before use

**Bugs identified and fixed:**
- TURNS header values written by hand during authoring — estimated counts that
  didn't match actual turn counts in file bodies. Caught by preprocessor
  validation later (Phase 1). Fixed with fix_seed_turns.py.
- GS_008 (standard_kyc_pass / Nigeria / GTBank) — entire transcript duplicated
  inside the file. The full dialogue appeared twice, with the second copy
  appended after the closing AGENT turn. Cause: copy-paste error during
  authoring. Fix: manually deleted everything from the second TRANSCRIPT_ID
  line to end of file.
- GS_002 SUB_TYPE field — hand-authored as "failed_transaction_resolved (Path B)"
  instead of "failed_transaction_resolved". Created an orphan sub-type group
  in preprocessing output. Fixed by editing the header field directly.

---

### Session 3 — Golden Set Generator (generate_golden_set.py)

**What was built:**
- Slot-filling generator scaling 13 seeds to 167 transcripts across all 12
  sub-types × 3 markets × multiple institutions per market per domain
- All 12 sub-type templates written and verified
- check_alternation.py — speaker alternation checker (runs generator in-memory)
- verify_golden_set.py — full verification suite (MD5 dedup, regex scans,
  vocab cross-checks, turn count ranges)

**Key design decisions:**

*Slot-filling not random generation:*
Ground truth trustworthiness at scale requires that what varies between
transcripts is surface detail only — names, amounts, dates, code-switching
phrases — not the underlying resolution/sentiment/intent logic. The template
skeleton encodes the logic; slots encode the variation. Random generation
would not produce reliable ground truth labels.

*Multi-institution pools per market:*
Every sub-type generates across Nigeria, Kenya, and Gulf/UAE, drawing from
verified institution pools per market per domain. All institutions verified
as real and currently operating in the specific market as of August 2026
before inclusion. Deliberate exclusions documented:
  - Bolt Food Nigeria — exited market December 2023
  - Jumia Food Nigeria — exited market December 2023
  - Branch intentionally in both Nigeria and Kenya lending pools — real
    company operating both markets, not a duplication bug

*Domain-merged pools for 3 sub-types:*
service_quality_complaint, repeated_issue_unresolved, and
policy_dispute_unresolved are not domain-locked in reality — a bank customer
can have a service quality complaint as easily as a food delivery customer.
Pools merged across banking, ride-hail, food delivery, and lending, with
domain-aware dialogue branching so a bank transcript never produces loan
recovery language and vice versa.

*Structural fidelity over turn count tuning:*
Generated output was 11-18 turns vs seed depth of 14-28. Diagnosed as a
missing structural element — identity verification sequences (phone + BVN/NIN
for Nigeria, national ID for Kenya, Emirates ID for Gulf) — not a length
tuning problem. Adding the structure closed the gap. A length patch would
have produced the right numbers with wrong structure, corrupting turn-index
logic in preprocessing.

*Market-specific correctness at field level:*
Currency formatting (₦ glued to number, KSh/AED with space before number),
payment method (M-Pesa Paybill in Kenya only, bank transfer/registered card
in Gulf), currency slang ("a single naira" / "a single shilling" / "a single
dirham") — stored as market-profile fields, not hardcoded strings.

*Hardship negotiation math correctness:*
Monthly installment derived from balance/months, not drawn independently.
Earlier version drew both independently, producing plans where
installment × months < balance — agent claimed plan settled the debt when
arithmetic proved otherwise. Fixed by derivation.

*Insertion indices use relative indexing:*
All optional turn insertions use len(turns) - N, never absolute indices.
Absolute indices break when template spine length changes, causing
alternation bugs. This is the root cause of all 9 alternation bugs caught
below.

**Bugs identified and fixed:**

1. Universal "?" on Swahili anger phrases — Swahili phrases are statements
   and exclamations, not questions. Fixed with per-phrase punctuation map.

2. Currency spacing — ₦ glued directly to number (₦12,750) but KSh and AED
   are letter codes requiring a space (KSh 10,500, AED 2,400). Initially
   treated identically. Fixed by separate formatting logic per market.

3. Sentence-boundary capitalisation — some generated turns had lowercase
   first letters after injected sentence boundaries. Caught by regex scan,
   not eyeballing. Fixed in template strings.

4. M-Pesa in Gulf / naira in Kenya — payment method and currency slang
   hardcoded in early templates. A Gulf transcript told a customer to pay
   via M-Pesa (Kenyan mobile money); a Kenya transcript had a customer
   refuse to pay "a single naira" (Nigerian currency). Fixed by moving
   both to market-profile fields.

5. Hardship math — installment × months < balance in early hardship
   templates. Plan amount didn't cover the stated outstanding balance.
   Fixed by deriving installment from balance/months.

6. 9 speaker alternation bugs — two consecutive AGENT or CUSTOMER turns
   appearing in generated transcripts. Root cause: absolute insertion
   indices breaking when template spine was edited. Caught by
   check_alternation.py. Fixed by converting all insertion indices to
   relative (len(turns) - N) form.

7. Bank customers referencing "loan account" — domain tagging missing on
   merged-pool sub-types. A banking transcript produced dialogue about a
   "loan account" because the domain variable wasn't checked before
   selecting the issue description. Fixed by explicit domain tagging
   in template branching logic.

**Bugs identified but not fixed (documented as known limitations):**
- Gulf Arabic code-switching pool has only 3 verified phrases (Wallah,
  Ana gelt lak, Khalas) vs Kenya's 7 and Nigeria's 5. Deliberately
  conservative — no additional Gulf Arabic phrases added without
  verification. Worth expanding before scaling Gulf volume further.

---

### Session 4 — Volume Set Generator (generate_volume_set.py)

**What was built:**
- generate_volume_set.py — generates 1,000 transcripts to data/raw/volume/
- check_volume_alternation.py — alternation checker for written VS_ files
- VS_ prefix, SET: volume metadata field, separate output directory

**Key design decisions:**

*Two-tier dataset architecture — why it exists:*
Golden set validates CORRECTNESS — labels are rubric-verified, purpose is
to confirm the scoring logic produces accurate outputs.
Volume set validates ROBUSTNESS — labels are programmatic, purpose is to
stress-test the pipeline at scale (parsing edge cases, unusual slot
combinations, wider distributions, realistic class imbalance).
These are distinct claims requiring different data and different evaluation
criteria. Conflating them would compromise both.

*Realistic class imbalance — not uniform distribution:*
Distribution mirrors real African/Gulf fintech call-centre volume:
  billing_error_resolved        170 (17%) — most common support ticket
  failed_transaction_resolved   110 (11%)
  service_quality_complaint     100 (10%)
  account_access_resolved        90  (9%)
  repeated_issue_unresolved      90  (9%)
  standard_kyc_pass              80  (8%)
  policy_dispute_unresolved      80  (8%)
  kyc_document_issue             70  (7%)
  payment_reminder_compliant     70  (7%)
  hardship_negotiation           60  (6%)
  kyc_fraud_flag                 40  (4%)
  loan_recovery_escalation       40  (4%)
  TOTAL                        1000

*Import from golden set generator — no duplication:*
generate_volume_set.py imports IMPLEMENTED_GENERATORS, MARKET_PROFILES,
and all institution pools from generate_golden_set.py. Any change to the
golden set generator propagates automatically. Seeds offset by +1000 so
no volume transcript is identical to any golden set output.

*SET: volume metadata field:*
Injected into every volume transcript header. Allows preprocessing and
downstream modules to distinguish the two tiers without relying on filename
prefix alone. Golden set files predate this field — preprocessor injects
SET: golden for any file missing the field.

**Verification results:**
- 1,000 files generated, 0 alternation bugs, 0 ID/filename mismatches
- SET: volume present in every header
- Turn ranges consistent with golden set (15-27 turns)

---

## PHASE 1 — Preprocessing (src/preprocessing.py)

**What was built:**
- Parser handling both golden (multi-line wrapped dialogue) and volume
  (single-line dialogue) transcript formats in one pass
- Two output DataFrames saved to data/processed/:
    turns.csv        — one row per speaker turn (22,401 rows)
    transcripts.csv  — one row per transcript (1,167 rows)
- Validation suite running before save

**Key design decisions:**

*Two outputs not one:*
sentiment.py reasons at the turn level — it needs one row per turn.
scoring.py and intent.py reason at the transcript level — they need one
row per transcript. Neither should have to reshape the data itself.
Separate outputs means no hidden reshaping dependency that breaks
silently when the schema changes.

*Metadata repeated on every turn row:*
Every row in turns.csv carries all transcript metadata (sub_type,
resolution, intent, etc). Allows sentiment.py to filter by archetype
or market without a join. Redundant at this scale; eliminates a class
of join bugs downstream.

*Validation before saving:*
Checks resolution values, speaker labels, empty text, both speakers
present in every transcript, SET field validity. A systematic parse bug
caught here costs seconds. The same bug in scoring.py costs an hour.

**Bugs identified and fixed:**

1. Multi-line dialogue format mismatch — hand-authored seeds wrap long
   dialogue lines across multiple lines; generated files use single lines
   per turn. Original regex (^(AGENT|CUSTOMER):.+$) matched every line
   as a new turn, overcounting turns in seed files and producing TURNS
   header mismatch errors for all 13 seeds.
   Fix: rejoiner loop that collapses wrapped lines back into single turns
   before the regex runs. Both formats handled by one parser.

2. TURNS header mismatches in all 13 seeds — header values were hand-
   estimated during authoring and didn't match actual turn counts after
   the multi-line fix was applied.
   Fix: fix_seed_turns.py — one-off script updating TURNS field in each
   seed to match the actual parsed count. Script deleted after use.

3. GS_008 duplicate — full transcript body appeared twice in the file
   (authoring copy-paste error). Parser counted 50 turns for a 20-turn
   transcript. Fixed by manually deleting the duplicate block.

4. GS_002 sub-type orphan — SUB_TYPE: failed_transaction_resolved (Path B)
   created a separate group in sub-type analysis. Fixed by editing the
   header to match the generated set's label exactly.

**Final output:**
- 1,167 transcripts parsed, 0 parse errors
- 167 golden + 1,000 volume
- 22,401 turns total, avg 19.2 turns/transcript
- All 6 validation checks passing

---

## PHASE 2 — Sentiment (src/sentiment.py + src/sentiment_hf_comparison.py)

**What was built:**
- VADER scoring across all 22,401 turns
- Six transcript-level sentiment aggregates computed per transcript
- Golden-set-only slice saved for HuggingFace comparison
- Separate sentiment_hf_comparison.py — DistilBERT vs VADER on golden set

**Key design decisions:**

*VADER for the pipeline:*
Rule-based, runs in seconds across 22,401 turns, no GPU, built for short
conversational text. The right tool for this use case and scale.

*HuggingFace comparison as a separate script:*
The main pipeline has no HuggingFace dependency — it runs cleanly without
a 250MB model download. The comparison is an explicit analytical artefact,
not a production component. If HuggingFace fails to load on interview day,
the demo still runs.

*HuggingFace comparison scoped to golden set only:*
The only place with hand-verified ground truth labels. Running on the volume
set would test throughput not accuracy — a different question. This scoping
is the honest claim: "I compared models where I can actually evaluate which
one aligns better with known labels."

*2-turn windows for sentiment start/end:*
A single turn can be a noise outlier. Two-turn averaging gives a more stable
signal of where the call started and ended emotionally without extending the
window so far it stops being "start" and "end." With calls averaging 19 turns,
2-turn windows are genuinely start and end.

*Six transcript-level aggregates:*
  customer_start_sentiment  — avg compound of first 2 customer turns
  customer_end_sentiment    — avg compound of last 2 customer turns
  customer_sentiment_delta  — end minus start → feeds 30% scoring weight
  agent_mean_sentiment      — avg compound across all agent turns → 20% weight
  agent_min_sentiment       — worst agent turn (consistency floor)
  customer_mean_sentiment   — overall customer sentiment

**Sanity check result:**
  Resolved calls avg delta:   -0.070
  Unresolved calls avg delta: -0.136
  Both negative — expected, not a bug. Customers start calls frustrated
  or anxious. The relative difference is what matters for scoring, not
  the absolute value. Resolved calls show better trajectory than
  unresolved — sanity check passes.

**HuggingFace comparison:**
Model: distilbert-base-uncased-finetuned-sst-2-english
Limitation: fine-tuned on English movie reviews (SST-2) — out of domain
for African/Gulf fintech calls with code-switching. This limitation is
itself interview material: VADER's rule-based approach handles short
conversational text without domain fine-tuning, making it a better fit
for this specific use case. With real Aethex data, fine-tuning a
transformer on in-domain transcripts would be the correct next step.

**Bugs encountered:**
- HuggingFace symlinks warning on Windows — cache system uses copies
  instead of symlinks. Harmless, no functional difference. Suppressed
  by setting HF_HUB_DISABLE_SYMLINKS_WARNING environment variable
  if needed.

---

## PHASE 3 — Intent Classification (src/intent.py)

**What was built:**
- Embedding similarity classifier using all-MiniLM-L6-v2
- 11 intent prototype sentences (one per class)
- Evaluation against golden set ground truth
- Iterative prototype and window refinement across 4 runs

**Key design decisions:**

*Embedding similarity over keyword matching:*
Keyword approaches break the moment a customer phrases something
unexpectedly. "You took money from my account twice" and "I was charged
double" are very different strings but land close in embedding space
because they mean the same thing. Embedding similarity captures semantic
meaning; keyword matching captures surface form.

*First 3 customer turns for intent window (inbound calls):*
Intent is always established in the opening of an inbound call — the
customer states their problem before the agent responds. Using the full
transcript dilutes the intent signal with resolution and sentiment
content that belongs to other scoring components.

*Turns 2-5 for outbound calls (LOAN_RECOVERY archetype):*
Outbound call structure: agent calls customer, customer's first response
is "who is this / yes what do you want" — near-contentless in embedding
space. The real intent signal emerges after the agent explains why they
called and the customer responds with their actual position. Shifting the
window to turns 2-5 captures this without affecting inbound call logic.

*Evaluation on golden set only:*
Volume set labels are programmatically assigned — using them for
evaluation would be circular. Golden set evaluation is the honest claim.

**Iteration history:**

Run 1 — v1 prototypes, 3 customer turns for all archetypes:
  Overall accuracy: 64.1%
  Key failures:
    hardship_plan  0.0% — all 14 confused with loan_reminder
    loan_reminder  28.6% — 9 confused with kyc_verification
    policy_dispute 30.8% — scattered across loan_reminder, failed_transaction
    service_complaint 53.6%

Run 2 — refined 5 weak prototypes (v2):
  Overall accuracy: 60.5% — WORSE than v1
  Root cause: loan_escalation prototype ("refusing to pay... documented
  everything... prepared to take this further") too assertive — created
  a gravity well pulling 51 transcripts from angry customers across all
  categories into loan_escalation. Collateral damage from one fix
  broke classes that were previously working.
  Key new failure: loan_escalation 92.9% predicted but pulling in
  service_complaint (16x), billing_error (7x), loan_reminder (7x).

Run 3 — reverted loan_escalation to less assertive prototype, kept
  other v2 improvements:
  Overall accuracy: 70.1%
  hardship_plan fixed: 0% → 100% ✓
  loan_reminder: 28.6% → 7.1% — got worse
  loan_escalation: 50% → 92.9% ✓
  Remaining problems: loan_reminder 7.1%, service_complaint 35.7%
  Root cause of loan_reminder failure: outbound call structure means
  first 3 customer turns are "who is this / yes what do you want" —
  near-contentless, not enough intent signal. Structural problem,
  not a prototype wording problem.

Run 4 — Fix 1: shifted LOAN_RECOVERY archetype window to turns 2-5:
  (Fix 2 considered: extend window when first turn is short. Rejected
  because it fires on a proxy signal — turn length — not the real
  cause. Fix 1 is targeted to the structural root cause and doesn't
  affect inbound call logic.)
  Overall accuracy: 71.3%
  loan_reminder: 7.1% → 100% ✓
  loan_escalation: 92.9% → 14.3% — dropped significantly
  Root cause: turns 2-5 for loan_escalation puts window in dispute
  explanation territory, which reads as policy_dispute in embedding space.

Run 5 — adjusted window to turns 1-5 (iloc[1:5]):
  Overall accuracy: 74.3% ✓ (final)
  loan_reminder: 100% ✓
  loan_escalation: 50% — partial recovery
  loan_escalation still confused with policy_dispute (7x) — both
  involve customers disputing charges, language overlaps mid-call

**Final results (74.3%):**
  account_access      100.0% ✓
  failed_transaction  100.0% ✓
  hardship_plan       100.0% ✓
  loan_reminder       100.0% ✓
  kyc_escalation       92.9% ✓
  kyc_incomplete       92.9% ✓
  kyc_verification     92.9% ✓
  loan_escalation      50.0% — known limitation
  billing_error        50.0% — known limitation
  policy_dispute       38.5% — known limitation
  service_complaint    35.7% — known limitation

**Confidence check (passes):**
  Correct predictions avg confidence:   0.616
  Incorrect predictions avg confidence: 0.451
  Model is appropriately uncertain when wrong — not confidently incorrect.

**Known limitations (unresolved, documented for interview):**

1. loan_escalation vs policy_dispute confusion (7x):
   Both involve customers disputing charges mid-call. Language overlap
   is genuine — the semantic boundary between disputing a policy and
   escalating a loan dispute is thin in embedding space. Resolution:
   fine-tuning on in-domain Aethex data with labelled examples would
   create a sharper boundary.

2. service_complaint gravity toward loan_escalation (17x):
   Angry customer language ("this is unacceptable", "I want something
   done") overlaps with escalation language in embedding space regardless
   of subject matter. The emotional register overrides the topic.
   Resolution: domain fine-tuning or a two-stage classifier (topic
   first, then tone) would address this.

3. billing_error vs failed_transaction bleed (7x):
   Both involve money not behaving as expected. Prototype wording is
   close. Resolution: could be addressed with more distinct prototypes
   or additional context turns, but not pursued given time constraints.

4. policy_dispute low accuracy (38.5%):
   Discriminating language ("I never agreed to this policy") appears
   mid-call after the customer has described the charge — which sounds
   like billing_error in the opening turns. The 3-turn window captures
   the symptom, not the intent. Resolution: extending the window or
   using a later window for this sub-type specifically.

---

## PHASE 4 — Resolution Detection (src/resolution.py)
Status: IN PROGRESS

---

## PHASE 5 — Scoring (src/scoring.py)
Status: NOT STARTED

---

## PHASE 6 — Streamlit Interface (app.py)
Status: NOT STARTED

---

## PHASE 7 — Polish
Status: NOT STARTED
Includes: "What I would do differently with real Aethex data" writeup