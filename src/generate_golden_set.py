"""
Golden Set Generator — Call Transcript Quality Scoring System
================================================================
Scales the 13 hand-authored, rubric-verified seed transcripts to a
~150 transcript golden set via controlled slot-filling.

DESIGN PRINCIPLE (why this exists):
Ground truth labels must remain trustworthy at scale. Instead of
randomly generating dialogue, each sub-type has a small number of
hand-authored SCENARIO TEMPLATES — full turn-by-turn skeletons with
slots for the details that don't change the underlying
resolution/sentiment/intent logic: customer name, agent name, amount,
date, reference number, and code-switching phrase choice.

MARKET COVERAGE (updated per review — see build_log):
Every sub-type now generates across all three markets (Nigeria, Kenya,
Gulf/UAE), AND across multiple institutions per market per domain —
not locked to the single institution each original seed happened to
use. Institutions are grouped by call-domain, since a real institution
handles all of ITS OWN sub-types, not just one:
  - Banking (billing / failed-txn / access / all 3 KYC sub-types):
      NG: Kuda, GTBank, OPay, Moniepoint
      KE: Equity Bank, KCB, Absa Kenya, NCBA
      GULF: Emirates NBD, Mashreq Neo, ADCB, RAKBANK
  - Ride-hail (service_quality_complaint):
      NG: InDrive, Bolt | KE: Bolt, Uber | GULF: Careem, Uber
  - Food delivery (repeated_issue_unresolved):
      NG: Chowdeck, Glovo | KE: Glovo, Bolt Food | GULF: Talabat, Deliveroo
  - Lending (policy_dispute_unresolved + all 3 loan-recovery sub-types):
      NG: FairMoney, OKash, Carbon, Branch
      KE: Mogo, M-Shwari, Tala, Branch
      GULF: ADIB, Dubai Islamic Bank

Every institution NOT already in the 13 seeds was verified via web
search before inclusion (confirmed real, currently-operating in that
specific market as of Aug 2026) — per the "research before attributing
company behavior" rule. Two things worth flagging from that research:
  - Bolt Food and Jumia Food both exited Nigeria's food delivery market
    (Dec 2023) — deliberately excluded from the NG food-delivery pool
    even though they're real brands, since including them would be
    inaccurate for the current market.
  - Branch operates as a lending app in both Nigeria and Kenya — it's
    intentionally in both pools, not a duplication bug.
Gulf currency is AED throughout since every Gulf institution here is
UAE-based.

KNOWN LIMITATION: Gulf Arabic code-switching currently has only 2
verified anger-register phrases (wallah, ana gelt lak) vs. Kenya's 4
and Nigeria's 2 Pidgin phrases — narrower variety by design, since I
won't add more Gulf Arabic slang without verifying it first. Worth
researching a few more before scaling Gulf volume much further.

STATUS: Engine + registry cover all 12 sub-types x 3 markets. Two
sub-types are fully templated as worked examples across all three
markets (billing_error_resolved, policy_dispute_unresolved). The
remaining 10 sub-types are registered with metadata but raise
NotImplementedError until their templates are written.
"""

import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path("data/raw")  # relative to demo-project root; adjust if run elsewhere

# ---------------------------------------------------------------------------
# Market profiles: names, currency, code-switching phrase pools
# ---------------------------------------------------------------------------

NIGERIAN_CUSTOMER_NAMES = [
    "Chiamaka", "Tunde", "Ngozi", "Emeka", "Yetunde", "Ifeoma",
    "Bolanle", "Chidi", "Amaka", "Segun", "Uche", "Folake",
]
KENYAN_CUSTOMER_NAMES = [
    "Wanjiru", "Otieno", "Achieng", "Kamau", "Njeri", "Odhiambo",
    "Wambui", "Mutua", "Akinyi", "Kariuki", "Nyambura", "Owino",
]
GULF_CUSTOMER_NAMES = [
    "Fatima", "Ahmed", "Mariam", "Khalid", "Aisha", "Omar",
    "Layla", "Rashid", "Noura", "Hassan", "Salim", "Huda",
]
AGENT_NAMES = [
    "Ada", "Kelechi", "Grace", "Femi", "Blessing", "Ifeanyi",
    "Chinwe", "David", "Zainab", "Michael",
]

# Anger/frustration phrases mapped to correct closing punctuation —
# these are STATEMENTS in the source language except where noted, so a
# universal "?" is wrong (this was the bug caught in the first pass).
# Expanded via research into authentic Nigerian Pidgin / Kenyan Swahili /
# Gulf Arabic frustration registers — not just textbook phrases.
NG_ANGER_PHRASES = {
    "E don do": "!",       # "it's enough / that's it" — exclamation
    "Na so e be": ".",     # "that's just how it is" — flat, resigned statement
    "Chai": "!",           # exclamation of frustration/disbelief
    "Na wa oo": "!",       # expresses disbelief/disappointment at what just happened
    "I no gree": "!",      # "I don't accept this" — assertive refusal
}
NG_OPEN_PHRASES = ["Abeg", "Please o"]  # "please" — softening openers, "o" is an emphatic Pidgin particle

KE_ANGER_PHRASES = {
    "Hii ni nini": "?",         # "what is this" — a genuine question
    "Sitaki mchezo": ".",       # "I don't want games" — flat statement
    "Watu hawajui": ".",        # "people don't know" — flat statement
    "Mwisho wa story": ".",     # "end of story" — closing statement
    "Hii haikubaliki": "!",     # "this is unacceptable" — strong, formal-register complaint
    "Mpaka lini": "?",          # "until when" — exasperated question, fits repeated-issue resignation
    "Nimechoka": ".",           # "I am tired / I'm done" — resigned statement
}
KE_OPEN_PHRASES = ["Sawa", "Pole", "Asante", "Tafadhali"]

GULF_ANGER_PHRASES = {
    "Wallah": "!",          # "I swear" — exclamation, intensifier
    "Ana gelt lak": ".",    # "I told you" — flat statement
    "Khalas": "!",          # "enough / done / stop it" — very natural frustration-closer in Gulf Arabic
}
GULF_OPEN_PHRASES = ["Shukran"]  # "thank you" — used as a polite opener

MARKET_PROFILES = {
    # currency: no trailing space (₦ is a glyph, glued to the number, per seed convention).
    # KSh/AED are letter codes and need a space before the number — that's the bug fixed below.
    # payment_method / currency_slang: market-appropriate substitutes, added after catching
    # a bug where a Gulf transcript told a customer to pay via M-Pesa (Kenyan mobile money)
    # and a Kenya transcript had a customer refuse to pay "a single naira" (Nigerian currency).
    "NG": dict(label="Nigeria", currency="₦", customer_names=NIGERIAN_CUSTOMER_NAMES,
               open_phrases=NG_OPEN_PHRASES, anger_phrases=NG_ANGER_PHRASES,
               payment_method="bank transfer or the app wallet", currency_slang="a single naira"),
    "KE": dict(label="Kenya", currency="KSh ", customer_names=KENYAN_CUSTOMER_NAMES,
               open_phrases=KE_OPEN_PHRASES, anger_phrases=KE_ANGER_PHRASES,
               payment_method="M-Pesa Paybill", currency_slang="a single shilling"),
    "GULF": dict(label="Gulf/UAE", currency="AED ", customer_names=GULF_CUSTOMER_NAMES,
                 open_phrases=GULF_OPEN_PHRASES, anger_phrases=GULF_ANGER_PHRASES,
                 payment_method="bank transfer or registered card", currency_slang="a single dirham"),
}

# ---------------------------------------------------------------------------
# Institution pools by call-domain (verified real operators per market)
# ---------------------------------------------------------------------------

BANKING = {
    "NG": [("Kuda", "KD", "bank"), ("GTBank", "GT", "bank"), ("OPay", "OP", "bank"), ("Moniepoint", "MP", "bank")],
    "KE": [("Equity Bank", "EQ", "bank"), ("KCB", "KCB", "bank"), ("Absa Kenya", "ABSA", "bank"), ("NCBA", "NCBA", "bank")],
    "GULF": [("Emirates NBD", "ENBD", "bank"), ("Mashreq Neo", "MQ", "bank"), ("ADCB", "ADCB", "bank"), ("RAKBANK", "RAK", "bank")],
}
RIDE_HAIL = {
    "NG": [("InDrive", "IND", "ride"), ("Bolt", "BOLT", "ride")],
    "KE": [("Bolt", "BOLT", "ride"), ("Uber", "UBR", "ride")],
    "GULF": [("Careem", "CRM", "ride"), ("Uber", "UBR", "ride")],
}
FOOD_DELIVERY = {
    "NG": [("Chowdeck", "CHD", "food"), ("Glovo", "GLV", "food")],
    "KE": [("Glovo", "GLV", "food"), ("Bolt Food", "BLTF", "food")],
    "GULF": [("Talabat", "TLB", "food"), ("Deliveroo", "DLV", "food")],
}
LENDING = {
    "NG": [("FairMoney", "FM", "lender"), ("OKash", "OK", "lender"), ("Carbon", "CRB", "lender"), ("Branch", "BRN", "lender")],
    "KE": [("Mogo", "MG", "lender"), ("M-Shwari", "MSW", "lender"), ("Tala", "TALA", "lender"), ("Branch", "BRN", "lender")],
    "GULF": [("ADIB", "ADIB", "lender"), ("Dubai Islamic Bank", "DIB", "lender")],
}

def merge_pools(*pools: dict) -> dict:
    """Combine institution pools across domains, market by market."""
    merged = {}
    for market in ("NG", "KE", "GULF"):
        combined = []
        for pool in pools:
            combined.extend(pool[market])
        merged[market] = combined
    return merged

# service_quality_complaint and repeated_issue_unresolved aren't limited to one
# domain in real life — a customer can have a service complaint with their bank
# just as easily as their ride-hail or food-delivery app. Merged, not domain-locked.
COMPLAINT_POOL = merge_pools(BANKING, RIDE_HAIL, FOOD_DELIVERY)

# policy_dispute_unresolved likewise isn't lending-only — banks impose disputed
# fees/terms too (account maintenance fees, FX conversion fees, etc).
DISPUTE_POOL = merge_pools(BANKING, LENDING)

SUBTYPE_INSTITUTION_POOL = {
    "billing_error_resolved": BANKING,
    "failed_transaction_resolved": BANKING,
    "account_access_resolved": BANKING,
    "standard_kyc_pass": BANKING,
    "kyc_document_issue": BANKING,
    "kyc_fraud_flag": BANKING,
    "service_quality_complaint": COMPLAINT_POOL,
    "repeated_issue_unresolved": COMPLAINT_POOL,
    "policy_dispute_unresolved": DISPUTE_POOL,
    "payment_reminder_compliant": LENDING,
    "hardship_negotiation": LENDING,
    "loan_recovery_escalation": LENDING,
}

def distribute_markets(count: int, markets=("NG", "KE", "GULF")) -> list:
    """Cycle through markets so a batch is spread roughly evenly."""
    return [markets[i % len(markets)] for i in range(count)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_date(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    month = rng.choice(["June", "July", "August"])
    return f"{month} {day}, 2026"

def date_to_yyyymmdd(rng: random.Random) -> str:
    month = rng.choice(["06", "07", "08"])
    day = f"{rng.randint(1, 28):02d}"
    return f"2026{month}{day}"

def non_round_amount(rng: random.Random, low: int, high: int) -> int:
    while True:
        val = rng.randint(low, high)
        if val % 500 != 0:
            return val

def ref_number(rng: random.Random, inst_abbr: str, purpose: str) -> str:
    return f"{inst_abbr}-{purpose}-{date_to_yyyymmdd(rng)}-{rng.randint(1000, 9999)}"


# ---------------------------------------------------------------------------
# Identity-verification data generators — added because the real seeds all
# open with a verification exchange (phone number + account/ID number)
# before the agent proceeds. The original templates skipped straight to
# "let me pull up your account," which was a structural gap, not a length
# problem — this is what actually needed fixing, not more insertions.
# ---------------------------------------------------------------------------

def gen_phone(market: str, rng: random.Random) -> str:
    if market == "NG":
        prefix = rng.choice(["070", "080", "081", "090", "091"])
        return f"{prefix}{rng.randint(1000000, 9999999)}"
    if market == "KE":
        return f"07{rng.randint(10, 39)} {rng.randint(100, 999)} {rng.randint(100, 999)}"
    return f"05{rng.randint(0, 9)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"  # GULF

def gen_bvn(rng: random.Random) -> str:
    return str(rng.randint(10000000000, 99999999999))

def gen_nin(rng: random.Random) -> str:
    return f"{rng.randint(10000, 99999)} {rng.randint(10000, 99999)}"

def gen_ke_national_id(rng: random.Random) -> str:
    return str(rng.randint(10000000, 39999999))

def gen_emirates_id(rng: random.Random) -> str:
    return f"784-{rng.randint(1980, 2005)}-{rng.randint(1000000, 9999999)}-{rng.randint(1, 9)}"

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}

def gen_dob(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    suffix = _ORDINAL_SUFFIX.get(day, "th")
    month = rng.choice(_MONTHS)
    year = rng.randint(1970, 2001)
    return f"{day}{suffix} {month} {year}"

def gen_last4(rng: random.Random) -> str:
    return str(rng.randint(1000, 9999))


def apply_variation(rng: random.Random, turns: list, insertions: list) -> list:
    """Turn-count variation mechanism, shared across all templates.

    WHY THIS EXISTS: without it, every transcript in a sub-type has identical
    turn count, which flattens call_efficiency (10% of the scoring model) —
    there'd be nothing for that component to differentiate on within a
    sub-type. Each insertion is an optional, scenario-appropriate exchange
    (a clarifying question, a re-confirmation) that a real call sometimes
    has and sometimes doesn't — never turns that change the resolution
    outcome, only ones that change how long it took to get there.

    insertions: list of (probability, insert_before_index, [(speaker, text), ...])
    Indices refer to positions in the ORIGINAL turns list. Applied highest
    index first so earlier insertions don't shift later insertion points.
    """
    result = list(turns)
    for prob, idx, extra in sorted(insertions, key=lambda x: -x[1]):
        if rng.random() < prob:
            result[idx:idx] = extra
    return result


# ---------------------------------------------------------------------------
# Transcript data model
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSpec:
    transcript_id: str
    archetype: str
    sub_type: str
    market: str
    resolution: str
    sentiment_arc: str
    agent_consistency: str
    intent: str
    turns: list

    def render(self) -> str:
        header = textwrap.dedent(f"""\
            TRANSCRIPT_ID: {self.transcript_id}
            ARCHETYPE: {self.archetype}
            SUB_TYPE: {self.sub_type}
            MARKET: {self.market}
            RESOLUTION: {self.resolution}
            SENTIMENT_ARC: {self.sentiment_arc}
            AGENT_CONSISTENCY: {self.agent_consistency}
            INTENT: {self.intent}
            TURNS: {len(self.turns)}
            ---
            """)
        body = "\n".join(f"{speaker}: {text}" for speaker, text in self.turns)
        return header + body + "\n"


# ---------------------------------------------------------------------------
# Sub-type registry (metadata for all 12; market/institution now resolved
# per-transcript at generation time via SUBTYPE_INSTITUTION_POOL)
# ---------------------------------------------------------------------------

SUBTYPE_REGISTRY = {
    "billing_error_resolved": dict(
        archetype="RESOLVED_SUPPORT", resolution="resolved",
        arc="frustrated -> relieved", agent_consistency="neutral_positive_throughout",
        intent="billing_error",
    ),
    "failed_transaction_resolved": dict(
        archetype="RESOLVED_SUPPORT", resolution="resolved",
        arc="anxious -> reassured", agent_consistency="neutral_positive_throughout",
        intent="failed_transaction",
    ),
    "account_access_resolved": dict(
        archetype="RESOLVED_SUPPORT", resolution="resolved",
        arc="frustrated -> relieved", agent_consistency="neutral_positive_throughout",
        intent="account_access",
    ),
    "service_quality_complaint": dict(
        archetype="UNRESOLVED_COMPLAINT", resolution="unresolved",
        arc="calm -> frustrated", agent_consistency="neutral_throughout",
        intent="service_complaint",
    ),
    "repeated_issue_unresolved": dict(
        archetype="UNRESOLVED_COMPLAINT", resolution="unresolved",
        arc="already_frustrated -> resigned", agent_consistency="neutral_throughout",
        intent="service_complaint",
    ),
    "policy_dispute_unresolved": dict(
        archetype="UNRESOLVED_COMPLAINT", resolution="unresolved",
        arc="polite -> hostile", agent_consistency="neutral_throughout",
        intent="policy_dispute",
    ),
    "standard_kyc_pass": dict(
        archetype="KYC_VERIFICATION", resolution="resolved",
        arc="neutral -> satisfied", agent_consistency="neutral_positive_throughout",
        intent="kyc_verification",
    ),
    "kyc_document_issue": dict(
        archetype="KYC_VERIFICATION", resolution="unresolved",
        arc="polite -> mildly_frustrated", agent_consistency="neutral_throughout",
        intent="kyc_incomplete",
    ),
    "kyc_fraud_flag": dict(
        archetype="KYC_VERIFICATION", resolution="unresolved",
        arc="neutral -> confused_defensive", agent_consistency="neutral_throughout",
        intent="kyc_escalation",
    ),
    "payment_reminder_compliant": dict(
        archetype="LOAN_RECOVERY", resolution="resolved",
        arc="slightly_defensive -> cooperative", agent_consistency="firm_non_aggressive",
        intent="loan_reminder",
    ),
    "hardship_negotiation": dict(
        archetype="LOAN_RECOVERY", resolution="resolved",
        arc="distressed -> cautiously_relieved", agent_consistency="empathetic_firm",
        intent="hardship_plan",
    ),
    "loan_recovery_escalation": dict(
        archetype="LOAN_RECOVERY", resolution="unresolved",
        arc="hostile_throughout", agent_consistency="neutral_professional",
        intent="loan_escalation",
    ),
}


# ---------------------------------------------------------------------------
# WORKED TEMPLATE 1: billing_error_resolved — now across NG / KE / GULF
# ---------------------------------------------------------------------------

def _billing_error_resolved(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["billing_error_resolved"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    cur = profile["currency"]
    amount = non_round_amount(rng, 400, 2200) if market == "GULF" else non_round_amount(rng, 4500, 22000)
    date = random_date(rng)
    ref = ref_number(rng, inst_abbr, "BILL")
    open_word = rng.choice(profile["open_phrases"])
    phone = gen_phone(market, rng)
    last4 = gen_last4(rng)

    scenario = idx % 3
    if scenario == 0:
        issue = f"I have been charged twice for the same transaction — {cur}{amount:,} taken out on {date}, and nobody has done anything about it"
        cause = "a network retry on our end double-submitted the transaction"
    elif scenario == 1:
        issue = f"I was charged {cur}{amount:,} twice for what should have been one payment on {date}, and I've been trying to sort this out for two days now"
        cause = "our merchant gateway logged two separate authorizations for a single tap"
    else:
        issue = f"you took {cur}{amount:,} from my account on {date} for a subscription I already cancelled last month, and this is not small money to me"
        cause = "the cancellation didn't sync to the billing cycle before it ran"

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"{open_word}, {agent}, good afternoon. My name is {name}. {issue[0].upper()}{issue[1:]}."),
        ("AGENT", f"Good afternoon, {name}. I'm sorry to hear that. I'll look into this for you right away. Can you please confirm your registered phone number and the last four digits of your account number so I can pull up your account?"),
        ("CUSTOMER", f"Yes. {phone}. Account ends in {last4}."),
        ("AGENT", f"Thank you, {name}. I can see your account now. Can you tell me approximately when this charge occurred and what the transaction was for?"),
        ("CUSTOMER", f"It was on {date}. I need this resolved today, please."),
        ("AGENT", f"I completely understand your frustration, {name}, and I sincerely apologize for the delay. Let me pull up your transaction history for {date} now."),
        ("CUSTOMER", "Please. Because this is not small money."),
        ("AGENT", f"I can see it here — {cause}. That is clearly a duplicate charge, and I am initiating a reversal for the second transaction now."),
        ("CUSTOMER", f"Okay. So the {cur}{amount:,} will come back?"),
        ("AGENT", f"Yes, exactly. The reversal for {cur}{amount:,} has been initiated. Your reference number is {ref}. The funds will reflect in your account within 24 to 48 hours, and you will also receive an SMS confirmation shortly."),
        ("CUSTOMER", f"Okay, {ref}. Good. So I should see it by tomorrow at the latest?"),
        ("AGENT", "That is correct, by tomorrow at the latest. Is there anything else I can help you with today?"),
        ("CUSTOMER", f"No, that is all. Thank you, {agent}. At least someone has finally sorted it out."),
        ("AGENT", f"You are welcome, {name}. I am glad we could resolve this for you. Have a lovely evening."),
        ("CUSTOMER", "You too. Bye."),
        ("AGENT", "Goodbye."),
    ]
    # Optional exchanges — vary turn count without touching resolution logic.
    insertions = [
        (0.35, 9, [
            ("CUSTOMER", "How did this even happen, is my account safe?"),
            ("AGENT", "Great question — this was purely a processing error, not a security issue. Your account and card details are completely safe."),
        ]),
        (0.3, 13, [
            ("CUSTOMER", "And this won't happen again, right?"),
            ("AGENT", "I've flagged this on our end so our systems team can look into the retry issue — it shouldn't recur, but do reach out immediately if it does."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_billing_error_resolved(count: int, start_id: int, seed: int = 1) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["billing_error_resolved"]
    for i, market in enumerate(markets):
        turns, inst = _billing_error_resolved(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="billing_error_resolved",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# WORKED TEMPLATE 2: policy_dispute_unresolved — now across NG / KE / GULF
# ---------------------------------------------------------------------------

def _policy_dispute_unresolved(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, domain = rng.choice(SUBTYPE_INSTITUTION_POOL["policy_dispute_unresolved"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    cur = profile["currency"]
    ref = ref_number(rng, inst_abbr, "DSP")
    anger, anger_punct = rng.choice(list(profile["anger_phrases"].items()))
    open_word = rng.choice(profile["open_phrases"])
    phone = gen_phone(market, rng)

    if market == "GULF":
        id_label, id_value = "Emirates ID number", gen_emirates_id(rng)
        regulator = "the Central Bank"
        fee_low, fee_high, prin_low, prin_high, grown_low, grown_high = 500, 2000, 15000, 35000, 4000, 12000
    elif market == "KE":
        id_label, id_value = "national ID number", gen_ke_national_id(rng)
        regulator = "the Central Bank of Kenya, and the Competition Authority"
        fee_low, fee_high, prin_low, prin_high, grown_low, grown_high = 3000, 9000, 100000, 350000, 40000, 200000
    else:
        id_label, id_value = "BVN", gen_bvn(rng)
        regulator = "the CBN"
        fee_low, fee_high, prin_low, prin_high, grown_low, grown_high = 3000, 9000, 100000, 350000, 40000, 200000

    other_lenders = {
        "NG": ["Carbon", "Palmcredit", "Branch"],
        "KE": ["Tala", "Zenka", "Branch"],
        "GULF": ["Emirates NBD", "ADCB", "RAKBANK"],
    }[market]
    competitors = ", ".join([o for o in other_lenders if o != inst][:2])

    scenario = idx % 3
    if domain == "lender":
        if scenario == 0:
            principal = non_round_amount(rng, prin_low, prin_high)
            grown = principal + non_round_amount(rng, grown_low, grown_high)
            months = rng.randint(3, 7)
            monthly_paid = non_round_amount(rng, (principal // months) - 500, (principal // months) + 500)
            issue = (f"I took a loan of {cur}{principal:,} and I have been repaying consistently every month for "
                      f"{months} months, not once missing a payment. But my outstanding balance is now showing "
                      f"{cur}{grown:,} — that is more than what I originally borrowed. How is that possible if I have "
                      f"been paying every month?")
            explain = (f"the current balance reflects the principal, accrued interest, and a variable processing "
                        f"fee that was applied to your account under our variable charges clause")
            pushback = (f"Nobody explained a variable charges clause to me when I took this loan. I was told my "
                         f"monthly repayment would be {cur}{monthly_paid:,} and that is what I have been paying. "
                         f"Nobody said anything about extra fees being added mid-loan without my knowledge.")
            math_check = (f"I have been paying {cur}{monthly_paid:,} a month for {months} months, that is "
                           f"{cur}{monthly_paid * months:,} paid already. Even with interest, the math does not "
                           f"produce a balance of {cur}{grown:,}. Something is wrong here.")
        elif scenario == 1:
            fee = non_round_amount(rng, fee_low, fee_high)
            issue = (f"you charged me a {cur}{fee:,} 'late processing fee' even though my payment went through "
                      f"on the exact due date — I have the transaction receipt. Nobody explained this fee when "
                      f"I took the loan, and I want to understand why it was applied.")
            explain = "the fee was applied automatically once the system flagged the payment as received after the cutoff time on the due date"
            pushback = ("There was no cutoff time mentioned anywhere when I took this loan. If there's a specific "
                         "time of day the payment needs to clear by, that should have been disclosed clearly, "
                         "not buried somewhere I'd never find it.")
            math_check = None
        else:
            principal = non_round_amount(rng, prin_low, prin_high)
            issue = (f"my loan of {cur}{principal:,} was refinanced without my signature, and now the new terms "
                      f"have a higher rate that I never agreed to. I only found out when I checked my statement "
                      f"this week.")
            explain = "the refinancing was applied under a clause that allows automatic restructuring after a missed payment date, which is standard across our loan agreements"
            pushback = ("I never missed a payment, and even if a clause like that exists, changing my interest "
                         "rate without telling me first is not something I would ever have agreed to.")
            math_check = None
    else:
        if scenario == 0:
            fee = non_round_amount(rng, fee_low, fee_high)
            issue = (f"you charged me a {cur}{fee:,} monthly account maintenance fee even though I've kept the "
                      f"minimum balance every single month since I opened this account. This was never explained "
                      f"to me when I opened it, and I want to understand why I'm being charged.")
            explain = "this is part of our standard account terms and conditions, under the fees and charges section"
            pushback = "Nobody read that to me, nobody explained it in a way I could understand. This is not fair."
            math_check = None
        elif scenario == 1:
            fee = non_round_amount(rng, fee_low, fee_high)
            issue = (f"I sent money abroad last week and you deducted a {cur}{fee:,} foreign currency conversion "
                      f"fee that was never disclosed before I confirmed the transfer. I would never have sent it "
                      f"through this app if I'd known there was a hidden charge like that.")
            explain = "the conversion fee is applied automatically on all cross-currency transfers and is outlined in our standard terms"
            pushback = "That fee was never shown to me on the confirmation screen before I sent the money. That's the whole problem."
            math_check = None
        else:
            fee = non_round_amount(rng, fee_low, fee_high)
            issue = (f"my account was dormant for a few months and now you've charged me a {cur}{fee:,} "
                      f"reactivation fee that nobody told me about when I tried to use my card again. I just "
                      f"wanted to use my own money.")
            explain = "dormancy reactivation fees are standard across the industry and are outlined in our account terms"
            pushback = "I've never heard of another bank charging a fee just to let someone use their own account again."
            math_check = None

    account_noun = "loan account" if domain == "lender" else "account"

    turns = [
        ("AGENT", f"Good day, thank you for calling {inst}, my name is {agent}, how can I assist you?"),
        ("CUSTOMER", f"Good day, {agent}. My name is {name}. {issue[0].upper()}{issue[1:]}"),
        ("AGENT", f"Good day, {name}. Of course, I'm happy to help. Can you please confirm your registered phone number and {id_label} so I can pull up your {account_noun}?"),
        ("CUSTOMER", f"{open_word}. {phone}. {id_value}."),
        ("AGENT", f"Thank you, {name}. I have your account here. Let me look into this for you."),
        ("CUSTOMER", "Please do, because this doesn't make sense to me."),
        ("AGENT", f"Thank you for waiting. I can see the charge — {explain}."),
        ("CUSTOMER", f"{anger}{anger_punct} {pushback}"),
        ("AGENT", "I completely understand your frustration, and I appreciate you raising this. Unfortunately I'm not able to override or reverse this — it's within our standard operating policy."),
        ("CUSTOMER", f"If this practice is so legitimate, why does it feel like {competitors if competitors else 'other providers'} don't do this? Only you people go about it this way." if domain == "lender" else "So your answer is that it's in the terms and that's final. You're not even willing to acknowledge that burying something like this where nobody reads it, and then charging me for it, is wrong."),
        ("AGENT", "I hear you, and those are valid concerns. I want to assure you this is fully disclosed in the agreement you accepted, and it's within our standard operating policy."),
    ]

    if math_check:
        turns += [
            ("CUSTOMER", math_check),
            ("AGENT", f"I understand the concern, {name}. What I can do is log a formal dispute so our review team can give you a full breakdown of every charge and the exact clause it falls under."),
        ]
    else:
        turns += [
            ("CUSTOMER", "So you are telling me there is nothing you can do? Who can do something then?"),
            ("AGENT", f"I can log a formal dispute for you, reference number {ref}, and escalate it to our terms review team. I can't promise a specific outcome or timeline beyond that, though."),
        ]

    turns += [
        ("CUSTOMER", "This is exactly the problem, always 'we will escalate', nothing ever changes."),
        ("AGENT", f"I understand the frustration, and I've logged everything accurately on my end. Your dispute reference is {ref}, and the review team will follow up within five to seven business days."),
        ("CUSTOMER", f"Five to seven business days, and in the meantime this keeps sitting on my account. {anger}{anger_punct} I know my rights — {regulator} exists for exactly this kind of thing, and I will be reporting this."),
        ("AGENT", "I hear you, and I respect your right to pursue that. I want to assure you we operate within all regulatory requirements, and our team will provide full transparency on your account."),
        ("CUSTOMER", f"{anger}{anger_punct} I don't have time for this. I want it on record that I am disputing this fully."),
        ("AGENT", f"That has been noted on your account, {name}. Your reference number is {ref} — is there anything else I can help with today?"),
        ("CUSTOMER", "No. There is nothing else you can do for me. Goodbye."),
        ("AGENT", "Thank you for —"),
    ]

    insertions = [
        (0.3, len(turns) - 4, [
            ("CUSTOMER", "This cannot be legal, surely there is someone I can report this to beyond just your team?"),
            ("AGENT", "You're welcome to escalate externally if you'd like — I just can't make that determination or override the charge from this line."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_policy_dispute_unresolved(count: int, start_id: int, seed: int = 2) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["policy_dispute_unresolved"]
    for i, market in enumerate(markets):
        turns, inst = _policy_dispute_unresolved(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="policy_dispute_unresolved",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 3: failed_transaction_resolved (banking, all markets)
# Two real-world paths, both ending resolved/anxious->reassured:
#   Path A: transaction genuinely failed, nothing moved, balance untouched
#   Path B: money left the account, showed failed, sitting in a clearing
#           account, reversed back, customer acknowledges
# ---------------------------------------------------------------------------

def _failed_transaction_resolved(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["failed_transaction_resolved"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    cur = profile["currency"]
    amount = non_round_amount(rng, 300, 2500) if market == "GULF" else non_round_amount(rng, 5000, 40000)
    ref = ref_number(rng, inst_abbr, "TXN")
    open_word = rng.choice(profile["open_phrases"])
    phone = gen_phone(market, rng)
    last4 = gen_last4(rng)
    recipient = ["my landlord", "my brother", "a supplier"][idx % 3]
    txn_kind = f"a transfer of {cur}{amount:,} to {recipient}"
    path = "A" if idx % 2 == 0 else "B"
    time_ago = rng.choice(["last night", "about an hour ago", "earlier this morning"])

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"{agent}, good afternoon. My name is {name}. I need urgent help. I made {txn_kind} {time_ago}. The transaction showed as failed on my end, but I'm not sure if the money left my account or if it's just gone. I've been worried about this."),
        ("AGENT", f"Good afternoon, {name}. I understand this is stressful and I will look into this for you right now. Can you please confirm your registered phone number and the last four digits of your account number so I can pull up your account?"),
        ("CUSTOMER", f"{open_word}. {phone}. Account ends in {last4}."),
        ("AGENT", f"Thank you, {name}. I have your account open now. One moment while I check the transaction."),
        ("CUSTOMER", "Okay, please."),
    ]

    if path == "A":
        turns += [
            ("AGENT", "I can confirm the transaction genuinely did not go through on our side — no funds left your account at all. Your balance is untouched."),
            ("CUSTOMER", "Are you sure? Because the app showed the amount was deducted for a moment."),
            ("AGENT", "That can happen when a transaction is pending confirmation before it fails — it briefly shows as a hold, but it was released automatically since the transfer didn't complete. Your available balance reflects that already."),
            ("CUSTOMER", "Okay, let me check my balance now... yes, I can see it's correct. That's a relief."),
            ("AGENT", "I'm glad that's cleared up for you. Is there anything else I can help you with today?"),
        ]
    else:
        turns += [
            ("AGENT", f"I can see the {cur}{amount:,} did leave your account, but it never reached the recipient — it's currently sitting in our clearing account because the transaction failed to settle on the receiving end."),
            ("CUSTOMER", "Okay so the money didn't just disappear? It is somewhere?"),
            ("AGENT", "Correct — it is not lost. Because the transaction failed at the settlement stage, the funds are sitting in a clearing account and will be reversed back to your account. I am initiating that reversal now."),
            ("CUSTOMER", "So where is my money right now, and how do I get it back?"),
            ("AGENT", f"I'm reversing it back to your account right now. This is an internal reversal, so it should reflect within 24 to 48 hours. Your reference number is {ref}, and you'll receive an SMS confirmation."),
            ("CUSTOMER", f"Okay, {ref}. So by tomorrow or the day after at the latest?"),
            ("AGENT", "That is correct. Once the reversal clears, you can retry the transfer and it will go through cleanly. Is there anything else I can help you with today?"),
        ]

    turns += [
        ("CUSTOMER", "No, that's all. Thank you so much."),
        ("AGENT", f"You're very welcome, {name}, I'm glad we could give you clarity on this. Have a good evening."),
        ("CUSTOMER", "You too. Bye."),
        ("AGENT", "Goodbye, and take care."),
    ]

    insertions = [
        (0.3, 5, [
            ("CUSTOMER", "This is really stressful, I needed that money today."),
            ("AGENT", "I completely understand, and I'll make sure this is resolved before we end the call."),
        ]),
        (0.3, len(turns) - 4, [
            ("CUSTOMER", "How do I make sure this doesn't happen again next time?"),
            ("AGENT", "This was a one-off settlement issue on the receiving end, not something on your account — there's nothing you need to change on your side."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_failed_transaction_resolved(count: int, start_id: int, seed: int = 3) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["failed_transaction_resolved"]
    for i, market in enumerate(markets):
        turns, inst = _failed_transaction_resolved(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="failed_transaction_resolved",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 4: account_access_resolved (banking, all markets)
# Suspicious-activity freeze -> secondary verification -> access restored,
# customer confirms live in the call. Arc: frustrated -> relieved.
# ---------------------------------------------------------------------------

def _account_access_resolved(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["account_access_resolved"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    ref = ref_number(rng, inst_abbr, "ACC")
    open_word = rng.choice(profile["open_phrases"])
    phone = gen_phone(market, rng)
    cur = profile["currency"]

    if market == "GULF":
        id_label, id_value = "Emirates ID number", gen_emirates_id(rng)
    elif market == "KE":
        id_label, id_value = "national ID number", gen_ke_national_id(rng)
    else:
        id_label, id_value = "BVN", gen_bvn(rng)

    scenario = idx % 3
    if scenario == 0:
        trigger = "a login attempt from a device we didn't recognize"
        context = "That was me — I was using a colleague's phone because mine had no charge."
    elif scenario == 1:
        trigger = "a login from a location that didn't match your usual pattern"
        context = "That was me, I'm travelling for work this week, that's probably why."
    else:
        trigger = "several incorrect PIN attempts in a short window"
        context = "That's my fault, I mistyped my PIN a few times this morning, sorry about that."

    beneficiary = rng.choice(profile["customer_names"])
    transfer_amount = non_round_amount(rng, 300, 3000) if market == "GULF" else non_round_amount(rng, 4000, 25000)
    branch = rng.choice(["the main branch", "the downtown branch", "the branch you originally opened the account at"])

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"{agent}, good afternoon. My name is {name}. I've been trying to log into my account since this morning and it is completely locked. I have an urgent payment to make and I cannot access anything."),
        ("AGENT", f"Good afternoon, {name}. I'm sorry to hear this and I understand how urgent this is. Before I can access your details, I need to verify your identity. Can you please confirm your registered phone number and your {id_label}?"),
        ("CUSTOMER", f"{open_word}. {phone}. {id_value}."),
        ("AGENT", f"Thank you, {name}. One moment while I pull up your account. I can see that your access was frozen this morning as a security precaution, triggered by {trigger}."),
        ("CUSTOMER", context),
        ("AGENT", "Understood — that explains it. This kind of thing triggers an automatic freeze as a security measure. I'll need to complete a secondary verification before I can restore your access. I'm going to ask you three security questions — please answer exactly as you registered."),
        ("CUSTOMER", "Okay, go ahead."),
        ("AGENT", "First — what is the name of the primary beneficiary on your account?"),
        ("CUSTOMER", f"{beneficiary}."),
        ("AGENT", "Correct. Second — what was the amount of your last inward transfer to this account?"),
        ("CUSTOMER", f"{cur}{transfer_amount:,}."),
        ("AGENT", f"Correct. Third — can you confirm {branch}?"),
        ("CUSTOMER", "Yes, that's correct."),
        ("AGENT", f"Thank you, {name}. All three verified. I'm restoring your access now — this will take approximately sixty seconds. Your reference number for this is {ref}."),
        ("CUSTOMER", "Okay, thank you."),
        ("AGENT", "Your access has been restored. Can you try logging in now while we're still on the call, just to confirm everything is working?"),
        ("CUSTOMER", "One moment... yes! I'm in, I can see my dashboard now. Thank you so much."),
        ("AGENT", "Perfect. For your security, the earlier session has been terminated and won't be able to access your account again. Is there anything else I can help you with today?"),
        ("CUSTOMER", "No, that's all, thank you."),
        ("AGENT", f"You're very welcome, {name}. I'm glad we could resolve this for you quickly."),
        ("CUSTOMER", "Thank you. Goodbye."),
        ("AGENT", "Goodbye, take care."),
    ]

    insertions = [
        (0.3, 7, [
            ("CUSTOMER", "Is my account actually safe though? Should I be worried?"),
            ("AGENT", "There's no indication of unauthorized access — this was a precautionary freeze, not a confirmed breach. Your funds were never at risk."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_account_access_resolved(count: int, start_id: int, seed: int = 4) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["account_access_resolved"]
    for i, market in enumerate(markets):
        turns, inst = _account_access_resolved(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="account_access_resolved",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs

# ---------------------------------------------------------------------------
# TEMPLATE 5: standard_kyc_pass (banking, all markets)
# All documents verified in-call, restriction lifted / access unlocked.
# Arc: neutral throughout, slight satisfaction at end.
# ---------------------------------------------------------------------------

def _standard_kyc_pass(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["standard_kyc_pass"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    ref = ref_number(rng, inst_abbr, "KYC")
    open_word = rng.choice(profile["open_phrases"])
    phone = gen_phone(market, rng)
    dob = gen_dob(rng)

    scenario = idx % 3
    if scenario == 0:
        months = rng.randint(6, 14)
        reason = f"your account has been inactive for {months} months, so we need to refresh a few details before fully reactivating it"
    elif scenario == 1:
        reason = "this is a routine periodic verification we're required to run on all accounts under current regulations"
    else:
        reason = "you're upgrading to a higher account tier, which requires an additional round of verification"

    old_area = rng.choice(["the old part of town", "my previous neighborhood", "across town"])
    new_area = rng.choice(["a new development nearby", "the other side of the city", "a place closer to work"])
    bill_doc = {"NG": "a recent utility bill", "KE": "a Safaricom statement", "GULF": "a recent DEWA bill"}[market]

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"{open_word}, {agent}. My name is {name}. I got a notification asking me to complete some verification on my account, so I'm calling to sort that out."),
        ("AGENT", f"Thank you for calling in about that, {name}. I can see {reason}. Before I proceed, can you confirm your registered phone number and date of birth?"),
        ("CUSTOMER", f"Yes. {phone}. Date of birth is {dob}."),
    ]

    if market == "NG":
        bvn = gen_bvn(rng)
        nin = gen_nin(rng)
        turns += [
            ("AGENT", "Thank you, that matches what we have on file. I'll also need to re-verify your BVN. Can you confirm that for me?"),
            ("CUSTOMER", f"{bvn}."),
            ("AGENT", "Thank you, confirmed. I also need your NIN — can you provide that?"),
            ("CUSTOMER", f"{nin}."),
            ("AGENT", "Thank you, confirmed as well. Now I need to verify your current residential address — can you confirm the address we have on file, or give me your current one if it's changed?"),
        ]
    elif market == "KE":
        nat_id = gen_ke_national_id(rng)
        turns += [
            ("AGENT", "Thank you, that matches what we have on file. I'll also need to re-confirm your national ID number — can you provide that?"),
            ("CUSTOMER", f"{nat_id}."),
            ("AGENT", "Thank you, confirmed. Now I need to verify your current residential address — can you confirm the address we have on file, or give me your current one if it's changed?"),
        ]
    else:
        emirates_id = gen_emirates_id(rng)
        turns += [
            ("AGENT", "Thank you, that matches what we have on file. I'll also need to re-confirm your Emirates ID number — can you provide that?"),
            ("CUSTOMER", f"{emirates_id}."),
            ("AGENT", "Thank you, confirmed. Now I need to verify your current residential address — can you confirm the address we have on file, or give me your current one if it's changed?"),
        ]

    turns += [
        ("CUSTOMER", f"The address on file is from {old_area}. I moved to {new_area} a few months ago — I'll need to update that."),
        ("AGENT", f"No problem, I'll update that for you now. I'll need {bill_doc} dated within the last three months showing your name and the new address. Do you have something like that available?"),
        ("CUSTOMER", "I have one, let me upload it now on the app... okay, it's uploaded."),
        ("AGENT", f"Perfect, I can see it. Everything else checks out. I can lift the restriction partially right now — you'll be able to log in and receive incoming transfers immediately. Outgoing transfers will be fully enabled once the address document clears review, typically within 24 hours. Your reference number is {ref}."),
        ("CUSTOMER", "That works. Can you confirm I'm able to log in now, while we're still on the call?"),
        ("AGENT", "Please go ahead and try."),
        ("CUSTOMER", "One moment... yes, I'm in, I can see my dashboard now. Thank you."),
        ("AGENT", "That's great to hear. Is there anything else I can help you with today?"),
        ("CUSTOMER", "No, that's all, thanks."),
        ("AGENT", f"You're welcome, thank you for banking with {inst}, take care."),
    ]

    insertions = [
        (0.3, len(turns) - 10, [
            ("CUSTOMER", "Why do you need all this again, didn't I already do this when I opened the account?"),
            ("AGENT", "Totally fair question — regulations require us to periodically refresh this information, it's not specific to you."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_standard_kyc_pass(count: int, start_id: int, seed: int = 5) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["standard_kyc_pass"]
    for i, market in enumerate(markets):
        turns, inst = _standard_kyc_pass(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="standard_kyc_pass",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 6: kyc_document_issue (banking, all markets)
# A submitted document is flagged; agent explains what's missing and next
# steps, call ends WITHOUT verification complete. Arc: polite -> mildly
# frustrated.
# ---------------------------------------------------------------------------

def _kyc_document_issue(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["kyc_document_issue"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    open_word = rng.choice(profile["open_phrases"])
    review_hours = rng.choice([24, 48, 72])
    phone = gen_phone(market, rng)
    dob = gen_dob(rng)
    ref = ref_number(rng, inst_abbr, "KYC")

    scenario = idx % 3
    if scenario == 0:
        years_old = rng.randint(6, 9)
        issue = f"the ID on file is from {years_old} years ago, and our system flags anything issued more than 5 years back for a compliance review — this isn't a government expiry issue, it's an internal rule"
        alt = "a more recent proof of identity, like a utility statement paired with your tax or national ID number"
    elif scenario == 1:
        issue = "the document you uploaded came through blurry, and our system couldn't read the details clearly enough to verify it"
        alt = "a clearer photo, ideally taken in good lighting with all four corners of the document visible"
    else:
        issue = "the name on the document you uploaded doesn't quite match the name on your account — even a small spelling difference can trigger this"
        alt = "a document that matches your account name exactly, or a supporting document explaining the name difference"

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"{open_word}, {agent}. My name is {name}. I uploaded my documents for verification but my account is still showing as restricted, so I wanted to check what's going on. This is a valid document, I don't understand what the problem is."),
        ("AGENT", f"I'm sorry to hear you've had trouble with that, {name}. Let me look into it. Can you please confirm your registered phone number and date of birth?"),
        ("CUSTOMER", f"Yes. {phone}. Date of birth is {dob}."),
        ("AGENT", f"Thank you, {name}. I have your account here. I can see that your document was flagged — {issue}."),
        ("CUSTOMER", "Nobody told me that when I uploaded it, the app just said 'processing'. That's a bit confusing — what am I supposed to do now?"),
        ("AGENT", f"I completely understand the confusion, and I apologize that wasn't communicated more clearly on the app. What I'd recommend is submitting {alt}."),
        ("CUSTOMER", "Okay, and how long will that take once I submit it?"),
        ("AGENT", f"Once it's submitted, review typically takes up to {review_hours} hours. I can't complete the verification on this call, unfortunately."),
        ("CUSTOMER", "So I still can't use my account after all this? I've already been waiting for two days."),
        ("AGENT", f"Not until the new document clears review, no — I'm sorry for the inconvenience. Your case reference number is {ref}. Please quote that when you upload the new document so our team picks it up quickly."),
        ("CUSTOMER", f"Okay, {ref}. I'll upload it today. I just hope this actually gets resolved because I have things I need to do on my account."),
        ("AGENT", f"I understand, {name}, and I hope so too. I'll leave detailed notes on your account so the review team has full context. Is there anything else I can help with today?"),
        ("CUSTOMER", "No, that's all."),
        ("AGENT", f"Thank you for banking with {inst}, take care."),
    ]
    insertions = [
        (0.3, len(turns) - 6, [
            ("CUSTOMER", "This app error message really should have told me this from the start."),
            ("AGENT", "That's fair feedback, and I'll pass it along to our product team."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_kyc_document_issue(count: int, start_id: int, seed: int = 6) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["kyc_document_issue"]
    for i, market in enumerate(markets):
        turns, inst = _kyc_document_issue(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="kyc_document_issue",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 7: kyc_fraud_flag (banking, all markets)
# Automated verification flags the account; agent follows protocol, never
# confirms or denies the specific reason, escalates internally. Arc:
# neutral -> confused/defensive.
# ---------------------------------------------------------------------------

def _kyc_fraud_flag(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["kyc_fraud_flag"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    open_word = rng.choice(profile["open_phrases"])
    days = rng.choice([3, 4, 5])
    phone = gen_phone(market, rng)
    ref = ref_number(rng, inst_abbr, "KYC")

    if market == "GULF":
        id_label, id_value = "Emirates ID number", gen_emirates_id(rng)
    elif market == "KE":
        id_label, id_value = "national ID number", gen_ke_national_id(rng)
    else:
        id_label, id_value = "BVN", gen_bvn(rng)

    context = [
        "a new account you opened recently",
        "your account during a routine automated screening",
        "a recent large transaction on your account",
    ][idx % 3]

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"{open_word}, {agent}. My name is {name}. My account has been flagged for some kind of review and I don't understand why — I haven't done anything unusual, and I submitted everything that was asked of me."),
        ("AGENT", f"I understand the confusion, {name}. Let me look into that for you. Can you please confirm your registered phone number and {id_label}?"),
        ("CUSTOMER", f"{phone}. {id_value}."),
        ("AGENT", f"Thank you, {name}. One moment while I pull up your application. I can confirm there's an extended review in progress on {context}. I apologize for the delay."),
        ("CUSTOMER", "Extended review — what does that mean exactly? What else could possibly be needed?"),
        ("AGENT", "I understand your frustration. An extended review means your application requires additional verification steps beyond the standard automated process. This can happen for a number of reasons and doesn't necessarily indicate a problem with your documents."),
        ("CUSTOMER", "So you're telling me you don't even know why, or you just can't tell me?"),
        ("AGENT", "I genuinely don't have visibility into the specific trigger from this screen — I can only confirm that a review is in progress and share the expected timeline. That's determined by an automated system on the compliance side."),
        ("CUSTOMER", "This doesn't make sense. Is someone accusing me of something?"),
        ("AGENT", "This isn't an accusation — extended reviews are a standard part of our verification process and don't necessarily mean anything was found. I know that's not a very satisfying answer, and I'm sorry for that."),
        ("CUSTOMER", "Okay... so what happens now?"),
        ("AGENT", f"The review typically takes {days} business days. I've escalated it internally with a note that you called in today — reference number {ref} — and you'll be notified directly once it's complete."),
        ("CUSTOMER", "Nobody contacted me to tell me this was happening in the first place, I had to call myself to find out. That's not good service."),
        ("AGENT", "You're right, and I sincerely apologize for that. You should have received a notification when your application moved to extended review. I'll log that as a service gap and flag it to the team."),
        ("CUSTOMER", "Alright. I still don't really understand, but okay."),
        ("AGENT", "I understand, and I appreciate your patience. Is there anything else I can help with today?"),
        ("CUSTOMER", "No, that's all."),
        ("AGENT", f"Thank you for banking with {inst}, take care."),
    ]
    insertions = [
        (0.3, len(turns) - 6, [
            ("CUSTOMER", "Can I at least speak to someone who does know the reason?"),
            ("AGENT", "I can escalate your request to speak with the review team directly, though I can't guarantee they'll be able to disclose more than I have."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_kyc_fraud_flag(count: int, start_id: int, seed: int = 7) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["kyc_fraud_flag"]
    for i, market in enumerate(markets):
        turns, inst = _kyc_fraud_flag(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="kyc_fraud_flag",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 8: service_quality_complaint (COMPLAINT_POOL: banks + ride-hail +
# food-delivery, all markets). Resolution: none — agent acknowledges, offers
# only apology or "we'll look into it," no committed action. Arc: calm ->
# frustrated.
# ---------------------------------------------------------------------------

def _service_quality_complaint(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, domain = rng.choice(SUBTYPE_INSTITUTION_POOL["service_quality_complaint"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    ref = ref_number(rng, inst_abbr, "CMP")
    anger, anger_punct = rng.choice(list(profile["anger_phrases"].items()))
    open_word = rng.choice(profile["open_phrases"])
    driver_name = rng.choice(AGENT_NAMES)
    date = random_date(rng)
    phone = gen_phone(market, rng)

    if domain == "ride":
        issue = (f"I booked a ride today from my home to my office and had a very bad experience with the driver. "
                  f"He showed up 20 minutes late, was rude when I mentioned it, and then tried to push me to cancel "
                  f"and rebook at a higher fare. I have heard about drivers behaving this way before, but today I "
                  f"experienced it myself, and this should not be happening.")
        what_kind, ref_label = "trip", "trip reference number"
    elif domain == "food":
        issue = (f"my order today arrived over an hour late and completely cold, and when I messaged the rider to "
                  f"ask where it was, he was extremely rude to me. This isn't the kind of service I expect to pay for.")
        what_kind, ref_label = "order", "order reference number"
    else:
        issue = (f"I was kept waiting over 40 minutes at the branch today, and when I finally got to the counter, "
                  f"the staff member was dismissive and unhelpful about a simple request. I don't think this is "
                  f"acceptable service from an institution I've banked with for years.")
        what_kind, ref_label = "branch visit", "registered phone number"

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I assist you today?"),
        ("CUSTOMER", f"Good day, {agent}. My name is {name}. I'm reaching out because {issue}"),
        ("AGENT", f"I am sorry to hear about your experience, {name}. I understand how you feel and I sincerely apologize for the inconvenience. To look into this, could you please provide your {ref_label}?"),
    ]
    if domain in ("ride", "food"):
        trip_ref = ref_number(rng, inst_abbr, "TRP")
        turns += [
            ("CUSTOMER", f"Yes, one moment. It's {trip_ref}."),
            ("AGENT", f"Thank you. And can you confirm the {'driver' if domain == 'ride' else 'rider'}'s name as it appeared in the app, and the approximate time of the {what_kind}?"),
            ("CUSTOMER", f"The name showed as {driver_name}. It was earlier today, {date}."),
        ]
    else:
        turns += [
            ("CUSTOMER", f"{phone}."),
            ("AGENT", f"Thank you, {name}. Let me pull up your account and the branch visit log for {date}."),
            ("CUSTOMER", "Okay, please."),
        ]

    turns += [
        ("AGENT", f"Thank you for providing that. I can see the record here. I want to assure you that {inst} takes service quality very seriously. I will escalate this complaint to our review team, and the appropriate action will be taken."),
        ("CUSTOMER", "What does appropriate action mean exactly? Because this is not the first time I'm hearing about this kind of thing, and I want to know something will actually be done, not just an apology."),
        ("AGENT", "I completely understand your concern. Our team will thoroughly review the complaint and take the necessary steps in line with policy. I'm not able to confirm the specific outcome at this time, but your feedback will not be ignored."),
        ("CUSTOMER", f"{anger}{anger_punct} So you cannot tell me if anything will actually happen. I am just supposed to wait and hope?"),
        ("AGENT", f"I sincerely apologize, {name}. What I can do is log your complaint formally — your reference number is {ref} — and ensure it's passed to the relevant team. You may receive a follow-up within five to seven business days."),
        ("CUSTOMER", "Five to seven business days. And in the meantime nothing changes for anyone else who deals with this."),
        ("AGENT", f"I hear your frustration, {name}, and I want to assure you we take these matters seriously. The complaint has been logged under reference {ref}."),
        ("CUSTOMER", f"{agent}, with all due respect, you've said sorry and 'we will look into it' more than once now, and nothing concrete has been said. I am not satisfied with this response at all."),
        ("AGENT", "I sincerely apologize that I haven't been able to provide the specific resolution you were hoping for today. This is the process we have in place, and I assure you your complaint is being taken seriously. Is there anything else I can help with?"),
        ("CUSTOMER", "No. There's nothing else. I just hope something actually gets done."),
        ("AGENT", f"Thank you for bringing this to our attention, {name}. We value your feedback and apologize again for your experience. Have a good day."),
    ]

    insertions = [
        (0.4, len(turns) - 6, [
            ("CUSTOMER", "Can you at least escalate this to someone who CAN offer compensation?"),
            ("AGENT", "I can flag it for a supervisor's attention, though I can't guarantee they'll be able to offer more than what I've already outlined."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_service_quality_complaint(count: int, start_id: int, seed: int = 8) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["service_quality_complaint"]
    for i, market in enumerate(markets):
        turns, inst = _service_quality_complaint(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="service_quality_complaint",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 9: repeated_issue_unresolved (COMPLAINT_POOL, all markets).
# Nth call about the same unresolved issue; agent can't access previous
# escalation outcomes, resolves nothing again. Arc: already frustrated ->
# resigned.
# ---------------------------------------------------------------------------

def _repeated_issue_unresolved(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, domain = rng.choice(SUBTYPE_INSTITUTION_POOL["repeated_issue_unresolved"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    anger, anger_punct = rng.choice(list(profile["anger_phrases"].items()))
    open_word = rng.choice(profile["open_phrases"])
    call_number = rng.choice([4, 5, 6])
    phone = gen_phone(market, rng)

    if domain == "food":
        item = rng.choice(["a large combo meal", "a family order", "a full house wrap combo"])
        issue = (f"every single time I place an order, something is missing. Tonight I ordered {item}, and part of "
                  f"it was missing again. This is the {call_number}th time something like this has happened.")
        ref_prefix, ref_purpose = inst_abbr, "ORD"
    elif domain == "ride":
        issue = (f"I keep being overcharged on my fare compared to what the app quoted me before the trip. This "
                  f"is the {call_number}th time I've called about the exact same problem.")
        ref_prefix, ref_purpose = inst_abbr, "TRP"
    else:
        issue = (f"a transaction that keeps failing to reflect correctly in my statement. I have called about "
                  f"this exact same issue {call_number - 1} times before, and it is still not fixed.")
        ref_prefix, ref_purpose = inst_abbr, "TXN"

    order_ref = ref_number(rng, ref_prefix, ref_purpose)

    turns = [
        ("AGENT", f"Good afternoon, thank you for calling {inst}, my name is {agent}, how may I help you today?"),
        ("CUSTOMER", f"I will be honest with you, {agent} — I am already tired before this call even started. My name is {name}. {issue[0].upper()}{issue[1:]}"),
        ("AGENT", f"I am very sorry to hear that, {name}. I completely understand your frustration. Can I please have your registered phone number so I can pull up your account?"),
        ("CUSTOMER", f"{open_word}. {phone}."),
        ("AGENT", f"Thank you. I can see your account here. Can you also provide the reference number for your most recent affected order or transaction?"),
        ("CUSTOMER", f"It is {order_ref}."),
        ("AGENT", f"Thank you, {name}. I can see previous notes referencing this issue, though I don't have visibility into the outcome of the earlier escalations from this screen."),
        ("CUSTOMER", "You people never actually fix it though, that's the problem."),
        ("AGENT", "I understand how frustrating that is. What I can do is log this again and escalate it with a note referencing your previous calls, so it's flagged as a recurring issue."),
        ("CUSTOMER", "That's exactly what the last agent told me. Word for word. Can you tell me what actually happened after my last complaint? Because nothing changed."),
        ("AGENT", f"I sincerely apologize, {name}. I am not able to access the outcome of previous escalations from my end, but I want to assure you your complaint is being taken seriously. I'll flag this as a high priority given how many times you've called."),
        ("CUSTOMER", f"{anger}{anger_punct} So every time I call, it starts from zero again? I have no way of knowing if anyone even looked at my previous complaints."),
        ("AGENT", f"I completely understand, {name}, and I sincerely apologize for the experience you've had. I have logged this and marked it urgent — our team will follow up within 24 to 48 hours."),
        ("CUSTOMER", f"Twenty-four to forty-eight hours, and then what? Another apology and nothing changes? {anger}{anger_punct} I don't even know why I keep calling at this point."),
        ("AGENT", f"I hear your frustration, {name}, and I genuinely apologize that we haven't met your expectations. Is there anything else I can assist you with today?"),
        ("CUSTOMER", "No. There's nothing else. Just make sure someone actually reads this complaint this time."),
        ("AGENT", "I assure you it will be reviewed. Thank you for your patience, and for being a valued customer."),
    ]

    insertions = [
        (0.5, 9, [
            ("CUSTOMER", "The first time I called, they said it would be fixed within 48 hours. The second time, someone told me it was already escalated. Nothing has changed."),
            ("AGENT", "I'm sorry you've had to repeat that history — I can see some of those notes, though not the full detail of what was promised each time."),
        ]),
        (0.4, 11, [
            ("CUSTOMER", "Is there anyone above you who can actually look into this properly?"),
            ("AGENT", "I can escalate this to a senior review, though I can't guarantee a faster outcome than the previous escalations."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_repeated_issue_unresolved(count: int, start_id: int, seed: int = 9) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["repeated_issue_unresolved"]
    for i, market in enumerate(markets):
        turns, inst = _repeated_issue_unresolved(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="repeated_issue_unresolved",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 10: payment_reminder_compliant (lending, all markets)
# Outbound call: agent confirms right person BEFORE identifying institution.
# Customer commits to a specific amount, date, and payment method.
# Arc: slightly defensive -> cooperative.
# ---------------------------------------------------------------------------

def _payment_reminder_compliant(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["payment_reminder_compliant"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    cur = profile["currency"]
    days_overdue = rng.randint(2, 7)
    principal = non_round_amount(rng, 3000, 15000) if market == "GULF" else non_round_amount(rng, 12000, 40000)
    interest = non_round_amount(rng, 200, 900) if market == "GULF" else non_round_amount(rng, 500, 3000)
    total = principal + interest
    reason = ["a delayed salary payment", "an unexpected family expense", "a delay from my employer's payroll"][idx % 3]
    day_of_week = ["Wednesday", "Thursday", "Friday", "Monday"][idx % 4]

    turns = [
        ("AGENT", f"Good afternoon, am I speaking with {name}?"),
        ("CUSTOMER", f"Yes, this is {name}. Who's calling?"),
        ("AGENT", f"Thank you for confirming. My name is {agent}, I'm calling from {inst} regarding your account."),
        ("CUSTOMER", "Okay, what's this about?"),
        ("AGENT", f"I'm calling because your payment of {cur}{principal:,} is now {days_overdue} days overdue. I wanted to check in and see how we can help get this sorted."),
        ("CUSTOMER", f"Yeah, sorry about that, it's been {reason} on my end, it wasn't intentional."),
        ("AGENT", "I understand, these things happen. When do you think you'd be able to make the payment?"),
        ("CUSTOMER", f"I can pay by {day_of_week} — I should have the funds by then."),
        ("AGENT", f"That works. With the accrued interest, the total due would be {cur}{total:,}. Can you confirm you're able to pay that full amount by {day_of_week}?"),
        ("CUSTOMER", f"Yes, that's fine, I'll pay {cur}{total:,} by {day_of_week}."),
        ("AGENT", f"Great, and will you be paying through the {inst} app, or another method?"),
        ("CUSTOMER", f"I'll do it through the app."),
        ("AGENT", f"Perfect. I've noted that on your account — reference number {ref_number(rng, inst_abbr, 'REM')}. Thank you for your cooperation, and please reach out if anything changes before then."),
        ("CUSTOMER", "Okay, will do, thanks."),
        ("AGENT", "Thank you for your time, have a good day."),
    ]
    insertions = [
        (0.3, 5, [
            ("CUSTOMER", "I don't appreciate being called about this like I'm avoiding it, I was going to pay anyway."),
            ("AGENT", "I completely understand, this is just a standard courtesy call, not an accusation — I appreciate you taking the time."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_payment_reminder_compliant(count: int, start_id: int, seed: int = 10) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["payment_reminder_compliant"]
    for i, market in enumerate(markets):
        turns, inst = _payment_reminder_compliant(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="payment_reminder_compliant",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 11: hardship_negotiation (lending, all markets)
# Outbound call: distressed customer (job loss / reduced income), agent
# negotiates a restructured plan — reduced installment, timeline, late fees
# frozen. Resolution = agreed plan, not money collected today.
# Arc: distressed -> cautiously relieved.
# ---------------------------------------------------------------------------

def _hardship_negotiation(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["hardship_negotiation"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    cur = profile["currency"]
    payment_method = profile["payment_method"]
    phone = gen_phone(market, rng)

    if market == "GULF":
        loan = non_round_amount(rng, 8000, 20000)
        partial_paid = non_round_amount(rng, 1000, 3000)
    else:
        loan = non_round_amount(rng, 20000, 50000)
        partial_paid = non_round_amount(rng, 3000, 8000)
    balance = loan - partial_paid
    months = rng.choice([6, 9, 12])
    # committed is derived FROM balance/months (not drawn independently) so the
    # stated plan total actually settles the stated balance — catching a real
    # bug where independent random draws produced a plan covering less than
    # half the outstanding balance while the agent claimed it would "settle" it.
    base_installment = balance // months
    committed = non_round_amount(rng, base_installment, base_installment + (200 if market == "GULF" else 1500))
    total_plan = committed * months
    day_of_month = rng.choice([1, 5, 10, 15])

    hardship_month = rng.choice(["April", "May", "June", "July"])
    partial_month = rng.choice(["May", "June", "July", "August"])
    hardship_reason = [
        f"I lost my job in {hardship_month} — the company I worked for did a round of retrenchments",
        f"my hours were cut significantly at work starting {hardship_month}, and my income dropped by more than half",
        f"I had a medical emergency in the family in {hardship_month} that used up everything I had saved",
    ][idx % 3]

    turns = [
        ("AGENT", f"Good afternoon. Am I speaking with {name}?"),
        ("CUSTOMER", "Yes, this is."),
        ("AGENT", f"Good afternoon. My name is {agent}, and I'm calling from {inst} loan recovery support regarding your loan account. Is this a good time to speak?"),
        ("CUSTOMER", f"Yes. I have been expecting a call. I know I have missed payments, and I want to explain what happened."),
        ("AGENT", "Of course, please go ahead. I'm here to listen and to see how we can find a way forward together."),
        ("CUSTOMER", (f"{hardship_reason}. I took this loan when I was still doing fine financially, and I've been "
                       f"trying to get back on my feet since. I paid what I could — {cur}{partial_paid:,} in "
                       f"{partial_month} — but I cannot pay the full outstanding amount right now. I'm not running "
                       f"away from this, I just genuinely do not have the money.")),
        ("AGENT", f"{name}, thank you for being honest with me about your situation. That's a serious hardship, and I can hear you're doing your best. The {cur}{partial_paid:,} payment has been noted on your account, and I want you to know that counts. Can I pull up your account so we can look at the full picture together?"),
        ("CUSTOMER", f"Sure. My phone number is {phone}."),
        ("AGENT", f"Thank you. I have your account here. Your original loan was {cur}{loan:,}. After your {partial_month} payment, the outstanding balance including accrued interest and late fees is currently {cur}{balance:,}. I want to be transparent with you about that number before we discuss options."),
        ("CUSTOMER", f"{cur}{balance:,}. That is almost what I borrowed. The interest and fees have eaten my payment completely — that's discouraging to hear."),
        ("AGENT", "I understand, and that frustration is completely valid. What I want to do today is find a restructured plan that works for your situation right now — not what it was before. Can I ask, what can you realistically commit to paying per month at the moment? Even a small amount — I want to work with what's actually possible for you."),
        ("CUSTOMER", f"Honestly, I can manage {cur}{committed:,} per month right now. Maybe more once things improve, but I can't promise that yet."),
        ("AGENT", f"I appreciate your honesty. {cur}{committed:,} a month is workable. Here's what I can offer — I'll restructure your loan over {months} months at {cur}{committed:,} per month. That covers {cur}{total_plan:,} over the full period, which settles the outstanding balance including the accrued interest. I'll also request that late fees stop accruing from today, provided you maintain the monthly payments consistently."),
        ("CUSTOMER", f"So {cur}{committed:,} every month for {months} months and the late fees stop — and that clears the full debt?"),
        ("AGENT", "Correct. As long as payments are made consistently each month by the agreed date, the late fees will be frozen from today and the plan will fully settle your account. If you miss a payment, the late fees would resume, so consistency matters — I want to be clear about that so there are no surprises."),
        ("CUSTOMER", f"I understand. And I pay through {payment_method} as usual?"),
        ("AGENT", f"Yes, same process you've been using. Each payment will reflect on your account within a few hours, and you'll get a confirmation. I'd suggest setting a reminder for the same date each month so it becomes routine. Which date works best for you?"),
        ("CUSTOMER", f"The {day_of_month}th of each month — that's usually when my income comes in."),
        ("AGENT", f"Perfect, the {day_of_month}th of each month starting next month. I'm documenting this agreement on your account now — you'll receive a confirmation message to your registered number shortly. Please keep that for your records."),
        ("CUSTOMER", f"Okay. {agent}, I was really dreading this call before I picked up. I thought you were going to threaten me or something. This isn't what I expected at all."),
        ("AGENT", "I'm glad we could find a way forward together. These situations happen, and the most important thing is working through them rather than avoiding them. You were honest with me, and that made everything easier. Is there anything else I can help with today?"),
        ("CUSTOMER", f"No. Just {cur}{committed:,} on the {day_of_month}th — I will not miss it."),
        ("AGENT", "I believe you. Good luck, and take care."),
        ("CUSTOMER", "Thank you. Goodbye."),
        ("AGENT", "Goodbye, take care."),
    ]
    insertions = [
        (0.3, len(turns) - 8, [
            ("CUSTOMER", "Are you sure the late fees will actually stop? I've been burned by promises like that before."),
            ("AGENT", "Completely understandable to ask — I'm noting it on your account right now, not just saying it. You'll see it reflected immediately."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_hardship_negotiation(count: int, start_id: int, seed: int = 11) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["hardship_negotiation"]
    for i, market in enumerate(markets):
        turns, inst = _hardship_negotiation(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="hardship_negotiation",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# TEMPLATE 12: loan_recovery_escalation (lending, all markets)
# Outbound call: customer disputes a fee/charge, refuses all payment options
# (full / partial / undisputed portion), escalated to legal/collections
# without agreement. Arc: hostile throughout.
# ---------------------------------------------------------------------------

def _loan_recovery_escalation(rng: random.Random, idx: int, market: str) -> tuple:
    profile = MARKET_PROFILES[market]
    inst, inst_abbr, _domain = rng.choice(SUBTYPE_INSTITUTION_POOL["loan_recovery_escalation"][market])
    name = rng.choice(profile["customer_names"])
    agent = rng.choice(AGENT_NAMES)
    cur = profile["currency"]
    anger, anger_punct = rng.choice(list(profile["anger_phrases"].items()))
    farewell, farewell_punct = rng.choice(list(profile["anger_phrases"].items()))
    ref = ref_number(rng, inst_abbr, "ESC")

    regulator = {"NG": "the CBN", "KE": "the Central Bank of Kenya", "GULF": "the Central Bank"}[market]

    if market == "GULF":
        arrears = non_round_amount(rng, 6000, 15000)
        disputed_fee = non_round_amount(rng, 500, 2000)
    else:
        arrears = non_round_amount(rng, 30000, 90000)
        disputed_fee = non_round_amount(rng, 3000, 12000)
    months_overdue = rng.randint(3, 5)
    undisputed = arrears - disputed_fee
    weeks_calling = rng.choice([3, 4, 5, 6])

    turns = [
        ("AGENT", f"Good afternoon. Am I speaking with {name}?"),
        ("CUSTOMER", "Who is asking?"),
        ("AGENT", f"Good afternoon. My name is {agent}, and I'm calling from {inst} loan recovery support. Am I speaking with {name}?"),
        ("CUSTOMER", "Yes. What do you want?"),
        ("AGENT", f"I'm reaching out regarding your loan account, which is currently {months_overdue} months in arrears, with an outstanding balance of {cur}{arrears:,} including accrued fees. I'm calling to discuss how we can resolve this together. Is this a good time?"),
        ("CUSTOMER", f"{months_overdue} months. You people have been calling me every week for {weeks_calling} weeks now. What is there to discuss that hasn't already been discussed? I told the last person who called — I am disputing this amount, and I'm not paying something I did not agree to."),
        ("AGENT", "I understand you've raised a dispute previously. I want to make sure we address that properly today — can you help me understand specifically which charges you're disputing?"),
        ("CUSTOMER", f"The restructuring fee, {cur}{disputed_fee:,}. Nobody told me about that fee when I took this loan. It appeared on my statement without any notification. If you want to add fees to my account, you need my consent — I did not give consent, and I'm not paying it. Simple."),
        ("AGENT", f"I hear you, and I want to look into that specifically. I can see the {cur}{disputed_fee:,} fee was applied following a payment arrangement set up earlier — this is outlined under the restructuring clause in your agreement. I can arrange for a full breakdown to be sent to your registered email if that would help clarify it."),
        ("CUSTOMER", f"I've read the agreement. That clause does not give you the right to add fees without written notification beforehand. You sent me nothing before adding that charge — nothing. And now you're calling me every week demanding money. {anger}{anger_punct} This is harassment. I have a good mind to report this to {regulator}."),
        ("AGENT", f"I want to assure you {inst} operates fully within regulatory guidelines, and all communication has been within permitted limits. You are absolutely within your right to file a complaint with {regulator} — that's your right as a customer, and I wouldn't discourage you from doing so. What I'd like to focus on today is whether there's a way to resolve the outstanding balance while your dispute is reviewed. The remaining balance outside the disputed fee is {cur}{undisputed:,} — would you consider a payment arrangement for that portion?"),
        ("CUSTOMER", "No. I'm not paying anything until the dispute is resolved. If I pay now, you'll say I accepted the charges. I know how this works. I'm not touching this account until you acknowledge the fee was applied incorrectly."),
        ("AGENT", f"I understand your position. I want to be transparent about what happens next so you're fully informed. Your account has been in arrears for {months_overdue} months. If no payment arrangement is made today, I'm required to refer this account to our legal recovery team — at that point it moves outside customer support and into a formal legal process, which may include a formal demand notice and further action under civil law. I'm not saying this to pressure you, I'm telling you so you can make an informed decision."),
        ("CUSTOMER", "Legal team. Fine, let them come. I've documented every call, every charge, every notification you failed to send me. If you want to take this to court, I'll see you there."),
        ("AGENT", f"I hear you, and I respect that you've documented your case. Before I escalate, I want to make one final attempt — is there any arrangement, even a partial payment against the undisputed portion, that you'd consider today to prevent this moving to legal recovery?"),
        ("CUSTOMER", f"No. I'm not paying {profile['currency_slang']} until this whole thing is sorted out properly."),
        ("AGENT", "I respect that. Since we're not able to reach an agreement today, I'll need to escalate this account to our recovery and legal team for further action."),
        ("CUSTOMER", "Do whatever you want. I'm not scared of that."),
        ("AGENT", f"I've logged the dispute details on the account so the legal team has full context before proceeding. Your account reference for this interaction is {ref}, and you'll receive formal written communication within five business days. Is there anything else you'd like on record?"),
        ("CUSTOMER", f"{farewell}{farewell_punct} Goodbye."),
        ("AGENT", f"Goodbye, {name}."),
    ]
    insertions = [
        (0.3, len(turns) - 6, [
            ("CUSTOMER", "This is exactly the kind of thing that should be illegal, adding fees nobody agreed to."),
            ("AGENT", "I understand the frustration — that's exactly why I'm logging it for the dispute review rather than just pushing for payment."),
        ]),
    ]
    turns = apply_variation(rng, turns, insertions)
    return turns, inst


def generate_loan_recovery_escalation(count: int, start_id: int, seed: int = 12) -> list:
    rng = random.Random(seed)
    markets = distribute_markets(count)
    specs = []
    meta = SUBTYPE_REGISTRY["loan_recovery_escalation"]
    for i, market in enumerate(markets):
        turns, inst = _loan_recovery_escalation(rng, i, market)
        specs.append(TranscriptSpec(
            transcript_id=f"GS_{start_id + i:03d}",
            archetype=meta["archetype"], sub_type="loan_recovery_escalation",
            market=f"{MARKET_PROFILES[market]['label']}/{inst}", resolution=meta["resolution"],
            sentiment_arc=meta["arc"], agent_consistency=meta["agent_consistency"],
            intent=meta["intent"], turns=turns,
        ))
    return specs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

IMPLEMENTED_GENERATORS = {
    "billing_error_resolved": generate_billing_error_resolved,
    "failed_transaction_resolved": generate_failed_transaction_resolved,
    "account_access_resolved": generate_account_access_resolved,
    "service_quality_complaint": generate_service_quality_complaint,
    "repeated_issue_unresolved": generate_repeated_issue_unresolved,
    "policy_dispute_unresolved": generate_policy_dispute_unresolved,
    "standard_kyc_pass": generate_standard_kyc_pass,
    "kyc_document_issue": generate_kyc_document_issue,
    "kyc_fraud_flag": generate_kyc_fraud_flag,
    "payment_reminder_compliant": generate_payment_reminder_compliant,
    "hardship_negotiation": generate_hardship_negotiation,
    "loan_recovery_escalation": generate_loan_recovery_escalation,
}

def run(sub_type: str, count: int, start_id: int, write: bool = True):
    if sub_type not in IMPLEMENTED_GENERATORS:
        raise NotImplementedError(
            f"'{sub_type}' has no template yet. Registered sub-types with metadata "
            f"but no generator: {sorted(set(SUBTYPE_REGISTRY) - set(IMPLEMENTED_GENERATORS))}"
        )
    specs = IMPLEMENTED_GENERATORS[sub_type](count, start_id)
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            out_path = OUTPUT_DIR / f"{spec.transcript_id}.txt"
            out_path.write_text(spec.render(), encoding="utf-8")
            print(f"wrote {out_path}  [{spec.market}]")
    else:
        for spec in specs:
            print(spec.render())
            print("=" * 60)
    return specs


if __name__ == "__main__":
    # 12-13 per sub-type to reach ~150 total alongside the 13 hand-authored seeds.
    run("billing_error_resolved", count=13, start_id=14, write=True)
    run("failed_transaction_resolved", count=13, start_id=27, write=True)
    run("account_access_resolved", count=13, start_id=40, write=True)
    run("service_quality_complaint", count=13, start_id=53, write=True)
    run("repeated_issue_unresolved", count=13, start_id=66, write=True)
    run("policy_dispute_unresolved", count=11, start_id=79, write=True)  # already has 2 seeds
    run("standard_kyc_pass", count=13, start_id=90, write=True)
    run("kyc_document_issue", count=13, start_id=103, write=True)
    run("kyc_fraud_flag", count=13, start_id=116, write=True)
    run("payment_reminder_compliant", count=13, start_id=129, write=True)
    run("hardship_negotiation", count=13, start_id=142, write=True)
    run("loan_recovery_escalation", count=13, start_id=155, write=True)