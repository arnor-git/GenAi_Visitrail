
import os
import re
import json
import time
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------------
# Config — tune these for your cohort
# ---------------------------------------------------------------------------

DIAGNOSTIC_TERMS = [
    "adhd", "autism", "autistic", "asd", "add", "attention deficit",
    "disorder", "diagnos", "dyslexi", "dyspraxi", "cognitive impairment",
    "intellectual disability", "developmental delay",
]

MIN_TRIALS_FOR_METRIC = 5          # below this, a trial-based metric (hit rate, RT) is too noisy
MIN_KINEMATIC_SAMPLING_HZ = 20     # below this, velocity-derived metrics (fixation %, gaze
                                    # coverage) can't be trusted — IVT needs near-continuous
                                    # samples to tell a real fixation from a gap between points

# Gemini's free tier, reached through its OpenAI-compatible endpoint.
# gemini-2.5-flash was retired for new API keys (as of your error above); Google's
# current stable Flash model is gemini-3.6-flash. If this one is ever retired too,
# check https://ai.google.dev/gemini-api/docs/models for the current name —
# nothing else in this file needs to change.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-3.6-flash"

# ⚠️  HARDCODED KEY — REMOVE BEFORE COMMITTING OR PUSHING THIS FILE ANYWHERE. ⚠️
# GitHub's secret scanning will find this in seconds if this file is ever pushed,
# public repo or private. The GEMINI_API_KEY environment variable, if set, always
# takes priority over this — set it instead when the repo leaves your machine.
GEMINI_API_KEY_HARDCODED = ""


# ---------------------------------------------------------------------------
# Stage 1 — Evidence pack: the ONLY thing that reaches the model
# ---------------------------------------------------------------------------

def build_evidence_pack(file_name, df, performance_metrics, spatial_metrics,
                         response_df, performance_df):
    duration_seconds = (df['Timestamp'].max() - df['Timestamp'].min()) / 1000.0
    sampling_rate = len(df) / duration_seconds if duration_seconds > 0 else 0
    fixation_rate = (df['movement_type'] == 'fixation').mean() * 100

    n_targets = int((performance_df['is_target'] == True).sum()) if len(performance_df) else 0
    rts = (performance_df.loc[performance_df['click_type'] == 'Correct', 'reaction_time'].dropna()
           if len(performance_df) else pd.Series(dtype=float))

    return {
        "session_id": file_name,
        "n_samples": int(len(df)),
        "duration_seconds": round(duration_seconds, 1),
        "sampling_rate_hz": round(sampling_rate, 1),
        "target_hit_rate": {
            "value": round(performance_metrics.get('target_hit_rate', 0), 1),
            "n_targets": n_targets,
            "unit": "%",
        },
        "false_alarm_rate": {
            "value": round(performance_metrics.get('false_alarm_rate', 0), 1),
            "unit": "%",
        },
        "avg_target_rt_ms": {
            "value": round(float(performance_metrics.get('avg_target_rt', 0)), 0),
            "n": int(len(rts)),
            "std_ms": round(float(rts.std()), 0) if len(rts) > 1 else None,
        },
        "fixation_rate_pct": {
            "value": round(fixation_rate, 1),
            "velocity_threshold_px_s": 721,
            "validated_on": "not yet population-validated for this cohort",
        },
        "screen_utilization_pct": {
            "value": round(spatial_metrics.get('screen_utilization', 0), 1),
        },
    }


def strip_identifiers_for_model(evidence):
    """The payload that actually leaves the machine. session_id (which may be a
    re-linkable pseudonym like a student ID, even on anonymized data) is kept
    locally for the audit log and the on-screen evidence panel, but is never
    included in what gets sent to the API."""
    return {k: v for k, v in evidence.items() if k != "session_id"}


def tag_reliability(evidence):
    """Not every metric fails the same way at a low sampling rate. Target hit rate and
    reaction time come from matching two event timestamps — they're fine even with
    sparse logging, as long as there are enough trials. Fixation rate and screen
    coverage come from a velocity calculation between consecutive gaze points — at a
    low sampling rate that math can't tell a real fixation from a large gap between
    samples, no matter how many trials there were. This tags each metric with its own
    "reliable" flag instead of using one blunt gate for the whole session, so a report
    can still be generated from what IS trustworthy while being explicit about what
    isn't — rather than silently guessing, or refusing everything because of one
    unrelated measurement.

    Note: false_alarm_rate isn't tagged here — the evidence pack doesn't currently carry
    a distractor count, so there's no principled n to check it against. Treated as
    reliable by default; worth revisiting if distractor counts get threaded through.
    """
    evidence = json.loads(json.dumps(evidence))  # cheap deep copy
    sampling_ok = evidence.get("sampling_rate_hz", 0) >= MIN_KINEMATIC_SAMPLING_HZ

    if "target_hit_rate" in evidence:
        evidence["target_hit_rate"]["reliable"] = (
            evidence["target_hit_rate"].get("n_targets", 0) >= MIN_TRIALS_FOR_METRIC
        )
    if "avg_target_rt_ms" in evidence:
        evidence["avg_target_rt_ms"]["reliable"] = (
            evidence["avg_target_rt_ms"].get("n", 0) >= MIN_TRIALS_FOR_METRIC
        )
    for kinematic_key in ("fixation_rate_pct", "screen_utilization_pct"):
        if kinematic_key in evidence:
            evidence[kinematic_key]["reliable"] = sampling_ok
            if not sampling_ok:
                evidence[kinematic_key]["unreliable_reason"] = (
                    f"sampling rate ({evidence.get('sampling_rate_hz')} Hz) is far below "
                    f"the ~{MIN_KINEMATIC_SAMPLING_HZ} Hz needed to classify eye movement "
                    f"from position samples this sparse"
                )
    return evidence


# ---------------------------------------------------------------------------
# Stage 2 — Refusal gate: don't generate on data too thin/noisy to interpret
# ---------------------------------------------------------------------------

def check_data_quality(evidence):
    """Hard refusal gate — blocks generation only when there isn't enough trial data
    to say anything trustworthy at all. Low gaze-sampling-rate is handled separately
    by tag_reliability(): it excludes specific kinematic metrics from generation
    rather than blocking the whole report, since hit rate and reaction time don't
    need a high gaze sampling rate to be valid."""
    failed = []
    if evidence["target_hit_rate"]["n_targets"] < MIN_TRIALS_FOR_METRIC:
        failed.append(f"Too few target trials ({evidence['target_hit_rate']['n_targets']} "
                       f"< {MIN_TRIALS_FOR_METRIC})")
    if evidence["avg_target_rt_ms"]["n"] < MIN_TRIALS_FOR_METRIC:
        failed.append(f"Too few valid reaction times ({evidence['avg_target_rt_ms']['n']} "
                       f"< {MIN_TRIALS_FOR_METRIC})")
    return (len(failed) == 0), failed


# ---------------------------------------------------------------------------
# Stage 3 — Constrained generation
# ---------------------------------------------------------------------------

AUDIENCE_OPTIONS = ["Teacher", "Psychologist", "Family / Parent"]

AUDIENCE_GUIDANCE = {
    "Teacher": (
        "Write for a classroom teacher. Keep it practical — what was observed, and what "
        "might be worth trying in the classroom next. Standard terms like 'reaction time' "
        "or 'target' are fine, the teacher will recognize them from the dashboard."
    ),
    "Psychologist": (
        "Write for a psychologist or specialist. Standard vocabulary around attention, "
        "response time, and visual search is fine. Where a pattern is notable, note that "
        "a fuller formal assessment would be needed to interpret it — never state a "
        "diagnosis yourself, even a tentative one."
    ),
    "Family / Parent": (
        "Write for a parent with no clinical or technical background. Use short, warm, "
        "everyday sentences. Do not use the words 'fixation', 'saccade', 'target hit "
        "rate', 'reaction time (ms)', or similar jargon — describe what happened in "
        "plain terms instead (e.g. 'how quickly they responded' rather than 'reaction "
        "time'). Avoid anything that sounds clinical, alarming, or like a verdict on "
        "the child."
    ),
}

PROMPT_TEMPLATE = """You are describing one child's eye-tracking session. {audience_guidance}

You are given ONLY the computed metrics below. Do not invent, estimate, or round-trip any
number that is not present in this evidence.

Evidence:
{evidence_json}

Rules:
- Every claim must list the evidence keys it is based on, using the exact key names above.
- Only state a NUMBER for a metric whose "reliable" field is true. For a metric marked \
"reliable": false, you may say in plain terms that it couldn't be measured reliably this \
session (use its "unreliable_reason" if present) — never state its value.
- Describe observed behaviour only. NEVER name or imply a specific diagnosis or clinical \
condition (e.g. ADHD, autism spectrum, a learning disorder). If a pattern is notable, say a \
specialist could look into it — do not say what they would find.
- Mark "certainty" as "tentative" whenever the relevant n is below 15, "moderate" otherwise. \
Never claim high certainty from a single session.
- Output ONLY valid JSON, no prose, no markdown fences, matching exactly:
{{"claims": [{{"text": str, "evidence_keys": [str], "certainty": "tentative"|"moderate"}}]}}
"""


def call_llm(evidence, audience, n_samples=3, temperature=0.4, max_retries=3, retry_base_delay=2.0):
    """`evidence` should already be stripped of identifiers (strip_identifiers_for_model)
    and reliability-tagged (tag_reliability) — this is the exact payload that leaves the
    machine, and the exact payload the model is told it may or may not cite numbers from.

    Free-tier Gemini occasionally returns a transient 503 (UNAVAILABLE) or 429 (rate
    limit) under load. Those are retried with exponential backoff (2s, 4s, 8s...) before
    giving up on that sample; anything else (bad key, bad model name) fails immediately
    rather than retrying a problem that won't go away on its own."""
    if OpenAI is None:
        raise RuntimeError("Run: pip install openai")
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY_HARDCODED
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError("Set GEMINI_API_KEY (env var, preferred) or fill in "
                            "GEMINI_API_KEY_HARDCODED above.")

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    prompt = PROMPT_TEMPLATE.format(
        audience_guidance=AUDIENCE_GUIDANCE[audience],
        evidence_json=json.dumps(evidence, indent=2),
    )

    outputs = []
    for _ in range(n_samples):
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                outputs.append(resp.choices[0].message.content or "")
                last_error = None
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                transient = any(s in msg for s in ("503", "429", "UNAVAILABLE",
                                                     "RESOURCE_EXHAUSTED", "rate limit"))
                if transient and attempt < max_retries - 1:
                    time.sleep(retry_base_delay * (2 ** attempt))
                    continue
                raise
        if last_error is not None:
            raise last_error
    return outputs


# ---------------------------------------------------------------------------
# Stage 4 — Verification: the actual "trustworthy" part
# ---------------------------------------------------------------------------

def _clean_json_text(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def _flatten_evidence_numbers(evidence, reliable_only=True):
    """Every number the model is allowed to have used — not just each metric's headline
    "value", but every numeric field in the evidence pack (n_targets, n, std_ms,
    duration_seconds, sampling_rate_hz...) plus any number written into the pack's own
    descriptive text (e.g. the "20 Hz" threshold named in unreliable_reason). That text
    is ours, not the model's, so any number in it is safe to treat as grounded — this is
    what lets the model correctly explain *why* a metric is unreliable without every
    number in that explanation being flagged as invented.

    When reliable_only is True (the default, used by verify_claims), a metric's "value"
    field specifically is skipped if that metric is marked "reliable": false — so the
    headline number itself stays ungrounded even though the metric's other fields
    (n_targets, unreliable_reason, etc.) are still fair game to cite."""
    numbers = set()

    def walk(node):
        if isinstance(node, dict):
            blocked = reliable_only and node.get("reliable") is False
            for k, v in node.items():
                if blocked and k == "value":
                    continue
                walk(v)
        elif isinstance(node, bool):
            pass  # bool is a subclass of int in Python — exclude explicitly
        elif isinstance(node, (int, float)):
            numbers.add(float(node))
        elif isinstance(node, str):
            for m in re.findall(r"-?\d+\.?\d*", node):
                numbers.add(float(m))

    for val in evidence.values():
        walk(val)
    return numbers


def verify_claims(raw_json_text, evidence, rel_tolerance=0.01, abs_tolerance=0.5):
    """Reject any claim citing an unknown key or containing a number not grounded in
    the evidence. Citing an unreliable metric's key is fine on its own — that's how a
    claim explains a metric couldn't be measured — what's actually blocked is stating
    that metric's numeric value, which _flatten_evidence_numbers already excludes for
    anything marked "reliable": false.

    Tolerance is intentionally tight (0.5 absolute, 1% relative) — just enough to
    forgive the model rounding 93.8 to "94". A looser tolerance (an earlier version
    used 5%) can accidentally treat two different real numbers as a match: 89.2 (an
    excluded, unreliable value) sits well within 5% of 93.8 (a reliable one), so a
    claim stating the wrong number would have been waved through as "grounded"."""
    try:
        parsed = json.loads(_clean_json_text(raw_json_text))
    except Exception:
        return [], [{"text": raw_json_text[:200], "reason": "model output was not valid JSON"}]

    valid_keys = set(evidence.keys())
    evidence_numbers = _flatten_evidence_numbers(evidence, reliable_only=True)

    verified, rejected = [], []
    for claim in parsed.get("claims", []):
        text = claim.get("text", "")
        keys = claim.get("evidence_keys", [])
        reasons = []

        bad_keys = [k for k in keys if k not in valid_keys]
        if bad_keys:
            reasons.append(f"cites unknown evidence keys: {bad_keys}")

        found_numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]
        for num in found_numbers:
            grounded = any(
                abs(num - v) <= max(abs_tolerance, rel_tolerance * abs(v))
                for v in evidence_numbers
            )
            if not grounded:
                reasons.append(f"ungrounded number: {num}")

        lowered = text.lower()
        hit_terms = [t for t in DIAGNOSTIC_TERMS if t in lowered]
        if hit_terms:
            reasons.append(f"diagnostic language used: {hit_terms}")

        if reasons:
            rejected.append({"text": text, "reason": "; ".join(reasons)})
        else:
            claim["_verified"] = True
            verified.append(claim)

    return verified, rejected


# ---------------------------------------------------------------------------
# Stage 5-7 — Streamlit UI: quality gate, self-consistency, human review, audit log
# ---------------------------------------------------------------------------

def render_trustworthy_recommendations_tab(file_name, result):
    df = result['df']
    performance_metrics = result['performance_metrics']
    spatial_metrics = result['spatial_metrics']
    response_df = result['response_df']
    performance_df = result['performance_df']

    evidence = build_evidence_pack(file_name, df, performance_metrics, spatial_metrics,
                                    response_df, performance_df)
    ok, failures = check_data_quality(evidence)

    st.markdown(f"#### AI-Assisted Interpretation — {file_name}")
    st.caption("Experimental. Every claim below is checked against the metrics shown here "
               "before you see it, and nothing is exported until you approve it.")

    with st.expander("Evidence actually sent to the model (session_id stripped)"):
        st.json(strip_identifiers_for_model(evidence))
    with st.expander("Full evidence pack (kept locally only, shown for your reference)"):
        st.json(evidence)

    tagged_evidence = tag_reliability(strip_identifiers_for_model(evidence))
    unreliable = [(k, v.get("unreliable_reason", "no reason given"))
                  for k, v in tagged_evidence.items()
                  if isinstance(v, dict) and v.get("reliable") is False]
    if unreliable:
        st.info("Some measurements from this session aren't precise enough to report on:")
        for key, reason in unreliable:
            st.write(f"- **{key}**: {reason}")

    if not ok:
        st.warning("Data quality insufficient for AI interpretation — showing rule-based "
                    "summary only:")
        for f in failures:
            st.write(f"- {f}")
        return

    audience = st.radio("Who is this summary for?", AUDIENCE_OPTIONS,
                         key=f"audience_{file_name}", horizontal=True)

    gen_key = f"gen_clicked_{file_name}_{audience}"
    if st.button(f"Generate AI interpretation for {audience}", key=f"btn_{file_name}_{audience}"):
        st.session_state[gen_key] = True

    if not st.session_state.get(gen_key):
        return

    claims_key = f"claims_{file_name}_{audience}"
    rejected_key = f"rejected_{file_name}_{audience}"
    agree_key = f"agreement_{file_name}_{audience}"

    if claims_key not in st.session_state:
        with st.spinner("Generating and verifying..."):
            try:
                outputs = call_llm(tagged_evidence, audience, n_samples=3)
            except Exception as e:
                st.error(f"LLM call failed: {e}. Falling back to rule-based summary only.")
                st.session_state[gen_key] = False
                return

            all_verified, all_rejected = [], []
            for out in outputs:
                v, r = verify_claims(out, tagged_evidence)
                all_verified.extend(v)
                all_rejected.extend(r)

            texts = [c["text"] for c in all_verified]
            agreement = {t: texts.count(t) / max(1, len(outputs)) for t in set(texts)}

            st.session_state[claims_key] = all_verified
            st.session_state[rejected_key] = all_rejected
            st.session_state[agree_key] = agreement

    claims = st.session_state.get(claims_key, [])
    rejected = st.session_state.get(rejected_key, [])
    agreement = st.session_state.get(agree_key, {})

    if rejected:
        with st.expander(f"{len(rejected)} claim(s) rejected by automatic verification"):
            for r in rejected:
                st.write(f"❌ {r['text']}")
                st.caption(r["reason"])

    if not claims:
        st.info("No claims survived verification. Rule-based summary stands as-is.")
        return

    seen = set()
    for i, claim in enumerate(claims):
        if claim["text"] in seen:
            continue
        seen.add(claim["text"])
        conf = agreement.get(claim["text"], 0) * 100

        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(claim["text"])
            st.caption(f"Evidence: {', '.join(claim['evidence_keys'])} · "
                       f"Certainty: {claim['certainty']} · Model agreement: {conf:.0f}%")
        with col2:
            decision = st.radio("Review", ["Pending", "Approve", "Edit", "Reject"],
                                 key=f"dec_{file_name}_{audience}_{i}", label_visibility="collapsed")

        st.session_state.setdefault("audit_log", []).append({
            "timestamp": datetime.utcnow().isoformat(),
            "file_name": file_name,
            "audience": audience,
            "claim_text": claim["text"],
            "evidence_hash": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()).hexdigest()[:12],
            "decision": decision,
        })

    if st.session_state.get("audit_log"):
        with st.expander("Audit log (timestamp, audience, evidence hash, reviewer decision)"):
            st.json(st.session_state["audit_log"][-20:])