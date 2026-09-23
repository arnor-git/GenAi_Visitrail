# VisiTrail

Eye-tracking analysis dashboard for children's performance on visual attention "serious games," built with Streamlit. VisiTrail processes gaze and game-event logs into fixation/saccade classification, target/distractor hit-rate and reaction-time metrics, and spatial gaze-coverage statistics — then summarizes each session for teachers and psychologists, with an experimental, audience-specific AI-assisted interpretation layer.

This tool was originally developed for sessions involving children with neurodevelopmental differences (NDD). It is a **research and observation aid, not a diagnostic instrument** — see [Data Privacy & Ethics](#data-privacy--ethics) below.

## Features

- CSV ingestion for gaze position + game event logs (object appearance, disappearance, clicks)
- I-VT (velocity-threshold) fixation/saccade classification
- Target hit-rate, reaction time, and false-alarm-rate computation
- Spatial gaze-coverage and screen-utilization metrics
- Multi-session comparison across game levels
- Rule-based recommendations for teachers and psychologists
- **AI-Assisted Interpretation (experimental)** — see below

## Trustworthy GenAI design

The AI Interpretation tab does not just ask an LLM to summarize a session. It is built around a verification pipeline, because an unverified LLM output feeding into an educational or clinical-adjacent recommendation is a real failure mode, not a hypothetical one:

1. **Evidence pack** — only aggregated numeric metrics (hit rate, reaction time, fixation rate, etc.) are ever sent to the model. Raw gaze coordinates never leave the machine, and the session identifier is stripped before the request is made.
2. **Per-metric reliability tagging** — trial-based metrics (hit rate, reaction time) and gaze-kinematic metrics (fixation rate, screen coverage) are judged separately, since a low sampling rate invalidates the second category without affecting the first. Unreliable metrics are excluded from what the model may cite a number for, not silently trusted or used to block the whole report.
3. **Refusal gate** — sessions with too few trials to say anything trustworthy are refused outright rather than interpreted.
4. **Constrained, audience-specific generation** — separate prompts for Teacher, Psychologist, and Family/Parent audiences, each with its own tone and vocabulary constraints. None of them may name a diagnosis.
5. **Automatic verification** — every generated claim is checked against the evidence pack: any number not traceable to the evidence (within a tight tolerance) is rejected, as is any claim using diagnostic language (ADHD, autism, disorder, etc.) or citing an evidence key that doesn't exist.
6. **Self-consistency sampling** — the model is queried multiple times per generation; agreement across samples is surfaced to the reviewer.
7. **Human-in-the-loop review** — every surviving claim requires an explicit Approve / Edit / Reject decision before use.
8. **Audit log** — timestamp, audience, a hash of the evidence pack, and the reviewer's decision are logged for every claim.

Groundedness is not the same as correctness: the verifier confirms a claim traces to the computed metrics, not that those metrics are themselves valid (see [Known Limitations](#known-limitations--roadmap)).

## Getting started

### Prerequisites

- Python 3.9+
- A free [Google AI Studio](https://aistudio.google.com/apikey) account, if you want the AI Interpretation tab (the rest of the dashboard works without it)

### Installation

```bash
git clone <this-repo-url>
cd visitrail
pip install -r requirements.txt
```

### API key setup

The AI Interpretation tab calls Gemini's free tier through its OpenAI-compatible endpoint.

```bash
export GEMINI_API_KEY="your-key-here"      # macOS/Linux
setx GEMINI_API_KEY "your-key-here"        # Windows (restart terminal after)
```

`trustworthy_genai.py` also has a `GEMINI_API_KEY_HARDCODED` fallback constant for local convenience. **Do not commit a real key there** — see [Secrets](#secrets) below.

### Running

```bash
streamlit run app.py
```

Upload one or more semicolon-separated CSVs (see expected format in-app) and click **Run Analysis**.

## Expected CSV format

Semicolon-separated, with these columns: `Timestamp`, `EyeTracker` (as `(x,y)`), `GameObjectPos (Screen Coordinates)` (as `(x,y)`), `ObjectName`, `ObjectState` (`Appear` / `Disappear` / `Correct` / `Incorrect`), `Label` (`Target` / `Distractor`). The app shows a sample when no file is uploaded.

## Data privacy & ethics

- Only the aggregated evidence pack (rounded numeric metrics, no coordinates, no session identifier) is sent to the LLM provider for AI Interpretation — this is visible in-app before generation, in the "Evidence actually sent to the model" panel.
- This tool assumes session data has already been appropriately de-identified/anonymized under your institution's protocol before it reaches VisiTrail. It does not perform anonymization itself.
- Gaze/eye-movement data can be a soft biometric identifier in some conditions; treat pseudonymized data (a re-linkable ID) as personal data, not anonymized data.
- The AI Interpretation layer is explicitly instructed never to name or imply a diagnosis, and a keyword filter blocks diagnostic language as a second check. It is not a substitute for clinical judgment and should not be treated as one.
- If deploying this against real participant data, confirm your institution's ethics/IRB approval covers the specific processing described here, including any use of a third-party LLM API.

## Known limitations / roadmap

Documented here deliberately, rather than left implicit — several of these affect the numbers the dashboard reports and are being addressed:

- **Reaction-time / hit-rate matching inconsistency.** `analyze_object_timeline()` and `analyze_click_performance_simple()` match clicks to targets using different time windows and different denominators (all objects vs. targets only). This can produce reaction-time and hit-rate figures on different tabs that look contradictory for the same session. A single, disappearance-aware, 1:1 matching function is planned to replace both.
- **Duplicate function definition.** `detect_fixations_ivt_pixel()` is currently defined twice in `app.py`; the second definition silently shadows the first. Current behavior is correct (the two are identical), but the duplication should be removed.
- **Unvalidated threshold.** The 721 px/s fixation/saccade velocity threshold has not been validated specifically for this cohort. A per-participant sensitivity analysis is planned before this threshold is treated as authoritative in any write-up.
- **Sampling-rate dependency.** Gaze-kinematic metrics (fixation rate, screen coverage) require a sampling rate the logging pipeline does not always reach. The AI Interpretation layer excludes these metrics when unreliable; the underlying "Eye Movement Patterns" and "Velocity Profile" plots do not yet carry the same caveat and should be read with that in mind.
- **Self-consistency granularity.** Independent generations that agree in substance but differ in phrasing are currently counted as disagreement (exact text match), understating true agreement.

## Secrets

Add a `.gitignore` (included) covering any file where you've filled in `GEMINI_API_KEY_HARDCODED`, and prefer the `GEMINI_API_KEY` environment variable for anything that leaves your machine. If a key is ever committed, revoke it in Google AI Studio immediately — GitHub's secret scanning will typically flag it within minutes, but the key should be considered compromised the moment it's pushed, public repo or private.

## Project structure

```
.
├── app.py                 # Streamlit dashboard: CSV processing, plots, rule-based recommendations
├── trustworthy_genai.py   # AI Interpretation tab: evidence pack, verification, review UI
├── requirements.txt
└── README.md
```

## License

Not yet specified — add a `LICENSE` file appropriate to your institution's requirements before treating this as open source (MIT and Apache-2.0 are common choices for research software).

## Citation

If you use VisiTrail in academic work, please cite this repository. A `CITATION.cff` or BibTeX entry can be added once the project has a stable release.
