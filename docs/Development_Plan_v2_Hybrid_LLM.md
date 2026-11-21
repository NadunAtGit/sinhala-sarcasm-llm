# Research-Oriented Development Plan (v2): Hybrid Encoder–LLM Framework

**Project:** Detect, Then Reason — Resolving Sarcasm-Induced Sentiment Errors in Sinhala and Singlish Social Media
**Prepared:** 21 November 2026 · **Start:** Monday 23 November 2026
**Follows:** `Research_Direction_LLM_Extension_v2.md` (RQ1–RQ4, RO1–RO4, H1–H5, conditions E1–HY3)

> **Purpose.** This is a plan for building *experiments*, not a product. Every module exists to produce evidence for a named hypothesis under controlled conditions. The software is finished when every condition in the experiment registry has been run once on the sealed test set, and every hypothesis has a result.
>
> **Defaults assumed** (from Research Direction v2, §14): the hybrid is the main system; four RQs; all three detector seeds (42, 43, 44); glossary condition optional. Where a choice changes the build, it is marked **[DECISION]**.

---

## Contents

1. [Research-to-Code Traceability](#1-research-to-code-traceability)
2. [System Architecture](#2-system-architecture)
3. [Experiment Registry](#3-experiment-registry)
4. [Repository Structure](#4-repository-structure)
5. [Data Contracts](#5-data-contracts)
6. [Module Specifications](#6-module-specifications)
7. [Validity Controls Built into the Code](#7-validity-controls-built-into-the-code)
8. [Build Schedule](#8-build-schedule)
9. [Pilot and Freeze Protocol (Gate G1)](#9-pilot-and-freeze-protocol-gate-g1)
10. [Main Runs (Gate G2)](#10-main-runs-gate-g2)
11. [Analysis Plan (Pre-Registered)](#11-analysis-plan-pre-registered)
12. [Research Log and Reproducibility](#12-research-log-and-reproducibility)
13. [Tests](#13-tests)
14. [Risks and Fallbacks](#14-risks-and-fallbacks)
15. [Day-1 Checklist](#15-day-1-checklist)

---

## 1. Research-to-Code Traceability

Every piece of code maps to an objective and a hypothesis. If a module cannot be placed in this table, do not build it.

| RO | What must exist | Code | Conditions | Hypotheses |
|---|---|---|---|---|
| RO1 — LLM reasoning framework | Retrieval index, 3 agents + synthesis, validated JSON output | `retrieval/`, `pipelines/single_pass.py`, `pipelines/multi_agent.py` | L1–L6 | H3 |
| RO2 — Hybrid framework | Encoder detector output → routing → LLM resolver | `baselines/export.py`, `pipelines/hybrid.py` | HY1–HY3 | H1, H2, H5 |
| RO3 — Components and controls | Ladder of LLM conditions; matched test-set oracles; composite conditions | `encoder_oracle/` (GPU), `analysis/composite.py` | E2, L1–L6, HY2, HY3, HYo | H4 |
| RO4 — Evaluation and error analysis | Subset metrics, penalty, Negative over-prediction rate, cluster bootstrap, McNemar + Holm, script strata, rationale review sheet | `eval/`, `analysis/` | All | H1–H5 |

---

## 2. System Architecture

### 2.1 Three systems, one scorer

```
                       ┌───────────────────────────── shared input: comment + parent-post caption ─────────────────────────────┐
                       │                                                                                                         │
 (A) ENCODER ALONE     │  XLM-R-large (3 seeds, existing) ──► sarcasm label + p(sarcastic) + sentiment label                     │
                       │                                                                                                         │
 (B) LLM ALONE         │  retrieve k train examples ─► [L1–L3 single pass]  or  [L4: literal │ incongruity │ intent ─► synthesis] │
                       │                                                     ─► sarcasm + literal sentiment + intended sentiment │
                       │                                                                                                         │
 (C) HYBRID            │  encoder detector ─► predicted sarcastic? ── no ──► keep encoder sentiment                              │
                       │                          │ yes                                                                          │
                       │                          └──► LLM resolver (L4 or L3) told "detector judged sarcastic [+ confidence]"   │
                       │                                 ─► intended sentiment                                                   │
                       └──────────────────────────────────────────────┬──────────────────────────────────────────────────────────┘
                                                                      ▼
                                   one scorer: same test IDs, same metrics, same bootstrap (99 test posts)
```

### 2.2 Why the hybrid is built this way (research rationale)

- **Division of labour follows the evidence.** The encoder's detection is the strongest available (Macro-F1 0.7547 ± 0.0035), but its sentiment on sarcastic rows is stuck (0.4309–0.4544). The LLM is used only where the encoder fails.
- **Routing makes the hybrid's ceiling measurable.** Sarcastic comments the detector misses never reach the LLM. The detector's sarcastic-class recall therefore bounds the hybrid, and must be reported alongside it. The oracle-routing composite (HYo, §3) gives the hybrid's upper bound without extra API calls.
- **Non-sarcastic predictions keep the encoder label.** This protects non-sarcastic performance (H2 requires non-inferiority there).

### 2.3 Tech stack

| Need | Choice |
|---|---|
| Language | Python 3.11 |
| Encoder work (E1 export, E2 oracle) | Existing thesis training code, HuggingFace Transformers + PyTorch, Colab A100 |
| LLM | Llama-3.3-70B instruct (F3 §3.7.2) via an OpenAI-compatible client; provider set in `.env` **[verify availability on day 1]** |
| Embeddings | `BAAI/bge-m3` (sentence-transformers); built once on Colab GPU |
| Vector store | ChromaDB (persistent folder) |
| Output validation | pydantic v2 |
| Retry, rate limiting | tenacity + async token bucket |
| Cache | SQLite, keyed by request hash |
| Prompts | Jinja2 templates under version control |
| Config / CLI | YAML + Typer |
| Statistics | numpy, scikit-learn, statsmodels |

---

## 3. Experiment Registry

The registry is the core research artefact. Each condition has one config file in `configs/conditions/`, with its hypothesis IDs written inside it.

| ID | Condition | Inputs | LLM calls per comment | Test rows | RQ / H |
|---|---|---|---|---|---|
| **E1** | XLM-R-large sentiment-only and multi-task (seeds 42–44) | Existing models | 0 | 6,875 | Baselines (all) |
| **E2** | XLM-R oracle on test (large; base if time) — gold sarcasm fed to the sentiment head | Thesis code with oracle flag, 3 seeds | 0 (GPU) | 6,875 | RQ3 / H4 |
| L1 | Zero-shot | Comment + caption | 1 | Eval set* | RQ1, RQ3 |
| L2 | + fixed few-shot (6 examples, one per sarcasm × sentiment cell) | + `few_shot.json` | 1 | Eval set* | RQ3 |
| L3 | + retrieved examples (k from pilot: 4 or 8) | + Chroma top-k | 1 | Eval set* | RQ3 |
| **L4** | + 3 agents + synthesis (full F3 design) | + agents | 4 | Eval set* | RQ1 / H3 |
| L5 *(optional)* | L4 + glossary hits | + glossary collection | 4 | Gold-sarcastic rows only | RQ3 (P RO4) |
| **L6** | LLM oracle: L4 + gold sarcasm stated | + gold label | 4 | Eval set* | RQ3 / H4 |
| **HY1** | Hybrid: E1 multi-task detector (each seed) → L4 resolver on predicted-sarcastic rows | Detector label | 4 per routed row | 6,875 (routed ≈ ⅓, to measure) | RQ2 / H1, H2, H5 |
| HY2 | HY1 + detector confidence band (low/medium/high) in the prompt | + band | 4 per routed row | 6,875 | RQ3 |
| HY3 | Hybrid with single-pass resolver (L3 prompt) | Detector label | 1 per routed row | 6,875 | RQ3 (value of agents) |
| HYo | *Composite:* encoder sentiment for gold-non-sarcastic rows + L6 output for gold-sarcastic rows | Computed offline from E1 + L6 | 0 extra | 6,875 | RQ2 upper bound |
| A1 | Context ablation of the best of L4 / HY1 (caption removed) | No caption | as source | Gold-sarcastic rows | RQ4 (secondary) |

\* **Eval set [DECISION at Gate G1]:** the full test set (6,875), or option B (all 2,277 sarcastic + 2,277 sampled non-sarcastic, stratified by script). The hybrid always runs on the full test set.

**Cache economics.** Hybrid prompts contain the detector's *binary* decision (and, in HY2, only a 3-level band), so most routed comments produce identical prompts for all three seeds. The cache serves the repeats; unique calls are roughly the union of comments flagged by any seed.

---

## 4. Repository Structure

Use a new repository, `sinhala-sarcasm-llm`. The thesis repo stays frozen; the E2 oracle training is run from the thesis code and its predictions are exported here.

```text
sinhala-sarcasm-llm/
├── README.md                      # reproduce every paper table in order
├── RESEARCH_LOG.md                # dated decisions, deviations, anomalies (§12)
├── PREREGISTRATION.md             # hypotheses, metrics, tests — frozen at G1
├── pyproject.toml                 # pinned versions (fixes the thesis limitation)
├── .env.example                   # LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
├── configs/
│   ├── base.yaml                  # temperature 0, max_tokens, paths, budget cap, frozen: false
│   ├── routing.yaml               # sarcasm threshold 0.5; confidence bands (fixed before pilot)
│   └── conditions/                # e1.yaml, l1.yaml … hy3.yaml (each lists its RQ/H IDs)
├── data/                          # git-ignored
│   ├── processed/                 # train / val / test .jsonl
│   ├── baselines/                 # E1 + E2 predictions with probabilities, per seed
│   ├── eval_sets/                 # pilot_val_300.txt, test_full.txt, test_optB.txt
│   ├── glossary/glossary.csv      # optional L5
│   └── chroma/                    # retrieval index (train only)
├── prompts/
│   ├── system.md.j2
│   ├── single_pass.md.j2          # L1–L3
│   ├── few_shot.json
│   ├── agents/{literal,incongruity,intent}.md.j2
│   ├── synthesis.md.j2
│   ├── oracle_note.md.j2          # L6
│   └── hybrid_note.md.j2          # HY1–HY3: detector decision (+ band)
├── src/sarcllm/
│   ├── config.py · data.py · schemas.py · prompts.py
│   ├── llm/{client,cache,ratelimit}.py
│   ├── retrieval/{embed,index,retrieve}.py
│   ├── baselines/export_check.py  # validate E1/E2 prediction files
│   ├── pipelines/{single_pass,multi_agent,hybrid}.py
│   ├── runner.py                  # batch, resume, budget cap, frozen-test guard
│   ├── eval/{metrics,bootstrap,significance}.py
│   └── analysis/{composite,tables,figures,error_sheet}.py
├── encoder_oracle/                # notebook + notes for E2 (runs the thesis code on Colab)
├── scripts/                       # Typer CLIs: prepare, build_index, run, score, stats, report
├── runs/                          # git-ignored; one folder per run_id
├── results/                       # paper tables (CSV + Markdown) and figures — committed
└── tests/
```

---

## 5. Data Contracts

### 5.1 Input record (`data/processed/*.jsonl`)

`id, post_id, text, context_text, script_type (sinhala|latin|mixed), sentiment (negative|neutral|positive), is_sarcastic (0|1), split`

**Required count check:** train 32,582 · val 6,938 · test 6,875; test sentiment 3,763 / 1,908 / 1,204; 2,277 sarcastic test rows (Negative 1,815 / Neutral 344 / Positive 118); 99 test posts.

### 5.2 Encoder prediction export (`data/baselines/<model>_seed<s>_<split>.csv`)

`id, sentiment_pred, p_neg, p_neu, p_pos, is_sarcastic_pred, p_sarcastic` (sarcasm fields empty for sentiment-only)

> **[DECISION / day-1 blocker]** The thesis kept best weights in memory and evaluated in-session (T §4.6). If checkpoints or probability-level predictions were not saved, the multi-task large model must be re-run for seeds 42–44 on Colab, to export `p_sarcastic` for validation and test. Budget one A100 day. Without probabilities, HY2 (confidence band) is not possible; HY1 still works from labels.

### 5.3 LLM output schema (all LLM conditions)

```json
{
  "literal_sentiment": "positive",
  "is_sarcastic": true,
  "intended_sentiment": "negative",
  "rationale": "≤ 60 words, citing the cue and the context",
  "evidence_ids": ["tr_10233"]
}
```

- **Agent outputs (L4–L6, HY1, HY2):** `{ "dimension", "judgement", "confidence": 0–1, "evidence" }`.
- **Hybrid resolver:** returns the same final schema. `is_sarcastic` is the LLM's own view, logged for analysis. The system's sarcasm label stays the detector's.

### 5.4 Prediction record (`runs/<run_id>/predictions.jsonl`)

`id, condition, run_id, detector_seed, routed (bool), is_sarcastic_pred, sentiment_pred, literal_sentiment_pred, rationale, retrieved_ids, parse_ok, n_calls, tokens_in, tokens_out, latency_s, cache_hit`

---

## 6. Module Specifications

| Module | Responsibility | Research-critical detail |
|---|---|---|
| `data.py` | Load, validate counts (§5.1), build eval sets | Option-B sampler stratified by script, seed 42, written once and committed as ID lists |
| `baselines/export_check.py` | Validate E1/E2 files | Every test ID present once; probabilities sum to 1; reproduces the ICARC-MS numbers (§13) |
| `llm/client.py` | Async chat calls, JSON mode, retries | Temperature 0; model string from the response saved per call |
| `llm/cache.py` | Request-hash cache | Key = SHA-256(model, messages, params) after NFC normalisation |
| `retrieval/index.py` | Embed train comments (`text` + `context_text`) into Chroma | Metadata: id, post_id, labels, script |
| `retrieval/retrieve.py` | Top-k neighbours | Train-only guard; returns IDs for `evidence_ids` validation |
| `pipelines/single_pass.py` | L1–L3 | One repair retry on invalid JSON, then logged failure |
| `pipelines/multi_agent.py` | L4–L6 | 3 agents in parallel with the same evidence, then synthesis; oracle note only in L6 |
| `pipelines/hybrid.py` | HY1–HY3 | Routes if `is_sarcastic_pred == 1` (threshold from `routing.yaml`); non-routed rows take the encoder sentiment unchanged; writes `routed` per row |
| `runner.py` | Batch execution | Resume from existing predictions; budget cap; **refuses `--split test` unless `frozen: true`** |
| `eval/metrics.py` | All metrics | Overall / sarcastic / non-sarcastic Macro-F1; penalty; per-class F1; **Negative over-prediction rate** (share of gold-sarcastic Neutral and Positive rows predicted Negative); detection Macro-F1 and sarcastic-class F1; routing recall; per script. Parse failures count as wrong and are also reported separately |
| `eval/bootstrap.py` | Post-level cluster bootstrap | 10,000 resamples of the 99 test posts; paired (same resample for both systems) |
| `eval/significance.py` | McNemar (continuity-corrected), Holm across H1–H5 | Per detector seed for hybrid comparisons, as in ICARC-MS |
| `analysis/composite.py` | HYo and any other offline composites | No API calls |
| `analysis/tables.py`, `figures.py` | Paper tables and the penalty figure (extends ICARC-MS Fig. 1 with the LLM and hybrid points) | Reads only from `runs/` and `data/baselines/` |
| `analysis/error_sheet.py` | Stratified sample of 100 rationales for native-speaker review | Strata: correct/incorrect × script; blind to condition |

### 6.1 Prompt rules

- Start from F3 §3.7.4 (system role, few-shot examples, user template). Remove the fields the corpus does not have (`sarcasm_type`, `rationale_span`).
- Instructions are in English; comments stay in their original script, untranslated.
- State the label rules from ICARC-MS §III-C verbatim in the system prompt. In particular, a sarcastic comment with Neutral surface sentiment stays Neutral. The LLM must use the same construct as the annotators.
- The hybrid note says only: "An automatic detector judged this comment to be sarcastic" (HY2 adds: "with low / medium / high confidence"). It does not hint at the expected sentiment.
- Prompts are frozen by git tag at Gate G1.

---

## 7. Validity Controls Built into the Code

| Threat to validity | Control | Where enforced |
|---|---|---|
| Test-set leakage through tuning | Test runs blocked until `frozen: true`; all prompt, k and threshold choices made on validation | `runner.py` |
| Retrieval leakage | Index built from train only; test asserts no val/test ID or post_id is ever retrieved | `retrieve.py`, `tests/test_leakage.py` |
| Few-shot leakage | Few-shot examples drawn from train; IDs logged | `tests/test_leakage.py` |
| Construct mismatch (LLM labels ≠ annotator labels) | Annotation rules copied into the system prompt; literal sentiment also requested | `prompts/system.md.j2` |
| Selective reporting | Pre-registration frozen at G1; every registered condition reported, including failures | `PREREGISTRATION.md` |
| LLM non-determinism | 300 validation comments run twice under L4; agreement reported | pilot |
| Pre-training contamination | Memorisation probe on 200 test comments (prefix completion; labels never used) | `scripts/probe.py` |
| Unfair comparison | Same IDs, same scorer, same bootstrap for encoder and LLM | `eval/` |
| Hybrid ceiling hidden | Routing recall and HYo upper bound always reported with HY1 | `metrics.py`, `composite.py` |
| Provider drift | Model string logged per call; all test runs inside one window | `client.py`, §10 |

---

## 8. Build Schedule

About 13 working days to Gate G1. GPU work (E2) runs in parallel with the LLM build.

| Day | Date | Build | Done when |
|---|---|---|---|
| 1 | Mon 23 Nov | Repo, pinned environment, `.env`; `prepare_data.py`; **check E1 prediction files** (§5.2); confirm LLM provider and model | Counts in §5.1 match; blocker resolved or re-run scheduled |
| 2 | Tue 24 Nov | `eval/metrics.py`, `bootstrap.py`, `significance.py`; import E1 | Reproduces ICARC-MS per-seed figures (§13) |
| 3 | Wed 25 Nov | *GPU:* start E2 oracle (large, 3 seeds) and, if needed, E1 multi-task re-runs with probabilities · *CPU:* `llm/client.py`, `cache.py`, `ratelimit.py`, `schemas.py` | 5 live calls succeed; second run 100% cache hits |
| 4 | Thu 26 Nov | Prompts for L1–L2; `single_pass.py`; `runner.py` (resume, budget cap, test guard) | L1 and L2 on 20 validation comments; parse rate logged |
| 5 | Fri 27 Nov | *Colab:* `embed.py`, `index.py`; save Chroma to Drive | 32,582 entries; leakage test passes |
| 6 | Mon 30 Nov | `retrieve.py`; L3; retrieval sanity check | Neighbours share the query's sarcasm label above the base rate (32.75%), reported by script |
| 7 | Tue 1 Dec | Agent and synthesis templates; `multi_agent.py` (L4) | L4 on 20 validation comments; `evidence_ids` valid |
| 8 | Wed 2 Dec | L6 oracle note; optional glossary collection (L5) | L6 and L5 run on 20 validation comments |
| 9 | Thu 3 Dec | `pipelines/hybrid.py` (HY1–HY3), using validation detector predictions | Routing counts per seed logged; non-routed rows identical to E1 |
| 10 | Fri 4 Dec | `analysis/composite.py` (HYo), `tables.py`, `figures.py`, `error_sheet.py`; memorisation probe | Dry-run report on validation produces every paper table |
| 11–12 | Mon 7 – Tue 8 Dec | **Pilot** (§9) | Pilot report written |
| 13 | Wed 9 Dec | **Gate G1:** freeze prompts, k, thresholds, eval set; write and commit `PREREGISTRATION.md`; tag `v1.0-frozen` | `frozen: true` set |

E2 must be finished and its test predictions exported by Gate G1. It is a GPU-only job using the thesis code; its test score is computed once, like every other condition.

---

## 9. Pilot and Freeze Protocol (Gate G1)

**Pilot set:** 300 validation comments, stratified by sarcasm × sentiment × script, with all sarcastic Positive and Neutral validation rows over-represented so that H5 can be checked early.

**Run:** L1–L4, L6, HY1–HY3 (plus L5 if kept), then score.

**Decide and record in `RESEARCH_LOG.md`:**

| Decision | Rule |
|---|---|
| k for retrieval (4 vs 8) | Higher sarcastic-subset Macro-F1 on the pilot; ties → smaller k (cheaper) |
| Keep L5 (glossary)? | Keep only if the glossary is ready and on schedule; its result is reported either way |
| Eval set for LLM-alone conditions | Full test if projected runtime ≤ 10 days and the budget fits; otherwise option B |
| Confidence bands (HY2) | Fixed *before* the pilot in `routing.yaml` (e.g., 0.5–0.65 / 0.65–0.85 / ≥ 0.85); not tuned |
| Prompt fixes | Only for format or parse failures; never to raise scores on specific items |
| Go / no-go | Parse rate ≥ 95%; non-determinism agreement reported; no leakage test failures |

**Freeze:** commit `PREREGISTRATION.md` containing H1–H5, primary metrics, tests, Holm family, eval set, k and thresholds. Tag the repo. Any later deviation is logged as a deviation, never silently.

---

## 10. Main Runs (Gate G2)

**Run window:** Thu 10 Dec – Fri 1 Jan (the holiday week is buffer). All test runs use one model version inside this window.

**Order** (most important first, so a budget or provider failure still leaves the core result):

1. HY1 (3 detector seeds) — H1, H2, H5
2. L4 — H3
3. L6 → HYo — H4 and the hybrid upper bound
4. HY3, then HY2 — components
5. L3 → L2 → L1 — the ladder
6. A1, then L5 if kept

Monitor the parse rate, cost and failures daily, and re-run failed items only. **Gate G2:** every registered condition scored once on test; no new conditions added afterwards for this paper.

---

## 11. Analysis Plan (Pre-Registered)

### 11.1 Primary comparisons

| Hypothesis | Comparison | Metric | Decision rule |
|---|---|---|---|
| H1 | HY1 vs E1 sentiment-only large | Sarcastic-subset sentiment Macro-F1 | Supported if the paired cluster-bootstrap 95% CI excludes zero for each detector seed (report all 3) |
| H2 | HY1 vs E1 multi-task large | Penalty; non-sarcastic Macro-F1 | Penalty CI excludes zero **and** non-sarcastic difference > −0.01 (non-inferiority margin, fixed at G1) |
| H3 | L4 vs E1 multi-task large | Detection Macro-F1 | Supported if E1 ≥ L4, or the CI includes zero |
| H4 | (L6 − L4) vs (E2 − E1 sentiment-only) | Sarcastic-subset gain | CI on the difference-in-gains excludes zero |
| H5 | HY1 vs E1 | Negative over-prediction rate on gold-sarcastic Neutral (n = 344) and Positive (n = 118) rows | CI on the proportion difference excludes zero; both cells reported |

Holm correction is applied across H1–H5 within each detector seed, as in ICARC-MS.

### 11.2 Secondary (descriptive; no confirmatory claims)

- The ladder L1 → L2 → L3 → L4: adjacent differences with CIs.
- HY1 vs HY2 (does confidence help?) and HY1 vs HY3 (do agents help inside the hybrid?).
- Routing recall per seed, and HY1 against HYo (how much is lost to detector misses).
- Script strata (Sinhala / Singlish / mixed) for all main conditions.
- Context ablation A1.

### 11.3 Error analysis

1. Confusion matrices on sarcastic rows for E1, L4 and HY1, set side by side.
2. Rows where HY1 fixes an E1 error vs where it introduces one (flip counts by class).
3. Native-speaker review of 100 rationales (blind to condition): is the stated reason plausible? Does it identify the actual cue?
4. Ambiguity check: report how many hybrid errors fall in the cells where the expert annotators disagreed (ICARC-MS: subtle sarcasm in comments labelled non-sarcastic).

### 11.4 Paper outputs produced by `scripts/report.py`

- **Table A** — main results: E1, E2, L4, L6, HY1, HYo; overall / sarcastic / non-sarcastic Macro-F1, penalty, detection.
- **Table B** — component ladder and hybrid variants.
- **Table C** — Negative over-prediction rates by condition.
- **Table D** — script-stratified results.
- **Figure 1** — penalty plot extending ICARC-MS Fig. 1.
- **Appendix** — prompts, routing counts, cost, parse rates, non-determinism, memorisation probe.

---

## 12. Research Log and Reproducibility

- **`RESEARCH_LOG.md`:** one dated entry per decision, anomaly or deviation, with its reason and commit hash. Reviewers ask about these; the log answers them, and it fixes the thesis's "checkpoint overwrite" type of problem.
- **Each run folder** stores the frozen config, git commit, model string, start and end times, token and cost totals, parse-failure list and cache-hit rate.
- **Pinned dependencies** in `pyproject.toml` (a limitation listed in ICARC-MS §VII).
- **Release:** prompts, configs, code and prediction files with the paper. The corpus is released according to the ethics decision.

---

## 13. Tests

| Test | Pass condition |
|---|---|
| `test_reproduce_baselines` | Scorer reproduces the ICARC-MS three-seed means from the exported predictions: sentiment-only large **0.6461**, multi-task large **0.6501** (overall Macro-F1); sarcastic-subset **0.4544** and **0.4502**; multi-task large detection Macro-F1 **0.7547** (tolerance ±0.0005) |
| `test_leakage` | No retrieved or few-shot ID, nor any `post_id`, from val or test |
| `test_hybrid_passthrough` | Non-routed rows in HY1 equal E1 predictions exactly |
| `test_parse` | Malformed, partial and extra-text JSON is handled; labels normalised; failures counted |
| `test_cache` | Same request gives the same key; a changed prompt gives a new key |
| `test_bootstrap` | Resampling is by post; a paired difference of identical systems gives CI [0, 0] |

---

## 14. Risks and Fallbacks

| Risk | Early sign | Fallback |
|---|---|---|
| E1 probabilities or checkpoints missing | Day-1 check fails | Re-run multi-task large with 3 seeds on Colab (≈ 1 A100 day); HY1 proceeds from labels meanwhile |
| Exported predictions don't reproduce ICARC-MS | `test_reproduce_baselines` fails | Stop; fix the alignment before any LLM run |
| Llama-3.3-70B unavailable or rate-limited | Day-1 check / pilot extrapolation | Same-size open model from another OpenAI-compatible host; log the switch in `RESEARCH_LOG.md` before G1 |
| LLM-alone too costly on the full test set | Pilot extrapolation > 10 days | Option B for L-conditions; the hybrid stays on the full test set |
| Low parse rate | < 95% in pilot | JSON mode / constrained output; one repair retry |
| bge-m3 retrieves poorly for Singlish | Day-6 check by script | Use the XLM-R encoder's [CLS] embeddings for retrieval; record it as a deviation |
| Hybrid gains vanish because routing recall is low | HY1 ≈ E1, HYo ≫ HY1 | This is a finding (detector misses are the bottleneck); report it as such |
| Schedule slips | A gate missed by more than 1 week | Cut L5, then A1, then HY2; never cut HY1, L4, L6/E2 or the bootstrap analysis |

---

## 15. Day-1 Checklist

- [ ] Create the `sinhala-sarcasm-llm` repo, pinned environment, `.env`
- [ ] Copy thesis splits into `data/processed/`; run the count checks in §5.1
- [ ] Locate E1 outputs: do validation and test predictions **with probabilities** exist for sentiment-only large and multi-task large, seeds 42–44? If not, schedule the Colab re-run for day 3
- [ ] Confirm the thesis code can run the oracle on the test split (E2)
- [ ] Confirm the LLM provider serves Llama-3.3-70B; record rate limits and price
- [ ] Start `RESEARCH_LOG.md` with today's decisions (defaults used, provider, model string)

---

*Aligned with: Research Direction v2 (21 Nov 2026) · ICARC-MS three-seed results (Tables I–III, §V–VII) · F3 §3.7 LLM design · Proposal §3.4 / §6.5–6.7 hybrid design. Supersedes the 17 Nov development plan, which predates the hybrid design and the three-seed baselines.*
