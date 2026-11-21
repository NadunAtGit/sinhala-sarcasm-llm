# Proposed Research Direction (v2): LLM-Based Extension after the Multi-Seed Results

**Prepared:** 21 November 2026
**Inputs analysed:** Research Proposal (P, Apr 2026) · First Three Chapters (F3, Jul 2026) · Final Thesis (T, Sep 2026) · Revised ICARC paper with three-seed results (ICARC-MS, Nov 2026)

> **One-line recommendation.** Restore the proposal's own **hybrid design** — XLM-RoBERTa for sarcasm detection, plus an LLM reasoning layer for intended-sentiment resolution. The multi-seed results now give that design a clear scientific justification: the encoder *can* detect sarcasm but *cannot* resolve what it means. The proposal changes very little; what changes is the reason for building it, and how it is tested.

---

## 1. What the Multi-Seed Results Changed

The three-seed re-runs (seeds 42, 43, 44; ICARC-MS §IV–V) closed the single-seed loophole. They confirmed the central finding, and corrected two earlier claims.

| Claim in the thesis (single seed) | Status after three seeds (ICARC-MS) | Consequence for the new study |
|---|---|---|
| Best sentiment Macro-F1 0.6565 (large multi-task) | Large models 0.6461–0.6501; **no reliable difference among them** (no bootstrap interval excludes zero) | Use the three-seed means as baselines, not 0.6565 |
| Scaling helps | **Confirmed.** +0.0174 to +0.0255 in every seed; multi-task base→large significant in 2 of 3 seeds, with bootstrap CI excluding zero in all 3 | XLM-R-large is the right baseline |
| Sarcasm detection is improvable | Large multi-task gives the best detection: Macro-F1 **0.7547 ± 0.0035**, sarcastic-class F1 **0.6793 ± 0.0018** | The encoder is a usable sarcasm *detector* |
| Sinhala-only training raised detection to 0.7155 | **Retracted.** Full-corpus large multi-task scores 0.7142 ± 0.0027 on the same Sinhala-script rows; the gain came from the easier subset | Script filtering is no longer a lever; drop it |
| Sarcasm-aware coupling slightly helps sarcastic rows | **Reversed.** Multi-task −0.34 / −0.42 points and hierarchical −0.79 / −0.57 points vs sentiment-only (base / large). The sentiment-only model is best on sarcastic rows at both scales | Encoder coupling does not help resolution: the core gap is now firmer |
| Sarcasm penalty 20–24 points | **Confirmed.** 19.4–24.4 points in **all 18 runs** | The primary target for the new study |
| Dominant error = Negative over-prediction | **Confirmed in all six configurations:** 44–49% of truly Neutral (n = 344) and 65–71% of truly Positive (n = 118) sarcastic comments predicted Negative; ironic praise misread as Positive only 5.7% (of 1,815) | A specific, measurable error the new method must reduce |
| Expert validation by accept/reject | **Fixed.** Blind expert annotation (n = 100): Krippendorff's α **0.955** (sentiment), **0.848** (sarcasm); expert disagreements fell on comments originally labelled non-sarcastic (under-detection of subtle sarcasm) | Labels are reliable enough to support a new method comparison |

### 1.1 The strengthened central finding

Across 18 runs, models that learn to detect sarcasm (sarcastic-class F1 0.6672–0.6810) are **no better** on sarcastic comments than a model that ignores sarcasm (sarcastic-subset sentiment Macro-F1 0.4309–0.4544; best = sentiment-only large, 0.4544 ± 0.0155). The paper's conclusion: *detecting sarcasm does not translate into resolving its intended sentiment, and resolution is the limiting factor* (ICARC-MS §VI-C).

This is the problem the LLM study must address.

### 1.2 Loopholes still open, and how the new study handles them

| Open loophole (ICARC-MS §VII) | Handled in the new study? | How |
|---|---|---|
| Oracle evaluated on validation only, single seed (0.4991 on sarcastic rows) | **Yes — essential** | Matched oracle on the **test** set: XLM-R oracle with 3 seeds, plus an LLM oracle |
| Negative-prior shortcut inferred, not verified | **Yes — central** | Measure the Negative over-prediction rate for every condition; test whether reasoning reduces it |
| Context gave no usable signal (single seed) | **Partly** | Test whether a reasoning model uses the parent post when an encoder did not (context on/off for the best LLM condition) |
| Ablations single-seed; scale confounded with learning rate and batch size | No (out of scope) | Declared as limitations of the baseline; not re-run |
| Thin sarcastic-Positive cells (118 test rows) | Reported, not fixed | Report per-class results with bootstrap CIs; no claims from that cell alone |
| Single primary annotator | Reported, not fixed | Blind expert α now supports reuse; listed as a limitation |

---

## 2. Alignment with the Original Proposal (Minimal-Change Check)

The proposal's architecture was a **two-stage hybrid**: a fine-tuned transformer extracts features and gives a preliminary classification with confidence scores; an LLM reasoning layer then produces the final sarcasm-aware sentiment label (P §3.4, §6.5). It also planned ablations of "the transformer alone, the agentic layer alone, and the combined ensemble" (P §6.6). F3 later reframed this as two independent arms (F3 §3.1.3).

The multi-seed results show *why* the original hybrid is the right design: the encoder handles detection reasonably, and fails exactly at the stage the LLM layer was meant to handle.

| Proposal element (source) | In the new study | Change | Reason |
|---|---|---|---|
| Hybrid: transformer features + confidence → LLM reasoning layer (P §3.4, §6.5; RO2) | **Kept — becomes the main system** | None in concept | Now motivated by the detection–resolution gap |
| Ablation: transformer alone / LLM alone / combined (P §6.6) | **Kept** | None | Gives RQ1 and RQ2 directly |
| Standalone RAG + multi-agent LLM arm, Llama-3.3-70B (F3 §3.7) | **Kept** as the "LLM alone" arm | None | F3 RQ2 is answered by it |
| LangGraph workflow (P §3.4) | Kept as an optional implementation | Plain Python orchestration is acceptable | Engineering choice; not a research variable |
| CNN/BiLSTM front-end on XLM-R (P §6.5, F3 §3.6.3) | **Dropped** | Use the existing three-seed [CLS] models | Already replaced in the thesis; retraining adds no research value |
| Retrieval over annotated training examples (F3 §3.7.5) | **Kept** | Built from the existing 32,582 training comments | As designed |
| Sarcasm lexicon / slang–emoji glossary (P RO4, F3 RO4) | **Kept, as a tested ablation** | Becomes an optional retrieval resource | Fulfils the unfulfilled RO4 at low cost |
| SinLlama silver labelling (P §2.6.4) | **Dropped** | — | The corpus is already human-labelled |
| ~5,000-post corpus, Facebook + X (P §3.4) | **Replaced** by the existing 46,395-comment corpus | No new data | Already exceeds the plan |
| Expert review of system outputs (P §6.7) | **Kept, scaled down** | Native-speaker review of a sample of LLM rationales and disagreement cases | Supports error analysis |
| Macro-F1 primary; held-out test (P §6.6) | **Kept** | Adds sarcastic-subset metrics, penalty and cluster bootstrap | Matches the multi-seed protocol |

**Net change:** the proposal's system is kept and its two unfulfilled objectives (RO2's LLM half, RO4's lexicon) are completed. The evaluation is aimed at the problem the thesis actually found.

---

## 3. Proposed Research Direction

### 3.1 Working title

**Detect, Then Reason: A Hybrid Encoder–LLM Framework for Resolving Sarcasm-Induced Sentiment Errors in Sinhala and Singlish Social Media**

Alternative: *Can Reasoning Close the Detection–Resolution Gap? Fine-Tuned Encoders, LLMs and Their Hybrid for Sarcasm-Aware Sentiment Analysis in Sinhala*

### 3.2 Background (one paragraph for the paper)

Prior work built the first sarcasm-annotated Sinhala social-media corpus (46,395 comments, 32.75% sarcastic) and established three-seed XLM-RoBERTa baselines. Sentiment Macro-F1 on sarcastic comments is 19.4–24.4 points below non-sarcastic comments in all 18 runs. Models trained to detect sarcasm (sarcastic-class F1 up to 0.6810) are no better on sarcastic comments than a sentiment-only model. Their dominant error is predicting Negative for sarcastic comments whose intended sentiment is Neutral or Positive. LLMs with retrieval and decomposed reasoning have been proposed to interpret communicative intent, but they have been evaluated mainly on English sarcasm *detection*, where they often trail fine-tuned encoders.

### 3.3 Research problem statement

Fine-tuned multilingual encoders detect sarcasm in Sinhala and Singlish text with moderate success, but do not convert that detection into correct intended sentiment. The failure persists across sarcasm-coupling architectures, model scales and text representations, and takes a specific form: sarcastic comments are pulled towards Negative regardless of their intended polarity. It is unknown whether an LLM reasoning layer — used alone, or given the encoder's sarcasm judgement — can resolve the intended sentiment of sarcastic comments where the encoder could not, and whether any improvement comes from retrieval, reasoning decomposition, or the sarcasm signal itself.

### 3.4 Research aim

To determine whether LLM-based reasoning, used alone or as a resolution layer on top of a fine-tuned XLM-RoBERTa sarcasm detector, reduces the sarcasm-induced sentiment penalty and the Negative over-prediction error in Sinhala and Singlish social-media text, and which components account for any improvement.

---

## 4. Research Gaps

Each gap follows the v1.4/v2.0 template (what exists → what is missing → why it matters → how this research addresses it). Evidence that has to be confirmed by the update SLR is marked *(verify)*.

| ID | Type | What exists | What is missing | Why it matters | Addressed by |
|---|---|---|---|---|---|
| **G1** | Methodological | Encoder baselines for Sinhala sarcasm-aware sentiment (ICARC-MS); LLM reasoning frameworks for English sarcasm (RAMAR, WM-SAR, RAM-SD) | No LLM-based or hybrid encoder–LLM system for Sinhala or Singlish *(verify)* | The only Sinhala system has a measured, unresolved ceiling on sarcastic text | RO1, RO2 |
| **G2** | Evaluative | LLM sarcasm studies report detection metrics (SarcasmBench; RAM-SD) | Almost no study measures sentiment resolution *on sarcastic items*, the sarcastic vs non-sarcastic penalty, or oracle conditions *(verify)* | Detection metrics hide the error that matters in practice (ICARC-MS shows detection and resolution diverge) | RO4 |
| **G3** | Diagnostic | Encoders gain nothing on sarcastic rows from learned sarcasm signals in 18 runs; the oracle is validation-only | No matched test-set evidence on whether *any* model can turn sarcasm knowledge into correct sentiment | Separates "the model cannot use the information" from "the task or labels are ambiguous" | RO3 (matched oracles) |
| **G4** | Error-mode | The Negative over-prediction pattern is documented in all six configurations (44–49% Neutral, 65–71% Positive) | No method has targeted it; the Negative-prior explanation is unverified | It is the concrete mechanism behind the penalty | RO4 |
| **G5** | Comparative / code-mixed | Prompted LLMs trail fine-tuned models for code-mixed sarcasm detection (Hinglish; Tamil/Malayalam) | No LLM vs encoder vs hybrid comparison on intended sentiment in a code-mixed low-resource language; no evidence on romanised Singlish | Tells the field whether LLM reasoning transfers to low-resource code-mixed registers | RO2, RO4 (script-stratified) |

---

## 5. Research Questions

**RQ1 (LLM alone — F3 RQ2 retained).** How does a retrieval-augmented, multi-agent LLM framework compare with fine-tuned XLM-RoBERTa in detecting sarcasm and in classifying the intended sentiment of Sinhala and Singlish social-media comments?

**RQ2 (hybrid — proposal RO2 retained).** Does a hybrid framework, in which a fine-tuned XLM-RoBERTa model detects sarcasm and an LLM reasoning layer resolves the intended sentiment of comments flagged as sarcastic, reduce the sarcasm penalty relative to either component alone?

**RQ3 (mechanism).** Which elements — retrieved annotated examples, multi-agent reasoning decomposition, the encoder's sarcasm signal, and gold sarcasm information (oracle) — account for any improvement? Does an LLM exploit sarcasm information more than the encoder does?

**RQ4 (error mode).** Does LLM reasoning reduce the Negative over-prediction error on sarcastic comments whose intended sentiment is Neutral or Positive, and how do the remaining errors vary by script (Sinhala, Singlish, mixed)?

> If four RQs are too many for the target venue, merge RQ3 into RQ2 and RQ4 into RQ1. The objectives below still hold.

---

## 6. Research Objectives

**RO1.** Implement an LLM reasoning framework on the existing corpus, comprising a retrieval index of annotated training comments, three reasoning agents (literal meaning, contextual incongruity, speaker intent) and an evidence-based synthesis step, as designed in F3 §3.7.

**RO2.** Construct the hybrid framework proposed in P §3.4. The three-seed XLM-RoBERTa-large multi-task model supplies the sarcasm prediction and its confidence; comments predicted sarcastic are routed to the LLM layer for intended-sentiment resolution, and the remaining comments keep the encoder's sentiment label.

**RO3.** Conduct a component and control analysis: the transformer alone, the LLM alone (zero-shot, few-shot, retrieval, multi-agent), the hybrid, the optional glossary-augmented retrieval (completing proposal RO4), and matched oracle conditions for both XLM-RoBERTa (three seeds, test set) and the LLM.

**RO4.** Evaluate all conditions on identical test comments using sentiment Macro-F1 on sarcastic and non-sarcastic subsets, the sarcasm penalty, the Negative over-prediction rate and sarcasm detection Macro-F1. Use post-level cluster-bootstrap confidence intervals and McNemar's test with Holm correction, and stratify results by script. Follow with a structured error analysis, including native-speaker review of a sample of LLM rationales.

---

## 7. Hypotheses

Baselines are the three-seed means from ICARC-MS Table II–III.

| ID | Hypothesis | Baseline to beat | Test |
|---|---|---|---|
| **H1** (RQ2) | The hybrid achieves higher sentiment Macro-F1 on sarcastic comments than the best encoder | 0.4544 ± 0.0155 (sentiment-only large) | Paired cluster-bootstrap CI excludes zero |
| **H2** (RQ2) | The hybrid has a smaller sarcasm penalty than XLM-R-large, without lowering non-sarcastic Macro-F1 | Penalty 0.2101–0.2227; non-sarcastic 0.6645–0.6724 | Bootstrap CI on the penalty difference; non-inferiority on the non-sarcastic subset |
| **H3** (RQ1) | XLM-R-large matches or beats the LLM alone on sarcasm detection | Detection Macro-F1 0.7547 ± 0.0035 | McNemar + bootstrap |
| **H4** (RQ3) | Supplying gold sarcasm status improves the LLM's sarcastic-subset sentiment by more than it improves the XLM-R oracle (both on test) | XLM-R test oracle (to be run, 3 seeds) | Difference-in-gains with bootstrap CI |
| **H5** (RQ4) | LLM-based resolution reduces the Negative over-prediction rate on truly Neutral and Positive sarcastic comments | 44–49% (Neutral, n = 344); 65–71% (Positive, n = 118) | Proportion difference with bootstrap CI; report per-class counts |

H3 is expected to *support* the encoder. That is deliberate: it locates where each component is strong, which is the argument for the hybrid.

---

## 8. Systems and Conditions Worth Studying

Each condition answers a specific RQ; none is included just to be thorough.

| ID | Condition | Components | Answers | Notes |
|---|---|---|---|---|
| E1 | XLM-R-large sentiment-only and multi-task | Existing 3-seed models | Baseline (all RQs) | No retraining; export predictions |
| E2 | XLM-R oracle on **test** | Base and/or large, 3 seeds | RQ3 / H4 | Closes the validation-only loophole; GPU only |
| L1 | LLM zero-shot | Llama-3.3-70B (F3 §3.7.2) | RQ1 floor | — |
| L2 | LLM + fixed few-shot | F3 Fig. 3.2 examples | RQ3 | — |
| L3 | LLM + retrieved examples | k nearest training comments | RQ3 | — |
| L4 | LLM + retrieval + multi-agent (full F3 design) | 3 agents + synthesis | RQ1 | "LLM alone" arm |
| L5 *(optional)* | L4 + slang/emoji glossary | Glossary from the training split | RQ3; completes P RO4 | Drop if behind schedule |
| L6 | LLM oracle | L4 + gold sarcasm status | RQ3 / H4 | — |
| **HY1** | **Hybrid** | E1 multi-task detector → L4 resolver on predicted-sarcastic comments | RQ2 / H1, H2, H5 | **Main system** |
| HY2 | Hybrid with confidence | HY1 + the detector's confidence band (low/medium/high) passed to the LLM | RQ3 | Tests whether the encoder signal helps the LLM |
| HY3 | Hybrid, single-pass resolver | E1 detector → L3 resolver (no agents) | RQ3 | Isolates the value of multi-agent decomposition inside the hybrid |
| A1 | Context ablation | Best of L4 / HY1 without the parent post | RQ4 (secondary) | Do reasoning models use the context encoders ignored? |

### Why the hybrid is affordable

The LLM is called only for comments the encoder predicts as sarcastic — about one-third of the test set, to be measured. Passing the detector's decision as a binary label plus a coarse confidence band keeps prompts identical across most of the three seeds, so the response cache serves repeated requests. This makes the **full 6,875-comment test set** feasible for the hybrid, preserving direct comparability with ICARC-MS.

### Model choice

Llama-3.3-70B via an OpenAI-compatible API, as in F3 §3.7.2. Confirm it is still served before the pilot. LLM fine-tuning and SinLlama remain out of scope (F3 §3.7.1 rationale) and are listed as future work.

---

## 9. Evaluation Protocol

- **Data:** the ICARC-MS corpus and post-level splits, unchanged; the test set is used once per finalised condition.
- **Primary metrics:** sentiment Macro-F1 on sarcastic test rows (n = 2,277); the sarcasm penalty.
- **Secondary metrics:** overall and non-sarcastic Macro-F1; per-class F1; Negative over-prediction rate; sarcasm detection Macro-F1 and sarcastic-class F1; script-stratified results; JSON parse rate; cost and latency per comment.
- **Statistics:** post-level cluster bootstrap over the 99 test posts (as in ICARC-MS); McNemar's test, continuity-corrected; Holm correction across H1–H5; encoder results as mean ± SD over three seeds; hybrid results reported per detector seed, plus their mean.
- **LLM controls:** temperature 0; frozen prompts after the validation pilot; a non-determinism check (300 comments run twice); a memorisation probe on 200 test comments; model string and run dates logged.
- **Human review:** a native speaker rates 100 LLM rationales (stratified by correct/incorrect and by script) for plausibility. This is the scaled-down form of P §6.7.

---

## 10. Scope and Delimitations

- Same corpus, labels, splits and three-seed encoder models as ICARC-MS; no new data collection or re-annotation.
- LLM adaptation is in-context only (prompting, retrieval, multi-agent reasoning); no LLM fine-tuning.
- One open-weight LLM; results are not claimed to generalise to other LLMs without testing.
- Text only; Facebook only; three-class intended sentiment and binary sarcasm, as in ICARC-MS.

---

## 11. Expected Contributions

**Methodological (hold regardless of results):**
1. The first LLM-based and hybrid encoder–LLM framework for sarcasm-aware sentiment analysis in Sinhala and Singlish, with prompts, retrieval configuration and routing logic released.
2. An evaluation protocol that separates detection from resolution — sarcastic-subset sentiment, penalty, matched test-set oracles and Negative over-prediction rate — applicable to any language.

**Empirical (direction depends on results):**
3. Whether LLM reasoning, alone or in the hybrid, closes any of the 19–24 point sarcasm penalty.
4. Which components matter (retrieval, decomposition, encoder signal, oracle information).
5. Whether reasoning corrects the Negative over-prediction error, and whether this differs for romanised Singlish.

---

## 12. Traceability Matrix

| Gap | RQ | RO | Hypothesis | Conditions | Thesis / ICARC-MS evidence |
|---|---|---|---|---|---|
| G1 | RQ1, RQ2 | RO1, RO2 | H1, H3 | L1–L4, HY1 | No LLM arm executed (T §1.8); future work names retrieval-augmented LMs (ICARC-MS §VII) |
| G2 | RQ2 | RO4 | H1, H2 | HY1 vs E1 | Penalty 19.4–24.4 in all 18 runs (Table II) |
| G3 | RQ3 | RO3 | H4 | E2, L6 | Coupling gives −0.34 to −0.79 points on sarcastic rows; oracle validation-only (§VI-C) |
| G4 | RQ4 | RO4 | H5 | All, error analysis | Negative over-prediction 44–49% / 65–71% (§V-B) |
| G5 | RQ1, RQ4 | RO2, RO4 | H3, H5 | Script-stratified results | Singlish 2.97 tokens/word; transliteration lowered detection (Table I, §V-D) |

---

## 13. Effect on the Existing Plans

| Existing document | Update needed |
|---|---|
| Fast-Track Roadmap | Add E2 (test oracle, week 2) and HY1–HY3 (weeks 5–7). Drop the Sinhala-only re-scoring task (resolved by ICARC-MS). Baselines become three-seed means |
| Chapter 1 Blueprint | Replace single-seed numbers (0.6565, 0.4330–0.4499, 0.7155) with three-seed figures; add the hybrid RQ2; restate the aim as above |
| Development Plan | Add `pipelines/hybrid.py` (routing + confidence band), XLM-R prediction export for all 3 seeds, and an E2 oracle script |
| SLR Blueprint v2.0 | Add search block S7: (`hybrid` OR `cascade` OR `routing` OR `"encoder–LLM"`) AND (`sentiment` OR `sarcasm`). Add section 5.4.6 "Hybrid encoder–LLM approaches". Gaps G1–G5 above replace the earlier G1–G5 wording |

---

## 14. Decisions Needed

1. **Main framing.** Hybrid as the main system (recommended), or the standalone LLM arm as main with the hybrid as an extension?
2. **Number of RQs.** Four (as above) or three (merge RQ3 into RQ2)?
3. **Glossary (L5).** Build it, which completes the proposal's RO4, or defer it?
4. **Detector for the hybrid.** Use all three seeds (recommended, with caching) or the single median seed?
5. **Target venue.** This determines the paper length and whether RQ4's script analysis goes in the main text.

---

*Sources: Research Proposal §3.4, §4.2, §6.5–6.7; First Three Chapters §1.6–1.7, §3.1, §3.7; Final Thesis §1.6–1.8, §5–7; Revised ICARC paper (three-seed) Abstract, §III–VII, Tables II–III. Literature cited in Section 4 as listed in the SLR Blueprint v2.0, Section 11.*
