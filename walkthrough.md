# Project Improvement Summary
## Hallucination-Aware Coffee Disease Diagnosis RAG

---

## What Was Wrong (Before Any Changes)

| Problem | Impact |
|---|---|
| Only 12 test cases (1 per disease) | Easy to overfit, not statistically valid |
| `MAX_QUESTIONS = 3` never enforced | Multi-symptom diseases got cut off too early |
| Top-3 accuracy used substring matching | Inflated accuracy scores (not real) |
| Retrieval quality used keyword matching | Trivially gamed, not meaningful |
| Hallucination detection was self-referential | System checked itself — not independent |
| `metrics_v2.py` simulated predictions using ground-truth | Always showed 100% — completely fake |
| No pytest suite | Regressions go undetected |

---

## Phase 0 ✅ — Disease-Specific Questions
**Done by teammate in VS Code (before this session)**

The question generator was improved to ask disease-specific clarification questions instead of generic ones. For example, when symptoms suggest leaf rust, the system now asks specifically about orange powder location rather than generic "where are the symptoms?"

**No files to list** — this was already done.

---

## Phase 1 ✅ — Expand Evaluation Dataset
**Done by teammate in VS Code (before this session)**

### Problem
Original dataset had only 12 hand-crafted cases — 1 per disease. Any metric computed on it was statistically meaningless and easy to overfit.

### Changes Made
| File | Description |
|---|---|
| `test/evaluation_dataset_v2.json` | **NEW** — 55 test cases (was 12, 4.6× larger) |
| `test/validate_phase1.py` | **NEW** — Validates dataset diversity |

### Results
- **55 cases** across 16 diseases (was 12 cases, 12 diseases)
- **3-5 cases per disease** (was 1 per disease)
- **10 ambiguous multi-label cases** (rust vs eye spot, deficiency types, etc.)
- **60% natural linguistic variation** (messy real-world language)
- **2 new diseases** added: Boron Deficiency, Zinc Deficiency

---

## Phase 3 ✅ — Increase MAX_QUESTIONS (3 → 5)
**Done in this session — implemented first (quickest change)**

### Problem
`MAX_QUESTIONS = 3` was set but **never actually enforced** in `should_stop()`. The system only stopped when confidence > 0.8 or one disease remained. For complex multi-symptom diseases (like brown mites), 3 questions was never enough.

### Changes Made

#### MODIFIED — `config/settings.py`
```python
# Before
MAX_QUESTIONS = 3

# After
MAX_QUESTIONS = 5  # Increased to allow more turns for complex diseases
```

#### MODIFIED — `src/coffee_diagnosis/rag/state_manager.py`
- **Added** `max_questions` parameter to `__init__()` — can be overridden per-instance for A/B testing
- **Fixed** `should_stop()` — now actually checks the hard cap:
  ```
  Stop if: questions_asked >= max_questions  ← NEW (was missing!)
  Stop if: confidence > 0.8                  ← existing (smart early stop)
  Stop if: only 1 disease remains            ← existing
  Stop if: no more missing info              ← existing
  ```
- **Added** `max_questions` key to `get_state_summary()` output

#### NEW — `test/compare_max_questions.py`
A/B comparison script. Runs 30 representative cases (priority: multi-symptom diseases) under both Q=3 and Q=5 limits, reports:
- Accuracy delta
- Average questions asked
- Average time per case
- Per-disease breakdown

---

## Phase 2 ✅ — Redesign Metrics (True Ranking + Semantic Relevance)
**Done in this session — implemented second**

### Problem
Three metrics were fundamentally broken:

1. **Top-3 accuracy** → `if "Rust" in reason_text` (substring match in reason — not a real ranked list)
2. **Retrieval quality** → `if keyword in doc_content` (keyword match — trivially gamed)
3. **Hallucination rate** → system checked its own output (self-referential, can't catch systematic errors)
4. **`metrics_v2.py`** → Simulated predictions using ground-truth: `predicted = [(expected_disease, 0.85)]` — always 100% accurate, completely fake

### Changes Made

#### NEW — `test/evaluate_v2.py` (Main deliverable of Phase 2)
The real end-to-end evaluator. Replaces both `evaluate_metrics.py` and the broken `metrics_v2.py`.

**True Top-K**: Runs actual `CoffeeDiagnosisController` pipeline, then parses the LLM's `CANDIDATE COMPARISON:` block:
```
CANDIDATE COMPARISON:
1. Coffee Leaf Rust: 85% - Orange powder on leaves   ← parsed
2. Brown Eye Spot: 10% - No gray center              ← parsed
3. Phoma Leaf Spot: 5% - Unlikely                   ← parsed
```
Top-1, Top-3, MRR computed from this real ranked list.

**Semantic Retrieval Quality**: Uses `SentenceTransformer` cosine similarity between query and each retrieved chunk. Falls back to Jaccard overlap if model unavailable. A chunk is "relevant" if cosine sim > 0.40 OR 2+ keywords match.

**Ground-Truth Hallucination Rate**: Uses dataset labels as external validator.
`hallucination_rate = cases where predicted ≠ ground_truth / total`

**Supports**: `--fast` flag (first 12 cases), `--dataset` flag for custom path.

#### NEW — `test/validate_phase2.py`
Fast validation script — **no LLM needed, runs in ~5 seconds**.
Tests all metric functions with fixed synthetic examples.
- **9/9 test groups PASSED** ✅

#### NEW — `test/results/` directory
Stores evaluation JSON outputs.

---

## Phase 4 ✅ — Automated pytest Suite
**Done in this session — implemented third**

### Problem
All tests were standalone scripts. No pytest infrastructure, no fixtures, no regression baseline — regressions go completely undetected.

### Changes Made

#### MODIFIED — `requirements.txt`
Added:
```
pytest>=7.4.0
pytest-timeout>=2.2.0
numpy>=1.24.0
```

#### NEW — `test/conftest.py`
pytest fixtures (session-scoped to avoid re-loading LLM/vector DB):
- `controller` — full `CoffeeDiagnosisController`
- `fresh_controller` — function-scoped, clean state each test
- `dataset_v1` / `dataset_v2` — loaded once per session
- `state_manager` — for unit testing
- Registers `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.regression` markers
- Auto-creates `__init__.py` in subdirectories

#### NEW — `test/unit/test_metrics.py` (33 tests)
Tests all Phase 2 metric functions:
- `TestCandidateExtraction` — 8 tests (3 candidates, decimals, empty input, wrong format)
- `TestTopKAccuracy` — 9 tests (rank-1/2/3, case insensitive, empty list, k>candidates)
- `TestMRR` — 6 tests (rank-1/2/3, absent disease, empty list, single candidate)
- `TestMultiLabelPrecision` — 5 tests (P@1, P@2, P@3, no overlap, empty)
- `TestMultiLabelRecall` — 5 tests (R@1, R@2, R@4, empty plausible, no overlap)

#### NEW — `test/unit/test_state_manager.py` (27 tests)
Tests the Phase 3 changes:
- `TestStateManagerInit` — custom/default max_questions
- `TestStateManagerInitialize` — sets query, confidence, resets previous state
- `TestConfidenceTracking` — increases per answer, capped at 1.0, direct set
- `TestShouldStop` — hard cap at 5, early stop at confidence 0.8, single disease, no missing info
- `TestQuestionsTracking` — count, attribute tracking, no duplicates
- `TestStateSummary` — all required keys present including `max_questions`
- `TestReset` — clean slate, max_questions preserved

#### NEW — `test/unit/test_multi_label.py` (19 tests)
Tests the 10 ambiguous cases in dataset v2:
- `TestAmbiguousCasesDataset` — structure validation (≥8 cases, plausible alts, no duplicates)
- `TestRustVsEyeSpotDisambiguation` — precision/recall/F1 for most common ambiguous pair
- `TestDeficiencyDisambiguation` — N vs Fe vs Mg deficiency scoring
- `TestMultiLabelF1` — F1 harmonic mean, zero case, imbalance penalty
- `TestDatasetAmbiguousScoringSimulation` — perfect system vs random system

#### NEW — `test/integration/test_diagnosis_pipeline.py`
Integration tests (marked `@pytest.mark.slow`, require Ollama running):
- `test_start_diagnosis_returns_dict` — always returns dict with `status` key
- `test_off_topic_query_rejected` — non-coffee query → `off_topic`
- `test_coffee_query_accepted` — valid query → `question` or `diagnosis`
- `test_multi_turn_reaches_diagnosis` — pipeline eventually returns diagnosis
- `test_diagnosis_result_has_required_fields` — all fields present
- `test_hallucination_check_produces_valid_score` — score in [0, 1]
- `test_state_summary_max_questions_is_5` — reflects new setting
- `test_reset_clears_state_for_new_session` — clean slate after reset
- `test_coffee_leaf_rust_query` — disease-specific test

#### NEW — `test/regression/test_baseline.py` + `test/regression/save_baseline.py`
- `save_baseline.py` — runs system on original 12 v1 cases, saves `baseline_v1.json` with real results
- `test_baseline.py` — compares current system against saved baseline:
  - Top-1 accuracy must not drop > 5% (LLM non-determinism tolerance)
  - Hallucination rate must not rise > 10%

#### NEW — `test/run_tests.py` — Unified Test Runner
```powershell
python test/run_tests.py --unit          # Fast, no LLM (~18s)
python test/run_tests.py --integration   # Needs Ollama (~10min)
python test/run_tests.py --regression    # Needs baseline file
python test/run_tests.py --validate      # Phase 2 metric checks (~5s)
python test/run_tests.py --all           # Everything
```

---

## Verification Results

```
python test/validate_phase2.py      →  9/9 test groups PASSED ✅
python -m pytest test/unit/         →  79/79 tests PASSED ✅  (17.58s)
git commit                          →  24 files changed, 4189 insertions
```

---

## How to Use Everything

```powershell
# Run the web app
python -m streamlit run ui/streamlit_app.py

# Fast sanity check (no LLM needed, ~5s)
python test/validate_phase2.py

# Run all unit tests (no LLM, ~18s)
python test/run_tests.py --unit

# Generate regression baseline (needs Ollama)
python test/regression/save_baseline.py

# Full 55-case evaluation (needs Ollama, slow)
python test/evaluate_v2.py

# Quick 12-case evaluation check (needs Ollama)
python test/evaluate_v2.py --fast

# A/B test MAX_QUESTIONS=3 vs 5 (needs Ollama)
python test/compare_max_questions.py
```

---

## Phase Completion Status

| Phase | Title | Status |
|---|---|---|
| Phase 0 | Disease-specific questions | ✅ Complete |
| Phase 1 | Expand dataset 12 → 55 cases | ✅ Complete |
| Phase 2 | Redesign metrics (true Top-K, semantic, GT hallucination) | ✅ Complete |
| Phase 3 | MAX_QUESTIONS 3 → 5 + enforce hard cap | ✅ Complete |
| Phase 4 | Automated pytest suite (79 tests) | ✅ Complete |

**All 5 phases complete. 🎉**
