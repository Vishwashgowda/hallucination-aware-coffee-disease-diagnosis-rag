# Coffee Disease Diagnosis with Balanced Hybrid RAG

## Abstract

This repository contains a coffee disease diagnosis system built on a balanced hybrid Retrieval-Augmented Generation (RAG) architecture. The system combines structured JSON disease knowledge and unstructured PDF agronomy references, performs multi-turn clarification to resolve ambiguous symptom descriptions, and verifies generated diagnoses for consistency. The implementation is optimized for local inference by default through an OpenAI-compatible Ollama endpoint, with optional cloud fallback.

## 1. Objectives and Scope

### 1.1 Objectives

The project addresses three practical failure modes in agricultural diagnosis assistants:
1. single-disease retrieval bias,
2. premature diagnosis on incomplete symptom input,
3. unsupported or inconsistent model outputs.

### 1.2 Scope

The system provides decision support for coffee disease diagnosis from text-based symptom descriptions. It is intended for research and educational use, not as a replacement for expert agronomists or field diagnostics.

## 2. System Design

### 2.1 End-to-End Pipeline

The diagnosis flow is:
1. ambiguity detection from user query,
2. multi-query expansion,
3. hybrid retrieval from JSON and PDF sources,
4. CRAG-style relevance filtering,
5. clarification question generation,
6. multi-turn interaction loop,
7. structured diagnosis generation,
8. consistency/hallucination verification.

### 2.2 Clarification Behavior

The multi-turn interface accepts:
- detailed symptom responses,
- short binary responses (e.g., `yes`, `no`, `partly`),
- uncertainty responses (e.g., `not sure`, `don't know`).

This allows natural user interaction without forcing long symptom restatements on every turn.

## 3. Dataset and Retrieval Strategy

### 3.1 Data Sources

- Structured dataset: `data/disease_knowledge.json` (15 diseases).
- Document corpus: PDFs under `data/pdfs/`.

### 3.2 Hybrid Retrieval Ratio

- 60% JSON retrieval (balanced disease coverage),
- 40% PDF retrieval (supporting evidence and detail).

### 3.3 Transparency

Evidence is tagged by source type (`[JSON]` and `[PDF]`) in the UI and internal context payloads.

## 4. Core Implementation Modules

### 4.1 `src/coffee_diagnosis/core`

- `pdf_loader.py`: PDF loading and chunking.
- `vector_store.py`: FAISS index build/load/search.
- `json_retriever.py`: structured disease retrieval.
- `retriever.py`: hybrid merge/rank logic.
- `llm_client.py`: LLM provider abstraction.

### 4.2 `src/coffee_diagnosis/rag`

- `ambiguity_detector.py`: missing-attribute detection.
- `retrieval_evaluator.py`: relevance filtering and chunk evaluation.
- `clarification_gen.py`: context-grounded question generation.
- `state_manager.py`: multi-turn state and history management.

### 4.3 `src/coffee_diagnosis/diagnosis`

- `controller.py`: full orchestration of diagnosis flow.
- `diagnosis_generator.py`: final diagnosis synthesis.
- `hallucination_checker.py`: multi-generation consistency checks.

### 4.4 UI

- `ui/streamlit_app.py`: interactive web interface with evidence visibility and stepwise interaction.

## 5. Repository Organization

```text
GenAi project/
├── config/
├── data/
│   ├── disease_knowledge.json
│   ├── pdfs/
│   ├── vector_db/
│   └── vector_db_json/
├── paper/
│   ├── coffee_disease_diagnosis_ieee.tex
│   ├── conference_101719.tex
│   ├── IEEE_paper.pdf
│   ├── PAPER_GUIDE.md
│   ├── PAPER_CHECKLIST.md
│   ├── DELIVERY_SUMMARY.md
│   ├── FINAL_STATUS.md
│   └── INTEGRATION_SUMMARY.md
├── scripts/
├── src/coffee_diagnosis/
├── test/
│   ├── test_diagnosis.py
│   ├── test_multiturn.py
│   ├── test_hybrid.py
│   ├── evaluate_metrics.py
│   ├── evaluation_dataset.json
│   └── TESTING_GUIDE.md
├── ui/
├── README.md
└── requirements.txt
```

## 6. Environment and Setup

### 6.1 Prerequisites

- Python 3.11 (recommended),
- Ollama installed,
- required model available locally (default: `phi3`).

### 6.2 Installation

```bash
pip install -r requirements.txt
```

### 6.3 `.env` (local default)

```bash
LLM_PROVIDER=openai_local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=phi3
LLM_API_KEY=ollama
LOG_LEVEL=INFO
```

### 6.4 Model Pull

```bash
ollama pull phi3
```

## 7. Running the Application

### 7.1 Streamlit UI

```bash
python -m streamlit run ui/streamlit_app.py
```

### 7.2 Direct Test Execution

```bash
python test/test_diagnosis.py "Leaves are turning yellow with brown spots"
python test/test_multiturn.py
python test/test_hybrid.py
python test/evaluate_metrics.py
```

## 8. Test Suite Description

### 8.1 `test/test_diagnosis.py`

Smoke-level diagnosis execution for direct symptom input.

### 8.2 `test/test_multiturn.py`

Validates question-answer interaction flow and controller transitions.

### 8.3 `test/test_hybrid.py`

Checks disease-distribution behavior under diverse prompts to detect dominant-disease bias.

### 8.4 `test/evaluate_metrics.py`

Runs labeled evaluation and reports:
- Top-1 accuracy (single-turn baseline),
- Top-1 accuracy (multi-turn),
- Top-3 accuracy (proxy),
- consistency score,
- hallucination rate,
- Precision@5 (keyword proxy),
- mean CRAG relevance score,
- multi-turn improvement.

## 9. Current Experimental Results

### 9.1 Labeled Evaluation (`test/evaluate_metrics.py`, 12 cases)

- Top-1 accuracy (single-turn): 50.00%
- Top-1 accuracy (multi-turn): 66.67%
- Top-3 accuracy (proxy): 83.33%
- Avg consistency score: 88.89%
- Hallucination rate: 25.00%
- Avg Precision@5 (keyword proxy): 48.19%
- Avg CRAG relevance score: 0.6588
- Top-1 improvement: +16.67%

### 9.2 Hybrid Distribution Check (`test/test_hybrid.py`, 8 cases)

- no single-disease dominance observed,
- Coffee Leaf Rust frequency: 2/8 (25.0%).

## 10. Paper Artifacts

All manuscript and submission artifacts are under `paper/`.

Primary manuscript:

`paper/coffee_disease_diagnosis_ieee.tex`

## 11. Configuration Notes

Primary runtime configuration is in `config/settings.py` and `.env`.

Commonly adjusted values include:
- maximum clarification turns,
- retrieval depth/top-k,
- relevance threshold,
- verification generation count,
- provider/model selection.

## 12. Failure Modes and Troubleshooting

### 12.1 Import or path errors

Run commands from project root and use `test/...` paths (not `tests/...`).

### 12.2 Slow first startup

Initial runs build/load vector indexes. Subsequent runs should reuse cached indexes.

### 12.3 Weak diagnosis quality

Provide symptom details for color, pattern, location, spread, and timing; allow clarification turns.

### 12.4 LLM backend issues

Confirm Ollama service and model availability:

```bash
ollama list
```

## 13. Reproducibility Notes

For consistent paper reporting:
1. use the same labeled dataset (`test/evaluation_dataset.json`),
2. run evaluation from a clean session,
3. report proxy metrics explicitly as proxies,
4. pair metric table with hybrid-distribution results.

## 14. Ethical and Practical Disclaimer

This system is a research prototype for decision support. It should be used with agronomic judgment and local field validation before operational deployment.

