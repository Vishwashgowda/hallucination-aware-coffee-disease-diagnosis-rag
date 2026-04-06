# ☕ Coffee Disease Diagnosis System - Advanced RAG with Hybrid Dataset

AI-powered coffee disease diagnosis using Advanced RAG with hybrid JSON+PDF retrieval and hallucination detection. Defaults to a local OpenAI-compatible model via Ollama (no cloud credits). Claude is optional.

## 🎯 Features

- **Hybrid Dataset Architecture**: Structured JSON knowledge (60%) + PDF evidence (40%) for balanced disease coverage
- **Multi-turn Interactive Diagnosis**: Iterative clarification questions to gather complete symptom information
- **Advanced RAG**: Multi-query expansion, hybrid scoring, source diversity, and deduplication
- **Ambiguity Detection**: Automatically identifies missing symptom details (color, pattern, location, spread)
- **Metadata Filtering**: Filter retrieval by symptom type, region, and affected plant parts
- **CRAG Filtering**: Evaluates and filters retrieved documents for relevance
- **RAC Question Generation**: Generates clarification questions grounded in retrieved context
- **Active Hallucination Gating**: Verification can request more clarification when uncertain
- **Streamlit UI**: User-friendly web interface showing JSON/PDF source tags
- **State Management**: Tracks conversation history and confidence levels

## 🏗️ System Architecture

```
USER INPUT
    ↓
AMBIGUITY DETECTION (missing info?)
    ↓
MULTI-QUERY EXPANSION (3 query variants)
    ↓
DUAL RETRIEVAL:
  ├─ JSON Retriever (60%, metadata filtering)
  └─ PDF Retriever (40%, multi-variant)
    ↓
HYBRID SCORING (vector + lexical + variants)
    ↓
DEDUPLICATION & SOURCE DIVERSITY
    ↓
CRAG FILTERING (relevance evaluation)
    ↓
CLARIFICATION (concise, targeted questions)
    ↓
MULTI-TURN LOOP (max 3 questions)
    ↓
STRUCTURED DIAGNOSIS (candidate comparison)
    ↓
ACTIVE HALLUCINATION GATING
    ├─ Safe to finalize → DIAGNOSIS
    └─ Not safe → MORE CLARIFICATION
    ↓
UI DISPLAY ([JSON]/[PDF] tagged results)
```

## 🗄️ Hybrid Dataset

The system uses a **balanced hybrid approach** to prevent retrieval bias:

### **Dataset Composition:**
- **Structured JSON** (`data/disease_knowledge.json`): 15 diseases with equal representation (~6-7% each)
  - Coffee Leaf Rust, Nitrogen Deficiency, Magnesium Deficiency
  - Coffee Berry Borer, Anthracnose, Brown Eye Spot
  - Root Rot, Iron Chlorosis, White Stem Borer
  - Coffee Wilt Disease, Red Spider Mites, Phoma Leaf Spot
  - Potassium Deficiency, Scale Insects, Zinc Deficiency

- **PDF Documents** (`data/pdfs/`): Supporting evidence and detailed information
  - `Coffee (1).pdf`
  - `coffee_cultivation_guide.pdf`
  - `i4985e.pdf`

### **Advantages:**
✅ **No bias** - Equal disease representation prevents "always Coffee Leaf Rust" problem  
✅ **Metadata filtering** - Filter by symptom type (foliar, nutritional, pest, etc.)  
✅ **Structured parsing** - Clean disease fields (symptoms, treatment, prevention)  
✅ **Karnataka-specific** - Regional information for local context  
✅ **Faster retrieval** - Metadata narrows search space  
✅ **Evidence backing** - PDFs provide supporting documentation

### **Retrieval Ratio:**
- **60% from JSON**: Primary source for balanced disease coverage
- **40% from PDF**: Supporting evidence and detailed information
- Sources tagged as **[JSON]** or **[PDF]** in UI for transparency

## 📦 Installation

### Prerequisites
- Python 3.11 recommended
- Ollama installed (OpenAI-compatible local endpoint): https://ollama.com/download
- GPU optional (improves speed)
- Optional: Claude key only if you choose cloud fallback

### Setup

1) Clone/Navigate to project directory:
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Create `.env` (local model defaults):
```bash
LLM_PROVIDER=openai_local
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=phi3
LLM_API_KEY=ollama
# Optional Claude fallback:
# ANTHROPIC_API_KEY=your_claude_key
# CLAUDE_MODEL=claude-3-haiku-20240307
LOG_LEVEL=INFO
```

4) Pull a local model (example: Phi-3):
```bash
ollama pull phi3
```

5) Ensure data files exist:
   - PDFs in `data/pdfs/`:
     - `Coffee (1).pdf`
     - `coffee_cultivation_guide.pdf`
     - `i4985e.pdf`
   - JSON in `data/`:
     - `disease_knowledge.json` (included in project)

## 🚀 Usage

### Web UI (Recommended)

Run the Streamlit application:
```bash
python -m streamlit run ui/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Testing
```bash
# Basic diagnosis test
python tests/test_diagnosis.py "Leaves are turning yellow with brown spots"

# Test hybrid retrieval balance
python tests/test_hybrid.py
```

## 🧪 Testing Hybrid Retrieval

Run the hybrid retrieval test to verify balanced disease distribution:

```bash
python tests/test_hybrid.py
```

This tests:
- ✅ No single disease dominates (e.g., not always "Coffee Leaf Rust")
- ✅ JSON vs PDF ratio (~60/40)
- ✅ Diverse diagnoses for diverse symptoms
- ✅ Source diversity (max 2 chunks per source)

**Expected Results:**
- 8 different symptom queries should yield varied diagnoses
- Coffee Leaf Rust should be <70% of total diagnoses
- Each query should show mix of [JSON] and [PDF] sources

## 📂 Project Structure

```
coffee_project/
├── data/
│   ├── pdfs/
│   │   ├── Coffee (1).pdf
│   │   ├── coffee_cultivation_guide.pdf
│   │   └── i4985e.pdf
│   ├── disease_knowledge.json      # Balanced JSON knowledge (15 diseases)
│   └── vector_db/                  # FAISS indexes (generated)
├── src/coffee_diagnosis/
│   ├── core/
│   │   ├── pdf_loader.py           # PDF loading & chunking
│   │   ├── vector_store.py         # FAISS vector database
│   │   ├── json_retriever.py       # JSON disease knowledge retriever
│   │   ├── retriever.py            # Hybrid retriever (JSON + PDF)
│   │   └── llm_client.py           # LLM abstraction (local/cloud)
│   ├── rag/
│   │   ├── ambiguity_detector.py   # Symptom analysis
│   │   ├── retrieval_evaluator.py  # CRAG filtering + deduplication
│   │   ├── clarification_gen.py    # Question generation
│   │   └── state_manager.py        # Conversation state
│   └── diagnosis/
│       ├── controller.py           # Multi-turn orchestration
│       ├── diagnosis_generator.py  # Structured diagnosis generation
│       └── hallucination_checker.py# Active gating verification
├── ui/
│   └── streamlit_app.py            # Web UI with source tags
├── tests/
│   ├── test_diagnosis.py           # CLI diagnosis test
│   ├── test_hybrid.py              # Hybrid retrieval balance test
│   └── test_multiturn.py           # Multi-turn conversation test
├── scripts/
│   └── diagnose.py                 # Diagnosis helper script
├── config/
│   └── settings.py                 # Configuration
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── TESTING_GUIDE.md                # Detailed testing instructions
```

## 🧠 Core Modules

### 1. PDF Loader (`pdf_loader.py`)
- Loads PDFs using PyPDF
- Splits into ~500 token chunks with overlap
- Preserves source metadata

### 2. Vector Store (`vector_store.py`)
- Creates FAISS index for similarity search
- Uses all-MiniLM-L6-v2 embeddings
- Performs efficient L2 similarity search

### 3. Retriever (`retriever.py`)
- Combines current query with conversation history
- Returns top-k relevant documents
- Supports relevance scoring

### 4. Ambiguity Detector (`ambiguity_detector.py`)
- Detects missing symptom information
- Categorizes by: color, pattern, location, spread, timing
- Prioritizes missing attributes

### 5. Retrieval Evaluator - CRAG (`retrieval_evaluator.py`)
- Scores chunks for relevance
- Filters by threshold (default 0.3)
- Identifies disease-specific keywords

### 6. Clarification Generator - RAC (`clarification_gen.py`)
- Generates questions using shared LLM client (default: local via Ollama)
- Ensures questions are grounded in context
- Validates question relevance

### 7. State Manager (`state_manager.py`)
- Tracks conversation state
- Maintains confidence score
- Checks stop conditions
- Provides conversation history

### 8. Multi-turn Controller (`controller.py`)
- Orchestrates complete pipeline
- Manages main interaction loop
- Coordinates all modules

### 9. Diagnosis Generator (`diagnosis_generator.py`)
- Generates final diagnosis using shared LLM client (default: local via Ollama)
- Provides treatment recommendations
- Formats output clearly

### 10. Hallucination Checker (`hallucination_checker.py`)
- Generates diagnosis multiple times (default 3; configurable via `NUM_GENERATIONS_FOR_VERIFICATION`)
- Compares for consistency
- Detects hallucinations
- Provides verification report

## 🔄 Multi-turn Flow

1. **User Input**: Describes symptoms
2. **Ambiguity Detection**: Identifies missing information
3. **Context Retrieval**: Fetches relevant PDF sections
4. **Question Generation**: Asks clarification question
5. **User Response**: Provides additional details
6. **Confidence Update**: Increases confidence score
7. **Stop Check**: Evaluates stop conditions
8. **Loop**: Repeats until stop condition or max turns
9. **Diagnosis**: Generates final output
10. **Verification**: Checks for hallucinations

## ⚙️ Stop Conditions

The diagnosis stops when:
- **Confidence > 80%**: Sufficient confidence achieved
- **1 Disease Remaining**: Clear diagnosis
- **3 Questions Asked**: Maximum turns reached

## 🔐 Hallucination Detection

The system uses a SelfCheckGPT-inspired approach:
- Generates diagnosis (default 1 generation to reduce calls; can be increased)
- Compares disease names for consistency when multiple generations are used
- Checks treatment recommendation similarity
- Flags inconsistencies with warnings
- Provides consistency score

## 📊 Example Interaction

**User**: "Leaves turning yellow with brown spots"

**System**: "Thank you. On which parts of the plant are these yellow leaves appearing most? Are they on the lower leaves, upper leaves, or throughout?"

**User**: "Mainly on lower leaves"

**System**: "Are the brown spots forming any particular pattern, such as rings or spreading outward?"

**User**: "They look like rings"

**System**: [Diagnosis]
- **Disease**: Coffee Leaf Rust
- **Confidence**: 85%
- **Reason**: Yellow leaves with concentric brown rings match classic rust symptoms
- **Treatment**: Apply copper fungicide spray during morning hours...

## 🛠️ Configuration

Edit `src/controller.py` or individual modules to customize:
- `chunk_size`: PDF chunk size (default: 500)
- `top_k`: Number of retrieved documents (default: 5)
- `max_questions`: Maximum clarification rounds (default: 3)
- `relevance_threshold`: Filtering threshold (default: 0.3)
- `num_generations`: Hallucination check generations (default: 3)

## 📝 Output Format

```
══════════════════════════════════════
            📋 DIAGNOSIS RESULT
══════════════════════════════════════

🏥 Disease: [Disease Name]
📊 Confidence: [XX.X]%

🔍 Reason:
[Detailed explanation of why this diagnosis]

💊 Treatment:
[Specific treatment steps]

🛡️ Prevention:
[Prevention measures]

📚 Source: [PDF source]
══════════════════════════════════════

HALLUCINATION CHECK
Consistency Score: XX.X%
Agreement Level: 100%
```

## ⚠️ Important Notes

1. **Context Only**: The system only provides diagnoses based on retrieved PDF context
2. **Not a Substitute**: Should not replace consultation with local agricultural experts
3. **Hallucination Warnings**: System flags potential hallucinations
4. **Local by Default**: Uses local OpenAI-compatible endpoint via Ollama (no cloud costs once the model is pulled)
5. **Optional Cloud**: Set `LLM_PROVIDER=anthropic` and add `ANTHROPIC_API_KEY` to use Claude instead

## 🔧 Troubleshooting

### No PDFs Found
- Ensure PDF files are in project root
- Check file names match exactly

### Slow Performance
- First run creates FAISS index (takes ~2-3 minutes)
- Subsequent runs use cached index
- Clear `vector_db/` to rebuild index

### LLM Errors
- Local: ensure Ollama is running and model pulled (`ollama list`)
- Claude (optional): set `LLM_PROVIDER=anthropic` and add a valid `ANTHROPIC_API_KEY`

### Poor Diagnoses
- Ensure PDFs are in `data/pdfs`
- Provide detailed symptoms (color, pattern, location, timing)
- Answer clarification questions fully

## 📚 Knowledge Base

The system uses 3 comprehensive PDFs:
- **Coffee.pdf**: General coffee cultivation information
- **coffee_cultivation_guide.pdf**: Detailed cultivation practices
- **i4985e.pdf**: Additional coffee disease references (FAO document)

## 🎓 Technical Stack

- **Python 3.11**: Core language
- **LangChain**: Framework for LLM orchestration
- **FAISS**: Vector similarity search
- **Sentence-Transformers**: Embeddings (all-MiniLM-L6-v2)
- **Local LLM (Ollama)**: Default LLM backend (OpenAI-compatible)
- **Anthropic Claude**: Optional fallback
- **Streamlit**: Web UI framework
- **PyPDF**: PDF processing

## 📖 References

- [RAG (Retrieval-Augmented Generation)](https://arxiv.org/abs/2005.11401)
- [CRAG (Corrective RAG)](https://arxiv.org/abs/2401.15884)
- [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
- [Corrective RAG](https://github.com/langchain-ai/langgraph)

## 🤝 Contributing

For improvements or bug reports:
1. Test changes thoroughly
2. Ensure no hallucinations are introduced
3. Verify with sample queries
4. Document changes

## 📄 License

This project is for educational purposes.

## 👥 Support

For issues or questions:
- Check troubleshooting section
- Review logs in terminal
- Verify all files are present
- Test with sample queries

---

**Last Updated**: March 2026
**Version**: 1.0.0
