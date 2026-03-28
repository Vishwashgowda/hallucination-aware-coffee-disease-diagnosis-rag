# ☕ Coffee Disease Diagnosis System - Hallucination-Aware RAG

AI-powered coffee disease diagnosis using RAG with hallucination detection. Defaults to a local OpenAI-compatible model via Ollama (no cloud credits). Claude is optional.

## 🎯 Features

- **Multi-turn Interactive Diagnosis**: Iterative clarification questions to gather complete symptom information
- **Ambiguity Detection**: Automatically identifies missing symptom details (color, pattern, location, spread)
- **RAG Retrieval**: Fetches relevant information from PDF knowledge base
- **CRAG Filtering**: Evaluates and filters retrieved documents for relevance
- **RAC Question Generation**: Generates clarification questions grounded in retrieved context
- **Hallucination Detection**: SelfCheckGPT-inspired verification across multiple generations
- **Streamlit UI**: User-friendly web interface for easy interaction
- **State Management**: Tracks conversation history and confidence levels

## 🏗️ System Architecture

```
USER INPUT
    ↓
AMBIGUITY DETECTION (missing info?)
    ↓
RAG RETRIEVAL (fetch documents)
    ↓
CRAG FILTERING (evaluate relevance)
    ↓
CLARIFICATION (ask grounded questions)
    ↓
MULTI-TURN LOOP (repeat until confident)
    ↓
DIAGNOSIS GENERATION (final output)
    ↓
HALLUCINATION CHECK (SelfCheckGPT)
    ↓
UI DISPLAY (results)
```

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

5) Ensure PDFs exist in `data/pdfs`:
   - `Coffee (1).pdf`
   - `coffee_cultivation_guide.pdf`
   - `i4985e.pdf`

## 🚀 Usage

### Web UI (Recommended)

Run the Streamlit application:
```bash
python -m streamlit run ui/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Testing (optional)
- `python test_diagnosis.py "Leaves are turning yellow with brown spots"`

## 📂 Project Structure

```
coffee_project/
├── data/
│   ├── Coffee (1).pdf
│   ├── coffee_cultivation_guide.pdf
│   └── i4985e.pdf
├── vector_db/                      # FAISS index (generated)
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py              # PDF loading & chunking
│   ├── vector_store.py            # FAISS vector database
│   ├── retriever.py               # RAG retrieval
│   ├── ambiguity_detector.py      # Symptom analysis
│   ├── retrieval_evaluator.py     # CRAG filtering
│   ├── clarification_gen.py       # Question generation (RAC)
│   ├── state_manager.py           # Conversation state
│   ├── controller.py              # Multi-turn orchestration
│   ├── diagnosis_generator.py     # Final diagnosis
│   └── hallucination_checker.py   # SelfCheckGPT verification
-├── app.py                         # (legacy UI, use ui/streamlit_app.py)
├── test_diagnosis.py              # CLI test script
├── requirements.txt               # Dependencies
└── README.md                      # This file
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
