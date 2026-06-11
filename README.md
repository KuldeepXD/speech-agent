# 🏥 Speech/Feeding Medical AI Agent

An AI agent built with **Google ADK** that classifies patient queries into **Speech** or **Feeding** categories, identifies medical ailments, and generates targeted treatment assessment questions — with full **conversational memory** and **LangGraph compatibility**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
│  "Patient coughs when swallowing liquids after stroke"  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Google ADK Agent (Gemini 2.5 Flash)         │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ classify_   │→ │ identify_    │→ │ generate_      │  │
│  │ query()     │  │ ailment()    │  │ treatment_     │  │
│  │             │  │              │  │ questions()    │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                                            │             │
│  ┌─────────────────────────────────────────┘             │
│  │  save_to_memory() ← Session State (Memory)           │
│  └───────────────────────────────────────────────────────│
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Output Dictionary:                                      │
│  {                                                       │
│    "category": "Feeding",                                │
│    "ailment": "Post-Stroke Oropharyngeal Dysphagia",     │
│    "treatment_questions": [q1, q2, q3],                  │
│    "conversation_context": { ... }                       │
│  }                                                       │
└──────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** — Get one free at [Google AI Studio](https://aistudio.google.com/apikey)

## 🚀 Quick Start

### 1. Set up virtual environment

```bash
cd Speech_Agent
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

```bash
# Copy the example and add your key
copy .env.example .env

# Edit .env and replace with your actual Gemini API key
# GOOGLE_API_KEY=your_actual_key_here
```

### 4. Run the agent

```bash
# Interactive mode (conversation with memory)
python main.py

# Single query mode
python main.py "My child has trouble pronouncing the R sound"
```

## 💬 Usage Examples

### Interactive Session

```
You ▶ My child stutters when nervous and has trouble with the R sound

Agent ▶ {
  "category": "Speech",
  "ailment": "Stuttering / Fluency Disorders with Articulation Disorder",
  "ailment_description": "Co-occurring fluency and articulation difficulties...",
  "treatment_questions": [
    "When did the stuttering first appear, and has the severity changed...",
    "How does the stuttering and articulation difficulty affect...",
    "Has your child received any speech therapy previously..."
  ],
  "conversation_context": {
    "previous_ailments": [],
    "interaction_number": 1
  }
}

You ▶ Now the same patient also coughs when drinking water

Agent ▶ {
  "category": "Feeding",
  "ailment": "Oropharyngeal Dysphagia",
  ...
  "conversation_context": {
    "previous_ailments": ["Stuttering / Fluency Disorders..."],  ← Memory!
    "interaction_number": 2
  }
}
```

### Available Commands

| Command | Description |
|---------|-------------|
| *any text* | Analyze a patient description |
| `history` | View all conditions discussed in this session |
| `help` | Show help information |
| `quit` | Exit the application |

## 🔗 LangGraph Integration

The agent is designed to be used in LangGraph workflows. Two integration options:

### Option 1: ADK's Native Bridge (Recommended)

```python
from langgraph_bridge import create_langgraph_agent

# Wrap the ADK agent for LangGraph
lg_agent = create_langgraph_agent()

# Use as a node in your LangGraph workflow
```

### Option 2: Custom StateGraph

```python
from langgraph_bridge import create_langgraph_workflow

# Create a pure LangGraph workflow
graph = create_langgraph_workflow()
result = graph.invoke({
    "query": "patient has trouble swallowing",
    "messages": [],
})
print(result["result"])
```

### Test the Bridge

```bash
python langgraph_bridge.py
```

## 📊 Evaluation Benchmarking Framework

A comprehensive benchmarking suite is included to evaluate the performance of the Proposed Therapy Guide-MAS against baselines. It uses **RAGAS**, **Cosine Similarity**, and an **LLM Judge** to compute metrics like Answer Correctness, Context Precision, and Latency across curated test queries.

### Baselines
1. **Baseline 1**: Single LLM + Web Search Tool (ReAct agent)
2. **Baseline 2**: Vanilla RAG (Retrieve + Generate directly)
3. **Proposed**: Full LangGraph Multi-Agent System (Therapy Guide-MAS)

### Running Evaluations

Use the CLI orchestrator to run benchmarks or compute metrics:

```bash
# Run the entire pipeline (all baselines + metrics + report)
python -m evaluation.run_evaluation --all

# Run only a specific baseline
python -m evaluation.run_evaluation --baseline1
python -m evaluation.run_evaluation --proposed

# Dry run with only 2 test queries
python -m evaluation.run_evaluation --all --limit 2

# Only compute metrics and generate the report (if logs already exist)
python -m evaluation.run_evaluation --metrics --report
```

Output reports (CSV, Tables, Radar Charts) will be saved in `evaluation/results/`.

## 📁 Project Structure

```
Speech_Agent/
├── Documents/                      # Medical reference PDFs
│   ├── Speech/                     # SLP assessment manuals
│   └── Feeding/                    # Dysphagia resources
├── agent/                          # Agent package
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Configuration & constants
│   ├── prompts.py                  # System instructions
│   ├── tools.py                    # Tool functions
│   └── speech_feeding_agent.py     # Agent + Runner factory
├── evaluation/                     # Benchmarking framework
│   ├── baseline1_single_llm.py     # Baseline 1 (Single LLM + Tool)
│   ├── baseline2_vanilla_rag.py    # Baseline 2 (Vanilla RAG)
│   ├── proposed_mas.py             # Proposed LangGraph MAS
│   ├── metrics.py                  # Evaluation metrics engine
│   ├── generate_report.py          # Tables and charts generator
│   ├── run_evaluation.py           # CLI orchestrator
│   └── test_queries.py             # Curated benchmark dataset
├── main.py                         # CLI entry point
├── langgraph_bridge.py             # LangGraph integration
├── requirements.txt                # Dependencies
├── .env.example                    # API key template
└── README.md                       # This file
```

## 🧠 Conversational Memory

The agent maintains session state with scoped persistence:

| Scope | Prefix | Behavior |
|-------|--------|----------|
| User-level | `user:` | Persists across sessions for the same user |
| Temp | `temp:` | Cleared after each conversation turn |

Memory tracks:
- **Previous ailments** discussed in the conversation
- **Interaction count** for context-aware responses
- **Full conversation history** for recall

## 🩺 Supported Conditions

### Speech
Aphasia, Apraxia of Speech, Dysarthria, Stuttering, Voice Disorders, Articulation Disorders, Phonological Disorders, Language Delay, Childhood Apraxia of Speech, and more.

### Feeding
Dysphagia (Oral/Pharyngeal/Esophageal), Pediatric Feeding Disorder, Aspiration, Texture Aversion, Food Refusal, Oral Motor Dysfunction, Post-Stroke Dysphagia, and more.
