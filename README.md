# Hiver AI Email Response System

> An AI-powered email reply suggestion system with comprehensive, multi-dimensional quality evaluation.

## 🎯 Overview

This system takes incoming customer support emails and generates intelligent suggested replies — grounded in a dataset of past email conversations using **RAG (Retrieval-Augmented Generation)**. It then evaluates each generated response across **7 quality dimensions** to produce per-response and overall system scores.

```
Incoming Email → [RAG Retriever] → [LLM Generator] → Suggested Reply
                       ↓                                     ↓
              Past Email Dataset                    Multi-Metric Evaluator
                                                          ↓
                                              Per-Response & System Scores
```

## 🏗️ Architecture

### 1. Dataset (`dataset/`)
- **50 curated customer support email pairs** across 8 categories
- Categories: billing, technical support, feature requests, account management, onboarding, complaints, general inquiries, follow-ups
- Each pair includes: incoming email, reference response, and metadata (urgency, sentiment, requires_action)
- **How it was built**: Hand-authored to be realistic and diverse — varying sender names, companies, tones (frustrated to polite), complexity (simple questions to multi-part issues), and realistic product details
- **Why synthetic**: Full control over category distribution, no privacy concerns, guaranteed quality, reproducible

### 2. Response Generator (`generator/`)
- **RAG Pipeline**: FAISS vector index over past emails → semantic retrieval → few-shot prompting
- **Retriever** (`retriever.py`): Uses `sentence-transformers` (all-MiniLM-L6-v2) to embed emails, FAISS for fast nearest-neighbor search. Returns top-3 most similar past conversations.
- **Responder** (`responder.py`): Constructs a prompt with retrieved few-shot examples + the new email, calls Groq API (Llama 3.3 70B Versatile) to generate a reply.
- **Why RAG over fine-tuning**: Interpretable (you can see which examples were retrieved), no training cost, works well with small datasets, easy to update by adding new emails to the dataset.

### 3. Evaluation System (`evaluation/`) — **Core Focus**

#### What "Accurate" Means for Email Replies

Exact match is far too strict — there are many valid ways to reply to an email. "Accuracy" for suggested replies is multi-dimensional:

| Metric | Weight | What It Measures | Method |
|--------|--------|------------------|--------|
| **Intent Coverage** | 25% | Does the reply address every question/concern? | LLM extracts intents from incoming, checks each is addressed |
| **Completeness** | 20% | Does it cover all key info from the reference? | LLM compares generated vs reference for missing points |
| **Semantic Similarity** | 15% | Is the meaning aligned with the reference? | Cosine similarity of sentence embeddings |
| **Tone** | 15% | Professional, empathetic, appropriate? | LLM rates tone quality |
| **Actionability** | 10% | Clear next steps for the customer? | LLM checks for specific actions/timelines |
| **Overall LLM Judge** | 10% | Holistic "would a customer be satisfied?" | LLM provides overall quality assessment |
| **Fluency** | 5% | Grammatically correct and readable? | LLM rates grammar and flow |

#### Why These Metrics Are Right

1. **Intent Coverage is weighted highest (25%)** because the #1 job of a reply is to answer every question the customer asked. A beautifully written response that misses a question is a bad response.

2. **Semantic similarity alone is insufficient**. Two perfectly valid replies ("I'll refund you" vs "I've processed your refund") can have moderate embedding similarity. That's why it's only 15% of the score.

3. **LLM-as-judge provides nuance** that embedding metrics miss — tone, empathy, actionability, and whether the customer would actually be satisfied.

4. **Multi-metric composite prevents gaming**. A response can't score high by being fluent but unhelpful, or complete but rude.

5. **Each metric has reasoning**: The LLM explains *why* it gave each score, making the evaluation interpretable and auditable.

#### Composite Score Formula

```
S_composite = Σ (weight_i × metric_i)
```

Where weights sum to 1.0 and each metric is on [0, 1].

#### Output Format

**Per-response:**
```
Email: "Invoice discrepancy for July"
├── Semantic Similarity:  0.82
├── Intent Coverage:      0.95  (3/3 intents addressed)
├── Tone Score:           0.90  (professional, empathetic)
├── Completeness:         0.85  (missing timeline detail)
├── Actionability:        0.80  (clear next step provided)
├── Fluency:              0.95
├── LLM Judge:            0.88
└── COMPOSITE SCORE:      0.88
```

**Overall system report** includes mean, median, std dev, min/max, per-category breakdown.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Setup

```bash
# Clone the repo
git clone <repo-url>
cd hiver-email-ai

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the System

```bash
# 1. Generate responses for all dataset emails
python main.py

# 2. Generate for a subset
python main.py --count 10

# 3. Generate for a single email
python main.py --email "Hi, I need help with my billing. I was charged twice this month."

# 4. Run full evaluation (generates responses + evaluates)
python evaluate.py

# 5. Evaluate a subset
python evaluate.py --count 5

# 6. Interactive demo
python demo.py
```

### Regenerate the Dataset (Optional)

```bash
# Uses Groq API to generate fresh synthetic emails
python dataset/generate_dataset.py
```

## 📁 Project Structure

```
hiver-email-ai/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
├── dataset/
│   ├── generate_dataset.py            # Synthetic dataset generator (uses Groq)
│   └── email_dataset.json             # Pre-built dataset (50 email pairs)
├── generator/
│   ├── __init__.py
│   ├── retriever.py                   # FAISS-based semantic retrieval
│   └── responder.py                   # Groq-powered response generation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # 7 individual metric implementations
│   └── evaluator.py                   # Orchestrator + reporting
├── main.py                            # End-to-end response generation
├── evaluate.py                        # End-to-end evaluation
└── demo.py                            # Interactive demo
```

## 🔧 Technical Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| **Llama 3.3 70B (Groq)** | Extremely fast, high-performance open-weights LLM, outstanding reasoning capabilities |
| **RAG over fine-tuning** | More interpretable, no training cost, easy to update, works with 50 emails |
| **FAISS for retrieval** | Production-grade vector search, fast even at scale |
| **sentence-transformers (all-MiniLM-L6-v2)** | Runs locally (no API cost for retrieval), good quality embeddings, fast |
| **LLM-as-judge** | Captures nuances (tone, empathy) that embedding similarity misses |
| **7-metric composite** | No single metric captures "good reply" — multi-dimensional view is essential |
| **Pre-built + generatable dataset** | Works out of the box, but can be regenerated for reproducibility |

## 🤖 How AI Tools Were Used

This project was built with the assistance of **Google Antigravity (Gemini-powered coding assistant)**:

- **Architecture & planning**: AI helped design the multi-metric evaluation framework and system architecture
- **Code generation**: AI generated the initial codebase across all components, with human review and iteration
- **Dataset creation**: The 50 email pairs were generated with AI assistance to ensure diversity and realism
- **Prompt engineering**: AI helped iterate on the LLM prompts for both response generation and evaluation
- **All code was reviewed and tested** by the developer to ensure correctness and completeness

The Groq API is also used at runtime for:
- **Response generation**: Llama 3.3 70b generates suggested email replies
- **LLM-based evaluation**: 6 of the 7 metrics use Groq as an evaluator/judge

## 📊 Validation Approach

The evaluation system validates itself through:

1. **Reasoning transparency**: Each LLM metric returns a `reasoning` field explaining the score, making it auditable
2. **Cross-metric consistency**: If intent coverage is high but completeness is low, that flags an interesting case worth investigating
3. **Per-category analysis**: Breakdowns by email category reveal if the system struggles with certain types
4. **Score distribution**: Standard deviation and min/max show how consistent the system is
5. **Reference comparison**: Semantic similarity against reference responses provides a grounding baseline

## License

MIT
