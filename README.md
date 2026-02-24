# 🧪 A/B Prompt Testing with Langfuse + Ministral 14B

> Compare two prompt strategies side-by-side using Langfuse's experiment framework and Ministral 14B.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LANGFUSE                           │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │  Prompt v1   │    │  Prompt v2   │                   │
│  │  label: "a"  │    │  label: "b"  │                   │
│  │  Concise     │    │  Detailed    │                   │
│  │  temp: 0.3   │    │  temp: 0.7   │                   │
│  └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                           │
│         ▼                   ▼                           │
│  ┌─────────────────────────────────────┐                │
│  │     Dataset: ab-test-eval-set       │                │
│  │     8 questions + expected output   │                │
│  └──────────────┬──────────────────────┘                │
│                 │                                       │
│      ┌──────────┴──────────┐                            │
│      ▼                     ▼                            │
│  ┌────────────┐    ┌────────────┐                       │
│  │  Run: A    │    │  Run: B    │                       │
│  │  8 traces  │    │  8 traces  │                       │
│  │  + scores  │    │  + scores  │                       │
│  └────────────┘    └────────────┘                       │
│                                                         │
│            📊 Compare in Dashboard                      │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │  Ministral 14B    │
              │  ministral-14b-   │
              │  latest           │
              └───────────────────┘
```

---

## 🆚 The Two Prompts

| | **Prompt A** | **Prompt B** |
|---|---|---|
| **Label** | `a` | `b` |
| **Strategy** | Concise & direct | Chain-of-thought |
| **System prompt** | _"Answer concisely in at most 2 sentences."_ | _"Think step by step. Provide a detailed explanation with examples."_ |
| **Temperature** | `0.3` | `0.7` |
| **Expected behavior** | Short, to-the-point answers | Longer, reasoned explanations |

---

## 📋 Evaluation Dataset

The dataset `ab-test-eval-set` contains 8 programming questions in Portuguese:

| # | Question | Expected Answer (summary) |
|---|---|---|
| 1 | What is recursion in programming? | Function that calls itself |
| 2 | Explain what a REST API is | Interface using HTTP + REST principles |
| 3 | Difference between list and tuple in Python? | Lists are mutable, tuples are not |
| 4 | What is Big O notation? | Describes algorithm complexity |
| 5 | Concept of closure in JavaScript? | Function accessing outer scope after return |
| 6 | What is a relational database? | Data in tables with relations |
| 7 | What is Docker used for? | Containers packaging apps + dependencies |
| 8 | What is machine learning? | Algorithms learning patterns from data |

---

## 📏 Scoring Metrics

Each response is evaluated with two metrics:

| Metric | What it measures | Score range | Ideal |
|---|---|---|---|
| **keyword_overlap** | % of expected keywords found in the response | `0.0` – `1.0` | `1.0` |
| **response_length** | Penalizes too short (<50 chars) or too long (>500 chars) responses | `0.0` – `1.0` | `1.0` (50–500 chars) |

Scores are recorded in Langfuse per trace, enabling side-by-side comparison in the dashboard.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your keys:

```env
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
MISTRAL_API_KEY=...
```

### 3. Create prompts + dataset in Langfuse

```bash
python setup_langfuse.py
```

### 4. Run the A/B test

```bash
python run_ab_test.py
```

---

## 📊 Sample Output

```
==================================================
Running experiment: ab-test-prompt-a
==================================================

  Q: What is recursion in programming?
  A: Recursion is when a function calls itself...

  Q: Explain what a REST API is.
  A: A REST API is an interface for communication...

  ...

============================================================
RESULTS COMPARISON
============================================================
Metric                       Prompt A     Prompt B     Winner
------------------------------------------------------------
avg_keyword_overlap             0.625        0.750          B
avg_response_length             0.950        0.820          A
```

---

## 📁 Project Structure

```
a-b-prompt-testing/
│
├── .env.example        # Environment variables template
├── .gitignore
├── requirements.txt    # Python dependencies
│
├── setup_langfuse.py   # Seeds prompts (a/b) + dataset into Langfuse
└── run_ab_test.py      # Runs both experiments and compares results
```

---

## 🔗 How Langfuse Tracks Everything

```
Prompt "a" ──► compile(question) ──► Ministral API ──► response ──► score ──► Langfuse Trace
                                                                              │
Prompt "b" ──► compile(question) ──► Ministral API ──► response ──► score ──► Langfuse Trace
                                                                              │
                                                                    ┌─────────┘
                                                                    ▼
                                                          Langfuse Dashboard
                                                          ┌─────────────────┐
                                                          │ Dataset Runs    │
                                                          │ ├─ prompt-a     │
                                                          │ └─ prompt-b     │
                                                          │                 │
                                                          │ Compare scores  │
                                                          │ side by side    │
                                                          └─────────────────┘
```

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| LLM | Ministral 14B (`ministral-14b-latest`) |
| Observability | Langfuse (prompt management + experiments) |
| Language | Python 3.10+ |
| Prompt management | Langfuse prompt versioning with labels |
| Evaluation | Custom keyword overlap + response length scorers |
