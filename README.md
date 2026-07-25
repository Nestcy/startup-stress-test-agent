# Startup Stress Test AI Agent

An AI agent that helps founders pressure-test a startup idea before they spend months building it. It asks three questions, in order, and won't let you skip ahead until each one has a real answer:

1. **Do people actually want this?**
2. **Does the math work as a business?**
3. **Can we actually build and execute it?**

Built with LangChain + LangGraph, running on Groq, with live web research via Tavily and a human checkpoint after every stage — the agent proposes an analysis, you react to it, and only then does it move forward.

## The Three Questions

### 🎯 Do people want this? (Desirability)

Before anything else, the agent digs into who you're building for and whether the problem is real enough that people will actually switch to your solution.

- Who the customer actually is, and how they'd be segmented
- What's changed recently that makes now the moment (market shifts, new triggers)
- What people currently do instead of your solution, and how good those alternatives already are
- Whether your solution actually fits the problem, not just sounds like it does
- A desirability score out of 100, purely as a signal — not a verdict

### 💰 Does the math work? (Viability)

Assuming people want it, can it survive as a business? This stage runs the actual numbers.

- Market size (how many people could realistically buy this)
- Bootstrap vs. VC-backed — which funding path fits, and what ARR target you're aiming for
- Pricing and what a healthy customer relationship looks like financially
- Churn and customer lifetime, based on real industry benchmarks
- A full breakdown of how customers actually get acquired — leads → activation → paying customer — with CAC, LTV, and payback period
- A viability score out of 100

### ⚙️ Can we build it? (Feasibility)

Assuming the idea is wanted and the math works, is it actually executable?

- What it would take technically — architecture, MVP scope
- A realistic growth roadmap at 3, 12, 24, and 36 months
- A NOW / NEXT / LATER plan: what to prove first, what to build next, what to scale later
- A feasibility score out of 100

## Why It's Built This Way

Early on, this agent used to kill an evaluation automatically if a score came back too low — e.g. exit immediately if desirability scored under 30. That's been removed. A low score on an unvalidated, early-stage idea usually means "we don't have enough real information yet," not "this idea is bad" — and you can't score an assumption as if it were a fact. The agent now walks through all three questions regardless of score, and only stops early if *you* decide to stop.

## Revising a Stage

Ideas change as you learn more. You can go back and redo any one of the three stages — say, change your idea description and re-run the desirability analysis — without losing the work you already did on the other two. If a stage you're revising has later stages that already have results (e.g. you revise desirability after viability and feasibility are done), the agent asks whether you want to redo those too, or keep them as-is. Nothing is silently overwritten or silently kept — you decide.

## Ask Follow-Up Questions

Once an evaluation is complete, you can ask the agent direct questions about the result — "why is my viability score low?", "what's the biggest risk here?" — and it answers using the actual analysis it produced, not a fresh guess. This is separate from revising: asking a question doesn't change any scores or analysis, it just helps you understand what's already there. If the answer makes you want to actually change something, that's what revising a stage is for.

## Project Structure

```
startup-stress-test-agent/
├── src/
│   ├── __init__.py
│   ├── state.py                 # State schema and types
│   ├── graph.py                 # LangGraph orchestration with human-driven routing
│   ├── api.py                   # HTTP API (FastAPI) — used by the Lovable frontend
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── desirability_node.py # Do people want this?
│   │   ├── viability_node.py    # Does the math work?
│   │   ├── feasibility_node.py  # Can we build it?
│   │   ├── human_review_node.py # Checkpoint + revision-confirmation interactions
│   │   └── report_node.py       # Final report generation
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search_tool.py       # Tavily search wrapper
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging setup
├── requirements.txt
├── .env.example
├── Procfile                      # Railway deployment entry point
├── main.py                       # CLI entry point
└── README.md
```

## Installation

1. **Clone the repository:**

```
git clone https://github.com/Nestcy/startup-stress-test-agent.git
cd startup-stress-test-agent
```

2. **Create a virtual environment:**

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```
pip install -r requirements.txt
```

4. **Set up environment variables:**

```
cp .env.example .env
# Add your API keys to .env
```

## Environment Variables

Create a `.env` file with:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=qwen/qwen3.6-27b
DATABASE_URL=your_postgres_connection_string   # optional but recommended for the API — see below
```

`DATABASE_URL` is optional for local CLI use. If you're running the API (`src/api.py`) — which is what the Lovable frontend talks to — set this, or every in-progress evaluation is lost the moment the server restarts. On Railway, attaching a Postgres addon sets this automatically.

## Usage

### CLI (local, one-off runs)

```
python main.py
```

You'll be prompted to describe your idea and walk through each stage interactively in the terminal.

### API (for the Lovable frontend, or any web client)

```
uvicorn src.api:app --reload
```

Key endpoints:

| Endpoint | What it does |
|---|---|
| `POST /evaluate/start` | Start a new evaluation |
| `POST /evaluate/{id}/feedback` | Submit feedback at the current checkpoint and move to the next stage |
| `POST /evaluate/{id}/revise` | Redo a specific stage (`desirability`, `viability`, or `feasibility`) |
| `POST /evaluate/{id}/confirm-downstream` | Answer whether to also redo later stages after a revise |
| `POST /evaluate/{id}/ask` | Ask a question about a completed evaluation |
| `GET /evaluate/{id}` | Check the current state of an evaluation |

## Key Financial Formulas

**Customers you need:**
```
Customers Needed = Annual Revenue Target ÷ Yearly Revenue per Customer
```

**Monthly churn:**
```
Churn Rate = 1 ÷ Customer Lifetime (in months)
```

**Payback period:**
```
CAC Payback Period = Monthly CAC ÷ Monthly Revenue per Customer
Typical healthy target: under 12 months
```

**Lifetime Value to CAC:**
```
LTV:CAC = Lifetime Value ÷ Customer Acquisition Cost
Target: > 3:1 (healthy), > 4:1 (excellent)
```

## Requirements

- Python 3.10+
- LangChain + LangGraph
- Groq Python Client
- Tavily Search API access
- FastAPI + Uvicorn (for the API)
- python-dotenv
- pydantic

## Contributing

Areas for enhancement:

- Persistent conversation history for `/ask` (currently stateless per question)
- Export to PDF/docs
- Benchmark comparisons across evaluations
- Historical tracking of revisions over time

## License

MIT

## Support

For issues, questions, or feature requests, open an issue on GitHub.

---

**Built with:** Groq (Qwen 3.6 27B) · LangChain + LangGraph · Tavily Search API · FastAPI · Python
