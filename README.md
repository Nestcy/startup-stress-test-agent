# Startup Stress Test AI Agent

An AI agent that helps founders pressure-test a startup idea before they spend months building it. Give it an idea and a description, and it runs straight through three questions, gathering real data along the way to ground its answers rather than guessing:

1. **Do people actually want this?** (Desirability)
2. **Does the math work as a business?** (Viability)
3. **Can we actually build and execute it?** (Feasibility)

The founder isn't interrupted mid-run. Once the full report is ready, that's when they step in — reading it, asking questions, and revising whichever stage needs another pass.

## How It Works

**1. You give it an idea.** Just a name and a description — no back-and-forth required to get started.

**2. It runs the full pipeline in one go.** Each stage does its own web research (via Tavily) to find real market data, competitor pricing, churn benchmarks, and technical references — so its scoring is grounded in something, not just the model's assumptions about your idea. This takes a few minutes since it's several search + LLM calls per stage.

**3. You get a complete report.** Desirability, viability, and feasibility scores, a full analysis for each, and an overall GO / CONDITIONAL / NO-GO recommendation.

**4. Now you're in control.** From here:
- **Ask questions** — "why is my viability score low?", "what's Competitor X charging right now?" (this can trigger a live search automatically if the question needs current info)
- **Revise a stage** — change your idea description and re-run desirability, viability, or feasibility specifically, without losing the other stages' results

## Why It's Built This Way

An earlier version of this agent used to kill an evaluation automatically if a score came back too low, and paused for approval after every single stage. Both of those got removed:

- **No more auto-kill on low scores.** A low score on an unvalidated, early-stage idea usually means "we don't have enough real information yet," not "this idea is bad" — you can't score an assumption as if it were a fact. The agent runs all three stages regardless of score.
- **No more pausing mid-run.** Approving three separate checkpoints before seeing a finished result added friction without adding much value — the founder's actual judgment matters most once they can see the whole picture, not one incomplete piece at a time.

## Revising a Stage

You can go back and redo any one of the three stages — say, change your idea description and re-run desirability — without losing the work already done on the other two. If the stage you're revising has later stages that already have results (e.g. you revise desirability after viability and feasibility are done), the agent asks whether you want to redo those too, or keep them as-is. Nothing is silently overwritten or silently kept — you decide.

## Ask Questions, With Live Search When Needed

Once an evaluation is complete, you can ask the agent anything about it. If your question needs information the evaluation doesn't already have — current pricing, a specific competitor, recent news — it automatically runs a fresh web search before answering, rather than guessing or refusing. If it doesn't need that, it just answers from the evaluation's own analysis.

## Project Structure

```
startup-stress-test-agent/
├── src/
│   ├── __init__.py
│   ├── state.py                  # State schema and types
│   ├── graph.py                  # Straight-through pipeline + revision routing
│   ├── api.py                    # HTTP API (FastAPI) — used by the Lovable frontend
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── desirability_node.py  # Do people want this? (with web research)
│   │   ├── viability_node.py     # Does the math work? (with web research)
│   │   ├── feasibility_node.py   # Can we build it? (with web research)
│   │   ├── revision_node.py      # The one interrupt point, used only during /revise
│   │   └── report_node.py        # Final report generation
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search_tool.py        # Tavily search wrapper
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logger.py             # Logging setup
│       ├── llm_json.py           # Safely extracts JSON from reasoning-model output
│       └── smart_search.py       # Decides whether a founder's question needs a live search
├── requirements.txt
├── .env.example
├── Procfile                       # Railway deployment entry point
├── main.py                        # CLI entry point
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

`DATABASE_URL` is optional for local CLI use. If you're running the API (`src/api.py`) — which is what the Lovable frontend talks to — set this, or a completed evaluation is lost the moment the server restarts, which breaks `/ask` and `/revise` for anything started before that restart. On Railway, attaching a Postgres addon sets this automatically.

## Usage

### CLI (local, one-off runs)

```
python main.py
```

Runs the full evaluation straight through, prints the report, then offers a small menu to ask questions or revise a stage.

### API (for the Lovable frontend, or any web client)

```
uvicorn src.api:app --reload
```

Key endpoints:

| Endpoint | What it does |
|---|---|
| `POST /evaluate/start` | Runs the full pipeline (desirability → viability → feasibility → report) and returns the finished result |
| `GET /evaluate/{id}` | Re-fetch a completed evaluation |
| `POST /evaluate/{id}/ask` | Ask a question about a completed evaluation (auto-searches if needed) |
| `POST /evaluate/{id}/revise` | Redo a specific stage (`desirability`, `viability`, or `feasibility`) |
| `POST /evaluate/{id}/confirm-downstream` | Answer whether to also redo later stages after a revise |

Note: `POST /evaluate/start` doesn't return until the whole pipeline finishes — expect it to take a few minutes given the number of search + LLM calls involved. Design your frontend's loading state accordingly.

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

- Streaming progress updates during the pipeline run (so the frontend can show "researching viability..." rather than a blank wait)
- Export to PDF/docs
- Benchmark comparisons across evaluations
- Historical tracking of revisions over time

## License

MIT

## Support

For issues, questions, or feature requests, open an issue on GitHub.

---

**Built with:** Groq (Qwen 3.6 27B) · LangChain + LangGraph · Tavily Search API · FastAPI · Python
