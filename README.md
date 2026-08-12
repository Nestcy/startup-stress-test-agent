# Startup Stress Test AI Agent

An AI agent that helps founders pressure-test a startup idea before they spend months building it. Give it an idea and a description, and it runs straight through three questions, gathering real data via web search to ground its answers along the way:

1. **Do people actually want this?** (Desirability)
2. **Does the math work as a business?** (Viability)
3. **Can we actually build and execute it?** (Feasibility)

The founder isn't interrupted mid-run. Once the full report is ready, that's when they step in — reading it, asking questions, and revising whichever stage needs another pass.

## Architecture: plain text, not JSON

Every stage writes a real, readable analysis in plain prose — no structured JSON schema, no per-field parsing. Each stage's write-up is appended to a shared **conversation buffer** (`state['conversation_history']`), and the next stage reads that whole buffer as context before writing its own analysis. That's the entire memory mechanism: a plain, growing list of text, passed along in the LangGraph state — no database, no vector store, nothing to serialize incorrectly.

The only thing pulled out of a response programmatically is a single number, from a line the model is asked to end every analysis with:

```
SCORE: NN/100
```

That's it — one regex, one number. Compare that to a multi-key JSON schema the model has to get exactly right every single call: far less to go wrong, and nothing gets silently dropped or truncated into a broken parse. If the score line is ever missing (a model formatting slip), the system falls back to a neutral 50, not 0 — so a parsing hiccup never looks like a harsh rejection of the idea.

The final report is compiled straight from the conversation buffer into one markdown document — the actual deliverable the founder reads.

## How It Works

**1. You give it an idea.** Just a name and a description.

**2. It runs the full pipeline in one go.** Each stage searches the web for real market data, competitor pricing, or technical references, then writes a grounded analysis ending in a score. Desirability's analysis becomes context for viability; both become context for feasibility.

**3. You get a complete report.** Three scores, three full write-ups, an overall recommendation — as one clean markdown document.

**4. Now you're in control.** From here:
- **Ask questions** — can trigger a live search automatically if the question needs current info
- **Revise a stage** — change your idea description and re-run a specific stage, without losing the others' results

## Why It's Built This Way

**No score-based auto-kill.** A low score on an unvalidated, pre-launch idea usually means the underlying opportunity is genuinely weak — not that the founder hasn't personally run interviews yet. Every prompt is explicit about this: score the opportunity the research actually found, not the founder's execution stage. Every idea evaluated here is pre-launch by definition, so that alone can never be the reason for a low score.

**No pausing mid-run.** The pipeline runs straight through. The founder's judgment matters most once they can see the whole picture, not one incomplete piece at a time.

**No JSON schema per phase.** Structured JSON output sounds more "reliable" on paper, but in practice it's more fragile: a model has to both reason correctly *and* format a multi-key schema perfectly, every call, with no way to gracefully degrade if it's cut off partway through. Plain text degrades gracefully — even a truncated analysis is still readable prose, not an unparseable fragment that silently becomes null.

## Revising a Stage

You can go back and redo any one of the three stages without losing the work already done on the other two. If the stage you're revising has later stages that already have results, the agent asks whether you want to redo those too, or keep them as-is.

## Ask Questions, With Live Search When Needed

Once an evaluation is complete, you can ask the agent anything about it. If your question needs current information the evaluation doesn't already have, it automatically runs a fresh web search before answering.

## Project Structure

```
startup-stress-test-agent/
├── src/
│   ├── __init__.py
│   ├── state.py                  # State schema -- conversation_history IS the memory
│   ├── graph.py                  # Straight-through pipeline + revision routing
│   ├── api.py                    # HTTP API (FastAPI)
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── desirability_node.py  # Plain-text analysis + search, ends with SCORE line
│   │   ├── viability_node.py     # Same, reads desirability from the buffer as context
│   │   ├── feasibility_node.py   # Same, reads desirability + viability as context
│   │   ├── revision_node.py      # The one interrupt point, used only during /revise
│   │   └── report_node.py        # Compiles the buffer into the final markdown report
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search_tool.py        # Tavily search wrapper
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logger.py             # Logging setup
│       ├── llm_json.py           # strip_think() + extract_score() -- no JSON parsing
│       └── smart_search.py       # Decides whether a founder's question needs a live search
├── requirements.txt
├── .env.example
├── Procfile
├── main.py                        # CLI entry point
└── README.md
```

## Installation

```
git clone https://github.com/Nestcy/startup-stress-test-agent.git
cd startup-stress-test-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Environment Variables

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=qwen/qwen3.6-27b
DATABASE_URL=your_postgres_connection_string   # optional but recommended for the API
```

`DATABASE_URL` is optional for local CLI use. If you're running the API, set this, or a completed evaluation is lost the moment the server restarts, which breaks `/ask` and `/revise` for anything started before that restart. On Railway/Render, attaching a Postgres addon usually sets this automatically.

## Usage

### CLI

```
python main.py
```

### API

```
uvicorn src.api:app --reload
```

| Endpoint | What it does |
|---|---|
| `POST /evaluate/start` | Runs the full pipeline and returns the finished result |
| `GET /evaluate/{id}` | Re-fetch a completed evaluation |
| `POST /evaluate/{id}/ask` | Ask a question about a completed evaluation (auto-searches if needed) |
| `POST /evaluate/{id}/revise` | Redo a specific stage |
| `POST /evaluate/{id}/confirm-downstream` | Answer whether to also redo later stages after a revise |

`POST /evaluate/start` doesn't return until the whole pipeline finishes — expect it to take a couple of minutes given the search + LLM calls involved.

## Requirements

- Python 3.10+
- LangChain + LangGraph
- Groq Python Client
- Tavily Search API access
- FastAPI + Uvicorn (for the API)
- python-dotenv, pydantic

## License

MIT
