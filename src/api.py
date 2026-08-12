"""HTTP API for the startup stress test agent.

Flow: POST /evaluate/start runs the entire pipeline -- desirability ->
viability -> feasibility -> report -- in one shot and returns the finished
report. There's no mid-run pausing; each stage node does its own web
research to ground its scoring in real data (see the node files), so the
founder doesn't need to feed it anything beyond the initial idea and
description.

The founder interacts with a finished evaluation two ways:
- POST /evaluate/{id}/ask -- ask questions about the result. Can trigger a
  live web search if the question needs current info (see smart_search.py).
- POST /evaluate/{id}/revise -- redo a specific stage with updated input,
  keeping the same thread_id so the other stages' results aren't lost. If
  later stages already have results from the prior run, the graph pauses
  at confirm_downstream to ask whether to redo those too.

Persistence: if `DATABASE_URL` is set (Railway's Postgres addon sets this
automatically once attached), state is stored in Postgres, so a completed
evaluation's thread_id can still be looked up (for /ask, /revise) after a
restart/redeploy, and works with more than one worker. If unset, this falls
back to `InMemorySaver` for local dev only -- state is lost on restart, and
it's only safe with a single worker.
"""
import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.graph import START
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.graph import build_graph
from src.state import create_initial_state
from src.utils.logger import logger
from src.utils.config import Config
from src.utils.smart_search import maybe_search

# --- Graph + checkpointer: created once, reused across all requests ---
_serde = JsonPlusSerializer(allowed_msgpack_modules=[("src.state", "EvaluationStatus")])

if Config.DATABASE_URL:
    from langgraph.checkpoint.postgres import PostgresSaver
    _pg_cm = PostgresSaver.from_conn_string(Config.DATABASE_URL, serde=_serde)
    checkpointer = _pg_cm.__enter__()  # keep the connection open for the app's lifetime
    checkpointer.setup()  # creates checkpoint tables if they don't exist yet; safe to call every boot
    logger.info("Using PostgresSaver for checkpoint storage.")
else:
    from langgraph.checkpoint.memory import InMemorySaver
    logger.warning("DATABASE_URL not set -- using InMemorySaver. State will NOT survive a restart.")
    checkpointer = InMemorySaver(serde=_serde)

graph = build_graph(
    checkpointer=checkpointer,
    interrupt_before=["confirm_downstream"],
)

# Which node to "pretend just finished" (via update_state(..., as_node=X))
# so that invoke(None, config) re-runs the target stage next. Desirability
# is the entry point, so rewinding to it means pretending we're at START.
REWIND_ANCHOR = {
    "desirability": START,
    "viability": "desirability",
    "feasibility": "viability",
}

# Fields to clear on the stage being revised. Downstream stages are
# deliberately left intact -- confirm_downstream asks the founder whether to
# redo them, rather than the revise endpoint silently wiping or silently
# keeping stale data.
STAGE_FIELDS_TO_CLEAR = {
    "desirability": [
        "desirability_analysis", "desirability_score",
        "final_report", "overall_score", "recommendation", "downstream_choice",
    ],
    "viability": [
        "viability_analysis", "viability_score",
        "final_report", "overall_score", "recommendation", "downstream_choice",
    ],
    "feasibility": [
        "feasibility_analysis", "feasibility_score",
        "final_report", "overall_score", "recommendation",
    ],
}

# Which prior-stage fields must already be populated before a given stage
# can be revised -- you can't revise viability on a thread that never
# completed desirability, since there'd be nothing to rewind to.
STAGE_PREREQUISITES = {
    "desirability": [],
    "viability": ["desirability_score"],
    "feasibility": ["desirability_score", "viability_score"],
}


app = FastAPI(
    title="Startup Stress Test Agent API",
    description="Search-grounded startup evaluation: desirability -> viability -> feasibility -> report, "
                "then founder review, revision, and Q&A.",
    version="2.0.0",
)

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/response schemas ---

class StartRequest(BaseModel):
    startup_idea: str = Field(..., min_length=1, description="Short name of the startup idea")
    idea_description: str = Field(..., min_length=1, description="Detailed description of the idea")
    funding_model: Optional[str] = Field(
        default=None, description="'bootstrap' or 'vc'. Defaults to 'bootstrap' if omitted."
    )
    arr_target: Optional[str] = Field(
        default=None, description="'100k', '1m', '10m', or a custom number. Defaults to '1m' if omitted."
    )


class ReviseRequest(BaseModel):
    stage: str = Field(..., description="'desirability', 'viability', or 'feasibility'")
    idea_description: Optional[str] = Field(
        default=None, description="Updated idea description, if the founder is changing it"
    )


class ConfirmDownstreamRequest(BaseModel):
    reevaluate: bool = Field(..., description="True to re-run downstream stages, False to keep existing scores")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Founder's question about the evaluation")


class AskResponse(BaseModel):
    thread_id: str
    question: str
    answer: str


class EvaluationResponse(BaseModel):
    thread_id: str
    startup_idea: str
    status: str  # "running" | "awaiting_review" | "completed"
    stage: str   # "confirm_downstream" | "completed"
    desirability_score: Optional[float] = None
    desirability_analysis: Optional[str] = None
    viability_score: Optional[float] = None
    viability_analysis: Optional[str] = None
    feasibility_score: Optional[float] = None
    feasibility_analysis: Optional[str] = None
    overall_score: Optional[float] = None
    final_report: Optional[str] = None
    recommendation: Optional[str] = None
    message: Optional[str] = None  # only set for confirm_downstream, copy for the frontend to display as-is


# --- Helpers ---

def _get_snapshot_or_404(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No evaluation found for thread_id '{thread_id}'")
    return config, snapshot


def _response_from_snapshot(thread_id: str, snapshot) -> EvaluationResponse:
    """Build the API response from wherever the graph currently is.

    `snapshot.next` is LangGraph's own record of which node(s) will run next.
    In this graph that's only ever non-empty if the run is paused at
    confirm_downstream (which only happens mid-/revise) -- otherwise the run
    has either not been started, or has already reached END.
    """
    values = snapshot.values
    startup_idea = values.get("startup_idea", "")

    base_fields = dict(
        thread_id=thread_id,
        startup_idea=startup_idea,
        desirability_score=values.get("desirability_score"),
        desirability_analysis=values.get("desirability_analysis"),
        viability_score=values.get("viability_score"),
        viability_analysis=values.get("viability_analysis"),
        feasibility_score=values.get("feasibility_score"),
        feasibility_analysis=values.get("feasibility_analysis"),
    )

    if snapshot.next and snapshot.next[0] == "confirm_downstream":
        source = values.get("_confirm_source", "an earlier stage")
        next_label = "viability and feasibility" if source == "desirability" else "feasibility"
        return EvaluationResponse(
            **base_fields,
            status="awaiting_review",
            stage="confirm_downstream",
            message=(
                f"You revised {source}. {next_label.capitalize()} still "
                f"{'has' if next_label == 'feasibility' else 'have'} results from before — "
                f"re-evaluate {next_label} too, or keep the existing scores?"
            ),
        )

    return EvaluationResponse(
        **base_fields,
        status="completed",
        stage="completed",
        overall_score=values.get("overall_score"),
        final_report=values.get("final_report"),
        recommendation=values.get("recommendation"),
    )


# --- Endpoints ---

@app.get("/health")
def health():
    """Simple liveness check for Railway (or any host) to poll."""
    return {"status": "ok"}


@app.post("/evaluate/start", response_model=EvaluationResponse)
def start_evaluation(payload: StartRequest):
    """Run the full evaluation pipeline in one shot: desirability ->
    viability -> feasibility -> report. Each stage node does its own web
    research to ground its assumptions, so this call can take a while
    (multiple LLM calls and searches per stage) -- the response only comes
    back once the whole pipeline has finished.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(
        payload.startup_idea, payload.idea_description, payload.funding_model, payload.arr_target
    )

    logger.info(f"[{thread_id}] Starting full evaluation for: {payload.startup_idea}")
    try:
        graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[{thread_id}] Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {e}")

    snapshot = graph.get_state(config)
    return _response_from_snapshot(thread_id, snapshot)


@app.get("/evaluate/{thread_id}", response_model=EvaluationResponse)
def get_evaluation(thread_id: str):
    """Re-fetch a completed (or, rarely, mid-revise) evaluation."""
    _, snapshot = _get_snapshot_or_404(thread_id)
    return _response_from_snapshot(thread_id, snapshot)


@app.post("/evaluate/{thread_id}/confirm-downstream", response_model=EvaluationResponse)
def confirm_downstream_choice(thread_id: str, payload: ConfirmDownstreamRequest):
    """Answer the 'do you want to re-evaluate viability/feasibility too?'
    checkpoint that appears after revising an earlier stage while later
    stages still hold results from a prior run. This is the only endpoint
    in the whole API that resumes a mid-flight graph -- every other
    interaction happens on a completed evaluation.
    """
    config, snapshot = _get_snapshot_or_404(thread_id)

    if not snapshot.next or snapshot.next[0] != "confirm_downstream":
        raise HTTPException(status_code=400, detail="No pending downstream confirmation for this evaluation.")

    choice = "reevaluate" if payload.reevaluate else "keep"
    graph.update_state(config, {"downstream_choice": choice})

    logger.info(f"[{thread_id}] Downstream choice recorded: {choice}, resuming...")
    try:
        graph.invoke(None, config=config)
    except Exception as e:
        logger.error(f"[{thread_id}] Error resuming after downstream confirmation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    snapshot = graph.get_state(config)
    return _response_from_snapshot(thread_id, snapshot)


@app.post("/evaluate/{thread_id}/revise", response_model=EvaluationResponse)
def revise_evaluation(thread_id: str, payload: ReviseRequest):
    """Rewind a completed evaluation back to a given stage and re-run it,
    keeping the same thread_id so the other stages' results aren't lost. If
    later stages still hold results from before, the graph pauses at
    confirm_downstream to ask whether to redo those too.
    """
    config, snapshot = _get_snapshot_or_404(thread_id)

    if payload.stage not in REWIND_ANCHOR:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{payload.stage}'")

    if snapshot.next:
        raise HTTPException(
            status_code=400,
            detail="Evaluation is paused on a pending downstream confirmation; resolve that first "
                   "via /confirm-downstream before revising.",
        )

    missing = [f for f in STAGE_PREREQUISITES[payload.stage] if snapshot.values.get(f) is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot revise '{payload.stage}': a prior required stage never completed. "
                    f"Missing: {missing}",
        )

    values_to_clear = {field: None for field in STAGE_FIELDS_TO_CLEAR[payload.stage]}
    if payload.idea_description:
        values_to_clear["idea_description"] = payload.idea_description

    anchor = REWIND_ANCHOR[payload.stage]
    logger.info(f"[{thread_id}] Revising from '{payload.stage}', rewinding to anchor '{anchor}'")

    graph.update_state(config, values_to_clear, as_node=anchor)

    try:
        graph.invoke(None, config=config)
    except Exception as e:
        logger.error(f"[{thread_id}] Error during revision: {e}")
        raise HTTPException(status_code=500, detail=f"Error during revision: {e}")

    snapshot = graph.get_state(config)
    return _response_from_snapshot(thread_id, snapshot)


_ASK_PROMPT = ChatPromptTemplate.from_template("""
You are answering a founder's question about a startup evaluation that has
already been completed. Use the information below as your primary source --
don't invent scores, facts, or analysis that isn't here. If fresh web
search results are provided below, prioritize the most current information
in your answer and say so.

Startup Idea: {startup_idea}
Idea Description: {idea_description}

Desirability ({desirability_score}/100):
{desirability_analysis}

Viability ({viability_score}/100):
{viability_analysis}

Feasibility ({feasibility_score}/100):
{feasibility_analysis}

Overall Score: {overall_score}/100
Recommendation: {recommendation}

Final Report:
{final_report}

{search_note}

Founder's question: {question}

Answer directly and concretely. Keep it focused -- a few sentences to a
short paragraph, not a restatement of the whole report.
""")


@app.post("/evaluate/{thread_id}/ask", response_model=AskResponse)
def ask_followup(thread_id: str, payload: AskRequest):
    """Answer a founder's question about a completed evaluation.

    If the question needs current, real-world information the evaluation
    doesn't already have (e.g. "what's Competitor X charging right now?"),
    this automatically runs a live web search and folds the results into
    the answer -- see src/utils/smart_search.py for how that decision gets
    made. Otherwise it just answers from the evaluation's own analysis.
    """
    _, snapshot = _get_snapshot_or_404(thread_id)

    if snapshot.next:
        raise HTTPException(
            status_code=400,
            detail="Evaluation has a pending downstream confirmation; resolve that first via /confirm-downstream.",
        )

    values = snapshot.values
    used_search, search_context = maybe_search(payload.question)

    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL, temperature=0.5)
    chain = _ASK_PROMPT | llm

    logger.info(f"[{thread_id}] Follow-up question: {payload.question}")
    try:
        response = chain.invoke({
            "startup_idea": values.get("startup_idea", ""),
            "idea_description": values.get("idea_description", ""),
            "desirability_score": values.get("desirability_score", 0),
            "desirability_analysis": values.get("desirability_analysis") or "Not available",
            "viability_score": values.get("viability_score", 0),
            "viability_analysis": values.get("viability_analysis") or "Not available",
            "feasibility_score": values.get("feasibility_score", 0),
            "feasibility_analysis": values.get("feasibility_analysis") or "Not available",
            "overall_score": values.get("overall_score", 0),
            "recommendation": values.get("recommendation", ""),
            "final_report": values.get("final_report", ""),
            "search_note": f"Recent web search results relevant to this question:\n{search_context}" if used_search else "",
            "question": payload.question,
        })
    except Exception as e:
        logger.error(f"[{thread_id}] Error answering follow-up: {e}")
        raise HTTPException(status_code=500, detail=f"Error answering follow-up: {e}")

    answer = response.content
    if used_search:
        answer += "\n\n_(I looked this up just now to answer that.)_"

    return AskResponse(thread_id=thread_id, question=payload.question, answer=answer)
