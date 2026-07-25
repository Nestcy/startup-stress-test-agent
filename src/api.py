"""HTTP API for the startup stress test agent.

Why this works (first principles):

LangGraph's `interrupt_before` pauses execution right before a named node
runs. A `checkpointer` is what makes that pause survive past the current
Python call stack -- it writes a snapshot of the state to storage, keyed by
`thread_id`, after every node. As long as the same checkpointer instance
(and thread_id) is used on a later call, `graph.invoke(None, config)` picks
the run back up exactly where it left off. That's the whole mechanism that
lets a request start an evaluation, return immediately at the first
checkpoint, and let a *different, later* HTTP request supply the human
feedback and carry the run forward.

The `graph` and `checkpointer` below are created once at module import time
(not per-request) precisely so that state persists *across* requests within
this running process.

Persistence: if `DATABASE_URL` is set (Railway's Postgres addon sets this
automatically once attached), state is stored in Postgres, so it survives
restarts/redeploys and works with more than one worker. If unset, this falls
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

from src.graph import build_graph
from src.state import create_initial_state
from src.utils.logger import logger
from src.utils.config import Config

# --- Graph + checkpointer: created once, reused across all requests ---
# `EvaluationStatus` (in src/state.py) is a custom Enum stored in state, so it
# gets written into every checkpoint. Registering it here avoids relying on
# langgraph's permissive-but-deprecated default for unregistered types --
# recent langgraph versions warn that unregistered types will eventually be
# rejected outright, which would break every checkpoint read/write.
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
    interrupt_before=[
        "human_review_desirability",
        "human_review_viability",
        "human_review_feasibility",
        "confirm_downstream",
    ],
)

# Maps the node the graph is currently paused before -> a human-readable stage
NODE_TO_STAGE = {
    "human_review_desirability": "desirability",
    "human_review_viability": "viability",
    "human_review_feasibility": "feasibility",
    "confirm_downstream": "confirm_downstream",
}

# Which state fields hold the analysis/score to show the reviewer at each stage
STAGE_ANALYSIS_FIELDS = {
    "desirability": ("desirability_analysis", "desirability_score"),
    "viability": ("viability_analysis", "viability_score"),
    "feasibility": ("feasibility_analysis", "feasibility_score"),
}

# Which state field a stage's feedback should be written into before resuming
STAGE_FEEDBACK_FIELDS = {
    "desirability": "desirability_human_feedback",
    "viability": "viability_human_feedback",
    "feasibility": "feasibility_human_feedback",
}

# Which node to "pretend just finished" (via update_state(..., as_node=X))
# so that invoke(None, config) re-runs the target stage next. Desirability
# is the entry point, so rewinding to it means pretending we're at START.
REWIND_ANCHOR = {
    "desirability": START,
    "viability": "human_review_desirability",
    "feasibility": "human_review_viability",
}

# Fields to clear on the stage being revised. Downstream stages are
# deliberately left intact -- confirm_downstream asks the founder whether to
# redo them, rather than the revise endpoint silently wiping or silently
# keeping stale data.
STAGE_FIELDS_TO_CLEAR = {
    "desirability": [
        "desirability_analysis", "desirability_score", "desirability_human_feedback",
        "final_report", "overall_score", "recommendation", "downstream_choice",
    ],
    "viability": [
        "viability_analysis", "viability_score", "viability_human_feedback",
        "final_report", "overall_score", "recommendation", "downstream_choice",
    ],
    "feasibility": [
        "feasibility_analysis", "feasibility_score", "feasibility_human_feedback",
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
    description="Human-in-the-loop startup evaluation: desirability -> viability -> feasibility -> report.",
    version="1.0.0",
)

# Lovable (or any browser-based frontend) calls this API cross-origin.
# Set ALLOWED_ORIGINS to a comma-separated list of your Lovable app URL(s) in
# production; defaults to "*" so it works out of the box.
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


class FeedbackRequest(BaseModel):
    feedback: str = Field(default="", description="Reviewer's free-text feedback for the current stage")


class ReviseRequest(BaseModel):
    stage: str = Field(..., description="'desirability', 'viability', or 'feasibility'")
    idea_description: Optional[str] = Field(
        default=None, description="Updated idea description, if the founder is changing it"
    )


class ConfirmDownstreamRequest(BaseModel):
    reevaluate: bool = Field(..., description="True to re-run downstream stages, False to keep existing scores")


class EvaluationResponse(BaseModel):
    thread_id: str
    startup_idea: str
    status: str  # "awaiting_review" | "completed"
    stage: str   # "desirability" | "viability" | "feasibility" | "confirm_downstream" | "completed"
    score: Optional[float] = None
    analysis: Optional[str] = None
    overall_score: Optional[float] = None
    final_report: Optional[str] = None
    recommendation: Optional[str] = None
    message: Optional[str] = None  # copy for the frontend to display as-is; only set for stages with no score/analysis


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
    An empty tuple means the run reached END -- there's nothing more
    authoritative to check than that, so we use it directly rather than
    inferring completion from which fields happen to be populated.
    """
    values = snapshot.values
    startup_idea = values.get("startup_idea", "")

    if snapshot.next:
        current_node = snapshot.next[0]
        stage = NODE_TO_STAGE.get(current_node, current_node)

        if stage == "confirm_downstream":
            source = values.get("_confirm_source", "an earlier stage")
            next_label = "viability and feasibility" if source == "desirability" else "feasibility"
            return EvaluationResponse(
                thread_id=thread_id,
                startup_idea=startup_idea,
                status="awaiting_review",
                stage="confirm_downstream",
                message=(
                    f"You revised {source}. {next_label.capitalize()} still "
                    f"{'has' if next_label == 'feasibility' else 'have'} results from before — "
                    f"re-evaluate {next_label} too, or keep the existing scores?"
                ),
            )

        analysis_field, score_field = STAGE_ANALYSIS_FIELDS.get(stage, (None, None))
        return EvaluationResponse(
            thread_id=thread_id,
            startup_idea=startup_idea,
            status="awaiting_review",
            stage=stage,
            score=values.get(score_field) if score_field else None,
            analysis=values.get(analysis_field) if analysis_field else None,
        )

    return EvaluationResponse(
        thread_id=thread_id,
        startup_idea=startup_idea,
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
    """Kick off an evaluation and run to the first checkpoint (after desirability)."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(
        payload.startup_idea, payload.idea_description, payload.funding_model, payload.arr_target
    )

    logger.info(f"[{thread_id}] Starting evaluation for: {payload.startup_idea}")
    try:
        graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[{thread_id}] Error starting evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting evaluation: {e}")

    snapshot = graph.get_state(config)
    return _response_from_snapshot(thread_id, snapshot)


@app.get("/evaluate/{thread_id}", response_model=EvaluationResponse)
def get_evaluation(thread_id: str):
    """Re-fetch the current checkpoint for an in-progress or completed evaluation."""
    _, snapshot = _get_snapshot_or_404(thread_id)
    return _response_from_snapshot(thread_id, snapshot)


@app.post("/evaluate/{thread_id}/feedback", response_model=EvaluationResponse)
def submit_feedback(thread_id: str, payload: FeedbackRequest):
    """Submit human feedback for whichever stage is currently paused, then resume the graph."""
    config, snapshot = _get_snapshot_or_404(thread_id)

    if not snapshot.next:
        raise HTTPException(
            status_code=400,
            detail="This evaluation has already completed; there is no pending checkpoint to review.",
        )

    current_node = snapshot.next[0]
    stage = NODE_TO_STAGE.get(current_node)
    if stage is None or stage not in STAGE_FEEDBACK_FIELDS:
        raise HTTPException(status_code=409, detail=f"Evaluation is paused at an unexpected node '{current_node}'")

    feedback_field = STAGE_FEEDBACK_FIELDS[stage]
    graph.update_state(config, {feedback_field: payload.feedback})

    logger.info(f"[{thread_id}] Feedback recorded for {stage}, resuming...")
    try:
        graph.invoke(None, config=config)
    except Exception as e:
        logger.error(f"[{thread_id}] Error resuming evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error resuming evaluation: {e}")

    snapshot = graph.get_state(config)
    return _response_from_snapshot(thread_id, snapshot)


@app.post("/evaluate/{thread_id}/confirm-downstream", response_model=EvaluationResponse)
def confirm_downstream_choice(thread_id: str, payload: ConfirmDownstreamRequest):
    """Answer the 'do you want to re-evaluate viability/feasibility too?'
    checkpoint that appears after revising an earlier stage while later
    stages still hold results from a prior run.
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
    """Rewind a completed evaluation back to a given stage and re-run from
    there, keeping the same thread_id so earlier/later stages' analysis and
    scores aren't lost. If later stages still hold results from before, the
    graph pauses at confirm_downstream to ask whether to redo those too.
    """
    config, snapshot = _get_snapshot_or_404(thread_id)

    if payload.stage not in REWIND_ANCHOR:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{payload.stage}'")

    if snapshot.next:
        raise HTTPException(
            status_code=400,
            detail="Evaluation is still in progress; revising only applies once it has a result "
                   "for the target stage. Submit feedback to advance it, or wait for completion.",
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
