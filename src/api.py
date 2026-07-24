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

Caveat: `InMemorySaver` keeps every paused run's state in this process's RAM.
That's fine for local dev and single-instance deployments, but:
  - a server restart or redeploy loses every in-flight (not yet completed)
    evaluation.
  - it will NOT work correctly if you run more than one instance/worker
    behind a load balancer, since a `/feedback` request could land on a
    process that never saw the `/start` request for that thread_id.
For a Railway deployment in particular, keep this to a single instance
(one worker, no autoscaling) unless you swap `InMemorySaver` for a
persistent checkpointer (e.g. `langgraph-checkpoint-postgres` pointed at a
Railway Postgres addon) -- `build_graph()` accepts any checkpointer, so
that's a one-line change here, not a redesign.
"""
import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.graph import build_graph
from src.state import create_initial_state
from src.utils.logger import logger

# --- Graph + checkpointer: created once, reused across all requests ---
# `EvaluationStatus` (in src/state.py) is a custom Enum stored in state, so it
# gets written into every checkpoint. Registering it here avoids relying on
# langgraph's permissive-but-deprecated default for unregistered types --
# recent langgraph versions warn that unregistered types will eventually be
# rejected outright, which would break every checkpoint read/write.
_serde = JsonPlusSerializer(allowed_msgpack_modules=[("src.state", "EvaluationStatus")])
checkpointer = InMemorySaver(serde=_serde)
graph = build_graph(
    checkpointer=checkpointer,
    interrupt_before=[
        "human_review_desirability",
        "human_review_viability",
        "human_review_feasibility",
    ],
)

# Maps the node the graph is currently paused before -> a human-readable stage
NODE_TO_STAGE = {
    "human_review_desirability": "desirability",
    "human_review_viability": "viability",
    "human_review_feasibility": "feasibility",
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


class EvaluationResponse(BaseModel):
    thread_id: str
    startup_idea: str
    status: str  # "awaiting_review" | "completed"
    stage: str   # "desirability" | "viability" | "feasibility" | "completed"
    score: Optional[float] = None
    analysis: Optional[str] = None
    overall_score: Optional[float] = None
    final_report: Optional[str] = None
    recommendation: Optional[str] = None


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
    if stage is None:
        # Defensive: only happens if the graph is paused at a node we didn't expect.
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
