"""State schema for LangGraph"""
from typing import TypedDict, Optional, List
from enum import Enum


class EvaluationStatus(str, Enum):
    """Status of evaluation at each gate"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class StartupStressTestState(TypedDict):
    """State for startup stress test evaluation.

    The pipeline runs straight through -- desirability -> viability ->
    feasibility -> report -- with each stage's own search calls (see the
    node files) gathering real market/tech data to ground its assumptions,
    rather than pausing for a human at every step. The founder only steps
    in at the end: reviewing the final report, asking questions, and
    revising a specific stage if something needs refining.
    """
    # Input
    startup_idea: str
    idea_description: str

    # Desirability
    desirability_analysis: Optional[str]
    desirability_status: EvaluationStatus
    desirability_score: Optional[float]

    # Viability
    viability_analysis: Optional[str]
    viability_status: EvaluationStatus
    viability_score: Optional[float]
    funding_model: Optional[str]  # "bootstrap" or "vc"; defaults to "bootstrap" if not set
    arr_target: Optional[str]     # e.g. "100k", "1m", "10m", or a raw number as a string; defaults to "1m"

    # Feasibility
    feasibility_analysis: Optional[str]
    feasibility_status: EvaluationStatus
    feasibility_score: Optional[float]

    # Final Report
    final_report: Optional[str]
    overall_score: Optional[float]
    recommendation: Optional[str]

    # Revision flow: set when a /revise on an earlier stage leaves later
    # stages holding results from a prior pass. `_confirm_source` records
    # which stage triggered the confirm_downstream checkpoint, so its
    # routing function knows whether "reevaluate" means go to viability
    # (came from a desirability revise) or feasibility (came from a
    # viability revise). `downstream_choice` holds the founder's answer:
    # "reevaluate" or "keep". This is the ONLY point in the whole graph
    # that ever pauses -- everything else runs straight through.
    _confirm_source: Optional[str]
    downstream_choice: Optional[str]

    # This IS the memory: each stage appends its own plain-text analysis
    # here as {"stage": "desirability"/"viability"/"feasibility", "content": "..."}.
    # The next stage's prompt includes this whole buffer as context, and
    # the final report is compiled straight from it. No separate database
    # or vector store -- it's a plain growing list of text, passed along
    # in the LangGraph state.
    search_results: Optional[List[dict]]
    conversation_history: List[dict]
    errors: List[str]


def create_initial_state(
    startup_idea: str,
    idea_description: str,
    funding_model: Optional[str] = None,
    arr_target: Optional[str] = None,
) -> StartupStressTestState:
    """Build a fresh state dict for a new evaluation.

    Single source of truth for what an "empty" evaluation looks like. Both
    the CLI (main.py) and the HTTP API (src/api.py) start a run by calling
    this, so a new state field only ever needs to be added in one place.

    `funding_model`/`arr_target` are optional; viability_node.py defaults
    them to "bootstrap" / "1m" if left as None.
    """
    return {
        "startup_idea": startup_idea,
        "idea_description": idea_description,
        "desirability_analysis": None,
        "desirability_status": EvaluationStatus.PENDING,
        "desirability_score": None,
        "viability_analysis": None,
        "viability_status": EvaluationStatus.PENDING,
        "viability_score": None,
        "funding_model": funding_model,
        "arr_target": arr_target,
        "feasibility_analysis": None,
        "feasibility_status": EvaluationStatus.PENDING,
        "feasibility_score": None,
        "final_report": None,
        "overall_score": None,
        "recommendation": None,
        "_confirm_source": None,
        "downstream_choice": None,
        "search_results": None,
        "conversation_history": [],
        "errors": [],
    }
