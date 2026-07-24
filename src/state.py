"""State schema for LangGraph"""
from typing import TypedDict, Optional, List
from enum import Enum


class EvaluationStatus(str, Enum):
    """Status of evaluation at each gate"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class StartupStressTestState(TypedDict):
    """State for startup stress test evaluation"""
    # Input
    startup_idea: str
    idea_description: str
    
    # Desirability Gate
    desirability_analysis: Optional[str]
    desirability_status: EvaluationStatus
    desirability_score: Optional[float]
    desirability_human_feedback: Optional[str]
    
    # Viability Gate
    viability_analysis: Optional[str]
    viability_status: EvaluationStatus
    viability_score: Optional[float]
    viability_human_feedback: Optional[str]
    funding_model: Optional[str]  # "bootstrap" or "vc"; defaults to "bootstrap" if not set
    arr_target: Optional[str]     # e.g. "100k", "1m", "10m", or a raw number as a string; defaults to "1m"
    
    # Feasibility Gate
    feasibility_analysis: Optional[str]
    feasibility_status: EvaluationStatus
    feasibility_score: Optional[float]
    feasibility_human_feedback: Optional[str]
    
    # Final Report
    final_report: Optional[str]
    overall_score: Optional[float]
    recommendation: Optional[str]
    
    # Metadata
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
        "desirability_human_feedback": None,
        "viability_analysis": None,
        "viability_status": EvaluationStatus.PENDING,
        "viability_score": None,
        "viability_human_feedback": None,
        "funding_model": funding_model,
        "arr_target": arr_target,
        "feasibility_analysis": None,
        "feasibility_status": EvaluationStatus.PENDING,
        "feasibility_score": None,
        "feasibility_human_feedback": None,
        "final_report": None,
        "overall_score": None,
        "recommendation": None,
        "search_results": None,
        "conversation_history": [],
        "errors": []
    }
