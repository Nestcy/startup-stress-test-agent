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
