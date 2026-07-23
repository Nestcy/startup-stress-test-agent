"""LangGraph state graph definition with conditional edges"""
from langgraph.graph import StateGraph, END
from src.state import StartupStressTestState
from src.nodes.desirability_node import desirability_node
from src.nodes.viability_node import viability_node
from src.nodes.feasibility_node import feasibility_node
from src.nodes.human_review_node import (
    human_review_desirability,
    human_review_viability,
    human_review_feasibility
)
from src.nodes.report_node import generate_final_report
from src.utils.logger import logger


def build_graph():
    """
    Build the LangGraph state graph for startup stress testing.
    
    Conditional edges enable early exit for non-viable ideas.
    """
    workflow = StateGraph(StartupStressTestState)
    
    # Add nodes
    workflow.add_node("desirability", desirability_node)
    workflow.add_node("human_review_desirability", human_review_desirability)
    workflow.add_node("viability", viability_node)
    workflow.add_node("human_review_viability", human_review_viability)
    workflow.add_node("feasibility", feasibility_node)
    workflow.add_node("human_review_feasibility", human_review_feasibility)
    workflow.add_node("generate_report", generate_final_report)
    
    workflow.set_entry_point("desirability")
    workflow.add_edge("desirability", "human_review_desirability")
    
    def route_after_human_desirability(state: StartupStressTestState) -> str:
        score = state.get('desirability_score', 0)
        if score < 30:
            logger.warning(f"Desirability score too low ({score}/100). Exiting early.")
            return "generate_report"
        logger.info(f"Desirability score acceptable ({score}/100). Continuing.")
        return "viability"
    
    workflow.add_conditional_edges(
        "human_review_desirability",
        route_after_human_desirability,
        {"viability": "viability", "generate_report": "generate_report"}
    )
    
    workflow.add_edge("viability", "human_review_viability")
    
    def route_after_human_viability(state: StartupStressTestState) -> str:
        viability_score = state.get('viability_score', 0)
        desirability_score = state.get('desirability_score', 0)
        
        if viability_score < 30:
            logger.warning(f"Viability score too low ({viability_score}/100). Exiting.")
            return "generate_report"
        
        if desirability_score < 40 and viability_score < 60:
            logger.warning(f"Combined weakness. Exiting.")
            return "generate_report"
        
        logger.info(f"Viability acceptable ({viability_score}/100). Continuing.")
        return "feasibility"
    
    workflow.add_conditional_edges(
        "human_review_viability",
        route_after_human_viability,
        {"feasibility": "feasibility", "generate_report": "generate_report"}
    )
    
    workflow.add_edge("feasibility", "human_review_feasibility")
    
    def route_after_human_feasibility(state: StartupStressTestState) -> str:
        feasibility_score = state.get('feasibility_score', 0)
        viability_score = state.get('viability_score', 0)
        desirability_score = state.get('desirability_score', 0)
        
        if feasibility_score < 25:
            logger.warning(f"Feasibility too low ({feasibility_score}/100). Cannot build.")
            return "generate_report"
        
        if feasibility_score < 40 and (viability_score < 50 or desirability_score < 50):
            logger.warning(f"Combined weakness. Exiting.")
            return "generate_report"
        
        logger.info(f"Feasibility acceptable ({feasibility_score}/100). Generating report.")
        return "generate_report"
    
    workflow.add_conditional_edges(
        "human_review_feasibility",
        route_after_human_feasibility,
        {"generate_report": "generate_report"}
    )
    
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()
