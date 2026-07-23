"""Human review checkpoint nodes"""
from src.state import StartupStressTestState, EvaluationStatus
from src.utils.logger import logger


def human_review_desirability(state: StartupStressTestState) -> StartupStressTestState:
    """Human review checkpoint after desirability analysis."""
    logger.info("Waiting for human review of desirability analysis...")
    
    print("\n" + "="*80)
    print("DESIRABILITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Score: {state['desirability_score']}/100")
    print(f"\nAnalysis:\n{state['desirability_analysis']}")
    print("="*80)
    
    feedback = input("\nProvide your feedback on desirability (or press Enter to continue): ").strip()
    state['desirability_human_feedback'] = feedback or "Approved to proceed"
    state['desirability_status'] = EvaluationStatus.APPROVED
    
    logger.info(f"Human feedback recorded")
    return state


def human_review_viability(state: StartupStressTestState) -> StartupStressTestState:
    """Human review checkpoint after viability analysis."""
    logger.info("Waiting for human review of viability analysis...")
    
    print("\n" + "="*80)
    print("VIABILITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Score: {state['viability_score']}/100")
    print(f"\nAnalysis:\n{state['viability_analysis']}")
    print("="*80)
    
    feedback = input("\nProvide your feedback on viability (or press Enter to continue): ").strip()
    state['viability_human_feedback'] = feedback or "Approved to proceed"
    state['viability_status'] = EvaluationStatus.APPROVED
    
    logger.info(f"Human feedback recorded")
    return state


def human_review_feasibility(state: StartupStressTestState) -> StartupStressTestState:
    """Human review checkpoint after feasibility analysis."""
    logger.info("Waiting for human review of feasibility analysis...")
    
    print("\n" + "="*80)
    print("FEASIBILITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Score: {state['feasibility_score']}/100")
    print(f"\nAnalysis:\n{state['feasibility_analysis']}")
    print("="*80)
    
    feedback = input("\nProvide your feedback on feasibility (or press Enter to continue): ").strip()
    state['feasibility_human_feedback'] = feedback or "Approved to proceed"
    state['feasibility_status'] = EvaluationStatus.APPROVED
    
    logger.info(f"Human feedback recorded")
    return state
