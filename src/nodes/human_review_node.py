"""Human review checkpoint nodes.

These nodes no longer block on input(). Each one runs *after* the graph has
already been resumed from an `interrupt_before` pause -- by that point,
whoever resumed the run (the CLI loop or an API request handler) has written
the reviewer's feedback into state via `graph.update_state(...)`. All a node
does now is fill in a default if no feedback was given and mark the gate
approved. The graph's conditional edges then decide where to go next based
on that feedback (see graph.py) -- not based on scores, since scores on an
early-stage, unvalidated idea reflect confidence in a guess, not a verdict
on the idea.
"""
from src.state import StartupStressTestState, EvaluationStatus
from src.utils.logger import logger


def human_review_desirability(state: StartupStressTestState) -> StartupStressTestState:
    """Finalize human feedback for the desirability checkpoint."""
    logger.info("Finalizing human review of desirability analysis...")

    state['desirability_human_feedback'] = state.get('desirability_human_feedback') or "Approved to proceed"
    state['desirability_status'] = EvaluationStatus.APPROVED

    logger.info("Human feedback recorded")
    return state


def human_review_viability(state: StartupStressTestState) -> StartupStressTestState:
    """Finalize human feedback for the viability checkpoint."""
    logger.info("Finalizing human review of viability analysis...")

    state['viability_human_feedback'] = state.get('viability_human_feedback') or "Approved to proceed"
    state['viability_status'] = EvaluationStatus.APPROVED

    logger.info("Human feedback recorded")
    return state


def human_review_feasibility(state: StartupStressTestState) -> StartupStressTestState:
    """Finalize human feedback for the feasibility checkpoint."""
    logger.info("Finalizing human review of feasibility analysis...")

    state['feasibility_human_feedback'] = state.get('feasibility_human_feedback') or "Approved to proceed"
    state['feasibility_status'] = EvaluationStatus.APPROVED

    logger.info("Human feedback recorded")
    return state


def confirm_downstream(state: StartupStressTestState) -> StartupStressTestState:
    """No-op node used purely as an interrupt point.

    The graph pauses here (via interrupt_before) whenever a /revise on an
    earlier stage left later-stage scores in place from a prior run. All the
    real branching logic lives in graph.py's route_after_confirm_downstream,
    which runs after this node resumes -- this node just gives LangGraph
    somewhere to pause and wait for `downstream_choice` to be written into
    state (via POST /evaluate/{thread_id}/confirm-downstream) before
    continuing.
    """
    return state
