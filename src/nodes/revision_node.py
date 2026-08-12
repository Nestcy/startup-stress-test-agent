"""Interrupt-point node for the revision flow.

This is the ONLY node in the whole graph that the run ever pauses at, and
only during a /revise call -- the normal desirability -> viability ->
feasibility -> report pipeline runs straight through with no pausing at all.
"""
from src.state import StartupStressTestState


def confirm_downstream(state: StartupStressTestState) -> StartupStressTestState:
    """No-op node used purely as an interrupt point.

    The graph pauses here (via interrupt_before) whenever a /revise on an
    earlier stage left later-stage scores in place from a prior run. The
    real branching logic lives in graph.py's route_after_confirm_downstream,
    which runs after this node resumes -- this node just gives LangGraph
    somewhere to pause and wait for `downstream_choice` to be written into
    state (via POST /evaluate/{thread_id}/confirm-downstream) before
    continuing.
    """
    return state
