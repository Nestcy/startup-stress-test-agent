"""LangGraph state graph definition with human-driven routing"""
from langgraph.graph import StateGraph, END, START
from src.state import StartupStressTestState
from src.nodes.desirability_node import desirability_node
from src.nodes.viability_node import viability_node
from src.nodes.feasibility_node import feasibility_node
from src.nodes.human_review_node import (
    human_review_desirability,
    human_review_viability,
    human_review_feasibility,
    confirm_downstream,
)
from src.nodes.report_node import generate_final_report
from src.utils.logger import logger

# Feedback that should end the evaluation at the current checkpoint instead
# of continuing to the next gate. Matched against the start of the founder's
# feedback (not the whole string), so phrases like "no thanks" or "stop here,
# I want to think about this" still count as a stop -- but feedback that just
# happens to be a critique, e.g. "no clear differentiation from competitors",
# doesn't accidentally match. Bare "no" is deliberately excluded since it's
# ambiguous between "no, stop" and "no, I disagree with this analysis".
_STOP_PHRASES = (
    "stop",
    "exit",
    "no thanks",
    "no thank you",
    "not now",
    "end here",
    "quit",
    "cancel",
)


def _human_requested_stop(feedback: str) -> bool:
    """True if the founder's feedback signals they want to stop here."""
    normalized = (feedback or "").strip().lower()
    return any(normalized.startswith(phrase) for phrase in _STOP_PHRASES)


def build_graph(checkpointer=None, interrupt_before=None):
    """
    Build the LangGraph state graph for startup stress testing.

    Routing is human-driven, not score-driven: every gate proceeds to the
    next phase by default, regardless of score, because a low score on an
    early-stage idea usually reflects missing validation, not a bad idea.
    A phase only ends early if the founder explicitly says to stop at that
    checkpoint (see `_STOP_PHRASES`). Scores are still computed by each node
    and shown in the final report so the founder can see where they're weak.

    Revision flow: if a founder revises an earlier stage (via POST
    /evaluate/{thread_id}/revise) after later stages already have results,
    the graph pauses at `confirm_downstream` to ask whether to re-run those
    later stages too, or keep the existing scores. That choice is read from
    `downstream_choice` in state.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. an
            `InMemorySaver`/`MemorySaver`, or a Postgres-backed one). When
            provided, LangGraph persists a snapshot of the state after every
            node under a `thread_id`, which is what makes it possible for a
            run to pause and later be resumed from a completely different
            process invocation -- e.g. a separate HTTP request hitting the
            same running server. Without a checkpointer the graph has
            nothing to resume from, so it can only ever be run start to
            finish in one `invoke()` call.
        interrupt_before: Optional list of node names to pause execution
            before. Pass the `human_review_*` and `confirm_downstream` node
            names here to stop the graph right before each checkpoint.
            Execution stays paused until something calls
            `graph.invoke(None, config)` again with the same `thread_id` --
            typically after writing feedback into state via
            `graph.update_state(config, {...})`.

    Both arguments default to None, which reproduces the original behavior:
    a single graph that runs straight through with no pausing.
    """
    workflow = StateGraph(StartupStressTestState)

    # Add nodes
    workflow.add_node("desirability", desirability_node)
    workflow.add_node("human_review_desirability", human_review_desirability)
    workflow.add_node("viability", viability_node)
    workflow.add_node("human_review_viability", human_review_viability)
    workflow.add_node("feasibility", feasibility_node)
    workflow.add_node("human_review_feasibility", human_review_feasibility)
    workflow.add_node("confirm_downstream", confirm_downstream)
    workflow.add_node("generate_report", generate_final_report)

    workflow.set_entry_point("desirability")
    workflow.add_edge("desirability", "human_review_desirability")

    def route_after_human_desirability(state: StartupStressTestState) -> str:
        feedback = state.get('desirability_human_feedback', '')
        if _human_requested_stop(feedback):
            logger.info("Founder stopped after desirability.")
            return "generate_report"

        # A populated viability_score at this point only happens after a
        # /revise on desirability -- a first-time run never reaches here
        # with viability already scored. That's the signal to ask, rather
        # than silently overwrite or silently keep stale downstream data.
        if state.get('viability_score') is not None:
            logger.info("Stale viability/feasibility data found. Asking founder whether to re-evaluate.")
            state['_confirm_source'] = 'desirability'
            state['downstream_choice'] = None
            return "confirm_downstream"

        logger.info("Desirability complete. Continuing to viability.")
        return "viability"

    workflow.add_conditional_edges(
        "human_review_desirability",
        route_after_human_desirability,
        {
            "viability": "viability",
            "confirm_downstream": "confirm_downstream",
            "generate_report": "generate_report",
        }
    )

    def route_after_confirm_downstream(state: StartupStressTestState) -> str:
        choice = (state.get('downstream_choice') or '').strip().lower()
        source = state.get('_confirm_source')

        if choice == "reevaluate":
            next_stage = "viability" if source == "desirability" else "feasibility"
            logger.info(f"Founder chose to re-evaluate. Continuing to {next_stage}.")
            return next_stage

        logger.info("Founder chose to keep existing downstream scores. Skipping to report.")
        return "generate_report"

    workflow.add_conditional_edges(
        "confirm_downstream",
        route_after_confirm_downstream,
        {"viability": "viability", "feasibility": "feasibility", "generate_report": "generate_report"}
    )

    workflow.add_edge("viability", "human_review_viability")

    def route_after_human_viability(state: StartupStressTestState) -> str:
        feedback = state.get('viability_human_feedback', '')
        if _human_requested_stop(feedback):
            logger.info("Founder stopped after viability.")
            return "generate_report"

        # Same logic as desirability: a populated feasibility_score here
        # only happens after a /revise on viability.
        if state.get('feasibility_score') is not None:
            logger.info("Stale feasibility data found. Asking founder whether to re-evaluate.")
            state['_confirm_source'] = 'viability'
            state['downstream_choice'] = None
            return "confirm_downstream"

        logger.info("Viability complete. Continuing to feasibility.")
        return "feasibility"

    workflow.add_conditional_edges(
        "human_review_viability",
        route_after_human_viability,
        {
            "feasibility": "feasibility",
            "confirm_downstream": "confirm_downstream",
            "generate_report": "generate_report",
        }
    )

    workflow.add_edge("feasibility", "human_review_feasibility")
    workflow.add_edge("human_review_feasibility", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
