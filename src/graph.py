"""LangGraph state graph definition with human-driven routing"""
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

# Feedback strings that a founder can enter at a checkpoint to end the
# evaluation there instead of continuing to the next gate. Kept as a small
# set of plain words rather than an LLM judgment call, since the point of
# this routing is to hand the "should we stop?" decision to the human, not
# have the model re-decide it.
_STOP_WORDS = {"stop", "exit", "no", "end", "quit"}


def _human_requested_stop(feedback: str) -> bool:
    """True only if the founder's feedback is explicitly a stop signal."""
    return (feedback or "").strip().lower() in _STOP_WORDS


def build_graph(checkpointer=None, interrupt_before=None):
    """
    Build the LangGraph state graph for startup stress testing.

    Routing is human-driven, not score-driven: every gate proceeds to the
    next phase by default, regardless of score, because a low score on an
    early-stage idea usually reflects missing validation, not a bad idea.
    A phase only ends early if the founder explicitly says to stop at that
    checkpoint (see `_STOP_WORDS`). Scores are still computed and shown in
    the final report so the founder can see where they're weak.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. an
            `InMemorySaver`/`MemorySaver`, or a SQLite/Postgres-backed one).
            When provided, LangGraph persists a snapshot of the state after
            every node under a `thread_id`, which is what makes it possible
            for a run to pause and later be resumed from a completely
            different process invocation -- e.g. a separate HTTP request
            hitting the same running server. Without a checkpointer the graph
            has nothing to resume from, so it can only ever be run start to
            finish in one `invoke()` call.
        interrupt_before: Optional list of node names to pause execution
            before. Pass the `human_review_*` node names here to stop the
            graph right before each human checkpoint. Execution stays paused
            until something calls `graph.invoke(None, config)` again with the
            same `thread_id` -- typically after writing feedback into state
            via `graph.update_state(config, {...})`.

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
    workflow.add_node("generate_report", generate_final_report)

    workflow.set_entry_point("desirability")
    workflow.add_edge("desirability", "human_review_desirability")

    def route_after_human_desirability(state: StartupStressTestState) -> str:
        score = state.get('desirability_score', 0)
        feedback = state.get('desirability_human_feedback', '')
        if _human_requested_stop(feedback):
            logger.info(f"Founder stopped after desirability (score {score}/100).")
            return "generate_report"
        logger.info(f"Desirability score {score}/100. Continuing to viability.")
        return "viability"

    workflow.add_conditional_edges(
        "human_review_desirability",
        route_after_human_desirability,
        {"viability": "viability", "generate_report": "generate_report"}
    )

    workflow.add_edge("viability", "human_review_viability")

    def route_after_human_viability(state: StartupStressTestState) -> str:
        score = state.get('viability_score', 0)
        feedback = state.get('viability_human_feedback', '')
        if _human_requested_stop(feedback):
            logger.info(f"Founder stopped after viability (score {score}/100).")
            return "generate_report"
        logger.info(f"Viability score {score}/100. Continuing to feasibility.")
        return "feasibility"

    workflow.add_conditional_edges(
        "human_review_viability",
        route_after_human_viability,
        {"feasibility": "feasibility", "generate_report": "generate_report"}
    )

    workflow.add_edge("feasibility", "human_review_feasibility")

    def route_after_human_feasibility(state: StartupStressTestState) -> str:
        score = state.get('feasibility_score', 0)
        logger.info(f"Feasibility score {score}/100. Generating report.")
        return "generate_report"

    workflow.add_conditional_edges(
        "human_review_feasibility",
        route_after_human_feasibility,
        {"generate_report": "generate_report"}
    )

    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
