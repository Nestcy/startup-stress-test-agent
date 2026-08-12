"""LangGraph state graph definition.

The pipeline runs straight through in a single invoke() call: desirability
-> viability -> feasibility -> report. Each analysis node does its own web
research (see src/nodes/desirability_node.py etc.) to ground its scoring in
real data rather than pure assumption, but nothing pauses for a human
mid-run. The founder only steps in once the report is done -- reviewing it,
asking questions (POST /ask), and revising a specific stage if something
needs refining (POST /revise).

The only place this graph ever pauses is `confirm_downstream`, and only as
a side effect of a /revise call: if revising an earlier stage leaves a
later stage holding results from the prior run, the graph asks whether to
redo that later stage too, or keep it as-is, rather than silently doing
either.
"""
from langgraph.graph import StateGraph, END, START
from src.state import StartupStressTestState
from src.nodes.desirability_node import desirability_node
from src.nodes.viability_node import viability_node
from src.nodes.feasibility_node import feasibility_node
from src.nodes.revision_node import confirm_downstream
from src.nodes.report_node import generate_final_report
from src.utils.logger import logger


def build_graph(checkpointer=None, interrupt_before=None):
    """
    Build the LangGraph state graph for startup stress testing.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. an
            `InMemorySaver`, or a Postgres-backed one). A normal run never
            pauses, so a checkpointer isn't needed for that -- but it IS
            needed for `/revise` and `/ask` to work, since those are
            separate HTTP requests made *after* the run completes, and need
            to look up the finished state by `thread_id`. Without a
            checkpointer, state only exists for the duration of a single
            `invoke()` call and can't be resumed or read afterward.
        interrupt_before: Optional list of node names to pause execution
            before. In this graph that should just be `["confirm_downstream"]`
            -- the one point where a /revise might need the founder's input
            before continuing.

    Both arguments default to None, which reproduces the simplest possible
    behavior: one straight run, no persistence, no pausing.
    """
    workflow = StateGraph(StartupStressTestState)

    workflow.add_node("desirability", desirability_node)
    workflow.add_node("viability", viability_node)
    workflow.add_node("feasibility", feasibility_node)
    workflow.add_node("confirm_downstream", confirm_downstream)
    workflow.add_node("generate_report", generate_final_report)

    workflow.set_entry_point("desirability")

    def route_after_desirability(state: StartupStressTestState) -> str:
        # A populated viability_score at this point only happens after a
        # /revise on desirability -- a first-time run never reaches here
        # with viability already scored. That's the signal to ask the
        # founder whether to redo the later stages too, rather than
        # silently overwriting or silently keeping stale downstream data.
        if state.get('viability_score') is not None:
            logger.info("Stale viability/feasibility data found. Asking founder whether to re-evaluate.")
            state['_confirm_source'] = 'desirability'
            state['downstream_choice'] = None
            return "confirm_downstream"
        return "viability"

    workflow.add_conditional_edges(
        "desirability",
        route_after_desirability,
        {"viability": "viability", "confirm_downstream": "confirm_downstream"}
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

    def route_after_viability(state: StartupStressTestState) -> str:
        # Same logic as desirability: a populated feasibility_score here
        # only happens after a /revise on viability.
        if state.get('feasibility_score') is not None:
            logger.info("Stale feasibility data found. Asking founder whether to re-evaluate.")
            state['_confirm_source'] = 'viability'
            state['downstream_choice'] = None
            return "confirm_downstream"
        return "feasibility"

    workflow.add_conditional_edges(
        "viability",
        route_after_viability,
        {"feasibility": "feasibility", "confirm_downstream": "confirm_downstream"}
    )

    # Feasibility is always last -- no further stage to check for staleness.
    workflow.add_edge("feasibility", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
