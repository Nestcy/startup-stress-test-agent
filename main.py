#!/usr/bin/env python3
"""Main entry point for startup stress test AI agent (CLI)."""
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.graph import build_graph
from src.state import create_initial_state
from src.utils.logger import logger

ALL_INTERRUPT_NODES = [
    "await_intake_response",
    "human_review_desirability",
    "human_review_viability",
    "human_review_feasibility",
    "confirm_downstream",
]

# Maps the node the graph is paused before -> which state fields to show/fill
STAGE_INFO = {
    "human_review_desirability": ("desirability", "desirability_analysis", "desirability_score", "desirability_human_feedback"),
    "human_review_viability": ("viability", "viability_analysis", "viability_score", "viability_human_feedback"),
    "human_review_feasibility": ("feasibility", "feasibility_analysis", "feasibility_score", "feasibility_human_feedback"),
}


def handle_intake_checkpoint(values: dict) -> dict:
    """Print the agent's latest intake question and collect the founder's
    reply. Returns the state update to write back before resuming.
    """
    stage = values.get("_intake_stage", "desirability")
    intake_history = values.get("intake_history") or {}
    history = intake_history.get(stage) or []
    last_message = history[-1]["content"] if history else "Tell me a bit more about your idea."

    print("\n" + "-"*80)
    print(f"[{stage} intake]")
    print(last_message)
    print("-"*80)
    answer = input("You: ").strip()

    conv_history = values.get("conversation_history") or []
    intake_history[stage] = history + [{"role": "human", "content": answer}]
    return {
        "intake_history": intake_history,
        "conversation_history": conv_history + [{"role": "human", "content": answer}],
    }


def handle_review_checkpoint(node_name: str, values: dict) -> dict:
    """Print the analysis for the current stage and collect feedback via input()."""
    stage_name, analysis_field, score_field, feedback_field = STAGE_INFO[node_name]
    print("\n" + "="*80)
    print(f"{stage_name.upper()} ANALYSIS COMPLETE")
    print("="*80)
    print(f"Score: {values.get(score_field)}/100")
    print(f"\nAnalysis:\n{values.get(analysis_field)}")
    print("="*80)
    feedback = input(f"\nProvide your feedback on {stage_name} (or press Enter to continue): ").strip()
    return {feedback_field: feedback or "Approved to proceed"}


def handle_confirm_downstream_checkpoint(values: dict) -> dict:
    """Ask whether to re-evaluate downstream stages after a revise."""
    source = values.get("_confirm_source", "an earlier stage")
    next_label = "viability and feasibility" if source == "desirability" else "feasibility"
    print("\n" + "-"*80)
    print(f"You revised {source}. {next_label.capitalize()} still have results from before.")
    answer = input(f"Re-evaluate {next_label} too? (y/N): ").strip().lower()
    return {"downstream_choice": "reevaluate" if answer == "y" else "keep"}


def main():
    """Run the startup stress test agent interactively from the terminal."""
    print("\n" + "="*80)
    print("STARTUP STRESS TEST AI AGENT")
    print("Evaluating startup ideas: Desirability -> Viability -> Feasibility")
    print("="*80 + "\n")

    startup_idea = input("Enter the startup idea name: ").strip()
    idea_description = input("Describe the startup idea in detail: ").strip()

    print("\nFunding strategy (used during the viability stage):")
    print("1. BOOTSTRAP: Self-funded, sustainable, profitability-focused")
    print("2. VC-BACKED: Venture-backed, growth-focused")
    funding_model = input("Funding strategy (bootstrap/vc) [bootstrap]: ").strip().lower() or "bootstrap"

    print("\n3-year ARR target: 100k / 1m / 10m, or a custom amount")
    arr_target = input("3-year ARR target [1m]: ").strip() or "1m"

    initial_state = create_initial_state(startup_idea, idea_description, funding_model, arr_target)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info("Building graph...")
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("src.state", "EvaluationStatus")])
        checkpointer = InMemorySaver(serde=serde)
        graph = build_graph(checkpointer=checkpointer, interrupt_before=ALL_INTERRUPT_NODES)

        logger.info(f"Starting evaluation for: {startup_idea}")
        print(f"\nStarting evaluation for: {startup_idea}")
        print(f"Description: {idea_description}\n")
        print("The agent will ask a few quick questions before each stage. Let's start:\n")

        graph.invoke(initial_state, config=config)
        snapshot = graph.get_state(config)

        # Walk through every kind of checkpoint -- intake questions, stage
        # reviews, and downstream-confirmation prompts -- feeding the
        # founder's response back in and resuming, until the graph has
        # nothing left to run (snapshot.next is empty).
        while snapshot.next:
            node_name = snapshot.next[0]

            if node_name == "await_intake_response":
                update = handle_intake_checkpoint(snapshot.values)
            elif node_name == "confirm_downstream":
                update = handle_confirm_downstream_checkpoint(snapshot.values)
            elif node_name in STAGE_INFO:
                update = handle_review_checkpoint(node_name, snapshot.values)
            else:
                logger.warning(f"Unhandled checkpoint node '{node_name}', approving with no feedback.")
                update = {}

            graph.update_state(config, update)
            graph.invoke(None, config=config)
            snapshot = graph.get_state(config)

        final_state = snapshot.values

        print("\n" + "="*80)
        print("FINAL EVALUATION REPORT")
        print("="*80)
        print(final_state['final_report'])
        print("\n" + "="*80)
        print(f"RECOMMENDATION: {final_state['recommendation']}")
        print(f"Overall Score: {final_state['overall_score']:.1f}/100")
        print("="*80 + "\n")

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
