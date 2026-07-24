#!/usr/bin/env python3
"""Main entry point for startup stress test AI agent (CLI)."""
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.graph import build_graph
from src.state import create_initial_state
from src.utils.logger import logger

HUMAN_REVIEW_NODES = [
    "human_review_desirability",
    "human_review_viability",
    "human_review_feasibility",
]

# Maps the node the graph is paused before -> which state fields to show/fill
STAGE_INFO = {
    "human_review_desirability": ("desirability", "desirability_analysis", "desirability_score", "desirability_human_feedback"),
    "human_review_viability": ("viability", "viability_analysis", "viability_score", "viability_human_feedback"),
    "human_review_feasibility": ("feasibility", "feasibility_analysis", "feasibility_score", "feasibility_human_feedback"),
}


def review_checkpoint(node_name: str, values: dict) -> str:
    """Print the analysis for the current stage and collect feedback via input().

    This is exactly the interaction that used to live inside
    src/nodes/human_review_node.py. It moved here because a node function
    shouldn't assume it's being run interactively -- the API resumes the same
    graph from an HTTP request instead. The CLI is just one of possibly many
    callers that can supply feedback and resume the run.
    """
    stage_name, analysis_field, score_field, _ = STAGE_INFO[node_name]
    print("\n" + "="*80)
    print(f"{stage_name.upper()} ANALYSIS COMPLETE")
    print("="*80)
    print(f"Score: {values.get(score_field)}/100")
    print(f"\nAnalysis:\n{values.get(analysis_field)}")
    print("="*80)
    return input(f"\nProvide your feedback on {stage_name} (or press Enter to continue): ").strip()


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
        graph = build_graph(checkpointer=checkpointer, interrupt_before=HUMAN_REVIEW_NODES)

        logger.info(f"Starting evaluation for: {startup_idea}")
        print(f"\nStarting evaluation for: {startup_idea}")
        print(f"Description: {idea_description}\n")

        graph.invoke(initial_state, config=config)
        snapshot = graph.get_state(config)

        # Walk through checkpoints, feeding feedback back in and resuming,
        # until the graph has nothing left to run (snapshot.next is empty).
        while snapshot.next:
            node_name = snapshot.next[0]
            _, _, _, feedback_field = STAGE_INFO[node_name]
            feedback = review_checkpoint(node_name, snapshot.values)
            graph.update_state(config, {feedback_field: feedback or "Approved to proceed"})
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
