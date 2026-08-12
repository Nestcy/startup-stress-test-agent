#!/usr/bin/env python3
"""Main entry point for startup stress test AI agent (CLI)."""
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START

from src.graph import build_graph
from src.state import create_initial_state
from src.utils.logger import logger
from src.utils.smart_search import maybe_search

REWIND_ANCHOR = {"desirability": START, "viability": "desirability", "feasibility": "viability"}
STAGE_FIELDS_TO_CLEAR = {
    "desirability": ["desirability_analysis", "desirability_score", "final_report", "overall_score", "recommendation", "downstream_choice"],
    "viability": ["viability_analysis", "viability_score", "final_report", "overall_score", "recommendation", "downstream_choice"],
    "feasibility": ["feasibility_analysis", "feasibility_score", "final_report", "overall_score", "recommendation"],
}


def print_report(values: dict):
    print("\n" + "="*80)
    print("FINAL EVALUATION REPORT")
    print("="*80)
    print(f"Desirability: {values.get('desirability_score')}/100")
    print(f"Viability:    {values.get('viability_score')}/100")
    print(f"Feasibility:  {values.get('feasibility_score')}/100")
    print("-"*80)
    print(values.get('final_report'))
    print("\n" + "="*80)
    print(f"RECOMMENDATION: {values.get('recommendation')}")
    print(f"Overall Score: {values.get('overall_score'):.1f}/100")
    print("="*80 + "\n")


def run_to_completion(graph, config, initial_state=None):
    """Invoke the graph and, if it needs a downstream confirmation
    (only happens mid-/revise), ask for it via input() and keep resuming
    until the run reaches END.
    """
    if initial_state is not None:
        graph.invoke(initial_state, config=config)
    snapshot = graph.get_state(config)

    while snapshot.next:  # only ever confirm_downstream in this graph
        source = snapshot.values.get("_confirm_source", "an earlier stage")
        next_label = "viability and feasibility" if source == "desirability" else "feasibility"
        answer = input(f"\nYou revised {source}. Re-evaluate {next_label} too? (y/N): ").strip().lower()
        graph.update_state(config, {"downstream_choice": "reevaluate" if answer == "y" else "keep"})
        graph.invoke(None, config=config)
        snapshot = graph.get_state(config)

    return snapshot.values


def handle_revise(graph, config, stage: str):
    print(f"\nRe-running {stage}...")
    new_description = input(f"Updated description (Enter to keep current): ").strip()

    values_to_clear = {field: None for field in STAGE_FIELDS_TO_CLEAR[stage]}
    if new_description:
        values_to_clear["idea_description"] = new_description

    graph.update_state(config, values_to_clear, as_node=REWIND_ANCHOR[stage])
    final_values = run_to_completion(graph, config)
    print_report(final_values)


def handle_ask(config, values: dict):
    question = input("\nYour question: ").strip()
    if not question:
        return
    used_search, search_context = maybe_search(question)
    if used_search:
        print("(Looking that up...)")

    # Reuses the same prompt shape as api.py's /ask -- kept inline here to
    # avoid importing FastAPI machinery into the CLI. If you're maintaining
    # both, keep this in sync with _ASK_PROMPT in src/api.py.
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from src.utils.config import Config

    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL, temperature=0.5)
    prompt = ChatPromptTemplate.from_template("""
    Answer the founder's question about this completed startup evaluation.

    Idea: {startup_idea}
    Desirability ({d_score}/100): {d_analysis}
    Viability ({v_score}/100): {v_analysis}
    Feasibility ({f_score}/100): {f_analysis}
    {search_note}

    Question: {question}
    """)
    chain = prompt | llm
    response = chain.invoke({
        "startup_idea": values.get("startup_idea"),
        "d_score": values.get("desirability_score"),
        "d_analysis": values.get("desirability_analysis") or "N/A",
        "v_score": values.get("viability_score"),
        "v_analysis": values.get("viability_analysis") or "N/A",
        "f_score": values.get("feasibility_score"),
        "f_analysis": values.get("feasibility_analysis") or "N/A",
        "search_note": f"Recent search results:\n{search_context}" if used_search else "",
        "question": question,
    })
    print(f"\n{response.content}")


def main():
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
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("src.state", "EvaluationStatus")])
        checkpointer = InMemorySaver(serde=serde)
        graph = build_graph(checkpointer=checkpointer, interrupt_before=["confirm_downstream"])

        print(f"\nRunning the full evaluation for: {startup_idea}")
        print("This runs straight through -- desirability, viability, then feasibility -- ")
        print("searching for real market/tech data along the way. This can take a few minutes.\n")

        final_values = run_to_completion(graph, config, initial_state=initial_state)
        print_report(final_values)

        # Post-completion loop: revise a stage, ask a question, or quit.
        while True:
            print("What would you like to do?")
            print("  1. Ask a question about this evaluation")
            print("  2. Revise a stage (desirability / viability / feasibility)")
            print("  3. Quit")
            choice = input("> ").strip()

            if choice == "1":
                handle_ask(config, graph.get_state(config).values)
            elif choice == "2":
                stage = input("Which stage? (desirability/viability/feasibility): ").strip().lower()
                if stage in REWIND_ANCHOR:
                    handle_revise(graph, config, stage)
                else:
                    print("Not a recognized stage.")
            else:
                break

        logger.info("Session ended")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
