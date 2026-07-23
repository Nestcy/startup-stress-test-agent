#!/usr/bin/env python3
"""Main entry point for startup stress test AI agent"""

from src.graph import build_graph
from src.state import StartupStressTestState, EvaluationStatus
from src.utils.logger import logger


def main():
    """
    Main function to run the startup stress test agent.
    """
    print("\n" + "="*80)
    print("STARTUP STRESS TEST AI AGENT")
    print("Evaluating startup ideas: Desirability → Viability → Feasibility")
    print("="*80 + "\n")
    
    startup_idea = input("Enter the startup idea name: ").strip()
    idea_description = input("Describe the startup idea in detail: ").strip()
    
    initial_state: StartupStressTestState = {
        "startup_idea": startup_idea,
        "idea_description": idea_description,
        "desirability_analysis": None,
        "desirability_status": EvaluationStatus.PENDING,
        "desirability_score": None,
        "desirability_human_feedback": None,
        "viability_analysis": None,
        "viability_status": EvaluationStatus.PENDING,
        "viability_score": None,
        "viability_human_feedback": None,
        "feasibility_analysis": None,
        "feasibility_status": EvaluationStatus.PENDING,
        "feasibility_score": None,
        "feasibility_human_feedback": None,
        "final_report": None,
        "overall_score": None,
        "recommendation": None,
        "search_results": None,
        "conversation_history": [],
        "errors": []
    }
    
    try:
        logger.info(f"Building graph...")
        graph = build_graph()
        
        logger.info(f"Starting evaluation for: {startup_idea}")
        print(f"\nStarting evaluation for: {startup_idea}")
        print(f"Description: {idea_description}\n")
        
        final_state = graph.invoke(initial_state)
        
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
