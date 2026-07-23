"""Final report generation node"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import StartupStressTestState
from src.utils.logger import logger
from src.utils.config import Config


def generate_final_report(state: StartupStressTestState) -> StartupStressTestState:
    """Generate comprehensive final report with all evaluations and recommendation."""
    logger.info(f"Generating final report for: {state['startup_idea']}")
    
    llm = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
        temperature=0.7
    )
    
    scores = [
        state.get('desirability_score', 0),
        state.get('viability_score', 0),
        state.get('feasibility_score', 0)
    ]
    overall_score = sum(scores) / len(scores) if scores else 0
    state['overall_score'] = overall_score
    
    prompt = ChatPromptTemplate.from_template("""
    Create executive summary report for startup evaluation.
    
    Startup Idea: {startup_idea}
    
    Scores:
    - Desirability: {desirability_score}/100
    - Viability: {viability_score}/100
    - Feasibility: {feasibility_score}/100
    - Overall: {overall_score}/100
    
    Generate:
    1. Executive Summary (2-3 sentences)
    2. Stress Test Results (table with scores)
    3. Key Strengths (3-4 bullets)
    4. Key Risks (3-4 bullets)
    5. Critical Success Factors
    6. Recommended Next Steps
    7. Go/No-Go Recommendation with justification
    
    Use markdown formatting.
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "startup_idea": state['startup_idea'],
        "desirability_score": state.get('desirability_score', 0),
        "viability_score": state.get('viability_score', 0),
        "feasibility_score": state.get('feasibility_score', 0),
        "overall_score": overall_score
    })
    
    state['final_report'] = response.content
    
    if overall_score >= 70:
        state['recommendation'] = "GO - Strong potential, proceed with next phase"
    elif overall_score >= 50:
        state['recommendation'] = "CONDITIONAL - Requires refinement in key areas"
    else:
        state['recommendation'] = "NO-GO - Significant concerns, reconsider approach"
    
    logger.info(f"Final report generated. Overall Score: {overall_score}/100")
    return state
