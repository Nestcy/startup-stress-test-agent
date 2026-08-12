"""Final report generation node -- compiles the conversation buffer
(desirability, viability, feasibility analyses) into a single markdown
report. This is the artifact the founder actually reads.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import StartupStressTestState
from src.utils.logger import logger
from src.utils.config import Config
from src.utils.llm_json import strip_think


_PROMPT = ChatPromptTemplate.from_template("""
Write an executive summary report for this startup evaluation, in markdown.

Startup Idea: {startup_idea}

Scores:
- Desirability: {desirability_score}/100
- Viability: {viability_score}/100
- Feasibility: {feasibility_score}/100
- Overall: {overall_score}/100

Full analysis from each stage:
{conversation_buffer}

Write, in this order:
1. Executive Summary (2-3 sentences)
2. Stress Test Results (a markdown table with the four scores above)
3. Key Strengths (3-4 bullets, grounded in what the analysis above actually found)
4. Key Risks (3-4 bullets, same)
5. Critical Success Factors
6. Recommended Next Steps
7. Go/No-Go Recommendation with a short justification

Base every claim on the analysis above -- don't introduce new facts. Write in plain
markdown prose, no JSON.
""")


def generate_final_report(state: StartupStressTestState) -> StartupStressTestState:
    """Generate the final report from the accumulated conversation buffer."""
    logger.info(f"Generating final report for: {state['startup_idea']}")

    llm = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
        temperature=0.7,
        max_tokens=8000,
    )

    scores = [
        state.get('desirability_score') or 0,
        state.get('viability_score') or 0,
        state.get('feasibility_score') or 0,
    ]
    overall_score = sum(scores) / len(scores) if scores else 0
    state['overall_score'] = overall_score

    history = state.get('conversation_history') or []
    conversation_buffer = "\n\n".join(
        f"--- {turn['stage'].upper()} ---\n{turn['content']}" for turn in history
    ) or "(no analysis available)"

    chain = _PROMPT | llm
    response = chain.invoke({
        "startup_idea": state['startup_idea'],
        "desirability_score": state.get('desirability_score', 0),
        "viability_score": state.get('viability_score', 0),
        "feasibility_score": state.get('feasibility_score', 0),
        "overall_score": overall_score,
        "conversation_buffer": conversation_buffer,
    })

    state['final_report'] = strip_think(response.content)

    if overall_score >= 70:
        state['recommendation'] = "GO - Strong potential, proceed with next phase"
    elif overall_score >= 50:
        state['recommendation'] = "CONDITIONAL - Requires refinement in key areas"
    else:
        state['recommendation'] = "NO-GO - Significant concerns, reconsider approach"

    logger.info(f"Final report generated. Overall Score: {overall_score}/100")
    return state
