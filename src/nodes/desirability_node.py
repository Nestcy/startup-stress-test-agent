"""Desirability evaluation node -- plain text, no JSON.

Writes a single, real-prose analysis grounded in a live web search, ending
with a "SCORE: NN/100" line. That analysis gets appended to the shared
conversation buffer (state['conversation_history']) so viability and
feasibility can read it as context later -- that's the whole "memory"
mechanism, no database or vector store needed.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import StartupStressTestState, EvaluationStatus
from src.tools.search_tool import SearchTool
from src.utils.logger import logger
from src.utils.config import Config
from src.utils.llm_json import strip_think, extract_score


_PROMPT = ChatPromptTemplate.from_template("""
You're a startup advisor assessing DESIRABILITY: do real people actually want this?

Startup Idea: {startup_idea}
Description: {idea_description}

Recent web research on this market/customer:
{search_data}

Write a clear, honest analysis covering:
- Who the real customer is, and how big that market is
- What's changed recently that makes this timely (or not)
- What people currently do instead (direct and indirect alternatives), and how good those
  alternatives already are
- Whether this specific solution actually fits the problem, and what's genuinely
  different about it
- The real risks here

IMPORTANT SCORING RULE: every idea being evaluated here is pre-launch by definition --
the founder hasn't personally run interviews or built a waitlist yet, and that is NOT
a strike against the idea. Score based on what the research above actually shows about
the opportunity (real demand signal, real timing, real competitive gaps) -- not on
the founder's execution stage. Only score low where the research itself points to weak
demand or a crowded, undifferentiated market.

Write in plain prose (a few clear paragraphs, markdown headers are fine). Do not use
JSON. End your response with exactly one line in this format:
SCORE: NN/100
""")


def desirability_node(state: StartupStressTestState) -> StartupStressTestState:
    """Run the desirability assessment and append it to the conversation buffer."""
    logger.info(f"Starting desirability evaluation for: {state['startup_idea']}")

    search_tool = SearchTool()
    llm = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
        temperature=0.7,
        max_tokens=8000,
    )

    search_query = f"{state['startup_idea']} target customers market demand competitors 2025 2026"
    results = search_tool.search(search_query, topic="general")
    search_data = "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in (results or [])[:5]
    ) or "No search results found for this query."

    chain = _PROMPT | llm
    response = chain.invoke({
        "startup_idea": state['startup_idea'],
        "idea_description": state.get('idea_description', ''),
        "search_data": search_data,
    })

    analysis = strip_think(response.content)
    score = extract_score(analysis)

    state['desirability_analysis'] = analysis
    state['desirability_status'] = EvaluationStatus.COMPLETED
    state['desirability_score'] = score
    state['conversation_history'] = (state.get('conversation_history') or []) + [
        {"stage": "desirability", "content": analysis}
    ]

    logger.info(f"Desirability evaluation completed. Score: {score}")
    return state
