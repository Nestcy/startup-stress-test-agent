"""Feasibility evaluation node -- plain text, no JSON.

Reads desirability + viability out of the shared conversation buffer as
context, then writes its own plain-prose analysis grounded in a live web
search of the technical landscape, ending with a "SCORE: NN/100" line.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import StartupStressTestState, EvaluationStatus
from src.tools.search_tool import SearchTool
from src.utils.logger import logger
from src.utils.config import Config
from src.utils.llm_json import strip_think, extract_score


def _format_buffer(history: list) -> str:
    if not history:
        return "(no prior analysis yet)"
    return "\n\n".join(f"--- {turn['stage'].upper()} ---\n{turn['content']}" for turn in history)


_PROMPT = ChatPromptTemplate.from_template("""
You're a startup advisor assessing FEASIBILITY: can this actually be built and executed?

Startup Idea: {startup_idea}

Prior analysis so far (use this as context, don't repeat it):
{conversation_buffer}

Recent web research on the technology needed to build this:
{search_data}

Write a clear, honest analysis covering:
- What it would actually take to build technically -- core components, a sensible
  tech stack, and what could reasonably be cut for a first version
- Team and skills needed
- A realistic growth path -- rough milestones over the next 12-36 months, without
  pretending you have certainty about exact numbers this far out
- A first-90-days plan: what to prove first, before building more

IMPORTANT SCORING RULE: don't penalize this idea for not having a team or codebase
yet -- that's true of every idea evaluated here. Score whether the technology itself
is buildable with a reasonable team and timeline. Only score low if the tech is
genuinely hard (e.g. needs research-grade AI, heavy regulation, physical
infrastructure) or the plan is genuinely unrealistic -- not because nothing exists yet.

Write in plain prose (a few clear paragraphs, markdown headers are fine). Do not use
JSON. End your response with exactly one line in this format:
SCORE: NN/100
""")


def feasibility_node(state: StartupStressTestState) -> StartupStressTestState:
    """Run the feasibility assessment and append it to the conversation buffer."""
    logger.info(f"Starting feasibility evaluation for: {state['startup_idea']}")

    search_tool = SearchTool()
    llm = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
        temperature=0.7,
        max_tokens=8000,
    )

    search_query = f"how to build {state['startup_idea']} technology stack architecture"
    results = search_tool.search(search_query, topic="general")
    search_data = "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in (results or [])[:5]
    ) or "No search results found for this query."

    chain = _PROMPT | llm
    response = chain.invoke({
        "startup_idea": state['startup_idea'],
        "conversation_buffer": _format_buffer(state.get('conversation_history') or []),
        "search_data": search_data,
    })

    analysis = strip_think(response.content)
    score = extract_score(analysis)

    state['feasibility_analysis'] = analysis
    state['feasibility_status'] = EvaluationStatus.COMPLETED
    state['feasibility_score'] = score
    state['conversation_history'] = (state.get('conversation_history') or []) + [
        {"stage": "feasibility", "content": analysis}
    ]

    logger.info(f"Feasibility evaluation completed. Score: {score}")
    return state
