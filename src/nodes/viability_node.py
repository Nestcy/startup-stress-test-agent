"""Viability evaluation node -- plain text, no JSON.

Reads the desirability analysis out of the shared conversation buffer
(state['conversation_history']) as context, then writes its own plain-prose
analysis grounded in a live web search, ending with a "SCORE: NN/100" line.
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


def _resolve_arr_target(arr_target_input: str) -> tuple:
    """Turn '100k' / '1m' / '10m' / a raw number into an actual dollar figure."""
    arr_targets = {"100k": 100000, "1m": 1000000, "10m": 10000000}
    normalized = (arr_target_input or "1m").strip().lower().replace("$", "").replace(",", "")

    if normalized in arr_targets:
        return arr_targets[normalized], f"Standard milestone: {normalized} ARR"
    try:
        return float(normalized), f"Custom target: ${float(normalized):,.0f}"
    except ValueError:
        return 1000000, "Default: $1M (unparseable ARR target)"


_PROMPT = ChatPromptTemplate.from_template("""
You're a startup advisor assessing VIABILITY: does this work as a business?

Startup Idea: {startup_idea}
Funding approach: {funding_model}
3-year ARR target: ${arr_target:,.0f}

Prior analysis so far (use this as context, don't repeat it):
{conversation_buffer}

Recent web research on market size, pricing, and unit economics for this kind of product:
{search_data}

Write a clear, honest analysis covering:
- Market size: is there realistically enough market to hit the ARR target?
- Pricing: what would this actually charge, anchored to comparable products in the
  research above wherever possible
- Customers needed: roughly how many paying customers get to the ARR target at that price
- Churn and customer lifetime: use real industry benchmarks from the research where you
  can find them
- Customer acquisition: realistic CAC, LTV, and payback period
- Whether the funding approach ({funding_model}) actually fits this model

IMPORTANT SCORING RULE: don't penalize this idea for not having launched yet -- every
idea evaluated here is pre-revenue by definition. Score the underlying business model
against what the research supports (real market size, real comparable pricing, real
unit economics) -- not against whether the founder has already proven it. Only score
low where the research itself shows the model doesn't work (market too small, pricing
unsupported, economics that don't close even in a realistic scenario).

Write in plain prose (a few clear paragraphs, markdown headers are fine). Do not use
JSON. End your response with exactly one line in this format:
SCORE: NN/100
""")


def viability_node(state: StartupStressTestState) -> StartupStressTestState:
    """Run the viability assessment and append it to the conversation buffer."""
    logger.info(f"Starting viability evaluation for: {state['startup_idea']}")

    search_tool = SearchTool()
    llm = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
        temperature=0.7,
        max_tokens=8000,
    )

    funding_model = (state.get('funding_model') or 'bootstrap').strip().lower()
    if funding_model not in ('bootstrap', 'vc'):
        funding_model = 'bootstrap'
    arr_target, _ = _resolve_arr_target(state.get('arr_target'))

    search_query = f"{state['startup_idea']} pricing competitors churn CAC benchmark market size"
    results = search_tool.search(search_query, topic="general")
    search_data = "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in (results or [])[:5]
    ) or "No search results found for this query."

    chain = _PROMPT | llm
    response = chain.invoke({
        "startup_idea": state['startup_idea'],
        "funding_model": funding_model,
        "arr_target": arr_target,
        "conversation_buffer": _format_buffer(state.get('conversation_history') or []),
        "search_data": search_data,
    })

    analysis = strip_think(response.content)
    score = extract_score(analysis)

    state['viability_analysis'] = analysis
    state['viability_status'] = EvaluationStatus.COMPLETED
    state['viability_score'] = score
    state['conversation_history'] = (state.get('conversation_history') or []) + [
        {"stage": "viability", "content": analysis}
    ]

    logger.info(f"Viability evaluation completed. Score: {score}")
    return state
