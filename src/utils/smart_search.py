"""Shared helper: decide whether a founder's question needs a live web
search before answering, and if so, run it. Currently used only by the
API's /discuss endpoint -- that's the one place in the conversation where a
founder can trigger a lookup (e.g. "what's Prodigy's current pricing?")
mid-checkpoint, without advancing the evaluation.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.tools.search_tool import SearchTool
from src.utils.config import Config
from src.utils.llm_json import extract_json
from src.utils.logger import logger

_search_tool = SearchTool()

_NEEDS_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
A founder said this: "{question}"

Does responding well require current, real-world information you wouldn't
already have (e.g. competitor pricing, a specific company/product, recent
market data, "look up X", "what's the latest on Y")? Or is it answerable
from context already available?

Respond as JSON: {{"needs_search": true or false, "search_query": "..."}}
"search_query" should be a short, specific web search query -- only
meaningful if needs_search is true.
""")


def maybe_search(question: str) -> tuple[bool, str]:
    """Returns (used_search, search_context).

    Runs a real Tavily search only if the LLM decides the question actually
    needs current information -- avoids searching on every single message,
    which would be slow and mostly irrelevant for questions answerable from
    the idea description or an existing analysis alone.
    """
    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL, temperature=0.3)
    chain = _NEEDS_SEARCH_PROMPT | llm

    decision = extract_json(
        chain.invoke({"question": question}).content,
        fallback_keys=("needs_search", "search_query"),
    )

    if not decision.get("needs_search"):
        return False, ""

    query = decision.get("search_query") or question
    logger.info(f"Detected need for current info -- searching: {query}")

    try:
        results = _search_tool.search(query, topic="general")
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return False, ""

    if not results:
        return False, ""

    context = "\n\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in results[:3]
    )
    return True, context
