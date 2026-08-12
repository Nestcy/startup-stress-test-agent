"""Shared helper: decide whether a founder's question needs a live web
search before answering, and if so, run it. Used by the API's /discuss
endpoint. No JSON here either -- the model answers in two plain lines,
parsed with simple regexes.
"""
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.tools.search_tool import SearchTool
from src.utils.config import Config
from src.utils.llm_json import strip_think
from src.utils.logger import logger

_search_tool = SearchTool()

_NEEDS_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
A founder said this: "{question}"

Does responding well require current, real-world information you wouldn't already
have (e.g. competitor pricing, a specific company/product, recent market data,
"look up X", "what's the latest on Y")? Or is it answerable from context already
available?

Answer in exactly two lines, nothing else:
NEEDS_SEARCH: yes or no
QUERY: a short, specific web search query (only meaningful if yes)
""")

_NEEDS_SEARCH_LINE = re.compile(r"NEEDS_SEARCH:\s*(yes|no)", re.IGNORECASE)
_QUERY_LINE = re.compile(r"QUERY:\s*(.+)")


def maybe_search(question: str) -> tuple[bool, str]:
    """Returns (used_search, search_context)."""
    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL, temperature=0.3, max_tokens=500)
    chain = _NEEDS_SEARCH_PROMPT | llm

    decision_text = strip_think(chain.invoke({"question": question}).content)

    needs_match = _NEEDS_SEARCH_LINE.search(decision_text)
    if not needs_match or needs_match.group(1).lower() != "yes":
        return False, ""

    query_match = _QUERY_LINE.search(decision_text)
    query = query_match.group(1).strip() if query_match else question
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
