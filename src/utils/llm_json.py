"""Helpers for extracting structured JSON from reasoning-model responses.

Models like Qwen 3.x prefix their answer with a <think>...</think> block
containing the model's internal reasoning before the actual JSON output.
`response.content` is therefore never valid JSON on its own -- a plain
`json.loads(response.content)` will always fail for these models, which is
why nodes were silently falling back to dumping the raw, unparsed response
(thinking process and all) into state.

Save this file as: src/utils/llm_json.py
"""
import json
import re
from typing import Dict

from src.utils.logger import logger

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(raw_text: str, fallback_keys: tuple = ()) -> Dict:
    """Strip any <think>...</think> block, then parse the first JSON object
    found in the remaining text.

    Falls back to an empty dict with `fallback_keys` set to None (rather
    than a raw_response dump) if no valid JSON can be found -- callers can
    then use `.get(key, default)` safely without leaking unparsed model
    reasoning into the report the founder sees.
    """
    cleaned = _THINK_BLOCK.sub("", raw_text).strip()

    # Models often wrap JSON in ```json ... ``` fences even after the
    # <think> block is removed -- strip those too if present.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the first {...} substring, in case there's stray
    # text before/after the JSON object despite the cleanup above.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from model response after cleaning. Using fallback.")
    return {key: None for key in fallback_keys}
