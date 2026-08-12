"""Helpers for working with reasoning-model output as plain text.

No JSON parsing anywhere in this codebase anymore -- every phase now just
writes a plain-text analysis, and the only thing pulled out of it
programmatically is a numeric score, via a simple regex on a "SCORE: NN/100"
line the prompts ask the model to end with. This is far less fragile than
requiring the model to produce valid multi-key JSON every single call: there's
one thing to get right (one line, one number) instead of a whole schema.
"""
import re

from src.utils.logger import logger

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SCORE_LINE = re.compile(r"SCORE:\s*(\d{1,3})\s*/?\s*100?", re.IGNORECASE)


def strip_think(raw_text: str) -> str:
    """Strip a <think>...</think> block from model output, leaving just the
    actual answer. Reasoning models like Qwen always prefix their response
    with this block; every piece of text stored in state or shown to the
    founder should have it removed first.
    """
    return _THINK_BLOCK.sub("", raw_text or "").strip()


def extract_score(text: str, default: int = 50) -> float:
    """Pull a numeric score out of a "SCORE: NN/100" line in plain-text
    model output. Falls back to `default` (not 0) if no such line is found,
    since a missing score line is a formatting slip, not evidence the idea
    scored zero -- treating it as zero would silently make every parsing
    hiccup look like a harsh rejection, which is exactly the failure mode
    this whole rewrite is trying to get away from.
    """
    match = _SCORE_LINE.search(text or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    logger.warning("Could not find a 'SCORE: NN/100' line in model output. Using default score.")
    return float(default)
