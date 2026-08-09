"""Conversational intake -- a short back-and-forth with the founder before
each of the three stages (desirability, viability, feasibility), so the
agent isn't analyzing a bare description cold. Reused for all three stages
rather than duplicated, since the only thing that changes per stage is what
it should focus on asking about (see _STAGE_FOCUS below).

Loops via the graph (see graph.py) rather than blocking on input(): this
node runs once per turn, asks one question (or decides it has enough), then
the graph pauses at `await_intake_response` for the founder's reply.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import StartupStressTestState
from src.utils.config import Config
from src.utils.llm_json import extract_json
from src.utils.logger import logger

_MAX_QUESTIONS = 3

# What each stage's intake should actually dig into. Keeps the three
# checkpoints distinct instead of asking the same generic questions three
# times in a row.
_STAGE_FOCUS = {
    "desirability": (
        "who the customer really is, what problem this solves for them, and any "
        "evidence of demand so far (interviews, waitlist, existing users)"
    ),
    "viability": (
        "their funding approach (bootstrap vs VC), rough revenue target, and how "
        "they're thinking about pricing -- you already have their desirability "
        "results as context, so don't re-ask about the customer"
    ),
    "feasibility": (
        "team size/technical skills available, any existing tech constraints or "
        "preferences, and rough timeline expectations -- you already have their "
        "desirability and viability results as context"
    ),
}

_PROMPT = ChatPromptTemplate.from_template("""
You're a startup advisor having a quick, informal chat with a founder before
running the {stage} stress test on their idea. Ask ONE short, sharp
clarifying question at a time, focused on: {focus}.

Ask at most {max_questions} questions for this stage. If the founder has
already given enough to work with -- or you've asked {max_questions} already
-- stop asking and say you're ready to run the {stage} analysis.

Idea: {startup_idea}
Description: {idea_description}
Prior results so far: {prior_context}

Conversation so far (this stage):
{conversation}

Respond as JSON: {{"ready": true or false, "message": "..."}}
"message" is either your next question, or (if ready) a brief, friendly
line letting them know you're about to start the analysis.
""")


def _format_conversation(history):
    if not history:
        return "(nothing yet -- this is the first message for this stage)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)


def _prior_context(state: StartupStressTestState, stage: str) -> str:
    parts = []
    if stage in ("viability", "feasibility") and state.get("desirability_score") is not None:
        parts.append(f"Desirability score: {state['desirability_score']}/100")
    if stage == "feasibility" and state.get("viability_score") is not None:
        parts.append(f"Viability score: {state['viability_score']}/100")
    return "; ".join(parts) if parts else "None yet"


def intake_node(state: StartupStressTestState) -> StartupStressTestState:
    stage = state.get("_intake_stage", "desirability")
    llm = ChatGroq(api_key=Config.GROQ_API_KEY, model=Config.GROQ_MODEL, temperature=0.6)

    # Track this stage's intake conversation separately (dynamic key) so
    # viability's questions don't get confused with desirability's leftover
    # history when the founder revisits a stage.
    key = f"_intake_history_{stage}"
    history = state.get(key) or []

    questions_so_far = sum(1 for t in history if t.get("role") == "assistant")
    if questions_so_far >= _MAX_QUESTIONS:
        state["intake_ready"] = True
        closing = {"role": "assistant", "content": f"Good enough -- let's run {stage}."}
        state[key] = history + [closing]
        state["conversation_history"] = (state.get("conversation_history") or []) + [closing]
        return state

    chain = _PROMPT | llm
    response = chain.invoke({
        "stage": stage,
        "focus": _STAGE_FOCUS[stage],
        "startup_idea": state["startup_idea"],
        "idea_description": state.get("idea_description", ""),
        "prior_context": _prior_context(state, stage),
        "conversation": _format_conversation(history),
        "max_questions": _MAX_QUESTIONS,
    })

    result = extract_json(response.content, fallback_keys=("ready", "message"))
    message = result.get("message") or f"Tell me more before I run {stage}."
    ready = bool(result.get("ready"))

    logger.info(f"[{stage} intake] {'ready' if ready else 'asking'}: {message}")

    state["intake_ready"] = ready
    assistant_turn = {"role": "assistant", "content": message}
    state[key] = history + [assistant_turn]
    state["conversation_history"] = (state.get("conversation_history") or []) + [assistant_turn]
    return state
