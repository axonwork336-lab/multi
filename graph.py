"""
LangGraph graph for the LLM-tool-calling Guest Booking Cancellation Agent.

REWRITTEN (see graph.py.pre_rewrite_backup for the old deterministic
20-node router). The graph is now a standard two-node ReAct-style loop:

    START -> load_config -> agent <-> tools -> END

  - load_config: loads/caches the tenant's client_config.csv +
    dialect_templates.csv (config.get_messages, UNCHANGED function) and
    builds the system prompt (prompts.build_system_prompt) - but only
    once per thread (checked via state.get("templates")), not every turn.
  - agent: calls the LLM, bound to every tool in tools.ALL_TOOLS. The LLM
    decides which tool(s) to call, if any.
  - tools: executes whatever tool call(s) the LLM just requested
    (LangGraph's prebuilt ToolNode - handles InjectedState wiring
    automatically) and appends the results as ToolMessages.
  - Routing: after "agent", if the LLM's latest message contains
    tool_calls, go to "tools"; otherwise the LLM's message IS the reply
    to the user this turn, so the graph ends (control returns to
    whichever caller - main.py's CLI or app.py's HTTP layer - the same
    way for both).

NOTE ON INTERRUPTS: the old graph used explicit interrupt() calls at
wait_for_otp/wait_for_selection/wait_for_confirmation/etc. Those are GONE
- there is no longer a fixed set of "steps that pause". Any time the
agent's response has no tool_calls (e.g. it's asking "which one?" or
"please confirm" or "what's the OTP?"), that's a natural, implicit pause:
the graph reaches END, and the next incoming HumanMessage (via app.py or
main.py) simply gets appended and the graph invoked again - the
checkpointer (MemorySaver, PRESERVED exactly per requirements) restores
the full chat history for that thread_id automatically. This IS how
interrupt/resume is achieved now - it's just no longer a graph-level
primitive, because the "steps" themselves no longer exist as distinct
nodes; the LLM decides turn-by-turn whether it needs another tool call
or needs to ask the user something.
"""

import json
import os
import sys
import logging
import re
import ast
from datetime import datetime
from typing import Dict, Optional

from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from openai import APIConnectionError as _OpenAIAPIConnectionError
from openai import APITimeoutError as _OpenAIAPITimeoutError

import agents
import config
import progress
import tools
from prompts import build_agent_system_prompt, build_system_prompt
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# LLM (bound to every tool) - api.py/config.py's OPENAI_* settings,
# UNCHANGED from the old project
# ==========================================================

_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY or "sk-not-configured",
    timeout=config.OPENAI_TIMEOUT_SECONDS,
)

_llm_with_tools = _llm.bind_tools(tools.ALL_TOOLS)

# The object `_llm_with_tools` was bound to at import time, kept as an
# identity sentinel. `_llm_for()` below compares against it to tell
# "nobody has touched this" apart from "a caller has swapped in their
# own LLM" (which the test suite does, by assigning to
# graph._llm_with_tools). When it HAS been swapped, every specialist
# uses the replacement - otherwise scripted-LLM tests would silently
# bypass the substitution the moment routing picked a specialist.
_DEFAULT_LLM_WITH_TOOLS = _llm_with_tools

# One binding per specialist, built once at import. Binding is pure
# schema work - no network call, no API key needed - so this is cheap
# and safe even where OPENAI_API_KEY is absent.
_AGENT_LLMS = {
    name: _llm.bind_tools(agents.tools_for(name))
    for name in agents.AGENT_NAMES
}


# ==========================================================
# LLM call resilience - the OpenAI request timing out (or a dropped
# connection) used to crash the WHOLE turn with no reply sent at all.
# ==========================================================
#
# CONFIRMED REAL PRODUCTION FAILURE (from the actual stack trace):
# openai.APITimeoutError raised straight out of `_llm_for(...).invoke(...)`,
# with no try/except anywhere between here and main.py. That is worse
# than the "empty reply" case main.send_message_with_signals already
# guards against - an empty reply still reaches that guard and gets
# turned into the clinic's own failure message; a raised exception never
# reaches it at all, because the turn never returns. The patient gets
# nothing, and (depending on how the HTTP layer above main.py handles an
# unhandled exception) the caller may not even see a clean error.
#
# One retry is attempted first - a slow or dropped request is often
# just that, slow - and only on a SECOND failure is a fallback reply
# returned instead of raising, so the graph ends this turn normally
# (exactly like any other no-tool-call reply) rather than crashing it.

_LLM_TIMEOUT_FALLBACK_TEXT = {
    "ar": "عذرًا، حصل تأخير مؤقت في الرد. ممكن تبعت رسالتك تاني؟ 🌷",
    "en": "Sorry, there was a temporary delay on our end. Could you please resend your last message?",
}


def _invoke_llm_resilient(llm, messages, *, agent_name: str, target_language: str, context: str):
    """`llm.invoke(messages)`, but a request timeout / connection drop is
    retried once and, if it fails again, converted into a graceful
    fallback AIMessage instead of propagating and crashing the turn.

    `context` is a short label for the logs only (which of the two call
    sites this was - the main turn or a verifier's correction retry)."""

    last_exc = None
    for attempt in (1, 2):
        try:
            return llm.invoke(messages)
        except (_OpenAIAPITimeoutError, _OpenAIAPIConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "agent[%s]: %s - LLM call failed on attempt %d/2 (%s: %s)",
                agent_name, context, attempt, type(exc).__name__, exc,
            )

    logger.error(
        "agent[%s]: %s - LLM call failed twice in a row (%s) - returning a "
        "fallback reply instead of letting this crash the whole turn",
        agent_name, context, type(last_exc).__name__ if last_exc else "?",
    )
    fallback_text = _LLM_TIMEOUT_FALLBACK_TEXT.get(target_language) or _LLM_TIMEOUT_FALLBACK_TEXT["ar"]
    return AIMessage(content=fallback_text)


def _llm_for(agent_name: str):
    """The LLM this specialist should use, bound to its own tool subset.

    Falls back to the shared `_llm_with_tools` whenever that would be
    wrong or risky: when a caller has replaced it, when tool scoping is
    switched off, or when the name isn't in the registry."""

    if _llm_with_tools is not _DEFAULT_LLM_WITH_TOOLS:
        return _llm_with_tools

    if not config.AGENT_TOOL_SCOPING:
        return _llm_with_tools

    return _AGENT_LLMS.get(agent_name, _llm_with_tools)


# ==========================================================
# Nodes
# ==========================================================

def load_config(state: AgentState) -> AgentState:
    """Loads this turn's client config - from `state["raw_client_config"]`
    if n8n sent one this turn, otherwise falling back to
    client_config.csv/dialect_templates.csv by client_id (config.py,
    UNCHANGED for that path) - and builds the system prompt EVERY turn.

    CHANGED: this used to skip rebuilding once state["templates"] was
    already set for a thread (a caching optimization). The real-world
    cost of that: any update to prompts.py or the CSVs would NEVER take
    effect for a conversation already in progress - only brand new
    threads picked it up - which is exactly what caused a live
    conversation to keep using stale, pre-fix wording long after
    multiple prompt improvements had already been deployed. Rebuilding
    every turn is cheap (in-memory CSV lookups, already cached at the
    row level by config.py's lru_cache, plus plain string formatting -
    no network calls), so there's no real performance reason to keep
    the old per-thread caching."""

    templates = config.get_messages(state["client_id"], client_row_override=state.get("raw_client_config"))

    state["templates"] = templates

    # THE BUILT PROMPT IS NOT STORED IN STATE WHEN IT ISN'T READ.
    #
    # `system_prompt` is ~130 KB of text. Everything in state is written
    # into the checkpoint by the checkpointer, on every super-step - so
    # a single 20-turn conversation was carrying something like 7.5 MB
    # of identical copies of a string that is rebuilt from `templates`
    # anyway (deliberately - see the docstring above). With MemorySaver
    # keeping checkpoints in process memory, that is the largest single
    # thing this service holds per conversation, for no benefit.
    #
    # In multi-agent mode `_run_agent` never reads this field: it builds
    # the SCOPED prompt itself from `templates`. It is only read on the
    # legacy single-agent path (MULTI_AGENT_ENABLED=false), so that is
    # the only case where it is stored - the rollback flag keeps working
    # byte for byte, and normal operation stops paying for it.
    if config.MULTI_AGENT_ENABLED:
        state["system_prompt"] = None
    else:
        state["system_prompt"] = build_system_prompt(templates)

    return state


_MORNING_CUES = ("صباح", "good morning", "morning")
_EVENING_CUES = ("مساء", "good evening", "evening")


# Mirrors the Arabic greeting item for item, in the same order, with the
# same emoji. The complaint/suggestion line (📝) used to be missing here
# while it was present in every client's Arabic greeting - so an English
# patient was told about five of the six things this assistant does, and
# the one thing that got dropped was the one for raising a problem.
_ENGLISH_GREETING_TEMPLATE = (
    "{salutation}\n"
    "I'm {agent_name}, the virtual assistant at {clinic_name}, and I'm happy to help you today.\n"
    "I can help you with:\n"
    "\U0001F5D3\uFE0F Booking a new appointment\n"
    "\u270F\uFE0F Modifying or cancelling an existing appointment\n"
    "\U0001FA7A Medical guidance to choose the right specialty or doctor\n"
    "\u2139\uFE0F Questions about the hospital's services and doctors\n"
    "\U0001F4DD Filing a complaint or a suggestion\n"
    "\U0001F464 Speaking with a customer service representative\n\n"
    "How can I help you today? \U0001F60A"
)


def _normalize_for_compare(text: str) -> str:
    """Collapse all whitespace (including \\r\\n vs \\n differences) into
    single spaces, for tolerant text comparison."""

    return re.sub(r"\s+", " ", (text or "").replace("\r", "\n")).strip()


def _already_contains_greeting(reply_text: str, greeting: str) -> bool:
    """
    Whether `reply_text` already includes the clinic greeting.

    Compares on a NORMALIZED SIGNATURE rather than the full exact text.
    Requiring an exact full-text match was the cause of a real
    production bug: the CSV greeting had \\r\\n line endings while the
    LLM's own reproduction of it used \\n, so the exact check failed and
    a second copy of the greeting was prepended to a reply that already
    contained one.

    The signature is the greeting's longest early line (usually the
    persona line, e.g. "أنا لطيفة، المساعدة الافتراضية في مستشفى ...") -
    distinctive enough that a false positive is very unlikely, and
    tolerant of cosmetic differences elsewhere in the message.
    """

    normalized_reply = _normalize_for_compare(reply_text)
    normalized_greeting = _normalize_for_compare(greeting)

    if not normalized_greeting:
        return False

    if normalized_greeting in normalized_reply:
        return True

    # Fall back to the most distinctive single line of the greeting.
    candidate_lines = [
        _normalize_for_compare(line)
        for line in greeting.replace("\r", "\n").split("\n")[:4]
    ]
    signature = max((line for line in candidate_lines if line), key=len, default="")

    if len(signature) >= 20 and signature in normalized_reply:
        return True

    return False


_NUMBER_EMOJIS = ["0️⃣", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]


def _numbered_prefix(n: int) -> str:
    """Emoji badge for a list position, for ANY number.

    1-10 use their own single badge (🔟 for ten). Beyond that, the
    keycap emoji for each digit are simply written side by side -
    1️⃣1️⃣ for 11, 1️⃣2️⃣ for 12 - so a long list stays visually
    consistent all the way down.

    WHY: the previous version fell back to a plain "11." / "12." past
    ten, which made the tail of a long slot list look like a different,
    unstyled list stapled onto the end of the emoji-numbered one.

    BIDI FIX: keycap digit emoji (e.g. 1️⃣) are NOT recognized by the
    Unicode bidi algorithm as "European Number" characters the way plain
    ASCII digits are - they're ordinary neutral symbols. A two-symbol
    sequence like "1️⃣2️⃣" is therefore free to be visually REORDERED by
    an RTL renderer (WhatsApp/Arabic paragraph context), and confirmed
    real production behavior: "12" (1️⃣2️⃣) rendered on-device as "21",
    while "11" (1️⃣1️⃣) looked fine only because reversing two identical
    digits is invisible. Wrapping the multi-digit sequence in a
    LEFT-TO-RIGHT ISOLATE (U+2066) ... POP DIRECTIONAL ISOLATE (U+2069)
    pair tells the bidi algorithm to render exactly what's enclosed in
    left-to-right order regardless of the surrounding RTL paragraph,
    which is what a genuine multi-digit number needs.
    """

    if n == 10:
        return "🔟"
    if 1 <= n <= 9:
        return _NUMBER_EMOJIS[n]
    digits = "".join(_NUMBER_EMOJIS[int(digit)] for digit in str(n))
    return f"\u2066{digits}\u2069"


def _build_slots_numbered_list_directive(messages: list) -> str:
    """
    If the LAST message is a ToolMessage from `get_available_reschedule_slots`
    with status "found", pre-build the EXACT numbered list of slot times in
    code and hand it to the model as a ready-made block to include
    verbatim - rather than only instructing it to format the list itself,
    which was not reliably followed even after an explicit prose
    instruction (observed directly in production - the numbered list
    format never appeared).

    Returns an empty string when the last message isn't a matching,
    successful slots-lookup tool result.
    """

    if not messages:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) not in ("get_available_reschedule_slots", "get_available_slots_for_booking"):
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if data.get("status") != "found":
        return ""

    slots = data.get("slots") or []
    if not slots:
        return ""

    lines = [f"{_numbered_prefix(i + 1)} {slot.get('time_display', '')}" for i, slot in enumerate(slots)]
    numbered_list = "\n".join(lines)

    first_slot = slots[0]
    date_display = first_slot.get("date_display") or ""
    weekday_display = first_slot.get("weekday_display") or ""
    service_name = first_slot.get("serviceName") or ""

    # NO PRICE HERE. An earlier version appended the service fee to this
    # header, which is exactly how prices ended up printed at people who
    # never asked about cost. Fees are private by default and only ever
    # revealed via `get_doctor_fees` on an explicit request (see
    # prompts.py's FEES rule); the tools no longer return servicePrice
    # in slot data at all, so this is now enforced on both sides.
    header_parts = []
    if date_display:
        day_label = f"{weekday_display} {date_display}".strip()
        header_parts.append(f"📅 المواعيد المتاحة ليوم {day_label}")
    if service_name:
        header_parts.append(f"— {service_name}")
    header = (" ".join(header_parts) + ":") if header_parts else ""

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "The available time slots were just looked up. Your ENTIRE reply "
        "must have EXACTLY this structure: the exact text between the "
        "START/END markers below (header line if present, then the "
        "numbered list), then ONE question asking them to reply with "
        "the number or the exact time. The START/END marker lines "
        "themselves are NOT part of the text to copy - never include "
        "them, or any other line of dashes/equals-signs, in your actual "
        "reply. Nothing else - do NOT also describe the slots in your "
        "own words anywhere in the reply.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{header}\n"
        f"{numbered_list}\n"
        "[END-EXACT-TEXT]\n\n"
    )


def _build_available_days_directive(messages: list, session_id: str) -> str:
    """
    If the LAST message is a ToolMessage from
    `list_available_days_for_booking` with status "found", pre-build the
    exact numbered day list in code - same approach (and same reasons)
    as the slots list above.

    Two things this guarantees that prose instructions did not:
      - ONLY the days the tool actually returned appear. The tool now
        returns the 3 nearest by default; the model must not pad that
        list back out with dates of its own, and must not silently drop
        one either.
      - When more days exist beyond these, the reply says so and offers
        the next three - so "these are the nearest three" never reads as
        "this is everything the doctor has".
    """

    if not messages:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) != "list_available_days_for_booking":
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if data.get("status") != "found":
        return ""

    days = data.get("days") or []
    if not days:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    doctor_name = session.get("doctor_display_name") or ""
    branch_name = session.get("branch_display_name") or ""

    single = len(days) == 1

    if single:
        header = "أقرب موعد متاح"
        if doctor_name:
            header = f"أقرب موعد متاح عند {doctor_name}"
        if branch_name:
            header = f"{header} في {branch_name}"
    else:
        header = "🗓️ المواعيد المتاحة"
        if doctor_name:
            header = f"🗓️ مواعيد {doctor_name} المتاحة"
        if branch_name:
            header = f"{header} في {branch_name}"
    header += ":"

    lines = []
    for i, day in enumerate(days):
        weekday = day.get("weekday_display") or day.get("weekday_name") or ""
        date_display = day.get("date_display") or ""
        first_time = day.get("firstTime") or ""
        last_time = day.get("lastTime") or ""
        lines.append(
            f"{_numbered_prefix(i + 1)} {weekday} {date_display} — من {first_time} إلى {last_time}".strip()
        )

    # IF THIS DAY LIST IS THE ANSWER TO A DAY THEY NAMED AND COULDN'T
    # HAVE, the correction leads the block. See
    # `_rejected_day_lead_for_day_list` - by the time this directive
    # runs, the `resolve_available_day` result that carried the bad news
    # is no longer the last message, so nothing else would carry it into
    # the reply and the patient's actual question would go unanswered.
    rejected_day_lead = _rejected_day_lead_for_day_list(messages, session_id)
    if rejected_day_lead:
        header = rejected_day_lead + "\n" + header

    if single:
        # PLAIN-PROSE SHAPE, NOT A LABELED BLOCK - explicit, direct
        # request: "أقرب موعد متاح عند [doctor] في [branch]: [day] —
        # [date] من [from] إلى [to] / هل يناسبك هذا الموعد؟", matching
        # this exact reference screenshot rather than the emoji-labeled
        # style used for OTHER message types (e.g. cancellation
        # confirmations). Do not reintroduce the labeled-block shape
        # here without checking first - it was tried and explicitly
        # rejected.
        day = days[0]
        weekday = day.get("weekday_display") or day.get("weekday_name") or ""
        block = (
            f"{header}\n"
            f"🗓️ {weekday} {day.get('date_display') or ''} — "
            f"من {day.get('firstTime') or ''} إلى {day.get('lastTime') or ''}"
        ).strip()
    else:
        block = header + "\n" + "\n".join(lines)


    if data.get("has_more") and single:
        more_instruction = (
            "This is the SOONEST date this doctor has open. Later dates "
            "exist, but do not list them - a weekly clinic just repeats "
            "the same appointment at different dates, which is noise "
            "rather than a choice. After the block, ask ONE question: "
            "does this date suit them - mentioning within that same "
            "single question that you can look for a later date if it "
            "doesn't. Only if they actually ask for another date, call "
            "`list_available_days_for_booking` again with "
            f"offset={data.get('next_offset')} (and limit=3 if they want "
            "to see a few); NEVER invent or calculate a date yourself.\n"
            "When they accept this date, your very next action is to "
            "call `get_available_slots_for_booking` with that day's "
            "from_date/to_date and show the times - not a phone/name "
            "question, which only comes after a time is chosen.\n"
        )
    elif data.get("has_more"):
        more_instruction = (
            "More available days exist beyond these. After the block, ask "
            "ONE question - which of these days suits them - and make "
            "clear in that same single question that you can show further "
            "dates if none of them work. If they say none suit them, call "
            f"`list_available_days_for_booking` again with offset="
            f"{data.get('next_offset')} to show the next few; NEVER invent "
            "or calculate a further date yourself.\n"
        )
    else:
        more_instruction = (
            "These are ALL the days this doctor currently has open - there "
            "are no further dates. After the block, ask ONE question: "
            "which day suits them. If none do, offer another doctor or a "
            "staff handoff rather than suggesting a date of your own.\n"
        )

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "The doctor's genuinely-available days were just looked up. Your "
        "ENTIRE reply must be the exact text between the START/END "
        "markers below, copied verbatim (translate the LABELS/connecting "
        "words only if the conversation is in a different language - keep "
        "the emoji, dates, and times unchanged either way), followed by "
        "exactly one question. The START/END marker lines themselves are "
        "NOT part of the text to copy - never include them, or any line "
        "of dashes/equals-signs, in your actual reply. Do NOT add, "
        "remove, reorder, or re-describe any day.\n"
        f"{more_instruction}\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
    )


# The three list-returning tools whose output the model used to format
# freehand, and the key each one puts its items under.
#
# WHY THIS EXISTS: slots and days already had pre-built blocks, so those
# two look identical for every patient. Doctors, specialties and
# branches did not - so the SAME roster came out as an emoji-numbered
# list for one patient, as "•" bullets for the next, and as a
# comma-separated run-on inside a sentence for a third, purely
# depending on what the model felt like that turn. Confirmed in
# production: a doctor's schedule rendered with "•" bullets in a
# conversation whose every other list was numbered.
#
# The list ITEMS are pre-built here; the numbering is `_numbered_prefix`,
# exactly the same function the slots and days blocks already use, so
# nothing about how lists are numbered changes.
_ENTITY_LIST_TOOLS = {
    "find_available_doctors": ("doctors", "الأطباء المتاحين"),
    "find_best_doctor_in_specialty": ("doctors", "الأطباء المتاحين"),
    "list_specialties": ("specialties", "التخصصات المتاحة"),
    "list_branches_for_specialty": ("branches", "الفروع المتاحة"),
    # `list_available_days_for_booking` returns a BRANCH list when the
    # branch isn't settled yet ("missing_branch"). That list is shown to
    # the patient and answered by number, so it needs the same fixed
    # shape as every other list - it was previously the only one the
    # model formatted freehand.
    "list_available_days_for_booking": ("branches", "الفروع اللي الدكتور متاح فيها"),
}


def _entity_list_line(item: dict) -> str:
    """One list item, with the same fields in the same order every time.

    Only fields the tool actually returned appear - a doctor with no
    degree simply has no degree on their line, rather than a blank
    label or a placeholder.
    """

    name = str(item.get("name") or "").strip()
    if not name:
        return ""

    details = []
    for key in ("degreeName", "specialtyName"):
        value = item.get(key)
        if value and str(value).strip() and str(value).strip() != name:
            details.append(str(value).strip())

    # Branches carry a doctor count instead of a degree/specialty.
    count = item.get("doctorCount")
    if count:
        details.append(f"{count} طبيب")

    # SAY IT, don't hide it. A branch the doctor is rostered at but has
    # nothing open at is still a real answer to "which branches?" - and
    # "fully booked" is the fact that actually helps, because it tells
    # the patient the branch and doctor are right and only the timing is
    # the problem. Omitting it made the assistant deny the branch
    # existed at all.
    if item.get("fully_booked"):
        details.append("محجوز بالكامل حاليًا")

    if details:
        return f"{name} — {' · '.join(details)}"
    return name


def _entity_type_for_tool_call(messages: list, tool_message) -> Optional[str]:
    """The `entity_type` argument a tool was called with, found by
    matching `tool_message`'s `tool_call_id` back to the AIMessage that
    made the call.

    Needed because `match_entity_for_booking` serves both doctors and
    branches from one status ("list"), so the result alone doesn't say
    which kind of list it is - only the call that produced it does.
    """

    tool_call_id = getattr(tool_message, "tool_call_id", None)
    if not tool_call_id:
        return None

    for msg in reversed(messages):
        calls = getattr(msg, "tool_calls", None) or []
        for call in calls:
            if isinstance(call, dict) and call.get("id") == tool_call_id:
                return (call.get("args") or {}).get("entity_type")

    return None


def _build_branches_only_no_doctors_directive(messages: list) -> str:
    """Fires when `list_branches_for_specialty` has just returned AND
    earlier in this same conversation a branch was reported empty
    (`noDoctorsAtBranch` / `not_found_in_branch`).

    That combination is exactly the "the branch you asked about has
    nobody - here are the others" moment, and it is where the reply has
    repeatedly turned into a roster dump. The branch list is what
    answers the question; the doctors at each of those branches are
    not, and they come later, once a branch is picked.
    """

    if not messages:
        return ""

    last = messages[-1]
    if getattr(last, "name", None) != "list_branches_for_specialty":
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if not (data.get("branches") or []):
        return ""

    # Only in the empty-branch follow-up context - a plain "which
    # branches have this specialty?" question is still allowed to show
    # each branch with its own doctors (that grouping is genuinely
    # useful there, and is what NB1d(b) asks for).
    empty_branch_seen = False
    for msg in messages[:-1]:
        if getattr(msg, "type", None) != "tool":
            continue
        try:
            previous = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(previous, dict):
            continue
        if previous.get("noDoctorsAtBranch") or str(previous.get("status") or "").lower() == "not_found_in_branch":
            empty_branch_seen = True
            break

    if not empty_branch_seen:
        return ""

    return (
        "============================================================\n"
        "OFFER THE OTHER BRANCHES WITHOUT LISTING THEIR DOCTORS\n"
        "============================================================\n"
        "The branch the patient asked about has nobody available, and "
        "you are now offering the alternatives. Show ONLY the branch "
        "NAMES, as a short emoji-numbered list, and ask which one they "
        "want.\n\n"
        "Do NOT list the doctors at those branches - not one name, even "
        "though this tool result contains them. CONFIRMED REAL "
        "PRODUCTION FAILURE: eleven doctor names across three branches "
        "went out in a single message to a patient who had asked about "
        "ONE branch. It is unreadable on a phone, and it buries the only "
        "question that matters (which branch instead?) under a roster "
        "nobody asked for. Once they pick a branch, THAT is when its "
        "doctors get shown.\n\n"
    )


def _build_entity_list_directive(messages: list) -> str:
    """
    If the LAST message is a ToolMessage from one of the list-returning
    tools with status "found", pre-build the exact numbered list in code -
    the same approach, and for the same reason, as the slots and days
    blocks above.
    """

    if not messages:
        return ""

    last = messages[-1]
    tool_name = getattr(last, "name", None)

    # `match_entity_for_booking` IN LIST MODE WAS THE ONE LIST-PRODUCING
    # PATH NEVER WIRED INTO THIS BLOCK, AND IT WAS DANGEROUS TO MISS.
    #
    # Every other list tool got this treatment because a freehand list
    # LOOKS wrong to a reader. This one is worse: it produces a list
    # whose NUMBERING silently disagrees with what gets stored for later
    # resolution, so the patient can pick option "1" and end up
    # confirmed for option "2" - a WRONG BRANCH, not just a wrong-looking
    # reply.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: the reply read "1️⃣ الدقي / 2️⃣
    # الشيخ زايد" (freehand, the model's own ordering), the patient typed
    # "1", and the code resolved position 1 against `_remember_list`'s
    # own stored order - which is the tool's ACTUAL array order, not
    # whatever order the model chose to print it in - and returned فرع
    # الشيخ زايد. One booking away from confirming the wrong branch.
    #
    # This tool serves two entity types from one status ("list"), so its
    # heading depends on which - read from the ORIGINAL tool call's
    # `entity_type` argument, not the result.
    if tool_name == "match_entity_for_booking":
        entity_type = _entity_type_for_tool_call(messages, last)
        if entity_type == "doctor":
            items_key, heading = "items", "الأطباء المتاحين"
        elif entity_type == "branch":
            items_key, heading = "items", "الفروع المتاحة"
        else:
            return ""

        try:
            data = json.loads(last.content)
        except (ValueError, TypeError):
            return ""

        if data.get("status") != "list":
            return ""
    else:
        spec = _ENTITY_LIST_TOOLS.get(tool_name)
        if not spec:
            return ""

        items_key, heading = spec

        try:
            data = json.loads(last.content)
        except (ValueError, TypeError):
            return ""

        # "missing_branch" is a list result too: it means "I can't go on
        # until you pick a branch, and here they are". Treated the same as
        # "found" so that list gets the same fixed shape as every other.
        if data.get("status") not in ("found", "missing_branch"):
            return ""

    items = data.get(items_key) or []

    # A SINGLE BRANCH FROM list_branches_for_specialty IS NOT A BRANCH
    # LIST TO SHOW - IT IS THE END OF THAT QUESTION. The real list, the
    # one the patient needs to pick from, is that one branch's DOCTORS -
    # which is exactly what `list_branches_for_specialty` remembers under
    # entity_type="doctor" in this exact case (see its own comment).
    # Rendered here from the SAME nested array, so display and memory
    # can never disagree - the same guarantee every other list in this
    # block already has.
    if tool_name == "list_branches_for_specialty" and len(items) == 1 and items[0].get("doctors"):
        branch_name = items[0].get("name") or ""
        items = items[0]["doctors"]
        items_key = "doctors"
        heading = f"الدكاترة المتاحين في فرع {branch_name}" if branch_name else "الدكاترة المتاحين"

    # `find_best_doctor_in_specialty` returns ONE doctor under "doctor",
    # not a list - it is a recommendation, not a roster, so it is left
    # to the flow's own wording rather than forced into a list of one.
    if not items and data.get("doctor"):
        return ""

    if not isinstance(items, list) or len(items) < 2:
        # A single result is not a list. Numbering one item reads oddly,
        # and the flows already have their own wording for the "only one
        # match" case.
        return ""

    lines = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = _entity_list_line(item)
        if text:
            lines.append(f"{_numbered_prefix(len(lines) + 1)} {text}")

    if len(lines) < 2:
        return ""

    block = f"{heading}:\n" + "\n".join(lines)

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "A list was just looked up. Your ENTIRE reply must be the exact "
        "text between the START/END markers below, copied verbatim "
        "(translate the HEADING and the connecting words only if the "
        "conversation is in another language - keep the numbering, the "
        "names and the order exactly as they are), followed by exactly "
        "ONE question asking which one they'd like. The START/END "
        "marker lines themselves are NOT part of the text to copy - "
        "never include them, or any line of dashes/equals-signs, in "
        "your actual reply.\n\n"
        "Do NOT add, remove, reorder, merge, or re-describe any item, "
        "and do NOT also list them in your own words anywhere else in "
        "the same reply. Do NOT mention any price unless the patient "
        "explicitly asked about cost and `get_doctor_fees` returned it.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
    )


def _build_resolved_day_directive(messages: list, session_id: str) -> str:
    """
    If the LAST message is a ToolMessage from `resolve_available_day`
    with status "found", pre-build the exact confirmation block in code -
    same approach, and same reason, as the slots and days lists above.

    WHY THIS ONE WAS MISSING AND WHY IT MATTERED: every other
    date-bearing result in this file has a pre-built block, so its shape
    is identical for every patient. This one did not, so the reply was
    written freehand each time. Confirmed in production, in a fully
    Arabic conversation:

        تم العثور على موعد متاح مع د. محمد زايد 🧑‍⚕️ الثلاثاء 2026-08-25 📅

    - a raw ISO date inside an Arabic sentence, emoji trailing their
    labels instead of leading them, and a field order that matched no
    other message in the same conversation. The tool now returns
    `date_display`/`weekday_display` (see tools.resolve_available_day),
    and this block pins the shape so the same information cannot come
    out looking different for the next patient.

    The closing question is deliberately left as it already was - asking
    whether they'd like to see that day's times. Only the SHAPE of the
    block above it is being fixed here, not the flow.
    """

    if not messages:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) != "resolve_available_day":
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if data.get("status") != "found":
        return ""

    weekday = data.get("weekday_display") or data.get("weekday_name") or ""
    date_display = data.get("date_display") or data.get("date") or ""

    if not date_display:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    doctor_name = session.get("doctor_display_name") or ""
    branch_name = session.get("branch_display_name") or ""

    first_time = data.get("first_time_display") or ""
    last_time = data.get("last_time_display") or ""

    # DID THE PATIENT CHOOSE THIS DAY, OR ARE WE OFFERING IT?
    #
    # The block below - "أقرب موعد متاح ... هل يناسبك هذا اليوم؟" - is
    # written for the second case, where the assistant proposes a day
    # and needs a yes. Sent to someone who asked for Tuesday by name, it
    # asks them to approve their own choice, and it contradicts the
    # GLOBAL HARD RULE that a SETTLED day is answered with its full time
    # list ("whether the patient named it themselves or accepted a day
    # you offered"). A day the patient named IS settled the moment the
    # tool confirms it exists, so that turn owes them times, not another
    # yes/no.
    named = _named_weekday_in_latest_human(messages)
    patient_chose_this_day = bool(
        named and named["english"].lower() == str(data.get("weekday_name") or "").lower()
    )

    if patient_chose_this_day:
        return (
            "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
            "The patient asked for " + weekday + " themselves, and it is "
            "genuinely available (" + date_display + "). The day is "
            "SETTLED - they chose it, you confirmed it exists, and there "
            "is nothing left to agree on.\n\n"
            "Do NOT ask whether this day suits them. Do NOT ask whether "
            "they would like to see the times. Both hand back a decision "
            "they have already made, and the second is pure dead weight - "
            "they named the day, so they obviously want its times.\n\n"
            "Your ONLY next action is to call "
            "`get_available_slots_for_booking` with this result's own "
            "`from_date`/`to_date`, copied verbatim - never a date you "
            "worked out yourself - and then show every time it returns, "
            "numbered, in the same reply as a one-line confirmation of "
            "the day.\n\n"
            "Call the tool now.\n\n"
        )

    # PLAIN-PROSE SHAPE - explicit, direct request: "أقرب موعد متاح عند
    # [doctor] في [branch]: [day] — [date] من [from] إلى [to] / هل
    # يناسبك هذا الموعد؟", matching this exact reference screenshot -
    # same shape as `_build_available_days_directive`'s single-day
    # case above, so this information looks identical everywhere in
    # the flow no matter which tool call produced it. Do not switch
    # this to a labeled emoji block without checking first - it was
    # tried and explicitly rejected.
    header = "أقرب موعد متاح"
    if doctor_name:
        header = f"أقرب موعد متاح عند {doctor_name}"
    if branch_name:
        header = f"{header} في {branch_name}"
    header += ":"

    # The DAY's overall available-time RANGE, not one narrow slot in
    # it. CONFIRMED REAL PRODUCTION CONFUSION this fixes: without it,
    # the model would grab the single nearest open SLOT's own
    # start/end (e.g. "11:00 - 11:30") and present that 30-minute
    # window as if it were the whole day's offer, instead of the day's
    # actual availability ("11:00 صباحًا - 3:00 مساءً").
    time_range = f" — من {first_time} إلى {last_time}" if first_time and last_time else ""
    block = f"{header}\n🗓️ {weekday} {date_display}{time_range}".strip()

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "The nearest genuinely-available date was just resolved. Your "
        "ENTIRE reply must be the exact text between the START/END "
        "markers below, copied verbatim (translate connecting words "
        "only if the conversation is in another language - keep the "
        "emoji, the date, the time range and the names unchanged "
        "either way), followed by exactly ONE question asking whether "
        "that DAY works for them - e.g. \"هل يناسبك هذا اليوم؟\". The "
        "START/END marker lines themselves are NOT part of the text to "
        "copy - never include them, or any line of dashes/equals-signs, "
        "in your actual reply.\n\n"
        "DO NOT ask \"هل يناسبك هذا اليوم والوقت؟\" (\"does this day AND "
        "TIME suit you\") or anything implying a specific time was "
        "already offered - the block above shows the DAY's whole "
        "working-hours RANGE (e.g. \"من 2:30 مساءً إلى 4:30 مساءً\"), "
        "not one bookable appointment time. Asking about \"the time\" "
        "here reads as if agreeing commits them to a specific slot, "
        "when in fact you still need to call `get_available_slots_for_"
        "booking` next and offer the actual soonest one - CONFIRMED "
        "REAL PATIENT CONFUSION this fixes. Ask about the DAY only.\n\n"
        "NEVER replace the time-range line below with one single "
        "specific appointment time (e.g. a slot's own start-end such as "
        "\"11:00 - 11:30\") - that is one bookable option among many, "
        "not the day's availability, and presenting it as the whole "
        "offer is misleading. The range in the block below (day's "
        "earliest to latest open time) is what belongs here; the "
        "individual bookable times are shown only in the NEXT step, "
        "after they confirm this day works.\n\n"
        "Never write the result's `date` field (the ISO \"YYYY-MM-DD\" "
        "form) anywhere in your reply - it is a machine value. The block "
        "below already contains the date in the form the patient should "
        "see.\n\n"
        "When they say yes, your ONLY next action is to call "
        "`get_available_slots_for_booking` with this result's own "
        "`from_date`/`to_date`, copied verbatim - never a date you "
        "worked out yourself - and then show every time it returns.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
    )


_ARABIC_DAY_NAMES = {
    "monday": "الاثنين", "tuesday": "الثلاثاء", "wednesday": "الأربعاء",
    "thursday": "الخميس", "friday": "الجمعة", "saturday": "السبت", "sunday": "الأحد",
}


def _arabic_time_12h(iso_string: str) -> str:
    """Format an ISO datetime string's TIME portion in Arabic 12-hour
    style (صباحًا/ظهرًا/مساءً), matching the exact reference examples."""

    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string)
    except ValueError:
        return ""

    hour, minute = dt.hour, dt.minute
    hour12 = hour % 12 or 12

    if hour == 12:
        period = "ظهرًا"
    elif hour < 12:
        period = "صباحًا"
    else:
        period = "مساءً"

    return f"{hour12}:00 {period}" if minute == 0 else f"{hour12}:{minute:02d} {period}"


def _fill_booking_ref(template_text: str, booking_ref: str) -> str:
    """Substitute the clinic template's booking-number placeholder with
    the real reference.

    The CSVs write it as `[booking id]`, but clinics edit these by hand,
    so accept the obvious variants rather than silently shipping a
    literal placeholder to a patient. If none is present, the reference
    is appended on its own line - a booking confirmation without its
    number is useless for cancelling or rescheduling later.
    """

    filled = template_text
    for placeholder in ("[booking id]", "[bookingId]", "[booking_id]", "[bookingRefNum]", "[booking ref]"):
        filled = filled.replace(placeholder, str(booking_ref))

    if str(booking_ref) not in filled:
        filled = f"{filled}\n🎉 رقم الحجز: {booking_ref}"

    return filled


# The clinic-authored success templates write their variable parts as
# {placeholders}, in several spellings across the two CSVs. Mapped to
# the fields `_shape_appointment` actually returns, so the block can be
# filled from a real tool result instead of from the model's memory.
_SUCCESS_TEMPLATE_FIELDS = {
    "patientFullName": ("patientFullName",),
    "date": ("date_display",),
    "time 12h ص/م": ("time_display",),
    "time": ("time_display",),
    "doctorName": ("doctorName",),
    "branchName": ("branchName",),
    # RESCHEDULE: new values come from the reschedule result, old values
    # from the lookup that preceded it. Both were previously mapped onto
    # the SAME appointment record, which has only one date on it - so
    # "the new appointment" and "the appointment it replaced" could only
    # ever have shown the same value, and {old_date}/{old_time} had no
    # source at all and reached a patient as literal text.
    # NO FALLBACK to `date_display` here on purpose. Falling back would
    # print the OLD appointment's date under the label "the new
    # appointment" - a card that looks perfectly filled in and is wrong,
    # which is worse than one that visibly fails. If the reschedule
    # result has no new time, the card is skipped entirely (see the
    # leftover-placeholder check below) and the model composes instead.
    "new_date": ("_new_date_display",),
    "new_time": ("_new_time_display",),
    "old_date": ("_old_date_display",),
    "old_time": ("_old_time_display",),
    "bookingRefNum": ("ref",),
}

# Anything still looking like {placeholder} after filling.
_UNFILLED_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_ /ص م\u0600-\u06FF]*\}")


def _fill_success_template(template_text: str, appointment: dict) -> str:
    """Replace every {placeholder} in a clinic's authored success
    template with the matching value from a real appointment record.

    A placeholder with no value available is left EXACTLY as it is
    rather than being blanked: an unfilled `{doctorName}` reaching a
    patient is obvious and gets reported, whereas a silently empty line
    ("👨‍⚕️ الطبيب: ") looks deliberate and hides the fault.
    """

    filled = template_text

    for placeholder, keys in _SUCCESS_TEMPLATE_FIELDS.items():
        value = next(
            (appointment.get(key) for key in keys if appointment.get(key)), None
        )
        if value:
            filled = filled.replace("{" + placeholder + "}", str(value))

    return filled


def _last_appointment_record(messages: list) -> dict:
    """The most recent appointment the booking system actually returned
    in this conversation.

    Scanned backwards from the end, so it is the freshest one - which is
    the record the flows require to have been re-fetched in the same
    turn before anything is cancelled or moved.
    """

    for msg in reversed(messages or []):
        if getattr(msg, "name", None) not in ("lookup_appointment", "check_booking_status"):
            continue
        try:
            data = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        appointment = data.get("appointment")
        if isinstance(appointment, dict):
            return appointment

    return {}


_TERMINAL_SUCCESS_TOOLS = {
    "cancel_appointment": ("msg_cancel_success", "cancellation"),
    "reschedule_appointment": ("msg_rescheduling_success", "reschedule"),
}


def _build_terminal_success_directive(messages: list, templates: dict) -> str:
    """
    If the LAST message is a successful `cancel_appointment` or
    `reschedule_appointment`, pre-build the clinic's OWN authored
    success message in code, filled from the real appointment record.

    WHY THIS EXISTS: `create_new_booking` already had this treatment, so
    a new booking's confirmation is identical for every patient. The two
    other terminal outcomes did not. Both tools return nothing but
    `{"status": "success"}` - no date, no doctor, no branch - so the
    model had to reconstruct every field of the clinic's authored
    template from memory of an earlier turn. That is the single worst
    place in the whole flow to be reconstructing from memory: it is the
    last message the patient receives, it is the one they screenshot and
    keep, and it is stating as fact what the clinic has just done to
    their appointment.

    The wording is the clinic's own, verbatim - only the {placeholders}
    are filled, and only from a real tool result.
    """

    if not messages:
        return ""

    last = messages[-1]
    spec = _TERMINAL_SUCCESS_TOOLS.get(getattr(last, "name", None))
    if not spec:
        return ""

    template_key, label = spec

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if data.get("status") != "success":
        return ""

    template_text = (templates or {}).get(template_key)
    if not template_text or not template_text.strip():
        return ""

    appointment = _last_appointment_record(messages[:-1])
    if not appointment:
        # Nothing real to fill it with. Better to leave the model to its
        # normal instructions than to emit a template full of visible
        # {placeholders}.
        return ""

    # The record fetched BEFORE the change is, by definition, the OLD
    # appointment - that is what the flow looked up in order to change
    # it. Its date/time therefore fill {old_date}/{old_time}, while the
    # NEW ones come from the reschedule result itself.
    values = dict(appointment)
    values["_old_date_display"] = appointment.get("date_display")
    values["_old_time_display"] = appointment.get("time_display")

    for key in ("new_date_display", "new_time_display"):
        if data.get(key):
            values[f"_{key}"] = data[key]

    block = _fill_success_template(
        template_text.replace("\r\n", "\n").replace("\r", "\n"), values,
    ).strip()

    if not block:
        return ""

    leftover = _UNFILLED_PLACEHOLDER_RE.findall(block)
    if leftover:
        # A card with a hole in it is worse than no card: it is the last
        # message the patient receives and the one they keep. Confirmed
        # in production - "تم إلغاء الموعد السابق المحدد بتاريخ
        # {old_date} الساعة {old_time}" went out exactly like that.
        #
        # Falling back to the model's normal instructions produces a
        # sentence that at least reads as a sentence, and the log line
        # names the field that had no source.
        logger.error(
            "terminal success template for %s has unfillable placeholder(s) %s - "
            "falling back to a composed reply rather than sending a card with holes in it",
            label, leftover,
        )
        return ""

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        f"The {label} has just completed successfully. Your ENTIRE reply "
        "must be the exact text between the START/END markers below, "
        "copied verbatim - it is the clinic's own authored confirmation "
        "message and it is already filled in with this patient's real "
        "details. The START/END marker lines themselves are NOT part of "
        "the text to copy - never include them, or any line of dashes/"
        "equals-signs, in your actual reply.\n\n"
        "Do NOT add a sentence of your own before or after it, do NOT "
        "reword it, and do NOT restate any of its details in your own "
        "words anywhere in the same reply. Do NOT add a question - this "
        "message already closes the conversation on its own terms.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
    )


def _build_booking_success_display_directive(messages: list, templates: dict) -> str:
    """
    If the LAST message is a ToolMessage from `create_new_booking` with
    status "success", pre-build the EXACT confirmation block in code -
    using the REAL booking_ref from the tool result (never inventable)
    and the patient's real name from that same tool call's own
    arguments - matching a specific format requested directly by the
    clinic. This also reinforces, at the code level, that a booking
    confirmation always carries its real reference number.
    """

    if not messages:
        return ""

    last = messages[-1]
    if getattr(last, "name", None) != "create_new_booking":
        return ""

    try:
        data = json.loads(last.content)
    except (json.JSONDecodeError, TypeError):
        return ""

    if data.get("status") != "success":
        return ""

    booking_ref = data.get("booking_ref")
    if not booking_ref:
        return ""

    patient_name = ""
    tool_call_id = getattr(last, "tool_call_id", None)
    for msg in reversed(messages[:-1]):
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if tc.get("id") == tool_call_id and tc.get("name") == "create_new_booking":
                patient_name = (tc.get("args") or {}).get("patient_full_name", "")
                break
        if patient_name:
            break

    clinic_name = (templates or {}).get("_clinic_name_ar") or (templates or {}).get("_clinic_name") or ""
    clinic_name = clinic_name.strip()

    if not clinic_name:
        clinic_line = "نشكر ثقتك بنا 🌷"
    elif clinic_name.startswith("مستشفى") or clinic_name.startswith("مركز") or clinic_name.startswith("عيادات"):
        # The Arabic clinic name usually already carries its own
        # "مستشفى"/"مركز" prefix - adding another produced
        # "نشكر ثقتك بمستشفى مستشفى ...".
        clinic_line = f"نشكر ثقتك بـ{clinic_name} 🌷"
    else:
        clinic_line = f"نشكر ثقتك بمستشفى {clinic_name} 🌷"

    greeting_line = f"✅ عزيزي/عزيزتي {patient_name}" if patient_name else "✅"

    # The MIDDLE of this block is the clinic's own authored
    # msg_booking_success template, read fresh from config every turn
    # and reproduced verbatim with only [booking id] substituted -
    # rather than a copy of its wording pasted into this file, which
    # would silently stop matching the moment the clinic edited the CSV.
    # The name line above it and the thank-you line below it are the
    # approved wrapper this clinic asked for around that template.
    success_template = (templates or {}).get("msg_booking_success") or ""
    success_template = success_template.replace("\r\n", "\n").replace("\r", "\n").strip()
    success_body = "\n".join(line.strip() for line in success_template.split("\n") if line.strip())

    if success_body:
        # Drop only the leading ✅ marker (the personalized greeting line
        # above already carries one) - never the line itself, which
        # holds the template's actual confirmation sentence.
        if success_body.startswith("✅"):
            success_body = success_body[1:].lstrip()
        success_body = _fill_booking_ref(success_body, booking_ref)
    else:
        success_body = (
            "تم تأكيد حجز موعدك بنجاح\n"
            f"🎉 رقم الحجز: {booking_ref}\n"
            "📌 احتفظ برقم الحجز — تقدر تستخدمه لإلغاء الموعد أو إعادة جدولة الموعد."
        )

    block = f"{greeting_line}\n{success_body}\n{clinic_line}"

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "The booking was just created successfully. Your ENTIRE reply "
        "must be EXACTLY the text between the START/END markers below, "
        "copied verbatim - translate only if the conversation is in a "
        "different language (keep the emoji and the actual booking_ref "
        "value unchanged either way). The START/END marker lines "
        "themselves are NOT part of the text to copy - never include "
        "them, or any other line of dashes/equals-signs, in your actual "
        "reply. Do NOT add anything else, anywhere in the reply.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
    )


def _build_booking_confirmation_requires_tool_directive(messages: list, session_id: str) -> str:
    """
    If a NEW BOOKING is deep in progress (doctor AND branch both
    confirmed in this conversation's booking session), the CURRENT last
    message is a fresh human message, and `get_patient_info` has been
    called more recently than `create_new_booking` (i.e. patient info
    was collected - the review card step - but the booking itself has
    not actually been created yet) - inject a hard reminder that
    claiming success requires actually calling `create_new_booking`.

    WHY THIS EXISTS: confirmed real production failure, the most severe
    version of a repeated pattern - the model replied "✅ booking
    confirmed" with a full summary, with ZERO tool calls made that turn
    (no trace of create_new_booking in the logs at all). This means no
    real booking existed in the system despite the user being told it
    was successful - a serious trust and safety issue, not just a
    scheduling inconvenience.
    """

    if not messages or not session_id:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    if not isinstance(messages[-1], _HumanMessage):
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id)
    if not session or not (session.get("doctor_id") and session.get("branch_id")):
        return ""

    for msg in reversed(messages[:-1]):
        name = getattr(msg, "name", None)
        if name == "create_new_booking":
            return ""
        if name == "get_patient_info":
            return (
                "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
                "Patient info was already collected for this new booking, "
                "but the booking has NOT actually been created yet. You "
                "are NOT allowed to say a booking is confirmed, "
                "successful, or booked - in any form, in any wording - "
                "without first calling `create_new_booking` THIS turn and "
                "using its real returned booking_ref. Confirmed real "
                "failure: claiming success with zero tool calls left no "
                "real booking in the system at all, while the patient was "
                "told otherwise. If the user just confirmed \"yes\" to the "
                "review card, call `create_new_booking` now.\n\n"
            )

    return ""


def _build_show_soonest_day_directive(messages: list, session_id: str) -> str:
    """When a doctor (and branch) are settled, the next message must SHOW
    the soonest available date - never ask the patient which day they
    want.

    WHY THIS EXISTS: confirmed real production failure, repeatedly. With
    د. طه مبروك and فرع الشيخ زايد both confirmed, the reply was "ممكن
    تخبرني اليوم اللي تفضله للحجز؟ مثلاً الجمعة، السبت؟" - asking the
    patient to guess. They have no way of knowing when the doctor works
    or which days still have space, so a wrong guess dead-ends the
    booking; and in earlier runs the model filled that gap by inventing
    weekdays outright.

    STEP NB3 already says to call `list_available_days_for_booking`
    immediately and show the date. It was not followed, so the
    instruction is injected here, at the exact turn it applies to,
    where it competes with nothing else."""

    if not messages or not session_id:
        return ""

    # THE PATIENT ASKED FOR A SPECIFIC DAY - STAND DOWN.
    #
    # This directive and `_build_named_day_directive` give directly
    # opposite instructions ("show the soonest date" vs "check the day
    # they named"), and both are correct in their own situation. When a
    # day has actually been named, this one must not be in the prompt at
    # all: it is by far the more emphatic of the two, and with both
    # present it won - the patient asked for Tuesday and was shown
    # Sunday, which is the exact complaint this pass exists to fix. Same
    # precedent as the appointment-display / wrong-tool pair in
    # `_run_agent`: when two directives contradict, one of them is
    # suppressed in code rather than left to compete.
    if _build_named_day_directive(messages, session_id):
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]

    if not isinstance(last, _HumanMessage):
        # ALSO FIRE ON THE TOOL TURN THAT JUST CONFIRMED THE DOCTOR.
        #
        # The confirmation arrives as a ToolMessage, not a human one -
        # the patient typed "1", `match_entity_for_booking` resolved it,
        # and the model is now composing the reply with that tool result
        # as the last message. Requiring a HumanMessage meant this
        # directive sat inert on precisely the turn it was written for.
        #
        # CONFIRMED REAL PRODUCTION FAILURE: with فرع الدقي ALREADY
        # confirmed and د. محمد زايد just resolved from position 1, the
        # reply was "اخترت د. محمد زايد ✅ / تحب تحجز في فرع معيّن، ولا
        # أعرض لك كل الفروع المتاحة عند د. محمد زايد؟" - asking which
        # branch when the branch had been settled several turns earlier,
        # instead of showing his available days.
        if getattr(last, "name", None) != "match_entity_for_booking":
            return ""
        try:
            _confirm_data = json.loads(last.content)
        except (ValueError, TypeError):
            return ""
        if not (isinstance(_confirm_data, dict) and _confirm_data.get("matched")
                and not _confirm_data.get("needsConfirmation")):
            return ""
        _session_now = tools._BOOKING_SESSIONS.get(session_id) or {}
        if not (_session_now.get("doctor_id") and _session_now.get("branch_id")):
            return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    doctor_confirmed = bool(session.get("doctor_id"))

    # A doctor can be settled in the CONVERSATION without ever having
    # been confirmed into the booking session - the medical flow shows a
    # doctor, the patient says yes, and nothing calls
    # `match_entity_for_booking`. Confirmed real production failure: with
    # د. طه مبروك agreed and فرع الشيخ زايد resolved, session["doctor_id"]
    # was still empty, so every session-based guard here sat inert and
    # the reply asked the patient to name a day. So this fires on either
    # signal, and tells the model to confirm the doctor first when the
    # session is the part that's missing.
    doctor_seen = False
    for msg in reversed(messages[:-1]):
        name = getattr(msg, "name", None)
        if name in ("list_available_days_for_booking", "get_available_slots_for_booking",
                    "resolve_available_day", "create_new_booking"):
            # Already past this step - days/slots have been handled.
            return ""
        if name in ("find_available_doctors", "match_entity_for_booking",
                    "list_branches_for_specialty", "find_best_doctor_in_specialty"):
            doctor_seen = True

    if not (doctor_confirmed or doctor_seen):
        return ""

    confirm_first = (
        ""
        if doctor_confirmed
        else ("The doctor has NOT been confirmed into the booking session yet - "
              "call `match_entity_for_booking(entity_type=\"doctor\")` with their "
              "exact name FIRST (that call is what saves them), then continue "
              "below in the same turn.\n\n")
    )

    return (
        "============================================================\n"
        "SHOW THE SOONEST DATE - DO NOT ASK WHICH DAY THEY WANT\n"
        "============================================================\n"
        + confirm_first
        + "A doctor has been settled for this booking. Your ONLY next "
        "action is to call `list_available_days_for_booking` and SHOW the "
        "soonest date it returns, then ask whether that date suits them.\n\n"
        "IF A BRANCH IS ALREADY CONFIRMED, THE BRANCH QUESTION IS OVER. "
        "Do not ask \"تحب تحجز في فرع معيّن، ولا أعرض لك كل الفروع "
        "المتاحة عند د. [اسم]؟\", and do not offer that doctor's other "
        "branches - the patient chose a branch earlier and has not asked "
        "to change it. Confirm the doctor in one short line and show the "
        "days in that SAME message. CONFIRMED REAL PRODUCTION FAILURE: "
        "with فرع الدقي settled several turns earlier and د. محمد زايد "
        "just picked from the list, the reply was \"اخترت د. محمد زايد "
        "✅ / تحب تحجز في فرع معيّن، ولا أعرض لك كل الفروع المتاحة عند "
        "د. محمد زايد؟\" - re-opening a question that was already "
        "answered, one step from showing real dates.\n\n"
        "You are NOT allowed to ask \"which day would you prefer?\", to "
        "suggest example days, or to name any weekday before that tool "
        "has returned. The patient does not know when this doctor works "
        "or which days still have space - asking them to guess is asking "
        "for information only the booking system has. Confirmed real "
        "production failure: with the doctor and branch both settled, the "
        "reply asked \"ممكن تخبرني اليوم اللي تفضله؟ مثلاً الجمعة، "
        "السبت؟\" - two weekdays that came from nowhere.\n\n"
        "Call the tool now.\n\n"
    )


# ============================================================
# THE DAY THE PATIENT NAMED THEMSELVES
# ============================================================
#
# WHY THIS WHOLE SECTION EXISTS: a booking request routinely arrives
# with the day already in it -
#
#     "عاوزه احجز معاد مع دكتور احمد العقيل يوم التلات"
#
# and the flow used to throw the day away. NB3's directive
# (`_build_show_soonest_day_directive`) fires the instant a doctor is
# settled and says, in the strongest terms available, "call
# `list_available_days_for_booking` and show the SOONEST date". That is
# exactly right when the patient has expressed no preference, and
# exactly wrong here: they named Tuesday, and they got offered Sunday.
#
# The two outcomes the patient actually needs are:
#   - the doctor DOES have Tuesday open -> show Tuesday's times, rather
#     than starting the day conversation over from the beginning;
#   - the doctor does NOT work Tuesday -> say so in plain words, then
#     show the days they DO work, exactly as the normal flow would.
#
# Neither is safe to answer from the model's own head, so both go
# through `resolve_available_day`, which already distinguishes "found"
# from "fully_booked" from "not_found". What was missing was anything
# telling the model to call it at all on this turn, plus a deterministic
# shape for the "he doesn't work that day" sentence.

# The specialists that can actually start a NEW booking - i.e. the ones
# bound to `match_entity_for_booking` AND `resolve_available_day`. Every
# named-day / multi-intent rule below is written in terms of those two
# calls, so it belongs only to these.
_NEW_BOOKING_AGENTS = ("booking", "concierge")

# The specialists that can look an EXISTING booking up - i.e. the ones
# bound to `lookup_appointment`. The reference/phone rules are written
# in terms of that call, so they belong only to these.
_EXISTING_BOOKING_AGENTS = ("cancel", "reschedule", "concierge")

_DAY_INTENT_TOOLS = (
    "resolve_available_day", "get_available_slots_for_booking",
    "select_appointment_slot", "create_new_booking",
)


def _latest_human_index(messages: list) -> int:
    """Index of the most recent HumanMessage, or -1."""

    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return i
    return -1


def _named_weekday_in_latest_human(messages: list) -> Optional[dict]:
    """The weekday the patient named in their OWN most recent message,
    or None.

    Deliberately reads only the latest human message: a day mentioned
    five turns ago has either been acted on already or been superseded,
    and treating it as a live request is how a patient ends up pushed
    back onto a day they had moved off.

    Colloquial spellings resolve here just as they do inside the tools -
    `tools.resolve_weekday_index` is the single source of truth for what
    counts as a day name, so a directive and a tool can never disagree
    about whether "التلات" is Tuesday.
    """

    index = _latest_human_index(messages)
    if index < 0:
        return None

    content = getattr(messages[index], "content", "")
    text = content if isinstance(content, str) else str(content)
    if not text.strip():
        return None

    weekday = tools.resolve_weekday_index(text)
    if weekday is None:
        return None

    english = ["Monday", "Tuesday", "Wednesday", "Thursday",
               "Friday", "Saturday", "Sunday"][weekday]

    return {
        "index": weekday,
        "english": english,
        "display": _ARABIC_DAY_NAMES.get(english.lower(), english),
        "human_index": index,
    }


def _tool_results_since_latest_human(messages: list, tool_names: tuple) -> list:
    """Every ToolMessage from `tool_names` that arrived AFTER the
    patient's latest message - i.e. work done in response to what they
    just said, not to something earlier in the conversation."""

    start = _latest_human_index(messages)
    if start < 0:
        return []

    return [
        msg for msg in messages[start + 1:]
        if getattr(msg, "name", None) in tool_names
    ]


def _build_named_day_directive(messages: list, session_id: str) -> str:
    """The patient named a day. Route this turn through
    `resolve_available_day` for THAT day instead of the soonest-date
    path, and spell out what to do with each of its results.

    Fires only while the day is still unanswered - as soon as
    `resolve_available_day` (or the slots call, or the booking itself)
    has run for this message, the day has been dealt with and the normal
    directives take over again.

    It fires whether or not the doctor is already in the booking
    session. When they are not, the instruction is to confirm the doctor
    FIRST and then check the day in the same turn - because the single
    most common shape of this request names both at once, and splitting
    it across two turns is exactly the "starting over from the
    beginning" this is here to stop.
    """

    if not messages or not session_id:
        return ""

    named = _named_weekday_in_latest_human(messages)
    if not named:
        return ""

    # Already handled on this turn - the day question is closed.
    if _tool_results_since_latest_human(messages, _DAY_INTENT_TOOLS):
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    # THE DAY IS ALREADY ON THE TABLE, WITH REAL DATES ATTACHED.
    #
    # When a day list has just been shown and the patient answers by
    # naming one of ITS days, the dates are already in hand and
    # `_build_day_confirmation_requires_tool_directive` already says the
    # right thing: take that day's from_date/to_date verbatim and call
    # `get_available_slots_for_booking`. Firing here as well would put a
    # second, differently-worded instruction ("call
    # `resolve_available_day`") beside it - the same
    # competing-directives problem this whole area has been bitten by
    # before, for the sake of re-deriving a date the tools already gave
    # us. Days NOT in that list still come here, which is exactly what
    # NB4 prescribes.
    last_list = session.get("last_list") or {}
    if last_list.get("entity_type") == "day":
        for day in last_list.get("items") or []:
            if str(day.get("weekday_name") or "").strip().lower() == named["english"].lower():
                return ""

    doctor_confirmed = bool(session.get("doctor_id"))

    # A day only means something once there is a doctor to check it
    # against. Without one - and without one even on the table - this is
    # a day mentioned in some other context ("كنت جاي الخميس اللي فات"),
    # not a booking preference, and forcing a tool call would be worse
    # than leaving the turn alone.
    doctor_on_the_table = doctor_confirmed
    if not doctor_on_the_table:
        for msg in reversed(messages):
            if getattr(msg, "name", None) in (
                "match_entity_for_booking", "find_available_doctors",
                "find_best_doctor_in_specialty", "list_branches_for_specialty",
            ):
                doctor_on_the_table = True
                break

    if not doctor_on_the_table:
        return ""

    doctor_name = session.get("doctor_display_name") or ""
    branch_name = session.get("branch_display_name") or ""
    who = " (" + doctor_name + ")" if doctor_name else ""

    confirm_first = "" if doctor_confirmed else (
        "THE DOCTOR IS NOT IN THE BOOKING SESSION YET. Call "
        "`match_entity_for_booking(user_input=<the doctor's name exactly "
        "as the patient typed it>, entity_type=\"doctor\")` FIRST - that "
        "call is what saves them - and then check the day, in this same "
        "turn. Do not stop after confirming the doctor and make the "
        "patient repeat a day they already gave you.\n\n"
    )

    fully_booked_line = (
        "  - \"fully_booked\": the doctor DOES work " + named["display"]
        + (" at " + branch_name if branch_name else "")
        + ", but every slot is taken. Say exactly that, then call "
        "`list_available_days_for_booking` in the same turn and show the "
        "days that are open.\n"
    )

    return (
        "============================================================\n"
        "THE PATIENT NAMED A DAY - CHECK THAT DAY, NOT THE SOONEST ONE\n"
        "============================================================\n"
        "Their latest message names a specific weekday: "
        + named["display"] + " (" + named["english"] + ").\n\n"
        + confirm_first
        + "Your ONLY next action for the day is:\n"
        "    resolve_available_day(weekday_name=\"" + named["english"] + "\")\n\n"
        "Do NOT call `list_available_days_for_booking` on this turn. That "
        "tool answers \"when is the soonest?\" - a question the patient "
        "did not ask. Calling it here offers them some other date and "
        "silently drops the day they chose, which reads as if you had "
        "not listened.\n\n"
        "Do NOT ask them which day they want. They just told you. Do NOT "
        "ask them to confirm the day back to you either - checking it IS "
        "the confirmation.\n\n"
        "Do NOT state, out of your own knowledge, whether the doctor"
        + who + " works " + named["display"] + ". You do not know that "
        "until the tool answers. Claiming the doctor does not come in on "
        + named["display"] + " - or that they do - before this call is a "
        "fabrication, and the most damaging kind here, because the "
        "patient will plan their day around it.\n\n"
        "WHAT TO DO WITH EACH RESULT:\n"
        "  - \"found\": that day genuinely has open slots. Confirm the day "
        "in one short line and, in the SAME turn, call "
        "`get_available_slots_for_booking` with the result's own "
        "`from_date`/`to_date` copied verbatim, then show the times. Do "
        "not go back to a day list - the day is settled.\n"
        + fully_booked_line
        + "  - \"not_found\": the doctor has no clinic on " + named["display"]
        + " here at all. Say exactly that - plainly, without apologising "
        "at length - and then call `list_available_days_for_booking` in "
        "the same turn and show the days they DO work. One message: the "
        "correction and the real days together.\n"
        "  - \"unrecognized_day\": ask which day they meant. Never guess.\n"
        "  - \"missing_branch\": the branch has to be settled before a day "
        "can be checked - handle the branch, then come straight back to "
        "this day.\n\n"
        "In every one of those cases the day the patient named is the "
        "subject of your reply. Never answer it with a different date as "
        "though the question had been \"when is your next opening?\".\n\n"
    )


# The exact opening sentence for "the day you asked for isn't bookable".
# Deterministic for the same reason every other patient-facing block in
# this file is: this sentence IS the answer to what they asked, and
# written freehand it came out differently every time - and sometimes
# not at all.
_DAY_UNAVAILABLE_LEAD = {
    ("not_found", "ar"): "{doctor} ما عنده عيادة يوم {day}{branch}.",
    ("not_found", "en"): "{doctor} doesn't hold a clinic on {day}{branch}.",
    ("fully_booked", "ar"): "{doctor} بيجي يوم {day}{branch}، بس كل المواعيد محجوزة.",
    ("fully_booked", "en"): "{doctor} does work on {day}{branch}, but every slot is already taken.",
}

_DAY_UNAVAILABLE_NO_DOCTOR = {
    ("not_found", "ar"): "ما فيه عيادة يوم {day}{branch}.",
    ("not_found", "en"): "There is no clinic on {day}{branch}.",
    ("fully_booked", "ar"): "يوم {day}{branch} كل المواعيد محجوزة.",
    ("fully_booked", "en"): "{day}{branch} is fully booked.",
}


def _day_unavailable_lead(messages: list, session_id: str, data: dict) -> str:
    """The one sentence that answers "what about the day I asked for?",
    built from a `resolve_available_day` payload.

    Shared by the directive that fires on that tool result AND by the
    day-list directive that fires one tool call later - see
    `_rejected_day_lead_for_day_list` for why the second one needs it.
    """

    status = data.get("status")
    if status not in ("not_found", "fully_booked"):
        return ""

    named = _named_weekday_in_latest_human(messages)
    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    # The day name comes from the TOOL when it supplied one
    # ("fully_booked" does), and from the patient's own message
    # otherwise ("not_found" carries no day back) - never from the
    # model's memory of the conversation.
    day_display = data.get("weekday_display") or (named or {}).get("display") or ""
    if not day_display:
        return ""

    language = "en" if _detect_target_language(messages) == "en" else "ar"
    doctor_name = session.get("doctor_display_name") or ""
    branch_name = session.get("branch_display_name") or ""

    if language == "ar":
        branch_part = (" في " + branch_name) if branch_name else ""
    else:
        branch_part = (" at " + branch_name) if branch_name else ""

    # A separate wording for the no-name case rather than formatting an
    # empty {doctor} into the normal one: "الدكتور  ما عنده عيادة" with
    # a hole where the name should be is worse than a sentence written
    # for that situation in the first place.
    table = _DAY_UNAVAILABLE_LEAD if doctor_name else _DAY_UNAVAILABLE_NO_DOCTOR
    return table[(status, language)].format(
        doctor=doctor_name, day=day_display, branch=branch_part,
    )


def _rejected_day_lead_for_day_list(messages: list, session_id: str) -> str:
    """The same sentence, recovered on the NEXT tool call.

    WHY THIS IS NEEDED AND WHY IT IS EASY TO MISS: the "he doesn't work
    Tuesday" directive fires on the `resolve_available_day` result. The
    model then correctly calls `list_available_days_for_booking`, and by
    the time the reply is actually WRITTEN the last message is that
    second tool result - so the first directive is long gone and only
    the day-list block remains. Without this, the finished reply shows
    the real days and never mentions Tuesday at all: the patient asked
    a direct question and simply never got an answer to it.

    Scans back only as far as the patient's own latest message, so a day
    rejected earlier in the conversation cannot resurface as a preamble
    to an unrelated day list later on.
    """

    for msg in reversed(_tool_results_since_latest_human(messages, ("resolve_available_day",))):
        try:
            data = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            lead = _day_unavailable_lead(messages, session_id, data)
            if lead:
                return lead
    return ""


def _build_day_unavailable_directive(messages: list, session_id: str) -> str:
    """Fires on the `resolve_available_day` result that says the named
    day is not bookable - either the doctor does not work it
    ("not_found") or it is full ("fully_booked").

    Pins BOTH halves of the answer: the exact sentence explaining what
    happened to the day they asked for, and the instruction to fetch and
    show the real days in the same turn - rather than stopping at the
    bad news, or asking them to name another day to guess at.
    """

    if not messages:
        return ""

    last = messages[-1]
    if getattr(last, "name", None) != "resolve_available_day":
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    status = data.get("status")

    if status == "unrecognized_day":
        return (
            "============================================================\n"
            "THAT WASN'T A DAY OF THE WEEK - ASK, DO NOT GUESS\n"
            "============================================================\n"
            "The word passed as the weekday was not recognised as a day. "
            "Ask the patient which day they meant, in one short question. "
            "Do NOT pick a day for them, and do NOT fall back to showing "
            "the soonest available date as if no day had been mentioned.\n\n"
        )

    if status not in ("not_found", "fully_booked"):
        return ""

    lead = _day_unavailable_lead(messages, session_id, data)
    if not lead:
        return ""

    return (
        "============================================================\n"
        "THE DAY THEY ASKED FOR ISN'T BOOKABLE - SAY SO, THEN SHOW THE REAL DAYS\n"
        "============================================================\n"
        "Open your reply with EXACTLY this sentence, unchanged:\n\n"
        "    " + lead + "\n\n"
        "Then, in this SAME turn, call `list_available_days_for_booking` "
        "and show what it returns, formatted by the directive that "
        "arrives with it. The patient asked about one specific day; they "
        "are owed a straight answer about that day AND a way forward, in "
        "the same message.\n\n"
        "Do NOT stop at the bad news and wait. Do NOT ask them to name "
        "another day - having no way of knowing which days exist is what "
        "put them here. Do NOT name any other day yourself before that "
        "tool has returned; you do not yet know a single one of them.\n\n"
        "Do NOT apologise beyond what the sentence above already does, "
        "and do not add a second question after the day list's own.\n\n"
    )


# ============================================================
# ONE MESSAGE, SEVERAL ANSWERS
# ============================================================
#
# WHY: the booking flow is written as a ladder - specialty, doctor,
# branch, day, time, phone, name - and the model climbs it one rung per
# message. That is right when the patient supplies one thing at a time.
# It is wrong, and reads as though nobody is listening, when a single
# WhatsApp message supplies four of them at once:
#
#     "عاوزه احجز معاد مع دكتور احمد العقيل يوم التلات في فرع الدقي"
#
# Answering that with "تحب تبدأ بالتخصص ولا بالدكتور؟" makes the patient
# say again what they already said. The ladder is not the problem; the
# problem is starting at the bottom of it when the patient has already
# climbed most of the way.
#
# This directive does no resolving of its own - it never claims a doctor
# or branch EXISTS, only that the patient MENTIONED one, quoting their
# own words back. Whether the name is real is still decided by
# `match_entity_for_booking`, exactly as before. What it changes is
# which rung the turn starts on, and it names each piece explicitly so
# the model cannot ask for one of them again in the same breath.

# A phone number: at least 9 digits, allowing the separators people
# actually type. Deliberately not a strict format check - that is
# `validate_phone_format`'s job; this only answers "did they give one?".
_MULTI_INTENT_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")

_MULTI_INTENT_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")

# An explicit clock time: "10:30", "الساعة 10", "10 صباحا", "3 pm".
_MULTI_INTENT_TIME_RE = re.compile(
    r"\d{1,2}\s*:\s*\d{2}|"
    r"(?:الساعه|الساعة)\s*\d{1,2}|"
    r"\d{1,2}\s*(?:صباحا|صباحًا|مساء|مساءً|ظهرا|ظهرًا|am|pm)\b",
    re.IGNORECASE,
)

# The cue words that introduce a name, and the words that end it. The
# stop list is what keeps "دكتور احمد العقيل يوم التلات" from being read
# as a doctor called "احمد العقيل يوم التلات".
_DOCTOR_CUE_RE = re.compile(
    r"(?:الدكتوره|الدكتورة|الدكتور|دكتوره|دكتورة|دكتور|د\.|استشاري|استشارية|dr\.?|doctor)\s+",
    re.IGNORECASE,
)
_BRANCH_CUE_RE = re.compile(r"(?:فرع|الفرع|branch)\s+", re.IGNORECASE)
_SPECIALTY_CUE_RE = re.compile(
    r"(?:تخصص|التخصص|عياده|عيادة|قسم|specialty|clinic)\s+", re.IGNORECASE,
)

_FRAGMENT_STOP_WORDS = (
    "يوم", "في", "فرع", "الفرع", "الساعه", "الساعة", "بكره", "بكرة",
    "عشان", "علشان", "لان", "لأن", "ومحتاج", "ومحتاجه", "واحجز",
    "وعايز", "وعاوز", "on", "at", "in", "branch", "day", "please",
    "لو", "ممكن", "بس", "او", "أو", "ولا",
)


def _fragment_after_cue(text: str, cue_re) -> str:
    """The words following a cue like "دكتور", stopped at the first word
    that clearly starts something else ("يوم", "في", "فرع", a digit).

    Returns the patient's own characters, untouched - this is quoted
    back to the model as "what they wrote", never treated as a resolved
    entity."""

    match = cue_re.search(text or "")
    if not match:
        return ""

    tail = (text or "")[match.end():]
    words = []
    for word in re.split(r"\s+", tail.strip()):
        clean = word.strip(".,،؟?!:؛()[]")
        if not clean:
            break
        if clean.lower() in _FRAGMENT_STOP_WORDS:
            break
        if re.search(r"\d", clean):
            break
        words.append(clean)
        if len(words) >= 4:
            break

    return " ".join(words).strip()


def _build_multi_intent_directive(messages: list, session_id: str) -> str:
    """Lists, in the system prompt, every distinct piece of booking
    information the patient's latest message contains - and forbids
    asking for any of them again.

    Fires only when there are at least TWO. One piece of information is
    the ordinary case the flow already handles well; two or more is the
    case where it used to ask for something it had just been told.
    """

    if not messages:
        return ""

    index = _latest_human_index(messages)
    if index < 0:
        return ""

    content = getattr(messages[index], "content", "")
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return ""

    # Nothing to harvest from a bare number, a "نعم", or a one-word
    # reply - and those are exactly the messages where a false reading
    # would do damage, because they are ANSWERS to a question rather
    # than fresh requests.
    if len(text.split()) < 3:
        return ""

    found = []

    doctor_fragment = _fragment_after_cue(text, _DOCTOR_CUE_RE)
    if doctor_fragment:
        found.append((
            "A DOCTOR",
            doctor_fragment,
            "resolve it with `match_entity_for_booking(user_input=\""
            + doctor_fragment + "\", entity_type=\"doctor\")`. Never ask "
            "them to send the doctor's name again.",
        ))

    branch_fragment = _fragment_after_cue(text, _BRANCH_CUE_RE)
    if branch_fragment:
        found.append((
            "A BRANCH",
            branch_fragment,
            "resolve it with `match_entity_for_booking(user_input=\""
            + branch_fragment + "\", entity_type=\"branch\")`. Never ask "
            "\"which branch?\" after this.",
        ))

    specialty_fragment = _fragment_after_cue(text, _SPECIALTY_CUE_RE)
    if specialty_fragment:
        found.append((
            "A SPECIALTY OR SERVICE",
            specialty_fragment,
            "act on it directly - it is more specific than the question "
            "\"تحب تبدأ بالتخصص ولا بالدكتور؟\", which must not be asked "
            "once this is on the table.",
        ))

    named = _named_weekday_in_latest_human(messages)
    if named:
        found.append((
            "A DAY",
            named["display"] + " (" + named["english"] + ")",
            "check THAT day with `resolve_available_day(weekday_name=\""
            + named["english"] + "\")` - not the soonest available date, "
            "and never by asking them which day they want.",
        ))

    if _MULTI_INTENT_TIME_RE.search(text):
        found.append((
            "A PREFERRED TIME",
            _MULTI_INTENT_TIME_RE.search(text).group(0).strip(),
            "hold on to it. When the real slots come back, point out the "
            "one nearest what they asked for instead of making them read "
            "the whole list again. If nothing is near it, say so.",
        ))

    phone_match = _MULTI_INTENT_PHONE_RE.search(text)
    if phone_match:
        found.append((
            "A PHONE NUMBER",
            phone_match.group(0).strip(),
            "use this one. Do not ask for a phone number again, and do "
            "not ask whether to use the WhatsApp number instead.",
        ))

    email_match = _MULTI_INTENT_EMAIL_RE.search(text)
    if email_match:
        found.append((
            "AN EMAIL",
            email_match.group(0).strip(),
            "use it as given; email is optional, so never ask twice.",
        ))

    if len(found) < 2:
        return ""

    lines = []
    for label, value, instruction in found:
        lines.append("  - " + label + ": \"" + value + "\"\n      -> " + instruction)

    return (
        "============================================================\n"
        "THEY ANSWERED SEVERAL QUESTIONS AT ONCE - USE ALL OF IT\n"
        "============================================================\n"
        "The patient's latest message carries more than one piece of "
        "booking information. Everything below is something they have "
        "ALREADY told you:\n\n"
        + "\n".join(lines) + "\n\n"
        "Start from the earliest step that is still genuinely unanswered "
        "AFTER all of the above is taken into account - not from the "
        "beginning of the flow. Asking for any item on that list is "
        "asking them to repeat themselves, and it is the single fastest "
        "way to make this conversation feel automated and deaf.\n\n"
        "Chain the tool calls in THIS turn rather than spreading them "
        "over one message each: resolve the doctor, then the branch if "
        "they named one, then the day, and get as far down the flow as "
        "the information allows before you write anything. The "
        "one-question-per-message rule governs what you SAY, not how "
        "many tools you may call - it was never a reason to hand a step "
        "back to the patient.\n\n"
        "Each quoted value above is the patient's own wording, not a "
        "verified record. Resolve every one of them through its tool as "
        "usual, and if a tool cannot match one, deal with THAT - do not "
        "silently start over, and do not assume a name is real because "
        "it is quoted here.\n\n"
        "Your reply still ends with at most ONE question, and only about "
        "something genuinely still missing.\n\n"
    )


def _build_day_confirmation_requires_tool_directive(messages: list) -> str:
    """
    If the most recent ToolMessage in the conversation (scanning
    backwards) was `get_doctor_schedule_for_booking` or
    `list_available_days_for_booking`, and the CURRENT last message is a
    fresh human message (i.e. the user is now replying to that schedule
    or day list - naming a day, or just confirming the one offered) with
    no more recent `resolve_available_day` /
    `get_available_slots_for_booking` call since - inject a hard
    reminder that this turn must call the right tool rather than either
    guessing at availability or skipping ahead to patient details.

    WHY THIS EXISTS: confirmed real production failure, even after the
    prose instruction was already explicit - the model concluded "no
    appointments available" for a day purely from the recurring
    schedule's stated hours, with ZERO tool calls made that turn (no
    trace of resolve_available_day or get_available_slots_for_booking
    in the logs at all). The recurring schedule can only say which
    weekdays a doctor generally works, never whether a specific upcoming
    date actually has an open slot.
    """

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    if not isinstance(messages[-1], _HumanMessage):
        return ""

    for msg in reversed(messages[:-1]):
        name = getattr(msg, "name", None)
        if name in ("resolve_available_day", "get_available_slots_for_booking", "create_new_booking"):
            return ""
        if name == "list_available_days_for_booking":
            return (
                "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
                "The user is replying to the available day(s) you just "
                "showed. If their reply accepts a day in ANY form - "
                "\"مناسب\", \"اه\", \"تمام\", \"yes\", a number from the "
                "list, or naming the date itself - your ONLY next action "
                "is to call `get_available_slots_for_booking` with that "
                "day's `from_date`/`to_date` copied VERBATIM from the "
                "days result, and show the times. Do it in THIS turn.\n\n"
                "Do NOT ask for a phone number, a name, an email, or "
                "anything else yet - a time has not been chosen, so the "
                "booking is not at that step. Confirmed real production "
                "failure: a confirmed day was followed by the phone "
                "question instead of the times, stranding the patient "
                "mid-booking. Patient details come at STEP NB6, only "
                "after a specific time slot has been picked.\n\n"
                "If instead they asked for a different or later date, "
                "call `list_available_days_for_booking` again with the "
                "result's own `next_offset` - never calculate a date "
                "yourself.\n\n"
            )
        if name == "get_doctor_schedule_for_booking":
            return (
                "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
                "The user is replying after seeing the doctor's recurring "
                "schedule - likely naming a day. You are NOT allowed to "
                "answer whether that day has an available slot from the "
                "schedule alone, and you are NOT allowed to say "
                "\"not available\" without checking first. Before saying "
                "anything about availability for a specific day, you MUST "
                "call `resolve_available_day` with that weekday. Confirmed "
                "real failure: answering from the recurring schedule alone, "
                "with zero tool calls, produced a false \"not available\" "
                "for a day that actually had real, bookable slots.\n\n"
            )
        if name is not None:
            # A different, unrelated tool fired more recently - this
            # schedule display is no longer the most relevant context.
            return ""

    return ""


def _build_schedule_display_directive(messages: list) -> str:
    """
    If the LAST message is a ToolMessage from `get_doctor_schedule` with
    status "found", pre-build the EXACT branch/day-grouped display block
    in code, matching one of three confirmed reference formats depending
    on how many branches/days are involved - rather than only instructing
    the model to format it itself, which (like the appointment display
    and numbered slots list before it) was not reliably followed.

    Returns an empty string when the last message isn't a matching,
    successful schedule-lookup tool result.
    """

    if not messages:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) not in ("get_doctor_schedule", "get_doctor_schedule_for_booking"):
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if data.get("status") != "found":
        return ""

    schedules = data.get("schedules") or []
    if not schedules:
        return ""

    doctor_name = next((s.get("doctorName") for s in schedules if s.get("doctorName")), "")

    # Group rows by branch, preserving first-seen order.
    #
    # THE FORMAT BELOW WAS SPECIFIED DIRECTLY, and replaces three
    # different emoji-labelled layouts this function used to pick
    # between depending on how many branches and days were involved.
    # Those three produced visibly different messages for what is the
    # same question ("when and where does this doctor work?"), which is
    # exactly the inconsistency this project exists to remove. There is
    # now ONE shape: the doctor's name, then each branch with its own
    # days underneath it.
    by_branch: dict = {}
    branch_order = []
    fully_booked_days: set = set()
    for s_row in schedules:
        branch = s_row.get("branchName") or ""
        if branch not in by_branch:
            by_branch[branch] = []
            branch_order.append(branch)
        days = s_row.get("recurringDaysNames") or [""]
        from_time = _arabic_time_12h(s_row.get("fromDateTime"))
        to_time = _arabic_time_12h(s_row.get("toDateTime"))
        service = (s_row.get("serviceName") or "").strip()
        # A roster entry is not availability - see
        # tools._mark_fully_booked_schedule_days.
        #
        # A day with nothing left is NOT LISTED. This block answers
        # "when does this doctor work?", and the useful answer is the
        # days they can actually book. Listing a full day next to the
        # bookable ones invites them to pick it and be turned away a
        # turn later; the earlier version that printed "محجوز بالكامل
        # حاليًا" inline still put a dead end in front of them.
        #
        # The flag stays in the tool result, so if the patient asks
        # about that specific day they get the real answer - see the
        # instruction text below.
        if s_row.get("fully_booked"):
            fully_booked_days.add(
                _ARABIC_DAY_NAMES.get((days[0] or "").strip().lower(), days[0]) if days else ""
            )
            continue
        for day in days:
            arabic_day = _ARABIC_DAY_NAMES.get((day or "").strip().lower(), day)
            entry = (arabic_day, from_time, to_time, service)
            if entry not in by_branch[branch]:
                by_branch[branch].append(entry)

    # Branches left with nothing bookable are dropped entirely, for the
    # same reason.
    for branch in list(by_branch):
        if not by_branch[branch]:
            del by_branch[branch]
    branch_order = [b for b in branch_order if b in by_branch]

    if not branch_order:
        # Everything this doctor has is taken. There is no list to
        # print, so the model composes - it still has the tool result,
        # including which days are full.
        return ""

    total_day_rows = sum(len(v) for v in by_branch.values())

    def _day_line(day: str, from_time: str, to_time: str, service: str) -> str:
        line = f"• {day}: من {from_time} لـ {to_time}"
        # NO PRICE, EVER. The service NAME is useful context ("كشف رمد");
        # its fee is private by default and only ever revealed through
        # `get_doctor_fees` on an explicit request - see prompts.py's
        # FEES rule.
        if service:
            line = f"{line} — {service}"
        return line

    branch_blocks = []
    for index, branch in enumerate(branch_order):
        heading = (
            f"مواعيد الدكتور {doctor_name} في فرع {branch}:"
            if index == 0
            else f"وفي فرع {branch}:"
        )
        lines = [heading]
        for day, from_time, to_time, service in by_branch[branch]:
            lines.append(_day_line(day, from_time, to_time, service))
        branch_blocks.append("\n".join(lines))

    block = "\n\n".join(branch_blocks)

    single_branch = len(branch_order) == 1

    # Days that exist on the rota but have nothing left. They are NOT in
    # the block above on purpose - the list should only contain days the
    # patient can book. This note exists for the one case where the fact
    # is useful: they ask about that specific day.
    hidden_full_days = sorted(d for d in fully_booked_days if d)
    if hidden_full_days:
        full_days_note = (
            "\nNOTE - NOT PART OF YOUR REPLY. These days are on this "
            f"doctor's rota but have nothing left: {', '.join(hidden_full_days)}. "
            "They are deliberately absent from the list above; do NOT add "
            "them, do NOT mention them, and do NOT explain why they are "
            "missing. Only if the patient ASKS about one of them "
            "specifically, tell them plainly that it is fully booked at "
            "the moment and offer the days that are listed.\n\n"
        )
    else:
        full_days_note = ""

    if total_day_rows == 1:
        only_day = next(iter(by_branch.values()))[0][0]
        closing_question_instruction = (
            f"  3. Exactly one question asking whether they'd like the "
            f"available times for {only_day} - the only day this doctor "
            f"works. Do NOT ask \"which day\" or \"which branch\" when "
            f"there is only one of each; that is not a choice, and "
            f"reads as though you didn't look.\n"
        )
    elif single_branch:
        closing_question_instruction = (
            "  3. Exactly one question asking which DAY they'd like - "
            "the branch is already settled, so do not ask about it.\n"
        )
    else:
        closing_question_instruction = (
            "  3. Exactly ONE question covering both, phrased as a "
            "single sentence: \"حابب تحجز في أنهي فرع وانهي يوم؟\" - "
            "not two separate questions, and not two turns.\n"
        )

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "The doctor's schedule was just looked up. Your ENTIRE reply "
        "must have EXACTLY this structure and nothing more:\n"
        "  1. At most one very short lead-in sentence (e.g. \"Here's the "
        "doctor's schedule:\") - or none at all.\n"
        "  2. The exact text between the START/END markers below, copied "
        "verbatim, unchanged (translate the LABELS only if the "
        "conversation is in a different language - keep the emoji icons "
        "and the actual values unchanged either way). The START/END "
        "marker lines themselves are NOT part of the text to copy - "
        "never include them, or any other line of dashes/equals-signs, "
        "in your actual reply to the user.\n"
        f"{closing_question_instruction}"
        "Nothing else, anywhere in the reply. Do NOT also describe, "
        "list, or summarize any day/time/branch from the schedule in "
        "your own words before or after the block - the block is the "
        "ONLY place this information appears. Confirmed repeatedly in "
        "production: writing your own version of the schedule ANYWHERE "
        "in the same reply as this block sends the same information "
        "twice.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
        f"{full_days_note}"
    )


def _build_wrong_tool_in_booking_flow_directive(messages: list, session_id: str) -> str:
    """
    If the LAST message is a ToolMessage from `lookup_appointment` or
    `check_booking_status` (the cancellation/reschedule flows' own
    tools), AND this conversation's booking session already has a
    confirmed doctor or branch (meaning a NEW BOOKING is genuinely in
    progress), this is very likely the exact confirmed production bug:
    the model mixed up "get this phone number's patient info for a NEW
    booking" with "look up an EXISTING booking to cancel/reschedule" -
    surfacing a different, unrelated patient's real appointment details
    mid-booking. Inject a hard, explicit correction rather than relying
    on prose alone, since that alone did not reliably prevent this.

    Returns an empty string when there's no matching tool result, or no
    booking is actually in progress for this session (a normal
    cancellation/reschedule conversation must not be affected).
    """

    if not messages or not session_id:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) not in ("lookup_appointment", "check_booking_status"):
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id)
    if not session or not (session.get("doctor_id") or session.get("branch_id")):
        return ""

    return (
        "============================================================\n"
        "WRONG TOOL CALLED - YOU ARE MID NEW-BOOKING, NOT CANCELLATION\n"
        "============================================================\n"
        "A doctor or branch is ALREADY confirmed in this conversation's "
        "NEW BOOKING session, meaning you are actively booking a new "
        "appointment right now - but the tool result just returned is "
        "from `lookup_appointment`/`check_booking_status`, which look up "
        "an EXISTING, DIFFERENT booking (for cancellation/reschedule). "
        "This is the wrong tool for this moment - confirmed real "
        "production bug: this exact mistake surfaced a different, "
        "unrelated patient's real appointment details mid-booking.\n\n"
        "Do NOT present these results to the user, do NOT mention any "
        "booking they show, and do NOT ask the user to pick one of "
        "them. Instead, call `get_patient_info` with the same phone "
        "number to continue the NEW BOOKING flow's own STEP NB6 - that "
        "is the correct tool here.\n\n"
    )


def _build_empty_day_recovery_directive(messages: list) -> str:
    """
    If `get_available_slots_for_booking` just came back "not_found" for
    a day the patient had already accepted, don't let the agent stop and
    ask a question about it - make it find the next real day itself.

    WHY THIS EXISTS: confirmed real production failure. The agent
    offered "أقرب موعد متاح ... الأربعاء 02/09/2026", the patient said
    "مناسب", and the per-day lookup returned zero slots - so the reply
    was "للأسف، ما في مواعيد متاحة في اليوم ده". From the patient's
    side the agent contradicted itself one message after promising the
    date, then asked THEM what to do about it.

    `list_available_days_for_booking` now verifies each day before
    offering it, so this should be rare - but a slot can still be taken
    by someone else in the seconds between the two calls. When that
    happens the recovery belongs to the agent, not the patient.
    """

    if not messages:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) != "get_available_slots_for_booking":
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if data.get("status") != "not_found":
        return ""

    # Only when this day came from an offered day list - otherwise the
    # user picked an arbitrary date and "nothing that day" is a normal,
    # correct answer.
    came_from_day_list = any(
        getattr(msg, "name", None) == "list_available_days_for_booking"
        for msg in messages[-8:]
    )

    if not came_from_day_list:
        return ""

    return (
        "============================================================\n"
        "THE DAY YOU OFFERED IS NOW EMPTY - RECOVER, DON'T APOLOGIZE\n"
        "============================================================\n"
        "The slots for the day the patient just accepted came back "
        "empty. They were told this date was available moments ago, so "
        "simply reporting that it isn't makes the agent look broken and "
        "leaves them with nothing.\n\n"
        "In THIS turn, call `list_available_days_for_booking` again with "
        "`offset` advanced past that day to get the next genuinely "
        "available one. Then, in one short message: say that the last "
        "free time on that day was just taken, give the NEXT available "
        "date, and ask ONE question - whether that one works.\n\n"
        "Do NOT ask the patient what to do about it, do NOT ask whether "
        "they'd like you to look for another day, and do NOT suggest a "
        "date you calculated yourself. Look it up and offer it.\n\n"
    )


def _build_empty_branch_directive(messages: list) -> str:
    """
    If the LAST message is a `match_entity_for_booking` result that
    confirmed a BRANCH but carries `noDoctorsAtBranch`, force the reply
    to say so plainly instead of announcing a doctor list that doesn't
    exist.

    WHY THIS EXISTS: confirmed real production failure - a branch was
    confirmed, `doctorsAtBranch` came back empty, and the reply was
    "فرع المعادي تم اختياره ✅ / هنا قائمة الدكاترة المتاحين في الفرع"
    followed by no names at all. The patient was left with a confirmed
    branch, an announced list that wasn't there, and no next step.
    """

    if not messages:
        return ""

    last = messages[-1]

    if getattr(last, "name", None) != "match_entity_for_booking":
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if not data.get("noDoctorsAtBranch"):
        return ""

    branch_name = ((data.get("item") or {}).get("altName")
                   or (data.get("item") or {}).get("name") or "")

    return (
        "============================================================\n"
        "THIS BRANCH HAS NO AVAILABLE DOCTORS - KEEP THE REPLY SMALL\n"
        "============================================================\n"
        f"The branch{(' (' + branch_name + ')') if branch_name else ''} "
        "was matched, but NOBODY is available there for booking. There "
        "is no doctor list to show for it.\n\n"
        "WHICH REPLY YOU GIVE DEPENDS ENTIRELY ON WHAT THEY ASKED:\n\n"
        "A) They were just ASKING ABOUT THE BRANCH (its address, its "
        "details, or they picked it from an info list) - they have NOT "
        "said they want to book there. Then this turn is a simple info "
        "answer:\n"
        "  - Give the branch's ADDRESS.\n"
        "  - Offer to tell them about the SERVICES this branch provides.\n"
        "  - Say NOTHING about doctors, availability, or other branches. "
        "They did not ask to book, so 'no doctors available here' is an "
        "answer to a question nobody asked - it just makes the branch "
        "sound broken.\n"
        "  - Do NOT call `list_branches_for_specialty`, "
        "`find_available_doctors`, or any other doctor lookup on this "
        "turn.\n\n"
        "B) They explicitly asked to BOOK at this branch. Only then:\n"
        "  - Say plainly that this branch has no doctors available for "
        "booking right now.\n"
        "  - Then call `list_branches_for_specialty` and offer the other "
        "branches as a short numbered list - names (and addresses if you have them). What must NOT appear is the DOCTORS at those branches.\n"
        "  - DO NOT list the doctors at those other branches. Not one "
        "name. CONFIRMED REAL PRODUCTION FAILURE: this produced a wall "
        "of eleven doctor names across three branches in a single "
        "message, when the patient had only asked about one branch - "
        "unreadable, and it buries the actual question (which branch "
        "instead?) under a roster they never requested. The doctors come "
        "later, AFTER they pick a branch.\n"
        "  - Ask ONE question: which of those branches they'd like.\n\n"
        "IF A SERVICE IS WHAT THEY'RE AFTER, ANSWER WITH WHERE IT'S "
        "AVAILABLE. When the patient has picked one of this branch's "
        "services and wants to book it, use "
        "`find_branches_offering_service` instead - it returns the "
        "branches that can actually book THAT service. Say this branch "
        "can't book it right now, list those branches by name, and ask "
        "which one. That answers the question they actually have "
        "(\"where can I get this?\") rather than just closing the door.\n\n"
    )


def _build_appointment_display_directive(messages: list) -> str:
    """
    If the LAST message is a ToolMessage from `lookup_appointment` or
    `check_booking_status` with a single found booking, pre-build the
    EXACT emoji-formatted appointment block in code and hand it to the
    model as ready-made text to include verbatim - rather than only
    instructing it to format the block itself, which was not reliably
    followed even after multiple explicit prose instructions (confirmed
    directly in production, more than once: the format kept reverting to
    plain dashes instead of the requested emoji icons).

    Returns an empty string when the last message isn't a matching tool
    result with exactly one booking found.
    """

    if not messages:
        return ""

    last = messages[-1]
    tool_name = getattr(last, "name", None)

    if tool_name not in ("lookup_appointment", "check_booking_status"):
        return ""

    try:
        data = json.loads(last.content)
    except (ValueError, TypeError):
        return ""

    if tool_name == "lookup_appointment":
        if data.get("status") != "found_one":
            return ""
        appt = data.get("appointment") or {}
    else:
        if data.get("status") != "active":
            return ""
        appt = data.get("appointment") or {}

    if not appt:
        return ""

    block = (
        f"👤 الاسم: {appt.get('patientFullName', '')}\n"
        f"👨\u200d⚕️ الطبيب: {appt.get('doctorName', '')}\n"
        f"🏥 الفرع: {appt.get('branchName', '')}\n"
        f"🗓️ التاريخ: {appt.get('date_display', '')}\n"
        f"🕐 الوقت: {appt.get('time_display', '')}"
    )

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "A booking was just found. Include the exact text between the "
        "START/END markers below, verbatim, in your reply (translate "
        "the LABELS only if the conversation is in a different language "
        "- keep the emoji icons and the actual values unchanged either "
        "way). The START/END marker lines themselves are NOT part of "
        "the text to copy - never include them, or any other line of "
        "dashes/equals-signs, in your actual reply to the user.\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
        "After this block, ask ONLY one single yes/no question appropriate "
        "to what's happening (e.g. confirm this is the booking to "
        "cancel/reschedule) - nothing else in this same reply.\n\n"
    )


def _split_bilingual_greeting(text: str) -> Optional[dict]:
    """Split a greeting template that is actually TWO greetings stapled
    together - one full paragraph in English, one full paragraph in
    Arabic (in either order) - into {"en": ..., "ar": ...}.

    Some clients' `msg_unknown_fallback` is authored this way on
    purpose, so the very first, language-unknown message can greet in
    both languages at once. That is fine for a channel that truly
    doesn't know the patient's language yet, but once this turn's
    language IS known (`target_language`, from `_detect_target_language`
    - here just "hi" is already enough), sending both halves puts two
    languages in one message. See the MIXED-LANGUAGE GREETING GUARD in
    `_run_agent` for the production failure this was written for.

    Heuristic, deliberately simple: scan line by line, tracking which
    script each line "commits" to (Arabic, Latin, or neither - a blank
    line or an emoji-only line carries the previous line's script
    forward). The first point where the running script flips is treated
    as the boundary between the two paragraphs. Returns None when no
    such flip is found (the template is single-language, or the mixing
    is too interleaved to safely separate) - callers must treat that as
    "could not split" and fall back to their existing behaviour, never
    guess and risk cutting a real client template in half.
    """

    lines = (text or "").split("\n")

    def _line_script(line: str) -> Optional[str]:
        if _looks_arabic(line):
            return "ar"
        if _has_latin_letters(line):
            return "en"
        return None

    scripts = [_line_script(line) for line in lines]

    running = []
    last_seen = None
    for s in scripts:
        if s:
            last_seen = s
        running.append(last_seen)

    if not running or running[0] is None:
        return None

    first_lang = running[0]
    split_at = None
    for i, s in enumerate(running):
        if s and s != first_lang:
            split_at = i
            break

    if split_at is None:
        # Single language throughout - nothing to split.
        return None

    part_a = "\n".join(lines[:split_at]).strip()
    part_b = "\n".join(lines[split_at:]).strip()

    if not part_a or not part_b:
        return None

    return {first_lang: part_a, ("ar" if first_lang == "en" else "en"): part_b}


def _build_greeting(templates: dict, user_message: str, target_language: str) -> str:
    """
    Build the deterministic opening greeting for this conversation.

    Always the clinic's own configured `msg_unknown_fallback` text
    (from n8n's client_config Data Table, or the CSV as a fallback -
    whichever config.get_messages() resolved this turn), verbatim, with
    only its fixed opening line optionally swapped for a time-of-day
    salutation (see _personalized_greeting) - regardless of which
    language the conversation is in. _personalized_greeting already
    handles both Arabic and Latin-script cues correctly, so there is no
    need for a separate English-only template here.

    The hardcoded English fallback below is a LAST RESORT ONLY, for a
    client whose config genuinely has no `msg_unknown_fallback` set at
    all (should be rare once every client's config source sets one) -
    it must never override a real, configured greeting just because the
    conversation happens to be in English. Confirmed real production
    bug: an English-language conversation was ALWAYS given this
    hardcoded generic template instead of the clinic's own configured
    greeting, even when one was properly set - so a fully-configured
    client (correct dialect, agent name, branches, everything) still
    showed a generic, un-branded opening line the moment the patient
    happened to type in English.
    """

    greeting = templates.get("msg_unknown_fallback")

    # A BILINGUAL TEMPLATE IS SPLIT BEFORE ANYTHING ELSE RUNS.
    #
    # Some clients author `msg_unknown_fallback` as one English
    # paragraph followed by one Arabic paragraph in a SINGLE field -
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): an
    # English "hi" got the full English AND the full Arabic paragraph
    # back, one after the other, in the very first message the clinic
    # sent. Isolating the half that matches this turn's language here
    # means the real client-authored wording (branding, service list,
    # emoji, exact phrasing) is what goes out - not the generic
    # hardcoded template below, which only happens to look right for
    # clients whose English paragraph happens to match it.
    if target_language in ("en", "ar"):
        split = _split_bilingual_greeting(greeting)
        if split and split.get(target_language):
            greeting = split[target_language]

    # LANGUAGE COMES FIRST, BRANDING COMES WITH IT.
    #
    # An earlier version always used the configured `msg_unknown_fallback`
    # verbatim, in whatever language it was authored in. Since every
    # client authors it in Arabic, a patient who opened the conversation
    # in English received the full Arabic greeting block and an English
    # reply stapled underneath it - two languages in the first message
    # the clinic ever sends. The version before THAT swung the other way
    # and gave English conversations a generic, un-branded template even
    # when the clinic was fully configured.
    #
    # Neither trade-off is necessary. The order is:
    #   1. A greeting the client authored in this language, if their
    #      config provides one (`msg_unknown_fallback_en`) - always wins.
    #   2. A bilingual configured greeting, already split above into
    #      just this conversation's language.
    #   3. The configured greeting, when it is already in the right
    #      language for this conversation (single-language template).
    #   4. The standard template rendered in the conversation's language,
    #      filled with THIS clinic's real agent name and clinic name -
    #      so it is branded, and it matches the structure, ordering and
    #      emoji of the Arabic one exactly. Same shape, different
    #      language, which is precisely what the response contract asks
    #      for.
    if target_language == "en":
        english_greeting = templates.get("msg_unknown_fallback_en")
        if english_greeting:
            greeting = english_greeting.replace("\r\n", "\n").replace("\r", "\n")
            return _personalized_greeting(greeting, user_message, target_language)

        if not greeting or _looks_arabic(greeting):
            lowered = (user_message or "").lower()
            if any(cue in lowered for cue in _MORNING_CUES):
                salutation = "Good morning! \U0001F60A"
            elif any(cue in lowered for cue in _EVENING_CUES):
                salutation = "Good evening! \U0001F60A"
            else:
                salutation = "Hi there! \U0001F44B"

            return _ENGLISH_GREETING_TEMPLATE.format(
                salutation=salutation,
                agent_name=templates.get("_agent_name") or "the assistant",
                clinic_name=templates.get("_clinic_name") or "the clinic",
            )

    if not greeting:
        return ""

    # Normalize line endings FIRST. The CSV can arrive with \r\n (Excel/
    # Git on Windows often rewrite it that way), while the LLM, when it
    # reproduces the same greeting from the system prompt, emits plain
    # \n. That mismatch made the "did the model already include the
    # greeting?" check in agent() fail, so a second copy got prepended -
    # the actual root cause of the duplicated greeting seen in
    # production (the two copies in the log differed ONLY in line
    # endings). Normalizing here fixes both the comparison and the text
    # we emit.
    greeting = greeting.replace("\r\n", "\n").replace("\r", "\n")

    return _personalized_greeting(greeting, user_message, target_language)


def _personalized_greeting(greeting: str, user_message: str, language_hint: str) -> str:
    """
    Swap ONLY the official greeting's fixed opening line ("أهلاً بيك 👋" /
    "أهلاً وسهلاً بك 👋") for a time-of-day salutation ("صباح النور!"/
    "مساء النور!"/"Good morning!"/"Good evening!") when the user's own
    first message clearly signals one - keeping the entire rest of the
    template (persona intro, service list, closing question) exactly as
    authored. Falls back to the original fixed opening line unchanged
    when the user's message doesn't give a clear time-of-day cue (e.g. a
    booking reference, a plain "hi", or anything else neutral).
    """

    lowered = (user_message or "").lower()

    if any(cue in lowered for cue in _MORNING_CUES):
        salutation = "صباح النور! 😊" if _looks_arabic(user_message) else "Good morning! 😊"
    elif any(cue in lowered for cue in _EVENING_CUES):
        salutation = "مساء النور! 😊" if _looks_arabic(user_message) else "Good evening! 😊"
    else:
        return greeting  # no clear time-of-day cue - use the template's own opening line as-is

    lines = greeting.split("\n", 1)
    if len(lines) == 2:
        return f"{salutation}\n{lines[1]}"
    return salutation


def _looks_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _has_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text or ""))


def _latin_word_count(text: str) -> int:
    """How many whole Latin-script words the text contains.

    Distinguishes an incidental proper noun that came back from the API
    in English ("Al Nozha", "Dr Smith") from an entire English paragraph
    riding along in an otherwise-Arabic message. `_has_latin_letters`
    cannot tell those apart - it is True for both - which is why the
    mixed-language greeting guard needs this instead. See the
    CONFIRMED REAL PRODUCTION FAILURE note at that guard.
    """

    return len(re.findall(r"[A-Za-z]{2,}", text or ""))


# ARABIZI / FRANCO-ARABIC: Arabic typed in Latin script, using digits
# for the Arabic letters that have no Latin equivalent - 2 for ء/أ,
# 3 for ع, 5 for خ, 6 for ط, 7 for ح, 8 for غ, 9 for ص. Extremely
# common in Egypt and the Gulf, and it is ARABIC, not English.
#
# The digit must sit INSIDE a word, or open one that is clearly a word
# rather than a time-of-day form, so that ordinary English containing a
# numeral - "I need 2 appointments", "book me for 3 pm", "3pm" - is
# never mistaken for Arabizi. Hence the 3-letter minimum on the
# word-initial case: "3ayez"/"3aleko" match, "3pm" does not.
_ARABIZI_RE = re.compile(
    r"[A-Za-z][2356789][A-Za-z]"       # digit inside a word:  mass2oo, a7gz
    r"|[A-Za-z]{2,}[2356789]\b"        # digit ends a word:    kha6er -> also mab3
    r"|\b[2356789][A-Za-z]{3,}"        # digit opens a word:   3ayez, 3aleko
)
# DELIBERATE GAP: a 2-letter tail after a leading digit stays English,
# because "3am"/"3pm"/"5pm" are real English times and breaking those
# would be worse than missing the rarer Franco spelling of "عم" in
# "ezayak ya 3am". Every longer Franco word ("3ayez", "3aleko",
# "3arabi") is still caught.


def _looks_arabizi(text: str) -> bool:
    """Whether Latin-script text is actually Arabic written in Franco.

    CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-31): the first
    message of a conversation was "mass2oo" (مسعود). Classified as
    English on the strength of its Latin letters alone, it produced an
    English greeting stapled to the model's own Arabic question - one
    message, two languages, on the clinic's very first contact.
    """

    return bool(_ARABIZI_RE.search(text or ""))


def _detect_target_language(messages: list) -> Optional[str]:
    """
    Determine which language THIS reply must be in, deterministically -
    by code, not left to the LLM to infer from a long system prompt.

    Scans the conversation's HumanMessages, most recent first, and
    returns "ar"/"en" based on the first one that gives a clear signal
    (Arabic script, or Latin letters). A message with neither (e.g. just
    digits like an OTP code, or "yes"/"نعم") is skipped in favor of an
    earlier message that does give a signal - this is what keeps the
    established language consistent through dialect-neutral replies
    without resetting.

    Returns None only if NO message in the whole conversation gives any
    signal at all (extremely unlikely in practice) - in that case the
    system prompt's own default dialect applies unmodified.

    WHY THIS EXISTS: relying solely on the LANGUAGE & DIALECT prose rule
    inside the (long, Arabic-reference-heavy) system prompt measurably
    did not reliably keep the reply in the user's actual language,
    including on a conversation that was purely English from its very
    first message - a plain prose instruction competing with thousands
    of characters of Arabic reference material was not a strong enough
    signal on its own.
    """

    for msg in reversed(messages):
        if getattr(msg, "type", None) != "human":
            continue
        content = msg.content or ""
        if _looks_arabic(content):
            return "ar"
        if _has_latin_letters(content):
            # Latin letters alone do not mean English - Franco-Arabic is
            # written in Latin script and is Arabic. See _looks_arabizi.
            return "ar" if _looks_arabizi(content) else "en"

    return None


_LANGUAGE_DIRECTIVE = {
    "en": (
        "============================================================\n"
        "MANDATORY LANGUAGE FOR THIS REPLY: ENGLISH\n"
        "============================================================\n"
        "This entire reply must be written in English only. Do not use "
        "any Arabic words, letters, or Arabic-script emoji captions "
        "anywhere in it. Ignore the Arabic dialect/reference-phrase "
        "sections further below for this reply - they do not apply.\n\n"
    ),
    "ar": (
        "============================================================\n"
        "MANDATORY LANGUAGE FOR THIS REPLY: ARABIC - EVERY SINGLE WORD\n"
        "============================================================\n"
        "This entire reply must be written in Arabic, following the "
        "dialect/tone and reference phrases further below. Not one "
        "English word anywhere in it. This explicitly includes:\n"
        "  - Times: write صباحًا / ظهرًا / مساءً - NEVER \"AM\"/\"PM\" and "
        "never a 24-hour or ISO timestamp.\n"
        "  - Weekday and month names: Arabic only (الثلاثاء, not "
        "\"Tuesday\").\n"
        "  - Names of the hospital, its branches, its doctors, its "
        "specialties and its services: use the Arabic form. The tools "
        "already return these in Arabic for an Arabic conversation - use "
        "the value they gave you. If a tool result somehow still carries "
        "only a Latin-script name for one of these, write it in Arabic "
        "yourself rather than pasting the English in mid-sentence.\n"
        "  - Labels, headers, and emoji captions: Arabic.\n"
        "Confirmed real user complaint: specialty names, the hospital "
        "name, and AM/PM were appearing in English inside otherwise "
        "fully-Arabic replies, which reads as careless and "
        "inconsistent.\n\n"
    ),
}


# Generic medical-guidance requests observed in production that carry NO
# actual symptom content - a plain ask for help, not a description of
# what's wrong. Kept short and literal on purpose: this only needs to
# catch the exact "just asking for guidance" pattern, not every possible
# phrasing - a real symptom mention naturally has more content and won't
# match these short strings.
_GENERIC_MEDICAL_GUIDANCE_PHRASES = (
    "توجيه طبي", "التوجيه الطبي", "عاوز توجيه طبي", "عاوزه توجيه طبي",
    "ابغى توجيه طبي", "ابغي توجيه طبي", "عايز توجيه طبي", "عايزه توجيه طبي",
    "محتاج توجيه طبي", "محتاجه توجيه طبي",
    "medical guidance", "i want medical guidance", "i need medical guidance",
)


def _is_generic_medical_guidance_request(user_message: str) -> bool:
    """
    True when `user_message` is a bare request for medical guidance with
    NO symptom described - e.g. "توجيه طبي" alone, as opposed to "توجيه
    طبي، عيني وجعاني" which already names a symptom.

    WHY THIS EXISTS: relying solely on the prose instruction not to
    invent a comfort suggestion before a symptom is named was NOT
    reliably followed even after being made explicit - observed directly
    in production, more than once, after the fix had already been
    deployed. This computes the same judgment deterministically instead,
    and a dominant directive is injected at the top of the system
    prompt for this specific turn (see agent()) rather than trusting a
    rule buried in a long prompt.
    """

    normalized = (user_message or "").strip().lower()

    if not normalized:
        return False

    # If the message is ONLY (allowing minor punctuation/whitespace) one
    # of the known generic phrases, treat it as symptom-free. Anything
    # with more content alongside it (even a few extra words) is assumed
    # to potentially carry real symptom detail and is left to the LLM.
    stripped = re.sub(r"[؟?!.,\s]+", " ", normalized).strip()

    return any(stripped == phrase for phrase in _GENERIC_MEDICAL_GUIDANCE_PHRASES)


_NO_SYMPTOM_YET_DIRECTIVE = (
    "============================================================\n"
    "NO SYMPTOM DESCRIBED YET\n"
    "============================================================\n"
    "The user has only asked for medical guidance in general - they have "
    "NOT described any actual symptom or health concern yet. Your ENTIRE "
    "reply must be limited to warmly asking what the symptom or issue "
    "is. Do NOT include any comfort/self-care suggestion in this reply - "
    "there is nothing yet to tailor one to, and inventing one (e.g. "
    "generic rest/warm-tea advice) with no actual symptom mentioned is "
    "worse than not giving one. Save the comfort suggestion for the turn "
    "after they've actually told you what's wrong.\n\n"
)


def _build_channel_identity_directive(channel_phone: Optional[str]) -> str:
    """
    Put the user's ACTUAL channel number (their WhatsApp sender number)
    into the system prompt as a real value.

    WHY THIS EXISTS: the NEW BOOKING and COMPLAINT flows both tell the
    model to ask "shall we use this same WhatsApp number ({channel_phone})?"
    - but that placeholder was never filled with anything, so the model
    was looking at literal braces rather than a number. Faced with a
    placeholder it couldn't show, it tended to skip the question
    altogether (confirmed: the yes/no confirmation simply never
    appeared) or fall back to asking the patient to type a number they
    had already been messaging from all along.
    """

    if not channel_phone:
        return (
            "============================================================\n"
            "CHANNEL IDENTITY: NONE AVAILABLE - HARD OVERRIDE\n"
            "============================================================\n"
            "There is NO verified WhatsApp/channel number for this "
            "conversation (this is a web-widget/Messenger conversation, "
            "not WhatsApp).\n\n"
            "THIS OVERRIDES ANY OTHER INSTRUCTION BELOW, INCLUDING ONES "
            "THAT SAY \"ALWAYS ASK\" OR \"NOT OPTIONAL\": those instructions "
            "(STEP NB6 in the NEW BOOKING flow, and the equivalent step in "
            "the COMPLAINT flow) are written assuming a channel identity "
            "exists. It does NOT exist in this conversation, so they do "
            "NOT apply here.\n\n"
            "Concretely: NEVER ask \"shall we continue with the same "
            "WhatsApp number?\" / \"نكمل بنفس رقم الواتساب ده؟\" or any "
            "variant of that yes/no question anywhere in this "
            "conversation - there is no number for it to refer to, so the "
            "question would be meaningless and confusing. Instead, "
            "whenever a phone number is genuinely needed, ask the user "
            "directly and openly for their mobile number (with country "
            "code), then proceed with the normal validate -> "
            "`compare_phone` -> `send_otp`/`verify_otp` flow exactly as "
            "you would for any first-time number.\n\n"
        )

    return (
        "============================================================\n"
        f"CHANNEL IDENTITY (THE USER'S OWN WHATSAPP NUMBER): {channel_phone}\n"
        "============================================================\n"
        "This conversation DOES have a verified WhatsApp/channel number "
        "(shown above, never to be printed in a reply).\n\n"
        "Wherever ANY flow needs the patient's phone number as a VALUE - "
        "passing it to a tool, LOOKING UP their existing appointment "
        "with it, saving it on a booking or on a complaint - that number "
        f"is {channel_phone}. Use it directly. Never ask them to type a "
        "number they are already messaging from, and never ask for a "
        "booking reference as a substitute for it.\n\n"
        "(That sentence was previously truncated mid-clause and read as "
        "nonsense - \"Wherever any flow number as a VALUE\" - which is "
        "very likely why the rule kept failing to hold. It is written "
        "out in full here.)\n\n"
        "DO NOT PRINT THE NUMBER IN YOUR REPLY. Both of you already know "
        "which number this is, so writing out the digits adds noise and "
        "makes a one-line question look like a form. Ask the short yes/no "
        "question exactly as the clinic wrote it - e.g. \"نكمل الحجز على "
        "نفس رقم الواتساب ده؟ ✅\" - with no digits, no country code, and "
        "no parenthetical. Then WAIT for their answer.\n\n"
        "If they say yes, use the number above as the phone value with no "
        "OTP - and ACT ON IT IN THAT SAME TURN. \"Yes\" is not an answer "
        "that needs a follow-up question; it is permission to proceed. "
        "For a cancellation or a change, that means calling "
        "`lookup_appointment` with the number above immediately.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: the assistant asked \"نكمل "
        "تعديل موعدك على نفس رقم الواتساب ده؟\", the patient answered "
        "\"اه\", and the reply was \"ممكن تعطيني رقم الحجز أو رقم جوالك "
        "عشان أقدر أجيب بيانات موعدك؟\" - asking for the very number it "
        "had just been given permission to use, one message after "
        "offering it. From the patient's side the assistant asked a "
        "question, got an answer, and ignored it.\n\n"
        "Only if they want a DIFFERENT number do the "
        "validate/compare/OTP steps apply - and only then does a number "
        "appear in the conversation at all, because they typed it.\n\n"
        "If they say NO (they want a different number, or a different "
        "way to be identified), your ENTIRE reply this turn must be "
        "asking them for ONE of: their mobile number WITH ITS COUNTRY "
        "CODE, or their booking reference number - nothing else. Do NOT "
        "proceed with ANY other step of whatever flow you're in (do NOT "
        "show reschedule day/slot options, do NOT call "
        "`get_available_reschedule_slots`, do NOT ask which day they'd "
        "like instead) until one of those two has actually been given "
        "and, for a typed phone number, it has gone through the normal "
        "validate -> `compare_phone` -> `send_otp`/`verify_otp` flow. "
        "The booking has not been identified yet at this point - only "
        "the identification METHOD has just changed from \"use the "
        "channel number\" to \"ask for it directly\" - so nothing "
        "downstream of identification can happen yet.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: the assistant asked \"نكمل "
        "تعديل موعدك على نفس رقم الواتساب ده؟\", the patient answered "
        "\"لا\" - and the very next message jumped straight into "
        "reschedule day options (\"وش اليوم تفضل تحجز بدال الثلاثاء؟\") "
        "as if a booking had already been found, with no phone number "
        "or booking reference ever asked for or given. From the "
        "patient's side, declining the channel number appeared to have "
        "been silently ignored and replaced with a booking the "
        "assistant had no way of actually knowing.\n\n"
    )


_SERVICES_QUESTION_CUES = (
    "خدمات", "الخدمات", "خدماتكم", "خدماتكو", "بتقدمو", "بتقدموا", "تقدمون",
    "وش تقدمون", "ايه اللي عندكم", "إيه اللي عندكم",
    "services", "what do you offer", "what services",
)


def _build_services_from_kb_directive(messages: list) -> str:
    """
    When the user asks what services the clinic offers, force the answer
    to come from the clinic's own knowledge base (`answer_hospital_faq`
    -> the RAG file), not from the specialty list and not from memory.

    WHY THIS EXISTS: "services" and "specialties" are different things.
    `list_specialties` returns the booking system's registered medical
    specialties; the clinic's actual service catalogue (with its real
    names, descriptions, hours and target audiences) lives in the
    knowledge base file. Answering a services question from the
    specialty list produces names the clinic doesn't use for its own
    services - the exact complaint raised about this flow.
    """

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    if not isinstance(messages[-1], _HumanMessage):
        return ""

    text = (messages[-1].content or "").lower()

    if not any(cue in text for cue in _SERVICES_QUESTION_CUES):
        return ""

    return (
        "============================================================\n"
        "SERVICES QUESTION - USE `list_hospital_services`\n"
        "============================================================\n"
        "The user is asking what services this clinic offers. Call "
        "`list_hospital_services` and present EVERY service it returns, "
        "in the order it returned them, as an emoji-numbered list, using "
        "its exact wording for each name. Then ask ONE question - whether "
        "they'd like details about one of them.\n\n"
        "Do NOT use `answer_hospital_faq` for this question. It returns "
        "the passages most SIMILAR to the question, which are detail "
        "paragraphs from inside one or two services - confirmed real "
        "failure: answering that way listed inpatient amenities (gardens, "
        "gym, art-therapy area, isolation rooms) as if they were "
        "services, while four of the clinic's six actual services were "
        "never mentioned at all. `answer_hospital_faq` is for AFTERWARDS, "
        "once they ask about one specific service.\n\n"
        "Do NOT answer from `list_specialties` either - registered "
        "booking specialties are a different list for a different "
        "purpose - and do not answer from memory or from anything earlier "
        "in this conversation. Add nothing to the list that "
        "`list_hospital_services` did not return, and drop nothing from "
        "it. If it returns \"not_found\"/\"not_configured\", say plainly "
        "that you don't have that information and offer a staff "
        "handoff.\n\n"
        "ANSWER ONLY THE SERVICES QUESTION. Do not open with, or append, "
        "anything about doctors or booking availability - not even when "
        "the branch under discussion happens to have no bookable doctor. "
        "CONFIRMED REAL PRODUCTION FAILURE: asked to show a branch's "
        "services, the reply began \"فرع المعادي مافي عنده دكاترة "
        "متاحين حاليا للحجز. لكن يقدم خدمات عديدة...\" - leading with a "
        "negative about a question that was never asked. Just answer "
        "what they asked.\n\n"
        "IS THIS ABOUT ONE BRANCH? If the question is about a SPECIFIC "
        "branch's services (\"خدمات فرع المعادي\", \"إيه الخدمات في "
        "الفرع ده؟\"), or a branch is the thing you were just discussing, "
        "then everything above about `list_hospital_services` does NOT "
        "apply - call `list_branch_services` instead. That reads the "
        "clinic's real service catalogue filtered to that branch; "
        "`list_hospital_services` reads the knowledge-base file, which "
        "has no per-branch information at all and would return the same "
        "hospital-wide list no matter which branch was asked about. "
        "CONFIRMED REAL PRODUCTION FAILURE: asked for فرع المعادي's "
        "services, the reply was the hospital-wide knowledge-base list "
        "verbatim.\n"
        "  - \"not_found\": this branch publishes no services - say that "
        "plainly for THAT branch; don't substitute the hospital-wide "
        "list.\n"
        "  - \"missing_branch\": ask which branch they mean.\n\n"
    )


_HOW_TO_BOOK_CUES = (
    "احجز ازاي", "أحجز ازاي", "احجز إزاي", "ازاي احجز", "إزاي أحجز",
    "كيف احجز", "كيف أحجز", "كيفية الحجز", "طريقة الحجز", "ازاي اقدر احجز",
    "كيف اقدر احجز", "كيف يمكنني الحجز", "ابغى احجز كيف", "الحجز ازاي",
    "how do i book", "how can i book", "how to book", "how do i make an appointment",
    "how can i make an appointment", "how do i schedule",
)


def _build_how_to_book_directive(messages: list) -> str:
    """
    When the user asks HOW to book, the answer is "right here, with me,
    now" - never a website, an app, a hotline, or a branch visit.

    WHY THIS EXISTS: confirmed real behavior - asked how to book, the
    agent pointed the patient at the clinic's website. The knowledge
    base legitimately contains website URLs and phone numbers (they're
    part of the clinic's own contact information), so they're right
    there in the retrieved passages, and it's an easy wrong turn to
    take. But this agent CAN complete a booking end to end, and sending
    someone elsewhere to do what it can do in the next three messages
    loses the booking for no reason.
    """

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    if not isinstance(messages[-1], _HumanMessage):
        return ""

    text = (messages[-1].content or "").lower()

    if not any(cue in text for cue in _HOW_TO_BOOK_CUES):
        return ""

    return (
        "============================================================\n"
        "\"HOW DO I BOOK?\" - THE ANSWER IS: RIGHT HERE, WITH YOU, NOW\n"
        "============================================================\n"
        "Tell them plainly that you can book it for them yourself in this "
        "same chat, then immediately start the NEW BOOKING FLOW (call "
        "`reset_booking_session`, then ask its first single question).\n\n"
        "NEVER answer this by pointing them anywhere else - not the "
        "clinic's website, not an app, not a phone/hotline number, not "
        "\"visit the branch\", not \"contact reception\". This is true even "
        "if `answer_hospital_faq` returns a passage containing a booking "
        "URL or a contact number: those are the clinic's general contact "
        "details, not an instruction to hand the patient off. You can "
        "complete a real booking end to end, so do that instead of "
        "redirecting them.\n\n"
    )


_QUESTION_MARKS = ("؟", "?")


def _template_question_sentences(templates: dict) -> set:
    """Every question sentence that appears inside the clinic's own
    authored templates, normalized for comparison.

    These are allowed to sit alongside another question in one reply -
    e.g. the approved booking review card ends with "هل جميع البيانات
    صحيحة وتود تأكيد الحجز؟", and some templates pair a statement with a
    follow-up ask. Those are signed-off wording, not the model piling on
    extra questions, so the one-question trimmer must leave them alone.
    """

    allowed = set()

    for key, value in (templates or {}).items():
        if not key.startswith("msg_") or not isinstance(value, str):
            continue
        for sentence in _split_sentences(value.replace("\r", "\n")):
            if any(mark in sentence for mark in _QUESTION_MARKS):
                normalized = _normalize_for_compare(sentence)
                if normalized:
                    allowed.add(normalized)

    return allowed


def _split_sentences(text: str) -> list:
    """Split into sentence-ish segments, keeping their terminators, and
    never merging across line breaks (numbered lists and labeled blocks
    rely on their own lines staying intact)."""

    segments = []

    for line in text.split("\n"):
        current = ""
        for char in line:
            current += char
            if char in ("؟", "?", "!", "۔"):
                segments.append(current)
                current = ""
        if current:
            segments.append(current)
        segments.append("\n")

    if segments and segments[-1] == "\n":
        segments.pop()

    return segments


def _strip_extra_questions(reply_text: str, templates: dict) -> tuple:
    """Keep the FIRST question in a reply and drop any later ones.

    Returns (possibly_trimmed_text, number_of_questions_removed).

    WHY THIS EXISTS: "exactly one question per message" is stated as a
    hard rule in the system prompt, and repeated inside the booking
    flow, and it is STILL the most frequently violated instruction -
    confirmed in production more than once ("تحب تحجز مع دكتور معيّن،
    ولا تخصص معيّن؟ أو تحب أشوف لك قائمة الدكاترة؟"). Stacked questions
    make people freeze or answer only one of them, which then desyncs
    the whole flow. Prose alone hasn't fixed it, so this enforces it
    after the fact, the same way the display blocks are enforced in
    code rather than trusted to the model.

    Only sentences that actually CARRY a question are removed - lists,
    labeled blocks, and ordinary statements after the question are left
    untouched, and question wording that came from the clinic's own
    approved templates is always kept.
    """

    if not reply_text or not any(mark in reply_text for mark in _QUESTION_MARKS):
        return reply_text, 0

    allowed = _template_question_sentences(templates)

    kept = []
    seen_question = False
    removed = 0

    for segment in _split_sentences(reply_text):
        is_question = any(mark in segment for mark in _QUESTION_MARKS)

        if not is_question:
            kept.append(segment)
            continue

        if not seen_question:
            seen_question = True
            kept.append(segment)
            continue

        if _normalize_for_compare(segment) in allowed:
            kept.append(segment)
            continue

        removed += 1

    if not removed:
        return reply_text, 0

    trimmed = "".join(kept)

    # Tidy up whatever the removal left behind: dangling connectors at
    # the end of a line ("... أو", "... ولا"), and blank-line runs.
    trimmed = re.sub(r"[ \t]*(?:أو|ولا|or)[ \t]*(?=\n|$)", "", trimmed)
    trimmed = re.sub(r"\n{3,}", "\n\n", trimmed)
    trimmed = "\n".join(line.rstrip() for line in trimmed.split("\n")).strip()

    return trimmed, removed


_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_PLAIN_LIST_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<number>[0-9\u0660-\u0669]{1,2})[.)\u061F]?[.)]?[ \t]+(?=\S)",
    re.MULTILINE,
)


def _emojify_list_numbers(text: str) -> str:
    """Rewrite plain "1. " / "2) " list markers at the start of a line
    into the same emoji badges the code-built lists use.

    WHY THIS EXISTS: only the lists this file pre-builds (time slots,
    available days) were emoji-numbered. Every OTHER list - doctors,
    branches, specialties - is written by the model, and those came out
    as plain "1. 2. 3." in production, so a single conversation mixed
    two visibly different list styles depending on which list it was.
    Normalizing here covers every list, including ones no directive
    exists for, without needing the model to remember anything.

    Only line-leading markers are touched, so ordinary text containing a
    number followed by a full stop is left alone.
    """

    if not text:
        return text

    def _replace(match):
        number = match.group("number").translate(_ARABIC_INDIC_DIGITS)
        try:
            value = int(number)
        except ValueError:
            return match.group(0)
        if value < 1:
            return match.group(0)
        return f"{match.group('indent')}{_numbered_prefix(value)} "

    return _PLAIN_LIST_MARKER_RE.sub(_replace, text)


def _is_redundant_closing_question_only(reply_text: str, greeting: str) -> bool:
    """
    True when `reply_text` is essentially just a repeat of the
    greeting's own last line (its closing question, e.g. "كيف أستطيع
    مساعدتك اليوم؟ 😊") and nothing substantially more.

    WHY THIS EXISTS: confirmed real production bug - on a bare opening
    greeting with no stated intent, the model is instructed to reply
    with nothing and let the greeting's own closing question stand
    alone. Instead, it sometimes writes that SAME closing question
    itself. `_already_contains_greeting` correctly does NOT flag this
    as "already contains the greeting" (a lone closing question doesn't
    match the persona-line signature), so the full greeting - which
    itself already ends with that same question - gets prepended
    anyway, and the question appears twice, back to back.
    """

    greeting_lines = [ln.strip() for ln in greeting.replace("\r", "\n").split("\n") if ln.strip()]
    if not greeting_lines:
        return False

    closing_line = greeting_lines[-1]
    normalized_closing = _normalize_for_compare(closing_line)
    normalized_reply = _normalize_for_compare(reply_text)

    if not normalized_closing or not normalized_reply:
        return False

    # The reply counts as "just the closing question" if, once that
    # question is removed, nothing meaningful (more than a few stray
    # characters of punctuation/whitespace) is left.
    remainder = normalized_reply.replace(normalized_closing, "", 1)
    return len(remainder) <= 3




# ==========================================================
# Invented-branch-name verifier
# ==========================================================
#
# WHY THIS EXISTS: the model has, more than once, listed branches it was
# never given. The worst confirmed case offered "فرع الدقي / فرع زايد /
# فرع مصر الجديدة" for a doctor who works at exactly one branch, with
# zero tool calls made that turn - so there was no wrong tool result to
# blame, and no prompt rule could have been "followed" at the moment it
# mattered, because the model simply wrote names from memory.
#
# Every other guard in this file shapes what the model does BEFORE it
# writes. This one checks what it actually wrote, which is the only
# place a name invented out of nothing can still be caught.
#
# Deliberately narrow: it only looks at branch names, only in final
# replies, and only flags a name that appears NOWHERE in this
# conversation's tool results or in the clinic's own configured
# branches. A name the model got from a tool is never flagged, however
# it's phrased.

# \b IS LOAD-BEARING: without it "فرع" also matches as a bare SUFFIX of
# "الفرع" ("the branch", definite article glued on with no space).
# Confirmed real production false-positive: "...أبغى أتأكد من اختيار
# الدكتور والفرع أولاً" ("...I want to confirm the doctor and the
# branch first") - a perfectly correct reply with no branch name in it
# at all - matched "فرع" inside "الفرع" and captured the next word
# "أولاً" ("first") as if it were an invented branch called "أولاً",
# forcing a pointless correction retry that re-asked the patient to
# reconfirm a doctor that was already settled. \b sees no boundary
# between "ل" and "ف" (both word characters), so "الفرع" is correctly
# left alone, while "فرع الدقي" (space before "فرع") still matches.
# Quote characters (straight, curly, Arabic) are excluded too: a real
# branch name never contains one, but a reply that CORRECTLY denies a
# name the patient invented ("ما لقيت فرع اسمه \"فرع النيل\"") quotes
# that invalid name right back at them, and without this exclusion the
# capture runs straight through the quote and swallows the rest of the
# denial sentence as if it were itself a branch name.
_BRANCH_MENTION_RE = re.compile(r"\bفرع\s+([^\n،,.؟?:()\[\]0-9️⃣\"'«»\u201c\u201d]{2,25})")

# Words that follow "فرع" in ordinary questions rather than naming one -
# "أي فرع تفضل؟" is not a branch called "تفضل". Without this the
# verifier flags its own clinic on a perfectly correct question, and a
# check that cries wolf gets switched off.
_NOT_A_BRANCH_NAME = {
    "تفضل", "تفضلين", "تفضلي", "معين", "معيّن", "تاني", "ثاني", "تانية", "ثانية",
    "قريب", "قريبة", "مناسب", "مناسبة", "محدد", "محددة", "يناسبك", "تحب", "تحبين",
    "اخر", "آخر", "أخرى", "اخرى", "غيره", "غيرها", "كذا", "معينة",
    "تحجز", "تحجزين", "احجز", "أحجز", "فيه", "فيها", "كمان", "ايضا", "أيضا",
    "ولا", "او", "أو", "من", "في", "علي", "على", "عند", "عندنا", "عندكم",
    "متاح", "متاحة", "المتاحة", "المتاح", "بس", "برضه", "برضو", "هو", "هي",
    "العيادة", "العياده", "عيادة", "عياده", "اللي", "الي", "التي", "الذي",
    "تزور", "تزوري", "تختار", "تختاري", "يزور",
    # Number-words following "فرع" describe HOW MANY branches, not name
    # one - "متوفر في فرع واحد" ("available in one branch"). Confirmed
    # real false positive: "واحد" was captured as an invented branch
    # name, forcing a pointless correction retry over a perfectly
    # accurate reply.
    "واحد", "واحده", "واحدة", "اثنين", "اثنان", "تلاته", "ثلاثة", "التالي",
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): "حابب
    # تحجز في أنهي فرع وانهي يوم؟" ("which branch and which day would
    # you like?") was parsed as a branch literally named "وانهي يوم" -
    # the second question word ("وانهي"/"which... and") landed right
    # after "فرع" because this is a compound question (branch AND day),
    # a shape none of the entries above were written to catch. The
    # reply was 100% correct (both real branches it named earlier in
    # the same message were already known) and still got discarded and
    # replaced with a generic error after two failed correction
    # retries.
    "اني", "أني", "انهي", "أنهي", "وانهي", "وأنهي", "اي", "أي", "وأي", "واي",
    "يوم", "ايه", "إيه",
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): "ما لقيت
    # فرع اسمه \"فرع النيل\"" - a correct denial that a patient-invented
    # branch exists - was itself parsed as naming a branch, because
    # "اسمه" ("named") introduces someone else's claim about a branch,
    # not a branch name. Without this the denial sentence gets treated
    # as the invention it's actually refuting.
    "اسمه", "اسمها", "اسمك", "اسمكم", "اسم",
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-31): a
    # perfectly correct recovery message - "ما قدرنا نلاقي فرع بالرقم
    # 1... تختار من: 1️⃣ المنار 2️⃣ النزهة" - was rejected as naming an
    # invented branch. "فرع بالرقم" ("branch BY NUMBER") refers to the
    # patient's numeric pick, not a name - the digit itself is already
    # excluded from the capture, but the word introducing it ("بالرقم"/
    # "برقم"/"رقم") wasn't, so it was captured alone as if it were the
    # branch's actual name.
    "رقم", "بالرقم", "برقم", "الرقم", "لرقم",
}

_NOT_A_BRANCH_NAME_NORM = {tools._normalize_arabic(w) for w in _NOT_A_BRANCH_NAME}

# STRICT MODE IS NOW THE DEFAULT.
#
# This flag governs all six reply verifiers below (invented branch,
# invented date/time, fabricated complaint confirmation, fabricated
# handoff confirmation, doctor-roster re-offer, unauthorised
# gynaecology). It used to default to FALSE, which meant every one of
# them only wrote a WARNING to the log while the offending reply was
# delivered to the patient unchanged - so a fabricated appointment or a
# branch that does not exist still reached them, and the only trace was
# a log line nobody was watching.
#
# These checks are deliberately narrow: each one fires only when the
# reply asserts something that appears in NO tool result in this
# conversation. When one does fire, "log it and send it anyway" is not a
# defensible default for a medical booking assistant - a patient can
# accept an appointment that exists nowhere. Strict mode re-asks the
# model once, with a targeted correction, and falls back to the original
# reply if that retry doesn't produce something clean, so the worst case
# is one extra LLM call on an already-suspect turn.
#
# Set BRANCH_VERIFIER_STRICT=false to restore the old log-only
# behaviour.
_BRANCH_VERIFIER_STRICT = os.getenv("BRANCH_VERIFIER_STRICT", "true").strip().lower() not in ("0", "false", "no", "off")


def _norm_ar(text: str) -> str:
    """Arabic-aware normalization, so a branch written with a different
    alef/ya/ta-marbuta form still matches the configured spelling.

    Confirmed real false positive: the reply's "الدقي" was flagged as
    invented while the config held the same branch under a slightly
    different spelling - whitespace-only comparison could not see they
    were the same name, and a verifier that accuses correct replies gets
    switched off."""

    return tools._normalize_arabic(_normalize_for_compare(text))


def _known_branch_text(state: AgentState) -> str:
    """Everything this conversation has legitimately been told about
    branches: the raw text of every tool result so far, plus the
    clinic's own configured branch names.

    Kept as one big string and tested with substring containment rather
    than parsed into a name list. Tool results nest branch names under
    several different keys depending on which tool produced them, and a
    parser that missed one would flag a perfectly real branch - a false
    accusation is worse here than a missed one, because it would block
    a correct reply."""

    parts = []

    for msg in state.get("messages", []):
        if getattr(msg, "type", None) == "tool" or getattr(msg, "name", None):
            content = getattr(msg, "content", "")
            if content:
                parts.append(str(content))

    templates = state.get("templates") or {}
    for entry in templates.get("_branch_aliases") or []:
        parts.extend(entry.get("aliases") or [])

    raw_config = state.get("raw_client_config") or {}
    for key, value in raw_config.items():
        if "branch" in key.lower() and isinstance(value, str):
            parts.append(value)

    # PERSISTENT MEMORY, NOT JUST THIS TURN'S MESSAGE HISTORY.
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-31): a reply
    # correctly named "فرع الطوارئ" - a branch the patient had already
    # been shown by name three turns earlier - and was rejected twice
    # as an invented branch anyway, because scanning `state["messages"]`
    # alone no longer surfaced that name as raw text by the time this
    # turn's reply was checked. `tools._remember_list` already tracks
    # every branch name ANY tool has ever returned in this session,
    # independent of whichever list was shown most recently - fold it
    # in here so this check can't go blind to a name the conversation
    # was legitimately told about earlier.
    for name in tools.get_known_entity_names(state.get("session_id"), "branch"):
        parts.append(name)

    return _norm_ar(" | ".join(str(p) for p in parts))


_DOCTOR_MENTION_RE = re.compile(
    r"(?:^|\n)\s*(?:[1-9]\uFE0F?\u20E3|[1-9][\.\)])\s*(?:د\.?|الدكتور[هة]?|دكتور[هة]?)?\s*"
    r"([^\n—\-·(]{3,40})"
)

# Lines that are numbered but are NOT people. A doctor guard that reads
# these as names flags a perfectly good times list as invented doctors.
#
# CONFIRMED REAL FALSE POSITIVE: a 20-item slot list ("1️⃣ 10:45 صباحًا",
# "2️⃣ 10:55 صباحًا"...) was rejected twice as "doctors that appear in no
# tool result", burning two model calls on a reply built entirely from
# real tool data.
_NON_DOCTOR_LIST_ITEM_RE = re.compile(
    r"^\s*\d{1,2}\s*[:：]\s*\d{2}"          # a clock time
    r"|صباح|مساء|ظهر|فجر|عصر|ليل"          # part-of-day words
    r"|^\s*\d{1,2}\s*/\s*\d{1,2}"           # a date
    r"|الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعه|السبت|الاحد"  # weekdays
    r"|^\s*(?:am|pm)\b|\b\d{1,2}\s*(?:am|pm)\b"
)


def _looks_like_a_person_name(candidate: str) -> bool:
    """Whether a numbered list entry plausibly names a person at all."""

    folded = _norm_ar(candidate)
    if not folded or len(folded) < 5:
        return False
    if _NON_DOCTOR_LIST_ITEM_RE.search(folded):
        return False
    if any(ch.isdigit() for ch in folded):
        return False
    return True


def _doctor_names_from_tools(state: AgentState) -> set:
    """Every doctor name any tool result in this conversation returned.

    ROBUST TO SERIALIZATION FORMAT: a tool's dict return value normally
    reaches here as valid double-quoted JSON (LangGraph's ToolNode tries
    `json.dumps` first), but this project's OWN code elsewhere routinely
    also checks tool-message content against a single-quoted Python
    repr form (e.g. `"'status': 'sent'" in content`), confirming that
    tool content is not reliably one format only in practice - a
    fallback error path, an older serialization, or a differently-typed
    return value can all produce it differently. `json.loads` alone
    silently drops every tool result it cannot parse - and since
    `known` being EMPTY makes this whole check stand down entirely (see
    below), any parsing gap here doesn't just miss one name, it disables
    the invented-doctor guard for the rest of the conversation.
    `ast.literal_eval` is tried as a fallback specifically to close that
    gap."""

    names = set()

    for msg in state.get("messages", []) or []:
        if getattr(msg, "type", None) != "tool":
            continue

        raw = getattr(msg, "content", None)

        data = None
        if isinstance(raw, (dict, list)):
            data = raw
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                try:
                    data = ast.literal_eval(raw)
                except (ValueError, SyntaxError, TypeError, MemoryError):
                    data = None

        if data is None:
            continue

        def _collect(node):
            if isinstance(node, dict):
                for key in ("name", "formatedName", "altName", "doctorName"):
                    value = node.get(key)
                    if value:
                        names.add(_norm_ar(str(value)))
                for value in node.values():
                    _collect(value)
            elif isinstance(node, list):
                for value in node:
                    _collect(value)

        _collect(data)

    # Same persistent-memory reasoning as `_known_branch_text` above -
    # fold in every doctor name `tools._remember_list` has ever recorded
    # for this session, not just what's still scannable in this turn's
    # message history.
    for name in tools.get_known_entity_names(state.get("session_id"), "doctor"):
        names.add(_norm_ar(name))

    return {n for n in names if n}


_TIME_LIST_RE = re.compile(r"[1-9]\uFE0F?\u20E3\s*\d{1,2}\s*[:：]\s*\d{2}")
_SOONEST_OFFER_RE = re.compile(
    r"اقرب\s*موعد|أقرب\s*موعد|هل\s*يناسبك|يناسبك\s*(?:هذا|ده|هالموعد)|"
    r"earliest\s*(?:available\s*)?appointment|does\s*(?:this|that)\s*work"
)


def _reply_dumps_times_without_offering_soonest(reply_text: str, state: AgentState) -> bool:
    """True when a reply answers a DAY choice with the full list of that
    day's times, instead of offering the soonest one first.

    The agreed flow is: day settled -> ONE concrete soonest appointment
    + "does that suit you?" -> only if they decline, the full list. A
    wall of twenty times is what that flow exists to avoid, and the
    prose rule for it has not held on its own.

    CONFIRMED REAL PRODUCTION FAILURE: the patient answered "الأحد", and
    got "المواعيد المتاحة ليوم الأحد 30/08/2026" followed by eight
    numbered times - no soonest-appointment offer at all, while the
    identical step in another flow did it correctly.

    Fires only when the patient's own last message was a DAY choice, so
    a time list they explicitly asked for is untouched.

    EXEMPTION - does NOT fire when the most recent tool call was
    `get_available_slots_for_booking` returning "found": that result is
    handled by `_build_slots_numbered_list_directive`, which FORCES the
    exact same full numbered list verbatim as the only acceptable reply
    for that turn. Without this exemption the two directives contradict
    each other on the identical trigger (patient names a weekday ->
    resolve_available_day/get_available_slots_for_booking run in the
    same turn -> full list comes back): the model is first ordered to
    print the whole list, then this check immediately rejects that same
    list and forces it back down to a single "soonest" offer instead.

    CONFIRMED REAL PRODUCTION FAILURE (the mirror-image one this
    exemption fixes): patient said "الثلاثاء", got the correct 7-slot
    numbered list, which this check then overrode into "أقرب موعد
    متاح ... من 11:00 إلى 11:30 - هل يناسبك؟". The patient replied
    "مناسب" believing they were confirming the single time shown, when
    in fact 7 times were available and the intended next step was to
    show all of them, not silently narrow to the first."""

    if not reply_text:
        return False

    if len(_TIME_LIST_RE.findall(reply_text)) < 3:
        return False

    if _SOONEST_OFFER_RE.search(_norm_ar(reply_text)):
        return False

    last_tool_msg = next(
        (m for m in reversed(state.get("messages") or []) if getattr(m, "type", None) == "tool"),
        None,
    )
    if getattr(last_tool_msg, "name", None) == "get_available_slots_for_booking":
        try:
            tool_data = json.loads(last_tool_msg.content)
        except (ValueError, TypeError):
            tool_data = None
        if isinstance(tool_data, dict) and tool_data.get("status") == "found":
            return False

    from langchain_core.messages import HumanMessage as _HumanMessage

    last_human = None
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, _HumanMessage):
            content = getattr(msg, "content", "")
            last_human = content if isinstance(content, str) else str(content)
            break

    if not last_human:
        return False

    folded = _norm_ar(last_human)

    picked_a_day = bool(
        re.search(
            r"الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعه|السبت|الاحد|"
            r"بكره|بكرا|اليوم|غدا|monday|tuesday|wednesday|thursday|friday|saturday|sunday",
            folded,
        )
    )

    if not picked_a_day:
        return False

    # If they asked outright to SEE the times, showing them is correct.
    if re.search(r"الاوقات|المواعيد\s*المتاحه|وريني|اعرض", folded):
        return False

    return True


_SOONEST_FIRST_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "OFFER THE SOONEST APPOINTMENT FIRST - NOT THE WHOLE DAY'S LIST\n"
    "============================================================\n"
    "The patient chose a DAY, and your previous draft answered with "
    "every time on it. That is a wall of options where one sentence "
    "would have finished the booking.\n\n"
    "Rewrite it as a single concrete offer - the EARLIEST time from the "
    "tool result you already have - plus one question, in this exact "
    "plain shape (the same one used everywhere else in this project "
    "for a single day/time offer):\n"
    "    أقرب موعد متاح عند [doctor name] في [branch name]:\n"
    "    🗓️ [weekday] [date] — [earliest time]\n"
    "    هل يناسبك هذا الموعد؟\n"
    "Take the doctor, branch, date and time verbatim from the tool "
    "result; invent nothing. Omit \"عند [doctor]\"/\"في [branch]\" only "
    "if that information genuinely isn't available yet.\n\n"
    "Only if they say it does NOT suit them do you show the rest of the "
    "day's times as a numbered list.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: the patient answered \"الأحد\" "
    "and received eight numbered times with no soonest-appointment "
    "offer, while the same step in another flow correctly offered one "
    "appointment and asked.\n\n"
)


# Broader than `_DOCTOR_CUE_WORD_RE` (defined later, used for a
# different purpose at the `match_entity_info` call site) - this one
# needs to catch the plural/title forms a doctor-roster REPLY actually
# uses ("الدكاترة", "الأطباء", "استشاري", "أخصائي"), not just the
# singular "دكتور"/"طبيب" a patient's own message tends to use.
_DOCTOR_LIST_CUE_RE = re.compile(
    r"دكتور|دكتوره|د\.|طبيب|أطباء|اطباء|الدكاترة|دكاترة|استشاري|أخصائي|اخصائي|doctor"
)


def _find_invented_doctors(reply_text: str, state: AgentState) -> list:
    """Doctor names presented in a numbered list that no tool result in
    this conversation ever returned.

    WHY THIS EXISTS: a doctor roster is the single most dangerous thing
    to invent - the patient picks a name, and everything downstream
    (schedule, slots, the booking itself) is built on a person who may
    not work here.

    CONFIRMED REAL PRODUCTION FAILURE: "الدكاترة المتاحين في تخصص طب
    الباطنة عندنا الآن: 1️⃣ د. طه مبروك 2️⃣ د. سارة عبد الله 3️⃣ د.
    محمود سليمان" went out with NO tool call at all in that turn - no
    `find_available_doctors`, no `_remember_list`, nothing. The names
    were recalled from earlier in the conversation and presented as a
    current, specialty-filtered roster.

    Only fires when the reply presents an actual numbered LIST of
    people and at least one entry matches nothing the tools returned -
    prose that merely mentions a doctor already discussed is left
    alone."""

    if not reply_text:
        return []

    # GATE - the reply must actually be about doctors before we treat any
    # numbered list in it as a doctor roster to verify. Without this, the
    # "is this a person name?" test below is a DENYLIST (reject only
    # times/dates/weekdays/anything-with-a-digit) rather than an
    # ALLOWLIST for doctor content, so it also fires on any other
    # numbered list a reply might legitimately contain - specialties,
    # services, symptoms, FAQ items - none of which are doctor names at
    # all, just multi-word Arabic phrases with no digits in them.
    # CONFIRMED REAL FALSE POSITIVE CLASS (found while fixing the branch
    # false positive above, same root cause): a specialty list
    # ("1️⃣ جراحة العظام", "2️⃣ طب الأطفال"...), a services list, and a
    # symptom list were ALL flagged as "invented doctors" with zero
    # doctor-related wording anywhere in the reply. `_find_invented_branches`
    # already guards itself this way (it requires "فرع" to appear at
    # all before scanning) - this mirrors that same pattern for doctors.
    if not _DOCTOR_LIST_CUE_RE.search(reply_text):
        return []

    known = _doctor_names_from_tools(state)

    # NOTE: `known` being empty is deliberately NOT treated as "nothing
    # to check" any more. It used to return [] here on the reasoning
    # that "other guards cover the no-tool-call case" - but no such
    # guard actually exists, and this exact gap is what let a fully
    # invented 4-doctor roster (covering four different specialties)
    # through with zero tool activity anywhere in the turn. CONFIRMED
    # REAL PRODUCTION FAILURE (medtown, 2026-08-30): the patient said
    # "معرفوش" (I don't know [which doctor]) and got "أعرض لك الدكاترة
    # المتاحين عندنا الحين" followed by four names and specialties, none
    # of which any tool in this conversation had ever returned. With
    # `known` empty, every person-like candidate below simply matches
    # nothing in it and is correctly reported as invented - this is the
    # SAME loop as the normal case, just with an empty comparison set
    # instead of a special early exit.
    known_branches = _known_branch_text(state)

    invented = []
    for match in _DOCTOR_MENTION_RE.finditer(reply_text):
        raw_candidate = match.group(1)
        if not _looks_like_a_person_name(raw_candidate):
            # Times, dates, weekdays - a numbered list is not always a
            # list of people. See _NON_DOCTOR_LIST_ITEM_RE.
            continue
        # A branch listing also uses "1️⃣ <name>" formatting, and a bare
        # branch name (e.g. "المنار") passes the person-name shape test
        # just as easily as a real doctor name does. Branch entries are
        # normally followed by an address, which no doctor entry ever
        # has - use that as the disambiguator so branch lists don't get
        # misread as an invented doctor roster.
        # CONFIRMED REAL FALSE POSITIVE (medtown, 2026-08-30): a 2-branch
        # list ("1️⃣ المنار / العنوان: ...", "2️⃣ النزهة / العنوان: ...")
        # was rejected twice as invented doctors, forcing the generic
        # fallback error message to go out instead of a valid answer.
        lookahead = reply_text[match.end():match.end() + 120]
        # SECOND CONFIRMED REAL FALSE POSITIVE (medtown, 2026-08-30,
        # same session, immediately after the first fix landed): the
        # literal word "العنوان" is not the only way an address shows
        # up. The model also writes it inline with no label at all -
        # "1️⃣ الطوارئ - مركز تناسق للرعاية الطبية العاجلة، RQMA3217، "
        # "7131 ابن النفيس، الرياض 14222" - which has no "العنوان"
        # anywhere, so that check alone still let a 3-branch list get
        # rejected twice and replaced with the generic fallback error.
        # Two more signals now count as "this is an address, not a
        # doctor", either one sufficient:
        #   1. The candidate name itself is one of this conversation's
        #      known branches (tool results / client config) - reuse
        #      the exact same source _find_invented_branches checks
        #      against, so the two guards can't disagree with each
        #      other about what a branch is.
        #   2. The text right after the name has an address shape: a
        #      run of 3+ digits (a building/unit number or postal
        #      code like "14222") together with a comma - a doctor
        #      entry is never followed by that combination.
        candidate = _norm_ar(raw_candidate)
        is_known_branch = candidate and (
            candidate in known_branches
            or any(part in known_branches for part in candidate.split() if len(part) >= 3)
        )
        looks_like_address = bool(
            "العنوان" in lookahead
            or (re.search(r"\d{3,}", lookahead) and "،" in lookahead)
        )
        if is_known_branch or looks_like_address:
            continue
        # Substring either way: tool results carry titles and suffixes
        # the reply trims ("د. طه مبروك" vs "طه مبروك — استشاري").
        if any(candidate in k or k in candidate for k in known):
            continue
        invented.append(match.group(1).strip())

    return invented


_INVENTED_DOCTORS_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "THOSE DOCTORS DID NOT COME FROM A TOOL - DO NOT LIST THEM\n"
    "============================================================\n"
    "Your previous draft presented a list of doctors that no tool "
    "result in this conversation returned. Some or all of those names "
    "were written from memory.\n\n"
    "A doctor roster is never something to recall or reconstruct. The "
    "patient picks a name from it, and the schedule, the slots and the "
    "booking are all built on that person - so an invented name sends "
    "them to someone who may not work here, or may not be available at "
    "all.\n\n"
    "Call `find_available_doctors` (or `match_entity_for_booking` in "
    "list mode) NOW and show ONLY the names it returns, in its exact "
    "order. If it returns nobody, say that plainly instead of listing "
    "anyone.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: a three-name specialty roster "
    "went out with no tool call in that turn at all - the names were "
    "carried over from earlier in the conversation and presented as a "
    "current, specialty-filtered list.\n\n"
)


def _find_invented_branches(reply_text: str, state: AgentState) -> list:
    """Branch names the reply mentions that this conversation was never
    actually given. Empty list means nothing to flag."""

    if not reply_text or "فرع" not in reply_text:
        return []

    known = _known_branch_text(state)
    if not known:
        # Nothing to compare against (no tool results yet, no config) -
        # stay silent rather than flagging everything.
        return []

    # Strip quoted spans before scanning. A reply that correctly denies
    # a name the PATIENT invented quotes that name straight back at
    # them ('ما لقيت فرع اسمه "فرع النيل"') - the quoted text is being
    # referenced, not asserted, and scanning it anyway means the denial
    # sentence gets flagged as the very invention it's refuting.
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): exactly
    # this sentence, followed by two genuinely correct branches, was
    # discarded twice and replaced with the generic fallback error.
    scan_text = re.sub(r"[\"'«»\u201c\u201d][^\"'«»\u201c\u201d\n]{1,40}[\"'«»\u201c\u201d]", " ", reply_text)

    invented = []
    for match in _BRANCH_MENTION_RE.finditer(scan_text):
        name = _norm_ar(match.group(1))
        if not name or len(name) < 3:
            continue
        # A branch name is the run of words right after "فرع", stopping
        # at the first word that clearly isn't part of a name ("أي فرع
        # تفضل تحجز فيه؟" -> nothing; "فرع المعادي كمان" -> "المعادي").
        # Names are 1-3 words in practice, so anything longer is prose
        # that happened to follow the word "فرع", not a name.
        words = []
        for word in name.split():
            if word in _NOT_A_BRANCH_NAME or word in _NOT_A_BRANCH_NAME_NORM:
                break
            words.append(word)
            if len(words) == 3:
                break
        if not words:
            continue
        name = " ".join(words)
        if len(name) < 3:
            continue
        if name in known:
            continue
        # Also accept a partial: "الشيخ زايد" mentioned as "زايد".
        if any(part in known for part in name.split() if len(part) >= 3):
            continue
        # DEFENSE IN DEPTH for the same class of bug
        # `_arabic_preferred_name` is now fixed at the source for
        # (Emergency -> الطوارئ etc.) - this covers any STALE session
        # still holding the old English-only name from before that fix
        # was deployed, or any other generic institutional word this
        # guard hasn't been told the translation of yet. If the
        # patient's own reply is naming a branch the tools reported
        # only in English, and the Arabic name given here is the known
        # standard translation of that English word, it isn't invented
        # - it's a correct translation of a real tool result. Checked
        # per WORD, not just the full candidate, for the same reason
        # the direct-match fallback above is per-word: the extracted
        # candidate can carry trailing words from the sentence after
        # the branch name ("الطوارئ يقدم خدمة") when the verb that
        # follows isn't in `_NOT_A_BRANCH_NAME`.
        translated_parts = {
            en for en, ar in tools._GENERIC_BRANCH_NAME_AR.items()
            if ar in name.split()
        }
        if translated_parts and any(_norm_ar(en) in known for en in translated_parts):
            continue
        if name not in invented:
            invented.append(name)

    return invented


_BRANCH_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU LISTED A BRANCH THAT DOES NOT EXIST - REWRITE YOUR REPLY\n"
    "============================================================\n"
    "Your previous draft named at least one branch that has NOT appeared "
    "in any tool result in this conversation and is not one of this "
    "clinic's configured branches: {names}\n\n"
    "You may only name branches that came from a tool result. If you "
    "need the branch list for a confirmed doctor, call the tool that "
    "returns it rather than writing names from memory - a doctor's "
    "branches are returned to you as `branchesForDoctor` the moment the "
    "doctor is confirmed, and `list_branches_for_specialty` returns them "
    "too.\n\n"
    "Rewrite the reply now using ONLY real branches, or call the tool "
    "first if you don't have them.\n\n"
)

# (?<![\w-]) / (?![\w-]) ARE LOAD-BEARING - confirmed false positive.
#
# A booking reference looks like "GBN-2026-06-20-151", and the bare
# pattern below happily matched the substring "26-06-20" INSIDE it. So
# every correct reply that quoted the patient's own reference number -
# which the cancellation flow does on literally every confirmation -
# was read as containing a date, and then judged against the tool
# results as if the model had invented it.
#
# In log-only mode that was noise. With the verifiers actually
# intervening it is worse than noise: a perfectly accurate reply gets
# thrown away and re-generated, and a check that cries wolf on correct
# output is one nobody can afford to leave switched on. Requiring that
# nothing word-like or hyphen-like sits on either side means a real
# date ("20/08/2026", "on 20-08-2026.") still matches while a run of
# digits embedded in a longer reference does not.
_DATE_IN_REPLY_RE = re.compile(r"(?<![\w-])\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?![\w-])")

# Same class of problem: a reference or an id can carry "...151:30...".
_TIME_IN_REPLY_RE = re.compile(r"(?<![\w:])\d{1,2}:\d{2}(?![\w:])")

# Weekday words that, appearing in a REPLY, must be backed by a real
# availability tool result (see `_reply_invents_availability`).
#
# The colloquial spellings matter as much as the formal ones here: a
# reply written in the patient's own Egyptian register ("الدكتور متاح
# التلات") carried no formal day name at all, so the fabricated-day
# check simply did not see it. Every form the model might echo back
# from a patient's message belongs in this map, not just the MSA ones.
_WEEKDAY_WORDS = {
    "الاثنين": "Monday", "الإثنين": "Monday", "الاتنين": "Monday",
    "الإتنين": "Monday", "التنين": "Monday",
    "الثلاثاء": "Tuesday", "التلات": "Tuesday",
    "التلاتاء": "Tuesday", "الثلثاء": "Tuesday",
    "الأربعاء": "Wednesday", "الاربعاء": "Wednesday",
    "الخميس": "Thursday",
    "الجمعة": "Friday", "الجمعه": "Friday",
    "السبت": "Saturday",
    "الأحد": "Sunday", "الاحد": "Sunday",
}
# DELIBERATELY ABSENT: "الحد", "الثلاث", "الاربع". They are real
# colloquial day names, but each is also an ordinary Arabic word in its
# own right ("الحد الأقصى", "الفروع الثلاث", "الفروع الاربع"), and even
# with word boundaries they would flag correct replies as inventing a
# day. Parsing what the PATIENT typed still understands all three -
# that happens in `tools.resolve_weekday_index`, which matches whole
# words in a message the model did not write. This map only governs
# what counts as a day CLAIM in the assistant's own reply, where a
# false positive is the expensive direction.

# WORD-BOUNDARY matching, not substring. The colloquial day names are
# short enough to hide inside ordinary words - "الحد" sits inside
# "الحدود"/"الحد الأقصى", "الثلاث" inside "الثلاثة" - and a substring
# hit there would flag a perfectly correct reply as inventing a day,
# which costs the patient a wasted correction round or, twice in a row,
# the generic fallback message. An Arabic letter is a \w character, so
#  does the right thing on both sides of these tokens.
_WEEKDAY_WORD_RES = {
    word: re.compile(r"" + re.escape(word) + r"")
    for word in _WEEKDAY_WORDS
}


_AVAILABILITY_TOOLS = (
    "list_available_days_for_booking", "get_available_slots_for_booking",
    "get_available_reschedule_slots", "resolve_available_day",
    "get_doctor_schedule", "get_doctor_schedule_for_booking",
    "find_best_doctor_in_specialty", "lookup_appointment",
    "check_booking_status", "create_new_booking",
)


_AVAILABILITY_DENIAL_RE = re.compile(
    r"ما\s*(?:في|فيه|ظهر|لقيت|لقينا|عندي|عندنا|عنده|عندها)[^.\n؟?]{0,40}مواعيد|"
    r"مفيش[^.\n؟?]{0,40}مواعيد|"
    r"لا\s*(?:يوجد|توجد)[^.\n؟?]{0,40}مواعيد|"
    r"مواعيد[^.\n؟?]{0,25}(?:متاحه|متاحة)?[^.\n؟?]{0,15}(?:حاليا?|الحين|هسه)\b[^.\n؟?]{0,15}(?:عند|مع)|"
    r"no\s+(?:available\s+)?(?:appointments?|slots?|availability)|"
    r"(?:doesn'?t|does\s+not)\s+have\s+(?:any\s+)?(?:available\s+)?(?:appointments?|slots?)"
)

_AVAILABILITY_LOOKUP_TOOLS = (
    "list_available_days_for_booking",
    "get_available_slots_for_booking",
    "resolve_available_day",
    "get_doctor_schedule_for_booking",
    "find_best_doctor_in_specialty",
    "get_available_reschedule_slots",
)


_MEDICATION_MENTION_RE = re.compile(
    # Arabic transliterations of drug names vary a lot in practice, so
    # these are deliberately loose. The confirmed production failure
    # spelled it "البارستامول" (no ي) - a stricter pattern missed it
    # entirely, which is exactly the failure mode to avoid here.
    #
    # DELIBERATELY EXCLUDES the bare generic noun "دواء"/"دوا" (just
    # "medicine"/"medication" with no specific drug or class named).
    # CONFIRMED REAL PRODUCTION FAILURE: a patient's own complaint was
    # "الدواء اللي اتوصفلي غلط" (the medication I was prescribed was
    # wrong) - a COMPLAINT ABOUT a medication, naming no drug at all -
    # and every reply that went on to discuss that complaint (asking for
    # details, summarizing it for confirmation, etc.) kept re-matching
    # this pattern on the bare word "دواء", permanently blocking the
    # entire medication-complaint flow into a generic fallback error.
    # A specific drug NAME or a drug CLASS (بنادول, بروفين, مضاد حيوي,
    # مسكن, ...) still means the assistant is naming/suggesting
    # something to take, which is the actual safety concern this check
    # exists for - the generic word "medicine" on its own does not.
    r"بنادول|باندول|بار[اي]?س?ي?تامول|باراس?ي?تامول|بروفين|بروفن|"
    r"اي?بوبروفين|إيبوبروفين|اسبرين|أسبرين|فولتارين|كتافلام|"
    r"اوجمنتين|أوجمنتين|زيرتك|كلاريتين|"
    r"مضاد\s*حيوي|خافض[^\n]{0,10}حرار|ادوي[هة][^\n]{0,15}حرار|"
    r"مسكن|حبوب[^\n]{0,10}مسكن|علاج\s*من\s*(?:ال)?صيدلي|"
    r"paracetamol|acetaminophen|panadol|ibuprofen|advil|tylenol|aspirin|"
    r"antibiotic|antihistamine|painkiller|analgesic|fever\s*reducer"
)

_MEDICATION_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOUR DRAFT NAMED A MEDICATION - REMOVE IT COMPLETELY\n"
    "============================================================\n"
    "Your previous draft named or suggested a medicine. You are a "
    "booking assistant, not a clinician: you cannot examine anyone and "
    "you do not know their history, allergies, weight, or what else "
    "they take. Medication advice over chat can genuinely hurt "
    "someone - especially a child.\n\n"
    "Rewrite the message with the medication REMOVED. Do not swap it "
    "for a different drug, a 'safe' dose, a drug class, or a vague "
    "\"something from the pharmacy\" - remove the idea entirely.\n\n"
    "What you may offer instead: ordinary non-medical comfort measures "
    "(rest, fluids, a quiet dark room, monitoring, warmth), a warm "
    "wish, and then the actual help you can give - the right specialty "
    "and a real appointment. If they asked what to take, say kindly "
    "that you can't advise on medication and the doctor will decide "
    "that after seeing them.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: a parent described a two-day "
    "fever in their child and the reply recommended fever-reducing "
    "medication \"مثل البارستامول\" adjusted \"لعمره ووزنه\" - naming a "
    "drug and giving dosing guidance for a child.\n\n"
)


_NUMBERED_LIST_ITEM_RE = re.compile(r"[1-9]\uFE0F?\u20E3|^\s*[1-9][\.\)]\s", re.MULTILINE)


_BOOKING_OFFER_RE = re.compile(
    r"تحب\s*(?:ت)?حجز|ترغب\s*(?:في\s*)?(?:ت)?حجز|تبي\s*(?:ت)?حجز|"
    r"احجز\s*لك|اساعدك\s*ب?حجز|نكمل\s*(?:ال)?حجز|"
    r"اكتب\s*اسم\s*(?:ال)?دكتور|"
    r"(?:would you like|want)\s*to\s*book|shall\s*i\s*book"
)


_GENERIC_BRANCH_QUESTION_RE = re.compile(
    r"اي\s*فرع\s*(?:تفضل|تحب|تبي|ترغب)|"
    r"في\s*انهي\s*فرع|"
    r"(?:الفروع|فروع)\s*(?:ال)?متاح|"
    r"which\s*branch\s*(?:would|do)\s*you"
)


_NOT_A_DIAGNOSIS_RE = re.compile(
    # Must match the REQUIRED notice ("...وليست تشخيصًا طبيًا مباشرة")
    # as well as the shorter dialect phrasings. Getting this wrong makes
    # the verifier demand a notice that is already there, so the
    # "وليست/ليست" form is the important one.
    r"وليست\s*تشخيص|ليست?\s*تشخيص|مش\s*تشخيص|مو\s*تشخيص|ما\s*هو\s*تشخيص|"
    r"معلومات\s*عامه|not\s*a\s*(?:medical\s*)?diagnosis|general\s*information"
)

_SPECIALTY_OFFER_RE = re.compile(
    r"عندنا\s*دكاتره|عندنا\s*اطباء|دكاتره\s*متاح|اطباء\s*متاح|"
    r"احجزلك|احجز\s*لك|اشوف\s*لك\s*(?:ال)?دكاتره|"
    r"التخصص\s*(?:ال)?مناسب|تحب\s*(?:ت)?حجز"
)


_SPECIALTY_CHOICE_QUESTION_RE = re.compile(
    r"(?:وش|اي|ايه|انهي)\s*(?:ال)?تخصص|"
    r"(?:ال)?تخصص\s*(?:اللي|الذي)\s*(?:تفضل|تحب|تبي|ترغب)|"
    r"which\s*specialt"
)


_MEDICAL_OFFER_PATTERN_RE = re.compile(
    r"وليست\s*تشخيص|ليست?\s*تشخيص|مش\s*تشخيص|معلومات\s*عامه|"
    r"not\s*a\s*(?:medical\s*)?diagnosis"
)


_ORGAN_SPECIALTY_EXPECTATIONS = (
    # (symptom words in the PATIENT's own message,
    #  specialty words that would genuinely treat it)
    #
    # Deliberately small and only for body parts where the mapping is
    # not a judgement call. This is a floor, not a diagnosis engine: it
    # exists to stop a clearly-wrong referral, not to choose the right
    # one.
    # NOTE the specialty side deliberately does NOT contain the bare
    # organ word ("عين"). CONFIRMED REAL PRODUCTION FAILURE: it did, and
    # the advice line "وما تفرك عينك" satisfied the check - so an eye
    # complaint offered طب الباطنة and this guard stayed silent. The
    # specialty side must only ever hold words that name a SPECIALTY.
    (("عين", "عيون", "عيني", "نظر", "الرؤيه", "eye", "vision"),
     ("عيون", "رمد", "بصريات", "شبكيه", "الجسم الزجاجي", "ophthal", "retina")),
    (("ضرس", "اسنان", "سناني", "لثه", "tooth", "teeth", "dental"),
     ("اسنان", "فم وفكين", "dental", "dent")),
    (("جلد", "بشره", "طفح", "حكه", "skin", "rash"),
     ("جلدي", "جلديه", "derma")),
    (("ودن", "سمعي", "طنين", "hearing"),
     ("انف واذن", "حنجره", "otolar")),
    (("عظم", "عظام", "كسر", "مفصل", "ركبه", "bone", "fracture", "joint"),
     ("عظام", "مفاصل", "ortho", "روماتيزم")),
)


def _medical_reply_offers_unrelated_specialty(reply_text: str, state: AgentState) -> bool:
    """True when the patient named a clearly organ-specific symptom and
    the reply offers a specialty that plainly does not treat it.

    Judging clinical relevance in general is not something code can do,
    and this does not try to. It covers only a handful of body parts
    where the mapping is not a matter of opinion - an eye complaint
    needs eyes, a tooth needs dentistry - and fires only when the
    offered specialty matches NONE of the plausible ones.

    CONFIRMED REAL PRODUCTION FAILURES, the same root cause twice: the
    clinic has no ophthalmology registered, and "عيني وجعاني وبتدمع"
    was answered first with an offer of طب الأطفال, then - after the
    contradiction guard forced internal consistency - with طب الباطنة
    in both lines. Consistent, and still an eye complaint routed to
    internal medicine. The correct answer in both cases was that this
    clinic has no eye doctor right now.

    The right response to this firing is NOT to find a different
    specialty - it is to say plainly that the clinic doesn't have one."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)

    if not _SPECIALTY_OFFER_RE.search(folded):
        return False

    # CHECK ONLY THE OFFER, NOT THE WHOLE REPLY.
    #
    # The advice line legitimately names the RIGHT specialty ("يفضل
    # تراجع طبيب عيون فورًا") while the offer names a wrong one.
    # Scanning the whole message let that advice line satisfy the check,
    # so the guard stayed silent on the exact failure it exists for.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: "...يفضل تراجع طبيب عيون فورًا
    # ... عندنا في مستشفى ميدتاون الطبية دكاترة في تخصص طب الأطفال
    # متاحين — تحب أحجزلك عند واحد منهم؟" - "عيون" was present in the
    # message, so the eye check passed, while the patient was in fact
    # being offered a paediatrician.
    # Split the RAW text into lines before normalizing: `_norm_ar`
    # collapses all whitespace (newlines included) into single spaces,
    # so splitting after it yields one line and the scoping is lost.
    #
    # A numbered doctor roster IS the offer, just spread across its own
    # lines rather than living on the closing question line ("تحب تحجز
    # عند أي واحد فيهم؟"). That closing line alone rarely repeats the
    # specialty word, so scoping to _SPECIALTY_OFFER_RE lines only was
    # throwing away the very lines that name the specialty being
    # offered - the roster entries above it.
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): "1️⃣ ...
    # · جراحة العظام\n2️⃣ ... · جراحة العظام\nتحب تحجز عند أي واحد
    # فيهم؟" for a patient who said "وجع في عظام رجلي" - the roster
    # lines correctly named جراحة العظام, but only the closing question
    # matched _SPECIALTY_OFFER_RE, so the specialty word never reached
    # `offer_text` and a completely correct referral was rejected twice.
    _NUMBERED_LIST_LINE_RE = re.compile(r"^\s*(?:[1-9]\uFE0F?\u20E3|[1-9][\.\)])\s*")
    offer_text = "\n".join(
        normalized_line
        for normalized_line in (_norm_ar(line) for line in reply_text.splitlines())
        if _SPECIALTY_OFFER_RE.search(normalized_line)
        or _NUMBERED_LIST_LINE_RE.match(normalized_line)
    ) or folded

    from langchain_core.messages import HumanMessage as _HumanMessage

    patient_text = " ".join(
        _norm_ar(str(getattr(m, "content", "")))
        for m in (state.get("messages") or [])
        if isinstance(m, _HumanMessage)
    )

    if not patient_text:
        return False

    for symptom_words, ok_specialty_words in _ORGAN_SPECIALTY_EXPECTATIONS:
        if not any(w in patient_text for w in symptom_words):
            continue
        # The symptom is in play. If the reply names ANY specialty that
        # could treat it, that's fine.
        if any(w in offer_text for w in ok_specialty_words):
            return False
        return True

    return False


_UNRELATED_SPECIALTY_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "THAT SPECIALTY DOES NOT TREAT WHAT THEY DESCRIBED\n"
    "============================================================\n"
    "The patient named a symptom in a specific part of the body, and "
    "your previous draft offers them a doctor in a specialty that does "
    "not treat it. Booking that appointment would cost them a trip and "
    "leave them still needing the right doctor.\n\n"
    "DO NOT SWAP IN ANOTHER SPECIALTY FROM THE LIST. If the fitting "
    "specialty is not registered at this clinic, that is the honest "
    "answer and you should give it:\n"
    "  1. Keep the warm line, the comfort measures, and the red flags.\n"
    "  2. Say plainly that this clinic doesn't currently have a doctor "
    "for this - e.g. \"للأسف ما عندنا دكتور عيون حاليًا في المستشفى\".\n"
    "  3. Offer to connect them with a staff member, or to help with "
    "something else.\n"
    "Do not offer any doctor here for this complaint.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: \"عيني وجعاني وبتدمع\" (eye pain "
    "with watering) was answered with an offer of طب الأطفال, and then "
    "on a retry with طب الباطنة - because this clinic has no "
    "ophthalmology and the reply would not say so. An honest \"not "
    "here\" is worth more than a confident wrong referral.\n\n"
)


def _medical_reply_names_two_specialties(reply_text: str, state: AgentState) -> bool:
    """True when a medical-guidance reply tells the patient to see one
    specialty and then offers an appointment in a DIFFERENT one.

    That contradiction is mechanically checkable even though "is this
    specialty relevant?" is not, and it is a reliable symptom of the
    underlying failure: the model knows which specialty the symptom
    needs, finds the clinic doesn't have it, and offers the nearest
    available one anyway - leaving both in the same message.

    CONFIRMED REAL PRODUCTION FAILURE: "عيني وجعاني وبتدمع" produced
    "راجع دكتور طب الأطفال أو استشاري عيون فورًا" followed by "عندنا في
    مستشفى ميدتاون دكاترة في طب الأطفال متاحين - تحب أحجز لك موعد عند
    واحد منهم؟" - advising an eye consultant while offering a
    paediatrician, for an adult's eye complaint."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)

    if not _SPECIALTY_OFFER_RE.search(folded):
        return False

    # A DOCTOR LIST LEGITIMATELY CARRIES SEVERAL SPECIALTIES.
    #
    # Each doctor is labelled with their own specialty, so a two-doctor
    # list naturally names two - informative, not contradictory. This
    # guard is about the ADVICE line and the OFFER line disagreeing,
    # which is a different thing.
    #
    # CONFIRMED REAL FALSE POSITIVE: "الأطباء المتاحين: 1️⃣ رانيا عبد
    # الرحمن — استشاري · باطنه عام / 2️⃣ فارس الشارخ — اخصائي · طب
    # الباطنة" was rejected twice as advising one specialty and offering
    # another. It was a correct list built from a real tool result, and
    # each retry burned a model call.
    if len(_NUMBERED_LIST_ITEM_RE.findall(reply_text)) >= 2:
        return False

    specialty_names = set()
    for msg in state.get("messages", []) or []:
        if getattr(msg, "name", None) != "list_specialties":
            continue
        try:
            data = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        for item in data.get("specialties") or data.get("items") or []:
            if isinstance(item, dict):
                for key in ("name", "altName"):
                    value = item.get(key)
                    if value:
                        normalized = _norm_ar(str(value))
                        if len(normalized) >= 4:
                            specialty_names.add(normalized)

    if not specialty_names:
        return False

    mentioned = {name for name in specialty_names if name in folded}

    # Two or more DIFFERENT registered specialties named in a single
    # guidance reply. One is the answer; two means the reply could not
    # decide, and the patient is the one left holding the ambiguity.
    return len(mentioned) >= 2


_TWO_SPECIALTIES_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOUR REPLY NAMES TWO DIFFERENT SPECIALTIES - PICK ONE\n"
    "============================================================\n"
    "Your previous draft points the patient at one specialty and offers "
    "an appointment in another. That is incoherent from their side: "
    "they cannot tell which doctor they are actually being sent to.\n\n"
    "Decide which specialty the SYMPTOM they described actually needs, "
    "and use that one in BOTH the advice line and the offer line.\n\n"
    "If this clinic does not have that specialty registered, do NOT "
    "substitute the nearest available one. Say plainly that there is no "
    "doctor for this here at the moment, keep the comfort advice and "
    "the red flags, and offer to connect them with a staff member "
    "instead. An honest \"not here\" is worth far more than a confident "
    "wrong referral.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: eye pain with watering was "
    "answered with \"راجع دكتور طب الأطفال أو استشاري عيون فورًا\" and "
    "then an offer of طب الأطفال doctors - a paediatrician offered for "
    "an adult's eye, because ophthalmology was not in the list.\n\n"
)


def _in_medical_guidance_handoff(state: AgentState) -> bool:
    """True when the assistant's last message was a medical-guidance
    offer ("...عندنا دكاترة باطنة متاحين، تحب أحجزلك؟") - i.e. the
    patient's current reply is them accepting it.

    WHY THIS IS NEEDED: the ROUTER moves the conversation out of the
    `medical` agent the moment the patient agrees - "اه" is classified
    as a bare affirmation answering a booking offer, so the very next
    turn runs under `booking`. Any guard gated on `agent_name ==
    "medical"` is therefore silent on exactly the turn after the
    guidance reply, which is where these failures happen.

    CONFIRMED REAL PRODUCTION FAILURE: a correct guidance reply named
    طب الباطنة and offered an appointment; the patient said "اه"; and
    the reply was "وش التخصص اللي حاب تحجز فيه؟" followed by all seven
    registered specialties. Both the specialty-choice guard and the
    catalogue-dump guard existed and neither fired, because the turn was
    no longer `medical`."""

    from langchain_core.messages import AIMessage as _AIMessage

    for msg in reversed(state.get("messages", []) or []):
        if not isinstance(msg, _AIMessage):
            continue
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        if not text.strip():
            continue
        return bool(_MEDICAL_OFFER_PATTERN_RE.search(_norm_ar(text)))

    return False


def _medical_reply_asks_which_specialty(reply_text: str, state: AgentState) -> bool:
    """True when a medical-guidance reply asks the patient to pick a
    specialty.

    They described a symptom - working out the specialty is the help
    they came for, and handing the question back is the one thing this
    flow must not do. It is especially wrong between near-identical
    entries ("طب الباطنة" vs "باطنه عام"), where the distinction is a
    registration detail on our side and means nothing to a patient in
    pain.

    CONFIRMED REAL PRODUCTION FAILURE: after a correct guidance reply
    the patient said "اه" - agreeing to be booked - and received "وش
    التخصص اللي تفضله من تخصصات الطب الباطنة المتوفرة؟ 1️⃣ طب الباطنة
    2️⃣ باطنه عام" instead of the doctor list.

    This complements `_reply_dumps_specialty_catalogue`, which catches
    the whole catalogue being printed; this catches the narrower "pick
    one of these two" question."""

    if not reply_text:
        return False

    return bool(_SPECIALTY_CHOICE_QUESTION_RE.search(_norm_ar(reply_text)))


_SPECIALTY_CHOICE_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "DON'T ASK WHICH SPECIALTY - SHOW THE DOCTORS\n"
    "============================================================\n"
    "Your previous draft asked the patient to choose a specialty. They "
    "described a symptom; working out the specialty is exactly the help "
    "they came for, so handing the question back leaves them stuck. "
    "Between near-identical entries like \"طب الباطنة\" and \"باطنه "
    "عام\" it is worse still - to them those are the same thing, and "
    "the difference is a registration detail on our side.\n\n"
    "Call `find_available_doctors` with EVERY plausibly matching "
    "specialty id at once - all of them, not one - and show the doctors "
    "that come back as a numbered list, then ask ONE question: which "
    "doctor. The specialty may appear as a label beside each name "
    "(\"رانيا عبد الرحمن — استشاري · باطنه عام\"), which is useful; it "
    "is never a choice they must make first.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: the patient said \"اه\" to being "
    "booked in and got \"وش التخصص اللي تفضله من تخصصات الطب الباطنة "
    "المتوفرة؟ 1️⃣ طب الباطنة 2️⃣ باطنه عام\" - a second triage question "
    "where the doctor list should have been.\n\n"
)


def _medical_reply_missing_not_a_diagnosis(reply_text: str, state: AgentState) -> bool:
    """True when a medical-guidance reply steers the patient toward a
    specialty or a doctor without saying this isn't a diagnosis.

    This is a compliance requirement, not a style preference: a
    booking assistant mapping symptoms to a specialty must not leave
    that reading as a clinical verdict. Prompt wording alone has
    already drifted on this once (it was softened to "leave it out if
    it doesn't fit"), so it is checked here.

    Deliberately narrow - it only fires when the reply is actually
    STEERING (offering doctors, offering to book, naming the fitting
    specialty). A follow-up question about the symptom, or a plain
    emergency instruction, needs no such clause."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)

    if not _SPECIALTY_OFFER_RE.search(folded):
        return False

    # ONLY WHILE STILL ROUTING - NOT ONCE A DOCTOR IS BEING CONFIRMED.
    #
    # The clause belongs to the guidance step, where a symptom is being
    # mapped to a specialty. Once the patient has picked a doctor from a
    # list, the reply confirming that choice is a booking step, and
    # attaching "this isn't a diagnosis" to it is noise.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: after the patient chose "1"
    # from a real doctor list, the reply "الدكتور المتاح عندنا حاليًا في
    # تخصص باطنة عام هو د. رانيا عبد الرحمن، دكتور استشاري - تحب أحجز
    # لك موعد عندها؟" was rejected for missing the clause. It then
    # retried in a loop that made roughly a hundred model calls over two
    # minutes and never produced a reply at all.
    session_id = state.get("session_id")
    if session_id:
        session = tools._BOOKING_SESSIONS.get(session_id) or {}
        if session.get("doctor_id"):
            return False
        last_list = session.get("last_list") or {}
        if last_list.get("entity_type") == "doctor" and last_list.get("items"):
            # A doctor list has been shown, so we are past routing and
            # into choosing. Naming a doctor here is a selection, not a
            # symptom-to-specialty suggestion.
            return False

    return not _NOT_A_DIAGNOSIS_RE.search(folded)


_NOT_A_DIAGNOSIS_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "ADD THE 'NOT A DIAGNOSIS' NOTICE - IT IS REQUIRED\n"
    "============================================================\n"
    "Your previous draft pointed the patient at a specialty or offered "
    "them a doctor without the required notice. It is mandatory on any "
    "medical-guidance reply that steers them: you are a booking "
    "assistant, not a clinician, and without it a symptom-to-specialty "
    "suggestion reads as a verdict on their condition.\n\n"
    "Add this line EXACTLY as written, on its own line, immediately "
    "before the line that offers the appointment:\n"
    "    \u2695\ufe0f تنبيه: هذه معلومات عامة وليست تشخيصًا طبيًا مباشرة.\n\n"
    "Keep the \u2695\ufe0f and the word \"تنبيه:\". This one line is "
    "deliberately a formal notice in Modern Standard Arabic, even though "
    "the rest of the message is in dialect.\n\n"
    "Then make sure the offer that follows is a COMPLETE, grammatical "
    "sentence of its own - not a fragment continuing from the notice. "
    "CONFIRMED REAL PRODUCTION FAILURE: bolting it on produced \"لتشخيص "
    "الطبي عندنا دكاترة باطنة متاحين\", which is not grammatical Arabic; "
    "it needs \"للتشخيص الطبي، عندنا في [اسم المستشفى] دكاترة باطنة "
    "متاحين — تحب أحجزلك عند واحد منهم؟\", or simply start it with "
    "\"عندنا في [اسم المستشفى]...\".\n\n"
    "Change nothing else about the reply.\n\n"
)


def _reply_asks_generic_branch_after_doctor(reply_text: str, state: AgentState) -> bool:
    """True when a doctor is settled and the reply asks a bare "which
    branch?" without having shown that doctor's own schedule.

    Once a doctor is chosen, the branches that matter are HERS. A
    generic branch list is a different question with a wrong answer
    attached: it offers branches she may not work at, and it hides the
    days and hours that would have let the patient just pick.

    CONFIRMED REAL PRODUCTION FAILURE: after "اخترت دكتورة رانيا عبد
    الرحمن ✅" the reply asked "أبشر، أي فرع تفضل تحجز فيه؟" and then
    listed "1️⃣ Al Nozha — 1 طبيب / 2️⃣ Al Manar — 1 طبيب" - the
    clinic's branches with headcounts, while that doctor works at only
    one of them.

    Suppressed once `get_doctor_schedule_for_booking` has actually run,
    since its own output legitimately names branches and asks which
    one."""

    if not reply_text:
        return False

    session_id = state.get("session_id")
    if not session_id:
        return False

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    if not session.get("doctor_id"):
        return False

    if session.get("branch_id"):
        # Branch already settled - nothing to ask, and any branch
        # wording here is incidental.
        return False

    for msg in state.get("messages", []) or []:
        if getattr(msg, "name", None) == "get_doctor_schedule_for_booking":
            return False

    return bool(_GENERIC_BRANCH_QUESTION_RE.search(_norm_ar(reply_text)))


_DOCTOR_SCHEDULE_INSTEAD_OF_BRANCHES_CORRECTION = (
    "============================================================\n"
    "SHOW THIS DOCTOR'S SCHEDULE - NOT A LIST OF CLINIC BRANCHES\n"
    "============================================================\n"
    "Your previous draft asked which branch the patient wants, and/or "
    "listed the clinic's branches. A doctor is already chosen, so the "
    "only branches that exist for this booking are HERS - a general "
    "branch list offers places she may not work at all.\n\n"
    "Call `get_doctor_schedule_for_booking` and show what it returns, "
    "grouped by branch, with the real days and hours at each:\n"
    "    مواعيد [الدكتور] في فرع [الفرع الأول]:\n"
    "    • [اليوم]: من [من] لـ [إلى] — [اسم الخدمة]\n"
    "    وفي فرع [الفرع الثاني]:\n"
    "    • [اليوم]: من [من] لـ [إلى] — [اسم الخدمة]\n"
    "    حابب تحجز في أنهي فرع وأنهي يوم؟\n\n"
    "If she works at only ONE branch, there is nothing to ask: show "
    "that branch's days and ask about the DAY directly (\"تحب أشوف لك "
    "المواعيد المتاحة ليوم [اليوم]؟\").\n\n"
    "Keep the confirmation line as it was and replace everything after "
    "it. Never print branch headcounts (\"— 1 طبيب\") in place of the "
    "doctor's actual schedule.\n\n"
)


def _reply_offers_booking_at_empty_branch(reply_text: str, state: AgentState) -> bool:
    """True when a reply offers to book at the branch the tools have
    already established has no bookable doctor.

    Prompt rules for this have been written three times and have leaked
    each time, and the cost falls entirely on the patient: they say yes
    to an appointment that cannot exist and only discover it several
    turns later.

    CONFIRMED REAL PRODUCTION FAILURE: at فرع الطوارئ (zero doctors) the
    sequence ran address -> "تحب تعرف عن الخدمات؟" -> services -> "هل
    ترغب تحجز موعد في هذا الفرع؟" -> "نكمل الحجز على نفس رقم الواتساب
    ده؟" -> "من فضلك اكتب اسم الدكتور اللي حابب تحجز معاه في فرع
    الطوارئ؟" - four turns walking someone into a booking at a branch
    with nobody in it."""

    if not reply_text:
        return False

    session_id = state.get("session_id")
    if not session_id:
        return False

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    empty_branch = session.get("info_branch_no_doctors")
    if not empty_branch:
        return False

    # THE REPLY MUST ACTUALLY BE ABOUT THAT BRANCH.
    #
    # The note survives on the session after the patient moves on, so
    # without this the guard fires on any later reply that happens to
    # offer a booking - including ones with no branch in play at all.
    #
    # CONFIRMED REAL FALSE POSITIVE: the patient browsed فرع الطوارئ
    # (empty), said "لا مش عاوزه", then described eye pain. The medical
    # reply offered an appointment - nothing to do with any branch - and
    # this guard blocked it twice because the stale note was still
    # there, burning two LLM calls on a reply that was never wrong.
    branch_mentioned = _norm_ar(empty_branch) in _norm_ar(reply_text)
    if not branch_mentioned:
        return False

    # Only while that empty branch is still the one in play - once a
    # different branch is confirmed, offering a booking is correct.
    current = session.get("branch_display_name") or session.get("info_branch_name")
    if current and _norm_ar(current) != _norm_ar(empty_branch):
        return False

    return bool(_BOOKING_OFFER_RE.search(_norm_ar(reply_text)))


_EMPTY_BRANCH_BOOKING_OFFER_CORRECTION = (
    "============================================================\n"
    "YOU OFFERED A BOOKING AT A BRANCH WITH NO DOCTORS - REWRITE\n"
    "============================================================\n"
    "Your previous draft offered to book an appointment (or asked for a "
    "doctor's name, or asked to continue a booking) at a branch the "
    "tools have already established has NO bookable doctor. If they say "
    "yes, there is nothing to give them - the offer cannot be kept.\n\n"
    "Rewrite it. Keep the useful parts - the address, the services - "
    "and replace the booking offer with an honest, useful question "
    "instead:\n"
    "    الفرع ده مفيهوش حجز حاليًا، تحب أعرض لك الفروع اللي فيها حجز؟\n\n"
    "Phrase it in this clinic's own dialect. If they then say yes, call "
    "`find_branches_offering_service` (when a service is in play) or "
    "`list_branches_for_specialty`, and list those branches by name.\n\n"
    "Do not ask for a doctor's name, do not ask to confirm a phone "
    "number for the booking, and do not start the booking flow for this "
    "branch in any other form.\n\n"
)


def _reply_dumps_specialty_catalogue(reply_text: str, state: AgentState) -> bool:
    """True when a MEDICAL-GUIDANCE reply prints the specialty catalogue
    as a numbered list for the patient to pick from.

    `list_specialties` is a catalogue for the ASSISTANT to match
    against, not an answer for the patient. Someone who has just
    described a symptom cannot be asked to work out which specialty it
    belongs to - that needs the medical knowledge they came here
    without, and it is the one thing they asked for help with.

    CONFIRMED REAL PRODUCTION FAILURE: "بطني وجعاني وعندي ترجيع" was
    answered with the right specialty named in prose AND then all seven
    registered specialties printed underneath ("1️⃣ طب الأطفال 2️⃣
    جراحة العظام 3️⃣ أمراض القلب..."), including several with no
    possible bearing on stomach pain. The prompt rule against this
    already existed and did not hold, so it is enforced here too.

    Deliberately scoped to replies that BOTH print three or more
    numbered items AND name specialties the tools returned - a doctor
    list or a branch list is numbered too, and those are legitimate."""

    if not reply_text:
        return False

    if len(_NUMBERED_LIST_ITEM_RE.findall(reply_text)) < 3:
        return False

    specialty_names = set()
    for msg in state.get("messages", []) or []:
        if getattr(msg, "name", None) != "list_specialties":
            continue
        try:
            data = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        for item in data.get("specialties") or data.get("items") or []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("altName")
                if name:
                    specialty_names.add(_norm_ar(str(name)))

    if not specialty_names:
        return False

    folded = _norm_ar(reply_text)
    matched = sum(1 for name in specialty_names if name and name in folded)

    return matched >= 3


_SPECIALTY_CATALOGUE_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU PRINTED THE SPECIALTY CATALOGUE - REMOVE IT\n"
    "============================================================\n"
    "Your previous draft listed the clinic's specialties for the patient "
    "to choose from. That list is for YOU to match against, not for "
    "them. They described a symptom - working out which specialty it "
    "belongs to is exactly the help they came for, and several entries "
    "in that list have no bearing on what they told you.\n\n"
    "Rewrite it WITHOUT the list. Name only the ONE fitting specialty in "
    "ordinary prose, say plainly that this clinic has doctors in it, and "
    "ask ONE question - whether they'd like an appointment. Keep the "
    "warmth, the comfort advice, and the 'this isn't a diagnosis' note "
    "exactly as they were.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: \"بطني وجعاني وعندي ترجيع\" was "
    "answered with the correct specialty in prose and then all seven "
    "registered specialties printed underneath, asking the patient to "
    "pick - including طب الأطفال and جراحة العظام.\n\n"
)


def _reply_recommends_medication(reply_text: str, state: AgentState) -> bool:
    """True when a reply names or suggests a medicine.

    Prompt rules alone have not held here, and the downside is real
    harm rather than an awkward turn, so this is enforced in code as
    well. Deliberately matches the drug NAME or class wherever it
    appears - there is no context in this product where a booking
    assistant should be putting a medication name in front of a
    patient."""

    if not reply_text:
        return False

    return bool(_MEDICATION_MENTION_RE.search(_norm_ar(reply_text)))


_DOCTOR_ESTABLISHING_TOOLS = (
    "find_available_doctors", "match_entity_for_booking",
    # RESCHEDULE establishes its current doctor differently than NEW
    # BOOKING does - by looking up the existing appointment's own
    # doctor, not by searching/matching a name. Without this, a fresh
    # `lookup_appointment` for a second, different booking later in the
    # same session does not reset the scope, and a stale availability
    # lookup for the FIRST doctor keeps satisfying this check for the
    # second one - the exact class of bug this scoping was added to
    # close, just reachable through reschedule's own doctor-resolution
    # path instead of booking's.
    "lookup_appointment",
)


def _reply_denies_availability_without_lookup(reply_text: str, state: AgentState) -> bool:
    """True when the reply tells the patient a doctor has no available
    appointments, while NO availability tool has run for the CURRENTLY
    discussed doctor.

    CONFIRMED REAL PRODUCTION FAILURE: the patient picked د. هشام عوض
    from position 1, `match_entity_for_booking` confirmed him - and the
    very next line was "لكن ما ظهر لي مواعيد متاحة حاليا عند د. هشام
    عوض", followed by an offer to look at other doctors. No days tool
    and no slots tool were called anywhere in that turn (both log
    unconditionally, and neither appears in the trace). The doctor was
    written off as unavailable on nothing at all, and the patient was
    steered away from a booking that may well have been possible.

    `_reply_invents_availability` is the mirror image of this - it
    catches dates/times asserted with no tool behind them. Between them
    both directions are covered: you may not invent availability, and
    you may not invent its absence.

    SCOPED TO THE CURRENT DOCTOR, NOT THE WHOLE CONVERSATION: an earlier
    version accepted ANY availability-lookup tool call anywhere in the
    conversation as satisfying this check - so once a real lookup had
    happened for one doctor, a denial for a COMPLETELY DIFFERENT doctor
    much later in the same session (a new search, a fresh
    `find_available_doctors` result) went unflagged, because the bar
    had already been satisfied by unrelated history. CONFIRMED REAL
    PRODUCTION FAILURE (medtown, 2026-08-30): `find_available_doctors`
    found "بدر تميمي" again for a fresh pediatrics search, the patient
    agreed to book ("اه"), and the reply was "ما عندي مواعيد لدكتور بدر
    تميمي الحين" with no availability tool called anywhere near that
    turn - while an unrelated availability lookup from earlier in the
    same long-running session (for whatever was being discussed back
    then) was still sitting in the message history, silently satisfying
    the old, unscoped check. Only lookups AFTER the doctor now being
    discussed was most recently (re-)established count."""

    if not reply_text:
        return False

    if not _AVAILABILITY_DENIAL_RE.search(_norm_ar(reply_text)):
        return False

    messages = state.get("messages", []) or []

    last_establish_idx = None
    for i, msg in enumerate(messages):
        if getattr(msg, "name", None) in _DOCTOR_ESTABLISHING_TOOLS:
            last_establish_idx = i

    start_idx = last_establish_idx + 1 if last_establish_idx is not None else 0

    for msg in messages[start_idx:]:
        if getattr(msg, "name", None) in _AVAILABILITY_LOOKUP_TOOLS:
            return False

    return True


_AVAILABILITY_DENIAL_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU SAID THERE ARE NO APPOINTMENTS - BUT YOU NEVER CHECKED\n"
    "============================================================\n"
    "Your previous draft told the patient a doctor has no available "
    "appointments. No availability tool has run in this conversation, "
    "so that claim has nothing behind it - you decided it yourself.\n\n"
    "A doctor being unavailable is a FACT, and it only ever comes from "
    "`list_available_days_for_booking` (or `resolve_available_day` / "
    "`get_available_slots_for_booking`). Call "
    "`list_available_days_for_booking` NOW and answer from what it "
    "actually returns:\n"
    "  - days come back -> show the soonest date and ask if it suits "
    "them.\n"
    "  - \"not_found\" -> only THEN may you say this doctor has nothing "
    "open, and only then offer another doctor.\n"
    "  - \"missing_branch\" -> settle the branch first.\n\n"
    "Do not offer other doctors or other branches in place of checking. "
    "CONFIRMED REAL PRODUCTION FAILURE: a doctor was confirmed and the "
    "next line was \"ما ظهر لي مواعيد متاحة حاليا عند د. [اسم]\" with no "
    "tool call at all - the patient was pushed off a booking that was "
    "never actually checked.\n\n"
)


def _reply_invents_availability(reply_text, state) -> bool:
    """True when the reply states an appointment date or times that no
    availability tool in this conversation ever returned.

    WHY: confirmed real production failure, the worst yet - after a
    branch was picked the reply offered "يوم الثلاثاء 30-05-2024" and a
    list of times, with NO availability tool called at all. The date was
    in the PAST and every time was invented; a patient could have
    accepted an appointment that existed nowhere. The date/time
    directives elsewhere only fire AFTER a tool has run, so when the
    model skips the tool entirely nothing else can catch it."""

    if not reply_text:
        return False

    dates = _DATE_IN_REPLY_RE.findall(reply_text)
    times = _TIME_IN_REPLY_RE.findall(reply_text)

    # Weekday names count too. Confirmed real production failure: the
    # reply offered "1️⃣ الخميس 2️⃣ السبت 3️⃣ الاثنين" as bookable days
    # with no availability tool called at all - it carried no digits, so
    # a date/time-only check saw nothing wrong while the patient was
    # being offered three days the doctor may not work at all.
    weekdays = [d for d, pattern in _WEEKDAY_WORD_RES.items() if pattern.search(reply_text)]

    if not dates and not times and not weekdays:
        return False

    tool_text = []
    for msg in state.get("messages", []):
        if getattr(msg, "name", None) in _AVAILABILITY_TOOLS:
            content = getattr(msg, "content", "")
            if content:
                tool_text.append(str(content))

    if not tool_text:
        return True

    joined = " ".join(tool_text)

    for value in dates:
        parts = [p.lstrip("0") for p in re.split(r"[-/]", value)]
        if not any(p and p in joined for p in parts):
            return True

    for value in times:
        hour = value.split(":")[0].lstrip("0")
        if hour and hour not in joined:
            return True

    for day in weekdays:
        # The tools return weekday names in both the conversation's
        # language and English, so accept either spelling.
        english = _WEEKDAY_WORDS[day]
        if day not in joined and english.lower() not in joined.lower():
            return True

    return False


_AVAILABILITY_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU STATED A DATE/TIME NO TOOL GAVE YOU - REWRITE YOUR REPLY\n"
    "============================================================\n"
    "Your previous draft named an appointment date or times that no "
    "availability tool returned in this conversation. Those are "
    "invented - a patient could accept an appointment that does not "
    "exist anywhere in the booking system.\n\n"
    "NEVER state a date or time from memory, from reasoning, or from a "
    "doctor's general working hours. Call "
    "`list_available_days_for_booking` for the real days, then "
    "`get_available_slots_for_booking` with that day's own from_date/"
    "to_date for the real times, and use ONLY what they return.\n\n"
    "Call the tool now instead of writing a date yourself.\n\n"
)


# ==========================================================
# Fabricated "no doctor/branch by that name" stop verifier
# ==========================================================
#
# WHY THIS EXISTS: STEP C1's PRIORITY check tells the model to extract
# a doctor/branch name from the patient's very first complaint message
# and verify it via `match_entity_info` before anything else. That
# instruction only applies when a name was ACTUALLY given - but the
# model can misfire it on a message that names nobody at all, invent
# some fragment of the message as if it were a name, get a real
# "not_matched" back for that invented fragment, and then correctly
# (from its own point of view) say the fixed "we couldn't find a
# doctor by that name" apology and stop the complaint entirely.
#
# CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): the patient's
# entire message was "عاوزه اشتكي علشان الدواء اتوصفلي غلط" ("I want to
# complain because the medication I was prescribed was wrong") - no
# doctor name, no word for "doctor" even appears in it - and the reply
# was the fixed STOP apology: "نعتذر، ما لقينا دكتور بهذا الاسم في
# مستشفى ميدتاون الطبية، لذلك ما نقدر نكمل تسجيل الشكوى..." The patient
# had said nothing that could be mistaken for a name; the flow was
# ended over a lookup the situation never called for.
#
# This is deliberately narrow: it only fires when the STOP apology is
# used and NEITHER of the two legitimate ways to reach it are present -
# (a) the assistant had just asked STEP C2b's own "تحت أي دكتور
# بالظبط؟"/"في أنهي فرع بالظبط؟" question, so any answer plausibly
# names one even without repeating the word itself, or (b) the
# patient's own message contains a recognizable doctor/branch cue word
# at all. Both are treated as "there was something to check" and left
# alone; only the case with neither is flagged.

_COMPLAINT_STOP_APOLOGY_RE = re.compile(
    r"ما\s*لقي(?:ت|نا)?\s*(دكتور|دكتوره|فرعا?)\s*بهذا\s*الاسم"
)

_ASKED_WHICH_DOCTOR_OR_BRANCH_RE = re.compile(
    r"تحت\s*أي\s*دكتور\s*بالظبط|في\s*أنهي\s*فرع\s*بالظبط|"
    r"which\s*doctor\s*exactly|which\s*branch\s*exactly"
)

_DOCTOR_CUE_WORD_RE = re.compile(r"دكتور|دكتوره|د\.|طبيب|doctor")
_BRANCH_CUE_WORD_RE = re.compile(r"فرع|branch")

# A generic word for "the doctor"/"a doctor" (or "the branch") ON ITS
# OWN, with nothing else - not an actual name. Matched against the
# FOLDED form of the exact `user_input` argument passed to
# `match_entity_info`, so this is the strongest possible signal that no
# real name was ever given: the model called the lookup tool with
# literally the generic noun itself rather than any name at all.
_GENERIC_DOCTOR_WORD_ONLY_RE = re.compile(
    r"^\s*(?:ال)?دكتور(?:ه)?\s*$|^\s*(?:ال)?طبيب(?:ه)?\s*$|^\s*د\s*$"
)
_GENERIC_BRANCH_WORD_ONLY_RE = re.compile(r"^\s*(?:ال)?فرع\s*$")


def _last_match_entity_info_user_input(state: AgentState, entity_type: str) -> Optional[str]:
    """The `user_input` most recently passed to `match_entity_info` for
    the given `entity_type` ("doctor"/"branch"), or None if it hasn't
    been called with that entity_type anywhere in this conversation."""

    for m in reversed(state.get("messages") or []):
        for call in (getattr(m, "tool_calls", None) or []):
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name != "match_entity_info":
                continue
            args = (call.get("args") if isinstance(call, dict) else getattr(call, "args", None)) or {}
            if (str(args.get("entity_type") or "")).strip().lower() != entity_type:
                continue
            return str(args.get("user_input") or "")

    return None


def _reply_fabricates_doctor_not_found_stop(reply_text: str, state: AgentState) -> bool:
    folded_reply = _norm_ar(reply_text or "")
    match = _COMPLAINT_STOP_APOLOGY_RE.search(folded_reply)
    if not match:
        return False

    is_branch = match.group(1).startswith("فرع")
    entity_type = "branch" if is_branch else "doctor"

    # STRONGEST SIGNAL FIRST: what was actually passed to
    # `match_entity_info`. If it was literally just the bare generic
    # word ("دكتور"/"الدكتور"/"طبيب"/"فرع") with nothing else, that is
    # never a real name no matter what the surrounding message looked
    # like - flag it regardless of the heuristics below.
    #
    # CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): the
    # patient said "دكتور كتبلي دواء غلط مش لحالتي" (a/the doctor
    # prescribed me the wrong medication) - genuinely mentions "دكتور"
    # as a common noun, with no name attached - and the call was made
    # as `match_entity_info(user_input="دكتور", entity_type="doctor")`.
    # The message-level heuristic below alone would have missed this
    # (the word "دكتور" DOES appear in the patient's message), which is
    # exactly why this stronger, argument-level check exists.
    generic_re = _GENERIC_BRANCH_WORD_ONLY_RE if is_branch else _GENERIC_DOCTOR_WORD_ONLY_RE
    last_call_input = _last_match_entity_info_user_input(state, entity_type)
    if last_call_input is not None and generic_re.match(_norm_ar(last_call_input)):
        return True

    cue_re = _BRANCH_CUE_WORD_RE if is_branch else _DOCTOR_CUE_WORD_RE

    messages = state.get("messages") or []

    last_human_text = ""
    prior_ai_text = ""
    seen_human = False
    for m in reversed(messages):
        mtype = getattr(m, "type", None)
        if mtype == "human" and not seen_human:
            last_human_text = str(getattr(m, "content", "") or "")
            seen_human = True
            continue
        if mtype == "ai" and seen_human:
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                prior_ai_text = content
                break

    # Legitimate case (a): the assistant had just explicitly asked which
    # doctor/branch, so this answer plausibly names one.
    if _ASKED_WHICH_DOCTOR_OR_BRANCH_RE.search(prior_ai_text):
        return False

    # Legitimate case (b): the patient's own message contains a
    # recognizable doctor/branch cue word somewhere.
    if cue_re.search(last_human_text):
        return False

    return True


_DOCTOR_NOT_FOUND_STOP_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU STOPPED THE COMPLAINT OVER A NAME THAT WAS NEVER GIVEN - REWRITE\n"
    "============================================================\n"
    "Your previous draft said no doctor/branch by that name could be "
    "found and stopped the complaint - but the patient's own message "
    "does not actually name a doctor or a branch, and you had not just "
    "asked them which one they meant. This means a name was invented "
    "(from the message itself, or from elsewhere) and checked against "
    "`match_entity_info` when there was nothing real to check.\n\n"
    "This complaint is NOT about a specific doctor or branch unless the "
    "patient actually said one. Continue the COMPLAINT FLOW normally: "
    "if their message already describes what went wrong, thank them and "
    "ask if there's anything else to add (STEP C1b) - do not ask "
    "'which doctor?' or 'which branch?' unless they raise one "
    "themselves. Do not call `match_entity_info` again unless they "
    "actually name someone.\n\n"
    "Rewrite the reply now, continuing the complaint flow instead of "
    "stopping it.\n\n"
)


# ==========================================================
# Complaint derailed into a handoff offer instead of STEP C3 verifier
# ==========================================================
#
# WHY THIS EXISTS: STEP C1b says that once the patient answers "لا"/
# "that's it" to "حابب تضيف أي تفاصيل تانية قبل ما نكمل؟", the flow
# moves on to STEP C2/C3 (determine the subject, then ask the
# patient's name) - collecting toward STEP C7's actual send. Nothing in
# the flow says to offer a human-handoff detour at that exact point,
# and STEP C8 only directs the PATIENT to ask for "موظف" themselves if
# THEY want that - it is not something to proactively offer mid-flow.
#
# CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): after "لا"
# to the STEP C1b question, the reply asked "هل تحبني أساعدك في
# التواصل مع أحد ممثلي خدمة العملاء مباشرةً؟" instead of asking for the
# patient's name - and when the patient said "لا" to THAT too, the
# conversation was simply closed with "شكرًا لك... هل تحتاج شيء ثاني
# الآن؟". The complaint was never sent, the patient was never told it
# wasn't sent, and STEP C3-C7 (name, phone, branch, confirm, send) were
# never reached at all - the complaint was silently dropped while the
# patient believed it had been handled ("شكرًا للتوضيح" had already
# been said earlier).

_MORE_DETAILS_QUESTION_RE = re.compile(
    r"حابب\s*تضيف\s*اي\s*تفاصيل\s*تانيه|اي\s*تفاصيل\s*تانيه\s*قبل\s*ما\s*نكمل|"
    r"anything\s*else\s*(?:you.?d\s*like\s*to\s*)?add"
)

_PLAIN_NEGATIVE_RE = re.compile(
    r"^\s*(?:لا+|لأ+|مفيش|ولا\s*حاجه|no+|nope|nothing)\b", re.IGNORECASE
)

_OFFERS_CUSTOMER_SERVICE_HANDOFF_RE = re.compile(
    r"(?:أساعدك|تحب|تحبني)[^.\n؟?]{0,25}(?:التواصل|أوصلك|أحولك)[^.\n؟?]{0,25}"
    r"(?:ممثل|موظف|خدمه\s*العملاء)|"
    r"customer\s*service\s*representative"
)


def _reply_derails_complaint_into_handoff_offer(reply_text: str, state: AgentState) -> bool:
    if not reply_text or not _OFFERS_CUSTOMER_SERVICE_HANDOFF_RE.search(_norm_ar(reply_text)):
        return False

    messages = state.get("messages") or []

    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            last_human_idx = i
            break

    if last_human_idx is None:
        return False

    last_human_text = str(getattr(messages[last_human_idx], "content", "") or "").strip()
    folded_last_human = _norm_ar(last_human_text)
    if not _PLAIN_NEGATIVE_RE.match(folded_last_human):
        return False

    # If the patient's own message ALSO separately, explicitly asks for
    # a person ("عايز اتكلم مع موظف حقيقي"), the offer isn't a
    # derailment - it's a normal response to a genuine request. Only
    # the bare-decline case (nothing more than "لا"/"nope") is the
    # confirmed failure pattern this guard exists for.
    if any(root in folded_last_human for root in tools._EXPLICIT_HUMAN_REQUEST_ROOTS):
        return False

    prior_ai_text = ""
    for i in range(last_human_idx - 1, -1, -1):
        m = messages[i]
        if getattr(m, "type", None) == "ai":
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                prior_ai_text = content
                break

    return bool(_MORE_DETAILS_QUESTION_RE.search(_norm_ar(prior_ai_text)))


_COMPLAINT_HANDOFF_DERAIL_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "DO NOT OFFER A HANDOFF HERE - ASK FOR THE PATIENT'S NAME (STEP C3)\n"
    "============================================================\n"
    "Your previous draft offered to connect the patient with a customer "
    "service representative, right after they said there was nothing "
    "more to add to their complaint. Offering a handoff at this exact "
    "point is not part of the COMPLAINT FLOW and derails a complaint "
    "that was proceeding normally - the patient did not ask for a "
    "human, and there has been no technical failure.\n\n"
    "Continue the flow instead: decide the complaint's subject (STEP "
    "C2), then ask ONLY for the patient's name if you don't already "
    "have it for this complaint (STEP C3). Do not mention a staff "
    "handoff at all unless the patient explicitly asks for one "
    "themselves.\n\n"
    "Rewrite the reply now, asking for the patient's name instead of "
    "offering a handoff.\n\n"
)


# ==========================================================
# Fabricated "your complaint was filed" verifier
# ==========================================================
#
# WHY THIS EXISTS: confirmed real production failure, arguably the
# worst kind - a patient filing a complaint about a doctor allegedly
# PRESCRIBING THE WRONG MEDICATION was told "حجزت لك الشكوى بخصوص دكتور
# عبدالله محمد" (your complaint has been registered) while the WHOLE
# exchange ran under the "booking" specialist, which does not have the
# `send_complaint_email` tool AT ALL - nothing was ever sent anywhere.
# The routing itself was the deeper bug (a complaint conversation never
# switched specialists and "booking" improvised the entire flow from
# its own general "answer whatever you safely can" instructions), but
# THIS check is the last line of defence regardless of which
# specialist is active or why: a reply that confirms a complaint was
# filed/registered/sent, with no successful `send_complaint_email` call
# anywhere in this conversation, is always false and must be blocked -
# a patient believing a genuine complaint about their care was
# delivered when it never was is a serious trust and safety failure,
# not a cosmetic one.

_COMPLAINT_CONFIRMATION_RE = re.compile(
    r"(?:حجزت|سجلت|تم\s*تسجيل|تم\s*استلام|تم\s*إرسال|تم\s*ارسال|استلمنا|"
    r"وصلتنا)\s*\w*\s*(?:شكوا?ي?ت?ك|الشكوى|شكواك)"
)


def _reply_fabricates_complaint_submission(reply_text: str, state: AgentState) -> bool:
    if not reply_text or not _COMPLAINT_CONFIRMATION_RE.search(reply_text):
        return False

    for msg in state.get("messages", []):
        if getattr(msg, "name", None) != "send_complaint_email":
            continue
        content = str(getattr(msg, "content", "") or "")
        if '"status": "sent"' in content or "'status': 'sent'" in content:
            return False  # a real send actually happened - not fabricated

    return True


_COMPLAINT_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU CONFIRMED A COMPLAINT WAS FILED WITHOUT SENDING IT - REWRITE\n"
    "============================================================\n"
    "Your previous draft told the patient their complaint was "
    "registered/sent/filed, but `send_complaint_email` was never called "
    "successfully in this conversation - nothing was actually sent "
    "anywhere. Telling a patient a complaint about their care was "
    "delivered when it wasn't is a serious trust failure, especially "
    "when the complaint concerns something like a wrong prescription.\n\n"
    "If this is genuinely a complaint (not a booking/medical/FAQ "
    "request), follow the COMPLAINT FLOW: verify any doctor/branch name "
    "via `match_entity_info` first, collect the remaining details, get "
    "explicit confirmation, and only THEN call `send_complaint_email` "
    "before claiming it was filed. If you don't have that tool "
    "available to you right now, say honestly that you'll connect them "
    "with a staff member for this instead of claiming success.\n\n"
    "Rewrite the reply now without claiming the complaint was filed.\n\n"
)


# ==========================================================
# Fabricated "you've been transferred to a human" verifier
# ==========================================================
#
# WHY THIS EXISTS: same class of failure as the fabricated complaint
# confirmation above, applied to human handoff. Confirmed real
# production failure: a patient said "لا هشتكي الدكتور دا" ("no, I'll
# complain about this doctor") - not a request for a human agent at all,
# and not agreement to any handoff that had been offered - and was told
# "تم تحويلك إلى أحد ممثلي خدمة العملاء" (you've been transferred to a
# customer service rep) with `request_human_handoff` never having been
# called anywhere in the conversation. The patient was NOT transferred;
# they were left believing a human was now handling something that
# nobody was ever notified about. `request_human_handoff` itself
# requires `patient_agreed=True` and returns no patient-facing text on
# purpose specifically so the model still has to write - and get right -
# the confirmation line itself; this check catches it when that line is
# written without the tool call that's supposed to back it up.

_HANDOFF_CONFIRMATION_RE = re.compile(
    r"تم\s*تحويلك|تم\s*التحويل|جاري\s*تحويلك|حولتك|حولناك|"
    r"سيتم\s*تحويلك|تم\s*تحويل\s*محادثتك"
)


def _reply_fabricates_handoff(reply_text: str, state: AgentState) -> bool:
    if not reply_text or not _HANDOFF_CONFIRMATION_RE.search(reply_text):
        return False

    for msg in state.get("messages", []):
        if getattr(msg, "name", None) != "request_human_handoff":
            continue
        content = str(getattr(msg, "content", "") or "")
        if '"status": "handoff_requested"' in content or "'status': 'handoff_requested'" in content:
            return False  # a real handoff was actually raised - not fabricated

    return True


_HANDOFF_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU CONFIRMED A HANDOFF WITHOUT REQUESTING ONE - REWRITE\n"
    "============================================================\n"
    "Your previous draft told the patient they've been transferred to a "
    "human staff member, but `request_human_handoff` was never called "
    "successfully in this conversation (or `patient_agreed` was not "
    "True) - nobody was actually notified. Telling a patient they've "
    "been handed off when they haven't is a serious trust failure: they "
    "may now be waiting indefinitely for a human who was never alerted.\n\n"
    "A handoff requires the patient's own explicit agreement - either "
    "they asked for a human themselves, or you offered one earlier and "
    "they just said yes to THAT offer. If that hasn't genuinely happened "
    "yet, do not call the tool or claim a transfer - ask them plainly "
    "whether they'd like to be connected to a staff member instead, and "
    "only call `request_human_handoff` with `patient_agreed=True` once "
    "they confirm.\n\n"
    "Rewrite the reply now without claiming a handoff happened.\n\n"
)


# ==========================================================
# "Doctor confirmed, then re-offered a doctor roster anyway" verifier
# ==========================================================
#
# WHY THIS EXISTS: this exact anti-pattern is already documented in the
# prompt with TWO confirmed prior examples ("دكتور شيماء جمعة تم
# اختياره ✅" and "أبشر بحجز موعد عند د. سارة عبد الله", both followed
# in the SAME message by an offer to list available doctors) - and it
# still happened a THIRD time: "استشاري محمود سليمان تم اختياره ✅ ...
# تحب تحجزين في فرع معيّن، ولا أعرض لك الدكاترة المتاحين؟". Repeating
# the same prompt warning a third time is unlikely to hold any better
# than the first two did, so this is promoted to a deterministic check,
# the same escalation already applied to invented branches and
# fabricated availability.

# MATCHED AGAINST A FOLDED COPY OF THE REPLY (see _norm_ar), not the raw
# text. CONFIRMED REAL MISS: the reply "اخصائى محمد زايد تم اختياره ✅ ...
# ولا أعرض لك كل الدكاترة المتاحين؟" - the exact anti-pattern this guard
# exists for, on its FOURTH occurrence - sailed straight through,
# because the model wrote "اخصائى" with alef maqsura while the pattern
# spelled it "اخصائي" with ya. One letter, and a guard that had already
# been escalated from a prompt rule to code did nothing.
#
# Folding both sides removes that entire class of miss: alef variants,
# ya/alef-maqsura, ta-marbuta/ha. Patterns here are written in their
# FOLDED form (ي not ى, ه not ة) so they match what _norm_ar produces.
_DOCTOR_CONFIRMED_RE = re.compile(
    r"(?:د\.|دكتور|دكتوره|استشاري|استشاريه|اخصائي|اخصائيه)"
    r"[^.\n؟?]{0,40}?(?:تم\s*اختياره|تم\s*اختيارها|✅)"
)

_DOCTOR_ROSTER_OFFER_RE = re.compile(
    r"(?:ال)?دكاتره\s*(?:ال)?متاحين|قائمه\s*(?:ال)?دكاتره|"
    r"(?:ا|أ)عرض\s*لك\s*(?:كل\s*)?(?:ال)?دكاتره|"
    r"(?:ال)?اطباء\s*(?:ال)?متاحين"
)


# The branch question - "a particular branch, or shall I show you...?".
# Asking this AT ALL means a doctor is already settled: there is no
# other reason to be choosing a branch in the booking flow. So a reply
# that asks it and offers the doctor roster in the same breath is
# self-contradictory, whether or not it happens to restate the
# confirmation line.
#
# CONFIRMED REAL MISS (the FIFTH occurrence of this anti-pattern): the
# reply was the bare question with no confirmation line above it -
# "تحب تحجزين في فرع معيّن، ولا أعرض لك كل الدكاترة المتاحين؟" - because
# the doctor had been agreed in the PREVIOUS turn. Requiring the
# confirmation text to be present in the same message meant the guard
# only ever caught the version where the model restated it.
_BRANCH_QUESTION_OFFER_RE = re.compile(
    r"(?:في\s*)?فرع\s*(?:معين|معينه|محدد|محدده)|"
    r"(?:اي|انهي|انهو)\s*فرع|"
    r"(?:a\s+)?(?:particular|specific|certain)\s+branch|which\s+branch"
)


def _reply_reoffers_doctor_roster_after_confirming_one(reply_text: str, state: AgentState = None) -> bool:
    """True when a reply both settles a doctor and re-offers the doctor
    roster in the same breath.

    GATED ON A DOCTOR ACTUALLY BEING SETTLED. This guard belongs to the
    BOOKING flow, where a doctor is already chosen and the roster must
    not be re-offered. It used to look at the reply text alone, which
    made it fire on replies that have nothing to do with a booking.

    CONFIRMED REAL FALSE POSITIVE: asked simply "ايه فروع المستشفي", the
    reply listed the branches and mentioned in passing that some of them
    "ما فيها دكاترة متاحين", ending with "تحب تعرف معلومات أكثر عن أي فرع
    منهم؟". The roster pattern matched "دكاتره متاحين" and the branch
    pattern matched "أي فرع" - two phrases that happen to co-occur in a
    perfectly ordinary branch listing, with no doctor confirmed anywhere
    in the conversation. The forced "correction" then rewrote a correct
    reply into a WORSE one that silently dropped three real branches
    from the list the patient had asked for.
    """

    if not reply_text:
        return False

    if state is not None:
        session = tools._BOOKING_SESSIONS.get(state.get("session_id")) or {}
        if not session.get("doctor_id"):
            return False

    folded = _norm_ar(reply_text)

    if not _DOCTOR_ROSTER_OFFER_RE.search(folded):
        return False

    return bool(
        _DOCTOR_CONFIRMED_RE.search(folded)
        or _BRANCH_QUESTION_OFFER_RE.search(folded)
    )


_DOCTOR_ROSTER_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU CONFIRMED A DOCTOR THEN OFFERED THE DOCTOR LIST AGAIN - REWRITE\n"
    "============================================================\n"
    "Your previous draft named a doctor as chosen (\"... تم اختياره ✅\") "
    "and then, in that SAME message, offered to show the available "
    "doctors again (\"الدكاترة المتاحين\"). The doctor question is "
    "already settled - re-offering the roster invites the patient to "
    "undo a choice they just made.\n\n"
    "Do NOT replace it with the branch question either. \"تحب تحجز في "
    "فرع معيّن، ولا أعرض لك الفروع اللي د. [اسم] متاح فيها؟\" is also "
    "forbidden in this flow - it spends a turn asking for something the "
    "tools can simply show.\n\n"
    "Instead, call `get_doctor_schedule_for_booking` and DISPLAY that "
    "doctor's real schedule grouped by branch, then ask ONE combined "
    "question:\n"
    "    مواعيد الدكتور [اسم الدكتور] في فرع [الفرع الأول]:\n"
    "    • [اليوم]: من [من] لـ [إلى] — [اسم الخدمة]\n"
    "    وفي فرع [الفرع الثاني]:\n"
    "    • [اليوم]: من [من] لـ [إلى] — [اسم الخدمة]\n"
    "    حابب تحجز في أنهي فرع وأنهي يوم؟\n\n"
    "With only ONE branch and ONE day, use the same layout and ask about "
    "that day directly (\"تحب أشوف لك المواعيد المتاحة ليوم [اليوم]؟\"). "
    "If a branch is ALREADY confirmed, skip the branch entirely and go "
    "straight to `list_available_days_for_booking`.\n\n"
    "Keep the confirmation line exactly as it was and rewrite only what "
    "follows it.\n\n"
)


_GYN_MENTION_RE = re.compile(
    r"نساء\s*و?\s*توليد|أمراض\s*النساء|امراض\s*النساء|"
    r"gyn[ae]colog\w*|obstetric\w*"
)

# Signals that the PATIENT (never the assistant) actually raised
# something gynaecological/obstetric themselves - only these justify the
# reply mentioning نساء وتوليد at all. Deliberately narrow: general
# symptoms (abdominal pain, nausea, dizziness) must NOT appear here, or
# they would silently "justify" the exact violation this guard exists to
# catch.
_PREGNANCY_SIGNAL_RE = re.compile(
    r"حمل|حامل|حاملة|الدوره|الدورة|دوره\s*شهريه|دورة\s*شهرية|الطمث|"
    r"تاخر\s*الدوره|تأخر\s*الدورة|اجهاض|إجهاض|ولاده|ولادة|رضاعه|رضاعة|"
    r"pregnan\w*|menstru\w*|period\s*late"
)


def _reply_offers_unauthorized_gynecology(reply_text: str, state: AgentState, agent_name: str = "") -> bool:
    """True when the reply names نساء وتوليد (directly or as a "would you
    also like me to check" secondary offer) despite the PATIENT never
    having raised anything gynaecological or obstetric themselves.

    SCOPED TO THE MEDICAL SPECIALIST ONLY. Confirmed real production
    false positive: the BOOKING specialist showed the complete, accurate
    roster of every doctor registered at a branch the patient had just
    picked (dentistry, internal medicine, retina surgery, gynaecology -
    whatever is actually there), with zero symptom involved and zero
    editorializing - just an honest list. نساء وتوليد being one real
    entry in a factual "who is at this branch" listing is not the same
    thing as the MEDICAL specialist recommending or offering it
    unprompted in response to a symptom; flagging it here would mean
    silently deleting a real, bookable doctor from an accurate roster,
    which is its own kind of wrong. Only "medical" ever gets to this
    check at all.

    WHY (for the medical case): confirmed real production failure,
    twice. First, abdominal pain and vomiting were routed straight to
    نساء وتوليد with an unprompted remark about "الجهاز التناسلي
    الأنثوي". After the prompt was fixed to forbid that, the SAME
    symptom correctly named طب الباطنة but then tacked on "أو تحبيني
    أدور لك دكاترة نساء وتوليد كمان؟" in the same message - still
    naming the specialty unprompted, just softened into an offer
    instead of the headline answer. Prompt instructions alone did not
    hold, so this is the same kind of deterministic last-line-of-
    defence as `_find_invented_branches`/`_reply_invents_availability`."""

    if agent_name and agent_name != "medical":
        return False

    if not reply_text or not _GYN_MENTION_RE.search(reply_text):
        return False

    for msg in state.get("messages", []):
        if getattr(msg, "type", None) != "human":
            continue
        content = getattr(msg, "content", "")
        if content and _PREGNANCY_SIGNAL_RE.search(str(content)):
            return False  # the patient genuinely raised it themselves

    return True


_GYN_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU NAMED نساء وتوليد WITHOUT THE PATIENT RAISING IT - REWRITE\n"
    "============================================================\n"
    "Your previous draft mentioned نساء وتوليد (gynaecology/obstetrics) - "
    "even if only as a second, optional offer (\"أو تحبيني أدور لك دكاترة "
    "نساء وتوليد كمان؟\") - but nothing the patient has said in this "
    "conversation raised pregnancy, menstruation, or anything "
    "gynaecological/obstetric themselves.\n\n"
    "Naming that specialty at all - even as an offered alternative - is "
    "not allowed on your own initiative. Rewrite your reply with ONLY "
    "the genuinely relevant general specialty (e.g. طب الباطنة). If you "
    "believe pregnancy is truly worth ruling out, ask the single plain "
    "question \"في احتمال يكون حمل؟\" INSTEAD of naming any specialty "
    "this turn, and let their answer decide what to search next.\n\n"
    "Rewrite the reply now without mentioning نساء وتوليد.\n\n"
)


# ==========================================================
# The single finaliser every outgoing reply passes through
# ==========================================================

# Scaffolding the DIRECTIVES use to delimit text the model must copy
# verbatim. It is instruction plumbing and must never reach a patient.
#
# CONFIRMED REAL PRODUCTION FAILURE: a greeting went out with
# "[END-EXACT-TEXT]" printed underneath it. Every directive tells the
# model in words not to include these lines, and that held right up
# until it didn't - which is the whole argument for stripping them in
# code instead of asking. The cost of being wrong here is a patient
# seeing internal machinery in the first message the clinic ever sends
# them; the cost of the strip is nothing, because no legitimate reply
# ever contains these tokens.
_DIRECTIVE_SCAFFOLD_RE = re.compile(
    r"^[ \t]*(?:"
    r"\[(?:BEGIN|END)-EXACT-TEXT\]|"
    r"\[/?INTERNAL[^\]\n]*\]|"
    r"\[(?:BEGIN|END)[A-Z\- ]*\]|"
    r"[=\-]{6,}"
    r")[ \t]*$\n?",
    re.MULTILINE,
)


def _strip_directive_scaffolding(text: str) -> tuple:
    """Remove directive scaffolding lines. Returns (cleaned, n_removed)."""

    if not text:
        return text, 0

    cleaned, removed = _DIRECTIVE_SCAFFOLD_RE.subn("", text)

    if removed:
        # Collapse the blank run a removed line can leave behind.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned, removed


def _apply_output_contract(
    text: str,
    state: AgentState,
    target_language: Optional[str],
    agent_name: str,
) -> str:
    """Put ONE reply into the shape every reply must have.

    THE ONLY PLACE this happens, on purpose. A reply can reach the
    patient by more than one route - the model's first draft, or a
    rewrite produced after one of the verifiers below rejected that
    draft - and those routes used to apply DIFFERENT amounts of
    processing: the first got question-trimming + emoji numbering + the
    shared response contract, the rewrite got emoji numbering only. The
    patient therefore saw a differently-shaped message depending on
    whether the model happened to get it right first time, which is
    invisible from the outside and impossible to explain.

    Order matters and is the original order:
      0. Strip directive scaffolding ([BEGIN/END-EXACT-TEXT], rule
         lines). First, so nothing downstream has to reason about it.
      1. Trim any question beyond the first (ONE QUESTION PER MESSAGE).
      2. Emoji list badges, so every list looks the same - including
         lists no pre-built directive exists for.
      3. The shared response contract (filler openers, "let me check
         that", persona re-introductions, leaked routing language,
         irregular blank lines).
    """

    descaffolded, scaffold_lines = _strip_directive_scaffolding(text)
    if scaffold_lines:
        logger.warning(
            "agent[%s]: reply contained %d line(s) of directive scaffolding - stripped. Original: %r",
            agent_name, scaffold_lines, text,
        )

    trimmed, removed = _strip_extra_questions(descaffolded, state.get("templates") or {})
    if removed:
        logger.warning(
            "agent[%s]: reply contained %d extra question(s) beyond the first - trimmed. Original: %r",
            agent_name, removed, text,
        )

    normalized = _emojify_list_numbers(trimmed)

    if not config.REPLY_NORMALIZATION_ENABLED:
        return normalized

    greeting_for_dedupe = (
        None if not state.get("greeted")
        else _build_greeting(
            state.get("templates") or {},
            state["messages"][0].content if state["messages"] else "",
            target_language or "ar",
        )
    )

    contracted, changed = agents.normalize_reply(normalized, greeting_for_dedupe)
    if changed:
        logger.info(
            "agent[%s]: reply normalized to the shared response contract", agent_name,
        )

    return contracted


def _safe_fallback_reply(
    state: AgentState, target_language: Optional[str], failure_description: Optional[str] = None,
) -> str:
    """The message sent instead of a reply that a verifier flagged TWICE
    in the same turn (original draft, then its corrective retry) - see
    the zero-tolerance fallback in the verifier loop below.

    Deliberately generic and deliberately NOT run back through the
    verifier table or the LLM: the whole point is a reply that cannot
    itself contain an invented doctor, branch, availability claim, etc.,
    because it names none of them.

    `failure_description` is the verifier's own `description` string (see
    the reply-verifier table below) - used ONLY to pick which one of a
    small set of PRE-WRITTEN, equally-safe messages to send. This never
    lets the failed verifier's own content back into the reply; it's a
    lookup key, not text that gets echoed. Before this, every twice-
    flagged reply - a medical mismatch, a fabricated cancellation, an
    unsent complaint confirmation, an invented appointment - all
    collapsed into the same "حدث خطأ تقني" (technical issue) wording.
    CONFIRMED CONFUSING IN PRACTICE: that phrasing tells the patient the
    SYSTEM is broken, when the real situation is usually "the assistant
    can't confidently answer this specific thing right now" - a
    difference that changes what the patient should reasonably do next
    (try again later vs. ask for a human). Categories below are chosen
    to be actionable without naming any doctor/branch/date the failed
    verifier flagged, so they stay just as safe as the generic message.

    Prefers the clinic's own authored `msg_On_failure` wording (same
    field main.py falls back to for an empty reply) so the voice stays
    consistent with the rest of the conversation for the GENERIC case;
    the category-specific messages below intentionally do NOT use
    `msg_On_failure`, since a clinic only authors one failure message and
    it's written for the generic case, not for e.g. "we can't confirm
    your appointment" specifically. Picks a plain English default over
    an Arabic template when the conversation itself is in English,
    rather than switching languages on the one message where the patient
    most needs clarity."""

    is_english = (target_language or "").strip().lower().startswith("en")
    desc = (failure_description or "").lower()

    # Ordered so a more specific match wins over a more general one when
    # a description could plausibly match more than one category.
    _CATEGORY_MESSAGES = (
        (
            ("medical-guidance", "specialty that does not treat", "specialty catalogue"),
            "معلش، مش قادرة أحدد لك التخصص الأنسب لحالتك بدقة كافية دلوقتي 🌷\n"
            "أفضل حاجة إنك تتواصل مع فريقنا الطبي مباشرة يوجهوك صح. تحب أحولك لهم؟",
            "Sorry, I can't confidently match your symptoms to the right "
            "specialty just now 🌷\nIt's best to speak directly with our "
            "medical team so they can guide you properly. Would you like "
            "me to connect you with them?",
        ),
        (
            ("fabricated appointment", "invents availability", "invented availability",
             "no availability tool"),
            "معلش، مش قادرة أتأكد من موعد فعلي متاح دلوقتي 🌷\n"
            "ممكن نرجع نشوف الأيام والمواعيد المتاحة تاني من الأول؟",
            "Sorry, I can't confirm a real available slot right now 🌷\n"
            "Shall we look at the available days and times again from the "
            "start?",
        ),
        (
            ("cancellation without", "confirm cancelling", "offers cancellation without lookup"),
            "معلش، مش لاقية حجز مؤكد بالمعلومات دي 🌷\n"
            "ممكن تبعتلي رقم الحجز أو رقم الجوال المسجل بيه الحجز؟",
            "Sorry, I can't find a confirmed booking with that information 🌷\n"
            "Could you send me the booking reference or the phone number "
            "the booking is under?",
        ),
        (
            ("complaint was filed", "fabricates complaint submission"),
            "معلش، مش قادرة أأكد تسجيل الشكوى فعلياً دلوقتي 🌷\n"
            "حابب أحولك لفريق خدمة العملاء يتابعوها معاك مباشرة؟",
            "Sorry, I can't confirm your complaint was actually filed yet "
            "🌷\nWould you like me to connect you with our customer "
            "service team so they can follow up directly?",
        ),
        (
            ("branch had nothing available", "denies a branch", "branch denial"),
            "معلش، حصل لبس عندي في معلومة الفرع 🌷\n"
            "ممكن تأكدلي اسم الفرع تاني؟",
            "Sorry, I mixed up the branch information 🌷\n"
            "Could you confirm the branch name again?",
        ),
    )

    for keys, ar_msg, en_msg in _CATEGORY_MESSAGES:
        if any(key in desc for key in keys):
            return en_msg if is_english else ar_msg

    templates = state.get("templates") or {}
    authored = (templates.get("msg_On_failure") or "").strip()

    if authored and not is_english:
        return authored
    if authored and is_english:
        # An authored template exists but the conversation is in
        # English - only reuse it if it looks like it's already
        # written in English (avoids answering an English question
        # with an Arabic-only failure template).
        has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in authored)
        if not has_arabic:
            return authored

    if is_english:
        return "Sorry, I ran into a technical issue just now - could you please try that again? 🌷"
    return "عذرًا، حصلت مشكلة تقنية. ممكن تبعت رسالتك تاني؟ 🌷"


# ==========================================================
# The reply verifiers, as data
# ==========================================================
#
# Each entry is (check, correction_directive, description):
#   check(reply, state, agent_name) -> True when the reply is wrong
#   correction_directive(reply, state) -> the text prepended to the
#       system message for the single corrective retry
#   description -> what gets logged
#
# The signatures are uniform even where a particular check ignores an
# argument, so the loop that runs them needs no special cases - the
# special case in the old hand-written version (the branch check, which
# needed the offending names interpolated into its directive) is what
# had caused that block to drift away from the other five.

# HOW MANY TIMES ONE TURN MAY BE SENT BACK FOR TOOLS BY A VERIFIER.
#
# A verifier that rejects a reply and gets tool_calls back re-enters the
# agent, where the same verifier can reject again - an unbounded cycle
# whenever the check is one the model cannot satisfy. Two attempts is
# plenty for a genuine "go fetch the real data" correction; beyond that
# the verifier is the thing that is wrong, and the patient must still
# get an answer.
_MAX_VERIFIER_TOOL_RETRIES = 2

# {session_id: (human_message_count, retries_used)} - the count resets
# automatically on the next patient message, so this is per-turn without
# needing to thread extra state through the graph.
_VERIFIER_TOOL_RETRIES: Dict[str, tuple] = {}


def _verifier_tool_retries_exhausted(state: AgentState) -> bool:
    """True when this turn has already been sent back for tools by a
    verifier as many times as allowed. Records the attempt as a side
    effect, so each call consumes one."""

    session_id = state.get("session_id")
    if not session_id:
        return False

    from langchain_core.messages import HumanMessage as _HumanMessage

    turn_key = sum(
        1 for m in (state.get("messages") or []) if isinstance(m, _HumanMessage)
    )

    recorded_turn, used = _VERIFIER_TOOL_RETRIES.get(session_id, (None, 0))
    if recorded_turn != turn_key:
        used = 0

    if used >= _MAX_VERIFIER_TOOL_RETRIES:
        return True

    _VERIFIER_TOOL_RETRIES[session_id] = (turn_key, used + 1)

    # Keep this from growing without bound in a long-lived process.
    if len(_VERIFIER_TOOL_RETRIES) > 5000:
        for stale in list(_VERIFIER_TOOL_RETRIES)[:1000]:
            _VERIFIER_TOOL_RETRIES.pop(stale, None)

    return False


# ==========================================================
# The day the patient named, ignored or answered from memory
# ==========================================================
#
# These two close the last gap around a named day. The directives above
# tell the model what to do BEFORE it writes; these catch the finished
# reply when it did something else anyway - which is the only class of
# error a pre-write directive structurally cannot prevent, because there
# is no tool call left to shape.

# "The doctor works / doesn't work on <day>" - a claim about a roster,
# stated as fact. Also catches the softer forms ("متاح يوم", "بيجي يوم")
# that carry exactly the same weight for the patient.
_DAY_ROSTER_CLAIM_RE = re.compile(
    r"(?:ما|مش|مو|لا)\s*(?:عنده|عندها|بيجي|بتجي|يجي|تجي|متاح|متاحه|متاحة)[^.\n؟?]{0,20}يوم|"
    r"(?:عنده|عندها|بيجي|بتجي|متاح|متاحه|متاحة)\s*(?:عياده|عيادة)?\s*يوم|"
    r"(?:يعمل|تعمل|does\s*not\s*work|doesn'?t\s*work|works)\s*(?:on\s*)?"
)


def _reply_ignores_named_day(reply_text: str, state: AgentState) -> bool:
    """True when the patient named a specific weekday and the reply
    talks about availability WITHOUT that day ever having been checked.

    THE FAILURE THIS CATCHES, in the patient's words: "لو انا قولتله
    عاوزه احجز معاد مع دكتور احمد العقيل يوم التلات" and the reply comes
    back offering some other date, or asking which day they would like,
    or announcing from nowhere that the doctor does not come in on
    Tuesday. All three have the same root - the day was never checked -
    and all three read as not having been listened to.

    Narrow on purpose. It requires ALL of:
      - the patient's own latest message names a weekday;
      - `resolve_available_day` has NOT run since that message (once it
        has, the day HAS been checked and whatever the reply says about
        it is grounded);
      - the reply is actually about days/dates/availability - it names a
        weekday, prints a date, asks which day, or makes a roster claim.

    A reply that simply moves the booking along without touching the day
    (asking for a phone number, confirming a name) is left alone.
    """

    if not reply_text:
        return False

    named = _named_weekday_in_latest_human(state.get("messages") or [])
    if not named:
        return False

    messages = state.get("messages") or []

    # The day was checked properly on this turn - nothing to catch.
    if _tool_results_since_latest_human(messages, ("resolve_available_day",)):
        return False

    normalized = _norm_ar(reply_text)

    mentions_other_weekday = any(
        pattern.search(reply_text)
        for word, pattern in _WEEKDAY_WORD_RES.items()
        if _WEEKDAY_WORDS[word] != named["english"]
    )
    prints_a_date = bool(_DATE_IN_REPLY_RE.search(reply_text))
    asks_which_day = bool(_ASKS_WHICH_DAY_RE.search(normalized))
    claims_roster = bool(_DAY_ROSTER_CLAIM_RE.search(normalized))

    return bool(mentions_other_weekday or prints_a_date or asks_which_day or claims_roster)


def _named_day_correction_directive(reply_text: str, state: AgentState) -> str:
    named = _named_weekday_in_latest_human(state.get("messages") or []) or {}
    display = named.get("display", "")
    english = named.get("english", "")

    return (
        "============================================================\n"
        "THEY ASKED ABOUT " + display.upper() + " - CHECK IT, DO NOT WORK AROUND IT\n"
        "============================================================\n"
        "Your previous draft answered a question about " + display + " with "
        "another date, with a question about which day they want, or with "
        "a claim about the doctor's roster - and " + display + " was never "
        "actually checked.\n\n"
        "Call `resolve_available_day(weekday_name=\"" + english + "\")` now. "
        "Then:\n"
        "  - \"found\": call `get_available_slots_for_booking` with its own "
        "from_date/to_date and show that day's times.\n"
        "  - \"not_found\" / \"fully_booked\": say plainly what the result "
        "says about " + display + ", then call "
        "`list_available_days_for_booking` and show the real days in the "
        "same message.\n\n"
        "You may not state whether the doctor works " + display + " until "
        "that tool has answered - not from the conversation, not from a "
        "schedule you saw earlier, not from inference.\n\n"
        "Call the tool now instead of rewriting the sentence.\n\n"
    )


# ==========================================================
# Re-asking for something the patient just supplied
# ==========================================================

# NB1-Q1's own question. Legitimate as an opener; a failure once the
# patient has already named a doctor, specialty or service.
_PATH_CHOICE_QUESTION_RE = re.compile(
    r"تبدا\s*بالتخصص|تبدأ\s*بالتخصص|بالتخصص\s*ولا\s*بالدكتور|"
    r"start\s*with\s*(?:the\s*)?(?:specialty|speciality)"
)


# Tools that RESOLVE something the patient named. Once one of these has
# run for the current message, the information was not ignored - it was
# looked up. Whatever the reply says next is a RESULT, not a re-ask.
_ENTITY_RESOLUTION_TOOLS = (
    "match_entity_for_booking", "match_entity_info",
    "list_branches_for_specialty", "find_available_doctors",
    "find_branches_offering_service", "find_best_doctor_in_specialty",
    "list_specialties", "list_branch_services", "resolve_available_day",
    "list_available_days_for_booking",
)

# "I don't have a branch by that name", "I couldn't find a doctor called
# ...". A reply that opens this way is ANSWERING the thing they named -
# the opposite of pretending they never said it.
_NOT_FOUND_STATEMENT_RE = re.compile(
    r"معنديش|ما\s*عندي|مفيش|ما\s*في\s*(?:عندنا)?|ما\s*لقي(?:ت|نا)|لم\s*(?:اجد|نجد)|"
    r"لا\s*يوجد|مش\s*موجود|غير\s*موجود|"
    r"(?:don'?t|do\s*not)\s*have|(?:couldn'?t|could\s*not|can'?t)\s*find|"
    r"no\s+(?:branch|doctor|clinic)\s+(?:called|named|by\s+that\s+name)"
)

# An actual list of options in the reply - emoji digits or "1." / "2)".
# Showing the real alternatives is the correct answer to a name that did
# not match; it is not a question handed back.
_SHOWS_A_LIST_RE = re.compile(r"[1-9]️?⃣|(?:^|\n)\s*[1-9][.)\-]\s+")


def _reply_reasks_something_just_given(reply_text: str, state: AgentState) -> bool:
    """True when the reply asks for a piece of information the patient's
    OWN latest message already contained.

    This is the verifier form of `_build_multi_intent_directive`: that
    directive tells the model what it has been given, this one checks
    that it did not hand one of those items straight back.

    Restricted to the three questions that are unambiguous when the
    corresponding information is present in the same message:
      - the specialty-vs-doctor opener, when a doctor/specialty/service
        was named;
      - "which branch?", when a branch was named;
      - "which day?", when a day was named.

    Every other kind of question is left alone - a booking that has a
    doctor and a day still legitimately needs a phone number and a name,
    and a verifier that fired on those would block the flow rather than
    fix it.

    THREE STAND-DOWNS, all learned from one confirmed production
    failure. The patient asked about "فرع النيل"; the tools found no
    such branch; the reply was:

        معنديش فرع اسمه النيل. لكن عندنا هالفروع المتاحة حاليًا:
        1️⃣ المنار
        2️⃣ النزهة
        هل تحب تعرف تفاصيل أي فرع؟

    - a completely correct answer. This verifier flagged it anyway
    ("الفروع المتاحة" matches the generic-branch-question pattern, and
    "النيل" was in their message), the corrected retry was flagged
    again, and the patient received "حدث خطأ تقني 😕" instead. A good
    reply was destroyed by a guard meant to protect it, which is worse
    than the bug the guard exists for.
    """

    if not reply_text:
        return False

    messages = state.get("messages") or []
    index = _latest_human_index(messages)
    if index < 0:
        return False

    content = getattr(messages[index], "content", "")
    text = content if isinstance(content, str) else str(content)
    if len(text.split()) < 3:
        return False

    # STAND-DOWN 1: it WAS acted on. A resolution tool ran for this very
    # message, so the reply is reporting what that tool found.
    if _tool_results_since_latest_human(messages, _ENTITY_RESOLUTION_TOOLS):
        return False

    # STAND-DOWN 2: the reply says the thing they named was not found.
    # That is an answer about their input, not a request to repeat it.
    if _NOT_FOUND_STATEMENT_RE.search(_norm_ar(reply_text)):
        return False

    # STAND-DOWN 3: the reply shows the real options. Listing what
    # actually exists is how a patient recovers from naming something
    # that does not - it is the fix, not the failure.
    if _SHOWS_A_LIST_RE.search(reply_text):
        return False

    normalized = _norm_ar(reply_text)

    if _PATH_CHOICE_QUESTION_RE.search(normalized):
        if (_fragment_after_cue(text, _DOCTOR_CUE_RE)
                or _fragment_after_cue(text, _SPECIALTY_CUE_RE)):
            return True

    if _GENERIC_BRANCH_QUESTION_RE.search(normalized):
        if _fragment_after_cue(text, _BRANCH_CUE_RE):
            return True

    if _ASKS_WHICH_DAY_RE.search(normalized):
        if tools.resolve_weekday_index(text) is not None:
            return True

    return False


def _reasked_information_correction_directive(reply_text: str, state: AgentState) -> str:
    messages = state.get("messages") or []
    index = _latest_human_index(messages)
    content = getattr(messages[index], "content", "") if index >= 0 else ""
    text = content if isinstance(content, str) else str(content)

    supplied = []
    doctor_fragment = _fragment_after_cue(text, _DOCTOR_CUE_RE)
    if doctor_fragment:
        supplied.append("the doctor: \"" + doctor_fragment + "\"")
    branch_fragment = _fragment_after_cue(text, _BRANCH_CUE_RE)
    if branch_fragment:
        supplied.append("the branch: \"" + branch_fragment + "\"")
    specialty_fragment = _fragment_after_cue(text, _SPECIALTY_CUE_RE)
    if specialty_fragment:
        supplied.append("the specialty/service: \"" + specialty_fragment + "\"")
    weekday = tools.resolve_weekday_index(text)
    if weekday is not None:
        english = ["Monday", "Tuesday", "Wednesday", "Thursday",
                   "Friday", "Saturday", "Sunday"][weekday]
        supplied.append("the day: " + _ARABIC_DAY_NAMES.get(english.lower(), english))

    listed = "\n".join("    - " + item for item in supplied)

    return (
        "============================================================\n"
        "YOU ASKED FOR SOMETHING THEY ALREADY TOLD YOU\n"
        "============================================================\n"
        "Your previous draft asked the patient for a piece of "
        "information their own last message already contained:\n\n"
        + listed + "\n\n"
        "Rewrite the turn. Act on what they gave you - resolve each item "
        "through its own tool (`match_entity_for_booking` for a doctor or "
        "branch, `resolve_available_day` for a day) and carry on from the "
        "first step that is genuinely still unanswered.\n\n"
        "Do not ask for any of the items above again. If a tool cannot "
        "match one of them, say what could not be found and offer the "
        "real options - that is a different message from asking the "
        "question over as though they had never answered it.\n\n"
    )


# ============================================================
# MESSAGES THE SCOPE REFUSAL MUST NEVER ANSWER
# ============================================================
#
# WHY THIS EXISTS: two kinds of message look "out of scope" to the model
# because no tool can answer them, and both were being met with the
# clinic's generic deflection - "عذرًا 🌷 أنا [name]، المساعدة
# الافتراضية في [clinic]، ومختصة بمساعدتك في خدمات المستشفى مثل حجز أو
# تعديل المواعيد..."
#
#   1. ASKING WHAT MEDICATION OR DOSE TO TAKE. CONFIRMED IN PRODUCTION:
#      a patient described a bad headache and fever, got a good medical
#      guidance reply, then pushed - "Just tell me the normal adult
#      dose, everyone knows it anyway." - and received the service menu.
#      Refusing the dose is right. Answering a person in pain with a
#      list of the things you CAN do instead is not: it reads as being
#      brushed off, and it drops both the reason and the offer of real
#      help.
#
#   2. SAYING THEY WANT TO HARM THEMSELVES. The same deflection here is
#      far worse than unhelpful.
#
# The medication ban and the crisis response already existed - but only
# inside the MEDICAL GUIDANCE section of the prompt, which reaches the
# `medical` and `concierge` specialists and nobody else. Neither message
# above scores on any router cue (`score_message` returns {} for "Just
# tell me the normal adult dose" and for "I want to kill myself"), so
# whichever specialist happened to be active kept the turn - and most of
# them had never been told any of this.
#
# These directives are therefore built in `_run_agent`, which every
# specialist runs, and keyed on the MESSAGE rather than on which agent
# won the routing.

_MEDICATION_REQUEST_RE = re.compile(
    # Arabic - "give me a medicine", "what do I take", "how many pills",
    # "what dose", "prescribe me something", "a painkiller".
    r"(?:ادي|اديني|اعطيني|عطيني|هات|هاتي|وصف|وصفلي|اكتبلي|اكتب\s*لي)\w*\s*"
    r"(?:\w+\s+){0,2}(?:دوا|دواء|علاج|مسكن|حبوب|برشام)|"
    r"(?:اخد|اخذ|اشرب|اتناول|استخدم|اعطي|اديه|اسقي)\s*(?:\w+\s+){0,2}"
    r"(?:ايه|ايش|وش|شنو|كام|قد\s*ايه)|"
    r"(?:ايه|ايش|وش|شنو|اي|أي)\s*(?:ال)?(?:دوا|دواء|علاج|مسكن|حبوب|برشام)|"
    r"(?:كام|قد\s*ايه|كم)\s*(?:\w+\s+){0,2}(?:حبه|حبة|قرص|مسكن|جرعه|جرعة|مره|مرة)|"
    r"(?:ال)?(?:جرعه|جرعة|الجرعات)|"
    r"(?:دوا|دواء|علاج|مسكن)\s*(?:\w+\s+){0,2}(?:للصداع|للحراره|للحرارة|للالم|للوجع|مناسب)|"
    # English.
    r"\b(?:what|which)\s+(?:medicine|medication|drug|painkiller|tablet|pill)s?\b|"
    r"\bwhat\s+(?:should|can|do)\s+i\s+take\b|"
    r"\bhow\s+(?:many|much|often)\b[^.\n?]{0,30}"
    r"\b(?:painkiller|paracetamol|panadol|ibuprofen|tablet|pill|dose|mg)\w*|"
    r"\b(?:dose|dosage|dosing)\b|"
    r"\bprescribe\s+(?:me\s+)?(?:a|some|any)?\s*\w*|"
    r"\b(?:give|recommend)\s+me\s+(?:a\s+|some\s+|any\s+)?"
    r"(?:medicine|medication|drug|painkiller|tablet|pill|something|anything)|"
    r"\bis\s+it\s+(?:safe|ok(?:ay)?)\s+to\s+take\b",
    re.IGNORECASE,
)

_CRISIS_RE = re.compile(
    # Arabic, including the colloquial future prefix ("هنتحر" = "I'm
    # going to kill myself") that a stem-only pattern misses.
    r"(?:ه|ح|سا|سأ)?انتحر|(?:ه|ح)نتحر|الانتحار|"
    r"(?:عايز|عاوز|بدي|ابي|ابغى|نفسي)\s*(?:\w+\s+){0,2}(?:اموت|امو ت|انهي\s*حياتي|اقتل\s*نفسي)|"
    r"(?:مش|ما|مو)\s*(?:عايز|عاوز|بدي|ابي)\s*(?:\w+\s+){0,2}(?:اعيش|اكمل|اكمل\s*حياتي)|"
    r"(?:اذي|أذي|اؤذي|أؤذي|اجرح|أجرح)\s*نفسي|"
    r"(?:انهي|أنهي|اخلص\s*من)\s*حيات|"
    r"(?:تعبت|زهقت|مليت)\s*من\s*(?:ال)?حياه|"
    # English.
    r"\bkill\s+my\s?self\b|\bsuicid\w*|\bend\s+(?:my|it\s+all)\b[^.\n]{0,12}\blife\b|"
    r"\bend\s+my\s+life\b|\bwant\s+to\s+die\b|\bhurt\s+my\s?self\b|"
    r"\bself[\s-]?harm\b|\bdon'?t\s+want\s+to\s+(?:live|be\s+here)\b|"
    r"\bno\s+reason\s+to\s+live\b",
    re.IGNORECASE,
)


def _latest_human_text(messages: list) -> str:
    """The patient's most recent message as plain text."""

    index = _latest_human_index(messages or [])
    if index < 0:
        return ""
    content = getattr(messages[index], "content", "")
    return content if isinstance(content, str) else str(content)


def _asks_for_medication(messages: list) -> bool:
    text = _latest_human_text(messages)
    return bool(text) and bool(_MEDICATION_REQUEST_RE.search(_norm_ar(text)))


def _signals_crisis(messages: list) -> bool:
    text = _latest_human_text(messages)
    return bool(text) and bool(_CRISIS_RE.search(_norm_ar(text)))


def _build_crisis_directive(messages: list, templates: dict) -> str:
    """A patient has said they want to harm themselves. Nothing else
    this turn matters.

    Deliberately available to EVERY specialist, not just medical: these
    messages score nothing on the router's cue table, so they land on
    whichever agent was already active - which in production was one
    that had never been given the crisis rules and answered with the
    service menu."""

    if not _signals_crisis(messages):
        return ""

    clinic = (templates or {}).get("_clinic_name_ar") or (templates or {}).get("_clinic_name") or ""
    at_clinic = (" at " + clinic) if clinic else ""

    return (
        "============================================================\n"
        "THIS PATIENT HAS SAID THEY WANT TO HARM THEMSELVES\n"
        "============================================================\n"
        "This overrides every other instruction in this prompt, "
        "including the out-of-scope refusal, the one-question rule, and "
        "whatever step of whatever flow you were in. Drop all of it.\n\n"
        "YOU MUST NOT send the out-of-scope refusal, the service menu, "
        "or any variation of \"I can only help with hospital services\". "
        "Answering this message with a list of what you are able to do "
        "instead is the single worst reply available to you here.\n\n"
        "Write, in the conversation's own language, warmly and plainly:\n"
        "  1. That you have heard them and are taking it seriously. Do "
        "not minimise it, do not reassure them that it will pass, and do "
        "not ask them to explain or justify it.\n"
        "  2. That they should not be alone with this right now - urge "
        "them to reach a mental-health professional, someone they trust, "
        "or their local emergency number straight away. If they are in "
        "immediate danger, emergency services come first.\n"
        "  3. ONE gentle offer of something concrete you can actually "
        "do: put them through to a member of staff" + at_clinic + ", or "
        "help them book with a doctor here.\n\n"
        "Do NOT diagnose. Do NOT name a medication or a dose. Do NOT "
        "print a doctor roster or a specialty list at them - a menu is "
        "not what this moment needs. Do NOT invent a helpline number: "
        "say \"a crisis line\" or \"your local emergency number\" unless "
        "a real one is configured for this clinic.\n\n"
        "Keep it short, human, and unhurried.\n\n"
    )


def _build_medication_request_directive(messages: list, templates: dict) -> str:
    """They asked what medicine, or how much of it, to take.

    The refusal itself was never in doubt - the medication ban is
    already absolute. What went wrong is what the refusal was replaced
    BY: the generic scope deflection, which drops the reason, drops the
    comfort, and drops the offer of an actual appointment. This pins all
    three back on."""

    if not _asks_for_medication(messages):
        return ""

    # A crisis message wins outright; two competing "override
    # everything" blocks in one prompt is exactly the failure mode this
    # file keeps having to design around.
    if _signals_crisis(messages):
        return ""

    clinic = (templates or {}).get("_clinic_name_ar") or (templates or {}).get("_clinic_name") or ""
    clinic_phrase = (" at " + clinic) if clinic else " here"

    return (
        "============================================================\n"
        "THEY ASKED WHAT MEDICINE OR WHAT DOSE TO TAKE\n"
        "============================================================\n"
        "You cannot answer that, and you already know it - the "
        "medication ban is absolute, and it does not soften because they "
        "asked twice, because they said \"everyone knows it anyway\", "
        "because the drug is over the counter, or because they only "
        "want \"the normal amount\".\n\n"
        "BUT DO NOT SEND THE OUT-OF-SCOPE REFUSAL. Not the service menu, "
        "not \"I'm the virtual assistant for [clinic] and I can help you "
        "with bookings...\", not any variation of it. CONFIRMED REAL "
        "PRODUCTION FAILURE: a patient with a headache and a fever asked "
        "for a normal adult dose and was answered with the list of "
        "things the assistant is able to do instead. That is not a "
        "refusal, it is a brush-off - and it arrived in Arabic in an "
        "English conversation on top of it. This is a health question "
        "from someone who feels unwell; it is squarely inside what you "
        "are for, and the answer is a real one.\n\n"
        "Write these THREE parts, in the conversation's own language, in "
        "this order, as one short message:\n\n"
        "  1. THE REFUSAL, WITH ITS REASON. You can't advise on "
        "medication or dosing, because the right choice and amount "
        "depend on their own health, allergies, other medicines they "
        "take and their weight - and only a doctor who has seen them can "
        "decide it safely. Say it warmly; they are unwell, not being "
        "difficult.\n"
        "  2. SOMETHING SAFE THEY CAN ACTUALLY DO NOW. Rest, fluids, a "
        "quiet dark room for a headache, keeping an eye on the symptom - "
        "non-drug comfort only. If the symptom has any warning sign "
        "attached (very high fever, a sudden severe headache, confusion, "
        "trouble breathing, vision changes), say plainly that this needs "
        "urgent care today.\n"
        "  3. ONE offer: helping them book an appointment" + clinic_phrase
        + " to see a doctor about it. That is the question the message "
        "ends on - and the only question in it.\n\n"
        "NEVER name a drug, a brand, a generic name, a dose, a "
        "frequency, or a \"safe\" amount, not even to say which one you "
        "are declining to recommend. Naming it is recommending it.\n\n"
        "If they ask again after this, hold the same line in fewer words "
        "and keep the offer of an appointment open. Do not escalate to "
        "the scope refusal on the second or third ask - repetition does "
        "not turn a health question into an off-topic one.\n\n"
    )


def _reply_scope_refuses_a_health_message(reply_text: str, state: AgentState) -> bool:
    """True when the finished reply is the generic out-of-scope
    deflection and the message it answers was about the patient's
    health - a medication question, or a crisis.

    The last line of defence for the two directives above: they shape
    the turn before it is written, this catches the turn that ignored
    them. Narrow by design - it fires only on the refusal text itself,
    identified by `_is_scope_refusal`, so an ordinary reply that happens
    to mention what the assistant can help with is untouched."""

    if not reply_text:
        return False

    messages = state.get("messages") or []
    if not (_asks_for_medication(messages) or _signals_crisis(messages)):
        return False

    return _is_scope_refusal(reply_text, state.get("templates") or {})


def _health_message_refusal_correction(reply_text: str, state: AgentState) -> str:
    messages = state.get("messages") or []
    templates = state.get("templates") or {}

    if _signals_crisis(messages):
        return (
            "============================================================\n"
            "YOU SENT THE SERVICE MENU TO SOMEONE IN CRISIS\n"
            "============================================================\n"
            "Your previous draft answered a patient who said they want "
            "to harm themselves with the out-of-scope refusal - a list "
            "of the things you are able to help with instead.\n\n"
            "Rewrite it completely.\n\n"
        ) + _build_crisis_directive(messages, templates)

    return (
        "============================================================\n"
        "YOU BRUSHED OFF A HEALTH QUESTION WITH THE SERVICE MENU\n"
        "============================================================\n"
        "Your previous draft answered someone who is unwell and asked "
        "about medication with the generic out-of-scope refusal. "
        "Declining the dose was right; replacing the whole answer with a "
        "list of your own capabilities was not - it drops the reason, "
        "the comfort, and the offer of an appointment.\n\n"
        "Rewrite it as the three parts below.\n\n"
    ) + _build_medication_request_directive(messages, templates)


# ============================================================
# THE IDENTIFIER IS ALREADY IN THE MESSAGE
# ============================================================
#
# STEP 1 of the cancel/reschedule flow says it in plain words: if the
# message already contains a booking reference or a phone number, use it
# and skip the "reference or phone?" question. It was not followed.
#
# CONFIRMED REAL PRODUCTION FAILURE (medtown, session
# 201003365691+medtown2, 2026-09-01 09:00):
#
#   patient : تعديل موعد برقم GuestBookingNum-2026-09-01-076
#   reply   : نكمل تعديل موعدك على نفس رقم الواتساب ده؟ ✅
#   patient : لا برقم حجز  GuestBookingNum-2026-09-01-076
#   reply   : ممكن رقم جوالك مع رمز الدولة أو رقم الحجز تكتبه لي،
#             عشان أقدر أجيب بيانات موعدك؟
#
# The reference was in the very first message, repeated in the second,
# and asked for a third time. `_reply_skips_reference_or_phone_question`
# already knew it was there - but only used that to STAND ITSELF DOWN.
# Nothing anywhere told the model to go and use it.
#
# This is the same failure as the booking side's named-day problem, in a
# different flow: information the patient supplied, discarded because
# the flow starts at its own step 1 rather than at the first step that
# is genuinely unanswered. So it gets the same treatment - a directive
# that states what was supplied, and a verifier that catches the reply
# which asks for it anyway.

# A booking reference: letters, then at least one hyphen-joined group,
# with a digit somewhere in it. Covers "GBN-2026-06-20-151" and
# "GuestBookingNum-2026-09-01-076" without matching ordinary hyphenated
# words ("مستشفى ميدتاون", "walk-in").
_BOOKING_REF_RE = re.compile(r"\b[A-Za-z]{2,}[A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")

# A phone number the patient typed. Checked only AFTER any reference has
# been cut out of the text: "GuestBookingNum-2026-09-01-076" ends in
# fourteen digits and three hyphens, and would otherwise be read as a
# phone number as well as a reference.
_SUPPLIED_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")


def _booking_reference_in(text: str) -> str:
    """The booking reference in this text, or "" - returned exactly as
    the patient typed it, so it can be passed straight to
    `lookup_appointment` without anyone retyping it."""

    for candidate in _BOOKING_REF_RE.findall(text or ""):
        if any(ch.isdigit() for ch in candidate):
            return candidate
    return ""


def _supplied_phone_in(text: str) -> str:
    """A phone number in this text, or "" - with any booking reference
    removed first (see `_SUPPLIED_PHONE_RE`)."""

    remainder = text or ""
    reference = _booking_reference_in(remainder)
    if reference:
        remainder = remainder.replace(reference, " ")

    match = _SUPPLIED_PHONE_RE.search(remainder)
    return match.group(0).strip() if match else ""


def _build_supplied_identifier_directive(messages: list) -> str:
    """The patient's own latest message contains the booking reference
    (or the phone number) needed to find their appointment. Use it.

    Stands down as soon as an identity tool has run for this message -
    at that point the identifier HAS been used and the flow is past
    STEP 1, where phone questions become legitimate again."""

    if not messages:
        return ""

    text = _latest_human_text(messages)
    if not text:
        return ""

    reference = _booking_reference_in(text)
    phone = "" if reference else _supplied_phone_in(text)

    if not (reference or phone):
        return ""

    # Already acted on this turn.
    if _tool_results_since_latest_human(messages, _IDENTITY_VERIFICATION_TOOLS):
        return ""

    if reference:
        supplied = "A BOOKING REFERENCE: " + reference
        action = (
            "    lookup_appointment(ref_number=\"" + reference + "\")\n\n"
            "Copy it exactly as written above - do not reformat it, "
            "do not strip the prefix, do not retype it from memory.\n\n"
            "A REFERENCE SKIPS IDENTITY VERIFICATION ENTIRELY. No OTP, "
            "no phone comparison, no \"is this the same WhatsApp "
            "number?\" - STEP 2 exists for the phone path and this is "
            "not the phone path. Go straight to showing them the "
            "booking.\n\n"
        )
    else:
        supplied = "A PHONE NUMBER: " + phone
        action = (
            "    validate_phone_format, then compare_phone, then "
            "lookup_appointment - with THIS number.\n\n"
            "Do not ask them to send a phone number, and do not ask "
            "whether to use the WhatsApp number instead: they chose one "
            "by typing it.\n\n"
        )

    return (
        "============================================================\n"
        "THEY ALREADY GAVE YOU WHAT YOU NEED TO FIND THE BOOKING\n"
        "============================================================\n"
        "Their latest message contains " + supplied + "\n\n"
        "Your ONLY next action is:\n"
        + action +
        "DO NOT ask \"reference number or phone number?\". That question "
        "exists to find out which one they have; they have just told "
        "you, in the message you are replying to. STEP 1 says in as many "
        "words to skip it in exactly this case.\n\n"
        "DO NOT ask for the reference or the phone number again in any "
        "wording, and do not ask them to \"confirm\" it. CONFIRMED REAL "
        "PRODUCTION FAILURE: a patient opened with \"تعديل موعد برقم "
        "GuestBookingNum-2026-09-01-076\", was asked whether to continue "
        "on the same WhatsApp number, replied \"لا برقم حجز "
        "GuestBookingNum-2026-09-01-076\" - and was then asked for their "
        "phone number or a booking reference. The same reference, asked "
        "for three times in a row.\n\n"
        "If the lookup comes back \"not_found\", THAT is worth a reply - "
        "say the reference did not match anything and ask them to check "
        "it. That is a different message from asking for it as though "
        "they had never answered.\n\n"
    )


_ASKS_FOR_REF_OR_PHONE_RE = re.compile(
    r"رقم\s*(?:ال)?حجز|رقم\s*(?:ال)?جوال|رقم\s*(?:ال)?موبايل|رقم\s*(?:ال)?تليفون|"
    r"رقم\s*(?:ال)?هاتف|رمز\s*(?:ال)?دوله|"
    r"booking\s*reference|reference\s*number|phone\s*number|mobile\s*number"
)


def _reply_reasks_an_identifier_already_supplied(reply_text: str, state: AgentState) -> bool:
    """True when the reply asks for a booking reference or a phone
    number that the patient's own latest message already contains.

    The verifier half of `_build_supplied_identifier_directive`. Narrow:
    it needs the reply to actually be ASKING (a question mark, or one of
    the request phrasings) and the identifier to be present in the very
    message being answered - a reference mentioned ten turns ago does
    not count, because the conversation may legitimately have moved to a
    different booking since."""

    if not reply_text:
        return False

    messages = state.get("messages") or []
    text = _latest_human_text(messages)
    if not text:
        return False

    if not (_booking_reference_in(text) or _supplied_phone_in(text)):
        return False

    # Already looked it up on this turn - anything the reply asks for
    # now is a legitimate later step, not a re-ask.
    if _tool_results_since_latest_human(messages, _IDENTITY_VERIFICATION_TOOLS):
        return False

    folded = _norm_ar(reply_text)
    asking = "?" in reply_text or "؟" in reply_text
    return bool(asking and _ASKS_FOR_REF_OR_PHONE_RE.search(folded))


def _supplied_identifier_correction(reply_text: str, state: AgentState) -> str:
    return (
        "============================================================\n"
        "YOU ASKED FOR THE REFERENCE THEY JUST GAVE YOU\n"
        "============================================================\n"
        "Your previous draft asked for a booking reference or a phone "
        "number. It is in the message you are replying to.\n\n"
    ) + _build_supplied_identifier_directive(state.get("messages") or [])


# ============================================================
# "CANCEL IT" / "CHANGE IT" - THE ONE THEY JUST BOOKED
# ============================================================
#
# A patient who has just finished booking and then says "الغيه",
# "الغي الحجز ده", "عدله" or "change it" is talking about the
# appointment still on their screen. Answering that with STEP 1's
# "reference number or phone number?" - or worse, the OTP dance - makes
# them identify a booking the assistant created itself thirty seconds
# earlier.
#
# WHY IT HAPPENS: `create_new_booking` deliberately wipes the booking
# session on success (so the NEXT booking starts clean), and the
# reference it returns then exists nowhere except inside a ToolMessage
# in the history. The cancel/reschedule specialist that the router
# hands the turn to starts at its own STEP 1, which knows nothing about
# any of it. Nothing was carrying the reference across that handover.
#
# The same applies one step earlier: a booking the patient has just been
# SHOWN by `lookup_appointment` is equally "this one" when they say
# "cancel it" next.

_CANCEL_OR_CHANGE_INTENT_RE = re.compile(
    # Arabic. `\w*` matters: the object pronoun attaches to the verb, so
    # "ألغيه", "ألغيها", "عدله", "أجله" are each a single word - the
    # very phrasings this directive exists for.
    r"(?:^|\s)(?:الغ|ابطل|بطل)\w*|"
    r"(?:^|\s)(?:عدل|اعدل|غير|اغير|اجل|اؤجل|انقل|قدم)\w*|"
    r"(?:ال)?(?:الغاء|تعديل|تاجيل|تغيير)\b|"
    r"\bcancel\b|\breschedul\w*|\bpostpon\w*|"
    r"\b(?:change|move|shift)\s+(?:it|this|that|the\s+(?:booking|appointment))\b",
    re.IGNORECASE,
)


def _just_created_booking_reference(messages: list) -> str:
    """The reference of a booking `create_new_booking` made in THIS
    conversation, or "".

    Returns "" once that booking has been cancelled - there is nothing
    left to point a later "cancel it" at, and silently re-targeting an
    already-cancelled booking would be worse than asking.
    """

    for index in range(len(messages or []) - 1, -1, -1):
        message = messages[index]
        if getattr(message, "name", None) != "create_new_booking":
            continue
        try:
            data = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if not (isinstance(data, dict) and data.get("status") == "success"):
            continue

        # Cancelled since? Then it is gone.
        for later in messages[index + 1:]:
            if getattr(later, "name", None) == "cancel_appointment":
                try:
                    later_data = json.loads(later.content)
                except (ValueError, TypeError):
                    continue
                if isinstance(later_data, dict) and later_data.get("status") == "success":
                    return ""

        return str(data.get("booking_ref") or "")

    return ""


def _reference_of_the_booking_on_screen(messages: list) -> str:
    """The reference of whichever booking this conversation is currently
    about - the one just created, or failing that the one most recently
    looked up and shown."""

    created = _just_created_booking_reference(messages)
    if created:
        return created

    appointment = _last_appointment_record(messages)
    return str(appointment.get("ref") or appointment.get("bookingRefNum") or "")


def _build_just_booked_directive(messages: list) -> str:
    """They said "cancel it"/"change it" about a booking already on the
    table. Point the flow at that booking instead of starting STEP 1
    over."""

    if not messages:
        return ""

    text = _latest_human_text(messages)
    if not text:
        return ""

    if not _CANCEL_OR_CHANGE_INTENT_RE.search(_norm_ar(text)):
        return ""

    # They typed a reference of their own - that one wins, and
    # `_build_supplied_identifier_directive` is already handling it.
    # Two directives naming two different references would be worse
    # than either alone.
    if _booking_reference_in(text):
        return ""

    reference = _reference_of_the_booking_on_screen(messages)
    if not reference:
        return ""

    # Already fetched on this turn.
    if _tool_results_since_latest_human(messages, _IDENTITY_VERIFICATION_TOOLS):
        return ""

    just_created = bool(_just_created_booking_reference(messages))

    which = (
        "the appointment you booked for them a moment ago, in this same "
        "conversation"
        if just_created else
        "the appointment you looked up and showed them earlier in this "
        "conversation"
    )

    identity_note = (
        "THEIR IDENTITY IS NOT IN QUESTION. You created this booking for "
        "them minutes ago, on a number that was already verified to make "
        "it. Do not send an OTP, do not compare phone numbers, and do "
        "not ask whether to continue on the same WhatsApp number.\n\n"
        if just_created else
        "If this booking was found earlier by a verified phone number or "
        "by its reference, that verification still stands - do not "
        "restart it.\n\n"
    )

    return (
        "============================================================\n"
        "\"IT\" MEANS THE BOOKING ALREADY ON THE TABLE\n"
        "============================================================\n"
        "Their latest message asks to cancel or change a booking, and "
        "they did not name one - because they are talking about " + which
        + ".\n\n"
        "Its reference is:\n"
        "    " + reference + "\n\n"
        "Your ONLY next action is:\n"
        "    lookup_appointment(ref_number=\"" + reference + "\")\n\n"
        "Copy the reference exactly as written above. Do not retype it "
        "from memory and do not reformat it.\n\n"
        "DO NOT ask \"reference number or phone number?\". That question "
        "finds out WHICH booking they mean; there is only one on the "
        "table and they just referred to it. Asking a patient to "
        "identify an appointment you made for them a moment ago reads as "
        "though the conversation restarted.\n\n"
        + identity_note +
        "After the lookup, continue the normal flow from the step that "
        "shows them the booking and asks for confirmation - cancelling "
        "still needs an explicit yes, and a reschedule still needs a new "
        "day and time. Nothing about this shortcut skips a confirmation; "
        "it only skips re-identifying a booking that was never in doubt.\n\n"
        "If the lookup comes back \"not_found\", say so plainly and ask "
        "them to check the reference - do not fall back to asking for "
        "their phone number as if the last few minutes had not happened.\n\n"
    )


def _reply_reasks_which_booking_when_only_one_is_on_the_table(
    reply_text: str, state: AgentState,
) -> bool:
    """True when the reply asks which booking they mean, while the
    conversation has exactly one on the table and their message just
    referred to it."""

    if not reply_text:
        return False

    messages = state.get("messages") or []
    if not _build_just_booked_directive(messages):
        return False

    folded = _norm_ar(reply_text)
    asking = "?" in reply_text or "؟" in reply_text
    if not asking:
        return False

    return bool(
        _ASKS_FOR_REF_OR_PHONE_RE.search(folded)
        or _SAME_WHATSAPP_QUESTION_RE.search(folded)
    )


def _just_booked_correction(reply_text: str, state: AgentState) -> str:
    return (
        "============================================================\n"
        "YOU ASKED WHICH BOOKING - THERE IS ONLY ONE\n"
        "============================================================\n"
        "Your previous draft asked the patient to identify a booking "
        "that is already on the table in this conversation.\n\n"
    ) + _build_just_booked_directive(state.get("messages") or [])


# ==========================================================
# What a verifier is actually protecting against
# ==========================================================
#
# The zero-tolerance fallback - swapping a twice-flagged reply for
# "حدث خطأ تقني 😕" - is exactly right for ONE kind of failure and
# exactly wrong for the other.
#
#   SAFETY  the reply ASSERTS SOMETHING UNTRUE: a doctor who does not
#           exist, a date no tool returned, a booking that was never
#           made, a medication. A patient can act on any of these and
#           be harmed by it. If the model cannot produce a clean version
#           in two attempts, saying nothing is genuinely better.
#
#   FLOW    the reply asks the wrong QUESTION, or asks at the wrong
#           step: re-requesting a reference they already gave, offering
#           the soonest date instead of the day they named, handing back
#           the specialty question. Clumsy, and worth one corrective
#           retry - but every word in it is true, and the patient can
#           still act on it. Replacing it with a technical error turns a
#           slightly awkward turn into a dead one.
#
# CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-09-01 09:12): a
# FLOW verifier misfired on a completely correct reply - "معنديش فرع
# اسمه النيل. لكن عندنا هالفروع المتاحة حاليًا: 1️⃣ المنار 2️⃣ النزهة" -
# fired again on the retry, and the patient received "حدث خطأ تقني 😕.
# تحاول مرة ثانية؟" instead of their answer. The verifier's own comment
# already says a verifier being wrong should cost one wasted call and
# never the whole conversation; for FLOW checks, this is what makes that
# true.
#
# SAFETY is the DEFAULT. An entry in `_REPLY_VERIFIERS` that does not
# say otherwise keeps today's strict behaviour exactly, so nothing
# loosens by omission - only the entries explicitly tagged below change.
_SAFETY = "safety"
_FLOW = "flow"

# Tagged by the distinctive part of each entry's own description, so the
# table itself stays a plain list of triples and this classification
# lives in one readable place instead of being scattered through it.
_FLOW_VERIFIER_MARKERS = (
    "already contained",
    "already on the table",
    "never checked with resolve_available_day",
    "already supplied",
    "generic out-of-scope service menu",
    "already been locked in",
    "just-shown day list",
    "reference-or-phone question ever being asked",
    "already specifically chose to identify by",
    "already has a verified phone",
    "before a doctor was confirmed and a time slot was selected",
    "should have been treated as another OTP retry",
    "right after the patient agreed to proceed on the channel number",
    "instead of showing the doctors",
    "instead of showing that doctor's own schedule",
    "printed the specialty catalogue",
    "re-send a full name that already had at least two parts",
    "offered the doctor roster again",
    "instead of continuing",
)


def _verifier_severity(description: str) -> str:
    """SAFETY unless the description matches a known FLOW marker."""

    for marker in _FLOW_VERIFIER_MARKERS:
        if marker in description:
            return _FLOW
    return _SAFETY


_REPLY_VERIFIERS = (
    (
        lambda reply, state, agent_name: _reply_scope_refuses_a_health_message(reply, state),
        lambda reply, state: _health_message_refusal_correction(reply, state),
        "reply answered a medication question or a self-harm disclosure with the "
        "generic out-of-scope service menu",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name in _EXISTING_BOOKING_AGENTS
            and _reply_reasks_which_booking_when_only_one_is_on_the_table(reply, state)
        ),
        lambda reply, state: _just_booked_correction(reply, state),
        "reply asked the patient to identify a booking that is already on the table "
        "in this conversation",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name in _EXISTING_BOOKING_AGENTS
            and _reply_reasks_an_identifier_already_supplied(reply, state)
        ),
        lambda reply, state: _supplied_identifier_correction(reply, state),
        "reply asked for a booking reference or phone number that the patient's own "
        "last message already contained",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name in _NEW_BOOKING_AGENTS
            and _reply_ignores_named_day(reply, state)
        ),
        lambda reply, state: _named_day_correction_directive(reply, state),
        "reply discussed availability while the day the patient explicitly named was "
        "never checked with resolve_available_day",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name in _NEW_BOOKING_AGENTS
            and _reply_reasks_something_just_given(reply, state)
        ),
        lambda reply, state: _reasked_information_correction_directive(reply, state),
        "reply asked for a doctor/branch/day that the patient's own last message "
        "already supplied",
    ),
    (
        lambda reply, state, agent_name: _reply_asks_for_a_slot_already_locked_in(reply, state),
        lambda reply, state: _selected_slot_correction_directive(reply, state),
        "reply asked for the appointment time, but a slot has already been locked in "
        "via select_appointment_slot for this booking",
    ),
    (
        lambda reply, state, agent_name: _reply_reasks_day_patient_already_named(reply, state),
        lambda reply, state: _DAY_ALREADY_NAMED_CORRECTION_DIRECTIVE,
        "reply re-asked which day, but the patient's own last message already named "
        "a day matching one in the just-shown day list",
    ),
    (
        lambda reply, state, agent_name: (
            # `concierge` is the full legacy agent - it carries the SAME
            # cancel/reschedule prompt sections and tools as those two
            # specialists (see agents/registry.py), and the router keeps
            # a weak, single-word cue like "تعديل" on whichever agent was
            # already active rather than switching - so `concierge` can
            # and does independently start this exact flow and produce
            # this exact bug. CONFIRMED REAL PRODUCTION FAILURE: routing
            # never switched off `concierge` (no strong enough cue in
            # "تعديل" alone), and `concierge` itself skipped straight to
            # "نكمل تعديل موعدك على نفس رقم الواتساب ده؟" - the same
            # violation this guard was written for, just from a
            # different agent than originally gated on.
            agent_name in ("cancel", "reschedule", "concierge")
            and _reply_skips_reference_or_phone_question(reply, state)
        ),
        lambda reply, state: _REFERENCE_OR_PHONE_CORRECTION_DIRECTIVE,
        "cancel/reschedule reply asked a phone-specific question (same WhatsApp "
        "number? / send your phone number) without STEP 1's reference-or-phone "
        "question ever being asked, and the patient never supplied either "
        "themselves",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name in ("cancel", "reschedule", "concierge")
            and _reply_reoffers_reference_after_phone_chosen(reply, state)
        ),
        lambda reply, state: _REFERENCE_REOFFER_CORRECTION_DIRECTIVE,
        "reply re-offered the booking reference as an identification method "
        "even though the patient already specifically chose to identify by "
        "phone at STEP 1",
    ),
    (
        lambda reply, state, agent_name: _reply_reasks_identity_after_verification(reply, state),
        lambda reply, state: _IDENTITY_REASK_CORRECTION_DIRECTIVE,
        "reply asked for a phone number or booking reference even though this "
        "session already has a verified phone AND lookup_appointment already "
        "found a real booking earlier in the conversation",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name in ("booking", "concierge")
            and _reply_asks_same_number_before_booking_ready(reply, state)
        ),
        lambda reply, state: _premature_same_number_directive(reply, state),
        "new-booking reply asked STEP NB6's same-WhatsApp-number question before "
        "a doctor was confirmed and a time slot was selected in the booking "
        "session",
    ),
    (
        lambda reply, state, agent_name: _reply_wrongly_scope_refuses_after_otp_failure(reply, state),
        lambda reply, state: _OTP_FAILURE_SCOPE_REFUSAL_CORRECTION_DIRECTIVE,
        "reply used the out-of-scope refusal right after a failed OTP attempt, "
        "when the patient's message should have been treated as another OTP "
        "retry instead",
    ),
    (
        lambda reply, state, agent_name: _reply_asks_for_a_phone_already_known(reply, state),
        lambda reply, state: _PHONE_ALREADY_KNOWN_CORRECTION_DIRECTIVE,
        "reply asked for a phone number (or a booking reference instead) right after "
        "the patient agreed to proceed on the channel number the service already has",
    ),
    (
        lambda reply, state, agent_name: (
            (agent_name == "medical" or _in_medical_guidance_handoff(state))
            and _medical_reply_offers_unrelated_specialty(reply, state)
        ),
        lambda reply, state: _UNRELATED_SPECIALTY_CORRECTION_DIRECTIVE,
        "medical-guidance reply offered a specialty that does not treat the "
        "body part the patient named",
    ),
    (
        lambda reply, state, agent_name: (
            (agent_name == "medical" or _in_medical_guidance_handoff(state))
            and _medical_reply_names_two_specialties(reply, state)
        ),
        lambda reply, state: _TWO_SPECIALTIES_CORRECTION_DIRECTIVE,
        "medical-guidance reply advised one specialty and offered an appointment "
        "in a different one",
    ),
    (
        lambda reply, state, agent_name: (
            (agent_name == "medical" or _in_medical_guidance_handoff(state))
            and _medical_reply_asks_which_specialty(reply, state)
        ),
        lambda reply, state: _SPECIALTY_CHOICE_CORRECTION_DIRECTIVE,
        "medical-guidance reply asked the patient to choose a specialty instead of "
        "showing the doctors",
    ),
    (
        lambda reply, state, agent_name: (
            agent_name == "medical" and _medical_reply_missing_not_a_diagnosis(reply, state)
        ),
        lambda reply, state: _NOT_A_DIAGNOSIS_CORRECTION_DIRECTIVE,
        "medical-guidance reply steered the patient to a specialty/doctor without "
        "the required 'not a diagnosis' clause",
    ),
    (
        lambda reply, state, agent_name: _reply_asks_generic_branch_after_doctor(reply, state),
        lambda reply, state: _DOCTOR_SCHEDULE_INSTEAD_OF_BRANCHES_CORRECTION,
        "reply asked a generic 'which branch?' after a doctor was settled, instead of "
        "showing that doctor's own schedule",
    ),
    (
        lambda reply, state, agent_name: _reply_offers_booking_at_empty_branch(reply, state),
        lambda reply, state: _EMPTY_BRANCH_BOOKING_OFFER_CORRECTION,
        "reply offered a booking at a branch the tools reported has no doctors",
    ),
    (
        lambda reply, state, agent_name: (
            (agent_name == "medical" or _in_medical_guidance_handoff(state))
            and _reply_dumps_specialty_catalogue(reply, state)
        ),
        lambda reply, state: _SPECIALTY_CATALOGUE_CORRECTION_DIRECTIVE,
        "medical-guidance reply printed the specialty catalogue for the patient to pick from",
    ),
    (
        lambda reply, state, agent_name: _reply_recommends_medication(reply, state),
        lambda reply, state: _MEDICATION_CORRECTION_DIRECTIVE,
        "reply named or suggested a medication",
    ),
    (
        lambda reply, state, agent_name: _reply_denies_availability_without_lookup(reply, state),
        lambda reply, state: _AVAILABILITY_DENIAL_CORRECTION_DIRECTIVE,
        "reply said a doctor has no available appointments while no availability "
        "tool has run in this conversation",
    ),
    (
        lambda reply, state, agent_name: _reply_denies_a_branch_the_tools_offered(reply, state),
        lambda reply, state: _BRANCH_DENIAL_CORRECTION_DIRECTIVE,
        "reply said a branch had nothing available, while a tool result in this "
        "conversation lists that branch as an option and none reported it empty",
    ),
    (
        lambda reply, state, agent_name: _reply_wrongly_rejects_full_name(reply, state),
        lambda reply, state: _NAME_REJECTION_CORRECTION_DIRECTIVE,
        "reply asked the patient to re-send a full name that already had at "
        "least two parts",
    ),
    (
        lambda reply, state, agent_name: _reply_offers_cancellation_without_lookup(reply, state),
        lambda reply, state: _CANCELLATION_CORRECTION_DIRECTIVE,
        "reply asked the patient to confirm cancelling an appointment, but no "
        "appointment has ever been looked up in this conversation",
    ),
    (
        lambda reply, state, agent_name: _reply_invents_availability(reply, state),
        lambda reply, state: _AVAILABILITY_CORRECTION_DIRECTIVE,
        "reply stated an appointment date/time that NO availability tool returned "
        "- this is a fabricated appointment",
    ),
    (
        lambda reply, state, agent_name: _reply_offers_unauthorized_gynecology(reply, state, agent_name),
        lambda reply, state: _GYN_CORRECTION_DIRECTIVE,
        "reply named نساء وتوليد with no patient-raised pregnancy/gynaecological "
        "signal in this conversation",
    ),
    (
        lambda reply, state, agent_name: _reply_fabricates_complaint_submission(reply, state),
        lambda reply, state: _COMPLAINT_CORRECTION_DIRECTIVE,
        "reply confirmed a complaint was filed but send_complaint_email was never "
        "called successfully in this conversation",
    ),
    (
        lambda reply, state, agent_name: _reply_fabricates_doctor_not_found_stop(reply, state),
        lambda reply, state: _DOCTOR_NOT_FOUND_STOP_CORRECTION_DIRECTIVE,
        "reply stopped the complaint over a doctor/branch name that the patient "
        "never actually gave",
    ),
    (
        lambda reply, state, agent_name: _reply_derails_complaint_into_handoff_offer(reply, state),
        lambda reply, state: _COMPLAINT_HANDOFF_DERAIL_CORRECTION_DIRECTIVE,
        "reply offered a customer-service handoff right after the patient said "
        "there was nothing more to add to their complaint, instead of continuing "
        "to STEP C3",
    ),
    (
        lambda reply, state, agent_name: _reply_fabricates_handoff(reply, state),
        lambda reply, state: _HANDOFF_CORRECTION_DIRECTIVE,
        "reply confirmed a human handoff but request_human_handoff was never "
        "raised successfully in this conversation",
    ),
    (
        lambda reply, state, agent_name: _reply_reoffers_doctor_roster_after_confirming_one(reply, state),
        lambda reply, state: _DOCTOR_ROSTER_CORRECTION_DIRECTIVE,
        "reply confirmed a doctor then offered the doctor roster again in the same message",
    ),
    (
        lambda reply, state, agent_name: bool(_find_invented_doctors(reply, state)),
        lambda reply, state: _INVENTED_DOCTORS_CORRECTION_DIRECTIVE,
        "reply listed doctor(s) that appear in no tool result in this conversation",
    ),
    (
        lambda reply, state, agent_name: bool(_find_invented_branches(reply, state)),
        lambda reply, state: _BRANCH_CORRECTION_DIRECTIVE.format(
            names=", ".join(_find_invented_branches(reply, state))
        ),
        "reply named branch(es) that appear in NO tool result and are not "
        "configured for this client",
    ),
)


def _is_answering_a_list(messages: list) -> bool:
    """True when the patient's latest message is a pick from a list the
    assistant just showed them, rather than a fresh open question.

    Used ONLY to choose the wording of the interim "please wait" line
    (see progress._LIST_LOOKUP_TOOLS) - it never affects routing, tool
    choice, or the reply itself.

    Deliberately narrow: a bare position ("2", "٢", "رقم 2") or a short
    reply, AND the assistant's own previous message actually contained a
    numbered list. Both conditions have to hold, so an open question
    that happens to be short is not mistaken for a selection.
    """

    history = list(messages or [])

    last_human = None
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "type", None) == "human":
            last_human = index
            break

    if last_human is None:
        return False

    content = getattr(history[last_human], "content", "")
    text = content if isinstance(content, str) else str(content)

    if not text.strip() or len(text.strip()) > 30:
        return False

    if tools._extract_selection_number(text) is None:
        return False

    for msg in reversed(history[:last_human]):
        if getattr(msg, "type", None) != "ai":
            continue
        previous = getattr(msg, "content", "")
        previous = previous if isinstance(previous, str) else str(previous)
        if not previous.strip():
            continue
        # A numbered list, in either the emoji badges this project emits
        # or plain digits.
        return bool(
            re.search(r"^\s*(?:[0-9\u0660-\u0669]{1,2}[.)]|[0-9]\uFE0F?\u20E3|\U0001F51F)", previous, re.MULTILINE)
        )

    return False


_ABANDONED_BOOKING_RESET_DIRECTIVE = (
    "============================================================\n"
    "A PREVIOUS BOOKING ATTEMPT IS STILL HALF-FINISHED\n"
    "============================================================\n"
    "Earlier in this conversation a doctor and/or a branch were settled "
    "for a booking that never completed - the patient changed their "
    "mind, or moved on to something else. Those choices are still "
    "remembered, and they will silently narrow every doctor, branch, day "
    "and slot lookup you make from here.\n\n"
    "The patient has now started a NEW request. Before anything else "
    "this turn, call `reset_booking_session` - then continue normally. "
    "Confirmed real production failure: a branch left over from an "
    "abandoned booking made an entire specialty the clinic genuinely "
    "staffs come back as \"no doctors available\", twice in a row, "
    "including after the patient explicitly asked to look across the "
    "whole hospital.\n\n"
)


def _build_abandoned_booking_directive(messages: list, session_id: str) -> str:
    """Fires when a booking session still holds a doctor/branch from a
    flow the patient has clearly moved on from.

    Deliberately narrow. It requires ALL of:
      - something actually remembered (a doctor or a branch), and
      - the patient's latest message being a fresh human turn, and
      - no booking tool having run since the patient walked away.

    A patient who merely detours mid-booking (asks the opening hours,
    then comes back) does NOT trip this: the booking specialist still
    owns those turns, and this directive is only ever added for
    specialists that cannot place a booking at all.
    """

    if not messages or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    if not (session.get("doctor_id") or session.get("branch_id")):
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    if not isinstance(messages[-1], _HumanMessage):
        return ""

    return _ABANDONED_BOOKING_RESET_DIRECTIVE


# The bare words a patient uses to ANSWER "by specialty or by doctor?"
# and "a particular branch, or shall I show you them?" - i.e. the word
# itself, with no name attached.
#
# NOTE: "دكتور"/"doctor" is deliberately NOT in this regex any more - it
# has its own, different handling below (_BARE_DOCTOR_ANSWER_RE), because
# the product decision for that one path changed: ask for the specific
# doctor's name first, rather than dumping the whole roster immediately.
# "تخصص" and "فرع" are unaffected and still show their list right away.
_BARE_ENTITY_ANSWER_RE = re.compile(
    r"^\s*(?:"
    r"(?:ال)?(?:تخصص|تخصصات|قسم|اقسام|أقسام)|"
    r"(?:ال)?(?:فرع|فروع)|"
    r"specialt(?:y|ies)|departments?|branch(?:es)?"
    r")\s*[.!؟?،,]*\s*$",
    re.IGNORECASE,
)

_BARE_ENTITY_ANSWER_DIRECTIVE = (
    "============================================================\n"
    "THEY ANSWERED WITH THE CATEGORY, NOT A NAME - SHOW THE LIST\n"
    "============================================================\n"
    "The patient's reply is the bare WORD (\"تخصص\", \"فرع\", "
    "\"specialty\", \"branch\") answering the choice you just offered "
    "them. It is an ANSWER, not the name of anything.\n\n"
    "They are telling you which way they want to go, and asking you to "
    "show them the options. So SHOW THEM, this turn:\n"
    "  - \"تخصص\"/\"specialty\" -> call `list_specialties` and show the "
    "numbered list of specialties.\n"
    "  - \"فرع\"/\"branch\"    -> call `list_branches_for_specialty` and "
    "show the numbered list of branches.\n\n"
    "Do NOT reply by asking for a name (\"أي تخصص؟\"). The patient does "
    "not know which specialties/branches exist - that is exactly why "
    "they asked you to show them. Asking them to name one is asking for "
    "information only the system has, and it wastes a turn.\n\n"
    "Never fuzzy-match the bare word itself against a real specialty or "
    "branch name. Confirmed real production failure: the bare word "
    "\"فرع\" was matched to an actual branch the patient had never named "
    "or seen, and the whole booking then ran against the wrong one.\n\n"
    "(If instead the bare word was \"دكتور\"/\"doctor\", that is handled "
    "separately - see the DOCTOR PATH directive.)\n\n"
)


# ==========================================================
# "دكتور" answered bare -> ASK FOR THE NAME, don't list yet
# ==========================================================
#
# Product decision: the patient choosing the DOCTOR PATH with the bare
# word "دكتور"/"doctor" (no name attached) should be asked to type the
# specific doctor's name FIRST, rather than immediately being shown the
# entire roster. The full list is still available - but only once the
# patient has said they don't have a particular doctor in mind, or asks
# outright to see everyone. This keeps the roster from being dumped on
# every patient who simply meant "let me pick by doctor, not by
# specialty" and may well already know who they want.
_BARE_DOCTOR_ANSWER_RE = re.compile(
    r"^\s*(?:ال)?(?:دكتور|دكتوره|دكتورة|دكاتره|دكاترة|طبيب|طبيبه|طبيبة|"
    r"اطباء|أطباء|doctors?)\s*[.!؟?،,]*\s*$",
    re.IGNORECASE,
)

_BARE_DOCTOR_ANSWER_DIRECTIVE = (
    "============================================================\n"
    "THEY CHOSE THE DOCTOR PATH - ASK FOR THE NAME, DO NOT LIST YET\n"
    "============================================================\n"
    "The patient's reply is the bare word (\"دكتور\"/\"doctor\") "
    "answering the specialty-vs-doctor choice you just offered them. It "
    "means they want to pick BY DOCTOR - it does NOT mean \"show me "
    "everyone\".\n\n"
    "Ask exactly ONE question this turn, and do NOT call any doctor-list "
    "tool yet - expressed in this clinic's own configured dialect (see "
    "the LANGUAGE & DIALECT rule), or in English if the patient is "
    "currently writing English - the Arabic below is only an "
    "illustration of what to ask, not fixed wording to force:\n"
    "    من فضلك اكتب اسم الدكتور اللي حابب تحجز معاه\n\n"
    "  - If they then give a NAME -> treat it exactly like any other "
    "named doctor: call `match_entity_for_booking` and continue the "
    "normal flow from STEP NB2.\n"
    "  - If they say they don't know one, or ask you to just show "
    "everyone (\"معرفش\", \"مش عارف\", \"ما اعرف\", \"اعرض كل "
    "الدكاتره\", \"ورينى الكل\", \"I don't know\", \"show me all\") -> "
    "THAT is the moment to call `find_available_doctors` (or "
    "`match_entity_for_booking` in list mode) and show the full "
    "numbered roster - never before it.\n\n"
    "Do not show the doctor list on this turn just because the bare "
    "word \"دكتور\" was said - that only tells you WHICH PATH they "
    "chose, not that they want the whole roster dumped on them.\n\n"
)


def _build_bare_doctor_answer_directive(messages: list) -> str:
    """Fires when the patient's latest message is just the bare word
    ("دكتور"/"doctor"), answering a specialty-vs-doctor choice the
    assistant just offered - see _BARE_DOCTOR_ANSWER_DIRECTIVE for the
    ask-for-a-name flow this should trigger instead of listing
    immediately."""

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _BARE_DOCTOR_ANSWER_RE.match(text.strip()):
        return ""

    return _BARE_DOCTOR_ANSWER_DIRECTIVE


# The follow-up turn: the assistant just asked "من فضلك اكتب اسم
# الدكتور اللي حابب تحجز معاه" (or an equivalent), and the patient's
# reply says they don't know one / wants to see everyone instead of
# naming one.
_DONT_KNOW_DOCTOR_NAME_RE = re.compile(
    r"معرفش|مش\s*عارف|مش\s*عارفه|ما\s*اعرف|لا\s*اعرف|مش\s*عارفة|"
    r"اعرض\s*(?:كل|جميع)?\s*(?:ال)?دكاتره|"
    r"ورين[ىي]\s*(?:كل|جميع)?\s*(?:ال)?دكاتره|"
    r"كل\s*(?:ال)?دكاتره\s*المتاح|وريني\s*الكل|اعرض\s*الكل|"
    r"don'?t\s*know|show\s*(?:me\s*)?all|show\s*(?:me\s*)?every"
)

# Matches the exact question _BARE_DOCTOR_ANSWER_DIRECTIVE tells the
# model to ask, loosely enough to survive minor rewording.
_ASKED_FOR_DOCTOR_NAME_RE = re.compile(r"اسم\s*(?:ال)?دكتور[^\n]{0,40}تحجز")

_SHOW_ALL_DOCTORS_AFTER_ASK_DIRECTIVE = (
    "============================================================\n"
    "THEY DON'T KNOW A DOCTOR'S NAME - SHOW THE FULL LIST NOW\n"
    "============================================================\n"
    "You just asked the patient to name a specific doctor, and their "
    "reply says they don't know one / asks you to just show everyone. "
    "This IS the moment to show the full roster: call "
    "`find_available_doctors` (or `match_entity_for_booking` in list "
    "mode) and present the numbered list of every currently available "
    "doctor. Do not ask them to try naming one again - they already "
    "told you they can't, and repeating the same question is a dead "
    "end for them.\n\n"
)


def _build_show_all_doctors_after_ask_directive(messages: list) -> str:
    """Fires the turn right after the assistant asked the patient to
    name a specific doctor (per _BARE_DOCTOR_ANSWER_DIRECTIVE), when the
    patient's reply says they don't know one / asks to see everyone
    instead of naming one."""

    history = list(messages or [])
    if len(history) < 2:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage

    last = history[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    if not _DONT_KNOW_DOCTOR_NAME_RE.search(_norm_ar(text)):
        return ""

    previous_ai = None
    for msg in reversed(history[:-1]):
        if isinstance(msg, _AIMessage):
            previous_ai = msg
            break
    if previous_ai is None:
        return ""

    previous_content = getattr(previous_ai, "content", "")
    previous_text = previous_content if isinstance(previous_content, str) else str(previous_content)
    if not _ASKED_FOR_DOCTOR_NAME_RE.search(_norm_ar(previous_text)):
        return ""

    return _SHOW_ALL_DOCTORS_AFTER_ASK_DIRECTIVE


_BOOKING_INTENT_RE = re.compile(
    r"احجز|اح جز|حجز|احجزلي|عايز.{0,12}حجز|عاوز.{0,12}حجز|ابغ[يى].{0,12}حجز|"
    r"اب[يى].{0,12}حجز|موعد|ميعاد|book|appointment|reserve"
)


_SERVICE_CHOSEN_DIRECTIVE = (
    "============================================================\n"
    "A SERVICE IS ALREADY CHOSEN - FIND ITS DOCTORS, DON'T RESTART\n"
    "============================================================\n"
    "The patient has picked a SERVICE and said they want to book it. "
    "That choice already tells you what they need - so do NOT ask "
    "\"تحب تبدأ بالتخصص ولا بالدكتور؟\", and do NOT show a specialty "
    "list. Both throw away a decision they have already made and "
    "restart the flow from zero.\n\n"
    "IF THE BRANCH THEY ARE LOOKING AT HAS NO BOOKABLE DOCTOR, the "
    "service is still bookable elsewhere - do not dead-end them. Call "
    "`find_branches_offering_service` and answer with WHERE they can "
    "get it:\n"
    "  - \"found\": say plainly that this branch can't book it right "
    "now, then list the branches that CAN, by name, emoji-numbered "
    "(no doctor names yet), and ask which one they'd like. Once they "
    "pick, that branch becomes the booking's branch and you continue "
    "normally.\n"
    "  - \"not_found\": only then say nobody offers this service right "
    "now, and offer to help with something else.\n"
    "Never announce a branch as offering the service unless this tool "
    "returned it - that fact is never yours to infer.\n\n"
    "Call `find_available_doctors` with `service_name` set to the "
    "service they chose (its name, or the number they picked from the "
    "service list), and `branch_name` set to the branch if one is "
    "already settled. That sends the service's real id to the doctors "
    "lookup, so only doctors who actually PROVIDE that service come "
    "back. Show them as a numbered list and ask ONE question: which "
    "doctor.\n\n"
    "YOU DO NOT NEED A SPECIALTY. `specialty_ids` is optional - omit it "
    "entirely. Do not call `list_specialties`, do not try to work out "
    "which specialty the service belongs to, and above all do not tell "
    "the patient you need one first. The service IS the filter.\n"
    "  - \"not_found\"/\"not_found_in_branch\": say plainly that nobody "
    "provides this service at this branch right now, and offer the "
    "branches that do.\n"
    "  - \"service_not_matched\": show the branch's service list again "
    "and let them pick from it - never guess.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURES, twice on this exact step: the "
    "patient was shown \"فحص النظر\", said \"اه\" to booking it, and got "
    "\"تحب تبدأ بالتخصص ولا بالدكتور؟\" with a four-item specialty list; "
    "then, with the branch settled too, \"راح أحتاج أعرف التخصص المناسب "
    "الأول عشان أقدر أجيب لك الدكاترة المتاحين\" - a prerequisite that "
    "does not exist, invented one step from finishing.\n\n"
)


_SERVICE_BOOKING_OFFER_RE = re.compile(
    r"تحجز\s*موعد\s*ل|تحجز\s*ل|حجز\s*موعد\s*ل|book\s*(?:an\s*)?appointment\s*for"
)


_SHOW_DOCTORS_REQUEST_RE = re.compile(
    r"(?:ال)?دكاتره|(?:ال)?دكاتره\s*المتاح|(?:ال)?اطباء|(?:ال)?دكتوره|"
    r"doctors?|physicians?"
)


def _build_doctors_scope_directive(messages: list, session_id: str) -> str:
    """Fires when the patient asks to see the doctors while a specific
    branch is the one under discussion - and scopes the roster to that
    branch.

    CONFIRMED REAL PRODUCTION FAILURE: فرع الدقي was the branch being
    discussed, the patient said "قولي كل الدكاتره المتاحه", and the
    reply listed all EIGHT doctors in the hospital - including several
    who do not work at الدقي at all. The lookup ran with branch_ids=None
    because the branch had been chosen through the INFO flow, so the
    booking session's own `branch_id` was still empty and nothing scoped
    the query.

    The rule the patient actually expects, and what this enforces:
      - a branch IS in play -> show THAT branch's doctors;
      - no branch in play  -> show every available doctor.
    """

    if not messages or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    branch_name = (
        session.get("branch_display_name")
        or session.get("info_branch_name")
    )
    if not branch_name:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _SHOW_DOCTORS_REQUEST_RE.search(_norm_ar(text)):
        return ""

    return (
        "============================================================\n"
        "DOCTORS MEANS THIS BRANCH'S DOCTORS\n"
        "============================================================\n"
        f"The branch under discussion is {branch_name}. When the patient "
        "asks for the available doctors here - including \"كل الدكاترة "
        "المتاحة\" - they mean everyone available AT THIS BRANCH, not "
        "every doctor in the hospital.\n\n"
        f"Call `find_available_doctors` with `branch_name=\"{branch_name}\"` "
        "and show what it returns, numbered, in its exact order. Do not "
        "widen the search, and do not pass `all_branches=True` - the "
        "word \"كل\" here means \"all of them at this branch\", not "
        "\"search the whole hospital\".\n\n"
        "HOW TO WRITE IT:\n"
        f"  - Head the list with the branch: \"الدكاترة المتاحين في "
        f"{branch_name}:\". NEVER write \"في كل الفروع\" - the list is "
        "scoped to ONE branch, so saying \"all branches\" is simply "
        "false.\n"
        f"  - Do NOT repeat \"في {branch_name}\" after every single "
        "doctor's name. The branch is already in the heading; adding it "
        "to all eight lines is noise.\n"
        "  - Do not narrate what you are doing (\"بوريك...\", \"خليني "
        "أعرض لك...\"). Just show the list, then ask ONE question: "
        "which doctor.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: the reply opened \"بوريك "
        "الدكاترة المتاحين الحين في كل الفروع:\" and then repeated \"في "
        "فرع الدقي\" on every line - narrating, mislabelling a "
        "single-branch list as hospital-wide, and repeating the branch "
        "four times over.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: with فرع الدقي under "
        "discussion, this request returned all eight hospital doctors, "
        "several of whom do not work at الدقي - so the patient could "
        "pick someone unreachable at the branch they had chosen.\n\n"
        "(Only when NO branch has been chosen at all does an unscoped, "
        "hospital-wide roster answer this question.)\n\n"
    )


_SERVICE_WORDED_REQUEST_RE = re.compile(
    # "خدمة" itself counts: patients often name the service by saying
    # the word - "خدمه اخصائي تغذيه" - and leaving it out meant a
    # repeat of the request went unrecognised a second time.
    r"جلس[هة]|استشار[هة]|إستشار[هة]|كشف|فحص|برنامج|تحليل|اشع[هة]|أشع[هة]|"
    r"جلسات|خدم[هة]|الخدم[هة]|"
    r"session|consultation|checkup|check-up|screening|programme|program|service"
)


_BARE_NEGATION_RE = re.compile(
    r"^\s*(?:لا|لأ|لاء|ﻻ|مش|مو|ما|لا\s*شكرا|لا\s*مش|مش\s*عاوز[هة]?|مش\s*عايز[هة]?|"
    r"ما\s*ابي|ما\s*ابغى|ما\s*اريد|مو\s*مناسب|مش\s*مناسب|غير\s*مناسب|"
    r"no|nope|nah|not\s*this|doesn'?t\s*work)"
    r"\s*[.!؟?،,]*\s*$",
    re.IGNORECASE,
)


# A refusal that also says WHAT is refused - "لا مش مناسب التلات دا".
# NOTE: no \b between the tokens. Arabic letters are all "word"
# characters, so \b never matches between "لا" and "مش" and the pattern
# silently fails - which is exactly how the confirmed failure below got
# through a first version of this regex.
_LEADING_REFUSAL_RE = re.compile(
    r"^\s*(?:لا|لأ|مش|مو|ما)[\s،,]*"
    r"(?:لا|مش|مو|ما)?[\s،,]*"
    r"(?:مناسب|يناسب|عاوز|عايز|عاوزه|عايزه|ابي|ابغى|اريد|ينفع|تمام|كويس|حلو|"
    r"no|not)"
)


def _build_negation_directive(messages: list) -> str:
    """Fires when the patient's whole message is a refusal ("لا", "مش
    مناسب", "no").

    A bare "no" is an answer, and it is answering whatever was just
    offered. The reply must move AWAY from that thing - it must never
    carry on as though the answer had been yes.

    CONFIRMED REAL PRODUCTION FAILURE: the patient replied "لا", and the
    next message was "أقرب موعد متاح عند رانيا عبد الرحمن في Al Nozha:
    الثلاثاء 01/09/2026 ... هل يناسبك هذا اليوم؟" - the refusal was
    swallowed and the same doctor was pushed forward regardless."""

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    folded = _norm_ar(text)

    # A refusal does not have to be bare. "لا مش مناسب التلات دا" is a
    # refusal that also names WHAT is being refused, and it is more
    # common than a plain "لا" - people say what didn't work.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: "لا مش مناسب التلات دا" did not
    # match the bare pattern, so nothing fired, and the alternatives
    # offered back still included Tuesday - the very day just rejected.
    refuses = bool(
        _BARE_NEGATION_RE.match(folded)
        or _LEADING_REFUSAL_RE.match(folded)
    )

    if not refuses:
        return ""

    previous_ai = None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, _AIMessage):
            previous_content = getattr(msg, "content", "")
            previous_text = (
                previous_content if isinstance(previous_content, str) else str(previous_content)
            )
            if previous_text.strip():
                previous_ai = previous_text.strip()
                break

    if not previous_ai:
        return ""

    return (
        "============================================================\n"
        "THEY SAID NO - DO NOT CARRY ON AS IF THEY SAID YES\n"
        "============================================================\n"
        "The patient's entire message is a refusal. It answers the "
        "question you just asked, which was:\n"
        f"    \"{previous_ai[:300]}\"\n\n"
        "Whatever that question offered - this doctor, this day, this "
        "time, this branch - they have declined it. Your reply must "
        "move AWAY from it:\n"
        "  - Asked whether to book/continue on the SAME WHATSAPP NUMBER "
        "(\"نكمل الحجز على نفس رقم الواتساب ده؟\" or similar) and they "
        "said no -> this is NOT a refusal of the day/time/doctor, even "
        "if a booking confirmation or an appointment detail sits right "
        "above that question in the same message. Your entire reply "
        "must ask for the alternative phone number (or booking "
        "reference) - see the CHANNEL IDENTITY instructions elsewhere "
        "in this prompt for the exact next step. Do NOT reopen the "
        "day/time/doctor choice, and do NOT call any availability tool.\n"
        "    CONFIRMED REAL PRODUCTION FAILURE: right after the Saturday "
        "slot was confirmed, the assistant asked \"نكمل الحجز على نفس "
        "رقم واتساب ده؟\"; the patient said \"لا\"; the reply then "
        "offered a DIFFERENT day (Wednesday) instead of asking for the "
        "other phone number - the day/time had already been settled and "
        "was never what \"لا\" was answering.\n"
        "  - Offered a DAY or a TIME and they said no -> show the OTHER "
        "days/times that are actually available (call the tool again "
        "with the next offset), never the same one reworded.\n"
        "    THE REFUSED DAY MUST NOT APPEAR IN THAT LIST AT ALL. If "
        "they named it (\"مش مناسب التلات\"), leave it out of the "
        "alternatives entirely - re-offering it as one of the options "
        "ignores what they just told you, and they can pick it back by "
        "accident. CONFIRMED REAL PRODUCTION FAILURE: a patient "
        "rejected Tuesday and the very next message offered \"1️⃣ "
        "الأحد 2️⃣ الاثنين 3️⃣ الثلاثاء\" - the rejected day still "
        "on the menu.\n"
        "  - Offered a DOCTOR and they said no -> offer the other "
        "available doctors, not that doctor's schedule.\n"
        "  - Offered a BRANCH and they said no -> the other branches.\n"
        "  - Asked whether to continue at all and they said no -> stop "
        "warmly and ask if there's anything else, without pushing.\n\n"
        "Do NOT re-present the same thing with different wording, and "
        "do NOT advance to the next step of the flow for the thing they "
        "just refused. If nothing else is available, say that plainly "
        "instead of quietly re-offering what they turned down.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: the patient replied \"لا\" "
        "and the very next message was \"أقرب موعد متاح عند رانيا عبد "
        "الرحمن ... هل يناسبك هذا اليوم؟\" - the refusal was ignored "
        "and the same doctor pushed forward anyway.\n\n"
    )


def _build_service_named_directive(messages: list, session_id: str) -> str:
    """Fires when the patient's booking request names a SERVICE rather
    than a specialty or a doctor ("عاوزة احجز جلسة أخصائي تغذية").

    A service is a complete answer to "what do you want booked" - more
    specific than a specialty, in fact - so the specialty-vs-doctor
    question does not apply. Treating it as a nameless booking request
    throws the information away and asks them to start again in terms
    they did not choose.

    CONFIRMED REAL PRODUCTION FAILURE: "عاوزه احجز جلسه اخصائي تغذيه"
    was answered with "نكمل الحجز على نفس رقم الواتساب ده؟", then "تحب
    تبدأ بالتخصص ولا بالدكتور؟", and when the patient repeated "خدمه
    اخصائي تغذيه" it still came back with "وش التخصص اللي حابة تحجزين
    فيه؟" - three turns, the service named twice, never once acted on.
    """

    if not messages or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    if session.get("doctor_id") or session.get("service_id"):
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    folded = _norm_ar(text)

    if not _SERVICE_WORDED_REQUEST_RE.search(folded):
        return ""

    return (
        "============================================================\n"
        "THEY NAMED A SERVICE - BOOK THAT, DON'T ASK FOR A SPECIALTY\n"
        "============================================================\n"
        "The patient's message names a SERVICE (a جلسة / استشارة / فحص / "
        "كشف / برنامج), not a specialty and not a doctor. That is "
        "already a complete answer to what they want booked - more "
        "specific than a specialty, not less.\n\n"
        "So do NOT ask \"تحب تبدأ بالتخصص ولا بالدكتور؟\", and do NOT "
        "ask \"وش التخصص اللي تحب تحجز فيه؟\". They told you. Asking "
        "again makes them restate it in words they did not choose.\n\n"
        "Call `find_available_doctors` with `service_name` set to what "
        "they said (leave `specialty_ids` out entirely - a service does "
        "not need one) and show the doctors who provide it, numbered, "
        "then ask ONE question: which doctor.\n"
        "  - If a branch is already settled, pass `branch_name` too.\n"
        "  - \"not_found\"/\"not_found_in_branch\": nobody provides it at "
        "that branch - use `find_branches_offering_service` to say where "
        "it IS available.\n"
        "  - \"service_not_matched\": you could not resolve what they "
        "named. Only then ask them to clarify - and do it by showing "
        "real services to pick from, never by falling back to the "
        "specialty question.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: \"عاوزه احجز جلسه اخصائي "
        "تغذيه\" got \"نكمل الحجز على نفس رقم الواتساب ده؟\" -> \"تحب "
        "تبدأ بالتخصص ولا بالدكتور؟\" -> and after the patient repeated "
        "\"خدمه اخصائي تغذيه\", still \"وش التخصص اللي حابة تحجزين "
        "فيه؟\". The service was named twice and acted on zero times.\n\n"
    )


def _build_service_chosen_directive(messages: list, session_id: str) -> str:
    """Fires when a service is what the patient is booking - the point
    where the flow used to reset to the specialty-vs-doctor question.

    Three ways a service counts as chosen, because it does not always
    arrive through `list_branch_services`:
      - a service list was shown and is remembered, or
      - a service id is already on the session, or
      - the assistant's own previous message offered to book a NAMED
        service ("حابب تحجز موعد لفحص النظر في فرع الدقي؟") and the
        patient agreed.
    That last case is the confirmed real failure: the service had been
    described from the knowledge base rather than picked off a list, so
    nothing was remembered, the directive never fired, and the reply to
    "اه" was "راح أحتاج أعرف التخصص المناسب الأول... تحب تبدأ بالتخصص
    ولا بالدكتور؟" - inventing a prerequisite and restarting the flow.
    """

    if not messages or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    if session.get("doctor_id"):
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    last_list = session.get("last_list") or {}
    service_shown = last_list.get("entity_type") == "service" and last_list.get("items")

    service_offered = False
    if not (service_shown or session.get("service_id")):
        for msg in reversed(messages[:-1]):
            if not isinstance(msg, _AIMessage):
                continue
            previous = getattr(msg, "content", "")
            previous = previous if isinstance(previous, str) else str(previous)
            if previous.strip():
                service_offered = bool(_SERVICE_BOOKING_OFFER_RE.search(_norm_ar(previous)))
            break

    if not (service_shown or session.get("service_id") or service_offered):
        return ""

    return _SERVICE_CHOSEN_DIRECTIVE


def _build_day_pick_directive(messages: list, session_id: str) -> str:
    """Resolves a bare number that is picking a DAY from the list just
    shown, and states the answer in the system prompt.

    WHY: a positional pick is routinely answered with no tool call at
    all, so nothing verifies which option the number refers to. Days
    were also the one list this project never remembered, which left the
    model matching a digit against dates from memory.

    CONFIRMED REAL PRODUCTION FAILURE: the patient said Tuesday didn't
    suit them, was shown "1️⃣ الأحد 30/08 2️⃣ الاثنين 31/08 3️⃣ الثلاثاء
    01/09", replied "1" - and was booked onto الثلاثاء 01/09/2026: the
    third option, and the exact day they had just rejected."""

    if not messages or not session_id:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    position = tools._extract_selection_number(text)
    if position is None:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    last_list = session.get("last_list") or {}

    if last_list.get("entity_type") != "day":
        return ""

    items = last_list.get("items") or []
    if not (1 <= position <= len(items)):
        return (
            "============================================================\n"
            "THAT NUMBER IS OUTSIDE THE DAY LIST YOU SHOWED\n"
            "============================================================\n"
            f"You showed {len(items)} day(s); they replied \"{position}\". "
            "Say how many options there are and ask them to pick within "
            "it. Do NOT guess a day, and do NOT tell them the day "
            "doesn't exist - the number came from your own list.\n\n"
        )

    chosen = items[position - 1]
    chosen_date = chosen.get("date") or ""
    weekday = chosen.get("weekday_display") or chosen.get("weekday_name") or ""
    display = chosen.get("date_display") or chosen_date

    return (
        "============================================================\n"
        "WHICH DAY THEY PICKED - RESOLVED, DO NOT RE-DERIVE IT\n"
        "============================================================\n"
        f"Their reply is a pick from the day list you just showed. "
        f"Option {position} is:\n"
        f"    weekday : {weekday}\n"
        f"    date    : {display}  (ISO {chosen_date})\n\n"
        "That is the ONLY day this booking may continue with. Do not "
        "read the date off an earlier message, do not re-count the "
        "list, and do not substitute a different option - pass THIS "
        "date to the next tool and name THIS weekday in your reply.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: a patient who had just "
        "rejected Tuesday picked option 1 (Sunday) and was confirmed "
        "onto Tuesday - option 3 - because the number was matched from "
        "memory instead of from the list.\n\n"
    )


def _build_branch_pick_directive(messages: list, session_id: str) -> str:
    """Fires when the patient's latest message is a bare number picking
    from a branch list that was just shown, and resolves it HERE, in the
    system prompt, from the remembered list.

    WHY THIS EXISTS: a positional pick is routinely answered with NO
    tool call at all - the model already has the list in the
    conversation, so it just replies from memory. That means
    `match_entity_info` never runs, its `hasAvailableDoctors` flag never
    arrives, and `tools._note_info_branch_availability` never fires.

    CONFIRMED REAL PRODUCTION FAILURE: shown the six branches, the
    patient replied "1" (فرع المعادي, zero doctors) and the reply -
    written with no tool call - offered "...أو تحب أساعدك بحجز موعد
    فيه؟", inviting a booking that cannot exist. Every earlier fix for
    this lived behind a tool call, so none of them could reach this
    turn. The remembered list already holds the answer; this puts it in
    front of the model on the one turn it is needed."""

    if not messages or not session_id:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    position = tools._extract_selection_number(text)
    if position is None:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    last_list = session.get("last_list") or {}

    if last_list.get("entity_type") != "branch":
        return ""

    items = last_list.get("items") or []
    if not (1 <= position <= len(items)):
        return ""

    chosen = items[position - 1]
    name = chosen.get("altName") or chosen.get("name") or ""
    address = chosen.get("address") or ""

    if not name:
        return ""

    # A branch row can reach here from two places: `match_entity_info`'s
    # list (which carries `hasAvailableDoctors`) or a booking-flow list
    # like `list_branches_for_specialty` (which does not). Fall back to
    # the session note so an empty branch is recognised either way -
    # otherwise the same branch is treated as bookable purely because of
    # which tool happened to list it.
    branch_is_empty = chosen.get("hasAvailableDoctors") is False
    if not branch_is_empty and session.get("info_branch_no_doctors"):
        branch_is_empty = _norm_ar(session["info_branch_no_doctors"]) == _norm_ar(name)

    if branch_is_empty:
        # Remember it, so the NEXT turn ("I want to book there") is also
        # covered - see _build_empty_branch_booking_intent_directive.
        session["info_branch_no_doctors"] = name
        return (
            "============================================================\n"
            "THEY PICKED A BRANCH THAT HAS NO BOOKABLE DOCTOR\n"
            "============================================================\n"
            f"Their reply is a pick from the branch list you just showed: "
            f"option {position} is {name}"
            f"{(', address: ' + address) if address else ''}.\n\n"
            "DO THE WHOLE THING IN ONE MESSAGE. They picked a branch to "
            "find out about it - so tell them about it now. Call "
            "`list_branch_services` in THIS turn and reply with:\n"
            f"  1. The address of {name}.\n"
            "  2. Its services, numbered, straight from that tool.\n\n"
            "Do NOT ask \"تحب تعرف عن الخدمات المتوفرة في هذا الفرع؟\" "
            "and wait for a yes. That is a turn spent asking permission "
            "to answer the question they already asked. CONFIRMED REAL "
            "PRODUCTION FAILURE: picking a branch produced the address "
            "plus that question, and the services only arrived a turn "
            "later.\n\n"
            "Use `list_branch_services` for this - NOT "
            "`list_hospital_services` and NOT `answer_hospital_faq`. "
            "Those read the knowledge-base file, which has no per-branch "
            "information and returns the same generic list for every "
            "branch. CONFIRMED REAL PRODUCTION FAILURE: asked for فرع "
            "المعادي's services, the reply was the hospital-wide "
            "knowledge-base list verbatim.\n\n"
            "THIS BRANCH CANNOT BE BOOKED AT. Never offer an "
            "appointment here - not \"هل ترغب تحجز موعد في هذا الفرع؟\", "
            "not a doctor question, not anything that starts a booking "
            "for it. There is no doctor to book, so every such offer is "
            "a promise you cannot keep.\n\n"
            "End the message with ONE question that is honest and "
            "actually useful - the branch can't take bookings, so point "
            "at the ones that can:\n"
            f"    الفرع ده مفيهوش حجز حاليًا، تحب أعرض لك الفروع اللي "
            "فيها حجز؟\n"
            "If they say yes, call `find_branches_offering_service` (when "
            "a service is in play) or `list_branches_for_specialty`, and "
            "list those branches BY NAME.\n\n"
            "CONFIRMED REAL PRODUCTION FAILURE, the exact sequence this "
            "replaces: address + \"تحب تعرف عن الخدمات؟\" -> services + "
            "\"هل ترغب تحجز موعد في هذا الفرع أو تحتاج تعرف عن فروع "
            "ثانية؟\" -> \"نكمل الحجز على نفس رقم الواتساب ده؟\" -> "
            "\"من فضلك اكتب اسم الدكتور اللي حابب تحجز معاه في فرع "
            "الطوارئ؟\" - four turns walking a patient into a booking at "
            "a branch that has no doctors at all.\n\n"
        )

    session.pop("info_branch_no_doctors", None)
    session["info_branch_id"] = chosen.get("id")
    session["info_branch_name"] = name

    # A SERVICE IS ALREADY CHOSEN - DON'T START OVER.
    #
    # When the patient picked a service, asked which branches offer it,
    # and has now picked one of those branches, the service question is
    # settled. Re-listing that branch's whole catalogue throws away the
    # choice they already made and asks it again.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: after choosing جلسة إستشارة
    # أخصائي التغذية and being shown the two branches that offer it, the
    # patient picked فرع النزهة - and the reply was "فرع النزهة فيه
    # الخدمات التالية: 1️⃣ برنامج علاج نفسي نهاري 2️⃣ إستشارة الطبيب
    # العام 3️⃣ جلسة إستشارة أخصائي التغذية / تبغى تحجز موعد في فرع
    # النزهة لإحدى هذه الخدمات؟" - offering them the service they had
    # already chosen, as one of three options.
    chosen_service = session.get("service_display_name") or session.get("service_id")
    if chosen_service:
        service_label = session.get("service_display_name") or "the service they chose"
        return (
            "============================================================\n"
            "BRANCH PICKED FOR AN ALREADY-CHOSEN SERVICE - KEEP GOING\n"
            "============================================================\n"
            f"Option {position} is {name}. The patient already chose "
            f"{service_label}, and this branch is one of the branches "
            "that offer it - that is why it was on the list.\n\n"
            "Do NOT list this branch's services again, and do NOT ask "
            "which service they want. Both re-ask a question that is "
            "already answered.\n\n"
            f"Call `find_available_doctors` with `branch_name=\"{name}\"` "
            "(the chosen service is already on the session and narrows "
            "it automatically) and show the doctors who provide it "
            "there, numbered, then ask ONE question: which doctor.\n\n"
            "CONFIRMED REAL PRODUCTION FAILURE: this exact step replied "
            "with the branch's full service catalogue and asked them to "
            "pick a service - the one they had chosen two turns earlier "
            "being item 3 on that list.\n\n"
        )

    return (
        "============================================================\n"
        "THEY PICKED A BRANCH FROM THE LIST\n"
        "============================================================\n"
        f"Their reply is a pick from the branch list you just showed: "
        f"option {position} is {name}"
        f"{(', address: ' + address) if address else ''}.\n\n"
        "DO THE WHOLE THING IN ONE MESSAGE. They picked this branch to "
        "find out about it, so answer that now rather than asking "
        "permission to. Call `list_branch_services` in THIS turn and "
        "reply with:\n"
        f"  1. The address of {name}.\n"
        "  2. Its services, numbered, straight from that tool.\n"
        "  3. ONE question: تحب تحجز موعد في الفرع ده؟\n\n"
        "Do NOT ask \"تحب تعرف عن الخدمات؟\" and wait for a yes - that "
        "spends a turn asking to answer a question they already asked. "
        "And do NOT offer \"الدكاترة المتوفرين فيه\" as an option: it "
        "makes a bare \"اه\" ambiguous, and that ambiguity is what "
        "previously sent the reply into dumping a specialty list nobody "
        "had asked for.\n\n"
        "  - Use `list_branch_services` for the services. Never "
        "`list_hospital_services` or `answer_hospital_faq` for a "
        "per-branch services question - they have no per-branch data and "
        "return the same generic hospital-wide list for every branch.\n"
        f"  - They say YES to booking -> the branch is {name}; now ask "
        "the single specialty-vs-doctor question (\"تحب تبدأ بالتخصص "
        "ولا بالدكتور؟\") and NOTHING else - do not call "
        "`list_specialties` and do not print a specialty list in that "
        "same message. Many patients already know their doctor's name; "
        "making them read a specialty list first is a wasted turn.\n\n"
        f"IF THEY ASK FOR DOCTORS AT ANY POINT, they mean the doctors at "
        f"{name} - the branch they are looking at - not every doctor in "
        "the hospital. Pass that branch through (`branch_name=\"" + name +
        "\"` on `find_available_doctors`) so the roster is scoped to it. "
        "CONFIRMED REAL PRODUCTION FAILURE: asked for the available "
        "doctors while فرع الدقي was the branch under discussion, the "
        "reply listed all eight doctors in the hospital, including ones "
        "who do not work there at all.\n\n"
        "Resolve the pick from the list you already showed - never "
        "guess, and never re-ask which branch they meant.\n\n"
    )


def _build_empty_branch_booking_intent_directive(messages: list, session_id: str) -> str:
    """Fires when the patient asks to BOOK at a branch the INFO flow has
    already established has no bookable doctor.

    WHY THIS EXISTS: that turn is routinely answered with NO tool call
    at all - the model has the branch from conversation memory, so it
    just replies. CONFIRMED REAL PRODUCTION FAILURE: right after being
    shown فرع المعادي (zero doctors), the patient said "عاوزه احجز فيه
    مع دكتور" and the reply was "اخترت فرع المعادي ✅ / تحب تحجز مع
    دكتور معيّن عند فرع المعادي، ولا تبي أعرض لك الدكاترة المتاحين
    هناك؟" - confirming a branch nothing can be booked at, and asking a
    doctor question with no possible answer. Only after the patient
    answered THAT did the tools finally run and report the branch empty.
    Two wasted turns before the one true sentence.

    `tools._note_info_branch_availability` puts the fact on the session
    the moment the branch is shown; this reads it back on the turn it
    matters."""

    if not messages or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    branch_name = session.get("info_branch_no_doctors")
    if not branch_name:
        return ""

    # THE NOTE MUST STILL BE ABOUT THE BRANCH IN PLAY.
    #
    # A branch confirmed for the booking (or browsed since) supersedes
    # the note entirely. Without this check the note survived the
    # patient moving on, and a DIFFERENT branch inherited a claim that
    # was never about it. CONFIRMED REAL PRODUCTION FAILURE: فرع المعادي
    # was noted empty, the patient moved to فرع الدقي, four real doctors
    # came back for الدقي in that same turn, and the reply still
    # announced that الدقي had none.
    current_branch = (
        session.get("branch_display_name")
        or session.get("info_branch_name")
    )
    if current_branch and _norm_ar(current_branch) != _norm_ar(branch_name):
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _BOOKING_INTENT_RE.search(_norm_ar(text)):
        return ""

    return (
        "============================================================\n"
        "THEY WANT TO BOOK AT A BRANCH THAT HAS NOBODY - SAY SO NOW\n"
        "============================================================\n"
        f"The patient is asking to book at {branch_name}, which was "
        "already established to have NO bookable doctor right now.\n\n"
        "Answer that in THIS reply. Do not confirm the branch, do not "
        "write \"اخترت فرع ... ✅\", and do not ask whether they want a "
        "particular doctor or the list of available doctors there - "
        "there is no doctor to pick and no list to show, so every one of "
        "those is a question with no possible answer.\n\n"
        "This reply, in one message - and it still TELLS THEM ABOUT THE "
        "BRANCH before closing the door on booking there:\n"
        f"  1. The address of {branch_name}.\n"
        "  2. Its services, numbered, from `list_branch_services`. They "
        "picked this branch to find out about it; a branch having no "
        "doctor free does not make its address and services irrelevant, "
        "and skipping straight to \"can't book here\" leaves them with "
        "nothing at all about the place they asked about.\n"
        "  3. Then, plainly, that this branch has no booking right now, "
        "and ONE question: تحب أعرض لك الفروع اللي فيها حجز؟\n"
        "  4. Only if they say yes, call `list_branches_for_specialty` "
        "and list the branches that CAN take bookings - names, and "
        "addresses if you have them, but no doctor names. Do NOT narrow "
        "that list to whatever service they were last looking at: the "
        "question you asked was which branches take bookings, so answer "
        "that. They pick a branch first; what they want there comes "
        "after. `find_branches_offering_service` is only for when they "
        "ASK directly which branches offer a named service.\n\n"
        "Do not collapse steps 1-2. CONFIRMED REAL REGRESSION: this turn "
        "previously replied with the address, the branch's one service, "
        "and then the honest question - and a later change reduced it to "
        "just \"الفرع هذا ما فيه حجز حاليًا، تحب أعرض لك الفروع اللي "
        "فيها حجز؟\", dropping the address and services the patient had "
        "actually asked for.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: this exact turn came back as "
        "\"اخترت فرع المعادي ✅ / تحب تحجز مع دكتور معيّن عند فرع "
        "المعادي، ولا تبي أعرض لك الدكاترة المتاحين هناك؟\" - and only "
        "after the patient answered that did the truth finally arrive. "
        "Two wasted turns, and a ✅ on a branch nothing can be booked "
        "at.\n\n"
    )


def _build_bare_entity_answer_directive(messages: list) -> str:
    """Fires when the patient's latest message is just the bare word
    ("تخصص"/"فرع"), AND the assistant's own previous message actually
    offered that choice (specialty-vs-doctor, or a branch-related
    question inside the booking flow).

    Both conditions matter: without checking the previous AI turn, this
    used to fire on a bare "فروع" that was NOT answering any such
    question - e.g. a patient opening cold with "فروع" as their very
    first, standalone question ("what branches do you have"). That is a
    general DOCTOR/BRANCH INFO question, not a choice being answered,
    and routing it into the booking flow's `list_branches_for_specialty`
    (which needs a specialty to work from) produced a confusing "which
    specialty do you want to know its branches?" reply instead of just
    showing the branches. CONFIRMED REAL PRODUCTION FAILURE: exactly
    that - a bare "فروع" typed as an opening question triggered a
    "جاري البحث عن الأطباء المتاحين" progress message and a specialty
    list, when the patient never asked about specialties at all. See
    _build_branches_info_directive for the correct handling of that
    case."""

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _BARE_ENTITY_ANSWER_RE.match(text.strip()):
        return ""

    previous_ai = None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, _AIMessage):
            previous_ai = msg
            break

    if previous_ai is None:
        return ""

    previous_content = getattr(previous_ai, "content", "")
    previous_text = previous_content if isinstance(previous_content, str) else str(previous_content)

    if not _OFFERED_SPECIALTY_OR_DOCTOR_OR_BRANCH_CHOICE_RE.search(_norm_ar(previous_text)):
        return ""

    return _BARE_ENTITY_ANSWER_DIRECTIVE


# Loosely matches the family of choice-questions this directive is meant
# to be answering - "تحب تبدأ بالتخصص ولا بالدكتور؟", "تحب تحجزين في فرع
# معيّن، ولا أعرض لك الدكاترة/الفروع المتاحين؟", and their variants.
# Deliberately keyed to phrases that only appear when that specific
# choice was actually offered, not to the bare category words themselves
# (which would just recreate the false-positive this guard exists to
# prevent).
_OFFERED_SPECIALTY_OR_DOCTOR_OR_BRANCH_CHOICE_RE = re.compile(
    r"بالتخصص[^\n]{0,10}بالدكتور|بالدكتور[^\n]{0,10}بالتخصص|"
    r"فرع\s*معي.ن[^\n]{0,40}(?:اعرض|أعرض)|"
    r"specialty[^\n]{0,15}doctor|doctor[^\n]{0,15}specialty|"
    r"particular\s*branch[^\n]{0,40}show"
)


# ==========================================================
# A STANDALONE "what branches do you have" question (NOT an answer to
# any choice the assistant offered) - route to the general INFO flow.
# ==========================================================
#
# CONFIRMED REAL PRODUCTION FAILURE, twice in one conversation: asked
# "ايه الفروع بتاعت المستشفي", the reply was the clinic's SERVICES list
# (from the knowledge base) - branches and services are different
# things entirely. Asked again with the bare word "فروع", the reply
# instead searched for doctors and showed the booking flow's SPECIALTY
# list ("أي تخصص ترغب تعرف فروعه؟") - again not branches, and forcing a
# specialty choice the patient never asked for. Both times, the one
# thing that should have happened - showing the branches themselves, via
# the DOCTOR/BRANCH INFO flow's own `match_entity_info` - never did.
_BRANCHES_INFO_QUESTION_CUES = (
    "فروع", "الفروع", "فروعكم", "فروعكو", "عناوين الفروع", "اماكن الفروع",
    "أماكن الفروع", "فين الفروع", "وين الفروع",
    "branch", "branches", "locations",
)


def _build_branches_info_directive(messages: list) -> str:
    """Fires when the patient asks a standalone question about the
    clinic's branches, that is NOT already handled as an answer to a
    specialty/doctor/branch choice the assistant just offered (that
    narrower case is `_build_bare_entity_answer_directive`, and takes
    priority - see the ordering where both are assembled)."""

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    normalized = _norm_ar(text).lower()

    if not any(cue in normalized for cue in _BRANCHES_INFO_QUESTION_CUES):
        return ""

    # If the bare-entity-answer directive already fired for this exact
    # turn (a real specialty/doctor/branch choice was actually offered
    # and answered), let THAT directive own this turn instead - don't
    # send two conflicting instructions for the same message.
    if _build_bare_entity_answer_directive(messages):
        return ""

    return (
        "============================================================\n"
        "BRANCHES QUESTION - USE `match_entity_info`, NOT SERVICES OR "
        "SPECIALTIES\n"
        "============================================================\n"
        "The patient is asking about the clinic's BRANCHES (locations) - "
        "not its services, and not which medical specialties it offers. "
        "These are three different things, and confirmed real production "
        "failures have answered a branches question with each of the "
        "other two by mistake.\n\n"
        "Call `match_entity_info(user_input=\"\", entity_type=\"branch\")` "
        "and present its \"list\" result as a numbered list (emoji "
        "digits), WITH EACH BRANCH'S ADDRESS beside its name:\n"
        "    1\ufe0f\u20e3 فرع المعادي - 133 شارع 9 المعادي\n"
        "    2\ufe0f\u20e3 فرع الدقي - 9 شارع الإمام الغزالي، الدقي\n"
        "The address is the useful half of the answer - \"which branches "
        "do you have\" is nearly always \"which one is near me\". A bare "
        "list of names makes them ask again. CONFIRMED REAL REGRESSION: "
        "this list previously carried each address and was reduced to "
        "names only (\"Emergency / Al Manar / Al Nozha\"), because a "
        "\"BY NAME ONLY\" rule written for a DIFFERENT case - offering "
        "alternative branches after an empty one - was applied here too. "
        "That rule is about not listing DOCTORS at those branches; it "
        "never meant dropping addresses from this list.\n\n"
        "Then ask if they'd like details on one of them. Do NOT "
        "call `list_hospital_services` (that answers a SERVICES "
        "question), and do NOT call `list_specialties` or "
        "`list_branches_for_specialty` (those require a specialty and "
        "belong to the BOOKING flow, not a general \"what branches do "
        "you have\" question) - and do not ask the patient to first pick "
        "a specialty or a doctor before you'll show them the branches "
        "list; that information was never asked for.\n\n"
        "SHOW EVERY BRANCH THE TOOL RETURNED, AND SAY NOTHING ABOUT "
        "DOCTOR AVAILABILITY. They asked which branches exist - that is "
        "a question about the hospital, not about who is bookable today. "
        "So:\n"
        "  - Never omit a branch. A branch that exists is part of the "
        "honest answer.\n"
        "  - Never append availability commentary to a row - not a "
        "sentence afterwards, and not a parenthetical like \"(لا يوجد "
        "أطباء متاحين حالياً)\" beside a branch name.\n"
        "  - Never offer booking in the same breath as the list. End "
        "with ONE question: whether they'd like to know more about one "
        "of them.\n"
        "  - The list result carries NO availability field, by design. "
        "If you are about to mention doctors while listing branches, you "
        "are answering a question nobody asked.\n"
        "CONFIRMED REAL PRODUCTION FAILURES, twice: the reply listed "
        "three branches and announced that المعادي، مصر الجديدة and بني "
        "سويف had no doctors - and the message the patient finally got "
        "had those three missing entirely (six real branches asked "
        "about, three shown). After that was fixed, the next reply "
        "listed all six but tagged those same three with \"(لا يوجد "
        "أطباء متاحين حالياً)\".\n\n"
    )


_BRANCH_QUESTION_RE = re.compile(
    r"(?:ال)?فروع|(?:ال)?فرع|branch(?:es)?"
)

_DOCTOR_BRANCHES_DIRECTIVE = (
    "============================================================\n"
    "SHOW THIS DOCTOR'S OWN BRANCHES AND DAYS - NOT A GENERIC LIST\n"
    "============================================================\n"
    "A doctor is already confirmed for this booking, and the patient is "
    "asking about branches. The question is therefore \"where and when "
    "does THIS doctor work?\" - not \"what branches does the hospital "
    "have?\".\n\n"
    "Call `get_doctor_schedule_for_booking` now. It returns exactly "
    "that: every branch this doctor works at, with the weekdays and "
    "hours at each. Do NOT call `list_branches_for_specialty` here - it "
    "answers a different question, and its answer includes branches "
    "this doctor may never attend.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: asked to show a confirmed "
    "doctor's branches, the reply listed all three hospital branches "
    "with their street addresses and opened with \"الدكاترة المتاحين "
    "عندنا مش محددين بعد في الفروع\" - which is simply untrue, the "
    "doctor's branches were one tool call away, and the patient could "
    "have picked a branch the doctor does not work at.\n\n"
    "Never say the doctor's branches are unknown or not yet assigned. "
    "If the tool genuinely returns nothing, say plainly that this "
    "doctor has no published schedule at the moment and offer another "
    "doctor - do not fall back to the hospital's branch list.\n\n"
)


def _build_doctor_branches_directive(messages: list, session_id: str) -> str:
    """Fires when a doctor is confirmed and the patient's latest message
    asks about branches."""

    if not messages or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    if not session.get("doctor_id"):
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _BRANCH_QUESTION_RE.search(_norm_ar(text)):
        return ""

    return _DOCTOR_BRANCHES_DIRECTIVE


_BRANCH_QUESTION_PHRASING_DIRECTIVE = (
    "============================================================\n"
    "DOCTOR CONFIRMED, NO BRANCH YET - SHOW THE SCHEDULE, DON'T ASK\n"
    "============================================================\n"
    "A doctor is settled for this booking and no branch is confirmed "
    "yet. Do NOT ask a branch question here (e.g. \"تحب تحجز في فرع "
    "معيّن، ولا أعرض لك كل الفروع اللي د. [اسم الدكتور] متاح فيهم؟\") - "
    "that just spends a turn asking when the real answer (this doctor's "
    "actual branches, days, and hours) is one tool call away. Never "
    "offer \"كل الدكاترة المتاحين\" or any doctor list here either - the "
    "doctor question is already settled.\n\n"
    "Call `get_doctor_schedule_for_booking` NOW, with no question asked "
    "first, and show its result grouped by branch - one line per "
    "weekday/hours per branch, using only the real names/days/hours the "
    "tool returned. Structure it like this (the Arabic below is only an "
    "illustration of shape/content - always phrase it in this clinic's "
    "own configured dialect per the LANGUAGE & DIALECT rule, or in "
    "English if the patient is currently writing English, never these "
    "exact words):\n\n"
    "    مواعيد الدكتور [اسم الدكتور] في فرع [اسم الفرع الأول]:\n"
    "    • [اليوم]: من [من الساعة] لـ [إلى الساعة] — [اسم الخدمة]\n\n"
    "    وفي فرع [اسم الفرع الثاني]:\n"
    "    • [اليوم]: من [من الساعة] لـ [إلى الساعة] — [اسم الخدمة]\n\n"
    "    حابب تحجز في أنهي فرع وأنهي يوم؟\n\n"
    "If the tool result has only ONE branch, there is still no ASKING "
    "to do (since there's nothing to choose between) - but you must "
    "still SHOW the schedule message exactly as above (again, in this "
    "clinic's own configured dialect per the LANGUAGE & DIALECT rule - "
    "or in English if the patient is currently writing English - not "
    "this specific wording), "
    "e.g.:\n\n"
    "    مواعيد الدكتور محمد زايد في فرع عيادات سكاي التخصصية:\n"
    "    • الاثنين: من 2:40 مساءً لـ 5:40 مساءً — جلسة تحليل سلوك تطبيقي\n\n"
    "With ONE branch and ONE day, the same layout, minus the choice - "
    "the last line asks about that day directly instead:\n\n"
    "    مواعيد الدكتورة سارة عبد الله في فرع الدقي:\n"
    "    • الاثنين: من 10:00 صباحًا لـ 8:00 مساءً — كشف عيادة النساء\n\n"
    "    تحب أشوف لك المواعيد المتاحة ليوم الاثنين؟\n\n"
    "EVERY branch and EVERY day the tool returned gets its own line, one "
    "under the other, in this same layout - never collapse two branches "
    "into a sentence, and never leave a day out.\n\n"
    "`get_doctor_schedule_for_booking` already auto-confirms the single "
    "branch into the session for you, so do NOT ask \"which branch?\" - "
    "but the patient should still see where and when this doctor works "
    "before you move to the day/time question. Never skip straight from "
    "\"doctor confirmed\" to asking about a day, or to "
    "`list_available_days_for_booking`, without first showing this "
    "schedule line - even when there was only ever one branch to show.\n\n"
    "AUTO-RESOLVING A PARTIAL ANSWER - only when the schedule you JUST "
    "showed makes it unambiguous:\n"
    "  - They name ONLY a day, and that day appears at exactly ONE of "
    "the branches you showed -> treat that branch as chosen "
    "automatically. Do not ask them to also name it.\n"
    "  - They name ONLY a branch, and that branch has exactly ONE day "
    "in the schedule you showed -> treat that day as chosen "
    "automatically the same way.\n"
    "  - In every OTHER case - including whenever you are not fully "
    "certain the combination they named genuinely matches a row in the "
    "schedule you just displayed - do not guess: confirm the branch "
    "with `match_entity_for_booking` (entity_type=\"branch\") and "
    "validate the day with `resolve_available_day`. A branch/day "
    "combination is never assumed valid just because each half looked "
    "plausible on its own; it must be confirmed by a real tool result.\n\n"
    "Only once a branch AND a day are genuinely confirmed - by the "
    "schedule's own unambiguous shape, or by these tools - do you move "
    "on to show the real nearest available appointment (STEP NB3/NB4) "
    "and ask whether it suits them. Never state or imply a 'nearest "
    "appointment' yourself; that fact only ever comes from "
    "`resolve_available_day` / `list_available_days_for_booking`'s "
    "actual result, never from your own reasoning about the schedule.\n\n"
)


def _doctor_is_settled(messages: list, session_id: str) -> bool:
    """Whether a specific doctor has been settled for this booking -
    either confirmed into the session, or named by a tool and then
    agreed to by the patient in the turn just gone.

    The second case matters: the patient can say "اه" to "تحب أحجز لك
    عند د. طه مبروك؟" and the doctor is settled from their point of view
    long before any tool has written it into the session.
    """

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    if session.get("doctor_id"):
        return True

    last_list = session.get("last_list") or {}
    if last_list.get("entity_type") == "doctor" and last_list.get("items"):
        return True

    return False


def _build_branch_question_directive(messages: list, session_id: str, agent_name: str) -> str:
    """Fires in the booking flow once a doctor is settled and no branch
    is confirmed yet - the exact point where a branch QUESTION used to
    be asked. Now instructs showing the doctor's real schedule (every
    branch, day, and hour) immediately instead of asking, per the
    product decision to remove that extra turn - see
    _BRANCH_QUESTION_PHRASING_DIRECTIVE."""

    # "medical" IS INCLUDED ON PURPOSE. The medical-guidance flow hands
    # over into booking without changing agent: symptom -> specialty ->
    # doctor list -> the patient picks one, all inside `medical`. Gating
    # this to booking/concierge meant the directive was silent on
    # exactly that path.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: after "اخترت دكتورة رانيا عبد
    # الرحمن ✅" the reply asked "أي فرع تفضل تحجز فيه؟" and then
    # printed a generic branch list ("Al Nozha — 1 طبيب", "Al Manar — 1
    # طبيب") - branches of the CLINIC, not of that doctor, offering a
    # branch she does not work at.
    if agent_name not in ("booking", "concierge", "medical") or not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    if session.get("branch_id"):
        return ""

    if not _doctor_is_settled(messages, session_id):
        return ""

    return _BRANCH_QUESTION_PHRASING_DIRECTIVE


# ==========================================================
# "Offered to cancel an appointment nothing looked up" verifier
# ==========================================================
#
# CONFIRMED REAL PRODUCTION FAILURE, mid-BOOKING: the patient was shown
# ten available slots, picked one ("٤"), and the reply came back
#
#     هذا هو موعدك الذي تبغى تلغيه؟
#     الطبيب: طه مبروك / الفرع: الشيخ زايد / التاريخ: 29/08/2026
#
# - the CANCELLATION confirmation, during a booking, with no
# cancellation tool called anywhere in the conversation. The patient was
# one "نعم" away from confirming the cancellation of an appointment that
# does not exist.
#
# `_reply_invents_availability` did not catch it: the date really did
# appear in a tool result - the slots list they were choosing FROM - so
# every date and time in the reply checked out. The fabrication was not
# in the values, it was in what the message claimed to be.
#
# The rule is simple and has no false-positive surface: you cannot ask
# someone to confirm cancelling an appointment you have never looked up.
_CANCELLATION_FRAMING_RE = re.compile(
    r"تلغيه\b|تلغيها\b|"
    r"تلغي\s*هذا\s*(?:ال)?موعد|الغاء\s*هذا\s*(?:ال)?موعد|"
    r"موعدك[^.\n؟?]{0,20}تلغي|"
    r"cancel\s+(?:this|your|the)\s+appointment|"
    r"appointment\s+(?:you|to)\s+(?:want\s+to\s+)?cancel"
)

# THE TWO BARE ALTERNATIVES REMOVED ABOVE ("تلغي موعد" / "الغاء موعد",
# with no possessive or demonstrative attached) MATCHED THE ROUTINE
# CAPABILITY MENU, NOT A CONFIRMATION.
#
# CONFIRMED REAL FALSE POSITIVE: every standard greeting lists what the
# assistant can help with, including "✏️ تعديل أو إلغاء موعد قائم"
# ("editing or cancelling an EXISTING appointment" - describing a
# SERVICE, not confirming a specific one). That phrase alone matched
# "الغاء\s*(?:ال)?موعد" and fired this verifier on literally the first
# message of every single conversation, spending an extra LLM call on a
# reply that had done nothing wrong.
#
# The real confirmed failure this guard exists for - "هذا هو موعدك الذي
# تبغى تلغيه؟" - is still caught: "تلغيه" (cancel IT, with the object
# pronoun attached) never appears in a generic capability description,
# only when referring back to an appointment already identified in the
# conversation. The remaining alternatives require an explicit "هذا"
# (this) or "موعدك" (YOUR appointment) - a routine menu bullet has
# neither.

_APPOINTMENT_LOOKUP_TOOLS = ("lookup_appointment", "check_booking_status")


def _reply_offers_cancellation_without_lookup(reply_text: str, state: AgentState) -> bool:
    """True when the reply frames something as an appointment the patient
    is about to cancel, while no appointment has been looked up."""

    if not reply_text:
        return False

    if not _CANCELLATION_FRAMING_RE.search(_norm_ar(reply_text)):
        return False

    for msg in state.get("messages", []) or []:
        if getattr(msg, "name", None) in _APPOINTMENT_LOOKUP_TOOLS:
            try:
                data = json.loads(msg.content)
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict) and data.get("appointment"):
                return False

    return True


_CANCELLATION_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU OFFERED TO CANCEL AN APPOINTMENT NOTHING LOOKED UP - REWRITE\n"
    "============================================================\n"
    "Your previous draft asked the patient to confirm cancelling an "
    "appointment. No appointment has been looked up in this "
    "conversation - `lookup_appointment` and `check_booking_status` have "
    "either never run or never returned one - so there is no "
    "appointment to cancel and the details you listed are not a real "
    "booking.\n\n"
    "Read the conversation again and answer what they ACTUALLY asked. "
    "If they picked a number from a list of available TIMES, they are "
    "choosing a slot to BOOK - continue the booking with that slot. "
    "Never turn a booking into a cancellation.\n\n"
    "If the patient genuinely does want to cancel something, you must "
    "first find the appointment with `lookup_appointment` and confirm "
    "the details it returns - never details you assembled yourself.\n\n"
    "Rewrite the reply now.\n\n"
)


def _build_out_of_scope_block(templates: dict, language: str = "ar") -> str:
    """The clinic's scope refusal, as ONE fixed text.

    Built from the client's own agent/clinic name so it is branded, and
    identical every time so an off-topic question gets the same answer
    for every patient - rather than the model improvising a different
    polite deflection each turn.

    A client can author their own via `msg_out_of_scope`.

    ENGLISH CONVERSATIONS GET AN ENGLISH BLOCK. This text used to be
    Arabic and nothing else, so the one message in the whole system that
    is pinned word-for-word was also the one guaranteed to come out in
    the wrong language. CONFIRMED IN PRODUCTION: a conversation held
    entirely in English received the full Arabic paragraph. Everything
    else in this file goes to some length to keep a reply in the
    patient's own language; this was quietly exempt from all of it.
    """

    authored = (templates or {}).get("msg_out_of_scope")
    if authored and authored.strip():
        return authored.replace("\r\n", "\n").replace("\r", "\n").strip()

    templates = templates or {}

    if language == "en":
        agent_name = (
            templates.get("_agent_name")
            or templates.get("_agent_name_ar")
            or "the virtual assistant"
        )
        clinic_name = (
            templates.get("_clinic_name")
            or templates.get("_clinic_name_ar")
            or "the hospital"
        )
        return (
            f"I'm sorry 🌷 I'm {agent_name}, the virtual assistant at "
            f"{clinic_name}, and I can help you with the hospital's own "
            "services - booking, changing or cancelling appointments, "
            "choosing the right specialty or doctor, questions about our "
            "services, filing a complaint, or putting you through to "
            "customer service.\n"
            "I'd be glad to help with any of those 😊"
        )

    # The block is Arabic, so the ARABIC name fields come first. Using
    # `_agent_name`/`_clinic_name` here put "أنا Latifa، المساعدة
    # الافتراضية في Dar El Oyoun Hospitals" into an otherwise Arabic
    # sentence - a Latin-script name mid-sentence in RTL text, in the
    # one message that is supposed to be the clinic's most polished.
    agent_name = (
        templates.get("_agent_name_ar")
        or templates.get("_agent_name")
        or "المساعدة الافتراضية"
    )
    clinic_name = (
        templates.get("_clinic_name_ar")
        or templates.get("_clinic_name")
        or "المستشفى"
    )

    return (
        f"عذرًا 🌷 أنا {agent_name}، المساعدة الافتراضية في {clinic_name}، "
        "ومختصة بمساعدتك في خدمات المستشفى مثل حجز أو تعديل المواعيد، "
        "إلغاء المواعيد، اختيار التخصص أو الطبيب المناسب، الاستفسار عن "
        "خدمات المستشفى، تقديم شكوى، أو التواصل مع خدمة العملاء.\n"
        "يسعدني مساعدتك في أي من هذه الخدمات 😊"
    )


def _build_scope_directive(templates: dict, language: str = "ar") -> str:
    """Always present, deliberately short.

    CONFIRMED REAL PRODUCTION FAILURE: asked "موسم الرياض خلص ولا لسه"
    - a question about a public entertainment season, with no connection
    to the hospital whatsoever - the assistant called
    `list_hospital_services` (a tool that cannot answer it), then
    ANSWERED it as fact ("موسم الرياض انتهى") and appended a service
    list nobody asked for. Two failures at once: a tool called because
    something had to be called, and a claim about the outside world
    stated with confidence and no source.
    """

    block = _build_out_of_scope_block(templates, language)

    return (
        "============================================================\n"
        "WHAT YOU ARE FOR - AND WHAT TO DO WITH EVERYTHING ELSE\n"
        "============================================================\n"
        "You handle this hospital's own services ONLY: booking, "
        "changing and cancelling appointments; choosing the right "
        "specialty or doctor; questions about the hospital's own "
        "services, doctors, branches and hours; complaints; and "
        "handing over to a human.\n\n"
        "ANYTHING ELSE gets this EXACT text as your ENTIRE reply, "
        "copied verbatim, with nothing added before or after it and no "
        "question appended:\n\n"
        "[BEGIN-EXACT-TEXT]\n"
        f"{block}\n"
        "[END-EXACT-TEXT]\n\n"
        "\"Anything else\" means anything the hospital's own systems "
        "cannot answer: news, public events, seasons and festivals, "
        "sport, weather, prices of things the hospital does not sell, "
        "religion, politics, other companies, general trivia, coding, "
        "translation, or any other request that is not about this "
        "hospital.\n\n"
        "IT DOES NOT MEAN ORDINARY CONVERSATION. Greetings (\"أهلاً\", "
        "\"السلام عليكم\", \"صباح الخير\", \"hi\"), thanks, apologies, "
        "goodbyes, \"كيف حالك\", a patient describing a symptom, saying "
        "yes or no, or any short reply that keeps this conversation "
        "moving are ALL in scope. Answer those normally and warmly.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: a patient opened with "
        "\"اهلا\" and received the welcome message with the refusal "
        "above stapled underneath it - told they were off-topic by the "
        "very first message the clinic ever sent them, for saying "
        "hello. If the message is a greeting, the greeting IS the "
        "reply. Never attach the refusal to it.\n\n"
        "The refusal is for a question you were asked and cannot "
        "answer. It is never an addition to a reply that already "
        "answered something - if you are already saying something else, "
        "the refusal does not belong in that message at all.\n\n"
        "Do NOT call a tool for such a question. No tool here can answer "
        "it, and calling one does not make an answer available - it just "
        "spends the patient's time and produces an irrelevant result you "
        "will then be tempted to build a reply around.\n\n"
        "Do NOT answer it from your own knowledge, not even when you are "
        "confident and not even in one word. You have no way to check "
        "whether it is still true, and a confident aside about the "
        "outside world is exactly what makes a patient trust the "
        "medical parts of this conversation more than they should.\n\n"
        "Do NOT mix the two: never answer the off-topic part AND then "
        "add hospital information. Reply with the text above, and "
        "nothing else.\n\n"
    )


_NAME_REJECTION_RE = re.compile(
    r"اسمين\s*علي\s*الاقل|اسمين\s*على\s*الأقل|"
    r"(?:ال)?اسم\s*(?:ال)?اول\s*و\s*(?:اسم\s*)?(?:ال)?عائله|"
    r"at\s+least\s+two\s+names|first\s+(?:name\s+)?and\s+(?:the\s+)?(?:family|last)\s+name"
)

# A name PART: two or more letters, Arabic or Latin. Deliberately not a
# dictionary check - "محمد" and "Aymen" are equally valid, and no list
# could ever cover the names real patients have.
_NAME_PART_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# Words that are never part of a name, so "اسمي ساره محمد" counts as two
# parts rather than three.
_NAME_FILLER_WORDS = {
    "اسمي", "اسمى", "انا", "أنا", "الاسم", "اسم", "هو", "my", "name", "is", "im",
}


def _count_name_parts(text: str) -> int:
    """How many name parts the patient actually supplied."""

    if not text:
        return 0

    parts = [
        part for part in _NAME_PART_RE.findall(text)
        if _norm_ar(part).lower() not in _NAME_FILLER_WORDS
    ]
    return len(parts)


def _reply_wrongly_rejects_full_name(reply_text: str, state: AgentState) -> bool:
    """True when the reply sends the patient back to re-type a name that
    ALREADY has at least two parts.

    CONFIRMED REAL PRODUCTION FAILURE: the patient answered "ساره محمد" -
    two names, exactly what was asked for - and was told
    "يجب أن يحتوي على اسمين على الأقل ... أعطني اسمك الكامل". There is
    nothing they could have typed to satisfy it; the name was already
    correct. The two-name requirement lived only in the prompt as a
    judgement for the model to make, and it made it wrong.

    Counting the parts in code removes the judgement entirely: the
    objection is only allowed to stand when the name genuinely has
    fewer than two parts.
    """

    if not reply_text or not _NAME_REJECTION_RE.search(_norm_ar(reply_text)):
        return False

    from langchain_core.messages import HumanMessage as _HumanMessage

    for msg in reversed(state.get("messages", []) or []):
        if not isinstance(msg, _HumanMessage):
            continue
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        # The name is whatever they last typed that wasn't an email or a
        # phone number.
        candidate = re.sub(r"\S+@\S+|\+?\d[\d\s\-()]{5,}", " ", text)
        if _count_name_parts(candidate) >= 2:
            return True
        if candidate.strip():
            return False

    return False


_NAME_REJECTION_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "THE NAME THEY GAVE IS ALREADY VALID - DO NOT ASK AGAIN\n"
    "============================================================\n"
    "Your previous draft told the patient their name needs at least two "
    "parts. It already HAS at least two parts - this was checked in "
    "code, not guessed. Asking again is asking them to retype something "
    "that was correct the first time, and there is nothing they could "
    "send that would satisfy you.\n\n"
    "Accept the name exactly as they wrote it, including any spelling "
    "you might have expected differently - it is their name, not "
    "yours - and continue the booking from where it stands. If they "
    "also gave an email, acknowledge it in the same breath and move "
    "on.\n\n"
    "Rewrite the reply now, accepting the name.\n\n"
)


_REVIEW_CARD_PHONE_DIRECTIVE = (
    "============================================================\n"
    "THE REVIEW CARD SHOWS THE NUMBER ITSELF, NOT A DESCRIPTION OF IT\n"
    "============================================================\n"
    "The phone line of the review card must contain the actual digits. "
    "For this conversation that is:\n"
    "    {phone}\n\n"
    "Write exactly that. Never write a DESCRIPTION of the number in its "
    "place - not \"رقم الواتساب الحالي\", not \"نفس الرقم\", not \"the "
    "number you're messaging from\", not \"your current WhatsApp "
    "number\".\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: a review card went out reading "
    "\"📱 الجوال: رقم الواتساب الحالي\". The patient is being asked to "
    "check their booking details are correct, and the one field they "
    "most need to verify was a sentence about itself instead of a "
    "number they could read back. The number was known the entire "
    "time.\n\n"
    "The same rule applies to every other line of the card: real "
    "values only, never a phrase standing in for one.\n\n"
)


def _build_review_card_phone_directive(state: AgentState, session_id: str) -> str:
    """Supplies the actual phone number whenever the booking is at or
    near the review-card step, so it can never be described in words."""

    channel_phone = (state or {}).get("channel_phone")
    if not channel_phone:
        return ""

    if not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    if not (session.get("doctor_id") and session.get("branch_id")):
        return ""

    normalized = tools.normalize_phone_number(channel_phone, state) or channel_phone

    return _REVIEW_CARD_PHONE_DIRECTIVE.format(phone=normalized)


# The refusal's distinctive middle clause. Matched rather than
# comparing the whole block, because the model may reword the clinic
# name or drop the trailing line while still clearly emitting "this is
# outside what I do".
_SCOPE_REFUSAL_ANCHOR_RE = re.compile(
    r"مختصه\s*بمساعدتك\s*في\s*خدمات\s*(?:ال)?مستشفي|"
    r"مختص\s*بمساعدتك\s*في\s*خدمات\s*(?:ال)?مستشفي|"
    r"only\s+help\s+(?:you\s+)?with\s+(?:the\s+)?hospital"
)


def _is_scope_refusal(reply_text: str, templates: dict) -> bool:
    """Whether this reply is (or contains) the out-of-scope refusal."""

    if not reply_text:
        return False

    if _SCOPE_REFUSAL_ANCHOR_RE.search(_norm_ar(reply_text)):
        return True

    # BOTH LANGUAGES, always. The block is language-specific now, but
    # this function's callers are not: the greeting guard passes no
    # language, and a verifier can be looking at a draft written in the
    # other one. Checking only the "current" language would let the
    # refusal through unrecognised in exactly the mixed-language cases
    # these guards exist for.
    normalized_reply = _normalize_for_compare(reply_text)
    for candidate_language in ("ar", "en"):
        block = _build_out_of_scope_block(templates, candidate_language)
        if block and _normalize_for_compare(block) in normalized_reply:
            return True

    return False


_BRANCH_DENIAL_RE = re.compile(
    r"ما\s*عنده\s*دكاتره|ما\s*في\s*دكاتره|مفيش\s*دكاتره|لا\s*يوجد\s*(?:اطباء|دكاتره)|"
    r"ما\s*عنده\s*مواعيد|مفيش\s*مواعيد|غير\s*متاح|مش\s*متاح|"
    r"no\s+doctors?\s+available|not\s+available\s+at"
)


def _reply_denies_a_branch_the_tools_offered(reply_text: str, state: AgentState) -> bool:
    """True when the reply says a branch has nothing available, while a
    tool result in this same conversation lists that branch as one of
    the options.

    CONFIRMED REAL PRODUCTION FAILURE: a doctor was chosen, the patient
    asked for فرع الدقي, and was told الدقي had no doctors available -
    then, one turn later, that الشيخ زايد had none either. Both were
    false: `list_available_days_for_booking` went on to return that same
    doctor's real working days AT الدقي. The patient was turned away
    twice from branches the doctor actually works at, and only got
    through by insisting.

    A denial is only allowed to stand when no tool has said otherwise.
    """

    if not reply_text or not _BRANCH_DENIAL_RE.search(_norm_ar(reply_text)):
        return False

    offered = _branches_named_by_tools(state)
    if not offered:
        return False

    folded_reply = _norm_ar(reply_text)

    for name in offered:
        folded_name = _norm_ar(name)
        if len(folded_name) < 3 or folded_name not in folded_reply:
            continue

        # The branch is named in a reply that denies availability, and a
        # tool result offered it. Only a tool result that ITSELF reported
        # emptiness can justify that.
        if not _tools_reported_branch_empty(state):
            return True

    return False


def _branches_named_by_tools(state: AgentState) -> set:
    """Every branch name any tool result in this conversation offered."""

    names = set()

    for msg in state.get("messages", []) or []:
        if getattr(msg, "type", None) != "tool":
            continue
        try:
            data = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue

        for key in ("branches", "schedules"):
            for item in data.get(key) or []:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("branchName")
                    if name:
                        names.add(str(name))

    # Same persistent-memory fix as `_known_branch_text`/
    # `_doctor_names_from_tools` above - a THIRD independent
    # message-scanning implementation of "which branches has this
    # conversation been told about", with the identical blind spot: a
    # branch this session legitimately saw several turns ago can stop
    # being visible here even though it was never actually withdrawn.
    # This function only ever produces a false NEGATIVE (missing a real
    # branch just means this particular guard stays quiet, rather than
    # wrongly rejecting a correct reply the way the invented-branch/
    # invented-doctor guards did) - lower severity, but the same fix
    # closes it the same way.
    names |= tools.get_known_entity_names(state.get("session_id"), "branch")

    return names


_EMPTY_RESULT_STATUSES = {
    "not_found", "no_doctors", "empty", "none", "no_slots", "not_available",
}


def _tools_reported_branch_empty(state: AgentState) -> bool:
    """Whether any tool result actually reported nothing available."""

    for msg in state.get("messages", []) or []:
        if getattr(msg, "type", None) != "tool":
            continue
        try:
            data = json.loads(msg.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue

        status = str(data.get("status") or "").lower()
        if status in _EMPTY_RESULT_STATUSES:
            return True
        # A branch reported empty by FLAG rather than by status. Both of
        # these are a tool explicitly saying "this branch has nobody" -
        # `noDoctorsAtBranch` from `match_entity_for_booking`, and
        # `not_found_in_branch` from `find_available_doctors`.
        #
        # CONFIRMED REAL FALSE POSITIVE this fixes: the patient asked
        # about فرع المعادي, `match_entity_for_booking` returned
        # `noDoctorsAtBranch: true`, and the reply correctly said so
        # while listing the OTHER branches as alternatives. Because
        # those alternative branch names appeared anywhere in the reply
        # text, and because this function only ever looked at `status`
        # (so the `noDoctorsAtBranch` flag was invisible to it), the
        # verifier concluded the reply was denying an available branch -
        # and fired twice on a reply that was entirely truthful.
        if data.get("noDoctorsAtBranch") or status == "not_found_in_branch":
            return True
        if status == "found" and not any(
            data.get(key) for key in ("doctors", "days", "slots", "branches", "schedules")
        ):
            return True

    return False


_BRANCH_DENIAL_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU TURNED THE PATIENT AWAY FROM A BRANCH THAT IS AVAILABLE\n"
    "============================================================\n"
    "Your previous draft told the patient a branch has nothing "
    "available. A tool result in THIS conversation lists that branch as "
    "one of the options, and no tool has reported it empty.\n\n"
    "Do not decide a branch is unavailable. Check it: call "
    "`list_available_days_for_booking` for the confirmed doctor at that "
    "branch, and answer from what it returns.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: a patient asking for فرع الدقي "
    "was told it had no doctors, then told الشيخ زايد had none either - "
    "and the same doctor's real working days at الدقي came back from "
    "the very next tool call. They were turned away twice from branches "
    "that were available the whole time, and only got through by "
    "insisting. Most patients do not insist; they leave.\n\n"
    "Rewrite the reply - or better, call the tool and answer from it.\n\n"
)


_ASKS_FOR_PHONE_RE = re.compile(
    r"رقم\s*(?:ال)?جوال|رقم\s*(?:ال)?موبايل|رقم\s*(?:ال)?تليفون|رقم\s*(?:ال)?هاتف|"
    r"رقم\s*(?:ال)?حجز|(?:ال)?رقم\s*(?:ال)?مرجعي|"
    r"your\s+(?:mobile|phone)\s+number|booking\s+(?:reference|number)"
)

# A SUMMARY/CONFIRMATION message (STEP C6's complaint summary, or the
# booking review card) legitimately RESTATES the phone number as an
# already-settled field ("📱 رقم الهاتف: نفس رقم الواتساب") - that is
# not a request for it. CONFIRMED REAL PRODUCTION FAILURE: the STEP C6
# summary "تأكيد إرسال الشكوى بهذا الشكل؟ ... رقم الهاتف: متضمن رقم
# الواتساب الحالي" was flagged and replaced by the safe fallback message
# TWICE in a row (the correction retry produced an equivalent summary
# and was flagged again), which meant a complaint that had reached its
# very last, ready-to-send step was thrown away into a generic
# "technical error" message instead of being sent. A summary is
# recognized by its own fixed confirmation phrasing and excluded here -
# only a reply with NEITHER of these cues is treated as a genuine
# re-ask.
_SUMMARY_OR_CONFIRMATION_CUE_RE = re.compile(
    r"تاكيد\s*(?:ال)?ارسال|تاكيد\s*(?:ال)?حجز|"
    r"هل\s*(?:جميع\s*)?(?:ال)?بيانات\s*صحيح|"
    r"confirm\s*(?:the\s*)?(?:sending\s*(?:the\s*)?)?(?:complaint|booking)|"
    r"is\s*everything\s*correct"
)

# A bare yes. The patient agreeing to a yes/no question the assistant
# itself asked.
_BARE_AFFIRMATION_RE = re.compile(
    r"^\s*(?:اه|ايه|أيوه|ايوه|ايوا|نعم|تمام|اوك|أوك|ok|okay|yes|yep|sure|"
    r"اكمل|كمل|اه\s*اكمل|ماشي|حاضر|طبعا|أكيد|اكيد)"
    r"\s*[.!؟?،,]*\s*$",
    re.IGNORECASE,
)


_BRANCH_SERVICES_OFFER_RE = re.compile(
    r"(?:ال)?خدمات[^\n]{0,40}(?:ال)?فرع|(?:ال)?فرع[^\n]{0,40}(?:ال)?خدمات|"
    r"services[^\n]{0,40}branch|branch[^\n]{0,40}services"
)


def _build_branch_services_affirmation_directive(messages: list, session_id: str) -> str:
    """Fires when the patient answers a bare "yes" to an offer to show a
    BRANCH's services.

    WHY THIS EXISTS: the services directive keys on the PATIENT's own
    wording ("خدمات", "services"), and a bare "اه" contains none of it -
    so nothing fired on the one turn where the tool choice actually
    mattered, and the model reached for `list_specialties` instead.

    CONFIRMED REAL PRODUCTION FAILURE: asked "هل تحب تعرف وش الخدمات
    المتوفرة في فرع الدقي؟", the patient said "اه", and the reply was
    "الخدمات المتاحة في فرع الدقي هي نفس التخصصات المتوفرة في المستشفى"
    followed by four SPECIALTIES (طب اسنان، جراحة الجسم الزجاجي
    والشبكية، نساء و توليد، طب الباطنة). Those are booking specialties,
    not services, and they are hospital-wide rather than that branch's -
    two different wrong answers in one sentence. The branch's real
    catalogue at that moment held two services (فحص النظر، كشف عيادة
    النساء)."""

    if not messages or not session_id:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage, AIMessage as _AIMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _BARE_AFFIRMATION_RE.match(_norm_ar(text)):
        return ""

    previous_ai = None
    for msg in reversed(messages[:-1]):
        if isinstance(msg, _AIMessage):
            previous_ai = msg
            break
    if previous_ai is None:
        return ""

    previous_content = getattr(previous_ai, "content", "")
    previous_text = previous_content if isinstance(previous_content, str) else str(previous_content)

    if not _BRANCH_SERVICES_OFFER_RE.search(_norm_ar(previous_text)):
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    branch_name = (
        session.get("info_branch_name")
        or session.get("branch_display_name")
        or ""
    )

    return (
        "============================================================\n"
        "THEY SAID YES TO SEEING THIS BRANCH'S SERVICES\n"
        "============================================================\n"
        "You offered to show the services at "
        f"{branch_name or 'this branch'}, and they agreed. Call "
        "`list_branch_services` now"
        + (f" (branch_name=\"{branch_name}\")" if branch_name else "")
        + " and show exactly what it returns, numbered, then ask if "
        "they'd like details on one.\n\n"
        "Do NOT call `list_specialties`. Specialties are the BOOKING "
        "system's medical categories - a different list, for a different "
        "purpose, and hospital-wide rather than per-branch. Do NOT call "
        "`list_hospital_services` or `answer_hospital_faq` either: those "
        "read the knowledge-base file, which has no per-branch data.\n\n"
        "Never say a branch's services \"are the same as the hospital's "
        "specialties\" - that is two wrong claims at once.\n"
        "  - \"not_found\": say plainly that this branch publishes no "
        "services right now. Do not substitute another list.\n\n"
        "CONFIRMED REAL PRODUCTION FAILURE: this exact turn answered "
        "with \"الخدمات المتاحة في فرع الدقي هي نفس التخصصات المتوفرة في "
        "المستشفى\" and four specialty names, while the branch's real "
        "catalogue held two actual services.\n\n"
    )


_SAME_WHATSAPP_QUESTION_RE = re.compile(
    r"نفس\s*رقم\s*(?:ال)?واتساب|نفس\s*(?:ال)?رقم\s*(?:اللي|الذي|ده)|"
    r"same\s*whatsapp\s*number|same\s*number\s*you"
)

# ONE shared word-list for "a phone number", reused by every pattern
# below so a synonym added in one place can't silently go missing from
# another. CONFIRMED REAL PRODUCTION FAILURE this fragment fixes: the
# patient answered STEP 1 with "رقم التليفون" ("telephone number") -
# a completely standard synonym, already recognised by the older
# `_ASKS_FOR_PHONE_RE` elsewhere in this file - but the guards below,
# written separately, only recognised "جوال"/"هاتف"/"phone"/"mobile"
# and missed "تليفون" entirely, so the patient's clear choice of phone
# went unrecognised and the reference option got wrongly re-offered.
_PHONE_WORD_FRAGMENT = (
    r"(?:رقم\s*(?:ال)?جوال|رقم\s*(?:ال)?موبايل|رقم\s*(?:ال)?تليفون|"
    r"رقم\s*(?:ال)?هاتف|mobile\s*number|phone\s*number|telephone\s*number)"
)

_ASKS_FOR_PHONE_DIRECTLY_RE = re.compile(_PHONE_WORD_FRAGMENT)

# SCOPING, NOT A TRIGGER ON ITS OWN. The "same WhatsApp number?"
# question ALSO appears in the completely unrelated NEW BOOKING flow
# (STEP NB6: "نكمل الحجز على نفس رقم الواتساب ده؟") - which has no
# "reference or phone?" STEP 1 at all, because there is no existing
# booking to identify yet. Since `concierge` handles every flow, this
# guard must only fire when the reply is actually about modifying or
# cancelling an EXISTING appointment - never on a plain "نكمل الحجز"
# (continuing a brand-new one), which does not mention "تعديل"/
# "إلغاء"/"موعدك" at all.
_CANCEL_OR_RESCHEDULE_CONTEXT_RE = re.compile(
    r"تعديل|إلغاء|الغاء|موعدك|"
    r"cancel\s*(?:your|the)?\s*appointment|reschedul"
)

# The STEP 1 question itself - "reference or phone?" - in any of the
# forms the clinic's own dialect/templates might render it, in Arabic
# or English. Matched LOOSELY (either keyword pair, in either order)
# since the exact wording varies per clinic template and per LLM
# phrasing.
_REFERENCE_OR_PHONE_QUESTION_RE = re.compile(
    r"(?:رقم\s*(?:ال)?حجز|(?:ال)?رقم\s*(?:ال)?مرجعي|reference\s*(?:number)?).{0,40}"
    + _PHONE_WORD_FRAGMENT + r"|"
    + _PHONE_WORD_FRAGMENT +
    r".{0,40}(?:رقم\s*(?:ال)?حجز|(?:ال)?رقم\s*(?:ال)?مرجعي|reference\s*(?:number)?)"
)

# A booking reference ("GBN-2026-06-20-151") or anything that looks
# like a real phone number (8+ digits) in the patient's OWN message -
# STEP 1's "smart detection" legitimately skips the question when
# either is already present, so this guard must not fire in that case.
_REF_OR_PHONE_GIVEN_RE = re.compile(r"[A-Za-z]{2,}-\d|\+?\d{8,}")

_IDENTITY_VERIFICATION_TOOLS = (
    "lookup_appointment", "compare_phone", "send_otp", "verify_otp", "check_booking_status",
)


def _reply_skips_reference_or_phone_question(reply_text: str, state: AgentState) -> bool:
    """True when a cancel/reschedule reply jumps straight to a
    phone-specific question - "same WhatsApp number?" or an open
    "send your phone number" - without STEP 1's "reference number or
    phone number?" question ever having been asked anywhere earlier in
    this conversation, and without the patient's own message ever
    having supplied a reference or a phone number themselves (which is
    the one case STEP 1 is explicitly allowed to skip via its own
    "smart detection" rule).

    CONFIRMED REAL PRODUCTION FAILURE: the patient said "عاوزه اعدل
    معاد" (wants to modify an appointment) - no reference, no phone -
    and the very next message was "نكمل تعديل موعدك على نفس رقم
    الواتساب ده؟", skipping straight past the reference-or-phone choice
    STEP 1 exists to ask. The patient never got a chance to say they'd
    rather identify the booking by its reference number."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)
    asks_same_number = bool(_SAME_WHATSAPP_QUESTION_RE.search(folded))
    asks_for_phone_directly = bool(_ASKS_FOR_PHONE_DIRECTLY_RE.search(folded))

    if not (asks_same_number or asks_for_phone_directly):
        return False

    # SCOPING - see `_CANCEL_OR_RESCHEDULE_CONTEXT_RE` above. Check the
    # reply itself first (the clinic's own "same number?" template
    # usually says "تعديل موعدك" in the same sentence), and fall back
    # to the patient's own last message (covers the bare "send your
    # phone number" phrasing, which doesn't always repeat "موعدك"
    # itself) - either one confirms this is actually the cancel/
    # reschedule flow and not an unrelated new booking.
    from langchain_core.messages import HumanMessage as _HumanMessage2
    last_human_text = ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, _HumanMessage2):
            content = getattr(msg, "content", "")
            last_human_text = content if isinstance(content, str) else str(content or "")
            break

    if not (
        _CANCEL_OR_RESCHEDULE_CONTEXT_RE.search(folded)
        or _CANCEL_OR_RESCHEDULE_CONTEXT_RE.search(_norm_ar(last_human_text))
    ):
        return False

    # The reply itself IS the "reference or phone?" question (presents
    # both options together) - that's STEP 1 being asked correctly
    # right now, not skipped. Without this, the question's own mention
    # of "phone number" as one of the two choices was matching
    # `_ASKS_FOR_PHONE_DIRECTLY_RE` and flagging STEP 1's own question
    # as having skipped itself.
    if _REFERENCE_OR_PHONE_QUESTION_RE.search(folded):
        return False

    messages = state.get("messages") or []

    # Already past STEP 1 in this conversation - an identity-
    # verification tool has already run, so a phone-specific follow-up
    # here is a legitimate LATER step, not a skipped first one.
    for msg in messages:
        if getattr(msg, "name", None) in _IDENTITY_VERIFICATION_TOOLS:
            return False

    from langchain_core.messages import AIMessage as _AIMessage, HumanMessage as _HumanMessage

    # STEP 1's own question already appeared somewhere earlier.
    for msg in messages:
        if isinstance(msg, _AIMessage):
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            if _REFERENCE_OR_PHONE_QUESTION_RE.search(_norm_ar(text)):
                return False

    # The patient already supplied a reference or a phone number
    # themselves - STEP 1's smart-detection rule allows skipping the
    # question in that case.
    for msg in messages:
        if isinstance(msg, _HumanMessage):
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            if _REF_OR_PHONE_GIVEN_RE.search(text):
                return False

    return True


_REFERENCE_OR_PHONE_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "ASK \"REFERENCE OR PHONE?\" FIRST - DO NOT SKIP STEP 1\n"
    "============================================================\n"
    "Your previous draft jumped straight to a phone-specific question "
    "(\"same WhatsApp number?\" or asking for the phone number directly) "
    "without ever asking the patient whether they'd rather identify "
    "their booking by its REFERENCE NUMBER or by PHONE NUMBER. The "
    "patient's own message contained neither, so this choice is theirs "
    "to make, not yours to assume.\n\n"
    "Rewrite this reply to ask that question instead - naturally, in "
    "this clinic's own dialect, e.g. \"تحب تلغي/تعدل الموعد برقم الحجز "
    "ولا برقم الجوال؟\". Only once they answer \"phone\" do you move on "
    "to the same-WhatsApp-number question.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: \"عاوزه اعدل معاد\" (no "
    "reference, no phone) was answered with \"نكمل تعديل موعدك على نفس "
    "رقم الواتساب ده؟\" - skipping the choice entirely.\n\n"
)


_CHOSE_PHONE_PATH_RE = re.compile(
    r"^\s*ب?\s*(?:رقم\s*)?(?:ال)?جوال\s*$|^\s*ب?\s*(?:رقم\s*)?(?:ال)?هاتف\s*$|"
    r"^\s*ب?\s*(?:رقم\s*)?(?:ال)?تليفون\s*$|^\s*ب?\s*(?:رقم\s*)?(?:ال)?موبايل\s*$|"
    r"^\s*(?:by\s*)?phone\s*(?:number)?\s*$|^\s*(?:by\s*)?mobile\s*(?:number)?\s*$|"
    r"^\s*(?:by\s*)?telephone\s*(?:number)?\s*$"
)

_CHOSE_REFERENCE_PATH_RE = re.compile(
    r"^\s*ب?\s*(?:رقم\s*)?(?:ال)?حجز\s*$|^\s*ب?\s*(?:ال)?رقم\s*(?:ال)?مرجعي\s*$|"
    r"^\s*(?:by\s*)?reference\s*(?:number)?\s*$"
)


def _reply_reoffers_reference_after_phone_chosen(reply_text: str, state: AgentState) -> bool:
    """True when the reply re-offers the booking-reference option
    ("...أو رقم الحجز") even though the patient already explicitly
    chose to identify by PHONE at STEP 1, earlier in this same
    conversation - re-opening a choice that was already made.

    CONFIRMED REAL PRODUCTION FAILURE: STEP 1 asked "تحب تعدل موعدك
    باستخدام رقم الحجز ولا رقم الجوال؟", the patient answered "رقم
    الجوال" - a clear, specific choice - then said "لا" to the
    same-WhatsApp-number follow-up. The next message asked "من فضلك
    أرسل رقم الجوال مع رمز الدولة أو رقم الحجز الخاص بك..." - reference
    number, back on the table, despite the patient never having been
    given a reason to reconsider. Once "phone" is chosen, every
    following question in this identification step must stay about
    phone numbers only - a different NUMBER is fine to ask for, a
    different METHOD is not, unless the patient brings it up again
    themselves."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)

    if not _REFERENCE_OR_PHONE_QUESTION_RE.search(folded):
        return False

    messages = state.get("messages") or []

    # Already past STEP 1 with a real lookup underway - a reference
    # number offered again here (e.g. a genuine "not found, try
    # something else?" recovery) is a different, legitimate situation,
    # not this one.
    for msg in messages:
        if getattr(msg, "name", None) in _IDENTITY_VERIFICATION_TOOLS:
            return False

    from langchain_core.messages import AIMessage as _AIMessage3, HumanMessage as _HumanMessage3

    # Scan for the exact pattern: an AI message asking STEP 1's
    # reference-or-phone question, immediately followed by a human
    # reply that is a bare, specific choice of "phone" (not "reference",
    # not something ambiguous - only a clean single-method pick counts).
    chose_phone = False
    asked_step1 = False
    for msg in messages:
        if isinstance(msg, _AIMessage3):
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            asked_step1 = bool(_REFERENCE_OR_PHONE_QUESTION_RE.search(_norm_ar(text)))
        elif isinstance(msg, _HumanMessage3) and asked_step1:
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content or "")
            folded_reply = _norm_ar(text)
            if _CHOSE_PHONE_PATH_RE.search(folded_reply):
                chose_phone = True
            elif _CHOSE_REFERENCE_PATH_RE.search(folded_reply):
                # They chose reference, not phone - re-offering both
                # later would be a different (currently unaddressed)
                # situation, not this bug.
                chose_phone = False
            asked_step1 = False

    return chose_phone


_REFERENCE_REOFFER_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "THE PATIENT ALREADY CHOSE \"PHONE\" - DON'T RE-OFFER \"REFERENCE\"\n"
    "============================================================\n"
    "Your previous draft asked for the phone number OR the booking "
    "reference - but the patient already specifically answered \"رقم "
    "الجوال\" (phone) when STEP 1 asked them to choose. Re-opening that "
    "choice now reads as if their answer was never registered.\n\n"
    "Rewrite the reply to ask ONLY for the phone number - e.g. \"من "
    "فضلك أرسل رقم الجوال مع رمز الدولة\" - with no mention of the "
    "booking reference as an alternative. If a phone-based lookup later "
    "genuinely comes back empty, THAT is when offering the reference "
    "number as a fallback becomes appropriate - not here, right after "
    "they picked phone and simply declined the same-WhatsApp-number "
    "shortcut.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: patient answered \"رقم الجوال\" "
    "at STEP 1, said \"لا\" to \"نكمل تعديل موعدك على نفس رقم الواتساب "
    "ده؟\", and was then asked for \"رقم الجوال ... أو رقم الحجز الخاص "
    "بك\" - reopening a decision already made one turn earlier.\n\n"
)


def _reply_reasks_identity_after_verification(reply_text: str, state: AgentState) -> bool:
    """True when the reply asks for a phone number or booking reference
    to identify the patient, even though identity is ALREADY fully
    established for this session: a phone number has already been
    verified (via `compare_phone`/`verify_otp`) AND `lookup_appointment`
    has already found a real booking earlier in this same conversation.
    Re-asking at that point isn't a scoping edge case - it's asking the
    patient to prove who they are a second time after they already did.

    WHY THIS IS SEPARATE FROM `_reply_skips_reference_or_phone_question`:
    that guard requires the reply (or the last human message) to
    mention "تعديل"/"إلغاء" wording, specifically so it doesn't misfire
    on the unrelated new-booking flow's own "same number?" question.
    That scoping is exactly why it MISSES this failure - many turns
    into an active, fully-identified reschedule flow, a bare "لا"
    carries none of that wording, yet the identity work was already
    done several turns earlier and must never be re-requested.

    CONFIRMED REAL PRODUCTION FAILURE: OTP succeeded, `lookup_appointment`
    found the booking ("👤 الاسم: حنين ايمن..."), the flow proceeded all
    the way to offering a specific reschedule time ("هل يناسبك هذا
    الموعد؟"), the patient said "لا" (declining that TIME, nothing to
    do with identity), and the very next reply was "أعطني رقم جوالك مع
    رمز الدولة أو رقم الحجز عشان أقدر أساعدك بالخطوة الجاية" - discarding
    a fully verified identity and a found booking to ask for both all
    over again."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)
    asks = bool(_ASKS_FOR_PHONE_DIRECTLY_RE.search(folded)) or bool(
        _REFERENCE_OR_PHONE_QUESTION_RE.search(folded)
    )
    if not asks:
        return False

    session_id = state.get("session_id")
    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    if not session.get("verified_phones"):
        return False

    for msg in state.get("messages") or []:
        if getattr(msg, "name", None) != "lookup_appointment":
            continue
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content or "")
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict) and data.get("status") in ("found_one", "found_many"):
            return True

    return False


_IDENTITY_REASK_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "IDENTITY IS ALREADY VERIFIED - DO NOT ASK FOR PHONE/REFERENCE AGAIN\n"
    "============================================================\n"
    "Your previous draft asked the patient for their phone number or "
    "booking reference - but this session already has a verified phone "
    "number AND `lookup_appointment` already found their booking earlier "
    "in this same conversation. Whatever the patient just said, it was "
    "not a request to re-identify themselves - continue the flow from "
    "where it actually is instead (the next date/time question, the "
    "confirmation step, or whatever comes next), never a fresh "
    "identification request.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: a fully verified reschedule flow "
    "(OTP succeeded, booking found, a specific new time offered) had the "
    "patient decline that TIME with a bare \"لا\" - and the reply then "
    "asked for the phone number or booking reference all over again, as "
    "if none of the identification that already happened had happened. "
    "\"لا\" to a time offer means \"show me another time\", not \"forget "
    "who I am.\"\n\n"
)


_NEW_BOOKING_SAME_NUMBER_QUESTION_RE = re.compile(
    r"نكمل\s*الحجز\s*.{0,10}نفس\s*رقم\s*(?:ال)?واتساب|"
    r"continue\s*(?:the\s*)?booking\s*.{0,10}same\s*whatsapp"
)


def _reply_asks_same_number_before_booking_ready(reply_text: str, state: AgentState) -> bool:
    """True when a NEW BOOKING reply asks "same WhatsApp number?"
    (STEP NB6) before a doctor is confirmed AND a time slot is
    actually selected in this booking session - i.e. before STEP NB6's
    own documented precondition is met.

    CONFIRMED REAL PRODUCTION FAILURE: the patient's very first message
    was "حجز جديد" (new booking, no specialty/doctor/time mentioned at
    all), and the very next reply was "نكمل الحجز على نفس رقم واتساب
    هذا؟ ✅" - STEP NB6's phone question, before STEP NB1 (specialty or
    doctor?) had even been asked once. When the patient said "اه" to
    that premature question, the NEXT reply then asked "وش التخصص اللي
    حابة تحجزين فيه؟" - the flow's actual FIRST question, now arriving
    dead last and out of order, reading as nonsensical to the patient."""

    if not reply_text:
        return False

    folded = _norm_ar(reply_text)

    if not _NEW_BOOKING_SAME_NUMBER_QUESTION_RE.search(folded):
        return False

    session_id = state.get("session_id")
    session = tools._BOOKING_SESSIONS.get(session_id) or {}

    doctor_ready = bool(session.get("doctor_id"))
    slot_ready = bool(session.get("selected_slot"))

    if doctor_ready and slot_ready:
        return False

    return True


_PREMATURE_SAME_NUMBER_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "TOO EARLY FOR \"SAME WHATSAPP NUMBER?\" - NO DOCTOR/SLOT YET\n"
    "============================================================\n"
    "Your previous draft asked whether to continue the booking on the "
    "same WhatsApp number - STEP NB6 - but no doctor is confirmed and/or "
    "no time slot has been selected in this booking session yet. STEP "
    "NB6 is explicitly the LAST step, only reached after a doctor AND a "
    "specific time slot are both locked in.\n\n"
    "Rewrite this reply to continue the flow from wherever it actually "
    "is instead.\n\n"
    "FIRST, USE WHAT THE PATIENT ALREADY TOLD YOU - do not throw it "
    "away and do not restart from the menu. If their messages so far "
    "already named a doctor, a specialty, a branch, or a day, carry "
    "straight on from there: match the doctor with "
    "`match_entity_for_booking` (a slightly misspelled name is still a "
    "name - match it rather than asking them to retype it), or take up "
    "the specialty/branch/day they gave, and ask only the next thing "
    "genuinely still missing.\n\n"
    "ONLY if nothing at all has been chosen yet, ask STEP NB1's opening "
    "question - whether they'd like to start by specialty or by doctor "
    "name. Never ask for phone/WhatsApp confirmation before the "
    "appointment itself (doctor, branch, day, time) is fully settled.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: \"حجز جديد\" (no specialty, no "
    "doctor mentioned) was answered with \"نكمل الحجز على نفس رقم "
    "واتساب هذا؟ ✅\" - the LAST step of the flow, asked as the FIRST "
    "reply. The patient said yes, and the very next message then asked "
    "\"وش التخصص اللي حابة تحجزين فيه؟\" - the flow's real first "
    "question, arriving after the phone question instead of before it.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE (2026-08-31): the opposite "
    "mistake, made while correcting this one. The patient opened with "
    "\"حجز مع دكتور احمد عقيا يوم الثلاثاء\" - a doctor AND a day - and "
    "the corrected reply printed the full 7-item specialty menu, "
    "discarding both. They answered \"دكتور احمد\" and were told to "
    "write the doctor's full name, which they had already given twice. "
    "Correcting a premature phone question must never cost the patient "
    "information they already provided.\n\n"
)


_FORGOT_TO_LOCK_SLOT_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "LOCK THE SLOT IN FIRST - THEN KEEP YOUR CONFIRMATION SENTENCE\n"
    "============================================================\n"
    "Your previous draft asked STEP NB6's same-WhatsApp-number "
    "question, and the doctor IS confirmed and a time list WAS shown to "
    "the patient - so the flow is in the right place. The only thing "
    "missing is mechanical: `select_appointment_slot` has not been "
    "called yet, so the time the patient just picked is not actually "
    "locked into this booking session.\n\n"
    "Call `select_appointment_slot` now with the slot they chose. Then "
    "send the SAME reply you already drafted, KEEPING the sentence that "
    "tells them which appointment was selected (day, date, time, doctor) "
    "before the WhatsApp-number question.\n\n"
    "Do NOT rewind the conversation to an earlier step, do NOT re-ask "
    "which day or which time, and do NOT drop the confirmation sentence "
    "- the patient must always be told exactly which appointment they "
    "are confirming before being asked about the phone number.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-31 12:46:47): "
    "the draft correctly said \"تم اختيار موعد الساعة 2:30 مساءً يوم "
    "الأحد 06/09/2026 مع دكتور نور عبد الرحمن\" followed by the "
    "WhatsApp question. The correction dropped that first sentence "
    "entirely, so the patient was asked to confirm a booking without "
    "ever being told which appointment it was.\n\n"
)


def _premature_same_number_directive(reply_text: str, state: AgentState) -> str:
    """Which of the two corrections this actually needs.

    The guard fires for two genuinely different situations, and telling
    them apart matters: a booking that has not started yet must go back
    to STEP NB1, but a booking whose doctor is confirmed and whose slot
    list has already been shown is in exactly the right place and just
    needs the slot committing - sending it back to NB1 there destroys
    correct work."""

    session = tools._BOOKING_SESSIONS.get(state.get("session_id")) or {}
    last_list = session.get("last_list") or {}

    if session.get("doctor_id") and last_list.get("entity_type") == "slot":
        return _FORGOT_TO_LOCK_SLOT_CORRECTION_DIRECTIVE

    return _PREMATURE_SAME_NUMBER_CORRECTION_DIRECTIVE


def _reply_wrongly_scope_refuses_after_otp_failure(reply_text: str, state: AgentState) -> bool:
    """True when the reply is the fixed out-of-scope refusal, but the
    most recent tool call in this conversation was `verify_otp`
    returning "otp_invalid" - meaning the patient's message should have
    been treated as ANOTHER OTP attempt (per this project's own
    explicit rule: "the next message after THAT is also automatically
    treated as the OTP"), not judged to be an unrelated, out-of-scope
    topic.

    CONFIRMED REAL PRODUCTION FAILURE: `verify_otp` returned
    "otp_invalid", the reply correctly said the code was wrong and
    asked to try again - then the patient answered "لا صحيح انا
    متاكده" (ambiguous pushback, not a 6-digit code) and got the
    clinic's canned "عذرًا 🌷 أنا لطيفة... مختصة بمساعدتك في خدمات
    المستشفى..." scope refusal, completely abandoning the OTP retry
    the patient was still actually mid-way through."""

    if not reply_text:
        return False

    templates = state.get("templates") or {}
    if not _is_scope_refusal(reply_text, templates):
        return False

    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "name", None) != "verify_otp":
            continue
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content or "")
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = None
        return isinstance(data, dict) and data.get("status") == "otp_invalid"

    return False


_OTP_FAILURE_SCOPE_REFUSAL_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "THIS IS AN OTP RETRY, NOT AN OUT-OF-SCOPE MESSAGE\n"
    "============================================================\n"
    "Your previous draft used the out-of-scope refusal, but the most "
    "recent thing that happened in this conversation was an INCORRECT "
    "OTP code. The patient is still in the middle of verifying their "
    "phone number - this exchange never left hospital-service scope, "
    "even if their message doesn't look like a fresh 6-digit code.\n\n"
    "Per this project's own rule: the message right after a failed OTP "
    "is ALWAYS treated as another OTP attempt. Call `verify_otp` again "
    "with the patient's exact message as the `otp` argument and the "
    "SAME phone number as before. If it's invalid again, tell them "
    "plainly and ask them to double-check the code that arrived and "
    "resend it - or offer a human handoff if this keeps failing. Never "
    "fall back to the generic capability menu here.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: after a wrong OTP, the patient "
    "wrote \"لا صحيح انا متاكده\" (ambiguous pushback, not a code) and "
    "was told \"عذرًا 🌷 أنا لطيفة... مختصة بمساعدتك في خدمات "
    "المستشفى...\" - the generic scope refusal, abandoning the OTP "
    "verification they were still actively going through.\n\n"
)


def _reply_asks_for_a_phone_already_known(reply_text: str, state: AgentState) -> bool:
    """True when the reply asks the patient for their phone number (or a
    booking reference in its place) immediately after they agreed to
    proceed on the channel number the service already has.

    CONFIRMED REAL PRODUCTION FAILURE: "نكمل تعديل موعدك على نفس رقم
    الواتساب ده؟" -> "اه" -> "ممكن تعطيني رقم الحجز أو رقم جوالك عشان
    أقدر أجيب بيانات موعدك؟". The assistant asked a question, was given
    an answer, and asked for the very thing it had just been granted.

    Requires ALL of: a channel number exists, the patient's last message
    is a bare yes, and the reply asks for a number - so a patient who
    genuinely wants to use a different number is unaffected, because
    that is never a bare "اه".
    """

    if not reply_text or not state.get("channel_phone"):
        return False

    if not _ASKS_FOR_PHONE_RE.search(_norm_ar(reply_text)):
        return False

    if _SUMMARY_OR_CONFIRMATION_CUE_RE.search(_norm_ar(reply_text)):
        # A summary/confirmation message restating the phone as an
        # already-settled field, not a request for it - see the
        # comment above _SUMMARY_OR_CONFIRMATION_CUE_RE.
        return False

    from langchain_core.messages import HumanMessage as _HumanMessage

    for msg in reversed(state.get("messages", []) or []):
        if not isinstance(msg, _HumanMessage):
            continue
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        return bool(_BARE_AFFIRMATION_RE.match(_norm_ar(text)))

    return False


_PHONE_ALREADY_KNOWN_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "YOU ASKED FOR A NUMBER YOU ALREADY HAVE - REWRITE\n"
    "============================================================\n"
    "Your previous draft asked the patient for their phone number, or "
    "for a booking reference instead of it. They have just answered "
    "\"yes\" to YOUR OWN question about continuing on the WhatsApp "
    "number they are messaging from. You have that number. Asking for "
    "it now means you asked a question, got an answer, and ignored "
    "it.\n\n"
    "Do not reply with a question at all this turn. Call "
    "`lookup_appointment` with the channel number given at the top of "
    "this prompt, and answer from what it returns.\n\n"
    "Only a patient who explicitly says they want a DIFFERENT number "
    "should ever be asked to type one - and that is never a bare "
    "\"اه\".\n\n"
)


_SELECTED_SLOT_DIRECTIVE = (
    "============================================================\n"
    "THE APPOINTMENT TIME IS ALREADY LOCKED IN - DO NOT ASK FOR IT AGAIN\n"
    "============================================================\n"
    "`select_appointment_slot` has already resolved and saved this "
    "booking's time. It is:\n"
    "    {date_display} {weekday_display} — {time_display}"
    "{service_suffix}\n\n"
    "Use these exact values whenever the flow needs them - especially "
    "`slot_start`/`slot_end` for `create_new_booking` - and never ask "
    "the patient for the time, the date, or which slot they meant "
    "again, no matter how many other questions (phone number, name, "
    "email) come between now and the booking itself.\n\n"
    "CONFIRMED REAL PRODUCTION FAILURE: a patient picked a slot, was "
    "asked to confirm their WhatsApp number, answered \"yes\" - and was "
    "then asked to give the appointment time again, as if the earlier "
    "answer had never happened. It had; nothing had reminded the model "
    "of it across the intervening question, so it was lost. This "
    "reminder exists so that never happens again: the values above are "
    "not something to recall from earlier in the conversation, they are "
    "written here, every turn, for exactly as long as this booking is "
    "in progress.\n\n"
)


def _build_selected_slot_directive(session_id: str) -> str:
    """Fires whenever this booking has a locked-in slot
    (`select_appointment_slot` succeeded) and the booking has not yet
    completed - keeps the exact chosen time in front of the model on
    every turn, the same way `_build_channel_identity_directive` keeps
    the phone number in front of it, so neither has to survive purely on
    the model's recollection of an earlier turn."""

    if not session_id:
        return ""

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    slot = session.get("selected_slot")

    if not slot:
        return ""

    service_name = (slot.get("serviceName") or "").strip()
    service_suffix = f" — {service_name}" if service_name else ""

    return _SELECTED_SLOT_DIRECTIVE.format(
        date_display=slot.get("date_display") or "",
        weekday_display=slot.get("weekday_display") or "",
        time_display=slot.get("time_display") or "",
        service_suffix=service_suffix,
    )


_ASKS_FOR_SLOT_RE = re.compile(
    r"وقت\s*(?:بالضبط|محدد)|الوقت\s*الذي|أي\s*وقت\s*تفضل|"
    r"الرقم\s*من\s*(?:ال)?قائمه|من\s*(?:ال)?قائمه\s*(?:ال)?سابقه|"
    r"exact\s*time|which\s*(?:time|slot)"
)


_ASKS_WHICH_DAY_RE = re.compile(
    r"أي\s*يوم|انهي\s*يوم|أنهي\s*يوم|اي\s*يوم|which\s*day"
)


def _reply_reasks_day_patient_already_named(reply_text: str, state: AgentState) -> bool:
    """True when the reply asks the patient which day they want, while
    their OWN latest message already named a weekday that matches one
    of the days most recently remembered (from `list_available_days_
    for_booking`) - AND the reply shows no times (so it genuinely is a
    re-ask, not a legitimate "these times don't work, want another day"
    follow-up after already showing that day's slots).

    CONFIRMED REAL PRODUCTION FAILURE: the patient answered "الاثنين",
    which the model correctly used to resolve which of two branches
    they meant - but the reply then asked "أي يوم يناسبك للحجز؟" again,
    re-showing the exact same day list, instead of extracting Monday's
    from_date/to_date and calling `get_available_slots_for_booking`.
    The patient had to type "الاثنين" a second time before the times
    were finally shown.
    """

    if not reply_text or not _ASKS_WHICH_DAY_RE.search(_norm_ar(reply_text)):
        return False

    # If the reply already contains a numbered list of times (rather
    # than days), this isn't the failure pattern - it's a normal times
    # list that happens to also mention "day" in passing.
    if re.search(r"صباح|مساء|am\b|pm\b|:\d{2}", reply_text, re.IGNORECASE):
        return False

    session_id = state.get("session_id")
    if not session_id:
        return False

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    last_list = session.get("last_list") or {}
    if last_list.get("entity_type") != "day":
        return False

    remembered_days = last_list.get("items") or []
    if not remembered_days:
        return False

    last_human = ""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", None) == "human":
            last_human = _norm_ar(str(getattr(m, "content", "") or ""))
            break

    if not last_human:
        return False

    for day in remembered_days:
        weekday_display = _norm_ar(str(day.get("weekday_display") or ""))
        weekday_name_en = str(day.get("weekday_name") or "").strip().lower()
        if weekday_display and weekday_display in last_human:
            return True
        if weekday_name_en and weekday_name_en in last_human.lower():
            return True

    return False


_DAY_ALREADY_NAMED_CORRECTION_DIRECTIVE = (
    "============================================================\n"
    "THE DAY WAS ALREADY NAMED - DO NOT RE-ASK, SHOW THE TIMES\n"
    "============================================================\n"
    "Your previous draft asked the patient which day they want. They "
    "already named a day, in their own last message, that matches one "
    "of the days you had just listed. Do NOT ask which day again and do "
    "NOT re-show the day list.\n\n"
    "Take that day's from_date/to_date from the day list you already "
    "have (do not recompute or guess it) and call "
    "get_available_slots_for_booking for it right now, then show the "
    "times in this same reply - exactly as STEP NB4 describes for "
    "'when they pick one of the days you listed'.\n\n"
)


def _reply_asks_for_a_slot_already_locked_in(reply_text: str, state: AgentState) -> bool:
    """True when the reply asks the patient for the appointment time,
    while a slot has already been resolved via `select_appointment_slot`
    for this booking.

    CONFIRMED REAL PRODUCTION FAILURE: a patient picked slot "2", was
    asked to confirm their WhatsApp number, answered "yes" - and was
    then asked to give the time again, as if the earlier answer had
    never happened. `_build_selected_slot_directive` now reminds the
    model of the locked-in time on every turn; this catches the case
    where the reminder didn't hold, the same escalation already applied
    to the phone-number equivalent in pass 11.
    """

    if not reply_text or not _ASKS_FOR_SLOT_RE.search(_norm_ar(reply_text)):
        return False

    session_id = state.get("session_id")
    if not session_id:
        return False

    session = tools._BOOKING_SESSIONS.get(session_id) or {}
    return bool(session.get("selected_slot"))


def _selected_slot_correction_directive(reply_text: str, state: AgentState) -> str:
    session = tools._BOOKING_SESSIONS.get(state.get("session_id")) or {}
    slot = session.get("selected_slot") or {}

    return (
        "============================================================\n"
        "THE TIME WAS ALREADY GIVEN - DO NOT ASK FOR IT AGAIN\n"
        "============================================================\n"
        "Your previous draft asked for the appointment time. It is "
        "already locked in for this booking:\n"
        f"    {slot.get('date_display', '')} {slot.get('weekday_display', '')} — "
        f"{slot.get('time_display', '')}\n\n"
        "Continue the booking from wherever it actually stands - do not "
        "ask for the time, the date, or which slot they meant.\n\n"
        "Rewrite the reply now, without asking for the time.\n\n"
    )


def _drop_orphaned_tool_calls(messages: list) -> list:
    """Remove AI messages whose tool_calls never received a matching
    ToolMessage, plus any ToolMessage with no matching call.

    WHY: OpenAI rejects a conversation where an assistant message with
    `tool_calls` is not followed by a response for every tool_call_id -
    with a 400, not a soft failure. And a turn can end mid-pair for
    reasons that have nothing to do with the model: the graph hitting
    its step ceiling, a crash between the agent and the tools node, a
    process restart. The half-pair is then PERSISTED by the
    checkpointer, so every later turn on that thread replays it and
    gets the same 400 forever.

    CONFIRMED REAL PRODUCTION FAILURE: after a turn ended without
    completing its tool calls, the next message returned
    "openai.BadRequestError ... tool_call_ids did not have response
    messages: call_N1c0Tp9HPZv9NF09d6zNrnV2" as an HTTP 500. The
    session was permanently unusable - every subsequent message failed
    the same way, because the bad pair never went away on its own.

    Dropping the orphan loses one incomplete step; keeping it loses the
    entire conversation."""

    messages = list(messages or [])

    responded_ids = {
        getattr(m, "tool_call_id", None)
        for m in messages
        if getattr(m, "type", None) == "tool"
    }
    responded_ids.discard(None)

    called_ids = set()
    for m in messages:
        for call in (getattr(m, "tool_calls", None) or []):
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            if call_id:
                called_ids.add(call_id)

    cleaned = []
    dropped = 0

    for m in messages:
        calls = getattr(m, "tool_calls", None) or []
        if calls:
            ids = [
                (c.get("id") if isinstance(c, dict) else getattr(c, "id", None))
                for c in calls
            ]
            if any(cid and cid not in responded_ids for cid in ids):
                dropped += 1
                continue

        if getattr(m, "type", None) == "tool":
            tool_call_id = getattr(m, "tool_call_id", None)
            if tool_call_id and tool_call_id not in called_ids:
                dropped += 1
                continue

        cleaned.append(m)

    if dropped:
        logger.warning(
            "_drop_orphaned_tool_calls: removed %d message(s) with unmatched tool calls - "
            "a previous turn ended mid-pair (step ceiling, crash or restart). Without this "
            "the provider rejects the whole conversation with a 400 and the session is "
            "permanently stuck.",
            dropped,
        )

    return cleaned


def _run_agent(state: AgentState, agent_name: str) -> dict:
    """The body every specialist runs. Calls the LLM with that
    specialist's SCOPED system prompt + the full chat history, and
    decides whether to call a tool or reply directly.

    MULTI-AGENT NOTE: everything below this docstring is the original
    single-agent node, unchanged in behaviour. Exactly two things differ:
    the system prompt is `build_agent_system_prompt(templates,
    agent_name)` instead of the full one, and the LLM is `_llm_for(
    agent_name)` instead of the single global binding. Every directive,
    the greeting guarantee, the one-question trimming and the emoji list
    numbering are shared by all specialists precisely BECAUSE they live
    here rather than in each agent - that is what stops the specialists
    from developing different output habits.

    GREETING GUARANTEE: if this call produces a final reply (no
    tool_calls, i.e. this turn is about to end) and the conversation
    hasn't been greeted yet, the clinic's exact opening greeting text is
    deterministically prepended in code - not left to the LLM to
    reproduce from the system prompt's reference phrases. This was
    added because relying on the LLM alone measurably did not keep the
    greeting's exact wording/structure consistent across separate
    conversations, despite explicit instructions to reuse it verbatim.
    The opening line specifically is swapped for a time-of-day salutation
    when the user's own first message signals one (see
    _personalized_greeting) - the rest of the template is untouched.

    DOUBLE-GREETING FIX: on the first turn, the LLM is ALSO told, via an
    extra instruction appended to the system message, not to write any
    greeting/opener of its own, AND not to jump ahead into asking about a
    reference/phone number before the user has actually said they want
    to cancel something - a bare greeting like "صباح الخير" states no
    intent yet, so the reply should be the greeting's own closing
    question only, waiting for the user's actual next message.

    DETERMINISTIC LANGUAGE DIRECTIVE: which language this reply must be
    in is computed by code (_detect_target_language) from the
    conversation's actual messages and placed at the very TOP of the
    system message - not left to the prose LANGUAGE & DIALECT rule
    buried inside the (long) system prompt, which measurably was not a
    strong enough signal on its own to keep replies in the user's actual
    language, even for a conversation that had been purely English from
    its first message."""

    target_language = _detect_target_language(state["messages"])
    language_directive = _LANGUAGE_DIRECTIVE.get(target_language, "")

    latest_user_message = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            latest_user_message = msg.content
            break

    no_symptom_directive = (
        _NO_SYMPTOM_YET_DIRECTIVE
        if _is_generic_medical_guidance_request(latest_user_message)
        else ""
    )

    slots_directive = _build_slots_numbered_list_directive(state["messages"])
    available_days_directive = _build_available_days_directive(state["messages"], state.get("session_id"))
    channel_identity_directive = _build_channel_identity_directive(state.get("channel_phone"))
    services_directive = _build_services_from_kb_directive(state["messages"])
    empty_branch_directive = _build_empty_branch_directive(state["messages"])
    branch_pick_directive = _build_branch_pick_directive(
        state["messages"], state.get("session_id"),
    )
    day_pick_directive = _build_day_pick_directive(
        state["messages"], state.get("session_id"),
    )
    service_chosen_directive = _build_service_chosen_directive(
        state["messages"], state.get("session_id"),
    )
    service_named_directive = _build_service_named_directive(
        state["messages"], state.get("session_id"),
    )
    negation_directive = _build_negation_directive(state["messages"])
    doctors_scope_directive = _build_doctors_scope_directive(
        state["messages"], state.get("session_id"),
    )
    branch_services_yes_directive = _build_branch_services_affirmation_directive(
        state["messages"], state.get("session_id"),
    )
    empty_branch_booking_directive = _build_empty_branch_booking_intent_directive(
        state["messages"], state.get("session_id"),
    )
    branches_only_directive = _build_branches_only_no_doctors_directive(state["messages"])
    empty_day_directive = _build_empty_day_recovery_directive(state["messages"])
    how_to_book_directive = _build_how_to_book_directive(state["messages"])
    wrong_tool_directive = _build_wrong_tool_in_booking_flow_directive(state["messages"], state.get("session_id"))

    # CRITICAL: these two directives both trigger on the SAME
    # lookup_appointment/check_booking_status tool result, and they say
    # opposite things - one pre-builds the appointment block "to include
    # verbatim", the other says "do NOT present these results". Confirmed
    # real production failure: with both present, the display directive
    # (earlier in the concatenation) won and the unrelated patient's
    # booking was shown anyway. When the wrong-tool directive fires, the
    # display directive must be suppressed entirely.
    appointment_display_directive = (
        "" if wrong_tool_directive
        else _build_appointment_display_directive(state["messages"])
    )
    schedule_display_directive = _build_schedule_display_directive(state["messages"])
    day_confirmation_directive = _build_day_confirmation_requires_tool_directive(state["messages"])
    show_soonest_directive = _build_show_soonest_day_directive(state["messages"], state.get("session_id"))

    # THE DAY THE PATIENT NAMED. `_build_show_soonest_day_directive`
    # already stands itself down when this one fires (see its own
    # comment) - the two say opposite things and must never both be in
    # the prompt.
    #
    # SCOPED TO THE SPECIALISTS THAT ACTUALLY HOLD THESE TOOLS. Every
    # instruction below names `match_entity_for_booking` and
    # `resolve_available_day`; `cancel`, `medical`, `faq` and
    # `complaint` are not bound to either, so handing them these
    # directives would order a call they cannot make. `reschedule` DOES
    # have `resolve_available_day`, but it runs its own day flow (STEPs
    # R3-R5) against an EXISTING booking, and a second, differently
    # worded set of day rules competing with that is how contradictory
    # directives caused trouble in the first place.
    booking_side = agent_name in _NEW_BOOKING_AGENTS
    named_day_directive = (
        _build_named_day_directive(state["messages"], state.get("session_id"))
        if booking_side else ""
    )
    day_unavailable_directive = (
        _build_day_unavailable_directive(state["messages"], state.get("session_id"))
        if booking_side else ""
    )
    multi_intent_directive = (
        _build_multi_intent_directive(state["messages"], state.get("session_id"))
        if booking_side else ""
    )
    booking_confirmation_directive = _build_booking_confirmation_requires_tool_directive(state["messages"], state.get("session_id"))
    booking_success_directive = _build_booking_success_display_directive(state["messages"], state.get("templates"))

    # The three display blocks added to close the remaining gaps: a
    # resolved date, an entity list (doctors/specialties/branches), and
    # the two terminal outcomes (cancelled / rescheduled). Before these,
    # those were the only patient-facing results still formatted freehand
    # by the model, which is exactly where its output shape varied from
    # one patient to the next.
    resolved_day_directive = _build_resolved_day_directive(state["messages"], state.get("session_id"))
    entity_list_directive = _build_entity_list_directive(state["messages"])
    terminal_success_directive = _build_terminal_success_directive(state["messages"], state.get("templates"))

    # Only the booking specialist is told to clear a half-finished
    # booking, and only when it has just TAKEN OVER the conversation -
    # i.e. the patient walked away from one booking and has started
    # another. A patient still inside their original booking is already
    # owned by this specialist from the previous turn, so nothing is
    # reset underneath them mid-flow.
    abandoned_booking_directive = ""
    if agent_name == "booking" and state.get("active_agent") == "booking":
        previous_owner = state.get("previous_agent")
        if previous_owner and previous_owner != "booking":
            abandoned_booking_directive = _build_abandoned_booking_directive(
                state["messages"], state.get("session_id"),
            )

    bare_entity_directive = _build_bare_entity_answer_directive(state["messages"])
    branches_info_directive = _build_branches_info_directive(state["messages"])
    bare_doctor_directive = _build_bare_doctor_answer_directive(state["messages"])
    show_all_doctors_directive = _build_show_all_doctors_after_ask_directive(state["messages"])
    doctor_branches_directive = _build_doctor_branches_directive(
        state["messages"], state.get("session_id"),
    )
    branch_question_directive = _build_branch_question_directive(
        state["messages"], state.get("session_id"), agent_name,
    )
    scope_directive = _build_scope_directive(
        state.get("templates") or {}, target_language or "ar",
    )

    # HEALTH MESSAGES THE SCOPE REFUSAL MUST NOT ANSWER. Built for every
    # specialist, not just `medical`: neither of these messages scores
    # on any router cue, so they stay with whichever agent was already
    # active - and that is exactly how a patient asking for a dose ended
    # up with the service menu.
    supplied_identifier_directive = (
        _build_supplied_identifier_directive(state["messages"])
        if agent_name in _EXISTING_BOOKING_AGENTS else ""
    )

    # "الغيه" / "عدله" about the booking already on the table. Suppressed
    # when the patient typed a reference of their own - that one wins,
    # and the directive above is already acting on it.
    just_booked_directive = (
        _build_just_booked_directive(state["messages"])
        if (agent_name in _EXISTING_BOOKING_AGENTS
            and not supplied_identifier_directive) else ""
    )

    crisis_directive = _build_crisis_directive(
        state["messages"], state.get("templates") or {},
    )
    medication_directive = _build_medication_request_directive(
        state["messages"], state.get("templates") or {},
    )
    review_phone_directive = _build_review_card_phone_directive(state, state.get("session_id"))
    selected_slot_directive = _build_selected_slot_directive(state.get("session_id"))

    # The scoped prompt for whoever owns this turn. Rebuilt per turn for
    # the same reason load_config rebuilds: a prompts.py/CSV edit must
    # reach conversations already in progress. The split itself is
    # cached by content hash inside agents.sections, so this is string
    # assembly, not re-parsing, on all but the first turn after a change.
    #
    # ORDER MATTERS. `state["system_prompt"]` is now only populated on
    # the legacy single-agent path (see load_config's comment on why the
    # 130 KB string is no longer written into every checkpoint), so the
    # multi-agent branch is checked FIRST and the stored value is only a
    # fallback. The final `build_system_prompt` fallback covers the one
    # remaining case - single-agent mode resuming a thread checkpointed
    # before this change, where the field is absent - so no path can
    # reach the LLM with an empty system prompt.
    templates = state.get("templates") or {}

    if config.MULTI_AGENT_ENABLED and templates:
        scoped_prompt = build_agent_system_prompt(templates, agent_name)
    else:
        scoped_prompt = state.get("system_prompt") or ""
        if not scoped_prompt and templates:
            scoped_prompt = build_system_prompt(templates)

    system_content = (
        language_directive + no_symptom_directive
        + services_directive + how_to_book_directive
        + slots_directive + available_days_directive
        + resolved_day_directive + entity_list_directive
        + abandoned_booking_directive + bare_entity_directive
        + branches_info_directive
        + bare_doctor_directive + show_all_doctors_directive
        + doctor_branches_directive + branch_question_directive
        + review_phone_directive + selected_slot_directive
        + supplied_identifier_directive + just_booked_directive + scope_directive
        + empty_branch_directive + branch_pick_directive + day_pick_directive
        + negation_directive
        + service_chosen_directive + service_named_directive
        + doctors_scope_directive
        + branch_services_yes_directive
        + empty_branch_booking_directive
        + branches_only_directive + empty_day_directive
        + appointment_display_directive + schedule_display_directive
        + wrong_tool_directive + day_confirmation_directive
        # MULTI-INTENT FIRST, then the day rules: the first says which
        # rung of the flow this turn starts on, the second says what to
        # do about the day specifically. Both come AFTER the day-pick /
        # day-confirmation directives above, which resolve a pick from a
        # list already shown and are about a narrower situation.
        + multi_intent_directive + named_day_directive + day_unavailable_directive
        + show_soonest_directive
        + booking_confirmation_directive + booking_success_directive
        + terminal_success_directive
        + scoped_prompt
        # CHANNEL IDENTITY goes LAST, after scoped_prompt (STEP NB6/
        # complaint flow), not before it. Confirmed real production bug:
        # with this directive earlier in the concatenation, the model
        # still obeyed STEP NB6's own "ALWAYS ASK THIS - NOT OPTIONAL"
        # wording and asked the same-WhatsApp-number question even with
        # an empty channel_phone, because that later, more emphatic
        # instruction won out. Placing the (now equally emphatic, and
        # explicitly overriding) channel-identity directive AFTER it
        # gives it the final word for this turn.
        + channel_identity_directive
        # LAST, and in this order. Both override the scope refusal, and
        # the scope refusal is itself deliberately emphatic - the same
        # reason channel identity had to be moved down here. Crisis goes
        # after medication because if a message is somehow both, the
        # crisis response is the only acceptable reply.
        + medication_directive + crisis_directive
    )

    if not state.get("greeted"):
        system_content += (
            "\n\n============================================================\n"
            "FIRST-TURN OVERRIDE\n"
            "============================================================\n"
            "This is the first message of a new conversation. The opening "
            "greeting/persona introduction has ALREADY been (or will be) sent "
            "separately, outside of what you write here. Do NOT write any "
            "greeting, self-introduction, or generic opener of your own, in "
            "any language (no 'صباح النور'/'مساء النور'/'Hi there! How can I "
            "help?' or similar).\n\n"
            "IMPORTANT - do not jump ahead: if the user's message is just a "
            "greeting or small talk with no stated intent yet (e.g. "
            "'صباح الخير', 'hi', 'مرحبا', 'good morning', with nothing else), "
            "do NOT ask about a booking reference or phone number yet - you "
            "don't know they want to cancel anything yet. In that case, "
            "simply write NOTHING here (an empty reply is fine) and let the "
            "greeting's own closing question stand on its own, waiting for "
            "them to say what they need. Only start STEP 1 (asking to "
            "identify the booking) once the user's message actually "
            "indicates they want to cancel an appointment."
        )

    system_message = SystemMessage(content=system_content)

    # Cap how much history actually gets sent to the LLM (config.MAX_HISTORY_MESSAGES).
    # start_on="human" guarantees the trimmed slice begins on a HumanMessage,
    # so we never cut in the middle of an AIMessage(tool_calls=...) /
    # ToolMessage pair and leave a dangling, invalid tool call behind.
    # The checkpointer still keeps the untrimmed full history for the
    # thread - this only shrinks what's sent on THIS call.
    # Strip any half-finished tool-call pair BEFORE trimming, so a turn
    # that ended mid-pair can't poison every later turn on this thread
    # with a provider 400 - see _drop_orphaned_tool_calls.
    safe_messages = _drop_orphaned_tool_calls(state["messages"])

    trimmed_history = trim_messages(
        safe_messages,
        strategy="last",
        token_counter=len,  # count messages, not real tokens - simple cap
        max_tokens=config.MAX_HISTORY_MESSAGES,
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    # Safety net: for a pathologically small MAX_HISTORY_MESSAGES (or an
    # unusual message-shape edge case), trim_messages can legitimately
    # return an empty list rather than violate the start_on="human"
    # constraint. Sending the LLM an empty turn would be worse than
    # sending it the untrimmed history, so fall back rather than trim.
    history = trimmed_history if trimmed_history else safe_messages
    response = _invoke_llm_resilient(
        _llm_for(agent_name), [system_message] + history,
        agent_name=agent_name, target_language=target_language, context="main turn",
    )

    updates: dict = {}

    # Publish the detected language into state BEFORE the tools node
    # runs, so every tool formats its human-readable fields (times,
    # weekday names, doctor/branch/specialty names) in the language this
    # conversation is actually being held in - see
    # tools.conversation_language.
    # Written unconditionally (even as None): tools validate state
    # strictly, so this field must never be missing on a turn that makes
    # a tool call.
    updates["target_language"] = target_language

    has_tool_calls = bool(getattr(response, "tool_calls", None))

    # INTERIM "PLEASE WAIT" MESSAGE. This is the exact moment we know
    # tools are about to run and know WHICH ones, so it is the only place
    # that can tell the patient something specific ("جاري البحث عن
    # الأطباء" rather than a generic spinner). Nothing is sent yet - a
    # timer is armed, and main.send_message cancels it when the turn
    # ends, so a fast turn stays a single message. See progress.py.
    if has_tool_calls:
        progress.schedule(
            session_id=state.get("session_id") or "",
            client_id=state.get("client_id") or "",
            tool_names=[
                call.get("name", "")
                for call in response.tool_calls
                if isinstance(call, dict)
            ],
            language=target_language,
            templates=state.get("templates"),
            answering_a_list=_is_answering_a_list(state["messages"]),
            tool_args={
                call.get("name", ""): (call.get("args") or {})
                for call in response.tool_calls
                if isinstance(call, dict)
            },
            agent_name=agent_name,
            channel_phone=state.get("channel_phone"),
        )

    # ONE QUESTION PER MESSAGE - enforced here, not just asked for in the
    # prompt. Applied to every final reply, before the greeting logic
    # below, so a trimmed reply is what gets stored in history too (the
    # model must never see its own multi-question message as a
    # precedent for the next turn).
    if not has_tool_calls and response.content:
        normalized = _apply_output_contract(
            response.content, state, target_language, agent_name,
        )

        # THE REPLY VERIFIERS. Each one catches a claim the model wrote
        # that NO tool result in this conversation supports - the only
        # class of error the pre-write directives structurally cannot
        # prevent, because there is no tool call to shape.
        #
        # They run as a table rather than as six copy-pasted blocks: the
        # blocks had already drifted apart (only some logged at ERROR,
        # the branch one alone reported a failed correction), and every
        # single one of them re-emitted the corrected reply through
        # `_emojify_list_numbers` ALONE - skipping the question trimming
        # and the shared response contract that the first-pass reply had
        # just been put through. A corrected reply therefore came out in
        # a visibly different shape from every other reply, which is
        # exactly the inconsistency this project exists to remove. One
        # loop, one finaliser, one behaviour.
        #
        # `used_safe_fallback` tracks whether `normalized` ends this
        # block as the generic zero-tolerance fallback message (see
        # `_safe_fallback_reply` below) rather than a real answer - the
        # opening-greeting step further down must never staple a
        # greeting UNDER this message the way it deliberately does for
        # everything else, for the same reason it already excludes scope
        # refusals: "حدث خطأ تقني" is not a substantive reply the
        # greeting should introduce, it is this turn failing outright.
        # CONFIRMED REAL PRODUCTION FAILURE: medtown, session
        # 201158877175+medtown2, 2026-08-30 11:30 - the fallback message
        # was sent glued underneath the full opening greeting/menu,
        # because nothing downstream distinguished it from a normal
        # first reply.
        used_safe_fallback = False
        for check, correction_directive, description in _REPLY_VERIFIERS:
            if not check(normalized, state, agent_name):
                continue

            logger.error(
                "agent[%s]: %s | strict_mode=%s | reply=%r",
                agent_name, description, _BRANCH_VERIFIER_STRICT, normalized,
            )

            if not _BRANCH_VERIFIER_STRICT:
                continue

            directive = correction_directive(normalized, state)

            try:
                retry = _llm_for(agent_name).invoke(
                    [SystemMessage(content=directive + system_content)] + history
                )
            except (_OpenAIAPITimeoutError, _OpenAIAPIConnectionError) as exc:
                # Unlike the main turn's call, there is already a usable
                # (if unverified) reply sitting in `normalized` - a
                # timeout here should cost this one verifier's
                # correction, not the whole turn. Log it and move on to
                # the next verifier (or out of the loop) with the
                # original reply intact, rather than overwriting a
                # perfectly fine draft with a generic apology.
                logger.warning(
                    "agent[%s]: verifier correction call failed (%s: %s) - keeping the "
                    "original, unverified reply for check '%s' instead of retrying further",
                    agent_name, type(exc).__name__, exc, description,
                )
                continue

            if getattr(retry, "tool_calls", None):
                # It chose to go and fetch the real data instead of
                # rewriting from memory - much better. Let the normal
                # tools loop run and send nothing this pass.
                #
                # BUT THIS PATH CAN LOOP. Returning here sends us back
                # through the tools node and into this agent again, so
                # the same verifier can fire, retry, ask for tools, and
                # return once more - forever, if the check is one the
                # model cannot satisfy.
                #
                # CONFIRMED REAL PRODUCTION FAILURE: a false-positive
                # "not a diagnosis" check on a doctor-selection reply
                # spun this loop for roughly a hundred model calls over
                # two minutes, and the patient received NOTHING - the
                # turn simply never ended. A verifier being wrong should
                # cost one wasted call, never the whole conversation.
                if _verifier_tool_retries_exhausted(state):
                    logger.error(
                        "agent[%s]: verifier '%s' has already sent this turn back for "
                        "tools %d time(s) - accepting the reply as-is rather than "
                        "looping. THIS MEANS THE VERIFIER IS PROBABLY WRONG HERE; the "
                        "reply is being delivered unmodified.",
                        agent_name, description, _MAX_VERIFIER_TOOL_RETRIES,
                    )
                    break

                updates["messages"] = [retry]
                updates["target_language"] = target_language
                return updates

            if not retry.content:
                continue

            if check(retry.content, state, agent_name):
                # FAILED THE SAME CHECK TWICE. What happens now depends
                # on WHAT the check protects - see `_verifier_severity`.
                severity = _verifier_severity(description)

                if severity == _FLOW:
                    # Every word of this reply is true; it just asks the
                    # wrong question or sits at the wrong step. Sending
                    # it costs the patient one clumsy turn. Sending
                    # "حدث خطأ تقني" costs them the answer entirely, and
                    # if the verifier itself is the thing that is wrong -
                    # which twice in a row strongly suggests - it costs
                    # them a perfectly good answer.
                    #
                    # CONFIRMED REAL PRODUCTION FAILURE this prevents:
                    # "معنديش فرع اسمه النيل. لكن عندنا هالفروع المتاحة
                    # حاليًا: 1️⃣ المنار 2️⃣ النزهة" - correct, useful,
                    # and replaced with a technical error because a flow
                    # check misread it as re-asking a question.
                    logger.error(
                        "agent[%s]: FLOW check failed twice (%s) - keeping the reply "
                        "rather than replacing it with the technical-error message. "
                        "THE VERIFIER IS PROBABLY WRONG HERE; nothing it guards is "
                        "unsafe to send. Reply: %r",
                        agent_name, description, normalized,
                    )
                    continue

                # ZERO-TOLERANCE FALLBACK, for SAFETY checks only.
                #
                # Before this existed, failing the SAME check twice
                # still ended with the original, already-flagged reply
                # going out unmodified. CONFIRMED REAL PRODUCTION
                # FAILURE: the branch-name verifier logged this exact
                # "STILL failed after correction" error and the patient
                # was sent the flagged reply anyway five seconds later.
                #
                # A safety verifier firing twice means the model cannot
                # stop asserting something no tool supports. A generic
                # "try again" is a much better outcome than a
                # confidently wrong claim the patient may act on.
                logger.error(
                    "agent[%s]: reply STILL failed the same check after correction (%s) - "
                    "replacing with the safe fallback message rather than sending the "
                    "twice-flagged reply",
                    agent_name, description,
                )
                normalized = _safe_fallback_reply(state, target_language, description)
                used_safe_fallback = True
                continue

            logger.info("agent[%s]: corrected on retry (%s)", agent_name, description)
            normalized = _apply_output_contract(
                retry.content, state, target_language, agent_name,
            )

        if normalized != response.content:
            if not (normalized or "").strip():
                # EVERY processing step above can, in principle, remove
                # everything: the question trimmer, the scaffolding
                # stripper, the response contract. Any of them producing
                # an empty string means the patient gets NOTHING back -
                # a silent turn, which is the worst failure this service
                # has, because it looks identical to being ignored and
                # there is nothing for them to react to.
                #
                # Confirmed reachable: a reply consisting only of a
                # scaffolding line strips to "".
                #
                # The original draft is kept instead. It may be
                # imperfect, but an imperfect answer is recoverable and
                # silence is not.
                logger.error(
                    "agent[%s]: output contract emptied the reply - keeping the original "
                    "rather than sending nothing. Original: %r",
                    agent_name, response.content,
                )
            else:
                response = AIMessage(content=normalized)

    if not has_tool_calls and not state.get("greeted"):
        first_user_message = state["messages"][0].content if state["messages"] else ""
        greeting = _build_greeting(state.get("templates") or {}, first_user_message, target_language or "ar")

        reply_so_far = response.content or ""

        # MIXED-LANGUAGE GREETING GUARD.
        #
        # The reference-phrases section elsewhere in the system prompt
        # tells the LLM to reproduce the clinic's configured greeting
        # "EXACT... word for word" every time. Some clients' configured
        # `msg_unknown_fallback` is itself bilingual - one English
        # paragraph followed by one Arabic paragraph, written that way
        # on purpose for the very first, language-unknown contact.
        # Reproduced verbatim in a conversation whose language THIS turn
        # already knows (target_language, from _detect_target_language),
        # that puts both languages in the very first message the clinic
        # ever sends.
        #
        # CONFIRMED REAL PRODUCTION FAILURE: medtown, session
        # 201003365691+medtown2, 2026-08-30 - the patient's first
        # message was the English word "hi"; the reply sent back was
        # the full English paragraph immediately followed by the full
        # Arabic paragraph, one after another in the same message.
        #
        # `_already_contains_greeting`'s signature match only confirms
        # that the persona line for THIS conversation's language is
        # present somewhere in the reply - it says nothing about
        # whether the OTHER language's entire block also rode along, so
        # a bilingual raw template slips past it undetected (the
        # English signature line is genuinely in there; so is an entire
        # second paragraph in Arabic). Detect that specific shape - both
        # scripts present, and a reply too long to be an incidental
        # word or two in the other language - and treat it as NOT
        # correctly greeted yet, so the pure, single-language
        # deterministic greeting below still replaces it instead of
        # being skipped.
        # A LATIN PROPER NOUN IS NOT AN ENGLISH BLOCK. The test below
        # deliberately counts Latin WORDS rather than asking whether any
        # Latin letters exist at all: entity names routinely come back
        # from the API in English ("Al Nozha", "Dr Smith") and land in
        # an otherwise perfectly Arabic reply.
        #
        # CONFIRMED REAL PRODUCTION FAILURE (medtown, session
        # 201158877175+medtown2, 2026-08-31 12:44:53): a correct 576-
        # character Arabic reply listing a doctor's weekly schedule was
        # discarded in full and replaced by the bare greeting, because
        # its branch name rendered as "Al Nozha" - two Latin words.
        # The patient asked a question and got no answer at all. The
        # bilingual template this guard actually exists to catch
        # carries an entire English paragraph, so a word count
        # separates the two cleanly where a boolean cannot.
        mixed_language_greeting = bool(
            target_language in ("en", "ar")
            and _looks_arabic(reply_so_far)
            and _latin_word_count(reply_so_far) >= 12
            and len(reply_so_far) > 200
        )
        if mixed_language_greeting:
            logger.warning(
                "agent[%s]: opening reply carried BOTH an Arabic and an English "
                "block while target_language=%s - replacing with the pure "
                "single-language deterministic greeting instead of sending both. "
                "Original: %r",
                agent_name, target_language, reply_so_far,
            )

        if greeting and (mixed_language_greeting or not _already_contains_greeting(reply_so_far, greeting)):
            # A reply already flagged as bilingual is discarded entirely
            # rather than kept as "extra content" underneath the
            # corrected greeting - it is, by definition, a duplicate of
            # the greeting itself (in the wrong language too), not
            # additional substance the patient asked for.
            reply_content = "" if mixed_language_greeting else reply_so_far

            # THE REFUSAL NEVER RIDES ALONG WITH THE GREETING.
            #
            # CONFIRMED REAL PRODUCTION FAILURE: a patient opened with
            # "اهلا" and got the welcome message with the out-of-scope
            # refusal stapled underneath - told they were off-topic by
            # the first message the clinic ever sent them, for saying
            # hello. The directive now says this in words too, but
            # words are what failed; the greeting turn is a place where
            # this can be settled in code, so it is.
            #
            # A greeting is in scope by definition. If the model
            # produced nothing else of substance, the greeting alone IS
            # the whole reply - which is exactly what it should have
            # been.
            if reply_content and _is_scope_refusal(reply_content, state.get("templates") or {}):
                logger.warning(
                    "agent[%s]: the scope refusal was attached to the opening greeting - "
                    "dropped. A greeting is in scope. Original: %r",
                    agent_name, reply_content,
                )
                reply_content = ""

            if reply_content and used_safe_fallback:
                # See `used_safe_fallback` above - a twice-flagged reply
                # that got swapped for the generic technical-error
                # message is not a real answer to staple a greeting
                # onto; drop it here exactly as scope refusals are
                # dropped just above, so the patient's very first
                # message from the clinic isn't "hi! here's what I can
                # help with... 😕 technical error, try again?".
                logger.warning(
                    "agent[%s]: the zero-tolerance fallback message was about to be "
                    "attached to the opening greeting - dropped. Original: %r",
                    agent_name, reply_content,
                )
                reply_content = ""

            if reply_content and _is_redundant_closing_question_only(reply_content, greeting):
                reply_content = ""
            combined = f"{greeting.strip()}\n\n{reply_content}".strip() if reply_content else greeting.strip()
            response = AIMessage(content=combined)

        updates["greeted"] = True

    updates["messages"] = [response]

    return updates



def agent(state: AgentState) -> dict:
    """The legacy single-agent node, preserved.

    Kept as a public name because it is what the old graph exposed and
    what external callers/tests reference. It now simply runs the shared
    implementation as whichever specialist the router selected, falling
    back to the full-access concierge - which is the old behaviour
    exactly - when nothing has been routed."""

    return _run_agent(state, state.get("active_agent") or agents.CONCIERGE)


def _specialist_node(agent_name: str):
    """Builds the graph node for one specialist."""

    def node(state: AgentState) -> dict:
        return _run_agent(state, agent_name)

    node.__name__ = f"agent_{agent_name}"
    return node


# ==========================================================
# Router (the supervisor)
# ==========================================================

def router(state: AgentState) -> dict:
    """Chooses which specialist owns this turn.

    Runs ONCE per user turn, at the top of the graph - deliberately not
    inside the agent<->tools loop, so a turn that makes six tool calls
    still routes exactly once and cannot change owner half way through
    its own tool sequence.

    Costs nothing in the default configuration: routing is pure pattern
    matching over the latest human message (see agents/router.py for why
    that is a feature rather than a shortcut), so no LLM call and no
    added latency."""

    previous = state.get("active_agent")

    chosen, reason = agents.route_turn(state["messages"], previous)

    if chosen != previous:
        logger.info(
            "router: %s -> %s (%s)", previous or "none", chosen, reason,
        )

    # `previous_agent` records who owned the turn BEFORE this one, which
    # is what tells a specialist whether it has just taken the
    # conversation over or has been running the same flow all along. The
    # booking specialist uses it to decide whether a half-finished
    # booking is a detour to be preserved or an abandoned one to clear -
    # see _build_abandoned_booking_directive.
    return {
        "active_agent": chosen,
        "routing_reason": reason,
        "previous_agent": previous,
    }


def route_to_specialist(state: AgentState) -> str:
    """Conditional edge out of the router: the node name of the chosen
    specialist. Never raises - an unrecognised value lands on the
    concierge, which has the full prompt and every tool."""

    chosen = state.get("active_agent")

    # LEAVING FOR MEDICAL GUIDANCE CLEARS THE BRANCH CONTEXT.
    #
    # `info_branch_*` describes a branch the patient was browsing. Once
    # they start describing a symptom instead, that branch is no longer
    # what the conversation is about, and leaving the note behind makes
    # later guards reason about a branch nobody mentioned.
    #
    # CONFIRMED REAL FALSE POSITIVE: after browsing an empty فرع
    # الطوارئ, "عيني وجعاني وبتدمع" produced a perfectly good medical
    # reply that was rejected twice as "offering a booking at a branch
    # with no doctors" - the branch was simply stale session state.
    if chosen == "medical":
        session_id = state.get("session_id")
        if session_id:
            session = tools._BOOKING_SESSIONS.get(session_id)
            if session:
                for key in ("info_branch_no_doctors", "info_branch_id", "info_branch_name"):
                    session.pop(key, None)

    if chosen not in agents.AGENT_NAMES:
        return _node_name(agents.CONCIERGE)
    return _node_name(chosen)


def route_after_agent(state: AgentState) -> str:
    """If the LLM's latest message requested tool call(s), run them.
    Otherwise its message is the reply for this turn - end the graph."""

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


def route_after_tools(state: AgentState) -> str:
    """Tool results go back to the SAME specialist that asked for them -
    never through the router again, which is what keeps one turn's
    reasoning with one owner."""

    return route_to_specialist(state)


def _node_name(agent_name: str) -> str:
    return f"agent_{agent_name}"


# ToolNode automatically injects graph state into any tool parameter
# annotated with InjectedState (see tools.py's `state` params) without
# exposing it to the LLM's function-calling schema.
_tool_node = ToolNode(tools.ALL_TOOLS)


# ==========================================================
# Build graph
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node("load_config", load_config)
builder.add_node("tools", _tool_node)
builder.set_entry_point("load_config")

if config.MULTI_AGENT_ENABLED:
    # ------------------------------------------------------
    #  load_config -> router -> agent_<specialist> <-> tools -> END
    # ------------------------------------------------------
    # The ToolNode deliberately still holds tools.ALL_TOOLS, not a
    # per-specialist subset. Scoping happens at BINDING time (which
    # tools a specialist's LLM can even see), so a specialist can only
    # request tools it owns - but if one ever does request something
    # outside its subset (a replayed checkpoint, a caller-supplied LLM,
    # a future change to the registry), the call still executes instead
    # of dying with "tool not found" mid-conversation.

    builder.add_node("router", router)
    builder.add_edge("load_config", "router")

    specialist_nodes = {}
    for _name in agents.AGENT_NAMES:
        _node = _node_name(_name)
        builder.add_node(_node, _specialist_node(_name))
        builder.add_conditional_edges(
            _node, route_after_agent, {"tools": "tools", END: END},
        )
        specialist_nodes[_node] = _node

    builder.add_conditional_edges("router", route_to_specialist, specialist_nodes)
    builder.add_conditional_edges("tools", route_after_tools, specialist_nodes)

    logger.info(
        "graph: multi-agent mode - specialists: %s (tool scoping=%s, router=%s)",
        ", ".join(agents.AGENT_NAMES), config.AGENT_TOOL_SCOPING, config.ROUTER_MODE,
    )
else:
    # ------------------------------------------------------
    #  load_config -> agent <-> tools -> END   (legacy shape)
    # ------------------------------------------------------
    # The exact pre-multi-agent graph. Set MULTI_AGENT_ENABLED=false to
    # get it back without touching a line of code.

    builder.add_node("agent", agent)
    builder.add_edge("load_config", "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    logger.info("graph: single-agent mode (MULTI_AGENT_ENABLED is off)")

checkpointer = MemorySaver()

# WHO PROVIDES PERSISTENCE DEPENDS ON WHO IS RUNNING THIS GRAPH.
#
#   - Run through main.py / app.py (the FastAPI deployment, and the
#     tests): nobody else is managing threads, so the graph carries its
#     own MemorySaver, exactly as before.
#   - Run through the LangGraph API server (`langgraph dev`, or a
#     LangGraph Platform deployment): the server manages persistence
#     itself and REFUSES to load a graph that brings its own
#     checkpointer - it fails at startup with GraphLoadError rather than
#     quietly ignoring it. Confirmed by running `langgraph dev` against
#     this project.
#
# Detected by whether langgraph_api is loaded, so neither path needs a
# flag set by hand and neither can be broken by forgetting one.
_RUNNING_UNDER_LANGGRAPH_API = "langgraph_api" in sys.modules

if _RUNNING_UNDER_LANGGRAPH_API:
    logger.info("Running under the LangGraph API server - using its built-in persistence, not MemorySaver")
    graph = builder.compile()
else:
    graph = builder.compile(checkpointer=checkpointer)
