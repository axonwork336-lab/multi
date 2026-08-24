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
from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

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

    if len(days) == 1:
        # A single date is not a list - numbering one item reads oddly.
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

    # Same label order as everywhere else in this project, per the
    # response contract: doctor -> branch -> day -> date.
    lines = []
    if doctor_name:
        lines.append(f"👨\u200d⚕️ الطبيب: {doctor_name}")
    if branch_name:
        lines.append(f"📍 الفرع: {branch_name}")
    lines.append(f"📅 اليوم: {weekday} {date_display}".rstrip())

    block = "\n".join(lines)

    return (
        "[INTERNAL INSTRUCTION - NOT FOR THE USER - READ CAREFULLY]\n"
        "The nearest genuinely-available date was just resolved. Your "
        "ENTIRE reply must be the exact text between the START/END "
        "markers below, copied verbatim (translate the LABELS only if "
        "the conversation is in another language - keep the emoji, the "
        "date and the names unchanged either way), followed by exactly "
        "ONE question asking whether they'd like to see the available "
        "times on that day. The START/END marker lines themselves are "
        "NOT part of the text to copy - never include them, or any line "
        "of dashes/equals-signs, in your actual reply.\n\n"
        "Never write the result's `date` field (the ISO \"YYYY-MM-DD\" "
        "form) anywhere in your reply - it is a machine value. The block "
        "below already contains the date in the form the patient should "
        "see.\n\n"
        "When they say yes, your ONLY next action is to call "
        "`get_available_slots_for_booking` with this result's own "
        "`from_date`/`to_date`, copied verbatim - never a date you "
        "worked out yourself.\n\n"
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

    from langchain_core.messages import HumanMessage as _HumanMessage

    if not isinstance(messages[-1], _HumanMessage):
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
        "THIS BRANCH HAS NO AVAILABLE DOCTORS - DO NOT ANNOUNCE A LIST\n"
        "============================================================\n"
        f"The branch{(' (' + branch_name + ')') if branch_name else ''} "
        "was matched, but NOBODY is available there for this booking. "
        "There is no doctor list to show.\n\n"
        "Do NOT write \"here are the available doctors\" or anything like "
        "it - there are none, and saying otherwise followed by an empty "
        "list is a confirmed real failure that dead-ended a booking. "
        "Instead: say plainly that this branch has no available doctors "
        "right now, then call `list_branches_for_specialty` (or "
        "`find_available_doctors` with no branch) and offer the branches "
        "that DO have someone. Ask ONE question after that.\n\n"
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
    #   2. The configured greeting, when it is already in the right
    #      language for this conversation.
    #   3. The standard template rendered in the conversation's language,
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
            return "en"

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
_BRANCH_MENTION_RE = re.compile(r"\bفرع\s+([^\n،,.؟?:()\[\]0-9️⃣]{2,25})")

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

    return _norm_ar(" | ".join(str(p) for p in parts))


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

    invented = []
    for match in _BRANCH_MENTION_RE.finditer(reply_text):
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

_WEEKDAY_WORDS = {
    "الاثنين": "Monday", "الإثنين": "Monday", "الثلاثاء": "Tuesday",
    "الأربعاء": "Wednesday", "الاربعاء": "Wednesday", "الخميس": "Thursday",
    "الجمعة": "Friday", "السبت": "Saturday", "الأحد": "Sunday", "الاحد": "Sunday",
}

_AVAILABILITY_TOOLS = (
    "list_available_days_for_booking", "get_available_slots_for_booking",
    "get_available_reschedule_slots", "resolve_available_day",
    "get_doctor_schedule", "get_doctor_schedule_for_booking",
    "find_best_doctor_in_specialty", "lookup_appointment",
    "check_booking_status", "create_new_booking",
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
    weekdays = [d for d in _WEEKDAY_WORDS if d in reply_text]

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


def _reply_reoffers_doctor_roster_after_confirming_one(reply_text: str) -> bool:
    if not reply_text:
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
    "undo a choice they just made, and it is the wrong question "
    "entirely: what is still unknown at this point is the BRANCH.\n\n"
    "The next question is about the branch, and the branches you offer "
    "must be THAT DOCTOR'S branches - naming the doctor explicitly, so "
    "it is obvious the choice stands:\n"
    "    تحب تحجز في فرع معيّن، ولا أعرض لك الفروع اللي د. [اسم الدكتور] "
    "متاح فيها؟\n\n"
    "Never offer \"كل الدكاترة\" or a general branch list here - the "
    "patient has a doctor, so only that doctor's branches are relevant. "
    "Keep the confirmation line exactly as it was and rewrite only the "
    "question.\n\n"
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

_REPLY_VERIFIERS = (
    (
        lambda reply, state, agent_name: _reply_asks_for_a_slot_already_locked_in(reply, state),
        lambda reply, state: _selected_slot_correction_directive(reply, state),
        "reply asked for the appointment time, but a slot has already been locked in "
        "via select_appointment_slot for this booking",
    ),
    (
        lambda reply, state, agent_name: _reply_asks_for_a_phone_already_known(reply, state),
        lambda reply, state: _PHONE_ALREADY_KNOWN_CORRECTION_DIRECTIVE,
        "reply asked for a phone number (or a booking reference instead) right after "
        "the patient agreed to proceed on the channel number the service already has",
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
        lambda reply, state, agent_name: _reply_fabricates_handoff(reply, state),
        lambda reply, state: _HANDOFF_CORRECTION_DIRECTIVE,
        "reply confirmed a human handoff but request_human_handoff was never "
        "raised successfully in this conversation",
    ),
    (
        lambda reply, state, agent_name: _reply_reoffers_doctor_roster_after_confirming_one(reply),
        lambda reply, state: _DOCTOR_ROSTER_CORRECTION_DIRECTIVE,
        "reply confirmed a doctor then offered the doctor roster again in the same message",
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
_BARE_ENTITY_ANSWER_RE = re.compile(
    r"^\s*(?:"
    r"(?:ال)?(?:دكتور|دكتوره|دكتورة|دكاتره|دكاترة|طبيب|طبيبه|طبيبة|اطباء|أطباء)|"
    r"(?:ال)?(?:تخصص|تخصصات|قسم|اقسام|أقسام)|"
    r"(?:ال)?(?:فرع|فروع)|"
    r"doctors?|specialt(?:y|ies)|departments?|branch(?:es)?"
    r")\s*[.!؟?،,]*\s*$",
    re.IGNORECASE,
)

_BARE_ENTITY_ANSWER_DIRECTIVE = (
    "============================================================\n"
    "THEY ANSWERED WITH THE CATEGORY, NOT A NAME - SHOW THE LIST\n"
    "============================================================\n"
    "The patient's reply is the bare WORD (\"دكتور\", \"تخصص\", \"فرع\", "
    "\"doctor\", \"branch\") answering the choice you just offered them. "
    "It is an ANSWER, not the name of anything.\n\n"
    "They are telling you which way they want to go, and asking you to "
    "show them the options. So SHOW THEM, this turn:\n"
    "  - \"دكتور\"/\"doctor\"  -> call `find_available_doctors` (or "
    "`match_entity_for_booking` in list mode) and show the numbered list "
    "of doctors.\n"
    "  - \"تخصص\"/\"specialty\" -> call `list_specialties` and show the "
    "numbered list of specialties.\n"
    "  - \"فرع\"/\"branch\"    -> call `list_branches_for_specialty` and "
    "show the numbered list of branches.\n\n"
    "Do NOT reply by asking for a name (\"اسم الدكتور إيه؟\", \"أي "
    "تخصص؟\"). The patient does not know who works here or which "
    "specialties exist - that is exactly why they asked you to show "
    "them. Asking them to name one is asking for information only the "
    "system has, and it wastes a turn.\n\n"
    "Never fuzzy-match the bare word itself against a real doctor, "
    "specialty or branch name. Confirmed real production failure: the "
    "bare word \"فرع\" was matched to an actual branch the patient had "
    "never named or seen, and the whole booking then ran against the "
    "wrong one.\n\n"
)


def _build_bare_entity_answer_directive(messages: list) -> str:
    """Fires when the patient's latest message is just the category word
    ("دكتور"/"تخصص"/"فرع"), answering a choice the assistant offered.

    `tools._is_generic_entity_word` already stops the bare word being
    fuzzy-matched to a real entity once a tool is reached. This is the
    other half: making sure a tool is reached at all, rather than the
    model answering with a second question.
    """

    if not messages:
        return ""

    from langchain_core.messages import HumanMessage as _HumanMessage

    last = messages[-1]
    if not isinstance(last, _HumanMessage):
        return ""

    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)

    if not _BARE_ENTITY_ANSWER_RE.match(text.strip()):
        return ""

    return _BARE_ENTITY_ANSWER_DIRECTIVE


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
    "THE BRANCH QUESTION - EXACT PHRASING\n"
    "============================================================\n"
    "A doctor is settled for this booking. The next thing you do not "
    "know is the BRANCH, so that is the one question this turn.\n\n"
    "Ask it in this shape, with the doctor's real name in place of "
    "[اسم الدكتور]:\n"
    "    تحب تحجز في فرع معيّن، ولا أعرض لك كل الفروع اللي "
    "د. [اسم الدكتور] متاح فيهم؟\n\n"
    "The alternative you offer is THAT DOCTOR'S BRANCHES. Never offer "
    "\"كل الدكاترة المتاحين\" or any doctor list here - the doctor "
    "question is answered, and re-offering the roster invites them to "
    "undo a choice they just made. Confirmed in production five separate "
    "times.\n\n"
    "If they then ask to see the branches, call "
    "`get_doctor_schedule_for_booking` - that returns this doctor's own "
    "branches with the days and hours at each.\n\n"
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
    is confirmed yet - the exact point at which the branch question gets
    asked, and the point at which it has repeatedly been asked wrong."""

    if agent_name not in ("booking", "concierge") or not session_id:
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


def _build_out_of_scope_block(templates: dict) -> str:
    """The clinic's scope refusal, as ONE fixed text.

    Built from the client's own agent/clinic name so it is branded, and
    identical every time so an off-topic question gets the same answer
    for every patient - rather than the model improvising a different
    polite deflection each turn.

    A client can author their own via `msg_out_of_scope`.
    """

    authored = (templates or {}).get("msg_out_of_scope")
    if authored and authored.strip():
        return authored.replace("\r\n", "\n").replace("\r", "\n").strip()

    # The block is Arabic, so the ARABIC name fields come first. Using
    # `_agent_name`/`_clinic_name` here put "أنا Latifa، المساعدة
    # الافتراضية في Dar El Oyoun Hospitals" into an otherwise Arabic
    # sentence - a Latin-script name mid-sentence in RTL text, in the
    # one message that is supposed to be the clinic's most polished.
    templates = templates or {}
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


def _build_scope_directive(templates: dict) -> str:
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

    block = _build_out_of_scope_block(templates)

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

    block = _build_out_of_scope_block(templates)
    return bool(block) and _normalize_for_compare(block) in _normalize_for_compare(reply_text)


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

# A bare yes. The patient agreeing to a yes/no question the assistant
# itself asked.
_BARE_AFFIRMATION_RE = re.compile(
    r"^\s*(?:اه|ايه|أيوه|ايوه|ايوا|نعم|تمام|اوك|أوك|ok|okay|yes|yep|sure|"
    r"اكمل|كمل|اه\s*اكمل|ماشي|حاضر|طبعا|أكيد|اكيد)"
    r"\s*[.!؟?،,]*\s*$",
    re.IGNORECASE,
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
    doctor_branches_directive = _build_doctor_branches_directive(
        state["messages"], state.get("session_id"),
    )
    branch_question_directive = _build_branch_question_directive(
        state["messages"], state.get("session_id"), agent_name,
    )
    scope_directive = _build_scope_directive(state.get("templates") or {})
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
        + doctor_branches_directive + branch_question_directive
        + review_phone_directive + selected_slot_directive + scope_directive
        + empty_branch_directive + empty_day_directive
        + appointment_display_directive + schedule_display_directive
        + wrong_tool_directive + day_confirmation_directive + show_soonest_directive
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
    trimmed_history = trim_messages(
        state["messages"],
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
    history = trimmed_history if trimmed_history else state["messages"]
    response = _llm_for(agent_name).invoke([system_message] + history)

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

            retry = _llm_for(agent_name).invoke(
                [SystemMessage(content=directive + system_content)] + history
            )

            if getattr(retry, "tool_calls", None):
                # It chose to go and fetch the real data instead of
                # rewriting from memory - much better. Let the normal
                # tools loop run and send nothing this pass.
                updates["messages"] = [retry]
                updates["target_language"] = target_language
                return updates

            if not retry.content:
                continue

            if check(retry.content, state, agent_name):
                logger.error(
                    "agent[%s]: reply STILL failed the same check after correction (%s)",
                    agent_name, description,
                )
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

        if greeting and not _already_contains_greeting(response.content or "", greeting):
            reply_content = response.content or ""

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
