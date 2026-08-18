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
    state["system_prompt"] = build_system_prompt(templates)

    return state


_MORNING_CUES = ("صباح", "good morning", "morning")
_EVENING_CUES = ("مساء", "good evening", "evening")


_ENGLISH_GREETING_TEMPLATE = (
    "{salutation}\n"
    "I'm {agent_name}, the virtual assistant at {clinic_name}, and I'm happy to help you today.\n"
    "I can help you with:\n"
    "\U0001F5D3\uFE0F Booking a new appointment\n"
    "\u270F\uFE0F Modifying or cancelling an existing appointment\n"
    "\U0001FA7A Medical guidance to choose the right specialty or doctor\n"
    "\u2139\uFE0F Questions about the hospital's services and doctors\n"
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
    """

    if n == 10:
        return "🔟"
    if 1 <= n <= 9:
        return _NUMBER_EMOJIS[n]
    return "".join(_NUMBER_EMOJIS[int(digit)] for digit in str(n))


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
    by_branch: dict = {}
    branch_order = []
    for s in schedules:
        branch = s.get("branchName") or ""
        if branch not in by_branch:
            by_branch[branch] = []
            branch_order.append(branch)
        days = s.get("recurringDaysNames") or [""]
        from_time = _arabic_time_12h(s.get("fromDateTime"))
        to_time = _arabic_time_12h(s.get("toDateTime"))
        for day in days:
            arabic_day = _ARABIC_DAY_NAMES.get((day or "").strip().lower(), day)
            by_branch[branch].append((arabic_day, from_time, to_time))

    total_day_rows = sum(len(v) for v in by_branch.values())

    if len(branch_order) == 1 and total_day_rows == 1:
        # Format 1: single branch, single day
        branch = branch_order[0]
        day, from_time, to_time = by_branch[branch][0]
        block = (
            f"📍 الفرع: {branch}\n"
            f"👩\u200d⚕️ الطبيب: {doctor_name}\n"
            f"📅 اليوم: {day}\n"
            f"🕙 المواعيد المتاحة: من {from_time} حتى {to_time}"
        )
    elif len(branch_order) == 1:
        # Format 3: single branch, multiple days
        branch = branch_order[0]
        lines = [f"👩\u200d⚕️ الطبيب: {doctor_name}", f"📍 {branch}"]
        for day, from_time, to_time in by_branch[branch]:
            lines.append(f"📅 {day} | 🕙 {from_time} – {to_time}")
        block = "\n".join(lines)
    else:
        # Format 2: multiple branches (each may have one or more days)
        branch_blocks = []
        for branch in branch_order:
            branch_lines = [f"📍 {branch}"]
            for day, from_time, to_time in by_branch[branch]:
                branch_lines.append(f"📅 {day}\n🕙 {from_time} – {to_time}")
            branch_blocks.append("\n".join(branch_lines))
        block = f"👩\u200d⚕️ الطبيب: {doctor_name}\n" + "\n\n".join(branch_blocks)

    if total_day_rows == 1:
        only_day = next(iter(by_branch.values()))[0][0]
        closing_question_instruction = (
            f"  3. Exactly one question asking if they'd like to see the "
            f"available times for {only_day} (the only day shown) - do "
            f"NOT ask \"which day\" when only one day exists at all, "
            f"that's not a real choice and reads as confusing/redundant.\n"
        )
    else:
        closing_question_instruction = "  3. Exactly one question asking which day they'd prefer.\n"

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

    if not greeting:
        if target_language == "en":
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
        "(shown above, never to be printed in a reply). Wherever any flow "
        f"number as a VALUE - passing it to a tool, saving it on a "
        f"booking or a complaint - that number is {channel_phone}. Use it "
        "directly; never ask them to type a number they are already "
        "messaging from.\n\n"
        "DO NOT PRINT THE NUMBER IN YOUR REPLY. Both of you already know "
        "which number this is, so writing out the digits adds noise and "
        "makes a one-line question look like a form. Ask the short yes/no "
        "question exactly as the clinic wrote it - e.g. \"نكمل الحجز على "
        "نفس رقم الواتساب ده؟ ✅\" - with no digits, no country code, and "
        "no parenthetical. Then WAIT for their answer.\n\n"
        "If they say yes, use the number above as the phone value with no "
        "OTP. Only if they want a DIFFERENT number do the "
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
    booking_confirmation_directive = _build_booking_confirmation_requires_tool_directive(state["messages"], state.get("session_id"))
    booking_success_directive = _build_booking_success_display_directive(state["messages"], state.get("templates"))

    # The scoped prompt for whoever owns this turn. Rebuilt per turn for
    # the same reason load_config rebuilds: a prompts.py/CSV edit must
    # reach conversations already in progress. The split itself is
    # cached by content hash inside agents.sections, so this is string
    # assembly, not re-parsing, on all but the first turn after a change.
    scoped_prompt = state.get("system_prompt") or ""
    if config.MULTI_AGENT_ENABLED and state.get("templates"):
        scoped_prompt = build_agent_system_prompt(state["templates"], agent_name)

    system_content = (
        language_directive + no_symptom_directive
        + services_directive + how_to_book_directive
        + slots_directive + available_days_directive
        + empty_branch_directive + empty_day_directive
        + appointment_display_directive + schedule_display_directive
        + wrong_tool_directive + day_confirmation_directive
        + booking_confirmation_directive + booking_success_directive
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
        )

    # ONE QUESTION PER MESSAGE - enforced here, not just asked for in the
    # prompt. Applied to every final reply, before the greeting logic
    # below, so a trimmed reply is what gets stored in history too (the
    # model must never see its own multi-question message as a
    # precedent for the next turn).
    if not has_tool_calls and response.content:
        trimmed, removed = _strip_extra_questions(response.content, state.get("templates") or {})
        if removed:
            logger.warning(
                "agent: reply contained %d extra question(s) beyond the first - trimmed. Original: %r",
                removed, response.content,
            )

        # Every numbered list gets the same emoji badges, not just the
        # ones this file pre-builds.
        normalized = _emojify_list_numbers(trimmed)

        # THE SHARED OUTPUT CONTRACT. Runs on every final reply, from
        # every specialist, so which agent happened to own the turn can
        # never be visible in the shape of the answer: no filler opener,
        # no "let me check that", no persona re-introduction on turn 9,
        # no leaked routing language, same vertical spacing every time.
        # Deliberately placed AFTER the question trimming so the stored
        # history contains exactly what the patient saw - the model must
        # never see an unnormalized message of its own as a precedent.
        if config.REPLY_NORMALIZATION_ENABLED:
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
                    "agent[%s]: reply normalized to the shared response contract",
                    agent_name,
                )
            normalized = contracted

        if normalized != response.content:
            response = AIMessage(content=normalized)

    if not has_tool_calls and not state.get("greeted"):
        first_user_message = state["messages"][0].content if state["messages"] else ""
        greeting = _build_greeting(state.get("templates") or {}, first_user_message, target_language or "ar")

        if greeting and not _already_contains_greeting(response.content or "", greeting):
            reply_content = response.content or ""
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

    return {"active_agent": chosen, "routing_reason": reason}


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
