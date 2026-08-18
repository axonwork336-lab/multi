"""
Interim "please wait" messages, pushed while tools are still running.

THE PROBLEM THIS SOLVES
-----------------------
`/chat` is request/response: n8n posts one message and blocks until the
whole turn is finished. So a "جاري البحث عن دكاترة" line returned inside
that same response is worthless - it would arrive in the same instant as
the answer it was supposed to precede.

For the patient to actually SEE it while waiting, something has to reach
Messenger/WhatsApp *during* the turn, on a second channel. That is what
this module does: the moment the agent decides to call a tool, it pushes
the interim line straight to n8n's webhook, while the tool call carries
on in the main request.

    n8n --POST /chat--------------------------------> (blocked, working)
    agent --POST progress webhook--> n8n --> patient   "جاري البحث..."
    n8n <--------------reply------------------------- "لقيت لك 3 دكاترة"

WHY IT WAITS BEFORE SENDING
---------------------------
Firing instantly would be worse than not firing at all. Most tool calls
finish in a few hundred milliseconds, and "جاري البحث..." followed 400 ms
later by the answer reads as noise - two notifications for one question.

So the message is SCHEDULED, not sent: a timer fires it only if the turn
is still running after PROGRESS_DELAY_SECONDS. A fast turn cancels the
timer before it ever fires and the patient sees exactly one message, as
before. A genuinely slow turn - a doctor search across branches, a
schedule lookup - is the only case that produces two.

ONE PER TURN
------------
A single turn can make six tool calls in sequence. Six "please wait"
lines for one question is spam, so only the first one to fire is
delivered; the rest are suppressed until the next user message.

THE CLOCK RUNS FROM THE FIRST TOOL PHASE, NOT THE START OF THE TURN
--------------------------------------------------------------------
Not from each tool call individually, and NOT from when the patient's
message first arrived either - both matter, for different reasons.

Not from the start of the turn: the time between the message arriving
and the model deciding to call a tool is pure LLM "thinking" latency,
often 1-1.5s on its own - nothing to do with how slow the tool phase
itself is. Confirmed real production complaint: the interim message was
firing on almost every tool call, including ones backed by an instant
local check with no real external request at all, simply because that
upstream decision latency had already eaten most of the delay budget
before the first tool was even chosen. So the clock is anchored lazily,
the first time THIS turn actually calls schedule() - i.e. right when a
tool call is first decided - not at begin_turn().

Not per individual tool call either: a turn made of three 1.2s tool
calls takes 3.6s, which is exactly the kind of wait the patient should
be told about. But if every new tool call re-armed the timer, each one
would reset the countdown before the previous had elapsed, and the
message would never fire at all - the LONGEST turns would be the ones
that went silent. Confirmed by scenario 4 in test_progress_scenarios.py,
which existed before this was fixed and caught it.

So the elapsed time is measured against the first schedule() call of
the turn. A later tool call in the same turn only updates the WORDING
of the pending message (so it describes whatever is running now); it
never restarts the clock.

OFF BY DEFAULT
--------------
This needs a webhook URL and a corresponding n8n branch to exist, so
PROGRESS_ENABLED defaults to false. Nothing about the agent's behaviour
changes until it is switched on - see README_MULTIAGENT.md.
"""

import json
import logging
import threading
import time
from typing import Dict, Iterable, List, Optional

import config

logger = logging.getLogger(__name__)


# ==========================================================
# What to say, per kind of work
# ==========================================================

# Tools are grouped by what the PATIENT is waiting for, not by which
# module they live in - "جاري البحث عن دكاترة" is true whether the doctor
# came from find_available_doctors or find_best_doctor_in_specialty.
_TOOL_GROUPS: Dict[str, tuple] = {

    "searching_doctors": (
        "find_available_doctors",
        "find_best_doctor_in_specialty",
        "list_specialties",
    ),

    "searching_branches": (
        "list_branches_for_specialty",
    ),

    # Looking for which DAYS a doctor has anything open.
    "searching_slots": (
        "list_available_days_for_booking",
        "get_doctor_schedule_for_booking",
        "get_doctor_schedule",
        "resolve_available_day",
        "get_next_weekday_date",
    ),

    # Looking for the TIMES inside a day the patient has already been
    # given and accepted. Kept separate from "searching_slots" on
    # purpose: by this point the date is settled, so saying "جاري البحث
    # عن المواعيد" again reads as if the appointment that was just
    # offered is being searched for all over again.
    "searching_times": (
        "get_available_slots_for_booking",
        "get_available_reschedule_slots",
    ),

    "finding_booking": (
        "lookup_appointment",
        "check_booking_status",
    ),

    # Pulling up the PATIENT's own saved details (name, phone, email) to
    # show them for review. Kept out of "finding_booking" on purpose:
    # during a NEW booking there is no booking to look up yet, so
    # "جاري البحث عن الحجز" describes something that doesn't exist and
    # reads like the not-yet-created appointment is being searched for.
    "finding_patient_details": (
        "get_patient_info",
    ),

    "creating_booking": (
        "create_new_booking",
    ),

    "cancelling": (
        "cancel_appointment",
    ),

    "rescheduling": (
        "reschedule_appointment",
    ),

    "sending_otp": (
        "send_otp",
        "verify_otp",
    ),

    "checking_info": (
        "answer_hospital_faq",
        "list_hospital_services",
        "get_doctor_fees",
    ),

    "sending_complaint": (
        "send_complaint_email",
    ),
}

_GROUP_FOR_TOOL: Dict[str, str] = {
    tool: group for group, tools in _TOOL_GROUPS.items() for tool in tools
}

# Fast entity-resolution tools that finish almost instantly and are
# ALWAYS immediately followed, within the same turn, by the actual slow
# thing the patient is waiting for (a slot search, a booking lookup...).
# Confirmed real production bug: `match_entity_for_booking` alone was
# enough to fire "جاري البحث عن الأطباء" - by the time the patient saw
# it, that quick resolver had already finished and the turn had moved on
# to a genuinely slow call (e.g. list_available_days_for_booking), but
# "one message per turn" meant the now-stale wording never got
# corrected. So a schedule() call made up ENTIRELY of these tools is
# deliberately a no-op: no timer, no delivery - leaving the turn's timer
# unarmed so the next (slower, real) tool call is what actually fires
# the message, with the RIGHT wording.
_SILENT_RESOLVER_TOOLS = frozenset({
    "match_entity_for_booking",
    "match_entity_info",
})

# Order of precedence when one turn calls several tools at once: the
# patient should be told about the most significant thing happening, not
# whichever tool the model happened to list first.
_GROUP_PRIORITY = (
    "creating_booking",
    "cancelling",
    "rescheduling",
    "sending_complaint",
    "sending_otp",
    "searching_slots",
    "searching_branches",
    "searching_doctors",
    "finding_booking",
    "checking_info",
)


# Defaults, kept deliberately plain. They are NOT written in any one
# regional dialect, because this text is not composed by the LLM and so
# never passes through the LANGUAGE & DIALECT mirroring - a hard-coded
# Egyptian line would read wrong for a Saudi tenant. Anything more
# specific belongs in the tenant's own CSV column (see below).
_DEFAULT_MESSAGES: Dict[str, Dict[str, str]] = {
    "searching_doctors":  {"ar": "لحظة من فضلك، جاري البحث عن الأطباء المتاحين… 🔎",
                           "en": "One moment please - looking up the available doctors… 🔎"},
    "searching_branches": {"ar": "لحظة من فضلك، جاري البحث عن الفروع المتاحة… 🏥",
                           "en": "One moment please - looking up the available branches… 🏥"},
    "searching_slots":    {"ar": "لحظة من فضلك، جاري البحث عن المواعيد المتاحة… 🗓️",
                           "en": "One moment please - checking the available days… 🗓️"},
    "searching_times":    {"ar": "لحظة من فضلك، جاري البحث عن الأوقات المتاحة… 🕐",
                           "en": "One moment please - checking the available times… 🕐"},
    "finding_booking":    {"ar": "لحظة من فضلك، جاري البحث عن الحجز… 🔎",
                           "en": "One moment please - looking up your booking… 🔎"},
    "finding_patient_details": {"ar": "لحظة من فضلك، جاري البحث عن بياناتك… 🔎",
                           "en": "One moment please - looking up your details… 🔎"},
    "creating_booking":   {"ar": "لحظة من فضلك، جاري تأكيد الحجز… ⏳",
                           "en": "One moment please - confirming your booking… ⏳"},
    "cancelling":         {"ar": "لحظة من فضلك، جاري تنفيذ طلب الإلغاء… ⏳",
                           "en": "One moment please - processing the cancellation… ⏳"},
    "rescheduling":       {"ar": "لحظة من فضلك، جاري تعديل الموعد… ⏳",
                           "en": "One moment please - moving your appointment… ⏳"},
    "sending_otp":        {"ar": "لحظة من فضلك، جاري إرسال رمز التحقق… 📲",
                           "en": "One moment please - sending the verification code… 📲"},
    "checking_info":      {"ar": "لحظة من فضلك، جاري الاستعلام… ⏳",
                           "en": "One moment please - checking that for you… ⏳"},
    "sending_complaint":  {"ar": "لحظة من فضلك، جاري تسجيل الشكوى… ⏳",
                           "en": "One moment please - registering your complaint… ⏳"},
    "generic":            {"ar": "لحظة من فضلك، جاري تنفيذ طلبك… ⏳",
                           "en": "One moment please - working on that… ⏳"},
}


def message_for(
    tool_names: Iterable[str],
    language: Optional[str] = "ar",
    templates: Optional[dict] = None,
) -> str:
    """The interim line for whatever is about to run.

    A tenant can override any of these by adding a column to
    client_config.csv named `msg_progress_<group>` (e.g.
    `msg_progress_searching_doctors`) - which is how you get the line in
    that clinic's own dialect and voice. Absent that, the neutral default
    above is used.
    """

    groups = {_GROUP_FOR_TOOL[name] for name in tool_names if name in _GROUP_FOR_TOOL}

    group = next((candidate for candidate in _GROUP_PRIORITY if candidate in groups), "generic")

    if templates:
        override = templates.get(f"msg_progress_{group}") or templates.get("msg_progress")
        if override and override.strip():
            return override.replace("\r\n", "\n").replace("\r", "\n").strip()

    key = "en" if (language or "ar").startswith("en") else "ar"
    return _DEFAULT_MESSAGES.get(group, _DEFAULT_MESSAGES["generic"])[key]


# ==========================================================
# Per-turn state
# ==========================================================

_lock = threading.Lock()
_timers: Dict[str, threading.Timer] = {}
_delivered: Dict[str, bool] = {}

# Sessions whose turn is CURRENTLY being processed. Set by begin_turn,
# removed by end_turn.
#
# threading.Timer.cancel() only prevents a timer that hasn't started yet;
# a timer that already fired and is midway through _fire/_deliver keeps
# going regardless. That leaves a real race: the countdown elapses just
# as the turn finishes, and the "please wait" line is delivered AFTER
# the actual answer has already been sent. Confirmed in production from
# a patient's own chat - the answer arrived, and only then did "لحظة من
# فضلك، جاري البحث عن الأطباء المتاحين" show up underneath it, which
# reads as though the assistant started working after it had replied.
# _deliver re-checks this set immediately before sending, so a turn that
# has ended can no longer emit anything.
_in_flight: Dict[str, bool] = {}

# When the tool-running part of the current turn began, per session -
# i.e. the moment schedule() is FIRST called for this turn (right when
# the LLM decides to call a tool), NOT when the turn itself started.
#
# Deliberately NOT set in begin_turn(): the time between the patient's
# message arriving and the model deciding to call a tool is pure LLM
# "thinking" latency, not tool latency, and is typically 1-1.5s on its
# own - i.e. it can consume the entire PROGRESS_DELAY_SECONDS budget by
# itself. Confirmed real production complaint: the interim message was
# firing on effectively every tool call, including ones backed by an
# instant local check (e.g. "not_configured", no real request at all),
# because the clock had already been ticking since before the tool was
# even chosen. Anchoring here instead means the countdown measures only
# the part the patient is actually shown a message about: how long the
# tool phase itself is taking. A genuinely slow external call still
# fires normally; a fast/instant one - even inside a turn where the LLM
# calls themselves took a while - correctly does not.
_turn_started: Dict[str, float] = {}

# The message a pending timer will send when it fires: (client_id, text).
# Held separately from the timer so a later tool call in the same turn
# can update the WORDING without touching the countdown.
_pending: Dict[str, tuple] = {}

# Last message actually delivered, per session. Exposed for tests and for
# PROGRESS_MODE=log, where there is no webhook to inspect.
last_delivered: Dict[str, str] = {}


def begin_turn(session_id: str) -> None:
    """Called once when a user message starts being processed."""

    with _lock:
        _cancel_locked(session_id)
        _in_flight[session_id] = True
        _delivered[session_id] = False
        # NOT set here on purpose - see _turn_started's comment above.
        # Left over from a PRIOR turn shouldn't leak in either, so it's
        # explicitly cleared; schedule() sets it fresh the first time
        # it's actually called for this turn.
        _turn_started.pop(session_id, None)


def end_turn(session_id: str) -> None:
    """Called when the turn finishes, however it finishes.

    Cancelling here is what makes a fast turn produce no interim message
    at all - the timer never gets to fire.
    """

    with _lock:
        # Cleared before anything else: a timer thread already inside
        # _fire is blocked on this same lock right now, and this is what
        # it will see the moment it acquires it.
        _in_flight.pop(session_id, None)
        _cancel_locked(session_id)
        _delivered.pop(session_id, None)
        _turn_started.pop(session_id, None)
        _pending.pop(session_id, None)


def _cancel_locked(session_id: str) -> None:
    timer = _timers.pop(session_id, None)
    if timer is not None:
        timer.cancel()


def schedule(
    session_id: str,
    client_id: str,
    tool_names: Iterable[str],
    language: Optional[str] = "ar",
    templates: Optional[dict] = None,
) -> None:
    """Arm the interim message for a tool phase that is about to start.

    Never raises: a failure here must not be able to break a turn that
    would otherwise have answered the patient perfectly well.
    """

    if not config.PROGRESS_ENABLED:
        return

    try:
        names = list(tool_names or [])
        if not names:
            return

        if names and all(n in _SILENT_RESOLVER_TOOLS for n in names):
            # Nothing worth announcing yet - see _SILENT_RESOLVER_TOOLS.
            # Leave any already-armed timer from earlier in this turn
            # alone; just don't let THIS call arm one or overwrite the
            # pending wording with a resolver-only description.
            return

        text = message_for(names, language, templates)
        fire_now = False

        with _lock:
            if _delivered.get(session_id):
                # Already told them once this turn - saying it again for
                # the next tool call in the same turn is spam.
                return

            # Newest wording wins, so the message describes what is
            # actually running when it goes out.
            _pending[session_id] = (client_id, text)

            if session_id in _timers:
                # A countdown from this turn's tool phase is already
                # running. Leaving it alone is the whole point - re-arming
                # here is what used to make long turns silent.
                return

            # Lazily anchor the clock HERE, the first time this turn
            # actually reaches a tool call - not back at begin_turn(). See
            # _turn_started's module-level comment for why.
            started = _turn_started.get(session_id)
            if started is None:
                started = time.monotonic()
                _turn_started[session_id] = started

            elapsed = time.monotonic() - started
            remaining = config.PROGRESS_DELAY_SECONDS - elapsed

            if remaining <= 0:
                # The turn has already been slow enough - no reason to
                # wait any longer.
                fire_now = True
            else:
                timer = threading.Timer(remaining, _fire, args=(session_id,))
                timer.daemon = True
                _timers[session_id] = timer
                timer.start()

        if fire_now:
            _fire(session_id)

    except Exception:
        logger.warning("progress: could not schedule an interim message", exc_info=True)


def _fire(session_id: str) -> None:
    """Timer callback: send whatever wording is pending for this turn."""

    with _lock:
        _timers.pop(session_id, None)
        pending = _pending.get(session_id)

    if not pending:
        return

    client_id, text = pending
    _deliver(session_id, client_id, text)


def _deliver(session_id: str, client_id: str, text: str) -> None:
    """Runs on the timer thread, only if the turn is still going."""

    with _lock:
        if _delivered.get(session_id):
            return
        if not _in_flight.get(session_id):
            # The turn finished while this timer was on its way in. The
            # real reply has already gone out (or is about to), so an
            # interim "please wait" now would arrive after the answer.
            logger.info(
                "progress[%s]: suppressed %r - the turn already finished",
                session_id, text,
            )
            return
        _delivered[session_id] = True
        _timers.pop(session_id, None)
        last_delivered[session_id] = text

    if config.PROGRESS_MODE == "log":
        logger.info("progress[%s]: would send %r", session_id, text)
        return

    url = config.PROGRESS_WEBHOOK_URL
    if not url:
        logger.warning(
            "progress: PROGRESS_ENABLED is on but PROGRESS_WEBHOOK_URL is unset - "
            "no interim message can be delivered. Set the URL, or use PROGRESS_MODE=log."
        )
        return

    payload = {
        "type": "progress",
        "session_id": session_id,
        "client_id": client_id,
        "reply": text,
    }

    try:
        # Imported lazily so this module stays importable in environments
        # where nothing ever sends a webhook.
        import requests

        response = requests.post(
            url,
            json=payload,
            timeout=config.PROGRESS_TIMEOUT_SECONDS,
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "progress[%s]: sent %r (HTTP %s)", session_id, text, response.status_code,
        )
    except Exception as exc:
        # Swallowed on purpose. A "please wait" line that fails to send is
        # a cosmetic loss; it must never surface to the patient or
        # interfere with the real reply that is still being produced.
        logger.warning("progress[%s]: delivery failed (%s)", session_id, exc)


def pending_tools(messages: List) -> List[str]:
    """The tool names the latest AI message is about to call."""

    if not messages:
        return []

    last = messages[-1]
    calls = getattr(last, "tool_calls", None) or []
    return [call.get("name", "") for call in calls if isinstance(call, dict)]


def describe_config() -> str:
    if not config.PROGRESS_ENABLED:
        return "progress messages: off"

    target = "log only" if config.PROGRESS_MODE == "log" else (config.PROGRESS_WEBHOOK_URL or "UNSET")
    return (
        f"progress messages: on, after {config.PROGRESS_DELAY_SECONDS}s, "
        f"one per turn, to {target}"
    )
