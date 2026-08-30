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

import logging
import threading
import time
from typing import Dict, Iterable, Optional

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
    ),

    # Kept as a group with no tools of its own: `list_specialties` is
    # silent (see _SILENT_RESOLVER_TOOLS), but a tenant may still
    # override this wording via msg_progress_searching_specialties, and
    # _list_mode_alias can still select it.
    "searching_specialties": (
        # Announced only when the patient actually asked for the
        # specialty list (the booking flow) - it is silenced in medical
        # guidance, where it is an internal step. See the
        # `list_specialties` handling in schedule().
        "list_specialties",
    ),

    # A doctor search NARROWED TO A SPECIALTY, which is what the patient
    # has just agreed to when they say "اه" to "تحب أشوف لك الدكاترة في
    # التخصص ده؟". Worth its own wording: at that moment the specialty
    # is the shared context of the conversation, and naming it confirms
    # the assistant understood which one they meant.
    "searching_specialty_doctors": (),

    "searching_branches": (
        "list_branches_for_specialty",
    ),

    # Looking for WHICH BRANCHES can book a service the patient has
    # already chosen. Worth its own wording: at that moment the service
    # is the shared context, and naming it confirms the assistant is
    # still working on the thing they picked rather than starting over.
    "searching_service_branches": (
        "find_branches_offering_service",
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
    # `list_specialties` is an INTERNAL step, not the thing the patient
    # is waiting for. They described a symptom and are waiting to be
    # told which doctor to see; the specialty lookup is how the
    # assistant works that out, and announcing it ("جاري مراجعة
    # التخصصات المتاحة") narrates the assistant's own reasoning at
    # someone who never asked about specialties.
    #
    # Confirmed directly: this line appeared after "بطني وجعاني وعندي
    # ترجيع" and was rejected - what the patient wants told to them is
    # the DOCTOR search, which is the step after they agree. Silencing
    # it here means the turn's timer stays unarmed and the genuinely
    # relevant next call is what speaks.
    "list_specialties",
})

# ...BUT ONLY IN RESOLVE MODE.
#
# `match_entity_for_booking` is dual-mode: with `user_input` filled it
# resolves the patient's text to one entity (instant, always followed by
# the real work), and with `user_input` EMPTY it lists every doctor or
# branch - which is a full roster fetch and is frequently the slowest
# thing in the turn.
#
# CONFIRMED REAL FAILURE: the patient answered "دكتور", the turn spent
# 5.1 seconds fetching and rendering all 8 doctors, and said nothing at
# all while they waited - because the only tool called was this one, and
# it was unconditionally treated as silent. `schedule()` is given the
# call's ARGUMENTS now so it can tell the two modes apart.
_LIST_MODE_ARG_NAMES = ("user_input",)


def _is_list_mode(tool_name: str, args: Optional[dict]) -> bool:
    """True when a dual-mode resolver is being used to LIST rather than
    to resolve - i.e. its entity argument is empty."""

    if tool_name not in _SILENT_RESOLVER_TOOLS:
        return False

    if args is None:
        # No argument information available: assume resolve mode, which
        # preserves the previous behaviour exactly.
        return False

    return not any(str(args.get(name) or "").strip() for name in _LIST_MODE_ARG_NAMES)


def _list_mode_alias(tool_name: str, args: Optional[dict]) -> str:
    """The tool whose wording describes what a list-mode resolver is
    actually fetching, chosen from its `entity_type` argument."""

    entity_type = str((args or {}).get("entity_type") or "").strip().lower()

    if entity_type.startswith("branch"):
        return "list_branches_for_specialty"

    return "find_available_doctors"

# Tools that LIST options. Announcing one of these as a search is right
# when the patient asked an open question ("what branches does he work
# at?"), and wrong when they have just answered with a pick from a list
# already on their screen.
#
# CONFIRMED FROM A REAL CHAT: the patient was shown three branches, typed
# "2", and was told "لحظة من فضلك، جاري البحث عن الفروع المتاحة… 🏥" -
# the branches had already been found and shown, and what the assistant
# was actually doing was resolving their choice and fetching days. The
# line described work from the previous turn, and re-announcing a search
# for something they just chose reads as though their answer was
# ignored. See `schedule(..., answering_a_list=True)`.
_LIST_LOOKUP_TOOLS = frozenset({
    "list_branches_for_specialty",
    "list_specialties",
    "find_available_doctors",
    "find_best_doctor_in_specialty",
})

# Order of precedence when one turn calls several tools at once: the
# patient should be told about the most significant thing happening, not
# whichever tool the model happened to list first.
#
# EVERY GROUP IN _TOOL_GROUPS MUST APPEAR HERE. `message_for` resolves
# the group by walking this tuple, so a group missing from it can never
# be selected at all - it silently falls through to the "generic" line.
# `searching_times` and `finding_patient_details` were both missing,
# which is why looking up a day's times said "جاري تنفيذ طلبك… ⏳"
# instead of "جاري البحث عن الأوقات المتاحة… 🕐", despite having a
# perfectly good message defined for it. There is a test below that
# fails if the two lists ever drift apart again.
_GROUP_PRIORITY = (
    "creating_booking",
    "cancelling",
    "rescheduling",
    "sending_complaint",
    "sending_otp",
    "searching_times",
    "searching_slots",
    "searching_service_branches",
    "searching_branches",
    "searching_specialty_doctors",
    "searching_doctors",
    "searching_specialties",
    "finding_booking",
    "finding_patient_details",
    "checking_info",
)

# Fail loudly at import rather than shipping a group nothing can select.
_UNREACHABLE_GROUPS = set(_TOOL_GROUPS) - set(_GROUP_PRIORITY)
if _UNREACHABLE_GROUPS:  # pragma: no cover - guards a coding mistake
    raise RuntimeError(
        "progress.py: these tool groups are not in _GROUP_PRIORITY and can "
        f"therefore never be selected: {sorted(_UNREACHABLE_GROUPS)}"
    )


# Defaults, kept deliberately plain. They are NOT written in any one
# regional dialect, because this text is not composed by the LLM and so
# never passes through the LANGUAGE & DIALECT mirroring - a hard-coded
# Egyptian line would read wrong for a Saudi tenant. Anything more
# specific belongs in the tenant's own CSV column (see below).
_DEFAULT_MESSAGES: Dict[str, Dict[str, str]] = {
    "searching_doctors":  {"ar": "لحظة من فضلك، جاري البحث عن الأطباء المتاحين… 🔎",
                           "en": "One moment please - looking up the available doctors… 🔎"},
    "searching_specialties": {"ar": "لحظة من فضلك، جاري مراجعة التخصصات المتاحة… 🩺",
                           "en": "One moment please - checking the available specialties… 🩺"},
    "searching_specialty_doctors": {"ar": "لحظة من فضلك، جاري مراجعة الدكاترة المتاحين في التخصص ده… 🩺",
                           "en": "One moment please - checking the available doctors in this specialty… 🩺"},
    "searching_branches": {"ar": "لحظة من فضلك، جاري البحث عن الفروع المتاحة… 🏥",
                           "en": "One moment please - looking up the available branches… 🏥"},
    "searching_service_branches": {"ar": "لحظة من فضلك، جاري البحث عن الفروع المتاح بها الخدمة… 🏥",
                           "en": "One moment please - looking up the branches offering this service… 🏥"},
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


def _message_for_group(
    group: str,
    language: Optional[str] = "ar",
    templates: Optional[dict] = None,
) -> str:
    """The interim line for one named group, bypassing tool lookup.

    Same tenant-override rule as `message_for`: a
    `msg_progress_<group>` column in the client's config wins, then
    `msg_progress`, then the neutral default.
    """

    if templates:
        override = templates.get(f"msg_progress_{group}") or templates.get("msg_progress")
        if override and override.strip():
            return override.replace("\r\n", "\n").replace("\r", "\n").strip()

    key = "en" if (language or "ar").startswith("en") else "ar"
    return _DEFAULT_MESSAGES.get(group, _DEFAULT_MESSAGES["generic"])[key]


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


# When a progress message was handed to the webhook, and whether one is
# still mid-flight. Both are what `end_turn` uses to guarantee ORDERING.
_delivery_finished_at: Dict[str, float] = {}
_delivery_in_progress: Dict[str, threading.Event] = {}

# How far ahead of the real answer an interim message must be.
#
# WHY THIS EXISTS AT ALL: the interim line and the answer travel by two
# INDEPENDENT paths - progress POSTs straight to the webhook from a
# timer thread, while the answer goes back as the /chat response and is
# then delivered by n8n. Nothing about "sent first" makes them "arrive
# first". Confirmed repeatedly in production: the line reading "لحظة من
# فضلك، جاري البحث عن الأوقات المتاحة… 🕐" landing UNDERNEATH the list
# of times it was supposed to precede.
#
# No amount of checking before sending can fix that - by the time the
# check runs, the only thing left to control is when the ANSWER goes
# out. So the answer is held back until the interim line has had a
# clear head start. Paid only on turns that actually sent one.
_MIN_ORDERING_GAP_SECONDS = 0.6

# Hard ceiling on the total hold, so a webhook that has gone slow or
# unresponsive can never stall a patient's answer. Exceeding this is
# logged: if it shows up, the webhook is the thing to look at.
_MAX_ORDERING_WAIT_SECONDS = 2.0


def begin_turn(session_id: str) -> None:
    """Called once when a user message starts being processed."""

    with _lock:
        _cancel_locked(session_id)
        _in_flight[session_id] = True
        _delivered[session_id] = False
        _delivery_finished_at.pop(session_id, None)
        _delivery_in_progress.pop(session_id, None)
        # NOT set here on purpose - see _turn_started's comment above.
        # Left over from a PRIOR turn shouldn't leak in either, so it's
        # explicitly cleared; schedule() sets it fresh the first time
        # it's actually called for this turn.
        _turn_started.pop(session_id, None)


def end_turn(session_id: str) -> None:
    """Called when the turn finishes, however it finishes.

    Cancelling here is what makes a fast turn produce no interim message
    at all - the timer never gets to fire.

    ALSO ENFORCES ORDERING. If an interim message went out during this
    turn, this blocks briefly so the answer cannot overtake it - see
    _MIN_ORDERING_GAP_SECONDS. This runs on the request thread, on
    purpose: the caller is about to return the reply, and delaying that
    return by a few hundred milliseconds is the only remaining way to
    control which of the two messages the patient sees first.

    Never raises, and never waits longer than
    _MAX_ORDERING_WAIT_SECONDS.
    """

    try:
        _await_ordering_gap(session_id)
    except Exception as exc:  # pragma: no cover - must never break a turn
        logger.warning("progress[%s]: ordering wait failed (%s)", session_id, exc)

    with _lock:
        # Cleared before anything else: a timer thread already inside
        # _fire is blocked on this same lock right now, and this is what
        # it will see the moment it acquires it.
        _in_flight.pop(session_id, None)
        _cancel_locked(session_id)
        _delivered.pop(session_id, None)
        _turn_started.pop(session_id, None)
        _pending.pop(session_id, None)
        _delivery_finished_at.pop(session_id, None)
        _delivery_in_progress.pop(session_id, None)


def _await_ordering_gap(session_id: str) -> None:
    """Hold the answer back until any interim message this turn is
    provably ahead of it."""

    with _lock:
        pending_send = _delivery_in_progress.get(session_id)
        finished_at = _delivery_finished_at.get(session_id)

    if pending_send is None and finished_at is None:
        # Nothing was sent this turn - the overwhelming majority of
        # turns - so there is nothing to order against.
        return

    started_waiting = time.monotonic()

    if pending_send is not None and not pending_send.is_set():
        # The POST is still in flight. Wait for it to land, otherwise
        # the gap below would be measured from the wrong moment.
        if not pending_send.wait(timeout=_MAX_ORDERING_WAIT_SECONDS):
            logger.warning(
                "progress[%s]: interim message still unsent after %.1fs - releasing the "
                "answer anyway; the two may arrive out of order. Check the progress webhook.",
                session_id, _MAX_ORDERING_WAIT_SECONDS,
            )
            return

        with _lock:
            finished_at = _delivery_finished_at.get(session_id)

    if finished_at is None:
        return

    elapsed_since_send = time.monotonic() - finished_at
    remaining_gap = _MIN_ORDERING_GAP_SECONDS - elapsed_since_send

    budget_left = _MAX_ORDERING_WAIT_SECONDS - (time.monotonic() - started_waiting)
    hold = min(remaining_gap, budget_left)

    if hold > 0:
        logger.info(
            "progress[%s]: holding the answer %.2fs so the interim message stays ahead of it",
            session_id, hold,
        )
        time.sleep(hold)


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
    answering_a_list: bool = False,
    tool_args: Optional[dict] = None,
    agent_name: Optional[str] = None,
    channel_phone: Optional[str] = None,
) -> None:
    """Arm the interim message for a tool phase that is about to start.

    `answering_a_list`: True when the patient's own latest message was a
    pick from a list they were just shown ("2", "الدقي", "الأول"). In
    that case a list-lookup tool is resolving their answer, not
    searching on their behalf, so it is treated as a silent resolver -
    see _LIST_LOOKUP_TOOLS.

    `tool_args`: {tool_name: arguments}, used only to tell a dual-mode
    resolver's LIST mode (a real roster fetch, worth announcing) from
    its RESOLVE mode (instant) - see _is_list_mode.

    `agent_name`: which flow is running. The SAME tool can be worth
    describing differently depending on why it was called - looking up
    times inside a reschedule is "moving your appointment", not
    "checking the available times" - so this narrows the wording where
    it genuinely changes what the patient is waiting for. Optional and
    additive: omit it and every existing behaviour is unchanged.

    `channel_phone`: the patient's own WhatsApp number for this
    session (state["channel_phone"]). Carried through to the webhook
    payload delivered to n8n (see _deliver) so the n8n flow can route
    the interim message on the same channel/variable it uses for the
    final reply, without having to re-derive it from session_id.
    Optional and additive: omit it and the payload simply omits the
    field, unchanged from before.

    Never raises: a failure here must not be able to break a turn that
    would otherwise have answered the patient perfectly well.
    """

    if not config.PROGRESS_ENABLED:
        return

    try:
        names = list(tool_names or [])
        if not names:
            return

        args_by_tool = tool_args or {}

        # NO INTERIM MESSAGE WHILE SOMEONE IS DESCRIBING A SYMPTOM.
        #
        # In the medical-guidance flow the patient has just told us they
        # are unwell ("بطني وجعاني وعندي ترجيع"). A mechanical "لحظة من
        # فضلك، جاري البحث عن الأطباء المتاحين… 🔎" lands as the FIRST
        # thing they get back - before any acknowledgement that they are
        # in pain - and reads like a system status line rather than a
        # person responding. The warm reply is worth waiting a moment
        # for; a spinner in front of it is worse than silence.
        #
        # This is deliberately the whole flow, not one tool: every
        # lookup here happens while the patient is waiting to be
        # answered about their symptom.
        if agent_name == "medical":
            return

        silent = {
            name for name in _SILENT_RESOLVER_TOOLS
            if not _is_list_mode(name, args_by_tool.get(name))
        }
        if answering_a_list:
            silent |= _LIST_LOOKUP_TOOLS

        # `list_specialties` IS SILENT ONLY IN MEDICAL GUIDANCE.
        #
        # There, the patient described a symptom and is waiting to be
        # told which doctor to see; the specialty lookup is the
        # assistant's own reasoning and announcing it narrates a step
        # nobody asked about (see _SILENT_RESOLVER_TOOLS).
        #
        # In the BOOKING flow it is the opposite: the patient answered
        # "تخصص" to "تحب تبدأ بالتخصص ولا بالدكتور؟", so the specialty
        # list is precisely what they asked for and are now waiting on.
        # CONFIRMED: that turn showed "جاري البحث عن الأطباء المتاحين"
        # and then produced a list of SPECIALTIES - describing the wrong
        # thing entirely.
        if agent_name in ("booking", "concierge") and "list_specialties" in names:
            silent.discard("list_specialties")

        if all(name in silent for name in names):
            # Nothing worth announcing yet - see _SILENT_RESOLVER_TOOLS.
            # Leave any already-armed timer from earlier in this turn
            # alone; just don't let THIS call arm one or overwrite the
            # pending wording with a resolver-only description.
            return

        # Whatever remains is what the patient is genuinely waiting for,
        # so the wording is taken from that and not from the resolvers
        # sharing the same turn.
        announceable = [name for name in names if name not in silent] or names

        # A dual-mode resolver in LIST mode has no group of its own -
        # what it is fetching depends on its `entity_type` argument. Map
        # it onto the tool whose wording already describes that fetch, so
        # listing doctors says "جاري البحث عن الأطباء" and listing
        # branches says "جاري البحث عن الفروع", rather than both falling
        # through to the generic line.
        announceable = [
            _list_mode_alias(name, args_by_tool.get(name))
            if _is_list_mode(name, args_by_tool.get(name)) else name
            for name in announceable
        ]

        # A doctor search narrowed to a specialty gets wording that says
        # so - it is the step the patient just agreed to, and naming the
        # specialty confirms the assistant understood which one.
        groups_override = None
        if any(
            name == "find_available_doctors"
            and (args_by_tool.get(name) or {}).get("specialty_ids")
            for name in announceable
        ):
            groups_override = "searching_specialty_doctors"

        # INSIDE A RESCHEDULE, THE PATIENT IS WAITING ON ONE THING.
        #
        # Every slot/time lookup in that flow exists to move an existing
        # appointment, so describing the mechanics ("جاري البحث عن
        # الأوقات المتاحة") tells them about a step rather than about
        # their request. CONFIRMED: during a reschedule the interim line
        # read "جاري البحث عن الأوقات المتاحة" immediately before the
        # old-vs-new appointment review - the patient is moving a
        # booking, and that is what the line should say.
        if agent_name == "reschedule" and any(
            _GROUP_FOR_TOOL.get(name) in ("searching_slots", "searching_times")
            for name in announceable
        ):
            groups_override = "rescheduling"

        text = (
            _message_for_group(groups_override, language, templates)
            if groups_override
            else message_for(announceable, language, templates)
        )
        fire_now = False

        with _lock:
            if _delivered.get(session_id):
                # Already told them once this turn - saying it again for
                # the next tool call in the same turn is spam.
                return

            # Newest wording wins, so the message describes what is
            # actually running when it goes out.
            _pending[session_id] = (client_id, text, channel_phone)

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

    client_id, text, channel_phone = pending
    _deliver(session_id, client_id, text, channel_phone)


def _deliver(session_id: str, client_id: str, text: str, channel_phone: Optional[str] = None) -> None:
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
        # The patient's own WhatsApp number for this session, so the n8n
        # flow can address the interim message the same way it addresses
        # the final reply, without re-parsing session_id. Present
        # whenever the caller had it (see schedule()'s channel_phone
        # param); omitted (None) only for the legacy/test call sites that
        # don't pass it, which keeps this additive rather than breaking.
        "channel_phone": channel_phone,
    }

    try:
        # Imported lazily so this module stays importable in environments
        # where nothing ever sends a webhook.
        import requests

        # LAST CHECK, IMMEDIATELY BEFORE THE REQUEST GOES OUT.
        #
        # The guard above runs while holding the lock, but the POST that
        # follows takes real time - measured at ~1.3s against the live
        # n8n webhook. The turn can, and does, finish inside that window:
        # the guard passed at T, the real answer left the app at T+40ms,
        # and the interim line was still in flight. Re-reading the flag
        # here shrinks that window to the width of the send call itself.
        if not _in_flight.get(session_id):
            logger.info(
                "progress[%s]: suppressed %r just before sending - the turn finished",
                session_id, text,
            )
            return

        # Opened BEFORE the POST and closed after it, so `end_turn` can
        # tell "nothing was sent" from "a send is still in flight" and
        # wait for the latter rather than releasing the answer into a
        # race - see _await_ordering_gap.
        sent_marker = threading.Event()
        with _lock:
            _delivery_in_progress[session_id] = sent_marker

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=config.PROGRESS_TIMEOUT_SECONDS,
                headers={"Content-Type": "application/json"},
            )
        finally:
            with _lock:
                _delivery_finished_at[session_id] = time.monotonic()
            sent_marker.set()

        # The window cannot be closed entirely from this side - the
        # request is already with n8n by now. Logging when it happens at
        # least makes it visible and measurable rather than showing up
        # only as a confused patient. If this line appears often, raise
        # PROGRESS_DELAY_SECONDS: the turns firing it are ones that were
        # never slow enough to need an interim message.
        if not _in_flight.get(session_id):
            logger.warning(
                "progress[%s]: sent %r but the turn finished while it was in "
                "flight - the patient may see it after the answer. Consider "
                "raising PROGRESS_DELAY_SECONDS (currently %ss).",
                session_id, text, config.PROGRESS_DELAY_SECONDS,
            )
        else:
            logger.info(
                "progress[%s]: sent %r (HTTP %s)", session_id, text, response.status_code,
            )
    except Exception as exc:
        # Swallowed on purpose. A "please wait" line that fails to send is
        # a cosmetic loss; it must never surface to the patient or
        # interfere with the real reply that is still being produced.
        logger.warning("progress[%s]: delivery failed (%s)", session_id, exc)
