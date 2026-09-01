"""
LangChain tools for the LLM-tool-calling Guest Booking Cancellation Agent.

REWRITTEN (see tools.py.pre_rewrite_backup for the old version). Every
tool now returns STRUCTURED DATA ONLY - no formatted sentences, no
message-template lookups, no natural language of any kind. The LLM
(driven by prompts.AGENT_SYSTEM_PROMPT_TEMPLATE) is solely responsible
for turning these status codes/data into user-facing replies. This is
the literal architecture change requested: tools never speak to the
user.

What did NOT change: api.py (all raw HTTP calls), config.py (client
config / base_url resolution), the timezone conversion math, the
active-booking filter, and the OTP dummy-provider mechanics. Those are
"Company APIs" / "booking logic" / "OTP logic" and were explicitly
required to stay untouched - only the OUTPUT SHAPE of the functions that
wrap them changed, from "already-formatted text" to "plain status/data".

Removed entirely (superseded by the LLM's own reasoning, since "the LLM
should decide" replaces every heuristic classifier):
  detect_message, extract_input_details, resolve_selection,
  parse_confirmation, detect_step_back, format_message,
  format_booking_card, format_booking_list, format_time_12h, format_date,
  find_matching_appointment (replaced by check_booking_status's ref-based
  re-lookup, simpler and equally safe since ref numbers are unique).
"""

import logging
import re
import smtplib
import time
from datetime import date, datetime, timedelta, timezone
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo
from typing import Annotated, Dict, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import api
import rag
from config import (
    DEFAULT_TIMEZONE,
    CANCELLABLE_STATUS_CODES,
    CANCELLED_STATUS_NAME,
    DEFAULT_COUNTRY_CODE,
    DOCTOR_AVAILABILITY_WINDOW_DAYS,
    DOCTOR_LIST_CACHE_SECONDS,
    OTP_PROVIDER,
    OTP_TTL_SECONDS,
    TEST_OTP,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_USE_TLS,
    SMTP_USE_SSL,
    COMPLAINT_WEBHOOK_URL,
)
import requests
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# Hard guard for request_human_handoff - complaint word alone is not
# consent to be transferred.
# ==========================================================
#
# WHY THIS EXISTS AS CODE, NOT JUST PROSE: the tool's own docstring
# below already explains, in detail, with a "confirmed real production
# failure" example, that a bare "شكوي" is a topic, not a request for a
# human. That prose was already in place and the LLM still called this
# tool with patient_agreed=True on exactly that input, reason logged as
# "patient asked for staff" - the same failure the docstring describes,
# happening again on the same wording. A second occurrence of an
# already-documented failure means the instruction alone cannot be
# trusted to hold on every turn, so the check is enforced here instead
# of only being asked for.
_COMPLAINT_ROOTS_FOR_HANDOFF_GUARD = ("شكو", "اشتك", "complaint")

# Words that show the patient is SEPARATELY, explicitly asking for a
# person - as opposed to just naming "complaint" as the topic. If any of
# these appear alongside a complaint word, the guard steps aside and
# lets the model's own call stand (e.g. "الشكوى معقدة عايز اتكلم مع حد").
_EXPLICIT_HUMAN_REQUEST_ROOTS = (
    "موظف", "خدمة العملاء", "خدمه العملاء", "ممثل خدمة", "اتكلم مع حد",
    "أتكلم مع حد", "كلمني حد", "كلميني حد", "حد يرد", "شخص حقيقي",
    "human", "representative", "agent", "someone", "speak to a person",
    "talk to a person",
)


def _latest_human_text_for_handoff_guard(state: AgentState) -> str:
    """The most recent HumanMessage's raw text, or "" if none is found.
    Deliberately tolerant of whatever message objects `state["messages"]`
    holds - only ever used to decide whether to BLOCK a handoff, never
    to allow one, so a missed/garbled message just means the guard has
    nothing to catch and the model's own decision goes through."""

    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content or "")
    return ""


def _latest_ai_text_before_handoff_guard(state: AgentState) -> str:
    """The most recent AIMessage's raw text (the assistant's own last
    turn) - used only to check whether a staff/customer-service handoff
    was actually OFFERED before this turn, never to allow a handoff on
    its own."""

    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", None) == "ai":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content or "")
    return ""


# ==========================================================
# Pure data helpers (unchanged in spirit from the old tools.py - these
# are data transforms, not user-facing text, so they stay)
# ==========================================================

"""Country calling codes, longest first, so the longest matching prefix
wins (e.g. "971" is checked before "97"). Used to recognize a number
that ALREADY carries a country code, whatever country that is.

This list only has to be good enough to tell "this already starts with
a country code" from "this is a bare local number" - it is not a
validation list, and an unknown code simply falls through to the
client's own default."""
_KNOWN_COUNTRY_CODES = (
    "966", "971", "973", "974", "965", "968", "962", "961", "970", "964", "963",
    "212", "213", "216", "218", "249", "252", "253", "222", "220",
    "234", "254", "255", "256", "233", "251",
    "353", "351", "352", "358", "359", "370", "371", "372", "380", "381", "385", "386",
    "420", "421", "852", "853", "855", "856", "886", "880", "886", "960", "961",
    "20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41", "43", "44",
    "45", "46", "47", "48", "49", "51", "52", "53", "54", "55", "56", "57", "58",
    "60", "61", "62", "63", "64", "65", "66", "81", "82", "84", "86", "90", "91",
    "92", "93", "94", "95", "98",
    "7", "1",
)


# The clinic's own country, derived from the IANA timezone every client
# row already sets correctly ("Africa/Cairo", "Asia/Riyadh", ...).
#
# WHY NOT country_codes_hint: that column is a list of codes this clinic
# ACCEPTS from patients, not the country the clinic is in, and it is
# written most-important-first for the reader, not for a parser. Both
# rows in client_config.csv read "+966, +20" - so taking the first code
# out of it made the EGYPTIAN clinic (Africa/Cairo) treat every bare
# local number as Saudi: "01155611045" became "+9661155611045", a number
# belonging to nobody, which was then sent to the booking API and used
# as the OTP storage key. The timezone column is unambiguous, already
# per-client, and already correct in every row.
_TIMEZONE_COUNTRY_CODES = {
    "africa/cairo": "20",
    "asia/riyadh": "966",
    "asia/dubai": "971",
    "asia/kuwait": "965",
    "asia/qatar": "974",
    "asia/bahrain": "973",
    "asia/muscat": "968",
    "asia/amman": "962",
    "asia/beirut": "961",
    "asia/baghdad": "964",
    "africa/tripoli": "218",
    "africa/tunis": "216",
    "africa/algiers": "213",
    "africa/casablanca": "212",
    "africa/khartoum": "249",
}


def _client_default_country_code(state=None) -> str:
    """The country code to assume for a BARE LOCAL number (one written
    with a leading 0, or with no country code at all), for this client.

    Resolution order:
      1. The client's own `timezone` column, mapped through
         _TIMEZONE_COUNTRY_CODES - this is the clinic's actual country.
      2. `phone_example`'s FIRST fully-written number, if the timezone
         is one this map doesn't know - a clinic that writes
         "+201155611045" as its example is telling us plainly which
         country its patients type local numbers for.
      3. DEFAULT_COUNTRY_CODE.

    `country_codes_hint` is deliberately NOT consulted: it lists the
    codes this clinic ACCEPTS, not the country it is in - see the
    comment above _TIMEZONE_COUNTRY_CODES.
    """

    templates = (state or {}).get("templates") or {}

    timezone_name = str(templates.get("_timezone") or "").strip().lower()
    code = _TIMEZONE_COUNTRY_CODES.get(timezone_name)
    if code:
        return code

    example = templates.get("_phone_example")
    if example:
        match = re.search(r"\+(\d{4,15})", str(example))
        if match:
            digits = match.group(1)
            # Longest known code first, so "966555123456" resolves to
            # "966" and not "96". A greedy \d{1,4} here would have
            # produced "9665", which is not a country code at all.
            for code in sorted(_KNOWN_COUNTRY_CODES, key=len, reverse=True):
                if digits.startswith(code):
                    return code

    return DEFAULT_COUNTRY_CODE


def normalize_phone_number(phone: Optional[str], state=None) -> Optional[str]:
    """Normalize a phone number to E.164 (e.g. "+201001255864").

    A number that ALREADY carries a country code is kept as-is, no
    matter which country that is - patients travel and register with
    foreign numbers, so an Egyptian number given to a Saudi clinic (or
    the reverse) is perfectly normal and must not be rewritten. Only a
    genuinely LOCAL number (leading 0, or no country code at all) gets
    this client's default country code attached.

    Confirmed real production bug: the default country code was a single
    global constant ("20"), so a Saudi number like "966568000000" was
    silently turned into "+20966568000000" - a number belonging to
    nobody - before being sent to the booking API.

    Pass `state` wherever it's available so the client's own configured
    country is used; without it, DEFAULT_COUNTRY_CODE applies."""

    if not phone:
        return phone

    cleaned = re.sub(r"[\s\-().]", "", str(phone).strip())

    if cleaned.startswith("+"):
        return cleaned
    if cleaned.startswith("00"):
        return "+" + cleaned[2:]

    default_code = _client_default_country_code(state)

    # Leading zero = local format for whichever country this client is
    # in ("01158877175" -> Egypt, "0568000000" -> Saudi).
    if cleaned.startswith("0"):
        return "+" + default_code + cleaned[1:]

    # No leading zero: this may already be a full international number
    # written without its "+". Accept it as such when it starts with a
    # real country code AND is long enough to be a complete number -
    # the length check is what stops a bare local number that happens to
    # begin with those digits from being misread as international.
    if cleaned.startswith(default_code) and len(cleaned) >= len(default_code) + 8:
        return "+" + cleaned

    for code in _KNOWN_COUNTRY_CODES:
        if cleaned.startswith(code) and len(cleaned) >= len(code) + 8:
            return "+" + cleaned

    return "+" + default_code + cleaned


def _is_valid_phone_format(phone: Optional[str]) -> bool:
    if not phone:
        return False
    return bool(re.match(r"^\+\d{7,15}$", phone.strip()))


def to_riyadh(utc_string: Optional[str], timezone_name: str = DEFAULT_TIMEZONE) -> Optional[str]:
    """ISO string -> the CLIENT'S OWN local time zone, as an ISO string.

    Despite the historical name (kept to minimize churn - this function
    used to be Riyadh-only), `timezone_name` is now a real per-client
    IANA zone name (e.g. "Africa/Cairo", "Asia/Riyadh" - both are real
    values already present in client_config.csv's own "timezone" column,
    exposed as state["templates"]["_timezone"]). This replaces a single
    hardcoded "+3 hours" that used to be applied to every clinic
    regardless of its actual location, which would have silently
    produced wrong times for any clinic outside Saudi Arabia, and
    doesn't account for DST where applicable.

    CRITICAL FIX (kept from the previous version): this used to blindly
    append a literal offset string to whatever `.isoformat()` produced,
    regardless of whether the parsed datetime was already timezone-aware.
    If the input was ALREADY timezone-aware (e.g.
    "2026-08-06T16:00:00+00:00" - confirmed directly from the real
    Doctors/GetDoctorScheduleSlots API response), that produced a
    doubled, invalid offset like "2026-08-06T19:00:00+00:00+03:00" -
    which caused a real production 400 error from GuestBookings/Update
    (it received an unparseable timestamp). Now: if the input is
    timezone-aware, convert it via astimezone(); if naive, assume UTC
    and attach the target zone directly on the datetime object - never
    by string concatenation."""

    if not utc_string:
        return None

    try:
        target_tz = ZoneInfo(timezone_name)
    except Exception:
        logger.warning("to_riyadh: unknown timezone %r, falling back to %s", timezone_name, DEFAULT_TIMEZONE)
        target_tz = ZoneInfo(DEFAULT_TIMEZONE)

    cleaned = utc_string.replace("Z", "+00:00")

    dt = None

    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        return utc_string

    if dt.tzinfo is not None:
        # Already timezone-aware - convert the actual instant to the
        # target zone (adjusts the wall-clock time correctly), don't
        # just relabel or append to it.
        local_dt = dt.astimezone(target_tz)
    else:
        # Naive - assume UTC, attach UTC first then convert, so DST
        # rules (where applicable) are resolved correctly rather than
        # applying a flat manual offset.
        local_dt = dt.replace(tzinfo=timezone.utc).astimezone(target_tz)

    return local_dt.isoformat()


def to_local_wallclock(value: Optional[str], timezone_name: str = DEFAULT_TIMEZONE) -> Optional[str]:
    """For WORKING-HOURS rows (a doctor's weekly rota), not instants.

    A rota row says "this doctor sits in clinic from 11:00 to 15:00".
    That is a WALL-CLOCK time at the branch. There is nothing to
    convert, and - critically - THE OFFSET THE API ATTACHES TO IT IS
    NOT REAL.

    EVIDENCE, from one production trace plus the clinic's own admin UI:
      - Admin UI for د. فارس الشارخ at Al Manar: 11:00 - 15:00.
        Patient saw: "من 2:00 مساءً لـ 6:00 مساءً" (14:00 - 18:00).
      - Raw row for د. رانيا عبد الرحمن, logged verbatim:
        fromDateTime '2026-08-06T07:45:00+00:00',
        toDateTime   '2028-01-18T14:15:00+00:00'
        Patient saw: "من 10:45 صباحًا لـ 5:15 مساءً" (10:45 - 17:15).
    Both are exactly +3 - the Asia/Riyadh offset applied to a number
    that was already local. The endpoint stamps "+00:00" on a
    wall-clock it never converted.

    So: take the clock components and drop the offset entirely. Never
    call `astimezone` on a rota row.

    (Note the same row's DATE parts - 2026-08-06 to 2028-01-18 - are the
    rota's effective RANGE, not a single day. Only the time-of-day
    component is the working window.)

    `timezone_name` is accepted so callers read consistently with
    `to_riyadh`, and is deliberately unused."""

    if not value:
        return None

    cleaned = value.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return value

    # Drop tzinfo rather than converting - the offset is decoration.
    return dt.replace(tzinfo=None).isoformat()


def _local_now_naive(timezone_name: str = DEFAULT_TIMEZONE) -> datetime:
    """"Now", as a NAIVE local datetime.

    Slot times are wall-clock with no real offset (see
    `to_local_wallclock`), so "has this slot already passed?" has to be
    compared against a naive local now. Comparing a naive datetime with
    an aware one raises TypeError, which - inside the try/except that
    wraps these filters - silently disables the past-slot filter and
    starts offering appointments earlier today."""

    try:
        return datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None)
    except Exception:
        return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).replace(tzinfo=None)


_ARABIC_WEEKDAY_NAMES = [
    "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد",
]


def conversation_language(state: Optional[dict]) -> str:
    """Which language THIS conversation is being conducted in ("ar"/"en").

    Computed deterministically by graph.py's `_detect_target_language`
    and written into state as `target_language` before any tool runs -
    tools read it from here so that every human-readable field they
    return (times, weekday names, doctor/branch/specialty names) is
    already in the conversation's own language.

    WHY THIS EXISTS: confirmed real user feedback - in an
    otherwise-Arabic conversation the agent was showing "10:30 AM" and
    English specialty/hospital names, because the tools formatted those
    fields in English regardless and the LLM (correctly) copied tool
    values verbatim rather than translating them. Fixing it at the
    source is far more reliable than asking the model to translate
    data fields it has been explicitly told never to alter.
    """

    language = (state or {}).get("target_language")
    return "en" if language == "en" else "ar"


def _display_time_12h(iso_string: Optional[str], language: str = "ar") -> str:
    """12-hour display string - DATA, not a sentence, so tools may
    still compute it (an LLM doing manual date arithmetic is unreliable;
    this is exactly why the hard rule in prompts.py tells it to use this
    field instead of formatting timestamps itself).

    Arabic conversations get صباحًا/ظهرًا/مساءً rather than AM/PM, so an
    Arabic reply never carries a stray English fragment.
    """

    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string

    if language == "en":
        return dt.strftime("%I:%M %p").lstrip("0") or dt.strftime("%I:%M %p")

    hour12 = dt.hour % 12 or 12
    if dt.hour == 12:
        period = "ظهرًا"
    elif dt.hour < 12:
        period = "صباحًا"
    else:
        period = "مساءً"

    return f"{hour12}:{dt.minute:02d} {period}"


def _display_weekday(iso_string: Optional[str], language: str = "ar") -> str:
    """Weekday name in the conversation's own language."""

    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return ""

    english = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][dt.weekday()]
    return english if language == "en" else _ARABIC_WEEKDAY_NAMES[dt.weekday()]


def _display_date(iso_string: Optional[str]) -> str:
    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string
    return dt.strftime("%d/%m/%Y")


_FIELD_MAP = (
    ("ref", ("bookingRefNum",)),
    # servicePrice deliberately NOT mapped - fees are private by default
    # and only ever revealed via `get_doctor_fees` on an explicit user
    # request (see prompts.py's FEES rule).
    ("patientFullName", ("patientFullName",)),
    ("mobileNumber", ("mobileNumber",)),
    ("email", ("email",)),
    ("statusName", ("statusName",)),
    ("branchName", ("branchName",)),
    ("doctorName", ("doctorName",)),
    ("doctorId", ("doctorId",)),
    ("serviceName", ("serviceName",)),
    ("specialtyName", ("specialtyName",)),
)


def _shape_appointment(item: dict, timezone_name: str = DEFAULT_TIMEZONE, language: str = "ar") -> dict:
    """Flatten one raw API booking item into plain data fields - no
    sentences, just values, for the LLM to reference directly.
    `timezone_name` should be this client's own IANA zone (from
    state["templates"]["_timezone"]) - see to_riyadh()."""

    shaped = {}
    for name, keys in _FIELD_MAP:
        for key in keys:
            if key in item:
                shaped[name] = item[key]
                break

    # WALL-CLOCK, like every other time in this API.
    #
    # A booking's stored time came from a slotStart, which is wall-clock
    # carrying a meaningless "+00:00" (see to_local_wallclock). Running
    # it through `to_riyadh` would shift it +3 and tell the patient the
    # wrong time for an appointment they already hold - the same class
    # of bug the rota and slot displays had.
    local_from = to_local_wallclock(item.get("bookingTimeFrom"), timezone_name)
    local_to = to_local_wallclock(item.get("bookingTimeTo"), timezone_name)

    shaped["bookingTimeFrom"] = local_from
    shaped["bookingTimeTo"] = local_to
    shaped["date_display"] = _display_date(local_from)
    shaped["time_display"] = _display_time_12h(local_from, language)
    shaped["weekday_display"] = _display_weekday(local_from, language)
    shaped["id"] = item.get("id")
    shaped["status"] = item.get("status")

    return shaped


def _filter_active(items: list) -> list:
    """Excludes bookings that can no longer be cancelled or rescheduled:
    already cancelled/completed/arrived/no-show, or past their own
    scheduled date. Applied to BOTH the reference-number and phone-number
    lookup paths (see lookup_appointment) - an earlier version only
    applied this to the phone path, preserving an asymmetry from the
    original n8n business logic; that asymmetry was explicitly removed
    per a later request: a past/inactive booking must never be offered
    for cancellation or rescheduling regardless of how it was found.

    CHANGED (explicit user request, based on a real dashboard screenshot):
    "active"/cancellable no longer requires a scheduled future visit date.
    It now means the booking's statusName indicates it HASN'T happened
    yet - i.e. anything other than Cancelled/Completed/Arrived. This
    specifically includes "New" bookings that don't have a visit date
    set yet at all (shown as "-" in the dashboard) - those are still
    perfectly cancellable and must appear.

    Previously this required `bookingTimeFrom` to be set AND in the
    future, which silently excluded every "New" booking without a
    visit date yet - that was the actual root cause of "no booking
    found" despite a visible, cancellable "New" row in the dashboard.

    ADDED BACK (explicit follow-up request): a booking with a scheduled
    visit date that has already passed must be excluded too, even if its
    status is still "New" (e.g. a no-show never updated in the source
    system) - it can't practically be cancelled anymore. A "New" booking
    with NO visit date set at all is still included (nothing to compare
    against - it hasn't happened by definition).

    STATUS CODES (confirmed directly from the Booking API's own
    documentation): New=1, Confirmed=2, Arrived=3, NoShow=4, Completed=5,
    Cancelled=6. Only New/Confirmed are cancellable. This now checks the
    NUMERIC `status` field as the primary, reliable mechanism (language-
    independent - no more guessing at Arabic vs English spelling), with
    the earlier string-based `statusName` matching kept only as a
    fallback for the rare item that might be missing a numeric status
    for some reason."""

    _excluded_keywords = (
        "cancelled", "canceled", "completed", "arrived", "no show", "no-show",
        "ملغ", "ألغي", "مكتمل", "منتهي", "وصل", "لم يحضر",
    )

    now = datetime.now(timezone.utc)

    active = []
    for item in items:
        status_code = item.get("status")

        if status_code is not None:
            if status_code not in CANCELLABLE_STATUS_CODES:
                continue
        else:
            # No numeric status on this item at all - fall back to the
            # string-based check as a defense-in-depth safety net.
            status_name = (item.get("statusName") or "").strip().lower()
            if any(keyword in status_name for keyword in _excluded_keywords):
                continue

        raw_from = item.get("bookingTimeFrom")
        if raw_from:
            try:
                dt = datetime.fromisoformat(raw_from.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    # Naive - assume UTC, same as this function always did.
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt <= now:
                    continue  # has a scheduled date, and it's already passed
            except ValueError:
                pass  # unparsable date - don't let a bad format hide an otherwise-active booking

        active.append(item)

    return active


def _base_url(state: AgentState) -> str:
    return state.get("templates", {}).get("_base_url") or "https://demo.catalystsystems.io:1102"


# ==========================================================
# Tools - each returns STATUS + DATA ONLY, never a sentence
# ==========================================================

@tool
def validate_phone_format(
    state: Annotated[AgentState, InjectedState],
    phone: str,
) -> dict:
    """Validate that a phone number is usable, and return it normalized
    to international format. Returns {"status": "valid",
    "normalized": "+201234567890"} or {"status": "invalid"}.

    A number from ANOTHER country is perfectly valid - patients
    routinely register with a foreign number - so never reject one for
    not matching the clinic's own country, and never rewrite its country
    code. Only a bare local number (leading 0, or no country code) is
    assumed to belong to this clinic's country."""

    normalized = normalize_phone_number(phone, state)

    if not _is_valid_phone_format(normalized):
        return {"status": "invalid"}

    return {"status": "valid", "normalized": normalized}


@tool
def compare_phone(
    state: Annotated[AgentState, InjectedState],
    provided_phone: str,
    channel_phone: str = "",
) -> dict:
    """Compare a user-provided phone number against the verified channel
    identity phone number (if any). Returns {"status": "match"} or
    {"status": "no_match"}. Never decide this yourself - always call
    this tool."""

    a = normalize_phone_number(provided_phone, state)
    b = normalize_phone_number(channel_phone, state) if channel_phone else None

    match = bool(a and b and a == b)

    logger.info(
        "compare_phone: provided=%r -> normalized=%r | channel=%r -> normalized=%r | match=%s",
        provided_phone, a, channel_phone, b, match,
    )

    if match:
        _mark_phone_verified(state, provided_phone)
        return {"status": "match"}

    return {"status": "no_match"}


@tool
def lookup_appointment(
    state: Annotated[AgentState, InjectedState],
    ref_number: str = "",
    phone: str = "",
    use_channel_identity: bool = False,
    language: str = "en",
) -> dict:
    """Look up bookings by reference number OR phone number.

    If the user chose to cancel by phone number and a verified channel
    identity (e.g. their WhatsApp number) is already known, call this
    with `use_channel_identity=True` and leave `phone` empty - this
    automatically searches using that verified number WITHOUT you ever
    needing to ask the user to type it, and WITHOUT you ever seeing the
    actual digits yourself. Any booking found this way is by definition
    already verified (it was found using their own verified channel
    number), so NO OTP is ever needed in this case - skip straight to
    STEP 3/4 of the flow.

    Only ask the user to type a phone number, and only then go through
    compare_phone/OTP, if `use_channel_identity` returns "no_channel_identity"
    (there is none available) or if the user explicitly says the booking
    is under a DIFFERENT number than the one they're messaging from.

    ALWAYS pass `language` as "ar" if you are about to reply to the user
    in Arabic (any dialect), or "en" if replying in English - this makes
    the booking system return doctor/branch/service names already
    spelled correctly in that language, so you never have to translate
    or transliterate a name yourself (which risks misspelling it).

    Returns one of:
    {"status": "not_found"}
    {"status": "found_one", "appointment": {...}}
    {"status": "found_many", "appointments": [...]}
    {"status": "found_but_inactive"}  # a booking exists under this ref/phone,
                          # but it's already cancelled, completed, or its
                          # own date/time has already passed - it can no
                          # longer be cancelled or rescheduled. Tell the
                          # user plainly why, don't just say "not found"
                          # (which would wrongly imply they mistyped
                          # something).
    {"status": "error"}  # the booking API call itself failed - a technical
                          # problem, NOT the same as "no booking exists"
    {"status": "no_channel_identity"}  # use_channel_identity was True but
                          # no verified channel number is available - ask
                          # the user to type their phone number instead
    {"status": "phone_not_verified"}  # you passed a phone number that
                          # hasn't been verified in this conversation yet
                          # (not the channel identity, no successful
                          # compare_phone match, no successful verify_otp).
                          # Go call compare_phone (and send_otp/verify_otp
                          # if it doesn't match) BEFORE calling this tool
                          # again with that number - never retry this call
                          # as-is expecting a different result.
    Appointment fields: ref, doctorName, branchName, serviceName,
    specialtyName, statusName, date_display, time_display, patientFullName,
    mobileNumber, email, id."""

    if use_channel_identity:
        channel_phone = state.get("channel_phone")
        logger.info("lookup_appointment: use_channel_identity=True, channel_phone=%r", channel_phone)
        if not channel_phone:
            return {"status": "no_channel_identity"}
        phone = channel_phone
    elif phone and not ref_number:
        # SERVER-SIDE ENFORCEMENT, NOT JUST A PROMPT RULE. The prompt
        # instructs calling `compare_phone`/`send_otp`+`verify_otp`
        # BEFORE this, for any phone that isn't the channel identity -
        # but nothing in this tool itself used to check that actually
        # happened, so a model that skipped straight here (a prompt-
        # following slip, not a deliberate bypass) would still return a
        # real patient's private appointment details - doctor, branch,
        # date, time - to whoever is messaging, for a phone number they
        # never proved was theirs. `_phone_is_verified` is TRUE only for
        # the conversation's own channel number, a real `compare_phone`
        # match, or a successful `verify_otp` recorded THIS session -
        # never just because the model asserts it already checked.
        if not _phone_is_verified(state, phone):
            logger.warning(
                "lookup_appointment: refusing phone-path lookup for an unverified number "
                "(session_id=%s) - compare_phone/verify_otp must succeed first",
                state.get("session_id"),
            )
            return {"status": "phone_not_verified"}

    base_url = _base_url(state)

    if ref_number:
        result = api.get_bookings_by_ref(base_url, ref_number, language=language)
    elif phone:
        result = api.get_bookings_by_phone(
            base_url, normalize_phone_number(phone, state), language=language,
            status_list=list(CANCELLABLE_STATUS_CODES),
        )
    else:
        return {"status": "not_found"}

    if not result["success"]:
        # IMPORTANT: this used to silently return "not_found" for ANY
        # failure - timeouts, wrong base_url, 4xx/5xx, bad JSON - making
        # a real connectivity/config problem indistinguishable from a
        # genuinely empty result, both to logs and to the user. Now it's
        # logged with the real reason and reported as a distinct
        # "error" status so the LLM (per prompts.py) tells the user
        # there was a technical problem instead of "no booking found".
        logger.error(
            "lookup_appointment API call failed: base_url=%s ref=%r phone=%r status_code=%s error=%s",
            base_url, ref_number, phone, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    if not items:
        return {"status": "not_found"}

    # CHANGED (explicit request): both paths now apply the same
    # active-only filter (excludes cancelled/completed/already-passed
    # bookings). This used to only apply to the phone path, matching the
    # original n8n business logic's asymmetry - that asymmetry is no
    # longer wanted: a past/inactive booking must never be offered for
    # cancellation or rescheduling, regardless of how it was looked up.
    active_items = _filter_active(items)
    if not active_items:
        # A booking WAS found, but every match is already past/cancelled/
        # completed - distinct from "no booking with that ref/phone at
        # all", so the LLM can say why plainly instead of implying they
        # may have mistyped something.
        return {"status": "found_but_inactive"}
    items = active_items

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    shaped = [_shape_appointment(i, timezone_name, conversation_language(state)) for i in items]

    if len(shaped) == 1:
        return {"status": "found_one", "appointment": shaped[0]}

    return {"status": "found_many", "appointments": shaped}


@tool
def check_booking_status(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
    language: str = "en",
) -> dict:
    """Re-fetch a booking by its reference number IMMEDIATELY before
    cancelling it - never trust anything earlier in the conversation as
    still current. ALWAYS pass `language` as "ar" or "en" matching what
    you're about to reply in (see lookup_appointment). Returns:
    {"status": "active", "appointment": {...}}
    {"status": "already_cancelled", "appointment": {...}}
    {"status": "not_found"}
    {"status": "error"}  # the booking API call itself failed - a technical
                          # problem, NOT the same as "booking not found"
    """

    base_url = _base_url(state)
    result = api.get_bookings_by_ref(base_url, ref_number, language=language)

    if not result["success"]:
        logger.error(
            "check_booking_status API call failed: base_url=%s ref=%r status_code=%s error=%s",
            base_url, ref_number, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    appt = _shape_appointment(items[0], timezone_name, conversation_language(state))

    if appt.get("statusName") == CANCELLED_STATUS_NAME:
        return {"status": "already_cancelled", "appointment": appt}

    return {"status": "active", "appointment": appt}


@tool
def cancel_appointment(
    state: Annotated[AgentState, InjectedState],
    booking_id: str,
) -> dict:
    """Cancel a booking by its internal id (from a previous tool's
    "appointment"/"id" field - NEVER the human-readable reference
    number). Always call check_booking_status on the same booking
    immediately before this. Returns {"status": "success"} or
    {"status": "error"}."""

    base_url = _base_url(state)
    result = api.cancel_booking_by_guid(base_url, booking_id)

    if result["success"]:
        return {"status": "success"}

    return {"status": "error"}


# ==========================================================
# OTP (dummy provider by default, Authentica when configured) - internal
# mechanics unchanged from the old tools.py, only the return shape changed
# ==========================================================

_otp_storage: Dict[str, dict] = {}

# An OTP record is unusable the moment it passes OTP_TTL_SECONDS -
# verify_otp already rejects it. Without eviction, though, the dict kept
# every code ever sent for the life of the process: a long-running
# container accumulates one permanent entry per patient who ever started
# identity verification, none of which can ever be used again. Pruned
# lazily (on write, which is the only path that grows the dict) rather
# than on a timer, so there is no background thread to reason about.
_OTP_PRUNE_EVERY = 50
_otp_writes_since_prune = 0


def _prune_otp_storage() -> None:
    global _otp_writes_since_prune

    _otp_writes_since_prune += 1
    if _otp_writes_since_prune < _OTP_PRUNE_EVERY:
        return

    _otp_writes_since_prune = 0
    now = time.time()
    expired = [
        key for key, record in _otp_storage.items()
        if now - record.get("created_at", 0) > OTP_TTL_SECONDS
    ]
    for key in expired:
        _otp_storage.pop(key, None)

    if expired:
        logger.info("_prune_otp_storage: evicted %d expired OTP record(s)", len(expired))


@tool
def send_otp(state: Annotated[AgentState, InjectedState], phone: str) -> dict:
    """Send an OTP code to the given phone number (the number ON FILE
    for the booking, not necessarily what the user typed). Returns
    {"status": "otp_sent"}, or {"status": "otp_not_needed_matches_channel"}
    if this number turns out to match the user's own verified channel
    identity (see note below) - in that case, treat it exactly like a
    successful compare_phone match: skip OTP entirely and continue
    straight to looking up the appointment.

    SAFETY NET: this checks the phone number against the channel
    identity itself before sending anything, even though you should
    already have called `compare_phone` before ever calling this tool -
    this is a defensive backstop in case that step was skipped, not a
    replacement for calling `compare_phone` first."""

    normalized = normalize_phone_number(phone, state)

    channel_phone = state.get("channel_phone")
    normalized_channel = normalize_phone_number(channel_phone, state) if channel_phone else None

    if normalized_channel and normalized and normalized_channel == normalized:
        logger.warning(
            "send_otp called for phone=%r which matches channel_phone=%r - "
            "skipping OTP entirely (compare_phone should have caught this "
            "before send_otp was ever called)",
            normalized, normalized_channel,
        )
        return {"status": "otp_not_needed_matches_channel"}

    if OTP_PROVIDER == "authentica":
        api.authentica_send_otp(normalized)
        return {"status": "otp_sent"}

    _otp_storage[normalized] = {"otp": TEST_OTP, "created_at": time.time()}
    _prune_otp_storage()
    logger.info("OTP sent for %s (test otp=%s)", normalized, TEST_OTP)
    return {"status": "otp_sent"}


@tool
def verify_otp(state: Annotated[AgentState, InjectedState], phone: str, otp: str) -> dict:
    """Verify a user-entered OTP code against the one sent to `phone`.
    Returns {"status": "otp_valid"} or {"status": "otp_invalid"}.

    IMPLEMENTATION NOTE (not something the caller has to do anything
    about): `state` is injected automatically and is used ONLY to
    normalize `phone` exactly the way `send_otp` normalized it when it
    stored the code.

    CONFIRMED REAL PRODUCTION BUG this fixes: `send_otp` normalized with
    the client's own country code while this function normalized without
    it (plain module default). For any patient who typed a local number
    ("01155611045") the two produced DIFFERENT keys - the code was
    stored under one and looked up under the other - so a correct OTP
    was rejected 100% of the time and identity verification could never
    complete.
    """

    normalized = normalize_phone_number(phone, state)

    if OTP_PROVIDER == "authentica":
        result = api.authentica_verify_otp(normalized, otp)
        if result["success"]:
            _mark_phone_verified(state, phone)
            return {"status": "otp_valid"}
        return {"status": "otp_invalid"}

    record = _otp_storage.get(normalized)

    if not record:
        return {"status": "otp_invalid"}

    if time.time() - record["created_at"] > OTP_TTL_SECONDS:
        return {"status": "otp_invalid"}

    if str(otp).strip() == str(record["otp"]):
        _mark_phone_verified(state, phone)
        return {"status": "otp_valid"}

    return {"status": "otp_invalid"}


# ==========================================================
# Medical Concierge (symptom -> specialty -> available doctor guidance)
# ==========================================================
#
# Confirmed directly by the user: the Doctors/Specialties API is scoped
# to the correct clinic by its own base_url alone (like GuestBookings),
# on a separate port (1102) from GuestBookings (1101). Response shapes
# confirmed directly from the API's own Swagger "Execute" output.

def _doctors_base_url(state: AgentState) -> Optional[str]:
    """Returns this client's configured Doctors/Specialties API base_url,
    or None if it isn't configured for this client at all. Deliberately
    NEVER falls back to some other client's URL - see config.py's
    extensive comment on why (a real cross-tenant data leak risk)."""

    return (state.get("templates") or {}).get("_doctors_base_url")


# ==========================================================
# Booking session store (moved ABOVE the doctor/specialty tools)
# ==========================================================
#
# This used to live further down, next to match_entity_for_booking. It
# was moved up because find_available_doctors / find_best_doctor_in_specialty /
# list_branches_for_specialty ALL need to record the list they just
# showed the user, so that a bare-number reply ("6") can be resolved
# against it later.
#
# THE BUG THIS FIXES (confirmed from a real production log): the agent
# listed 7 doctors via find_available_doctors, the user replied "6", and
# match_entity_for_booking answered "الدكتور رقم 6 في القائمة غير موجود".
# The number path only ever consulted session["last_list"], and ONLY
# match_entity_for_booking ever wrote to it - so any list produced by a
# different tool left it empty and every numeric pick failed, which in
# turn meant no booking could ever complete from a specialty search.

_BOOKING_SESSIONS: Dict[str, dict] = {}

# A booking session holds a doctor list, a branch list and specialty ids
# - a few KB per conversation. `create_new_booking` clears it on
# success, but an ABANDONED booking (the overwhelming majority: the
# patient stops replying half way through) left its entry behind
# forever. On a container that stays up for weeks that is an unbounded
# leak, one entry per abandoned conversation.
#
# Sessions older than this are dropped. Deliberately generous - far
# longer than config.SESSION_TIMEOUT_SECONDS, after which the
# conversation itself starts fresh anyway - so this can only ever
# collect sessions that are already dead, never one still in use.
_BOOKING_SESSION_TTL_SECONDS = 6 * 3600
_BOOKING_SESSION_PRUNE_EVERY = 100
_booking_session_touches_since_prune = 0


def _prune_booking_sessions() -> None:
    global _booking_session_touches_since_prune

    _booking_session_touches_since_prune += 1
    if _booking_session_touches_since_prune < _BOOKING_SESSION_PRUNE_EVERY:
        return

    _booking_session_touches_since_prune = 0
    now = time.monotonic()
    stale = [
        key for key, session in _BOOKING_SESSIONS.items()
        if now - session.get("_touched_at", now) > _BOOKING_SESSION_TTL_SECONDS
    ]
    for key in stale:
        _BOOKING_SESSIONS.pop(key, None)

    if stale:
        logger.info("_prune_booking_sessions: evicted %d abandoned booking session(s)", len(stale))


def _get_booking_session(session_id: str) -> dict:
    session = _BOOKING_SESSIONS.setdefault(session_id, {
        "doctor_id": None, "branch_id": None, "service_id": None,
        "last_list": None,  # {"entity_type": "doctor"/"branch", "items": [shaped items]}
        "specialty_ids": None,  # remembered so later steps reuse the same specialties
        # Every branch/doctor name any tool has EVER returned in this
        # conversation, kept SEPARATE from "last_list" and never
        # overwritten by a later list - see _remember_list below for why
        # this exists alongside last_list rather than replacing it.
        "known_branch_names": set(),
        "known_doctor_names": set(),
        # Every phone number this session has actually PROVEN belongs
        # to the person messaging - via a `compare_phone` match against
        # their own channel identity, or a successful `verify_otp` - see
        # `_mark_phone_verified` below. This is a SERVER-SIDE gate, not
        # a prompt instruction: `lookup_appointment`'s phone path and
        # `create_new_booking` both refuse to run against a phone
        # number that isn't in here (or isn't the channel identity
        # itself), regardless of what the model believes it already
        # did. Prompt instructions alone are not enforcement - this is.
        "verified_phones": set(),
    })
    session["_touched_at"] = time.monotonic()
    _prune_booking_sessions()
    return session


def _mark_phone_verified(state: AgentState, phone: Optional[str]) -> None:
    """Record that `phone` has been proven to belong to this
    conversation's user - via a channel-identity match or a successful
    OTP - so `lookup_appointment`/`create_new_booking` will accept it
    without re-verifying. See `_get_booking_session`'s
    `verified_phones` for why this exists as a separate, persistent,
    session-scoped set rather than trusting the model's own account of
    what it already checked."""

    session_id = state.get("session_id")
    normalized = normalize_phone_number(phone, state) if phone else None
    if not session_id or not normalized:
        return

    session = _get_booking_session(session_id)
    session["verified_phones"].add(normalized)


def _phone_is_verified(state: AgentState, phone: Optional[str]) -> bool:
    """True when `phone` is either this conversation's own verified
    channel identity, or has been proven via `_mark_phone_verified`
    (a real compare_phone match or a successful verify_otp) earlier in
    THIS session. False for anything else - including a phone number
    that merely LOOKS like it was mentioned earlier, or one the model
    simply asserts is fine."""

    normalized = normalize_phone_number(phone, state) if phone else None
    if not normalized:
        return False

    channel_normalized = normalize_phone_number(state.get("channel_phone"), state) if state.get("channel_phone") else None
    if channel_normalized and normalized == channel_normalized:
        return True

    session_id = state.get("session_id")
    if not session_id:
        return False

    session = _BOOKING_SESSIONS.get(session_id)
    if not session:
        return False

    return normalized in (session.get("verified_phones") or set())



def _remember_specialty_ids(session: dict, specialty_ids: Optional[list]) -> None:
    """Fold newly-used specialty ids into the session's remembered set,
    rather than replacing it outright.

    WHY: confirmed real production failure. The model is repeatedly told
    to pass ALL plausibly-matching specialty ids in ONE call (see
    find_available_doctors/list_branches_for_specialty docstrings), but
    in practice it sometimes calls the same tool once PER specialty
    instead - e.g. find_available_doctors(['باطنة']) immediately followed
    by find_available_doctors(['نساء وتوليد']). Each call used to
    OVERWRITE session["specialty_ids"] outright, so after the second
    call the session only remembered نساء وتوليد - the patient's actual
    (باطنة) doctor then vanished from every later specialty-filtered
    lookup. `match_entity_for_booking` uses this same session value to
    narrow its doctor query, and with only the wrong specialty id left
    in it, a real, bookable doctor came back as "couldn't find them in
    the system" - a booking dead-end for a doctor who was on-screen
    moments earlier. Merging (union, order-preserving) means a second
    call for a different specialty ADDS to what's remembered instead of
    silently discarding the first."""

    if not specialty_ids:
        return

    existing = session.get("specialty_ids") or []
    merged = list(existing)
    for sid in specialty_ids:
        if sid not in merged:
            merged.append(sid)
    session["specialty_ids"] = merged


def _remember_list(state: AgentState, entity_type: str, items: list) -> None:
    """Record the list of doctors/branches just returned to the model, so
    a later bare-number reply can be resolved against the SAME ordering
    the user actually saw. Every tool that returns a user-facing list
    must call this - see the comment above _BOOKING_SESSIONS.

    ALSO folds any branch/doctor name in `items` into the session's
    PERMANENT known-name memory (`known_branch_names`/
    `known_doctor_names`), separate from and in addition to
    `last_list`.

    WHY THIS SECOND, NEVER-OVERWRITTEN STORE EXISTS: `last_list` holds
    only the MOST RECENT list - the very next tool call (a different
    branch's services, a different specialty's doctors) replaces it
    outright, by design, since it exists purely to resolve THIS turn's
    bare-number reply against THIS turn's list. graph.py's
    invented-branch/invented-doctor guards need the opposite property:
    "has this name ever legitimately come from a tool in this whole
    conversation", which must NOT be forgotten the moment a newer list
    is shown. Before this, those guards read only `state["messages"]`
    (the raw chat history) - CONFIRMED REAL PRODUCTION FAILURE
    (medtown, 2026-08-31): a reply correctly named "فرع الطوارئ" (a
    branch the patient had already been shown by name three turns
    earlier) and was rejected twice as an invented branch, forcing the
    generic fallback error, because whatever the guard could still see
    in that turn's message history no longer surfaced that name as
    substring text. `last_list` had already moved on to that turn's
    service list by the time the check ran, so it couldn't help either
    - this dedicated accumulator is the fix: once a name is seen, it
    stays known for the rest of the conversation, independent of
    whatever else has been shown since."""

    session_id = state.get("session_id")
    if not session_id:
        return

    session = _get_booking_session(session_id)
    session["last_list"] = {"entity_type": entity_type, "items": list(items)}

    if entity_type in ("branch", "doctor"):
        bucket = session.setdefault(
            "known_branch_names" if entity_type == "branch" else "known_doctor_names",
            set(),
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("name", "altName", "formatedName", "doctorName"):
                value = item.get(key)
                if value:
                    bucket.add(str(value))

    logger.info(
        "_remember_list: session_id=%s entity_type=%s count=%d",
        session_id, entity_type, len(items),
    )


def get_known_entity_names(session_id: Optional[str], entity_type: str) -> set:
    """Public accessor for graph.py's invented-branch/invented-doctor
    guards - every branch or doctor name any tool has returned in this
    session so far, regardless of what the most recent list was. See
    `_remember_list` for why this is tracked separately from
    `last_list`. Returns an empty set for a missing/unknown session
    rather than creating one - a read-only check should never have the
    side effect of starting a fresh booking session."""

    if not session_id:
        return set()

    session = _BOOKING_SESSIONS.get(session_id)
    if not session:
        return set()

    key = "known_branch_names" if entity_type == "branch" else "known_doctor_names"
    return set(session.get(key) or set())


# Arabic-Indic (٠-٩) and Extended Arabic-Indic (۰-۹) digits -> ASCII.
# Patients on Arabic keyboards routinely reply "٦" rather than "6"; the
# old code path only ever handled ASCII, so those replies fell through
# to fuzzy name matching and failed.
_ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _extract_selection_number(user_input: str) -> Optional[int]:
    """Return the list position the user meant, or None if their message
    isn't a positional pick. Accepts a bare number in either digit set
    ("6", "٦"), and also short phrasings that wrap one ("رقم 6",
    "الدكتور 6", "no. 6") - but deliberately NOT a long sentence that
    merely happens to contain a digit, which would risk hijacking a
    real name/date."""

    if not user_input:
        return None

    text = user_input.translate(_ARABIC_DIGIT_MAP).strip()

    if text.isdigit():
        return int(text)

    # Short wrapper phrases only - cap the length so e.g. a free-text
    # sentence with a number in it isn't misread as a selection.
    if len(text) <= 25:
        numbers = re.findall(r"\d+", text)
        if len(numbers) == 1:
            leftover = re.sub(r"[\d\s.،,:\-#]", "", text)
            if leftover in ("رقم", "الرقم", "دكتور", "الدكتور", "د", "فرع", "الفرع",
                            "no", "No", "num", "number", "option", "choice"):
                return int(numbers[0])

    return None


@tool
def list_specialties(state: Annotated[AgentState, InjectedState]) -> dict:
    """List every medical specialty this clinic actually offers. ALWAYS
    call this before suggesting a specialty to a user describing a
    symptom/concern - never guess whether this clinic has a given
    specialty. Returns:
    {"status": "found", "specialties": [{"id": ..., "name": ...,
      "has_available_doctors": true}, ...]}
    {"status": "no_bookable_specialties", "unstaffed_specialties": [names]}
        # The clinic offers specialties on paper but has NO bookable
        # doctor behind ANY of them right now. Say that plainly in your
        # very next reply and offer a staff handoff - do NOT name these
        # specialties as a recommendation, and never ask "shall I fetch
        # the available doctors?" about them; nobody is available.
    {"status": "not_found"}  # this clinic has no specialties registered
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}  # the API call itself failed

    IMPORTANT: `specialties` contains ONLY specialties that actually
    have a bookable doctor right now. Ones with no available doctor are
    deliberately left out, so anything in this list is safe to
    recommend, and you can never accidentally walk a patient toward an
    empty specialty. Never add a specialty of your own that isn't here.
    (When availability couldn't be checked at all this call, the
    `has_available_doctors` field is omitted and nothing is filtered -
    treat that as unknown, not as unavailable.)

    IMPLEMENTATION NOTE: this uses the Specialties/GetList endpoint
    directly. An earlier version derived specialties from the doctors
    endpoint instead, after an initial test call to Specialties/GetList
    returned mismatched placeholder data ("New NEw", unrelated ids).
    That turned out to be stale/unrelated test data, not a real problem
    with the endpoint - a follow-up call (after fixing pageNumber=1 and
    the /GetList path) returned the correct, complete specialty list,
    confirmed to share the exact same ids as the doctors' own
    specialtyId field. Using this endpoint (rather than deriving from
    doctors) is more correct: it includes every specialty this clinic
    has registered, even ones with zero doctors currently assigned -
    letting the agent correctly say "we don't offer that" only when
    truly true, rather than only when nobody happens to be staffed."""

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("list_specialties called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_specialties(base_url, language=conversation_language(state))

    if not result["success"]:
        logger.error(
            "list_specialties API call failed: base_url=%s status_code=%s error=%s",
            base_url, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    # The API's name fields are usually populated (confirmed against
    # real data), but fall back through the alternatives defensively -
    # dropping anything with no usable name at all rather than
    # surfacing a blank to the user.
    specialties = []
    for item in items:
        if not item.get("id"):
            continue

        # Arabic conversations must see the Arabic specialty name -
        # confirmed real user feedback: English specialty names were
        # appearing inside otherwise-Arabic replies.
        name = _preferred_name(item, conversation_language(state)) or item.get("code")

        if not name or not str(name).strip():
            logger.warning("Skipping specialty with no usable name: id=%s", item.get("id"))
            continue

        specialties.append({"id": item["id"], "name": str(name).strip()})

    # Flag which specialties actually have a BOOKABLE doctor behind them.
    # The specialties endpoint lists everything the clinic has
    # registered, including specialties with nobody currently staffed or
    # schedulable - which is correct for answering "do you offer X?",
    # but on its own it leads the booking/guidance flows into a dead
    # end. Confirmed real production failure: a patient describing
    # headaches and insomnia was recommended "طب نفسي / علاج نفسي",
    # asked whether to fetch doctors, said yes - and only then was told
    # nobody is available in either, after being led all the way there.
    # One extra call here lets the flow steer to a specialty that can
    # actually be booked BEFORE making an offer it can't keep.
    doctors_result = api.get_doctors(base_url, page_size=200, language=conversation_language(state))
    staffed_specialty_ids = set()
    availability_known = False
    if doctors_result["success"]:
        availability_known = True
        for doctor in (doctors_result["data"] or {}).get("items", []):
            # Apply the SAME hasSlots filter find_available_doctors uses.
            # Counting every registered doctor here made this check
            # weaker than the tool it's meant to protect: a specialty
            # whose only doctors have no bookable slots passed as
            # "available", and find_available_doctors then returned
            # nothing for it - reproducing the exact dead end this check
            # exists to prevent. "hasSlots is not False" (rather than
            # "is True") mirrors that tool exactly, so a missing field
            # stays optimistic in both places instead of silently
            # hiding a bookable specialty.
            if doctor.get("hasSlots") is False:
                continue
            specialty_id = doctor.get("specialtyId")
            if specialty_id:
                staffed_specialty_ids.add(specialty_id)
    else:
        logger.warning(
            "list_specialties: could not check doctor availability per specialty (status_code=%s error=%s) - "
            "returning specialties without availability flags",
            doctors_result.get("status_code"), doctors_result.get("error"),
        )

    unstaffed_names = []

    if availability_known:
        for specialty in specialties:
            specialty["has_available_doctors"] = specialty["id"] in staffed_specialty_ids

        # Physically SEPARATE the unstaffed specialties out of the main
        # list rather than just flagging them. The prompt already forbids
        # recommending a specialty with has_available_doctors=false, and
        # it kept happening anyway: confirmed real production failure
        # (twice) - a patient with headaches and insomnia was recommended
        # two psychiatry specialties, asked "تحبين أجيب لك دكاترة متاحين
        # في هالتخصصات؟", said yes, and only THEN was told nobody is
        # available in either. A flag the model has to remember to check
        # is not a guarantee; a specialty that isn't in the list it reads
        # from cannot be recommended at all.
        staffed = [s for s in specialties if s.get("has_available_doctors")]
        unstaffed = [s for s in specialties if not s.get("has_available_doctors")]
        unstaffed_names = [s.get("name") for s in unstaffed if s.get("name")]
        specialties = staffed

    logger.info(
        "list_specialties: %d specialties returned, %d bookable, %d with no available doctors (%s)",
        len(items), len(specialties), len(unstaffed_names), unstaffed_names or "none",
    )

    if not specialties:
        # Every specialty this clinic offers is currently unstaffed (or
        # none matched). Say so in the FIRST reply instead of offering to
        # look for doctors that don't exist.
        if unstaffed_names:
            return {"status": "no_bookable_specialties", "unstaffed_specialties": unstaffed_names}
        return {"status": "not_found"}

    # CRITICAL: record the exact list, in the exact order, that the
    # model is about to show - the same reason every other user-facing
    # list in this file does this (see _remember_list). Before this,
    # `list_specialties` was the one list-producing tool that never
    # called it, so a bare "1" reply had nothing deterministic to
    # resolve against and the model had to recall the specialty id from
    # memory/context instead - see `_resolve_specialty_for_booking` and
    # `find_available_doctors`'s `specialty_name` parameter.
    _remember_list(state, "specialty", specialties)

    return {"status": "found", "specialties": specialties}


def _shape_doctor_list(raw_items: list, language: str = "ar") -> list:
    """Shape raw Doctors/GetList rows into the dict the model sees AND
    the dict remembered for positional selection - one function so the
    two can never drift apart. `altName`/`formatedName` are carried
    through (not just a single collapsed `name`) so that a later
    positional pick can still resolve the Arabic-preferred display name
    via _arabic_preferred_name, even when the live roster is filtered
    differently and the row can't be re-fetched.

    `name` (the field the model actually reads out) is resolved in the
    CONVERSATION's language - Arabic conversations get `altName`, the
    confirmed Arabic translation of `name`/`formatedName`. Previously
    `name` (English) always won here, which is why English doctor and
    specialty names kept appearing inside otherwise-Arabic replies even
    though the model was following its instructions to copy tool values
    verbatim.
    """

    doctors = []

    for i in raw_items:
        name = _preferred_name(i, language) or i.get("name") or i.get("formatedName") or i.get("altName")

        if not name or not str(name).strip():
            logger.warning("Skipping doctor with no usable name: id=%s", i.get("id"))
            continue

        doctors.append({
            "id": i.get("id"),
            "name": str(name).strip(),
            "formatedName": i.get("formatedName") or i.get("name"),
            "altName": i.get("altName"),
            "specialtyName": i.get("specialtyAltName") if (language != "en" and i.get("specialtyAltName")) else i.get("specialtyName"),
            "degreeName": i.get("degreeAltName") if (language != "en" and i.get("degreeAltName")) else i.get("degreeName"),
        })

    return doctors


def _doctors_with_real_slots(state: AgentState, base_url: str, doctor_ids: list,
                              branch_id: Optional[str] = None) -> Optional[set]:
    """Which of `doctor_ids` genuinely have at least one open (non-booked)
    schedule slot - at `branch_id` if given, or across ALL branches when
    `branch_id` is None (used by `find_available_doctors`'s clinic-wide/
    no-branch-chosen search) - within the normal booking window. ONE
    batched call for the whole roster, not one per doctor.

    WHY THIS EXISTS: confirmed real production failure. The doctor-list
    endpoint's own `hasSlots` flag is trusted leniently on purpose
    (excluding a doctor only when it is explicitly False - see
    `find_available_doctors`'s own comment on why treating a missing/
    null flag as "unavailable" was a worse, previously-fixed bug). That
    leniency means a doctor whose `hasSlots` came back null/missing but
    who genuinely has ZERO open slots still gets shown as a bookable
    choice. A patient picked exactly such a doctor from a branch's
    roster and only found out two turns later - after already
    committing to her - that `list_available_days_for_booking` had
    nothing at all for her. This cross-checks against the SAME real
    schedule-slots endpoint used for actual booking, in one call for
    every candidate doctor at once, so the roster shown matches what
    booking will actually honour.

    Returns None (meaning "unknown, don't filter anything out") if EVERY
    verification call fails - a transient error here must not silently
    hide every doctor, the same principle `_open_slots_on_day` already
    follows elsewhere in this file.

    ONE QUERY PER DOCTOR, ON PURPOSE.
    --------------------------------
    This used to make a single batched call for the whole roster and
    then group the returned slots by each slot's `doctorId`. That is the
    EXACT pattern `_branches_with_real_slots` already had to abandon:
    the slots endpoint returns slotStart/slotEnd/isBooked, and the
    id fields it echoes back are not something this project can rely on.
    When `doctorId` is absent, the grouping finds nothing for anybody,
    and every doctor whose `hasSlots` flag was ambiguous gets dropped
    from the roster.

    CONFIRMED REAL PRODUCTION FAILURE: د. وائل عويس has a live rota at
    فرع الدقي (Mon/Wed/Thu 10:00-12:00, effective 2026-07-13 to
    2026-08-31 - i.e. genuinely in effect), and `find_available_doctors`
    listed him at that branch correctly. The branch roster built through
    THIS function silently dropped him, so the patient could never pick
    him. `find_available_doctors` does not use this cross-check, which
    is exactly why the same doctor appeared in one list and not the
    other.

    Asking per doctor means the ANSWER is what the query was scoped to,
    rather than something inferred from a field that may not be there.
    That is a handful of calls, made once, at the only point it matters.
    """

    if not doctor_ids:
        return set()

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now = datetime.now(tz)
    window_end = now + timedelta(days=DOCTOR_AVAILABILITY_WINDOW_DAYS)

    have_slots = set()
    any_lookup_succeeded = False

    for doctor_id in doctor_ids:
        if not doctor_id:
            continue

        result = api.get_doctor_schedule_slots(
            base_url, doctor_ids=[doctor_id],
            branch_ids=[branch_id] if branch_id else None,
            from_date=now.isoformat(), to_date=window_end.isoformat(),
            is_booked=False, page_size=1000,
            language=conversation_language(state),
        )

        if not result["success"]:
            # This doctor's check failed. Treat them as available rather
            # than hiding someone who may well have appointments.
            logger.warning(
                "_doctors_with_real_slots: verification call failed for doctor_id=%s at "
                "branch_id=%s (status_code=%s) - keeping them rather than hiding real "
                "availability on a transient error",
                doctor_id, branch_id, result.get("status_code"),
            )
            have_slots.add(doctor_id)
            continue

        any_lookup_succeeded = True

        for item in (result["data"] or {}).get("items", []):
            if item.get("isBooked"):
                continue
            # The query was scoped to THIS doctor, so any open slot in
            # the response is theirs - no id field needs to be trusted.
            have_slots.add(doctor_id)
            break

    if not any_lookup_succeeded:
        logger.warning(
            "_doctors_with_real_slots: every verification lookup failed at branch_id=%s - "
            "reporting 'unknown' so nobody gets dropped on a transient error",
            branch_id,
        )
        return None

    return have_slots


def _doctors_at_branch(state: AgentState, base_url: str, branch_id: str) -> list:
    """Fetch the available doctors at one branch, narrowed to the
    specialty AND service this booking is already about (when known),
    and remember the list for positional selection.

    Exists because "which doctors are here" changes the moment a branch
    is confirmed - not every doctor works at every branch. Without this,
    the model would re-display doctor names it had shown BEFORE the
    branch was picked, which is both wrong (some of them don't work
    there) and unselectable (the remembered list at that point is the
    BRANCH list, so a reply of "2" resolves to nothing).

    CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-31): a booking
    that started from a SERVICE pick ("جلسة إستشارة أخصائي التغذية")
    never had `specialty_ids` set on the session - only `service_id`
    was. The patient then typed a branch name directly ("النزهة"),
    which resolves through this function - and this function only ever
    filtered by `specialty_ids`, never by `service_id`, even though
    `api.get_doctors` supports `service_ids` for exactly this case (see
    its own docstring) and other call sites in this same file already
    use it correctly. The result: this returned ALL 4 doctors who work
    at that branch generally, with zero filtering for whether any of
    them actually offer this specific service - and the model then
    told the patient "no doctors available for this service", a claim
    the (unfiltered) tool result never actually supported either way."""

    session = _get_booking_session(state.get("session_id"))
    specialty_ids = session.get("specialty_ids") or None
    service_ids = [session["service_id"]] if session.get("service_id") else None

    now = datetime.utcnow()
    result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids,
        service_ids=service_ids,
        branch_ids=[branch_id],
        has_published_service=True,
        has_service_schedule=True,
        intersection_start=now.isoformat() + "Z",
        intersection_end=(now + timedelta(days=DOCTOR_AVAILABILITY_WINDOW_DAYS)).isoformat() + "Z",
     language=conversation_language(state),)

    if not result["success"]:
        logger.error(
            "_doctors_at_branch: get_doctors failed for branch_id=%s: status_code=%s error=%s",
            branch_id, result.get("status_code"), result.get("error"),
        )
        return []

    available = [i for i in (result["data"] or {}).get("items", []) if i.get("hasSlots") is not False]
    doctors = _shape_doctor_list(available, conversation_language(state))

    # ONE extra batched call cross-checks the roster against real
    # schedule slots, so a doctor whose hasSlots flag was ambiguous but
    # who genuinely has nothing open is not offered as a bookable
    # choice - see _doctors_with_real_slots for why.
    verified_ids = _doctors_with_real_slots(
        state, base_url, [d["id"] for d in doctors if d.get("id")], branch_id,
    )
    if verified_ids is not None:
        before = len(doctors)
        dropped = [d for d in doctors if d.get("id") not in verified_ids]
        doctors = [d for d in doctors if d.get("id") in verified_ids]
        if dropped:
            # NAMES, not just a count. A doctor silently vanishing from a
            # branch roster is a reported complaint ("الدكتور وائل
            # متعرضش مع انه موجود"), and a bare count made it impossible
            # to tell whether the filter was right (genuinely nothing
            # open) or wrong (a slot-sweep gap, like the one
            # `_branches_with_real_slots` needed its future-rota
            # exemption for). Logging who was dropped makes that
            # answerable from a single production trace.
            logger.info(
                "_doctors_at_branch: dropped %d of %d doctor(s) at branch_id=%s - no open "
                "slot found in the booking window despite an ambiguous hasSlots flag: %s",
                len(dropped), before, branch_id,
                [{"id": d.get("id"), "name": d.get("name")} for d in dropped],
            )

    if doctors:
        _remember_list(state, "doctor", doctors)

    # KEEP THE "THIS BRANCH IS EMPTY" NOTE HONEST.
    #
    # `_note_info_branch_availability` sets `info_branch_no_doctors`
    # when the INFO flow shows an empty branch, and the booking-intent
    # directive reads it back on a later turn. Nothing was clearing it
    # when the patient moved to a DIFFERENT branch through the booking
    # tools (which go through here, not through `match_entity_info`), so
    # the note went stale.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: فرع المعادي was browsed first
    # (correctly noted as empty), the patient then moved to فرع الدقي,
    # this function returned FOUR real doctors there - and the reply
    # still said "فرع الدقي ما فيه دكاترة متاحين حاليا للحجز", straight
    # from the stale note, contradicting the tool result in the very
    # same turn.
    session_id = state.get("session_id")
    if session_id:
        booking_session = _get_booking_session(session_id)
        if doctors:
            booking_session.pop("info_branch_no_doctors", None)
        elif booking_session.get("info_branch_id") == branch_id:
            booking_session["info_branch_no_doctors"] = booking_session.get("info_branch_name")

    logger.info(
        "_doctors_at_branch: branch_id=%s specialty_ids=%s -> %d doctor(s)",
        branch_id, specialty_ids, len(doctors),
    )

    return doctors


def _branch_ids_with_available_doctors(
    state: AgentState, base_url: str, specialty_ids: Optional[list] = None,
    days_ahead: int = DOCTOR_AVAILABILITY_WINDOW_DAYS,
) -> Optional[set]:
    """Which branch ids CURRENTLY have at least one bookable, scheduled
    doctor - narrowed to `specialty_ids` when given (the specialty this
    booking is already about), clinic-wide otherwise.

    Exists so a branch that is real in the system but currently has
    nobody staffed there (e.g. a "فرع المعادي" with zero doctors) is
    never offered as a fuzzy-match suggestion or silently confirmed as
    if it were a real option. Uses the exact same
    intersection_start/intersection_end availability window as every
    other doctor lookup in this file, via `get_doctors` + the doctor
    schedule endpoint's per-row `branchId` (the Doctors endpoint itself
    doesn't reliably carry which branch each doctor is at - see
    `list_branches_for_specialty` for the same pattern).

    Returns None on an API failure (caller should then skip filtering
    rather than hiding every branch on a transient error), or a
    (possibly empty) set of branch ids on success."""

    now = datetime.utcnow()
    intersection_start = now.isoformat() + "Z"
    intersection_end = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    doctors_result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids or None,
        has_published_service=True,
        has_service_schedule=True,
        intersection_start=intersection_start,
        intersection_end=intersection_end,
        language=conversation_language(state),
    )

    if not doctors_result["success"]:
        logger.warning(
            "_branch_ids_with_available_doctors: get_doctors failed - status_code=%s error=%s",
            doctors_result.get("status_code"), doctors_result.get("error"),
        )
        return None

    doctor_items = [i for i in (doctors_result["data"] or {}).get("items", []) if i.get("hasSlots") is not False]
    doctor_ids = [i.get("id") for i in doctor_items if i.get("id")]

    if not doctor_ids:
        return set()

    effective_date = None
    try:
        timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
        effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    except Exception:
        logger.exception("_branch_ids_with_available_doctors: failed to compute today's date")

    schedule_result = api.get_doctor_schedule(
        base_url, doctor_ids=doctor_ids, page_size=500,
        effective_date=effective_date, include_future=True,
        language=conversation_language(state),
    )

    if not schedule_result["success"]:
        logger.warning(
            "_branch_ids_with_available_doctors: get_doctor_schedule failed - status_code=%s error=%s",
            schedule_result.get("status_code"), schedule_result.get("error"),
        )
        return None

    return {row.get("branchId") for row in (schedule_result["data"] or {}).get("items", []) if row.get("branchId")}


def _resolve_branch_by_name(base_url: str, branch_name: str, language: str = "ar", state=None) -> Optional[dict]:
    """Fuzzy-match the user's raw branch text against the clinic's real
    branch list. Returns the raw branch row, or None if nothing matched
    confidently. Used by the specialty -> branch -> doctor sequence so
    the model never has to carry a branch id itself.

    `state` is optional only so existing/direct callers keep working;
    pass it whenever available, since it carries the client's configured
    bilingual branch names - without it, a branch typed in a different
    language than the API returns it in silently fails to match (see
    _with_branch_aliases)."""

    branches_result = api.get_branches(base_url, page_size=200, language=language)

    if not branches_result["success"]:
        logger.error(
            "_resolve_branch_by_name: get_branches failed: status_code=%s error=%s",
            branches_result.get("status_code"), branches_result.get("error"),
        )
        return None

    branch_items = (branches_result["data"] or {}).get("items", [])
    if state is not None:
        branch_items = _with_branch_aliases(branch_items, state)

    # TWO-TIER RESOLUTION - mirrors match_entity_info's own fix for the
    # identical underlying problem (see there for the full rationale).
    #
    # An EXACT/near-exact reference - the patient really did type this
    # specific branch's name - must resolve to THAT branch, regardless
    # of whether it currently has a doctor. Only a WEAK/guessed match
    # should ever be restricted to branches that currently have one.
    #
    # CONFIRMED REAL PRODUCTION FAILURE (worse than the one this
    # filtering was originally added to fix): the patient typed
    # "المعادي" - a REAL, EXACTLY-NAMED branch, just one with no doctors
    # right now - and because Maadi had been filtered out of the
    # candidate pool BEFORE matching even started, the fuzzy match was
    # forced to guess among the remaining (active) branches and silently
    # locked in "الدقي" (Dokki) instead - a completely different, real
    # branch that the patient never mentioned. `find_available_doctors`
    # then confirmed that WRONG branch into the session with no
    # confirmation step at all, and displayed its doctors while still
    # labeling the reply "فرع المعادي". Filtering the pool BEFORE
    # checking for an exact reference silently swaps one real branch for
    # another - worse than the original bug (a typo pointing at an empty
    # branch), because here the patient's own exact words are discarded.
    exact_probe = _fuzzy_match(branch_name, branch_items, ["name", "altName", "formatedName", "cityName", "_configAliases"])

    if exact_probe["result"] == "matched" and exact_probe.get("score", 0) >= 0.95:
        # A confident, explicit reference - never narrowed further. Let
        # the normal downstream availability check (in
        # `find_available_doctors`, via `not_found_in_branch`) be the one
        # to say honestly "this branch has nobody right now" - that is
        # a true statement about a real branch, not a wrong branch
        # pretending to be the right one.
        return exact_probe["item"]

    if state is not None:
        # Not a confident/explicit reference - this is a GUESS, so it
        # must never be allowed to land on a branch with no currently
        # available doctor. Narrow the candidate pool to branches that
        # ACTUALLY have a bookable doctor right now, before guessing.
        # CONFIRMED REAL PRODUCTION FAILURE: the raw text "فرع المنار"
        # (not a real branch name at all) fuzzy-matched to "فرع المعادي"
        # - a real branch that currently has zero doctors - and the
        # patient was walked into a dead end instead of being shown a
        # branch that actually has someone.
        #
        # Falls back to the UNFILTERED list whenever the filter itself
        # can't be trusted: an API failure (`None`), or a filter that
        # would leave zero candidates (better to fuzzy-match against the
        # real names and let the normal "not_found_in_branch" handling
        # further downstream explain the branch has nobody, than to
        # silently refuse to match a real branch name at all).
        session = _get_booking_session(state.get("session_id"))
        specialty_ids = session.get("specialty_ids") or None
        active_branch_ids = _branch_ids_with_available_doctors(state, base_url, specialty_ids)

        if active_branch_ids:
            narrowed = [b for b in branch_items if b.get("id") in active_branch_ids]
            if narrowed:
                branch_items = narrowed
            else:
                logger.info(
                    "_resolve_branch_by_name: no currently-staffed branch to narrow to for "
                    "specialty_ids=%s - falling back to the unfiltered branch list",
                    specialty_ids,
                )

    match_result = _fuzzy_match(branch_name, branch_items, ["name", "altName", "formatedName", "cityName", "_configAliases"])

    if match_result["result"] == "matched":
        return match_result["item"]

    return None


@tool
def list_branches_for_specialty(
    state: Annotated[AgentState, InjectedState],
    specialty_ids: list,
    days_ahead: int = DOCTOR_AVAILABILITY_WINDOW_DAYS,
) -> dict:
    """Show WHICH BRANCHES actually have available doctors in the given
    specialties, together with the doctors at each one.

    Call this when a patient has chosen a specialty and either asks
    which branches exist, says they don't know the branches, or asks
    where a given specialty is available - so you can answer with
    something real ("فرع الدقي: د. كذا، د. كذا / فرع زايد: ...") rather
    than naming branches from memory. Never list branches you did not
    get from this tool or from `match_entity_for_booking`.

    Pass ALL plausibly-matching specialty ids together as a list, the
    same rule as `find_available_doctors` - a general specialty and its
    sub-specialty routinely both matter for one complaint. CONFIRMED
    REAL FAILURE: "رمد" was passed alone and returned nothing, because
    that specialty has zero registered doctors while its sub-specialty
    "جراحة الشبكية" has seven - the patient was told the clinic has no
    eye doctors at all. If two specialties could both cover what the
    patient asked for, pass BOTH ids. Returns:
    {"status": "found", "branches": [{"id", "name", "doctorCount", "doctors": [{"id", "name", "degreeName"}, ...]}, ...]}
    {"status": "found_broader_search", "branches": [...]}  # the given specialty_ids had nobody, so this is EVERY branch/doctor clinic-wide - say so honestly and show each doctor's own specialtyName; don't present them as that specialty
    {"status": "not_found"}  # no branch has an available doctor at all, even clinic-wide
    {"status": "not_configured"} / {"status": "error"}

    The branch list is remembered automatically, so the patient can
    reply with just its number and `match_entity_for_booking` (or
    `find_available_doctors`'s `branch_name`) will resolve it."""

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("list_branches_for_specialty called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    session = _get_booking_session(state.get("session_id"))

    if not specialty_ids:
        specialty_ids = session.get("specialty_ids") or []

    _remember_specialty_ids(session, specialty_ids)

    now = datetime.utcnow()
    intersection_start = now.isoformat() + "Z"
    intersection_end = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    def _fetch(ids):
        result = api.get_doctors(
            base_url,
            specialty_ids=ids or None,
            has_published_service=True,
            has_service_schedule=True,
            intersection_start=intersection_start,
            intersection_end=intersection_end,
         language=conversation_language(state),)

        if not result["success"]:
            return None

        available = [i for i in (result["data"] or {}).get("items", []) if i.get("hasSlots") is not False]
        return _shape_doctor_list(available, conversation_language(state))

    doctors = _fetch(specialty_ids)

    if doctors is None:
        logger.error("list_branches_for_specialty: get_doctors failed for specialty_ids=%s", specialty_ids)
        return {"status": "error"}

    broadened = False

    if not doctors:
        # SAME safety net find_available_doctors already carries, and for
        # the same confirmed reason: the model routinely passes only ONE
        # of several plausibly-relevant specialty ids. Real example from
        # production - "رمد" (7b33ac7b) has ZERO registered doctors while
        # its sub-specialty "جراحة الشبكية" (f33c9b73) has seven, so
        # passing only the general id makes a fully-staffed clinic look
        # empty and dead-ends the booking. Broaden clinic-wide rather
        # than trusting the model to get the id list right, and flag it
        # so the reply can be honest that it's a wider result.
        logger.info("list_branches_for_specialty: 0 doctors for specialty_ids=%s - broadening clinic-wide", specialty_ids)

        doctors = _fetch(None)
        broadened = True

        if doctors is None:
            return {"status": "error"}

    if not doctors:
        logger.info("list_branches_for_specialty: no available doctors even clinic-wide")
        return {"status": "not_found"}

    doctors_by_id = {d["id"]: d for d in doctors if d.get("id")}

    # Which branch is each doctor at? The Doctors endpoint doesn't
    # reliably carry that, but DoctorSchedules does (branchId/branchName
    # per row) - and one call covers every doctor at once.
    #
    # `effective_date`/`include_future` for the same reason as every
    # other schedule lookup in this file: without them, a doctor whose
    # assignment to a branch has already LAPSED still counts as working
    # there for the purposes of this specialty-wide branch listing.
    specialty_effective_date = None
    try:
        timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
        specialty_effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    except Exception:
        logger.exception("list_branches_for_specialty: failed to compute today's date")

    schedule_result = api.get_doctor_schedule(
        base_url, doctor_ids=list(doctors_by_id.keys()), page_size=500,
        effective_date=specialty_effective_date, include_future=True,
     language=conversation_language(state),)

    if not schedule_result["success"]:
        logger.error(
            "list_branches_for_specialty: get_doctor_schedule failed: status_code=%s error=%s",
            schedule_result.get("status_code"), schedule_result.get("error"),
        )
        return {"status": "error"}

    # Cross-reference the real branch list so the Arabic altName/address
    # are used, rather than only the schedule row's plain branchName.
    branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
    branch_rows = {}

    if branches_result["success"]:
        branch_rows = {b.get("id"): b for b in (branches_result["data"] or {}).get("items", []) if b.get("id")}

    grouped: Dict[str, dict] = {}

    for row in (schedule_result["data"] or {}).get("items", []):
        branch_id = row.get("branchId")
        doctor_id = row.get("doctorId")

        if not branch_id or doctor_id not in doctors_by_id:
            continue

        branch_row = branch_rows.get(branch_id) or {}
        entry = grouped.setdefault(branch_id, {
            "id": branch_id,
            "name": _arabic_preferred_name(branch_row) or row.get("branchName"),
            "address": branch_row.get("address"),
            "cityName": branch_row.get("cityName"),
            "altName": branch_row.get("altName"),
            "formatedName": branch_row.get("formatedName"),
            "doctors": [],
            "_doctor_ids": set(),
        })

        if doctor_id not in entry["_doctor_ids"]:
            entry["_doctor_ids"].add(doctor_id)
            doctor = doctors_by_id[doctor_id]
            entry["doctors"].append({
                "id": doctor["id"],
                "name": doctor["name"],
                "degreeName": doctor.get("degreeName"),
                "specialtyName": doctor.get("specialtyName"),
            })

    branches = []

    for entry in grouped.values():
        entry.pop("_doctor_ids", None)
        entry["doctorCount"] = len(entry["doctors"])
        branches.append(entry)

    if not branches:
        logger.info("list_branches_for_specialty: doctors found but no branch mapping for specialty_ids=%s", specialty_ids)
        return {"status": "not_found"}

    branches.sort(key=lambda b: b["doctorCount"], reverse=True)

    _remember_list(state, "branch", branches)

    # A SINGLE BRANCH IS NOT A CHOICE - AND ITS DOCTORS ARE THE REAL LIST.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: exactly this case. Only one
    # branch came back, so the model skipped the pointless "which
    # branch?" step and presented that branch's DOCTORS as a numbered
    # list instead - "1️⃣ د. سارة عبد الله" - which is the right call to
    # make. But nothing had EVER remembered that list under
    # entity_type="doctor" - only the (single-item, never-shown) branch
    # list was remembered - so when the patient answered "1", the tool
    # correctly reported "no doctor list is remembered for this
    # session", and the reply that followed had to deny understanding a
    # number the patient was only ever given because it appeared in the
    # bot's own message a turn earlier.
    #
    # The branch is auto-confirmed here too, for the same reason
    # get_doctor_schedule_for_booking and list_available_days_for_booking
    # already auto-confirm a doctor's one-and-only branch: there is no
    # real decision left to make about it.
    if len(branches) == 1:
        only_branch = branches[0]
        session_id = state.get("session_id")
        if session_id and only_branch.get("id"):
            session = _get_booking_session(session_id)
            if not session.get("branch_id"):
                session["branch_id"] = only_branch["id"]
                session["branch_display_name"] = _arabic_preferred_name(only_branch) or only_branch.get("name")
                logger.info(
                    "list_branches_for_specialty: auto-confirmed the only branch_id=%s (%s)",
                    only_branch["id"], session["branch_display_name"],
                )

        if only_branch.get("doctors"):
            _remember_list(state, "doctor", only_branch["doctors"])
            logger.info(
                "list_branches_for_specialty: remembered %d doctor(s) at the only branch, "
                "under entity_type=doctor, so a positional reply resolves",
                len(only_branch["doctors"]),
            )

    logger.info(
        "list_branches_for_specialty: specialty_ids=%s broadened=%s -> %d branch(es), %d doctor(s)",
        specialty_ids, broadened, len(branches), len(doctors),
    )

    if broadened:
        return {"status": "found_broader_search", "branches": branches}

    return {"status": "found", "branches": branches}


def _booking_branch_is_stale(state, session) -> bool:
    """Whether the branch remembered in the booking session should be
    ignored for the question being asked right now.

    A branch is remembered so that the REST OF THAT BOOKING stays
    consistent with it. It is not a global preference, and it must not
    outlive the flow that set it.

    The signal is which specialist owns this turn. `booking` and
    `reschedule` are the two flows a branch legitimately constrains -
    they are placing or moving a specific appointment. `medical`, `faq`
    and the concierge are answering a different question, and silently
    filtering their results by a branch chosen for an abandoned booking
    turns a specialty the clinic genuinely staffs into "nobody is
    available".

    Returns False when the active agent is unknown, so nothing changes
    on any path that does not go through the router.
    """

    active_agent = (state or {}).get("active_agent")

    if not active_agent:
        return False

    return active_agent not in ("booking", "reschedule")


def _resolve_service_for_booking(state, service_text: str) -> Optional[dict]:
    """Resolve the patient's service text - a name ("فحص النظر") or a
    bare number picking from the service list they were just shown - to
    {"id", "name"}.

    Checks the remembered service list first (that IS the list they saw,
    so a positional pick can only mean one thing), then falls back to
    fuzzy-matching the name against it."""

    if not service_text:
        return None

    session = _get_booking_session(state.get("session_id"))
    last_list = session.get("last_list") or {}

    items = last_list.get("items") or [] if last_list.get("entity_type") == "service" else []

    if items:
        position = _extract_selection_number(service_text)
        if position is not None and 1 <= position <= len(items):
            chosen = items[position - 1]
            if chosen.get("id"):
                return {"id": chosen["id"], "name": chosen.get("name")}

        match = _fuzzy_match(service_text, items, ["name"])
        if match["result"] == "matched" and match["item"].get("id"):
            return {"id": match["item"]["id"], "name": match["item"].get("name")}

    # Nothing remembered (or no match in it) - look the service up in
    # the branch's real catalogue.
    base_url = _doctors_base_url(state)
    branch_id = session.get("branch_id")
    if not base_url:
        return None

    result = api.get_services(
        base_url, branch_ids=[branch_id] if branch_id else None,
        is_published=True, language=conversation_language(state),
    )
    if not result["success"]:
        logger.error(
            "_resolve_service_for_booking: get_services failed: status_code=%s error=%s",
            result.get("status_code"), result.get("error"),
        )
        return None

    language = conversation_language(state)
    candidates = [
        {"id": i.get("id"), "name": _preferred_name(i, language)}
        for i in (result["data"] or {}).get("items", [])
        if i.get("id") and _preferred_name(i, language)
    ]

    match = _fuzzy_match(service_text, candidates, ["name"])
    if match["result"] == "matched":
        return {"id": match["item"]["id"], "name": match["item"].get("name")}

    return None


def _resolve_specialty_for_booking(state, specialty_text: str) -> list:
    """Resolve the patient's specialty text - a name ("طب الأطفال") or a
    bare number picking from the specialty list they were just shown -
    to a list of {"id", "name"} (a list because a resolved specialty is
    still expanded to its siblings by `_expand_specialty_ids` later).

    Checks the remembered specialty list first (that IS the list they
    saw, so a positional pick can only mean one thing), then falls back
    to fuzzy-matching the name against it. Returns [] when nothing
    matches - the caller must not guess an id in that case.

    WHY THIS EXISTS: every OTHER list this project shows (doctors,
    branches, services, days, slots) is remembered via `_remember_list`
    so a bare "1" resolves by POSITION against the exact list the
    patient actually saw - specialties were the one list-producing tool
    (`list_specialties`) that never called `_remember_list`, leaving the
    model to recall the specialty id from memory/context instead of
    from a deterministic lookup. CONFIRMED REAL PRODUCTION FAILURE
    (medtown, 2026-08-30): the patient picked "1" for طب الأطفال
    (position 1 in the list just shown), and the doctor returned for
    that specialty was later shown with a schedule labelled "إستشارة
    الطبيب العام" (general physician consultation) - the same class of
    bug already fixed for days/branches/doctors elsewhere in this file,
    now closed the same way here.
    """

    if not specialty_text:
        return []

    session = _get_booking_session(state.get("session_id"))
    last_list = session.get("last_list") or {}

    items = last_list.get("items") or [] if last_list.get("entity_type") == "specialty" else []

    if not items:
        return []

    position = _extract_selection_number(specialty_text)
    if position is not None and 1 <= position <= len(items):
        chosen = items[position - 1]
        if chosen.get("id"):
            return [{"id": chosen["id"], "name": chosen.get("name")}]

    match = _fuzzy_match(specialty_text, items, ["name"])
    if match["result"] == "matched" and match["item"].get("id"):
        return [{"id": match["item"]["id"], "name": match["item"].get("name")}]

    return []


_SPECIALTY_STOPWORDS = {"طب", "جراحه", "امراض", "علاج", "قسم", "عام", "عامه", "استشارات"}


def _expand_specialty_ids(state, base_url: str, specialty_ids: list) -> list:
    """Add SIBLING specialties whose names share a real medical stem
    with the ones chosen.

    WHY: clinics routinely register the same field twice under slightly
    different names - "طب الباطنة" and "باطنه عام", "طب وجراحة العيون"
    and "جراحة الجسم الزجاجي والشبكية" - and the doctors are split
    across them. Searching only the id whose name matched the patient's
    wording most literally silently hides everyone registered under the
    other one.

    CONFIRMED REAL PRODUCTION FAILURE: an internal-medicine search
    returned only د. فارس الشارخ (طب الباطنة), while د. رانيا عبد
    الرحمن (باطنه عام) - shown correctly in an earlier turn of the same
    session - was missing. The patient had no way to know she existed.

    This is a DATA-DRIVEN expansion, not a guess: a sibling is added
    only when it shares a meaningful word with a chosen specialty, after
    dropping generic words ("طب", "عام", "جراحة"...) that would
    otherwise link unrelated fields. Never narrows, and returns the
    input unchanged on any failure."""

    if not specialty_ids:
        return specialty_ids

    try:
        result = api.get_specialties(base_url, language=conversation_language(state))
        if not result["success"]:
            return specialty_ids

        items = (result["data"] or {}).get("items", [])
        by_id = {i.get("id"): i for i in items if i.get("id")}

        def _tokens(item):
            words = set()
            for key in ("name", "altName"):
                value = item.get(key)
                if not value:
                    continue
                for word in _normalize_arabic(str(value)).split():
                    # Strip the definite article. Without this "الباطنه"
                    # and "باطنه" are different tokens, and the two
                    # registrations this whole helper exists to link -
                    # "طب الباطنة" and "باطنه عام" - never match.
                    if word.startswith("ال") and len(word) > 4:
                        word = word[2:]
                    if len(word) >= 4 and word not in _SPECIALTY_STOPWORDS:
                        words.add(word)
            return words

        chosen_tokens = set()
        for sid in specialty_ids:
            if sid in by_id:
                chosen_tokens |= _tokens(by_id[sid])

        if not chosen_tokens:
            return specialty_ids

        expanded = list(specialty_ids)
        added = []
        for sid, item in by_id.items():
            if sid in expanded:
                continue
            if _tokens(item) & chosen_tokens:
                expanded.append(sid)
                added.append(_preferred_name(item, conversation_language(state)))

        if added:
            logger.info(
                "_expand_specialty_ids: added sibling specialty/ies %s to %s",
                added, specialty_ids,
            )
        return expanded

    except Exception:
        logger.exception("_expand_specialty_ids: failed - using the original ids")
        return specialty_ids


@tool
def find_available_doctors(
    state: Annotated[AgentState, InjectedState],
    specialty_ids: list = None,
    days_ahead: int = DOCTOR_AVAILABILITY_WINDOW_DAYS,
    branch_name: str = "",
    allow_broader_search: bool = True,
    all_branches: bool = False,
    service_name: str = "",
    specialty_name: str = "",
) -> dict:
    """Find doctors who currently have a bookable service AND an available
    schedule slot within the next `days_ahead` days, across one or more
    specialties. ALWAYS call `list_specialties` first to get correct ids
    - never guess or invent one.

    `specialty_ids` IS OPTIONAL. Leave it out entirely when a SERVICE or
    a BRANCH is what the patient actually chose - you do NOT need to
    work out a specialty first, and you must not ask them for one just
    to satisfy this parameter. CONFIRMED REAL PRODUCTION FAILURE: with a
    service and a branch both already settled, the reply was "راح أحتاج
    أعرف التخصص المناسب الأول عشان أقدر أجيب لك الدكاترة المتاحين. تحب
    تبدأ بالتخصص ولا بالدكتور؟" - inventing a prerequisite that does not
    exist and restarting a flow that was two steps from done.

    `specialty_name`: PREFER THIS over hand-typing `specialty_ids`
    whenever the patient just answered a specialty list `list_specialties`
    showed them - pass their raw text here (a bare number like "1", or
    the name they typed). It is resolved against the EXACT list they
    were just shown (by position first, then by fuzzy name match) and
    turned into the correct id for you - you never need to recall or
    retype an id from memory. Only fall back to typing `specialty_ids`
    yourself when there is no specialty list in this conversation to
    resolve against. Leave both empty when neither applies. CONFIRMED
    REAL PRODUCTION FAILURE: a patient picked "1" from a freshly shown
    specialty list, and the doctor found for the id then used turned
    out to be scheduled under an unrelated, more generic service - the
    same class of bug this parameter's remembered-list resolution
    exists to prevent for every other list in this project (doctors,
    branches, services, days).

    `branch_name`: optional. Pass the user's raw branch text when they've
    said which branch they want (e.g. "الدقي", "فرع زايد") - the branch
    is resolved and CONFIRMED into the booking session automatically, and
    only doctors working at that branch are returned. Leave it empty when
    the user hasn't picked a branch (or said they don't mind). If the
    user doesn't know which branches exist, call
    `list_branches_for_specialty` instead of guessing.

    `service_name`: pass the SERVICE the patient chose (e.g. "فحص
    النظر"), or a bare number picking one from a service list you just
    showed. The service is resolved against the branch's real catalogue
    and its id is sent as `serviceIds` - so only doctors who actually
    provide THAT service come back. Use this whenever a service has been
    chosen: it answers "who can do this for me?" directly, and asking
    "specialty or doctor?" instead throws away a choice the patient has
    already made. CONFIRMED REAL PRODUCTION FAILURE: the patient picked
    "فحص النظر" and said yes to booking, and the reply was "تحب تبدأ
    بالتخصص ولا بالدكتور؟" followed by a specialty list - restarting the
    flow from scratch.

    `all_branches=True`: search the WHOLE hospital, ignoring any branch    settled earlier in the conversation. Pass this whenever the user asks
    to look more widely - "شوف في أي دكتور في المستشفى", "في فروع
    ثانية؟", "anywhere", "any branch" - and whenever you are answering a
    NEW question that has nothing to do with an earlier booking attempt.
    Without it, a branch chosen earlier keeps narrowing every later
    search.

    The returned list is remembered automatically, so the user can simply
    reply with its number ("3") and `match_entity_for_booking` will
    resolve it - you never need to repeat the names back as ids.

    IMPORTANT: pass ALL plausibly-matching specialty ids in ONE call as a
    list, not just the single most obvious one. Clinics often have both
    a general specialty and a more specific sub-specialty that could
    both reasonably cover the same complaint (e.g. "Ophthalmology" AND
    "Vitreoretinal Surgery" both relate to eye problems). If more than
    one specialty from `list_specialties` could plausibly match what the
    user described, include all of their ids here together - e.g.
    specialty_ids=["<ophthalmology-id>", "<vitreoretinal-surgery-id>"] -
    so a doctor registered under any of them is found. Do not conclude
    "no doctors available" after checking only one plausible specialty.

    NEVER call this tool ONCE PER SPECIALTY as a substitute for one
    combined call - e.g. calling it with ["<internal-medicine-id>"] and
    then immediately again with ["<gynaecology-id>"]. Confirmed real
    production failure: doing exactly that made the SECOND call's
    specialty silently become the only one this booking remembers going
    forward (later steps reuse the session's remembered specialties),
    so a doctor who was correctly found and shown under the FIRST
    specialty came back "couldn't find them in the system" minutes
    later when the patient tried to actually book with him - because
    the booking lookup was, by then, only searching the second,
    irrelevant specialty. One call, one list containing every relevant
    id, every time.

    `allow_broader_search`: pass False whenever the specialty was chosen
    to match a SYMPTOM the patient described. With True (the default)
    this tool falls back to every doctor in the clinic when the given
    specialties have nobody - useful while BOOKING, where the patient has
    already decided they want to be seen here and just needs someone
    available. It is actively wrong for medical guidance: a patient with
    abdominal pain offered a list of retina surgeons has been given a
    worse answer than "we don't have that specialty". Confirmed real
    production failure - pass False in the MEDICAL GUIDANCE flow, every
    time.

    Returns:
    {"status": "found", "doctors": [{"id", "name", "specialtyName", "degreeName"}, ...]}
    {"status": "found_broader_search", "doctors": [...]}  # the given specialty_ids had nobody available, but other doctors clinic-wide currently are. These are NOT a specialty match - never offer them as an answer to a symptom
    {"status": "not_found_in_specialty"}  # allow_broader_search=False and these specialties have nobody available. Say so plainly; do NOT substitute other doctors
    {"status": "not_found"}  # nobody at all currently has availability, even clinic-wide
    {"status": "branch_not_matched"}  # branch_name given but no branch matches it - show the branch list instead
    {"status": "not_found_in_branch", "branch": {...}}  # the branch is real, but has nobody in these specialties - offer other branches
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}  # the API call itself failed"""

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("find_available_doctors called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    session = _get_booking_session(state.get("session_id"))

    # `specialty_ids` is optional - a service or a branch is enough on
    # its own. Normalized here so every use below is safe.
    specialty_ids = specialty_ids or []

    # RESOLVE `specialty_name` (a bare number or raw text picking from
    # the specialty list just shown) INTO ITS REAL ID.
    #
    # This is the specialty equivalent of the `service_name`/`branch_name`
    # resolution just below - see `_resolve_specialty_for_booking` for
    # why this exists (specialties were the one remembered-list omission
    # in this project). Merged into any `specialty_ids` the model also
    # passed directly, rather than replacing them, so both paths can be
    # used together safely.
    if specialty_name and specialty_name.strip():
        resolved_specialties = _resolve_specialty_for_booking(state, specialty_name)
        if resolved_specialties:
            for item in resolved_specialties:
                if item.get("id") and item["id"] not in specialty_ids:
                    specialty_ids.append(item["id"])
            logger.info(
                "find_available_doctors: resolved specialty_name=%r -> %s",
                specialty_name, [i.get("id") for i in resolved_specialties],
            )
        else:
            logger.info(
                "find_available_doctors: specialty_name=%r did not match the remembered "
                "specialty list or any fuzzy candidate",
                specialty_name,
            )

    # Pull in sibling specialties registered under a near-identical name
    # ("طب الباطنة" / "باطنه عام"), so doctors filed under the other one
    # are not silently invisible. See _expand_specialty_ids.
    if specialty_ids:
        specialty_ids = _expand_specialty_ids(state, base_url, specialty_ids)

    # Remember which specialties this search used, so later steps
    # (list_branches_for_specialty, "who's soonest?") reuse exactly the
    # same set instead of the model having to re-derive them. Merged
    # rather than overwritten - see _remember_specialty_ids.
    _remember_specialty_ids(session, specialty_ids)

    branch_ids = None
    matched_branch = None

    if branch_name and branch_name.strip():
        matched_branch = _resolve_branch_by_name(base_url, branch_name, conversation_language(state), state=state)

        if matched_branch is None:
            logger.info("find_available_doctors: branch_name=%r did not match any branch", branch_name)
            return {"status": "branch_not_matched"}

        branch_ids = [matched_branch["id"]]
        session["branch_id"] = matched_branch["id"]
        session["branch_display_name"] = _arabic_preferred_name(matched_branch)
        logger.info("find_available_doctors: confirmed branch_id=%s (%s) from branch_name=%r", matched_branch["id"], session["branch_display_name"], branch_name)

    elif all_branches:
        # An explicit clinic-wide search. The remembered branch is also
        # CLEARED, not merely ignored for this one call: the patient has
        # just said the branch isn't the constraint, so leaving it in the
        # session would re-narrow the very next lookup.
        if session.get("branch_id"):
            logger.info(
                "find_available_doctors: all_branches=True - clearing remembered branch_id=%s",
                session.get("branch_id"),
            )
        session["branch_id"] = None
        session["branch_display_name"] = None

    elif session.get("branch_id"):
        # A branch was already confirmed earlier in this booking - keep
        # the doctor list consistent with it.
        #
        # ONLY when that branch belongs to the flow actually in progress.
        # CONFIRMED REAL PRODUCTION FAILURE: a patient picked a branch,
        # abandoned the booking ("لا مش عايزة كده خلاص"), then described
        # stomach pain. The medical search inherited that dead branch, so
        # a specialty the clinic genuinely staffs came back as "no
        # doctors available" - and when the patient then asked, in as
        # many words, to look across the whole hospital, the identical
        # branch-filtered query ran again and gave the identical wrong
        # answer. A branch chosen for a booking that is no longer
        # happening must not silently constrain a different question.
        if _booking_branch_is_stale(state, session):
            logger.info(
                "find_available_doctors: ignoring branch_id=%s - it belongs to an "
                "abandoned/unrelated flow, not the question being asked now",
                session.get("branch_id"),
            )
        else:
            branch_ids = [session["branch_id"]]

    now = datetime.utcnow()
    intersection_start = now.isoformat() + "Z"
    intersection_end = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    # RESOLVE A CHOSEN SERVICE -> its id, for Doctors/GetList serviceIds.
    service_ids = None
    if service_name and service_name.strip():
        resolved_service = _resolve_service_for_booking(state, service_name)
        if resolved_service:
            service_ids = [resolved_service["id"]]
            session["service_id"] = resolved_service["id"]
            session["service_display_name"] = resolved_service.get("name")
            logger.info(
                "find_available_doctors: resolved service %r -> id=%s (%s)",
                service_name, resolved_service["id"], resolved_service.get("name"),
            )
        else:
            logger.info("find_available_doctors: service_name=%r did not match any service", service_name)
            return {"status": "service_not_matched"}
    elif session.get("service_id"):
        # A service chosen earlier in this booking keeps narrowing the
        # doctor list, the same way a confirmed branch does.
        service_ids = [session["service_id"]]

    result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids,
        branch_ids=branch_ids,
        service_ids=service_ids,
        has_published_service=True,
        has_service_schedule=True,
        intersection_start=intersection_start,
        intersection_end=intersection_end,
     language=conversation_language(state),)

    if not result["success"]:
        logger.error(
            "find_available_doctors API call failed: base_url=%s specialty_ids=%s branch_ids=%s status_code=%s error=%s",
            base_url, specialty_ids, branch_ids, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    # The server already filtered by hasPublishedService +
    # hasServiceSchedule + the intersection window, so treat hasSlots as
    # a refinement only: exclude a doctor ONLY when the API explicitly
    # says hasSlots is False. Previously this required hasSlots to be
    # truthy, which silently discarded EVERY doctor whenever the field
    # was absent or null in the response - indistinguishable from
    # "nobody is available".
    available = [i for i in items if i.get("hasSlots") is not False]

    # RESCUE DOCTORS THE hasSlots FLAG WRONGLY EXCLUDED.
    #
    # `hasSlots` is already trusted leniently (only excluded when
    # explicitly False - see the comment above), but that still means a
    # doctor with a stale/incorrect False flag never even reaches this
    # list, no matter how the patient asks. CONFIRMED REAL PRODUCTION
    # CLASS OF BUG: د. وائل عويس had a live, in-effect rota and genuinely
    # open slots at a branch, yet a roster built from this same
    # hasSlots-filtered field silently excluded him (see
    # `_doctors_with_real_slots`'s docstring) - fixed there by
    # cross-checking the real schedule-slots endpoint per doctor rather
    # than trusting the flag. This applies that identical cross-check
    # here, for every doctor this call is about to tell the patient
    # "isn't available" - scoped to the same branch(es) this search
    # already used, or clinic-wide when no branch was chosen - so a
    # doctor who is actually bookable is never hidden by a bad flag.
    excluded_ids = [i.get("id") for i in items if i.get("hasSlots") is False and i.get("id")]
    if excluded_ids:
        rescue_branch_id = branch_ids[0] if branch_ids else None
        verified_ids = _doctors_with_real_slots(state, base_url, excluded_ids, rescue_branch_id)
        if verified_ids:
            rescued = [i for i in items if i.get("id") in verified_ids]
            if rescued:
                logger.info(
                    "find_available_doctors: rescued %d doctor(s) whose hasSlots=False did not "
                    "match real open slots: %s",
                    len(rescued), [{"id": i.get("id"), "name": i.get("name")} for i in rescued],
                )
                already_in = {i.get("id") for i in available}
                available = available + [i for i in rescued if i.get("id") not in already_in]

    logger.info(
        "find_available_doctors: specialty_ids=%s api_returned=%d after_hasSlots_filter=%d",
        specialty_ids, len(items), len(available),
    )

    if not available and branch_ids:
        # The user explicitly named a branch. Broadening clinic-wide here
        # would silently hand back doctors at OTHER branches as if they
        # answered the question that was asked - say plainly that this
        # branch has nobody instead, and let the reply offer the others.
        logger.info("find_available_doctors: no doctors in specialty_ids=%s at branch_id=%s", specialty_ids, branch_ids)
        return {
            "status": "not_found_in_branch",
            "branch": {"id": (matched_branch or {}).get("id") or branch_ids[0],
                       "name": session.get("branch_display_name")},
        }

    if not available:
        if not allow_broader_search:
            # Symptom-driven search. Broadening here would answer
            # "which doctor suits my abdominal pain?" with a list of
            # retina surgeons - a confidently wrong answer, which is
            # worse for the patient than an honest "we don't have
            # anyone for that". Confirmed real production failure.
            logger.info(
                "find_available_doctors: no doctors in specialty_ids=%s and "
                "allow_broader_search=False - not substituting other specialties",
                specialty_ids,
            )
            return {"status": "not_found_in_specialty"}

        # Safety net: the given specialty_ids found nobody, but a related
        # specialty under a different name might still have doctors -
        # confirmed real, repeated production bug where the model only
        # passed one of several plausibly-relevant specialty ids (e.g.
        # a general "Ophthalmology" with zero registered doctors, never
        # also passing its "Vitreoretinal Surgery" sub-specialty where
        # doctors actually are). Rather than relying solely on the model
        # getting this right, automatically broaden to ALL currently
        # available doctors clinic-wide before giving up - flagged
        # distinctly so the reply can be honest that this is a broader
        # result, not an exact specialty match.
        broader_result = api.get_doctors(
            base_url,
            specialty_ids=None,
            has_published_service=True,
            has_service_schedule=True,
            intersection_start=intersection_start,
            intersection_end=intersection_end,
         language=conversation_language(state),)
        if broader_result["success"]:
            broader_items = (broader_result["data"] or {}).get("items", [])
            broader_available = [i for i in broader_items if i.get("hasSlots") is not False]

            # Same rescue as the primary search above - a stale hasSlots
            # flag must not hide a genuinely bookable doctor here either.
            broader_excluded_ids = [i.get("id") for i in broader_items if i.get("hasSlots") is False and i.get("id")]
            if broader_excluded_ids:
                broader_verified_ids = _doctors_with_real_slots(state, base_url, broader_excluded_ids, None)
                if broader_verified_ids:
                    broader_rescued = [i for i in broader_items if i.get("id") in broader_verified_ids]
                    if broader_rescued:
                        logger.info(
                            "find_available_doctors (broader search): rescued %d doctor(s) whose "
                            "hasSlots=False did not match real open slots: %s",
                            len(broader_rescued), [{"id": i.get("id"), "name": i.get("name")} for i in broader_rescued],
                        )
                        already_in_broader = {i.get("id") for i in broader_available}
                        broader_available = broader_available + [i for i in broader_rescued if i.get("id") not in already_in_broader]

            logger.info(
                "find_available_doctors: narrow search found 0, broadened to all specialties: api_returned=%d after_hasSlots_filter=%d",
                len(broader_items), len(broader_available),
            )
            if broader_available:
                doctors = _shape_doctor_list(broader_available, conversation_language(state))
                if doctors:
                    _remember_list(state, "doctor", doctors)
                    return {"status": "found_broader_search", "doctors": doctors}

        return {"status": "not_found"}

    doctors = _shape_doctor_list(available, conversation_language(state))

    if not doctors:
        return {"status": "not_found"}

    # CRITICAL: record the exact list, in the exact order, that the model
    # is about to show. Without this a reply of "6" cannot be resolved -
    # the original production failure.
    _remember_list(state, "doctor", doctors)

    return {"status": "found", "doctors": doctors}


# ==========================================================
# Reschedule Appointment (change an existing booking's time)
# ==========================================================
#
# Reuses lookup_appointment/compare_phone/send_otp/verify_otp exactly
# as-is for identifying the booking and verifying identity - see
# prompts.py's RESCHEDULE FLOW, which mirrors the same STEP 1-3 logic
# already used for cancellation. These three tools cover what's new:
# checking the doctor's schedule/availability and performing the update.

def _resolve_doctor_id(state: AgentState, ref_number: str, language: Optional[str]) -> dict:
    """Internal helper: look up a booking by its reference number and
    return its doctorId, so schedule/slot tools know which doctor to
    query without the LLM ever having to know or pass a doctor's GUID
    directly. Returns {"status": "found", "doctor_id": ...} or an error
    status matching lookup_appointment's own conventions."""

    base_url = _base_url(state)
    result = api.get_bookings_by_ref(base_url, ref_number, language=language)

    if not result["success"]:
        logger.error("_resolve_doctor_id: API call failed for ref_number=%s error=%s", ref_number, result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    doctor_id = items[0].get("doctorId")
    if not doctor_id:
        logger.warning("_resolve_doctor_id: booking found but has no doctorId - ref_number=%s", ref_number)
        return {"status": "error"}

    return {"status": "found", "doctor_id": doctor_id}


# WEEKDAY VOCABULARY - deliberately much wider than "correct" Arabic.
#
# WHY: this map is the ONLY thing standing between "the patient named a
# day" and "the day was silently thrown away". Every spelling that
# fails to resolve here makes `resolve_available_day` return an error,
# and the model then falls back to showing the soonest date instead -
# i.e. it ignores the day the patient actually asked for.
#
# CONFIRMED REAL GAP: "عاوزه احجز معاد مع دكتور احمد العقيل يوم التلات"
# - "التلات" is how Tuesday is written in everyday Egyptian, and it was
# not in this map at all, so the request resolved to nothing. The same
# was true of "الاتنين", "الاربع", "الحد", "الجمعه" and every other
# form people actually type. Formal MSA spellings are the exception in
# a WhatsApp message, not the rule.
#
# Keys are matched against `_fold_weekday_token`'s output, so hamza
# forms (أ/إ/آ -> ا), ta-marbuta (ة -> ه), tatweel, diacritics and the
# leading "ال" are already normalised away by the time a lookup runs -
# every key below is written in that same folded form.
_WEEKDAY_NAMES = {
    # English + common short forms (case-insensitive)
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,

    # Arabic - MSA, Egyptian and Gulf colloquial, all in folded form.
    "اثنين": 0, "الاثنين": 0, "تنين": 0, "التنين": 0, "اتنين": 0, "الاتنين": 0,
    "ثلاثاء": 1, "الثلاثاء": 1, "تلاتاء": 1, "التلاتاء": 1,
    "ثلاث": 1, "الثلاث": 1, "تلات": 1, "التلات": 1, "ثلثاء": 1, "الثلثاء": 1,
    "اربعاء": 2, "الاربعاء": 2, "اربع": 2, "الاربع": 2, "روبع": 2, "الروبع": 2,
    "خميس": 3, "الخميس": 3,
    "جمعه": 4, "الجمعه": 4, "جمعة": 4,
    "سبت": 5, "السبت": 5,
    "احد": 6, "الاحد": 6, "حد": 6, "الحد": 6,

    # Arabizi / franco-arabe, as typed on a Latin keyboard.
    "eltalat": 1, "talat": 1, "eltalata": 1, "talata": 1,
    "eltnen": 0, "etnen": 0, "itnin": 0, "elitnen": 0,
    "elarbaa": 2, "arbaa": 2, "arbe3": 2, "elarbe3": 2,
    "elkhamis": 3, "khamis": 3, "el5amis": 3, "5amis": 3,
    "elgomaa": 4, "gomaa": 4, "goma3a": 4, "elgom3a": 4, "gom3a": 4,
    "elsabt": 5, "sabt": 5, "essabt": 5,
    "elhad": 6, "had": 6, "elahad": 6, "ahad": 6,
}

# Arabic letters that different keyboards/habits render differently.
# Folding them means one map entry covers every variant instead of the
# map needing a row per spelling.
_WEEKDAY_FOLD_MAP = {
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ة": "ه",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
}

_WEEKDAY_DIACRITICS_RE = re.compile(r"[ً-ْـ]")

# Punctuation, quotes and whitespace clinging to either end of a word -
# stripped so "(التلات)" and "التلات،" fold to the same key.
_WEEKDAY_EDGE_PUNCT_RE = re.compile(r"^[\s\W_]+|[\s\W_]+$", re.UNICODE)


def _fold_weekday_token(text: Optional[str]) -> str:
    """Normalise one word so every spelling of a weekday lands on the
    same `_WEEKDAY_NAMES` key: strip diacritics/tatweel, unify hamza and
    ta-marbuta, drop surrounding punctuation, lowercase Latin text.

    The leading "ال" is NOT stripped here - both the bare and the
    prefixed forms are listed explicitly in the map instead, because
    blind prefix-stripping would turn unrelated words ("الحجز") into
    near-misses for real day names."""

    if not text:
        return ""

    folded = _WEEKDAY_DIACRITICS_RE.sub("", str(text).strip())
    folded = "".join(_WEEKDAY_FOLD_MAP.get(ch, ch) for ch in folded)
    folded = _WEEKDAY_EDGE_PUNCT_RE.sub('', folded)
    return folded.lower()


# Day names that are ALSO ordinary Arabic words. Read out of a longer
# sentence they produce real false positives - "ايه الحد الأقصى للحجز؟"
# is a question about a limit, not a request for Sunday, and
# "الفروع الثلاث" is a count, not Tuesday. Inside a longer text these
# only count as a day when something marks them as one.
_AMBIGUOUS_WEEKDAY_KEYS = frozenset({
    "حد", "الحد", "ثلاث", "الثلاث", "اربع", "الاربع",
    "تنين", "التنين", "سبت", "جمعه", "احد",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun", "thur",
})

# The words that mark the next token as a day of the week.
_WEEKDAY_CUE_WORDS = frozenset({
    "يوم", "اليوم", "ايام", "الايام", "يومي", "بيوم", "ليوم",
    "day", "on", "this", "next", "coming",
})


def resolve_weekday_index(weekday_text: Optional[str]) -> Optional[int]:
    """Weekday index (Monday=0 .. Sunday=6) for ANY spelling a patient
    or the model might use, or None when the text names no weekday.

    Accepts a bare day name ("التلات"), a short phrase around one
    ("يوم التلات", "on tuesday"), or a whole sentence - the words are
    folded one at a time and the first that resolves wins. Word-level
    matching (rather than substring) is what stops "الجمعية" from being
    read as Friday.

    A token in `_AMBIGUOUS_WEEKDAY_KEYS` only counts when the text is
    JUST that word (someone answering "الحد" to "which day?" means
    Sunday and nothing else) or when a cue word marks it as a day
    ("يوم الحد"). Anywhere else in a sentence it is ignored, because the
    cost of reading "الحد الأقصى" as Sunday - a booking steered onto a
    day nobody asked for - is far higher than the cost of missing one
    unusual phrasing, which merely falls back to the normal flow.
    """

    if not weekday_text:
        return None

    whole = _fold_weekday_token(weekday_text)
    if whole in _WEEKDAY_NAMES:
        # The entire message IS the day name - no ambiguity to resolve.
        return _WEEKDAY_NAMES[whole]

    words = [w for w in re.split(r"[\s/،,]+", str(weekday_text)) if w]

    for position, word in enumerate(words):
        folded = _fold_weekday_token(word)
        index = _WEEKDAY_NAMES.get(folded)
        if index is None:
            continue
        if folded not in _AMBIGUOUS_WEEKDAY_KEYS:
            return index
        previous = _fold_weekday_token(words[position - 1]) if position else ""
        if previous in _WEEKDAY_CUE_WORDS:
            return index

    return None


@tool
def get_next_weekday_date(
    weekday_name: str,
    after_date: str = "",
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict:
    """Resolve a weekday NAME (e.g. "Thursday"/"الخميس") to an actual
    calendar date, computed exactly - NEVER work out which calendar date
    a weekday name falls on yourself, your own mental date arithmetic is
    not reliable enough for this and has caused real incorrect answers
    before (e.g. calling a date "Thursday" that was not actually a
    Thursday). ALWAYS call this tool instead, every time a user names a
    day of the week rather than a specific date.

    Two modes:
    - `after_date` OMITTED (empty): returns the NEXT upcoming date for
      that weekday counting from TODAY. If today itself already IS that
      weekday, returns TODAY's date.
    - `after_date` GIVEN (format "YYYY-MM-DD", e.g. from an earlier call
      to this same tool, or from get_available_reschedule_slots): returns
      the next occurrence of that weekday STRICTLY AFTER that date - use
      this whenever the user refers to a day relative to one you already
      discussed (e.g. "the following Monday" / "الاثنين اللي بعده" after
      you'd already established a specific Monday's date) - do NOT ask
      them to clarify what date they mean, just call this directly.
    Returns:
    {"status": "found", "date": "YYYY-MM-DD", "weekday_name": "Thursday"}
    {"status": "error"}  # unrecognized weekday name or bad after_date"""

    target_weekday = resolve_weekday_index(weekday_name)

    if target_weekday is None:
        logger.warning("get_next_weekday_date: unrecognized weekday_name=%r", weekday_name)
        return {"status": "error"}

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    if after_date:
        try:
            reference = date.fromisoformat(after_date.strip())
        except ValueError:
            logger.warning("get_next_weekday_date: invalid after_date=%r", after_date)
            return {"status": "error"}
        days_ahead = (target_weekday - reference.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # strictly AFTER the reference date, never the same day
    else:
        reference = datetime.now(tz).date()
        days_ahead = (target_weekday - reference.weekday()) % 7

    target_date = reference + timedelta(days=days_ahead)
    english_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_weekday]

    return {"status": "found", "date": target_date.isoformat(), "weekday_name": english_name}


@tool
def get_doctor_schedule(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
    target_date: str = "",
    language: str = "en",
) -> dict:
    """Get the GENERAL RECURRING weekly schedule of the doctor on a given
    booking - which weekdays they work, their daily start/end times, and
    the date range this schedule is valid for. Call this BEFORE offering
    to reschedule, to know which days of the week are even worth
    checking - this does NOT return specific open time slots (use
    `get_available_reschedule_slots` for that once you've picked a
    target date).

    `target_date` (format "YYYY-MM-DD"), if you already have one in mind
    (e.g. from `get_next_weekday_date`), filters to only the schedule
    row(s) actually valid/effective on that specific date - avoiding
    stale/expired or not-yet-started schedule rows for the same doctor.
    If omitted, defaults to today.
    Returns:
    {"status": "found", "schedules": [{"recurringDaysNames": [...], "fromDateTime": ..., "toDateTime": ...}, ...]}
    {"status": "not_found"}  # booking or schedule doesn't exist
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}"""

    resolved = _resolve_doctor_id(state, ref_number, language)
    if resolved["status"] != "found":
        return resolved

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("get_doctor_schedule called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    if target_date:
        effective_date = target_date
    else:
        try:
            effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            effective_date = None

    result = api.get_doctor_schedule(
        base_url, doctor_ids=[resolved["doctor_id"]], effective_date=effective_date,
        include_future=not target_date,
        language=conversation_language(state),
    )

    if not result["success"]:
        logger.error("get_doctor_schedule API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    schedules = [
        {
            "recurringDaysNames": item.get("recurringDaysNames"),
            "fromDateTime": to_local_wallclock(item.get("fromDateTime"), timezone_name),
            "toDateTime": to_local_wallclock(item.get("toDateTime"), timezone_name),
            "branchName": item.get("branchName"),
            "doctorName": item.get("doctorName"),
        }
        for item in items
    ]

    return {"status": "found", "schedules": schedules}


@tool
def get_available_reschedule_slots(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
    from_date: str,
    to_date: str,
    language: str = "en",
) -> dict:
    """Get the doctor's ACTUAL open time slots (not just working days)
    for the booking's doctor, within [from_date, to_date] - both in ISO
    format, e.g. "2026-05-01T09:00:00+03:00". Only genuinely available
    (not already booked) slots are returned. Call `get_doctor_schedule`
    first to know which weekdays/hours are worth checking, then call
    this with a specific day's full working-hours range to see the
    exact bookable times. Returns:
    {"status": "found", "slots": [{"slotStart": ..., "slotEnd": ..., "date_display": ..., "time_display": ..., "doctorName": ..., "serviceName": ...}, ...]}
    {"status": "not_found"}  # no open slots in this range
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}"""

    resolved = _resolve_doctor_id(state, ref_number, language)
    if resolved["status"] != "found":
        return resolved

    # Safety net: if the range came in backwards (from_date after
    # to_date), swap them. Confirmed directly in production: the LLM
    # passed from_date=09:00 and to_date=07:00 (inverted) - the real API
    # appears to silently ignore date filtering entirely when given a
    # nonsensical inverted range, returning generic/unfiltered slots
    # instead (which is what caused already-passed times to still
    # appear). Guaranteeing a valid, forward-ordered range here removes
    # dependence on the LLM getting the order right.
    try:
        if from_date and to_date and datetime.fromisoformat(from_date) > datetime.fromisoformat(to_date):
            logger.warning(
                "get_available_reschedule_slots: from_date=%r was AFTER to_date=%r - swapping them",
                from_date, to_date,
            )
            from_date, to_date = to_date, from_date
    except ValueError:
        pass  # let the API itself reject a genuinely malformed date string

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("get_available_reschedule_slots called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[resolved["doctor_id"]],
        from_date=from_date, to_date=to_date, is_booked=False,
     language=conversation_language(state),)

    if not result["success"]:
        logger.error("get_available_reschedule_slots API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    # Defense in depth: exclude any item explicitly marked isBooked=True,
    # even though is_booked=False was already sent as a request filter -
    # other endpoints in this same system have been observed to not
    # always respect their own request filters (e.g. the inverted
    # from_date/to_date range issue), so don't rely on the request filter
    # alone for something this important (double-booking a doctor).
    items = [i for i in items if i.get("isBooked") is not True]
    if not items:
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    language = conversation_language(state)
    slots = []
    for item in items:
        slot_start = to_local_wallclock(item.get("slotStart"), timezone_name)
        slot_end = to_local_wallclock(item.get("slotEnd"), timezone_name)
        slots.append({
            "slotStart": slot_start,
            "slotEnd": slot_end,
            "date_display": _display_date(slot_start),
            "weekday_display": _display_weekday(slot_start, language),
            "time_display": _display_time_12h(slot_start, language),
            "doctorName": item.get("doctorName"),
            "serviceName": _service_name(item, language),
            # servicePrice is deliberately NOT returned: fees are private
            # by default and must only ever be revealed through
            # `get_doctor_fees` when the user explicitly asks. Confirmed
            # real issue - a price the model could see in the slot list
            # ended up printed in the availability message unprompted.
        })

    # Exclude slots that have already passed - a slot for TODAY earlier
    # than right now must never still be offered (observed directly:
    # 9:00 AM was still shown while the conversation was happening at
    # ~5pm the same day). Compared in the client's own local timezone,
    # matching how slotStart itself was already converted.
    try:
        now_local = _local_now_naive(timezone_name)
        slots = [
            s for s in slots
            if s["slotStart"] and datetime.fromisoformat(s["slotStart"]) > now_local
        ]
    except Exception:
        logger.exception("get_available_reschedule_slots: failed to filter past slots, showing all")

    if not slots:
        return {"status": "not_found"}

    # Always chronological - the API's own return order was observed to
    # be scrambled in production (slots came back neither ascending nor
    # descending), and relying on the LLM to re-sort dozens of items
    # correctly by eye is not realistic. Sort here, once, in code.
    slots.sort(key=lambda s: s["slotStart"] or "")

    # Deduplicate by exact start time - confirmed directly in production:
    # the API returned every distinct time TWICE in a row (e.g. "11:00
    # AM, 11:00 AM, 12:00 PM, 12:00 PM, ..."), likely once per underlying
    # resource/service sharing the same schedule slot. The user must
    # never see the same bookable time offered more than once.
    seen_starts = set()
    deduped = []
    for s in slots:
        key = s["slotStart"]
        if key in seen_starts:
            continue
        seen_starts.add(key)
        deduped.append(s)
    if len(deduped) != len(slots):
        logger.warning("get_available_reschedule_slots: removed %d duplicate slot(s) with the same start time", len(slots) - len(deduped))
    slots = deduped

    # Cap to a reasonable, actually-usable count for a chat interface.
    # Observed in production: a too-wide [from_date, to_date] query
    # returned 44 slots spanning nearly 24 hours - regardless of why
    # that range was too wide, showing dozens of options in a chat
    # message is not usable. This guarantees a sane result independent
    # of whether the date-range scoping prompt guidance is followed.
    MAX_SLOTS_TO_SHOW = 20
    if len(slots) > MAX_SLOTS_TO_SHOW:
        logger.warning(
            "get_available_reschedule_slots: %d slots returned for range [%s, %s] - "
            "capping to the first %d chronologically (this usually means the "
            "queried date range was wider than a single day's actual working hours)",
            len(slots), from_date, to_date, MAX_SLOTS_TO_SHOW,
        )
        slots = slots[:MAX_SLOTS_TO_SHOW]

    return {"status": "found", "slots": slots}


@tool
def reschedule_appointment(
    state: Annotated[AgentState, InjectedState],
    booking_id: str,
    new_time_from: str,
    new_time_to: str,
) -> dict:
    """Change an existing booking to a new time. `booking_id` MUST be the
    booking's own "id" field (a GUID) from a FRESH `lookup_appointment`
    or `check_booking_status` call in THIS conversation - never invent
    or reuse an old value from memory. `new_time_from`/`new_time_to` must
    be the EXACT slotStart/slotEnd values from `get_available_reschedule_slots`
    - never modify or recompute them yourself. Returns:
    {"status": "success"} or {"status": "error"}"""

    # NOTE: confirmed directly from the user's own curl - GuestBookings/Update
    # lives on the SAME port as Doctors/Specialties (1302), NOT the regular
    # GuestBookings port used for cancellation (1101), despite the "GuestBookings"
    # name. Trusting the confirmed URL over the path-name convention.
    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("reschedule_appointment called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "error"}

    result = api.reschedule_booking(base_url, booking_id, new_time_from, new_time_to)

    if not result["success"]:
        logger.error(
            "reschedule_appointment API call failed: booking_id=%s status_code=%s error=%s",
            booking_id, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    language = conversation_language(state)
    timezone_name = (state.get("templates") or {}).get("_timezone")
    # `new_time_from` is the slotStart the patient picked, already
    # wall-clock - converting it here would confirm a time three hours
    # later than the one they chose.
    local_new_from = to_local_wallclock(new_time_from, timezone_name)

    # THE NEW TIME, READY TO DISPLAY.
    #
    # This used to return {"status": "success"} and nothing else, while
    # the clinic's own success message asks for BOTH the new appointment
    # and the old one it replaced. With no source for either, its
    # placeholders were filled from whatever the model remembered - and
    # confirmed in production, {old_date} and {old_time} reached a
    # patient as literal text.
    #
    # The OLD values are deliberately NOT returned here: they come from
    # the `lookup_appointment` record this flow is required to have
    # fetched first, which is the authoritative copy of what was
    # actually booked. See graph._build_terminal_success_directive.
    return {
        "status": "success",
        "new_date_display": _display_date(local_new_from),
        "new_time_display": _display_time_12h(local_new_from, language),
        "new_weekday_display": _display_weekday(local_new_from, language),
    }


# ==========================================================
# General Hospital FAQ (RAG over a per-client knowledge base document)
# ==========================================================
#
# Fully generic - reads whichever file client_config.csv's
# knowledge_base_file column points to for THIS client_id (see rag.py).
# Adding a new clinic's FAQ knowledge base is just adding a text file and
# setting that column - no code changes needed.

@tool
def list_hospital_services(state: Annotated[AgentState, InjectedState]) -> dict:
    """List the services this clinic offers, read straight from its own
    knowledge base, complete and in the clinic's own wording/order.

    CALL THIS - not `answer_hospital_faq` - whenever the user asks what
    services the clinic provides ("إيه الخدمات اللي عندكم؟", "وش
    الخدمات؟", "what services do you offer?"). Present exactly the
    services it returns, all of them, as a numbered list.

    Do NOT use `answer_hospital_faq` for this question: it returns the
    passages most SIMILAR to the question, which are detail paragraphs
    from inside one or two services. Confirmed real failure - answering
    that way produced a list mixing inpatient amenities (gardens, gym,
    art therapy area) with services, while four of the six actual
    services were missing entirely.

    Use `answer_hospital_faq` afterwards, when the user asks about ONE
    specific service in detail.

    NOT FOR A SINGLE BRANCH'S SERVICES. This reads the knowledge-base
    file, which describes the hospital's service lines as a whole and
    carries NO per-branch information - so it returns the identical list
    no matter which branch was asked about. For "خدمات فرع كذا" / "what
    services does this branch have?", call `list_branch_services`, which
    reads the real service catalogue filtered to that branch.

    Returns:
    {"status": "found", "services": ["...", ...]}
    {"status": "not_found"}  # no services section in this clinic's knowledge base
    {"status": "not_configured"}  # this clinic has no knowledge base set up yet
    """

    kb_file = (state.get("templates") or {}).get("_knowledge_base_file", "")

    if not kb_file:
        logger.warning(
            "list_hospital_services called but no knowledge_base_file is configured for client_id=%s",
            state.get("client_id"),
        )
        return {"status": "not_configured"}

    services = rag.list_services(kb_file, conversation_language(state))

    if not services:
        return {"status": "not_found"}

    return {"status": "found", "services": services}


@tool
def list_branch_services(
    state: Annotated[AgentState, InjectedState],
    branch_name: str = "",
) -> dict:
    """List the services a SPECIFIC BRANCH provides, read from the
    clinic's real service catalogue (the Services endpoint), filtered to
    that branch and to published services only.

    CALL THIS - not `list_hospital_services`, and not
    `answer_hospital_faq` - whenever the question is about ONE BRANCH's
    services ("خدمات فرع المعادي", "إيه الخدمات في الفرع ده؟", "what
    services does this branch have?"). Those other two answer from the
    knowledge base file, which describes the hospital's service lines as
    a whole and carries NO per-branch information at all - so using them
    here returns the same six generic service lines for every branch,
    which is not an answer to the question that was asked. CONFIRMED
    REAL PRODUCTION FAILURE: asked for فرع المعادي's services, the reply
    listed the hospital-wide knowledge-base list verbatim.

    `branch_name`: optional. Pass the patient's raw branch text when
    they named one. Leave it empty to use the branch already confirmed
    in this booking session, or the branch most recently shown to them.

    Returns:
    {"status": "found", "branch": {"id", "name"}, "services": [{"name", "description"}, ...]}
    {"status": "not_found", "branch": {...}}  # this branch publishes no services
    {"status": "branch_not_matched"}  # branch_name given but nothing matched
    {"status": "missing_branch"}  # no branch named and none remembered - ask which branch
    {"status": "not_configured"} / {"status": "error"}"""

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning(
            "list_branch_services called but no doctors_base_url is configured for client_id=%s",
            state.get("client_id"),
        )
        return {"status": "not_configured"}

    session = _get_booking_session(state.get("session_id"))

    branch_id = None
    branch_display = None

    if branch_name and branch_name.strip():
        matched = _resolve_branch_by_name(base_url, branch_name, conversation_language(state), state=state)
        if matched is None:
            logger.info("list_branch_services: branch_name=%r did not match any branch", branch_name)
            return {"status": "branch_not_matched"}
        branch_id = matched.get("id")
        branch_display = _arabic_preferred_name(matched)
    else:
        branch_id = session.get("branch_id")
        branch_display = session.get("branch_display_name")

        if not branch_id:
            # Fall back to the branch most recently SHOWN to them - the
            # patient routinely asks "and its services?" right after
            # picking one from a list, with no name repeated.
            last_list = session.get("last_list") or {}
            if last_list.get("entity_type") == "branch":
                items = last_list.get("items") or []
                if len(items) == 1:
                    branch_id = items[0].get("id")
                    branch_display = _arabic_preferred_name(items[0]) or items[0].get("name")

    if not branch_id:
        return {"status": "missing_branch"}

    result = api.get_services(
        base_url, branch_ids=[branch_id], is_published=True,
        language=conversation_language(state),
    )

    if not result["success"]:
        logger.error(
            "list_branch_services: get_services failed for branch_id=%s: status_code=%s error=%s",
            branch_id, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    language = conversation_language(state)
    services = []
    seen = set()

    for item in (result["data"] or {}).get("items", []):
        # The Services endpoint returns name/altName directly (unlike
        # slot rows, which carry serviceName/serviceAltName), so
        # `_preferred_name` is the right helper here, not `_service_name`.
        name = _preferred_name(item, language)
        if not name or name in seen:
            continue
        seen.add(name)
        description = (
            item.get("altDescription") if language != "en" else item.get("description")
        ) or item.get("description") or item.get("altDescription")
        # `id` IS REQUIRED. Picking a service is a real booking step:
        # the id gets passed to `find_available_doctors(service_name=...)`
        # -> Doctors/GetList `serviceIds`, which is what makes "who does
        # فحص النظر?" answerable at all.
        services.append({"id": item.get("id"), "name": name, "description": description})

    branch_info = {"id": branch_id, "name": branch_display}

    logger.info(
        "list_branch_services: branch_id=%s (%s) -> %d published service(s)",
        branch_id, branch_display, len(services),
    )

    if not services:
        return {"status": "not_found", "branch": branch_info}

    # Remembered so a bare "1" picks a SERVICE by position, and so the
    # chosen service's id can be recovered on a later turn.
    _remember_list(state, "service", services)

    return {"status": "found", "branch": branch_info, "services": services}


@tool
def find_branches_offering_service(
    state: Annotated[AgentState, InjectedState],
    service_name: str = "",
    days_ahead: int = DOCTOR_AVAILABILITY_WINDOW_DAYS,
) -> dict:
    """Which OTHER branches can actually book a given service right now,
    and how many doctors provide it at each.

    Use this when the patient wants a service at a branch that has no
    bookable doctor: instead of a dead end ("this branch has nobody"),
    it answers the question they actually have - where CAN I get this?

    `service_name`: the service's name, or a bare number picking one
    from a service list just shown. Falls back to the service already
    chosen in this booking session.

    Returns:
    {"status": "found", "service": {"id", "name"},
     "branches": [{"id", "name", "doctorCount"}, ...]}
    {"status": "not_found", "service": {...}}  # nobody offers it anywhere
    {"status": "service_not_matched"} / {"status": "not_configured"} / {"status": "error"}

    Every branch listed is one the tools VERIFIED has a bookable doctor
    for this service inside the availability window - never infer a
    branch offers something because its name or address suggests it."""

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning(
            "find_branches_offering_service called but no doctors_base_url is configured for client_id=%s",
            state.get("client_id"),
        )
        return {"status": "not_configured"}

    session = _get_booking_session(state.get("session_id"))

    resolved_service = None
    if service_name and service_name.strip():
        resolved_service = _resolve_service_for_booking(state, service_name)
    elif session.get("service_id"):
        resolved_service = {
            "id": session["service_id"],
            "name": session.get("service_display_name"),
        }

    if not resolved_service:
        logger.info(
            "find_branches_offering_service: service_name=%r did not match any service",
            service_name,
        )
        return {"status": "service_not_matched"}

    now = datetime.utcnow()
    result = api.get_doctors(
        base_url,
        service_ids=[resolved_service["id"]],
        has_published_service=True,
        has_service_schedule=True,
        intersection_start=now.isoformat() + "Z",
        intersection_end=(now + timedelta(days=days_ahead)).isoformat() + "Z",
        page_size=200,
        language=conversation_language(state),
    )

    if not result["success"]:
        logger.error(
            "find_branches_offering_service: get_doctors failed: status_code=%s error=%s",
            result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    doctor_items = [
        i for i in (result["data"] or {}).get("items", [])
        if i.get("hasSlots") is not False and i.get("id")
    ]

    if not doctor_items:
        return {"status": "not_found", "service": resolved_service}

    # The Doctors endpoint doesn't reliably say WHICH branch each doctor
    # sits at, so the branch comes from the schedule rows - the same
    # pattern `_branch_ids_with_available_doctors` uses, for the same
    # reason.
    effective_date = None
    try:
        timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
        effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
    except Exception:
        logger.exception("find_branches_offering_service: failed to compute today's date")

    schedule_result = api.get_doctor_schedule(
        base_url, doctor_ids=[i["id"] for i in doctor_items], page_size=500,
        effective_date=effective_date, include_future=True,
        language=conversation_language(state),
    )

    if not schedule_result["success"]:
        logger.error(
            "find_branches_offering_service: get_doctor_schedule failed: status_code=%s error=%s",
            schedule_result.get("status_code"), schedule_result.get("error"),
        )
        return {"status": "error"}

    doctors_per_branch = {}
    for row in (schedule_result["data"] or {}).get("items", []):
        branch_id = row.get("branchId")
        doctor_id = row.get("doctorId")
        if branch_id and doctor_id:
            doctors_per_branch.setdefault(branch_id, set()).add(doctor_id)

    if not doctors_per_branch:
        return {"status": "not_found", "service": resolved_service}

    branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
    names_by_id = {}
    if branches_result["success"]:
        for b in _with_branch_aliases((branches_result["data"] or {}).get("items", []), state):
            if b.get("id"):
                names_by_id[b["id"]] = _arabic_preferred_name(b) or b.get("name")

    branches = [
        {
            "id": branch_id,
            "name": names_by_id.get(branch_id),
            "doctorCount": len(doctor_ids),
        }
        for branch_id, doctor_ids in doctors_per_branch.items()
        if names_by_id.get(branch_id)
    ]

    if not branches:
        return {"status": "not_found", "service": resolved_service}

    _remember_list(state, "branch", branches)

    logger.info(
        "find_branches_offering_service: service=%s (%s) -> %d branch(es)",
        resolved_service["id"], resolved_service.get("name"), len(branches),
    )

    return {"status": "found", "service": resolved_service, "branches": branches}


@tool
def answer_hospital_faq(
    state: Annotated[AgentState, InjectedState],
    question: str,
) -> dict:
    """Look up this clinic's own general information (vision, mission,
    values, goals, details about a SPECIFIC service, branch addresses/
    contact details, policies, partners, etc.) to answer an FAQ-style
    question - NOT for schedules, availability, or booking (those have
    their own tools), and NOT for "what services do you offer?" (use
    `list_hospital_services`, which returns the complete list rather
    than the passages that happen to look most similar).
    Returns the most relevant passages found; summarize them naturally
    in 2-3 sentences rather than reproducing them verbatim. Returns:
    {"status": "found", "passages": ["...", ...]}
    {"status": "not_found"}  # nothing relevant enough was found
    {"status": "not_configured"}  # this clinic has no FAQ knowledge base set up yet"""

    kb_file = (state.get("templates") or {}).get("_knowledge_base_file", "")

    if not kb_file:
        logger.warning("answer_hospital_faq called but no knowledge_base_file is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    passages = rag.search(kb_file, question)

    if not passages:
        return {"status": "not_found"}

    return {"status": "found", "passages": passages}


# ==========================================================
# Doctor/Branch info lookup (fuzzy name matching + listing)
# ==========================================================
#
# READ-ONLY FAQ/info lookup - never touches booking/availability. Fully
# generic: works off whatever Doctors/GetList and Branches/GetList
# return for THIS client_id, no per-clinic hardcoding.

def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy comparison: strip diacritics and
    collapse common letter variants (alef forms, ta marbuta/ha, alef
    maksura/ya) so typo/spelling variations still match."""

    if not text:
        return ""

    text = str(text).strip().lower()
    # Strip Arabic diacritics (tashkeel)
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # Normalize alef variants -> ا
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize ta marbuta -> ه, alef maksura -> ي
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_entity_filler(text: str) -> str:
    """Remove the generic words patients naturally put around a name -
    "فرع الدقي", "مستشفى تناسق", "دكتور أحمد", "Dokki branch".

    These carry no identifying information, but they DO wreck fuzzy
    matching: "فرع النزهة" against a stored "النزهة" scores far lower
    than the bare name, and against an English "Al Nozha" it fails
    outright. Confirmed real production failure: "النزهة" matched
    correctly while "فرع النزهة" - the far more natural way to ask -
    came back as a branch that doesn't exist.

    Only ever used to build an EXTRA comparison candidate; the original
    text is still matched too, so a clinic whose branch is genuinely
    called e.g. "مركز الفرع" can't be broken by this."""

    if not text:
        return ""

    stripped = str(text)
    for filler in (
        "فرع", "فروع", "الفرع", "مستشفى", "المستشفى", "مستشفي", "عيادة", "العيادة",
        "مركز", "المركز", "دكتور", "الدكتور", "دكتوره", "دكتورة", "د.", "طبيب", "الطبيب",
        "branch", "hospital", "clinic", "center", "centre", "doctor", "dr.", "dr",
    ):
        stripped = re.sub(rf"(?:^|\s){re.escape(filler)}(?=\s|$)", " ", stripped, flags=re.IGNORECASE)

    stripped = re.sub(r"\s+", " ", stripped).strip()

    # If stripping removed everything, the filler WAS the name - keep the
    # original rather than matching on an empty string.
    return stripped or str(text).strip()


def _fuzzy_match(user_input: str, candidates: list, name_keys: list) -> dict:
    """Match `user_input` against `candidates` (list of raw API items),
    checking each of `name_keys` per candidate. Returns:
    {"result": "matched", "item": {...}, "score": 0.0-1.0}
    {"result": "ambiguous", "items": [...]}   # 2+ close, similarly-scored matches
    {"result": "not_matched"}

    `score` lets callers distinguish a high-confidence match (exact or
    unique) from a lower-confidence one that's still worth confirming
    with the user (likely typo) - see match_entity_for_booking."""

    import difflib

    normalized_input = _normalize_arabic(user_input)
    if not normalized_input:
        return {"result": "not_matched"}

    # Also compare against the input with generic words removed, so
    # "فرع النزهة" matches a branch stored as "النزهة" (or "Al Nozha")
    # just as well as the bare name does. Both variants are tried and
    # the better score wins, so this can only ever help - see
    # _strip_entity_filler.
    candidates_input = [normalized_input]
    stripped_input = _normalize_arabic(_strip_entity_filler(user_input))
    if stripped_input and stripped_input != normalized_input:
        candidates_input.append(stripped_input)

    scored = []
    for item in candidates:
        best_score = 0.0
        for key in name_keys:
            value = item.get(key)
            if not value:
                continue
            for normalized_value in {_normalize_arabic(value), _normalize_arabic(_strip_entity_filler(value))}:
                if not normalized_value:
                    continue
                for candidate_input in candidates_input:
                    if candidate_input == normalized_value:
                        best_score = max(best_score, 1.0)
                    elif candidate_input in normalized_value or normalized_value in candidate_input:
                        best_score = max(best_score, 0.96)
                    else:
                        ratio = difflib.SequenceMatcher(None, candidate_input, normalized_value).ratio()
                        best_score = max(best_score, ratio)
        if best_score >= 0.6:
            scored.append((item, best_score))

    if not scored:
        return {"result": "not_matched"}

    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_score = scored[0][1]
    close_matches = [item for item, score in scored if score >= top_score - 0.08]

    if len(close_matches) == 1 or top_score >= 0.98:
        return {"result": "matched", "item": close_matches[0], "score": top_score}

    return {"result": "ambiguous", "items": close_matches[:5]}


def _branch_alias_map(state) -> dict:
    """Build {normalized alias -> [all normalized names for that branch]}
    from the client's configured bilingual branch names.

    Lets a branch typed in one language match an API record that only
    carries the other one - see config.get_messages()'s _branch_aliases
    for why this can't be solved by fuzzy matching alone."""

    templates = (state or {}).get("templates") or {}
    alias_map: dict = {}

    for entry in templates.get("_branch_aliases") or []:
        names = [_normalize_arabic(n) for n in (entry.get("aliases") or []) if n]
        names = [n for n in names if n]
        if len(names) < 2:
            continue
        for name in names:
            alias_map.setdefault(name, [])
            for other in names:
                if other not in alias_map[name]:
                    alias_map[name].append(other)

    return alias_map


def _with_branch_aliases(items: list, state) -> list:
    """Return `items` with each branch's configured other-language
    name(s) attached under `_configAliases`, so _fuzzy_match can match
    against those too (it's given "_configAliases" as a name key).

    Non-destructive: works on shallow copies, so the original API items
    (and the ids/fields every caller relies on) are untouched."""

    alias_map = _branch_alias_map(state)
    if not alias_map:
        return items

    enriched = []
    for item in items:
        extra: list = []
        for key in ("name", "altName", "formatedName"):
            normalized = _normalize_arabic(item.get(key))
            if not normalized:
                continue
            for alias in alias_map.get(normalized, []):
                if alias and alias not in extra:
                    extra.append(alias)
        if extra:
            copied = dict(item)
            # _fuzzy_match reads one string per name key, so join with a
            # separator it will still substring-match against.
            copied["_configAliases"] = " | ".join(extra)
            enriched.append(copied)
        else:
            enriched.append(item)

    return enriched


def _note_info_branch_availability(state, branch_row: dict) -> None:
    """Record, on the booking session, the branch the patient has just
    been shown via the INFO flow and whether anything can be booked
    there.

    Exists because the very next turn ("I want to book there") is
    routinely answered with no tool call at all, straight from the
    model's memory of the conversation - so the availability fact has to
    already be somewhere the next turn's system prompt can read it. See
    graph._build_empty_branch_booking_intent_directive, which turns this
    into an instruction on exactly that turn."""

    if not state:
        return

    session_id = state.get("session_id")
    if not session_id:
        return

    session = _get_booking_session(session_id)
    name = _arabic_preferred_name(branch_row) or branch_row.get("name")

    # The branch the patient is currently looking at, whether or not it
    # has doctors. Kept SEPARATE from the booking session's own
    # `branch_id` (which means "confirmed for a booking in progress") so
    # browsing a branch never silently confirms it - but available so a
    # later "show me the doctors" is scoped to the branch they are
    # actually looking at, instead of the whole hospital.
    session["info_branch_id"] = branch_row.get("id")
    session["info_branch_name"] = name

    if branch_row.get("hasAvailableDoctors") is False:
        session["info_branch_no_doctors"] = name
        logger.info(
            "_note_info_branch_availability: session_id=%s - %r has no bookable doctors",
            session_id, name,
        )
    else:
        session.pop("info_branch_no_doctors", None)


@tool
def match_entity_info(
    state: Annotated[AgentState, InjectedState],
    user_input: str,
    entity_type: str,
) -> dict:
    """FAQ/info lookup for doctors and branches - fuzzy name matching +
    listing. READ-ONLY: never touches booking, schedules, or
    availability - use the other tools for those.

    DUAL MODE:
      LIST MODE (user_input=""): returns ALL doctors or ALL branches as
        a list for display.
      RESOLVE MODE (user_input="user's raw text"): fuzzy-matches to ONE
        entity and returns its details. Tolerates Arabic typos, letter
        substitutions, and partial names - always pass the user's raw
        text, don't pre-process it yourself.

    `entity_type`: "doctor" or "branch".

    Returns one of:
    {"status": "list", "items": [...]}
    {"status": "matched", "item": {...}}
    {"status": "possible_match", "item": {...}}  # low-confidence guess
        (score < 0.95) - likely a typo, OR the input may not really be a
        branch/doctor in the system at all. Do NOT state this as fact -
        ask the user "هل تقصد [altName/name]؟" and WAIT for them to
        confirm before giving out its address/details/etc. For
        branches, this guess is already restricted to ones that
        currently have a real available doctor (see the "not_matched" +
        `available_branches` case below for when none do) - but it is
        still only a GUESS, never a confirmed fact, until the patient
        agrees. CONFIRMED REAL PRODUCTION FAILURE: "فرع المنار" (not a
        real branch) scored a mediocre 0.615 similarity against "فرع
        المعادي" (a real, unrelated, and currently doctor-less branch)
        and was reported as an outright match - "الفرع اللي ذكرته هو
        فرع المعادي" - stated as settled fact with no confirmation
        asked, pointing at a branch that could never actually help this
        patient. A user's "yes" to the follow-up question is what makes
        the match - don't act on the guessed branch/doctor until they've
        actually agreed it's the one they meant.
    {"status": "ambiguous", "candidates": [...]}  # show each candidate's
        name and ask the user which one they meant
    {"status": "not_matched"}  # doctors, or a branch with no viable
        alternative available at all
    {"status": "not_matched", "available_branches": [...]}  # branches
        only: no confident/real match was found (or the only guesses
        were branches with zero doctors right now, which are never
        offered even as a guess) - `available_branches` already lists
        the branches that DO currently have a doctor. Say plainly you
        couldn't find a branch by that name, then show this list in the
        SAME reply - don't ask a follow-up question just to get it.
    {"status": "not_configured"}  # no doctors_base_url set up for this client
    {"status": "error"}

    NUMBERED SELECTION: after this tool returns a "list" (or a
    "not_matched" with `available_branches`), a later bare number reply
    ("2", "٢", "رقم 2") resolves by POSITION against that exact list -
    pass it straight through as `user_input`, don't fuzzy-match a digit
    against names yourself. Extra statuses only for that case:
    {"status": "out_of_range", "list_size": N}  # number bigger than the
        list you showed - say how many there are, ask them to pick
        within it. Never say the doctor/branch "doesn't exist".
    {"status": "no_list_shown"}  # a number was given but nothing was
        listed for this entity_type yet in this conversation - show the
        list first (user_input="").

    Doctor fields: formatedName, altName, degreeName, specialtyName,
    defaultServiceName (serviceName). Fees are NOT included here - use
    `get_doctor_fees` if (and only if) the user explicitly asks a price.
    Branch fields: name, altName, address, cityName, countryName,
    stateName, email, mobile, hasAvailableDoctors.

    `hasAvailableDoctors` appears ONLY on a SINGLE matched branch (a
    positional pick or a name match) - never on a branch LIST, so a list
    can never be filtered or annotated by it. FALSE means that branch
    has no bookable doctor right now. Its only purpose: never offer to
    book at, or start a booking flow for, such a branch - not even as a
    friendly "...or would you like to book there?". Give the address,
    offer its SERVICES, and leave booking out of it. If THEY ask to book
    there, only then say the branch has nobody available and offer the
    branches that do, by name."""

    entity_type = (entity_type or "").strip().lower()
    if entity_type not in ("doctor", "branch"):
        return {"status": "error"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("match_entity_info called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    if entity_type == "doctor":
        result = api.get_doctors(base_url, page_size=200, language=conversation_language(state))
        name_keys = ["formatedName", "altName", "name"]
    else:
        result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
        name_keys = ["name", "altName", "formatedName", "cityName", "_configAliases"]

    if not result["success"]:
        logger.error("match_entity_info API call failed: entity_type=%s status_code=%s error=%s", entity_type, result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if entity_type == "branch":
        # Let a branch typed in one language match an API record
        # carrying only the other one - see _with_branch_aliases.
        items = _with_branch_aliases(items, state)

    if not user_input or not user_input.strip():
        if entity_type == "doctor":
            shaped = [
                {
                    "id": i.get("id"),
                    "formatedName": _arabic_preferred_name(i) or i.get("formatedName"),
                    "altName": i.get("altName"),
                    "specialtyName": i.get("specialtyName"),
                    "degreeName": i.get("degreeName"),
                }
                for i in items
            ]
        else:
            # WHICH BRANCHES CAN ACTUALLY BE BOOKED AT.
            #
            # DELIBERATELY *NOT* RETURNED IN LIST MODE - only stored in
            # the remembered list below, so a later positional pick can
            # carry it. The RETURNED rows carry no availability field at
            # all.
            #
            # WHY THE FIELD IS HIDDEN HERE RATHER THAN JUST DOCUMENTED:
            # it was originally returned on every row with prose rules
            # (in three separate places) saying "never announce this,
            # never filter on it". CONFIRMED REAL PRODUCTION FAILURES,
            # twice, despite those rules: first the reply appended a
            # paragraph naming the three empty branches, then - after
            # the rules were tightened - it appended "(لا يوجد أطباء
            # متاحين حالياً)" to each of those three rows instead. A
            # field visible in the tool result is a field the model will
            # eventually surface; the only reliable fix is not to send
            # it when it isn't needed.
            #
            # It IS still returned for a SINGLE branch (positional pick,
            # name match), which is the only moment it's actually needed
            # - deciding whether to offer a booking there.
            bookable_branch_ids = _branch_ids_with_available_doctors(state, base_url) or set()
            # ARABIC NAME FIRST. `_arabic_preferred_name` exists exactly
            # for this: `altName` is the Arabic form across every
            # endpoint in this API, and putting the English `name` in
            # the "name" field means that is what gets printed.
            #
            # CONFIRMED REAL PRODUCTION FAILURE: the branch list went
            # out as "Emergency / Al Manar / Al Nozha" inside an
            # otherwise fully Arabic reply, because this list shaped
            # `name` from the English field while every other branch
            # path in this file uses the helper.
            shaped = [
                {
                    "id": i.get("id"),
                    "name": _arabic_preferred_name(i) or i.get("name"),
                    "altName": i.get("altName"),
                    "address": i.get("address"),
                    "cityName": i.get("cityName"),
                }
                for i in items
            ]
            if not shaped:
                return {"status": "not_matched"}
            # The remembered copy keeps the flag - it never reaches the
            # model directly, and it's what a bare "4" resolves against.
            remembered_with_flag = [
                dict(row, hasAvailableDoctors=bool(row.get("id") in bookable_branch_ids))
                for row in shaped
            ]
            _remember_list(state, entity_type, remembered_with_flag)
            return {"status": "list", "items": shaped}
        if not shaped:
            return {"status": "not_matched"}
        # Remembered so a later bare number ("2") resolves by position
        # against this EXACT list - this tool was missing this entirely,
        # even though every other listing tool in the file writes it.
        # CONFIRMED REAL PRODUCTION FAILURE: shown a numbered branch
        # list, the patient replied "1", and the reply was "هل تقصد فرع
        # عيادات سكاي التخصصية؟" - fuzzy-matching the digit "1" against
        # branch names (which of course fails) instead of just taking
        # the first item of the list it had JUST shown.
        _remember_list(state, entity_type, shaped)
        return {"status": "list", "items": shaped}

    # Positional pick ("6", "٦", "رقم 6") -> resolve against the list the
    # user was ACTUALLY shown, from whichever call produced it (list mode
    # here, or the `available_branches` fallback below) - never fuzzy-
    # match a bare digit against names, which always fails.
    position = _extract_selection_number(user_input)
    if position is not None:
        session = _get_booking_session(state.get("session_id"))
        last_list = session.get("last_list")

        if not (last_list and last_list.get("entity_type") == entity_type):
            return {"status": "no_list_shown"}

        list_items = last_list.get("items") or []
        if not (1 <= position <= len(list_items)):
            return {"status": "out_of_range", "list_size": len(list_items)}

        remembered = list_items[position - 1]
        chosen_id = remembered.get("id")
        chosen_raw = next((i for i in items if i.get("id") == chosen_id), None) if chosen_id else None

        if entity_type == "doctor":
            shape_fn_for_pos = lambda i: {
                "id": i.get("id"),
                "formatedName": _arabic_preferred_name(i) or i.get("formatedName"),
                "altName": i.get("altName"),
                "degreeName": i.get("degreeName"),
                "specialtyName": i.get("specialtyName"),
                "serviceName": i.get("defaultServiceName") or i.get("serviceName"),
            }
        else:
            shape_fn_for_pos = lambda i: {
                "id": i.get("id"),
                "name": _arabic_preferred_name(i) or i.get("name"),
                "altName": i.get("altName"),
                "address": i.get("address"),
                "cityName": i.get("cityName"),
                "countryName": i.get("countryName"),
                "stateName": i.get("stateName"),
                "email": i.get("email"),
                "mobile": i.get("mobile"),
            }

        shaped_pos = shape_fn_for_pos(chosen_raw) if chosen_raw else dict(remembered)

        # Carry the availability flag through a positional pick too.
        # The remembered row already has it (every branch list this tool
        # writes now includes it); the freshly-shaped row does not, and
        # a positional pick is EXACTLY the path the confirmed failure
        # went through - the patient typed "1" for فرع المعادي and the
        # reply then offered to book there.
        if entity_type == "branch" and "hasAvailableDoctors" not in shaped_pos:
            if "hasAvailableDoctors" in remembered:
                shaped_pos["hasAvailableDoctors"] = remembered["hasAvailableDoctors"]
            else:
                try:
                    shaped_pos["hasAvailableDoctors"] = bool(
                        shaped_pos.get("id") in (_branch_ids_with_available_doctors(state, base_url) or set())
                    )
                except Exception:
                    logger.exception("match_entity_info: failed to compute hasAvailableDoctors for a positional pick")

        # REMEMBER AN EMPTY BRANCH ON THE SESSION.
        #
        # The next turn is very often "I want to book there", and it is
        # answered with NO tool call at all - straight from conversation
        # memory. Without this, that turn has no access to the fact that
        # the branch is empty, and the reply asks a doctor question that
        # can never be answered. CONFIRMED REAL PRODUCTION FAILURE: after
        # picking فرع المعادي the patient said "عاوزه احجز فيه مع دكتور"
        # and got "اخترت فرع المعادي ✅ / تحب تحجز مع دكتور معيّن...؟" -
        # a wasted turn, and a confirmation of a branch nothing can be
        # booked at.
        if entity_type == "branch":
            _note_info_branch_availability(state, shaped_pos)

        return {"status": "matched", "item": shaped_pos}

    match_candidates = items

    if entity_type == "branch":
        # TWO-TIER RESOLUTION for branches.
        #
        # An EXACT/near-exact reference - the patient really did type
        # this specific branch's name - is answered regardless of
        # whether it currently has a doctor: that's a genuine, honest
        # question ("info about فرع المعادي") and deserves a genuine
        # answer even for a branch with nobody in it right now.
        #
        # But anything WEAKER than that - a guess, not an explicit
        # reference - must never be allowed to land on a branch with no
        # currently-available doctor. CONFIRMED REAL PRODUCTION FAILURE:
        # "فرع المنار" (not a real branch at all) fuzzy-matched, at a
        # mediocre 0.615 score, to "فرع المعادي" - a real branch with
        # ZERO doctors available right now - and was confidently
        # reported as the match. Checking availability only AFTER the
        # fact (the `possible_match` confidence gate above) is not
        # enough by itself: it still means asking "هل تقصد فرع
        # المعادي؟" about a branch that can never actually help this
        # patient. The availability check has to happen BEFORE a guess
        # is even offered, exactly like `_resolve_branch_by_name` (used
        # by the booking flow) already does.
        exact_probe = _fuzzy_match(user_input, items, name_keys)
        if not (exact_probe["result"] == "matched" and exact_probe.get("score", 0) >= 0.95):
            active_branch_ids = _branch_ids_with_available_doctors(state, base_url)
            if active_branch_ids:
                narrowed = [b for b in items if b.get("id") in active_branch_ids]
                # Only narrow when it leaves something to guess from -
                # an API failure or a genuinely empty clinic falls back
                # to the unfiltered list rather than manufacturing a
                # false "not_matched" out of a transient error.
                if narrowed:
                    match_candidates = narrowed

    match_result = _fuzzy_match(user_input, match_candidates, name_keys)
    logger.info(
        "match_entity_info: entity_type=%s user_input=%r api_returned=%d result=%s%s",
        entity_type, user_input, len(items), match_result["result"],
        f" score={match_result.get('score')}" if match_result["result"] == "matched" else "",
    )

    if match_result["result"] == "not_matched":
        if entity_type == "branch":
            # Don't leave the patient with a bare "couldn't find it" -
            # hand back the branches that ARE currently available, so
            # the reply can say "لم أجد فرعًا باسم [x]، هذه الفروع
            # المتاحة للمستشفى" and show them in the SAME turn, rather
            # than guessing at an empty one or asking a second question
            # just to get the same list.
            available_branch_ids = _branch_ids_with_available_doctors(state, base_url)
            available_items = (
                [b for b in items if b.get("id") in available_branch_ids]
                if available_branch_ids else items
            )
            shaped_available = [
                {
                    "id": b.get("id"),
                    "name": _arabic_preferred_name(b) or b.get("name"),
                    "altName": b.get("altName"),
                    "address": b.get("address"),
                    "cityName": b.get("cityName"),
                }
                for b in available_items
            ]
            # Remembered too, for the identical reason as the plain list
            # mode above - the patient is very likely to reply with a
            # bare number to pick one of these. These are, by
            # construction, branches that DO have someone, so the
            # remembered copy says so; the returned rows carry no
            # availability field, same as list mode.
            _remember_list(
                state, "branch",
                [dict(row, hasAvailableDoctors=True) for row in shaped_available],
            )
            return {
                "status": "not_matched",
                "available_branches": shaped_available,
            }
        return {"status": "not_matched"}

    def _shape_doctor(i):
        return {
            "id": i.get("id"),
            "formatedName": _arabic_preferred_name(i) or i.get("formatedName"),
            "altName": i.get("altName"),
            "degreeName": i.get("degreeName"),
            "specialtyName": i.get("specialtyName"),
            "serviceName": i.get("defaultServiceName") or i.get("serviceName"),
            # No fee here on purpose: a doctor-info lookup is a routine
            # listing, and a price visible in the tool result reliably
            # ends up printed in the reply unprompted. Fees go through
            # `get_doctor_fees`, only when the user explicitly asks.
        }

    def _shape_branch(i):
        shaped_branch = {
            "id": i.get("id"),
            "name": _arabic_preferred_name(i) or i.get("name"),
            "altName": i.get("altName"),
            "address": i.get("address"),
            "cityName": i.get("cityName"),
            "countryName": i.get("countryName"),
            "stateName": i.get("stateName"),
            "email": i.get("email"),
            "mobile": i.get("mobile"),
        }
        # Same reason as the list-mode flag above: whether this branch
        # can be booked at travels WITH the branch, so a reply can never
        # offer a booking at an empty one.
        try:
            shaped_branch["hasAvailableDoctors"] = bool(
                i.get("id") in (_branch_ids_with_available_doctors(state, base_url) or set())
            )
        except Exception:
            logger.exception("match_entity_info: failed to compute hasAvailableDoctors")
        return shaped_branch

    shape_fn = _shape_doctor if entity_type == "doctor" else _shape_branch

    if match_result["result"] == "matched":
        # LOW-CONFIDENCE GUESS GATE - mirrors match_entity_for_booking's        # own needs_confirmation logic exactly (same 0.95 cutoff), which
        # this tool was missing entirely.
        #
        # CONFIRMED REAL PRODUCTION FAILURE: "فرع المنار" (not a real
        # branch) scored only 0.615 against "فرع المعادي" (an unrelated,
        # real branch) - a mediocre score, but the ONLY candidate above
        # the bare 0.6 inclusion floor, so it came back as a flat
        # "matched" with no hint that this was a guess. The reply then
        # stated it as settled fact ("الفرع اللي ذكرته هو فرع المعادي")
        # instead of confirming first. `match_entity_for_booking` already
        # guards against exactly this by asking "did you mean X?" below
        # score 0.95 - this read-only info lookup had no equivalent
        # guard at all, despite using the very same `_fuzzy_match`.
        if match_result.get("score", 0) < 0.95:
            return {"status": "possible_match", "item": shape_fn(match_result["item"])}
        matched_item = shape_fn(match_result["item"])
        if entity_type == "branch":
            # Same reason as the positional-pick path above.
            _note_info_branch_availability(state, matched_item)
        return {"status": "matched", "item": matched_item}

    ambiguous_candidates = [shape_fn(i) for i in match_result["items"]]
    # Remembered too - a follow-up bare number ("2") should pick between
    # exactly these candidates, not require the patient to retype a name.
    _remember_list(state, entity_type, ambiguous_candidates)
    return {"status": "ambiguous", "candidates": ambiguous_candidates}


# ==========================================================
# New Booking (create a brand new appointment)
# ==========================================================
#
# Uses an internal per-session "booking session" store (module-level
# dict keyed by session_id) so the LLM never has to handle or pass raw
# doctor/branch UUIDs itself. This mirrors a deliberate, battle-tested
# design confirmed directly from a real production n8n system: even
# with full conversation history available to the model, having it
# re-type or pass UUIDs reliably was NOT safe enough in practice there.
# Tools read/write these fields directly via session_id; the LLM only
# ever passes plain names/text, never IDs.

# (_BOOKING_SESSIONS / _get_booking_session are defined near the top of
# this module now - see the comment there for why they had to move.)


@tool
def reset_booking_session(state: Annotated[AgentState, InjectedState]) -> dict:
    """Clear any previously-confirmed doctor/branch/service for a NEW
    booking. Call this as the FIRST action whenever the user starts a
    brand new booking ("حجز جديد"/"new booking"/"ابي احجز"), or
    explicitly wants to change branch or start completely over - this
    prevents a stale doctor/branch from a PREVIOUS booking earlier in
    this same conversation from silently carrying over and filtering
    results. Do NOT call this mid-flow otherwise (e.g. not just because
    the user picked a different day or time - only for a genuine restart
    or explicit branch change). Returns {"status": "reset"}."""

    session_id = state.get("session_id")
    _BOOKING_SESSIONS[session_id] = {
        "doctor_id": None, "branch_id": None, "service_id": None,
        "last_list": None, "specialty_ids": None,
        "_touched_at": time.monotonic(),
    }
    return {"status": "reset"}


_GENERIC_ENTITY_WORDS = {
    "doctor": {
        "دكتور", "الدكتور", "دكتوره", "دكتورة", "دكاتره", "دكاترة", "الدكاتره",
        "الدكاترة", "طبيب", "الطبيب", "طبيبه", "طبيبة", "اطباء", "الاطباء",
        "doctor", "doctors", "a doctor", "the doctor",
    },
    "branch": {
        "فرع", "الفرع", "فروع", "الفروع", "فرعكم", "فرع معين", "فرع معيّن",
        "branch", "branches", "a branch", "the branch",
    },
}


def _is_generic_entity_word(user_input: str, entity_type: str) -> bool:
    """True when the user's text is just the WORD "doctor"/"branch"
    rather than the NAME of one.

    WHY THIS EXISTS: confirmed real production failure. Asked "تحب
    تحجزين في فرع معيّن، ولا أعرض لك الدكاترة المتاحين؟" the patient
    replied "فرع" - meaning "yes, a branch" - and that bare word was
    fuzzy-matched against the branch list and confirmed as an actual
    branch ("فرع المعادي تم اختياره ✅") that the patient had never
    named or seen. Everything downstream then ran against the wrong
    branch, which in that case had no doctors at all, dead-ending the
    booking.

    The prompt already documents the same trap for the bare word
    "دكتور", but a rule the model has to remember is not enough here:
    fuzzy matching will always find *something* plausible for a short
    generic word, so this is caught before matching runs at all. Treated
    as list mode - which is what the patient was asking for anyway.
    """

    normalized = _normalize_arabic((user_input or "").strip().lower())
    normalized = re.sub(r"[^\w\u0600-\u06FF ]+", "", normalized).strip()

    if not normalized:
        return False

    candidates = _GENERIC_ENTITY_WORDS.get(entity_type, set())
    return any(normalized == _normalize_arabic(word) for word in candidates)


# Words that make a message a REQUEST TO SEE THE LIST rather than a name.
# Both the definite and bare forms are listed on purpose: patients drop
# the "ال" constantly ("اعرض الدكاتره متاحه"), and a cue list that only
# had "المتاحه" let that exact sentence fall through to name matching -
# confirmed in production.
_LIST_REQUEST_CUES = (
    "اعرض", "اعرضلي", "عرض", "وريني", "اوريني", "ورني", "شوفني", "اشوف",
    "شوف", "هات", "هاتلي", "جبلي", "ادينى", "ادينی", "قائمه", "قائمة",
    "لستة", "لسته", "كل", "جميع", "كافه", "كافة", "مين", "ايه", "ما",
    "متاح", "متاحه", "متاحة", "متاحين", "المتاح", "المتاحه", "المتاحة",
    "المتاحين", "متوفر", "متوفره", "متوفرين", "المتوفر", "المتوفره",
    "المتوفرين", "موجود", "موجوده", "موجودين", "الموجودين", "الموجوده",
    "عندكو", "فاضي", "فاضيين",
    "show", "list", "all", "available", "who", "which", "see", "view",
)

# Filler that carries no naming information, so it doesn't count as a
# leftover "name" when we check what the message is really asking for.
_LIST_REQUEST_FILLER = (
    "لو", "سمحت", "من", "فضلك", "ممكن", "عايز", "عاوز", "عايزه", "عاوزه",
    "ابغى", "ابغي", "اريد", "بدي", "حابب", "حابه", "لي", "لى", "عندك",
    "عندكم", "عندنا", "في", "فى", "هو", "هي", "دول", "ديه", "ده", "دي",
    "please", "can", "you", "me", "i", "want", "the", "a", "for", "us",
    "your", "have", "do",
)


def _is_entity_list_request(user_input: str, entity_type: str) -> bool:
    """True when the message asks to SEE the doctors/branches rather than
    naming one.

    WHY THIS IS SEPARATE FROM _is_generic_entity_word: that function
    requires the whole message to BE the bare word ("دكتور"). A real
    patient writes a sentence - "اعرض كل الدكاتره المتاحه" - which
    slipped straight past it into fuzzy name matching, and from there
    into a live API lookup that timed out after 16 seconds and ended the
    turn with "فيه مشكلة تقنية". Confirmed real production failure. The
    patient asked a perfectly clear question; nothing about it was a
    name.

    The test is deliberately conservative: the message must contain a
    generic entity word AND a list cue, and after removing those plus
    ordinary filler there must be NOTHING meaningful left. That last
    part is what keeps "اعرض مواعيد دكتور محمد" out of list mode - the
    residue "مواعيد محمد" shows a real name was given, so it still goes
    to name matching.
    """

    normalized = _normalize_arabic((user_input or "").strip().lower())

    # Arabic punctuation (؟ ، ؛ ٪ ...) lives INSIDE the \u0600-\u06FF
    # block, so a range-based "keep Arabic letters" rule keeps it glued
    # to the word. That left "المتاحين؟" as one token, which matched
    # nothing, and a plain question like "مين الدكاترة المتاحين؟" fell
    # through to name matching. Strip it explicitly first.
    normalized = re.sub(r"[\u060C\u061B\u061F\u066A-\u066D\u06D4\u00BF]+", " ", normalized)
    normalized = re.sub(r"[^\w\u0600-\u06FF ]+", " ", normalized)

    tokens = [t for t in normalized.split() if t]
    if not tokens:
        return False

    generic = {_normalize_arabic(w) for w in _GENERIC_ENTITY_WORDS.get(entity_type, set())}
    cues = {_normalize_arabic(w) for w in _LIST_REQUEST_CUES}
    filler = {_normalize_arabic(w) for w in _LIST_REQUEST_FILLER}

    has_entity_word = any(token in generic for token in tokens)
    has_list_cue = any(token in cues for token in tokens)

    if not (has_entity_word and has_list_cue):
        return False

    residue = [t for t in tokens if t not in generic and t not in cues and t not in filler]

    return not residue


# ==========================================================
# Doctor list fetch for booking: cached, with a cheaper fallback
# ==========================================================

# (base_url, specialty_key, branch_key) -> (fetched_at, result)
_DOCTOR_LIST_CACHE: dict = {}


def _fetch_doctors_for_booking(
    state: AgentState,
    base_url: str,
    specialty_ids: Optional[list],
    branch_ids: Optional[list],
    service_ids: Optional[list] = None,
) -> dict:
    """Fetch the bookable-doctor list, defensively.

    WHY THIS IS NOT JUST ONE api.get_doctors CALL - measured, in
    production, on the same tenant within one hour:

        10:12  same payload, unfiltered by specialty  ->  8 doctors, 0.5s
        11:18  same payload, unfiltered by specialty  ->  TIMEOUT, 29.4s

    The request was byte-identical both times, so the query wasn't wrong;
    the endpoint's own latency moved by two orders of magnitude. Three
    things follow from that, and this function does all three:

      1. CACHE briefly. A patient who asks to see the doctors, picks one,
         then changes their mind should not pay for that call three
         times. The TTL is short (DOCTOR_LIST_CACHE_SECONDS) because a
         roster does change - this is a latency shield, not a data store.

      2. DROP THE EXPENSIVE FILTER ON FAILURE. `intersectionStart/End`
         asks the API to compute each doctor's schedule intersection with
         a 14-day window - almost certainly the costly part of the query,
         since the plain list has no such join. On timeout, retry once
         without it: the result then includes doctors who may not have a
         free slot, which the booking flow checks anyway at the slot
         step. A slightly looser list beats "فيه مشكلة تقنية".

      3. ASK FOR LESS. page_size drops on the retry too. A clinic with
         eight bookable doctors does not need a 200-row page.

    Returns api.get_doctors' own result dict, so callers are unchanged.
    """

    # `service_ids` IS PART OF THE KEY. Without it, a service-filtered
    # result and an unfiltered one for the same branch collide, and
    # whichever ran first is served to both.
    cache_key = (
        base_url,
        tuple(sorted(specialty_ids)) if specialty_ids else None,
        tuple(sorted(branch_ids)) if branch_ids else None,
        tuple(sorted(service_ids)) if service_ids else None,
    )

    cached = _DOCTOR_LIST_CACHE.get(cache_key)
    if cached and (time.monotonic() - cached[0]) < DOCTOR_LIST_CACHE_SECONDS:
        age = time.monotonic() - cached[0]
        logger.info(
            "doctor list: served from cache (age %.1fs, %d items)",
            age, len((cached[1].get("data") or {}).get("items", [])),
        )
        return cached[1]

    language = conversation_language(state)
    now = datetime.utcnow()
    window_start = now.isoformat() + "Z"
    window_end = (now + timedelta(days=DOCTOR_AVAILABILITY_WINDOW_DAYS)).isoformat() + "Z"

    # --- Attempt 1: the precise query -----------------------------
    started = time.monotonic()
    result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids or None,
        branch_ids=branch_ids,
        service_ids=service_ids or None,
        has_published_service=True,
        has_service_schedule=True,
        intersection_start=window_start,
        intersection_end=window_end,
        page_size=200,
        language=language,
    )
    elapsed = time.monotonic() - started

    if result.get("success"):
        items = (result.get("data") or {}).get("items", [])
        logger.info(
            "doctor list: precise query took %.1fs -> %d items "
            "(specialty_ids=%s branch_ids=%s service_ids=%s)",
            elapsed, len(items), specialty_ids, branch_ids, service_ids,
        )
        _DOCTOR_LIST_CACHE[cache_key] = (time.monotonic(), result)
        return result

    logger.warning(
        "doctor list: precise query FAILED after %.1fs (error=%s) - retrying "
        "without the schedule-intersection window",
        elapsed, result.get("error"),
    )

    # --- Attempt 2: drop the expensive join, ask for less ---------
    started = time.monotonic()
    fallback = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids or None,
        branch_ids=branch_ids,
        # The service filter is kept on the fallback too. Dropping the
        # SCHEDULE-INTERSECTION window is a deliberate, documented
        # loosening (the slot step re-checks availability anyway);
        # dropping the service would be a different thing entirely -
        # doctors who do not provide what the patient asked for are not
        # a "slightly looser list", they are wrong answers.
        service_ids=service_ids or None,
        has_published_service=True,
        has_service_schedule=True,
        page_size=50,
        language=language,
    )
    elapsed = time.monotonic() - started

    if fallback.get("success"):
        items = (fallback.get("data") or {}).get("items", [])
        logger.info(
            "doctor list: fallback query took %.1fs -> %d items (no window, page_size=50)",
            elapsed, len(items),
        )
        # Cached too: if the precise query is currently unhealthy, the
        # next few turns should not each rediscover that the slow way.
        _DOCTOR_LIST_CACHE[cache_key] = (time.monotonic(), fallback)
        return fallback

    logger.error(
        "doctor list: fallback query ALSO failed after %.1fs (error=%s) - "
        "the Doctors/GetList endpoint is not responding",
        elapsed, fallback.get("error"),
    )

    return result


def _branches_with_real_slots(state: AgentState, base_url: str, doctor_id: str,
                                branch_ids: list, future_branch_ids: Optional[set] = None) -> Optional[set]:
    """Which of `branch_ids` genuinely have at least one open (non-booked)
    slot for `doctor_id` within the booking window.

    WHY THIS EXISTS: branches for an ALREADY-CONFIRMED doctor are
    derived from her general DoctorSchedules rows - which branch(es) she
    is assigned to recurringly - not from whether she currently has
    anything bookable there. A patient was shown a branch purely because
    a schedule ROW exists for that pairing, picked it, and only found
    out afterwards that she had nothing open there.

    `future_branch_ids`: branches with a rota that has not STARTED yet
    (effectiveFrom in the future). These are ALWAYS returned as having
    slots, regardless of what the slot sweep finds.

    WHY THIS MATTERS, CONFIRMED IN PRODUCTION: a doctor's only Thursday
    rota at a branch had expired, and her Monday rota there (effective
    a future date) genuinely had an open slot - but the branch was
    reported "محجوز بالكامل حاليًا" anyway. Publishing a rota does not
    guarantee the underlying booking system has already generated
    bookable slots for it; the slot sweep below simply may not reach
    that far yet. `_mark_fully_booked_schedule_days` (the day-level
    schedule display) already carried this exemption - this branch-level
    check did not, and the gap is exactly what produced the false
    "fully booked".

    ONE QUERY PER BRANCH, ON PURPOSE.
    --------------------------------
    This used to make a single batched call for every branch at once and
    then group the results by each slot's `branchId`. The slots endpoint
    returns slotStart/slotEnd/isBooked - `branchId` is NOT a field it is
    documented or confirmed to return. When it is absent, the grouping
    finds nothing for any branch, and the function reports that EVERY
    branch is full.

    CONFIRMED REAL PRODUCTION FAILURE: a doctor with genuine open
    appointments at الدقي was reported as fully booked there -
    `list_available_days_for_booking`, which does not depend on that
    field, returned her real days at that same branch moments later.

    Asking per branch means the ANSWER is what the query was scoped to,
    rather than something inferred from a field that may not be there.
    That is a handful of calls (two or three in practice), made once, at
    the only point where it matters.

    Returns None ("unknown - don't filter anything") if EVERY lookup
    failed, so a transient outage can never mark a real branch full.
    """

    if not branch_ids:
        return set()

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now = datetime.now(tz)
    window_end = now + timedelta(days=DOCTOR_AVAILABILITY_WINDOW_DAYS)

    have_slots = set(future_branch_ids or ())
    any_lookup_succeeded = False

    for branch_id in branch_ids:
        if not branch_id or branch_id in have_slots:
            # Already counted as available via future_branch_ids - no
            # need to spend a call confirming what we've already decided
            # not to doubt.
            continue

        slots = _open_slots_on_day(
            state, base_url, doctor_id, branch_id,
            now.isoformat(), window_end.isoformat(), timezone_name,
        )

        if slots is None:
            # This branch's check failed. Treat it as available rather
            # than hiding a branch that may well have appointments.
            have_slots.add(branch_id)
            continue

        any_lookup_succeeded = True
        if slots:
            have_slots.add(branch_id)

    if not any_lookup_succeeded and not future_branch_ids:
        logger.warning(
            "_branches_with_real_slots: every availability lookup failed for doctor_id=%s - "
            "reporting 'unknown' so nothing gets marked full on a transient error",
            doctor_id,
        )
        return None

    return have_slots


@tool
def match_entity_for_booking(
    state: Annotated[AgentState, InjectedState],
    user_input: str,
    entity_type: str,
) -> dict:
    """Resolve a doctor or branch by the user's raw text for a NEW
    BOOKING, AND automatically confirm+remember it in this booking's
    session - you NEVER need to track, save, or pass any ID yourself;
    this tool handles that entirely, including filtering doctors to an
    already-confirmed branch automatically.

    DUAL MODE:
      LIST MODE (user_input=""): lists all doctors/branches. If a
        branch is already confirmed in this booking session and
        entity_type="doctor", the list is automatically filtered to
        doctors at that branch only - you don't need to filter it
        yourself or pass the branch.
      RESOLVE MODE (user_input="user's raw text"): matches to ONE
        entity. This also accepts a bare number referring to a position
        in the list you most recently showed via this same tool (e.g.
        user replies "2" after you displayed a numbered list) - always
        pass the user's raw text/number as-is, the tool handles both
        cases.

    `entity_type`: "doctor" or "branch".

    Returns one of:
    {"matched": true, "needsConfirmation": false, "item": {...}}
        -> CONFIRMED AND SAVED to the booking session automatically -
           do NOT ask "are you sure" for this case, proceed directly.
           When entity_type="branch" AND no doctor was already confirmed
           in this booking, this ALSO carries "doctorsAtBranch": [...] -
           the doctors who actually work at that branch (narrowed to
           this booking's specialty when known) and already remembered
           for numeric selection. Show THAT list, numbered - never
           re-show doctor names from before the branch was chosen,
           because not every doctor works at every branch.
    {"matched": true, ..., "fullyBooked": true}
        -> the branch is REAL and this doctor does work there, but has
           no open slot in the booking window right now. Say exactly
           that - "الفرع ده محجوز بالكامل حاليًا عند د. [name]" - and
           offer the other branch, or a later date. NEVER say the
           branch doesn't exist, and never act as though the patient
           named something wrong: they named a branch you yourself
           showed them.
    {"matched": true, ..., "doctorAlreadyConfirmed": true}
        -> the branch was confirmed while a DOCTOR was already
           confirmed earlier in this booking. There is no doctor list
           here on purpose - one was already picked, so do not ask
           "which doctor?" or show any doctor roster. Go straight to
           `list_available_days_for_booking` for the doctor+branch pair
           already on file. Confirmed real, repeated production
           failure: a confirmed doctor kept getting silently dropped the
           moment a branch was confirmed afterward, with the reply
           reverting to "here are the available doctors" as if no
           doctor had ever been chosen.
    {"matched": true, "needsConfirmation": true, "item": {...}}
        -> a close-but-not-exact match (likely a typo) - nothing was
           saved yet. Ask the user "did you mean [item]?" and WAIT.
           Their "yes" is NOT a confirmation by itself - call this tool
           AGAIN with the corrected name on that turn (that call is what
           actually saves it) before proceeding.
    {"matched": false, "ambiguous": true, "candidates": [...]}
        -> multiple similarly-close matches - show each candidate's name
           and ask the user to pick one; nothing was saved.
    {"matched": false, "ambiguous": false}
        -> no match at all.
    {"matched": true, ..., "noDoctorsAtBranch": true}
        -> the branch was confirmed but NOBODY works there (for this
           booking's specialty). Never claim there's a list of doctors:
           say plainly that this branch has no available doctors right
           now and offer the branches that do.
    {"matched": false, "status": "out_of_range", "list_size": N}
        -> they gave a number bigger than the list you showed. Say the
           list only has N options and ask them to pick within it - do
           NOT say the doctor/branch "doesn't exist".
    {"matched": false, "status": "no_list_shown"}
        -> they gave a number but no list has been shown yet for this
           entity_type. Show the list first (user_input=""), then let
           them pick - again, never say the doctor "doesn't exist".
    {"status": "list", "items": [...]}
        -> list mode result (user_input was empty).
    {"status": "not_configured"} / {"status": "error"}"""

    entity_type = (entity_type or "").strip().lower()
    if entity_type not in ("doctor", "branch"):
        return {"matched": False, "ambiguous": False, "status": "error"}

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)

    # Set only on the doctor-filtered branch path below. None means
    # "no availability check ran", which is NOT the same as "nothing
    # is available" - see the fullyBooked flag near the end.
    verified_branch_ids = None

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("match_entity_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"matched": False, "ambiguous": False, "status": "not_configured"}

    # DECIDE LIST-VS-NAME BEFORE FETCHING, not after.
    #
    # This check used to sit further down, AFTER the API call, purely
    # because it only needed to blank `user_input` before matching ran.
    # That ordering had a real cost once the fetch started depending on
    # the answer: "اعرض الدكاتره المتاحه" still looked like a name up
    # here, so the resolve-mode widening retry below could fire for it -
    # turning one query into two, and one timeout into a 25-second one.
    # Deciding it once, up front, is both correct and cheaper.
    wants_list = (
        not (user_input or "").strip()
        or _is_generic_entity_word(user_input, entity_type)
        or _is_entity_list_request(user_input, entity_type)
    )

    if wants_list and (user_input or "").strip():
        logger.info(
            "match_entity_for_booking: %r is asking to SEE the %ss, not naming one - list mode",
            user_input, entity_type,
        )
        user_input = ""

    if entity_type == "doctor":
        branch_filter = [session["branch_id"]] if session.get("branch_id") else None

        # NARROW THE QUERY. This used to ask the hospital's API for every
        # doctor in the system with no filters at all, which on a real
        # tenant took longer than the 15s HTTP timeout and ended the turn
        # with "فيه مشكلة تقنية" - while find_available_doctors, hitting
        # the SAME endpoint with these filters applied, answered the same
        # question in under a second. Confirmed against production logs
        # minutes apart.
        #
        # It is also the more correct list: this tool exists to pick a
        # doctor for a NEW BOOKING, and a doctor with no published
        # service or no schedule cannot be booked, so listing them only
        # offers the patient choices that dead-end.
        # A SERVICE ALREADY CHOSEN MUST NARROW THIS LIST TOO.
        #
        # `find_available_doctors` honours the session's `service_id`,
        # but this path did not - so the same booking showed a
        # service-filtered roster from one tool and an unfiltered,
        # longer one from the other, for the same branch. The extra
        # names are doctors who do not provide the service the patient
        # picked.
        result = _fetch_doctors_for_booking(
            state, base_url, session.get("specialty_ids"), branch_filter,
            service_ids=[session["service_id"]] if session.get("service_id") else None,
        )

        logger.info(
            "match_entity_for_booking: doctor fetch (mode=%s) -> success=%s items=%d",
            "list" if wants_list else "name",
            result.get("success"),
            len((result.get("data") or {}).get("items", [])) if result.get("success") else -1,
        )

        # RESOLVE MODE SAFETY NET: the patient named a specific doctor
        # and the narrowed list didn't contain them. Widen once before
        # concluding "no such doctor" - they may be real but fully
        # booked, and "we couldn't find that doctor" would be wrong and
        # confusing.
        #
        # Deliberately NOT done in list mode: there, an empty narrowed
        # list is the honest answer, and widening would double the cost
        # of the exact request that was already too slow.
        if not wants_list and result.get("success"):
            narrowed = (result["data"] or {}).get("items", [])
            if not narrowed:
                logger.info(
                    "match_entity_for_booking: narrowed doctor list was empty for %r - widening once",
                    user_input,
                )
                widen_started = time.monotonic()
                result = api.get_doctors(
                    base_url, branch_ids=branch_filter, page_size=50,
                    language=conversation_language(state),
                )
                logger.info(
                    "match_entity_for_booking: widened doctor query took %.1fs -> success=%s",
                    time.monotonic() - widen_started, result.get("success"),
                )

        name_keys = ["formatedName", "altName", "name"]
    elif session.get("doctor_id"):
        # A doctor is already confirmed - only offer branches where THIS
        # doctor actually has a schedule, derived from their own
        # DoctorSchedules rows (same source used to display schedules),
        # rather than every clinic branch regardless of relevance.
        #
        # `effective_date`/`include_future` ARE REQUIRED HERE.
        #
        # CONFIRMED REAL PRODUCTION FAILURE: this call used to omit both,
        # so it returned EVERY row ever created for this doctor -
        # including branches whose assignment had fully expired weeks
        # earlier. A doctor whose only three schedule rows at فرع الشيخ
        # زايد had all ended in May/July was still offered that branch
        # as an option in August, then correctly found nothing bookable
        # there and called it "fully booked" - which is not what was
        # true. She does not work there any more; the assignment lapsed,
        # it isn't merely full.
        #
        # `get_doctor_schedule_for_booking` (the schedule DISPLAY call)
        # already passes these two and correctly excludes lapsed
        # branches - confirmed directly, in the same production trace,
        # where it saw only one still-valid branch while this call, right
        # next to it, still offered two. Bringing this call in line with
        # that one, rather than inventing a second mechanism.
        today_iso = None
        try:
            timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
            today_iso = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            logger.exception("match_entity_for_booking (branch, doctor-filtered): failed to compute today's date for effective_date filtering")

        schedule_result = api.get_doctor_schedule(
            base_url, doctor_ids=[session["doctor_id"]],
            effective_date=today_iso, include_future=True,
            language=conversation_language(state),
        )
        if not schedule_result["success"]:
            logger.error("match_entity_for_booking (branch, doctor-filtered): get_doctor_schedule failed: status_code=%s error=%s", schedule_result.get("status_code"), schedule_result.get("error"))
            return {"matched": False, "ambiguous": False, "status": "error"}

        schedule_items = (schedule_result["data"] or {}).get("items", [])

        # SAME DIAGNOSTIC AS get_doctor_schedule_for_booking, for direct
        # comparison - both calls now use the same effective_date/
        # include_future filtering, so their raw results should agree.
        # If they don't, the difference between the two calls (not a
        # guess) is what explains it.
        logger.info(
            "match_entity_for_booking (branch, doctor-filtered): doctor_id=%s effective_date=%s -> "
            "api returned %d raw row(s): %s",
            session["doctor_id"], today_iso, len(schedule_items),
            [
                {
                    "branchId": it.get("branchId"), "branchName": it.get("branchName"),
                    "recurringDaysNames": it.get("recurringDaysNames"),
                    "fromDateTime": it.get("fromDateTime"), "toDateTime": it.get("toDateTime"),
                }
                for it in schedule_items
            ],
        )

        doctor_branch_ids = {s.get("branchId") for s in schedule_items if s.get("branchId")}

        if not doctor_branch_ids:
            return {"matched": False, "ambiguous": False, "status": "not_matched"}

        # Cross-reference against the full branch list so altName (Arabic
        # name), address, etc. aren't lost - DoctorSchedules/GetList only
        # gives branchName, not altName.
        all_branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
        if not all_branches_result["success"]:
            logger.error("match_entity_for_booking (branch, doctor-filtered): get_branches failed: status_code=%s error=%s", all_branches_result.get("status_code"), all_branches_result.get("error"))
            return {"matched": False, "ambiguous": False, "status": "error"}

        all_branch_items = (all_branches_result["data"] or {}).get("items", [])
        candidate_branches = [b for b in all_branch_items if b.get("id") in doctor_branch_ids]

        # ONE extra batched call cross-checks these candidate branches
        # against real schedule slots, so a branch that's only a general
        # schedule assignment - with nothing actually bookable there
        # right now - is not offered as a choice. See
        # _branches_with_real_slots for why.
        # MATCH FIRST, REPORT AVAILABILITY SECOND.
        #
        # This used to DROP branches with no open slot before matching,
        # so a patient naming a branch that happened to be fully booked
        # got "not_matched" - which the model faithfully reported as
        # "ما لقيت فرع اسمه الدقي".
        #
        # CONFIRMED REAL PRODUCTION FAILURE: the assistant printed the
        # doctor's schedule AT الدقي, the patient answered "الدقي", and
        # was told no such branch exists. The branch was real, the
        # doctor works there, and the only true statement was "it is
        # fully booked" - the one thing the patient was never told.
        #
        # The branches stay in the candidate list so the NAME resolves.
        # Whether anything is open there is reported separately, on the
        # match, as `fully_booked`.
        #
        # A branch with a NOT-YET-STARTED rota is never counted as full
        # here - see _branches_with_real_slots's `future_branch_ids`
        # parameter for why: this is the second, and more damaging, of
        # the two places that check needed this exemption. The day-level
        # schedule display already had it; this branch-level check did
        # not, and CONFIRMED REAL PRODUCTION FAILURE followed directly
        # from the gap - a doctor whose only Thursday rota had expired
        # and whose Monday rota (Effective From 2026-10-01) genuinely had
        # an open slot was reported as "محجوز بالكامل حاليًا" at that
        # branch, because the slot sweep simply hadn't reached that far
        # yet. Publishing a rota does not guarantee the underlying
        # booking system has generated bookable slots for it immediately
        # - this project's own day-level check already accounted for
        # that; this one now does too.
        future_branch_ids = {
            s.get("branchId")
            for s in schedule_items
            if s.get("branchId")
            and (lambda d: d and d > date.today())(
                _parse_iso_date(s.get("effectiveFrom") or s.get("fromDateTimeFrom"))
            )
        }

        verified_branch_ids = _branches_with_real_slots(
            state, base_url, session["doctor_id"],
            [b["id"] for b in candidate_branches if b.get("id")],
            future_branch_ids=future_branch_ids,
        )
        if verified_branch_ids is not None:
            full = [b for b in candidate_branches if b.get("id") not in verified_branch_ids]
            if full:
                logger.info(
                    "match_entity_for_booking (branch, doctor-filtered): %d branch(es) are "
                    "rostered for doctor_id=%s but fully booked - keeping them matchable and "
                    "flagging them rather than denying they exist",
                    len(full), session["doctor_id"],
                )

        result = {"success": True, "data": {"items": candidate_branches}, "error": None}
        name_keys = ["name", "altName", "formatedName", "cityName", "_configAliases"]
    else:
        # NO doctor confirmed yet.
        #
        # TWO-TIER RESOLUTION - mirrors the fix in `_resolve_branch_by_name`
        # and `match_entity_info` for the identical underlying problem.
        #
        # An EXACT/near-exact reference - the patient really did type
        # this specific branch's name - must resolve to THAT branch,
        # regardless of whether it currently has a doctor. Narrowing the
        # candidate pool BEFORE checking for an exact reference silently
        # swaps one real branch for a DIFFERENT real branch when the
        # named one has no doctors - CONFIRMED REAL PRODUCTION FAILURE:
        # the patient typed "المعادي" (a real, exactly-named branch,
        # just currently empty of doctors), and because Maadi had
        # already been filtered out of the candidate pool before
        # matching even started, the fuzzy match was forced to guess
        # among the remaining branches and silently locked in "الدقي"
        # (Dokki) instead - a completely different, real branch the
        # patient never mentioned - with no confirmation step at all.
        #
        # Only a WEAK/guessed match (not an explicit reference) is ever
        # restricted to branches that currently have a doctor - for the
        # ORIGINAL reason this narrowing exists: "فرع المنار" (not a real
        # branch at all) must never be allowed to guess its way onto a
        # real-but-empty branch like "فرع المعادي".
        all_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
        result = all_result
        if all_result["success"]:
            raw_items = (all_result["data"] or {}).get("items", [])
            aliased_items = _with_branch_aliases(raw_items, state)
            exact_probe = _fuzzy_match(
                user_input, aliased_items,
                ["name", "altName", "formatedName", "cityName", "_configAliases"],
            )
            if not (exact_probe["result"] == "matched" and exact_probe.get("score", 0) >= 0.95):
                active_branch_ids = _branch_ids_with_available_doctors(
                    state, base_url, session.get("specialty_ids") or None,
                )
                if active_branch_ids:
                    narrowed_items = [b for b in raw_items if b.get("id") in active_branch_ids]
                    if narrowed_items:
                        result = {"success": True, "data": {"items": narrowed_items}, "error": None}
                    else:
                        logger.info(
                            "match_entity_for_booking (branch, no doctor yet): no currently-staffed "
                            "branch to narrow to for specialty_ids=%s - falling back to the "
                            "unfiltered branch list",
                            session.get("specialty_ids"),
                        )
        name_keys = ["name", "altName", "formatedName", "cityName", "_configAliases"]

    if not result["success"]:
        logger.error("match_entity_for_booking API call failed: entity_type=%s status_code=%s error=%s", entity_type, result.get("status_code"), result.get("error"))
        return {"matched": False, "ambiguous": False, "status": "error"}

    items = (result["data"] or {}).get("items", [])
    if entity_type == "branch":
        # Same bilingual-name bridge as match_entity_info above.
        items = _with_branch_aliases(items, state)

    _lang = conversation_language(state)

    def _shape(i):
        if entity_type == "doctor":
            return {
                "id": i.get("id"),
                # `name` is included so this tool's shape matches what
                # find_available_doctors / list_branches_for_specialty
                # return. Without it, a doctor resolved by number came
                # back under different keys than the same doctor in the
                # list the user was shown.
                "name": i.get("altName") or i.get("formatedName") or i.get("name"),
                "formatedName": i.get("formatedName") or i.get("name"),
                "altName": i.get("altName"),
                # Specialty/degree/branch in the CONVERSATION's language
                # too - the same reason the name is. Leaving these on the
                # English fields is how "استشاري · Internal Medicine" or
                # an English branch name ends up inside an Arabic reply.
                "degreeName": i.get("degreeAltName") if (_lang != "en" and i.get("degreeAltName")) else i.get("degreeName"),
                "specialtyName": i.get("specialtyAltName") if (_lang != "en" and i.get("specialtyAltName")) else i.get("specialtyName"),
                "branchId": i.get("branchId"),
                "branchName": i.get("branchAltName") if (_lang != "en" and i.get("branchAltName")) else i.get("branchName"),
            }
        return {
            "id": i.get("id"),
            "name": i.get("name") or i.get("formatedName"),
            "altName": i.get("altName"),
            "address": i.get("address"),
            "cityName": i.get("cityName"),
        }

    shaped_items = [_shape(i) for i in items]

    if not user_input or not user_input.strip():
        _remember_list(state, entity_type, shaped_items)
        if not shaped_items:
            return {"matched": False, "ambiguous": False, "status": "not_matched"}
        return {"status": "list", "items": shaped_items}

    # Positional pick ("6", "٦", "رقم 6") -> resolve against the list the
    # user was ACTUALLY shown, whichever tool produced it. Two things
    # changed here, both of which independently broke real bookings:
    #
    #   1. The list is now written by _remember_list from every listing
    #      tool, not only from this one's own list mode.
    #   2. The chosen item is taken straight from the remembered list
    #      instead of requiring a second lookup to re-find it in this
    #      call's freshly-fetched `items`. That re-lookup silently failed
    #      whenever the two queries didn't line up - e.g. the list came
    #      from an availability-filtered/specialty-filtered search but
    #      this call fetches the unfiltered roster (or a branch-filtered
    #      one), so the id simply wasn't there and the pick was reported
    #      to the user as "that number doesn't exist".
    position = _extract_selection_number(user_input)
    last_list = session.get("last_list")

    if position is not None and last_list and last_list.get("entity_type") == entity_type:
        list_items = last_list.get("items") or []

        if 1 <= position <= len(list_items):
            remembered = list_items[position - 1]
            chosen_id = remembered.get("id")

            # Enrich from the live roster when the same entity is present
            # there (gives altName/branch info the remembered copy may
            # lack); otherwise trust the remembered item as-is.
            chosen_raw = next((i for i in items if i.get("id") == chosen_id), None)
            shaped = _shape(chosen_raw) if chosen_raw else dict(remembered)

            if shaped.get("id"):
                session[f"{entity_type}_id"] = shaped["id"]
                session[f"{entity_type}_display_name"] = _arabic_preferred_name(shaped)
                logger.info(
                    "match_entity_for_booking: resolved position %d -> %s_id=%s (%s)",
                    position, entity_type, shaped["id"], session[f"{entity_type}_display_name"],
                )
                response = {"matched": True, "needsConfirmation": False, "item": shaped}

                if entity_type == "branch":
                    if session.get("doctor_id"):
                        # A doctor is ALREADY confirmed - there is no
                        # roster to browse, so don't compute or return
                        # one. Confirmed real, repeated production
                        # failure: doctorsAtBranch being present at all
                        # (even naming the SAME already-confirmed
                        # doctor among others, or coming back empty)
                        # kept tempting the model into re-presenting a
                        # doctor choice that was already settled,
                        # discarding the confirmed doctor entirely.
                        # Removing the data removes the temptation.
                        response["doctorAlreadyConfirmed"] = True
                    else:
                        doctors_here = _doctors_at_branch(state, base_url, shaped["id"])
                        response["doctorsAtBranch"] = doctors_here
                        if not doctors_here:
                            response["noDoctorsAtBranch"] = True

                return response

        logger.warning(
            "match_entity_for_booking: position %d out of range for last_list of %d %s item(s)",
            position, len(list_items), entity_type,
        )
        return {
            "matched": False, "ambiguous": False,
            "status": "out_of_range", "list_size": len(list_items),
        }

    if position is not None:
        # A number, but nothing was listed for this entity_type yet -
        # tell the model that plainly rather than letting it fall through
        # to fuzzy-matching a digit against doctor names (which always
        # fails and produced the misleading "that doctor doesn't exist").
        logger.warning(
            "match_entity_for_booking: got positional input %r but no %s list is remembered for this session",
            user_input, entity_type,
        )
        return {"matched": False, "ambiguous": False, "status": "no_list_shown"}

    match_result = _fuzzy_match(user_input, items, name_keys)

    if match_result["result"] == "not_matched":
        return {"matched": False, "ambiguous": False}

    if match_result["result"] == "ambiguous":
        return {"matched": False, "ambiguous": True, "candidates": [_shape(i) for i in match_result["items"]]}

    # matched - decide confidence: high score (exact/unique) auto-confirms
    # and saves to session; lower score is a likely typo needing "did you
    # mean X?" confirmation before anything is saved.
    shaped = _shape(match_result["item"])
    needs_confirmation = match_result["score"] < 0.95

    if not needs_confirmation:
        session[f"{entity_type}_id"] = shaped["id"]
        session[f"{entity_type}_display_name"] = _arabic_preferred_name(shaped)

    response = {"matched": True, "needsConfirmation": needs_confirmation, "item": shaped}

    # The branch resolved, but has nothing open. Reported here so the
    # reply can say "fully booked" - which is true, useful, and lets the
    # patient ask about a later date - instead of the branch being
    # dropped before matching and reported as not existing.
    if (
        entity_type == "branch"
        and shaped.get("id")
        and verified_branch_ids is not None
        and shaped["id"] not in verified_branch_ids
    ):
        response["fullyBooked"] = True

    if entity_type == "branch" and not needs_confirmation and shaped.get("id"):
        if session.get("doctor_id"):
            # Same reasoning as the positional-pick branch above: a
            # doctor is already confirmed, so there is no roster to
            # browse - don't return one, don't tempt a re-presentation
            # of a decision that's already made.
            response["doctorAlreadyConfirmed"] = True
        else:
            doctors_here = _doctors_at_branch(state, base_url, shaped["id"])
            response["doctorsAtBranch"] = doctors_here
            if not doctors_here:
                # Explicit flag rather than just an empty list: confirmed
                # real failure - with an empty list the reply still said
                # "هنا قائمة الدكاترة المتاحين في الفرع" and then listed
                # nobody, leaving the patient with a confirmed branch and no
                # way forward.
                response["noDoctorsAtBranch"] = True
            response["noDoctorsAtBranch"] = True

    return response


# Generic, universal hospital/clinic department words that carry a
# single unambiguous standard Arabic rendering everywhere - as opposed
# to a clinic-specific PROPER NOUN branch name (e.g. "Al Manar",
# "Downtown"), which has no safe generic translation and must keep
# falling through to whatever the API actually has. Only used as a
# LAST RESORT below, when the API row has no altName at all.
#
# WHY THIS EXISTS: CONFIRMED REAL PRODUCTION FAILURE (medtown,
# 2026-08-31, recurring across 3 separate turns/sessions) - the
# "Emergency" branch has no Arabic altName on file in the API, so this
# function correctly (per its own contract) fell back to the raw
# English "Emergency". The model, correctly following its own
# instruction to always answer in the conversation's language, then
# said "فرع الطوارئ" in its Arabic reply - a completely legitimate
# translation of a generic, universal term. But every invented-branch
# guard checks the reply against what tools actually RETURNED, and
# every tool result on file only ever said "Emergency" in English -
# never "الطوارئ" anywhere - so a 100% correct reply was rejected
# twice as a fabricated branch, every single time this branch came up.
# Filling in the standard Arabic name here, at the source, keeps the
# tool's own data and the model's own (correct) reply in agreement,
# rather than teaching every downstream guard to guess at translation
# equivalence on its own.
_GENERIC_BRANCH_NAME_AR = {
    "emergency": "الطوارئ",
    "emergency department": "الطوارئ",
    "er": "الطوارئ",
    "reception": "الاستقبال",
    "outpatient": "العيادات الخارجية",
    "outpatient clinic": "العيادات الخارجية",
    "pharmacy": "الصيدلية",
    "laboratory": "المختبر",
    "lab": "المختبر",
    "radiology": "الأشعة",
}


def _looks_arabic_text(text: str) -> bool:
    """Whether the text contains any Arabic script at all.

    graph.py has its own `_looks_arabic`, but tools.py cannot import
    graph (graph imports tools), so this small check is duplicated
    rather than shared.
    """

    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _arabic_preferred_name(shaped_entity: dict) -> str:
    """Pick the Arabic-preferred display name for a doctor/branch:
    `altName` is confirmed, across every specialty/doctor/branch endpoint
    in this API, to be the Arabic translation of `name`/`formatedName` -
    prefer it whenever present. Falls back to whatever name is available
    otherwise. Confirmed real user feedback: English names (e.g. "Al
    Manar", "Omar Almodayfer") were showing up mixed into otherwise-
    Arabic replies, which reads as unprofessional/inconsistent."""

    alt = (shaped_entity.get("altName") or "").strip()
    if alt:
        return alt

    fallback = (shaped_entity.get("formatedName") or shaped_entity.get("name") or "").strip()

    # Only a generic institutional word, never a clinic-specific proper
    # noun, gets auto-translated - see _GENERIC_BRANCH_NAME_AR above.
    generic_ar = _GENERIC_BRANCH_NAME_AR.get(fallback.lower())
    if generic_ar:
        return generic_ar

    return fallback


def _service_name(slot_item: dict, language: str = "ar") -> str:
    """The service name for a slot, in the conversation's language.

    The API is asked for localized content via accept-language (see
    api._post_json), which is the primary mechanism - a service, unlike
    a doctor or a branch, has no altName field to fall back on. Some
    deployments do expose one alongside it, so prefer that when present
    rather than showing an English service name ("Eye Vision Check")
    inside an Arabic slot list.
    """

    slot_item = slot_item or {}

    if language != "en":
        alt = (slot_item.get("serviceAltName") or slot_item.get("serviceNameAr") or "").strip()
        if alt:
            return alt

    return slot_item.get("serviceName")


def _preferred_name(entity: dict, language: str = "ar") -> str:
    """The display name for a doctor/branch/specialty in the
    CONVERSATION's language: `altName` (Arabic) for Arabic
    conversations, `name`/`formatedName` (English) for English ones.
    Falls back to whatever name exists rather than returning nothing."""

    entity = entity or {}

    if language == "en":
        return (
            entity.get("formatedName")
            or entity.get("name")
            or entity.get("altName")
            or ""
        ).strip()

    return _arabic_preferred_name(entity)


@tool
def get_doctor_fees(state: Annotated[AgentState, InjectedState]) -> dict:
    """Get the currently-confirmed doctor's published services and
    prices for a NEW BOOKING. Reads the doctor from the booking session
    automatically - you never pass an ID. A doctor MUST already be
    confirmed (via `match_entity_for_booking`, needsConfirmation=false)
    before calling this - if none is confirmed yet, this returns
    {"status": "no_doctor_confirmed"} and you should ask which doctor
    they're asking about first.

    IMPORTANT: fees are PRIVATE BY DEFAULT - only call this when the
    user EXPLICITLY asks about price/cost/fee. Never mention a fee
    proactively, and never quote one from schedule/slot data instead of
    this tool. Returns:
    {"status": "found", "fees": [{"service": ..., "price": ...}, ...]}
    {"status": "no_doctor_confirmed"}
    {"status": "not_found"}  # doctor has no published services
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")

    if not doctor_id:
        return {"status": "no_doctor_confirmed"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_doctor_fees called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_doctor_fees(base_url, doctor_ids=[doctor_id], language=conversation_language(state))

    if not result["success"]:
        logger.error("get_doctor_fees API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    fees = [{"service": i.get("serviceName"), "price": i.get("price")} for i in items]
    return {"status": "found", "fees": fees}


@tool
def get_patient_info(state: Annotated[AgentState, InjectedState], mobile_number: str) -> dict:
    """Look up whether a patient is already registered by phone number,
    for a NEW BOOKING - to avoid re-asking for their name/email if
    they've booked before. Returns:
    {"status": "found", "patientFullName": ..., "mobileNumber": ..., "email": ...}
        # exactly ONE patient is registered under this number - use
        # their name (and email, if present) directly, don't re-ask.
    {"status": "found_multiple", "patients": [{"patientFullName": ..., "email": ...}, ...]}
        # MORE THAN ONE patient is registered under this number (a
        # shared family phone is common). Show each name as a short
        # numbered list and ask which one this booking is for - or
        # whether they'd like to add a NEW name instead. Never silently
        # pick the first one, and never merge/guess between them.
    {"status": "not_found"}  # not registered - collect name/email fresh
    {"status": "too_early"}
        # No appointment time has been shown yet, so it is too early to
        # be collecting personal details - go back and show the
        # available times for the chosen day first, let the patient pick
        # one, and only then ask for their phone number.
    {"status": "not_configured"} / {"status": "error"}
    {"status": "phone_not_verified"}  # mobile_number isn't the channel
        # identity and hasn't been verified in this conversation (no
        # successful compare_phone match, no successful verify_otp). Go
        # complete that verification for this exact number BEFORE
        # calling this tool again - never retry as-is."""

    session = _get_booking_session(state.get("session_id"))
    if not session.get("slots_shown"):
        # Collecting personal details before a time exists is the wrong
        # order and it costs the patient real effort for nothing:
        # confirmed real production failure - the phone question was
        # asked straight after a branch was picked, so the patient could
        # have handed over their details and only then discovered no
        # suitable time was free. Times first, always.
        logger.warning(
            "get_patient_info called before any slot times were shown for session_id=%s - refusing",
            state.get("session_id"),
        )
        return {
            "status": "too_early",
            "reason": "no appointment time has been shown or chosen yet - show the available times first",
        }

    # SERVER-SIDE ENFORCEMENT, NOT JUST A PROMPT RULE - same reasoning
    # as `lookup_appointment`/`create_new_booking`'s equivalent checks:
    # this reveals a real patient's name/email for a phone number, so
    # it must not run against a number nobody has proven belongs to the
    # person messaging.
    if not _phone_is_verified(state, mobile_number):
        logger.warning(
            "get_patient_info: refusing lookup for an unverified mobile_number "
            "(session_id=%s) - compare_phone/verify_otp must succeed first",
            state.get("session_id"),
        )
        return {"status": "phone_not_verified"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_patient_info called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_patient_info(base_url, mobile_number)

    if not result["success"]:
        logger.error("get_patient_info API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    data = result["data"] or {}
    items = data.get("items", [])
    if not items or not data.get("totalCount"):
        return {"status": "not_found"}

    # CONFIRMED REAL GAP: this used to silently take items[0] and
    # discard the rest, even though a phone number shared by a family
    # can legitimately have several registered patients on file. Taking
    # the first one silently risks booking the appointment under the
    # WRONG family member's name - the tool never told the model there
    # was ever a choice to make.
    if len(items) > 1:
        return {
            "status": "found_multiple",
            "patients": [
                {
                    "patientFullName": i.get("patientFullName"),
                    "email": i.get("email"),
                }
                for i in items
            ],
        }

    item = items[0]
    return {
        "status": "found",
        "patientFullName": item.get("patientFullName"),
        "mobileNumber": item.get("mobileNumber"),
        "email": item.get("email"),
    }


@tool
def resolve_available_day(
    state: Annotated[AgentState, InjectedState],
    weekday_name: str,
    after_date: str = "",
) -> dict:
    """For a NEW BOOKING: find the NEAREST date of a given weekday that
    the currently-confirmed doctor (and branch, if also confirmed)
    ACTUALLY has a real, non-booked slot available - not just any
    calendar date matching that weekday. Reads doctor_id/branch_id from
    the booking session automatically - both must already be confirmed
    via `match_entity_for_booking` first, or this returns an error
    telling you which is missing.

    NEVER compute or guess a date yourself for a new booking - always
    call this. `after_date` (format "YYYY-MM-DD"), if given, finds the
    next occurrence STRICTLY AFTER that date - use this for "next
    Thursday"/"الخميس اللي بعده" relative to one already discussed, or
    to retry after a day turned out fully booked.
    Returns:
    {"status": "found", "date": "YYYY-MM-DD", "weekday_name": "Thursday",
     "date_display": "25/08/2026", "weekday_display": "الثلاثاء",
     "first_time_display": "11:00 صباحًا", "last_time_display": "3:00 مساءً",
     "from_date": ..., "to_date": ...}
        # SHOW `weekday_display` and `date_display` to the patient -
        # never `date`, which is a machine value in ISO format and reads
        # as a raw timestamp inside a sentence. Pass `from_date`/
        # `to_date` VERBATIM into `get_available_slots_for_booking`.
        # `first_time_display`/`last_time_display` are the EARLIEST and
        # LATEST open slot start times on that day - present them as a
        # RANGE ("من 11:00 صباحًا إلى 3:00 مساءً"), never as one specific
        # appointment time. The DAY itself, not one slot in it, is what
        # you're offering at this step; the individual bookable times
        # only come after the patient confirms the day, via
        # `get_available_slots_for_booking`.
    {"status": "fully_booked", "weekday_name": "Thursday", "weekday_display": "الخميس"}
        # The doctor DOES work that weekday here, but every slot is
        # taken. Say exactly that - "الخميس محجوز بالكامل حاليًا" - and
        # offer the days that ARE available. This is the only place that
        # fact should be volunteered: the schedule list deliberately
        # leaves full days out so nobody is invited to pick one, and
        # this status is what comes back when they ask anyway.
    {"status": "not_found"}  # the doctor does not work that weekday here at all
        # Say EXACTLY that - the doctor has no clinic on that weekday at
        # this branch - and then, in the SAME turn, call
        # `list_available_days_for_booking` and show the days they DO
        # have. Never answer a named day by quietly showing the soonest
        # date as if the patient had not named one.
    {"status": "unrecognized_day", "weekday_text": "..."}
        # `weekday_name` was not a day of the week at all. Ask which day
        # they meant - do NOT guess one, and do NOT fall through to
        # showing the soonest date.
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "not_configured"} / {"status": "error"}

    WEEKDAY SPELLING: pass the patient's own word through unchanged if
    you like - Egyptian/Gulf colloquial ("التلات", "الاتنين", "الحد"),
    MSA ("الثلاثاء"), English ("Tuesday"/"tue") and franco-arabe
    ("eltalat") all resolve. You never need to translate or "correct"
    the day name before calling."""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}

    target_weekday = resolve_weekday_index(weekday_name)
    if target_weekday is None:
        # NOT "error" - the two need completely different handling.
        # "error" means the lookup itself broke and the patient should
        # be told something went wrong; THIS means the word was not
        # recognised as a day at all, and the only correct response is
        # to ask which day they meant. Falling back to "show the
        # soonest date" is exactly how a day the patient named used to
        # get silently discarded.
        logger.warning("resolve_available_day: unrecognized weekday_name=%r", weekday_name)
        return {"status": "unrecognized_day", "weekday_text": weekday_name}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("resolve_available_day called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    if not branch_id:
        # Try to auto-disambiguate: if this doctor's schedule shows the
        # requested weekday at only ONE of their branches, confirm that
        # branch automatically rather than asking - confirmed real
        # production frustration where the model kept asking "which
        # branch?" despite the schedule it had ALREADY shown uniquely
        # determining the answer from the day the user just named.
        #
        # `effective_date`/`include_future` applied here for the same
        # reason as every other schedule lookup in this file: without
        # them, a branch whose assignment for this doctor has already
        # LAPSED can still get auto-confirmed as "the" branch for a
        # weekday, on the strength of a row that no longer applies.
        disambiguation_effective_date = None
        try:
            disambiguation_timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
            disambiguation_effective_date = datetime.now(ZoneInfo(disambiguation_timezone_name)).date().isoformat()
        except Exception:
            logger.exception("resolve_available_day: failed to compute today's date for the branch-disambiguation lookup")

        schedule_result = api.get_doctor_schedule(
            base_url, doctor_ids=[doctor_id],
            effective_date=disambiguation_effective_date, include_future=True,
            language=conversation_language(state),
        )
        if schedule_result["success"]:
            english_weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            target_name_en = english_weekday_names[target_weekday]
            matching_branch_ids = set()
            for item in (schedule_result["data"] or {}).get("items", []):
                raw_days = item.get("recurringDaysNames") or []
                if isinstance(raw_days, str):
                    raw_days = [raw_days]
                for raw_day in raw_days:
                    day_token = str(raw_day).split(":")[-1].strip()
                    if day_token.lower() == target_name_en.lower() and item.get("branchId"):
                        matching_branch_ids.add(item.get("branchId"))
            if len(matching_branch_ids) == 1:
                branch_id = next(iter(matching_branch_ids))
                session["branch_id"] = branch_id
                logger.info("resolve_available_day: auto-resolved branch_id=%s from weekday=%s (unique match in doctor's schedule)", branch_id, weekday_name)
                try:
                    branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
                    if branches_result["success"]:
                        match = next((b for b in (branches_result["data"] or {}).get("items", []) if b.get("id") == branch_id), None)
                        if match:
                            session["branch_display_name"] = _arabic_preferred_name(match)
                except Exception:
                    logger.exception("resolve_available_day: failed to enrich auto-resolved branch name")
        else:
            logger.error("resolve_available_day: schedule lookup for branch auto-disambiguation failed: status_code=%s error=%s", schedule_result.get("status_code"), schedule_result.get("error"))

    if not branch_id:
        return {"status": "missing_branch"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now = datetime.now(tz)
    horizon_days = 42  # matches the confirmed production booking window
    from_date = now.isoformat()
    to_date = (now + timedelta(days=horizon_days)).isoformat()

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=from_date, to_date=to_date, is_booked=False, page_size=1000,
     language=conversation_language(state),)

    if not result["success"]:
        logger.error("resolve_available_day API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    logger.info(
        "resolve_available_day: doctor_id=%s branch_id=%s weekday=%s after_date=%r from_date=%s to_date=%s api_returned=%d",
        doctor_id, branch_id, weekday_name, after_date, from_date, to_date, len(items),
    )

    # NAIVE, to match the wall-clock slot times it is compared against.
    # `now` stays timezone-aware because the API request params use it;
    # only the comparison value is stripped. Mixing the two raises
    # TypeError - CONFIRMED REAL PRODUCTION CRASH: "can't compare
    # offset-naive and offset-aware datetimes" at `dt <= lead_time`,
    # which took down the whole turn.
    lead_time = now.replace(tzinfo=None) + timedelta(hours=12)  # 12h minimum advance booking lead
    after_dt = None
    if after_date:
        try:
            after_dt = date.fromisoformat(after_date.strip())
        except ValueError:
            after_dt = None

    candidates = []
    for item in items:
        if item.get("isBooked"):
            continue
        slot_start_local = to_local_wallclock(item.get("slotStart"), timezone_name)
        if not slot_start_local:
            continue
        try:
            dt = datetime.fromisoformat(slot_start_local)
        except ValueError:
            continue
        if dt <= lead_time:
            continue
        if dt.weekday() != target_weekday:
            continue
        if after_dt and dt.date() <= after_dt:
            continue
        candidates.append(dt)

    if not candidates:
        logger.info(
            "resolve_available_day: not_found - %d raw items, none matched (weekday=%s, lead_time=%s, after_date=%s). Sample raw items: %s",
            len(items), weekday_name, lead_time.isoformat(), after_dt, items[:3],
        )

        # WHY the day is unavailable decides what the patient is told.
        #
        # If the doctor is ROSTERED on this weekday at this branch and
        # there is simply nothing left, that is "fully booked" - a real,
        # useful answer they can act on (ask for another day, or a later
        # date). It is also the only moment that fact should ever be
        # volunteered: the schedule list deliberately hides full days so
        # nobody is invited to pick one, and this is the branch reached
        # when they ask about that day anyway.
        #
        # If the doctor does not work that weekday at all, "not_found"
        # stays - a different thing, and saying "fully booked" would
        # falsely imply the day normally exists.
        rostered = _doctor_works_weekday(
            state, base_url, doctor_id, branch_id, target_weekday,
        )

        if rostered:
            return {
                "status": "fully_booked",
                "weekday_name": _ENGLISH_WEEKDAY_BY_INDEX.get(target_weekday, ""),
                "weekday_display": _display_weekday_name(target_weekday, conversation_language(state)),
            }

        # THE WEEKDAY RIDES BACK EVEN ON A MISS.
        #
        # The reply to a "not_found" has to NAME the day - "الدكتور ما
        # عنده عيادة يوم الثلاثاء" - and graph.py's
        # `_reply_invents_availability` verifier flags any weekday in a
        # reply that appears in no availability-tool result. A bare
        # {"status": "not_found"} therefore made the one correct answer
        # to this situation look like a fabricated day, and the verifier
        # would reject it twice and fall through to the generic error
        # message. Echoing the resolved day back keeps the check honest
        # and costs nothing.
        return {
            "status": "not_found",
            "weekday_name": _ENGLISH_WEEKDAY_BY_INDEX.get(target_weekday, ""),
            "weekday_display": _display_weekday_name(target_weekday, conversation_language(state)),
        }

    candidates.sort()
    chosen_dt = candidates[0]
    chosen_date = chosen_dt.date()
    english_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_weekday]
    logger.info("resolve_available_day: found date=%s (weekday=%s) from %d candidate(s)", chosen_date.isoformat(), english_name, len(candidates))

    # The EARLIEST and LATEST open slot start times on chosen_date, so
    # the day can be offered as a RANGE ("من 11:00 صباحًا إلى 3:00
    # مساءً") instead of the model latching onto the single nearest slot
    # and presenting a 30-minute window as if that were the only option.
    # No extra API call needed - `candidates` already holds every open
    # slot on this weekday across the whole window (it's how chosen_date
    # itself was picked), so this is just a filter over data already in
    # hand.
    #
    # CONFIRMED REAL PRODUCTION CONFUSION this fixes: the patient was
    # told the nearest appointment was "من 11:00 إلى 11:30" - the
    # nearest SLOT's own start/end, not the day's actual availability
    # (11:00 صباحًا to 3:00 مساءً) - which reads as if that one narrow
    # window were the whole offer.
    same_day_candidates = [dt for dt in candidates if dt.date() == chosen_date]
    first_time_dt = same_day_candidates[0] if same_day_candidates else chosen_dt
    last_time_dt = same_day_candidates[-1] if same_day_candidates else chosen_dt

    day_start = datetime.combine(chosen_date, datetime.min.time(), tzinfo=chosen_dt.tzinfo)
    day_end = datetime.combine(chosen_date, datetime.max.time().replace(microsecond=0), tzinfo=chosen_dt.tzinfo)

    language = conversation_language(state)

    # DISPLAY FIELDS, in the conversation's own language.
    #
    # This tool used to return `date` (a bare ISO "2026-08-25") and
    # `weekday_name` ("Tuesday") and NOTHING a reply could show as-is.
    # The model, correctly told never to reformat a tool's values, then
    # printed the ISO string straight into an Arabic sentence -
    # confirmed in production: "الثلاثاء 2026-08-25". Every OTHER
    # date-bearing tool in this file already returns `date_display` /
    # `weekday_display`, so the same date appeared in two different
    # formats depending only on which tool happened to fetch it.
    #
    # `date` and `weekday_name` are kept exactly as they were: they are
    # the MACHINE values, and `from_date`/`to_date` get passed verbatim
    # into the next call. The display fields are additions, not
    # replacements.
    return {
        "status": "found",
        "date": chosen_date.isoformat(),
        "weekday_name": english_name,
        "date_display": _display_date(chosen_dt.isoformat()),
        "weekday_display": _display_weekday(chosen_dt.isoformat(), language),
        "first_time_display": _display_time_12h(first_time_dt.isoformat(), language),
        "last_time_display": _display_time_12h(last_time_dt.isoformat(), language),
        "from_date": day_start.isoformat(),
        "to_date": day_end.isoformat(),
    }


_ENGLISH_WEEKDAY_INDEX = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _weekdays_with_open_slots(state, base_url: str, doctor_id: str, branch_id: str,
                              timezone_name: str, window_days: int):
    """Which weekdays this doctor still has an OPEN slot on, at this
    branch, within the booking window.

    Returns a set of Python weekday indexes (Monday=0), or None when the
    lookup failed - None means "unknown", and the caller must not mark
    anything as full on the strength of it.

    ONE call per branch, covering the whole window, rather than one per
    day.
    """

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now_local = datetime.now(tz)
    window_end = now_local + timedelta(days=window_days)

    slots = _open_slots_on_day(
        state, base_url, doctor_id, branch_id,
        now_local.isoformat(), window_end.isoformat(), timezone_name,
    )

    if slots is None:
        return None

    return {slot.weekday() for slot in slots}


_ARABIC_WEEKDAY_BY_INDEX = {
    0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس",
    4: "الجمعة", 5: "السبت", 6: "الأحد",
}

_ENGLISH_WEEKDAY_BY_INDEX = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}


def _display_weekday_name(weekday_index: int, language: str = "ar") -> str:
    """A weekday's name in the conversation's language, from its index."""

    if (language or "ar").startswith("en"):
        return _ENGLISH_WEEKDAY_BY_INDEX.get(weekday_index, "")
    return _ARABIC_WEEKDAY_BY_INDEX.get(weekday_index, "")


def _doctor_works_weekday(state, base_url: str, doctor_id: str, branch_id: str,
                          weekday_index: int) -> bool:
    """Whether this doctor's ROSTER includes that weekday at that branch.

    Used to tell "fully booked" (rostered, nothing left) apart from "the
    doctor doesn't work that day" - two different answers, and saying
    the wrong one either invents a day that never existed or hides a
    real one behind a vague refusal.

    Returns False when the lookup fails, so a transient error produces
    the more conservative "not_found" rather than asserting the day is
    full.
    """

    try:
        # `effective_date`/`include_future` for the same reason as every
        # other schedule lookup here: without them, an assignment that
        # LAPSED weeks ago still counts as "she works this day" and this
        # would answer "fully_booked" (implying an active rota, just
        # full) for a weekday she no longer works at all - "not_found"
        # is the honest answer in that case.
        today_iso = None
        try:
            timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
            today_iso = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            logger.exception("_doctor_works_weekday: failed to compute today's date")

        result = api.get_doctor_schedule(
            base_url, doctor_ids=[doctor_id],
            branch_ids=[branch_id] if branch_id else None,
            effective_date=today_iso, include_future=True,
            language=conversation_language(state),
        )
        if not result["success"]:
            return False

        for item in (result["data"] or {}).get("items", []):
            for name in item.get("recurringDaysNames") or []:
                if _ENGLISH_WEEKDAY_INDEX.get(str(name).strip().lower()) == weekday_index:
                    return True
    except Exception:
        logger.exception("_doctor_works_weekday: roster lookup failed")

    return False


def _mark_fully_booked_schedule_days(state, base_url: str, doctor_id: str,
                                     schedules: list, timezone_name: str) -> list:
    """Flag schedule rows whose weekday has no open slot left.

    WHY: a schedule row is a ROSTER entry - "this doctor works Thursdays
    here". It says nothing about whether any Thursday slot is still
    free. Confirmed in production: a doctor's Thursday rota was
    presented as available while every Thursday slot had already been
    taken by other patients, so the patient was walked forward into a
    day that could not be booked.

    A row is marked ONLY when its weekday actually occurs inside the
    booking window and has nothing open. A rota that has not STARTED yet
    (the clinic has published it for a later period) is left unmarked -
    "not open yet" is a different thing from "full", and the doctor may
    well be taking bookings for it. That distinction is why this cannot
    simply mark every weekday the sweep didn't see.
    """

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    today = datetime.now(tz).date()

    open_weekdays_by_branch = {}

    for row in schedules:
        branch_id = row.get("branchId")
        if not branch_id:
            continue

        names = row.get("recurringDaysNames") or []
        indexes = [
            _ENGLISH_WEEKDAY_INDEX[str(n).strip().lower()]
            for n in names
            if str(n).strip().lower() in _ENGLISH_WEEKDAY_INDEX
        ]
        if not indexes:
            continue

        starts_on = _schedule_row_effective_from(row)

        if branch_id not in open_weekdays_by_branch:
            open_weekdays_by_branch[branch_id] = _weekdays_with_open_slots(
                state, base_url, doctor_id, branch_id, timezone_name,
                DOCTOR_AVAILABILITY_WINDOW_DAYS,
            )

        open_weekdays = open_weekdays_by_branch[branch_id]
        if open_weekdays is None:
            # Lookup failed - unknown is not full.
            continue

        if any(index in open_weekdays for index in indexes):
            # Something is open on this weekday - nothing to flag.
            continue

        # A rota the clinic has published for a FUTURE period is left
        # alone. Publishing it IS opening it - the patient can book
        # against it, and the sweep simply hasn't reached that far or
        # the slots are generated closer to the date. Flagging it in any
        # way would discourage a booking that is perfectly possible.
        #
        # Only a rota that is in effect RIGHT NOW and has nothing open
        # is genuinely full.
        if starts_on and starts_on > today:
            continue

        row["fully_booked"] = True
        logger.info(
            "_mark_fully_booked_schedule_days: %s at branch %s is in effect but has no open "
            "slot in the next %d days - marking it fully booked",
            names, branch_id, DOCTOR_AVAILABILITY_WINDOW_DAYS,
        )

    return schedules


def _parse_iso_date(value):
    """A date from an ISO-ish string, or None."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


# THE FIELD NAME BELOW HAS NEVER BEEN CONFIRMED AGAINST A REAL RESPONSE.
#
# Every other field this file reads off a DoctorSchedules row -
# `recurringDaysNames`, `fromDateTime`, `toDateTime`, `branchId` - is
# marked in api.py as "confirmed directly from the API's real response".
# The validity-range field (the one behind the admin panel's "Effective
# From" / "Effective To" columns) never got that same confirmation. It
# was guessed as "effectiveFrom" when the future-rota exemption was
# first written, and every fully-booked check since has quietly
# inherited that same unverified guess.
#
# CONFIRMED SUSPECT: a doctor's branch was reported "fully booked" in
# production while her own admin-panel schedule showed a genuinely
# upcoming rota at that same branch (Effective From a later date). If
# the real field is spelled differently, `_parse_iso_date` above always
# returns None for it, `starts_on` is always None, and the future-rota
# exemption this file relies on in three places never fires - it looks
# like working code and silently does nothing.
#
# This checks every plausible spelling rather than one, and - the part
# that actually closes the question - logs the row's own keys the FIRST
# time none of them match, once per process. The next production log
# will show exactly what the field is really called, ending the
# guessing rather than extending it.
_EFFECTIVE_FROM_CANDIDATE_KEYS = (
    "effectiveFrom", "fromDateTimeFrom", "effectiveDate", "effectiveFromDate",
    "validFrom", "startDate", "fromDate", "scheduleFrom", "startEffectiveDate",
    # ANSWERED, 2026-08-31: the warning below finally fired on a real
    # medtown row and printed its actual keys. There is no
    # "effectiveFrom"-style field on this API at all - the row's
    # validity window is `fromDateTime`/`toDateTime`, both of which
    # api.py already marks as confirmed against a real response
    # (observed: fromDateTime=2026-08-31T12:30:00+00:00,
    # toDateTime=2026-09-30T19:30:00+00:00). Its DATE is therefore the
    # "Effective From" the admin panel shows, and the future-rota
    # exemption - dead in production until now, since every candidate
    # above returned None - can finally fire.
    #
    # Deliberately LAST: if any deployment does expose a dedicated
    # field under one of the names above, that stays authoritative and
    # this fallback is never reached.
    "fromDateTime",
)

_effective_from_field_unknown_logged = False


def _schedule_row_effective_from(row: dict):
    """The date this schedule row's validity BEGINS, or None if no
    candidate field name matched - see the module-level note above for
    why this is a temporary, defensive lookup rather than a single
    trusted key."""

    global _effective_from_field_unknown_logged

    for key in _EFFECTIVE_FROM_CANDIDATE_KEYS:
        parsed = _parse_iso_date(row.get(key))
        if parsed:
            return parsed

    if not _effective_from_field_unknown_logged and row:
        _effective_from_field_unknown_logged = True
        logger.warning(
            "_schedule_row_effective_from: none of %s matched on a real schedule row - "
            "the future-rota exemption cannot fire until the correct field name is known. "
            "Row's actual keys: %s",
            _EFFECTIVE_FROM_CANDIDATE_KEYS, sorted(row.keys()),
        )

    return None


def _branches_with_real_availability(state, base_url: str, doctor_id: str, branch_options: list,
                                     future_branch_ids: Optional[set] = None) -> list:
    """Mark each branch with whether the doctor actually has anything
    bookable there. Returns the SAME branches, annotated - never fewer.

    A schedule row means ROSTERED, not AVAILABLE: the roster can be
    full, or the schedule can have lapsed. Each candidate is therefore
    checked with the same slot query the next step will run, and the
    ones with nothing open get `fully_booked: True`.

    `future_branch_ids`: branches with a rota that has not STARTED yet
    (effectiveFrom in the future) are NEVER marked fully booked,
    regardless of what the slot sweep finds - publishing a rota does not
    guarantee the underlying system has already generated bookable slots
    for it. See `_branches_with_real_slots` for the confirmed production
    failure this prevents; this is the same exemption, for the sibling
    function used by `list_available_days_for_booking`'s branch list.

    WHY ANNOTATE RATHER THAN DROP: an earlier version removed them
    outright, which was wrong in two ways at once. The patient could see
    from the doctor's own schedule that she works Thursdays at الدقي,
    and the assistant then behaved as though that branch did not exist -
    "ما لقيت فرع اسمه الدقي" - which is both false and impossible to
    argue with. And it withholds the one fact that actually helps: the
    branch is right, the doctor is right, the slots are simply taken.
    A patient told "fully booked" can ask about a later date; a patient
    told the branch doesn't exist can only give up.

    A branch whose check FAILS (transient API error) is left unmarked -
    unknown is not the same as full, and the next step surfaces the
    truth anyway.
    """

    if len(branch_options) <= 1:
        # Nothing to choose between - the caller's single-branch
        # auto-confirm path already handled that case.
        return branch_options

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now_local = datetime.now(tz)
    window_end = now_local + timedelta(days=DOCTOR_AVAILABILITY_WINDOW_DAYS)
    future_branch_ids = future_branch_ids or set()

    for option in branch_options:
        branch_id = option.get("id")
        if not branch_id:
            continue

        if branch_id in future_branch_ids:
            # A not-yet-started rota exists here - never call this full,
            # regardless of what the sweep below would have found.
            continue

        slots = _open_slots_on_day(
            state, base_url, doctor_id, branch_id,
            now_local.isoformat(), window_end.isoformat(), timezone_name,
        )

        if slots is None:
            logger.info(
                "_branches_with_real_availability: availability check failed for branch_id=%s - "
                "leaving it unmarked (unknown is not the same as full)",
                branch_id,
            )
        elif not slots:
            option["fully_booked"] = True
            logger.info(
                "_branches_with_real_availability: branch %r (%s) is rostered but has no open "
                "slots in the window - marking it fully booked",
                option.get("name"), branch_id,
            )

    return branch_options


def _open_slots_on_day(state, base_url: str, doctor_id: str, branch_id: str,
                       from_iso: str, to_iso: str, timezone_name: str):
    """Open slot start times for ONE day, fetched with exactly the same
    query `get_available_slots_for_booking` uses.

    Returns a sorted list of local datetimes, [] when the day genuinely
    has nothing open, or None when the check itself failed (transient
    API error) - None means "unknown", and the caller keeps the day
    rather than hiding real availability because of a network blip.
    """

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=from_iso, to_date=to_iso, is_booked=False, page_size=200,
        language=conversation_language(state),
    )

    if not result["success"]:
        logger.warning(
            "_open_slots_on_day: verification lookup failed for %s (status_code=%s) - "
            "keeping the day rather than discarding it on a transient error",
            from_iso, result.get("status_code"),
        )
        return None

    # Naive, to match the wall-clock slot times - see _local_now_naive.
    now_local = _local_now_naive(timezone_name)

    starts = []
    for item in (result["data"] or {}).get("items", []):
        if item.get("isBooked") is True:
            continue
        local = to_local_wallclock(item.get("slotStart"), timezone_name)
        if not local:
            continue
        try:
            dt = datetime.fromisoformat(local)
        except ValueError:
            continue
        if dt > now_local:
            starts.append(dt)

    return sorted(starts)


@tool
def list_available_days_for_booking(
    state: Annotated[AgentState, InjectedState],
    limit: int = 3,
    offset: int = 0,
) -> dict:
    """For a NEW BOOKING: list the doctor's REAL upcoming days that
    actually have open slots, each with its actual calendar date. Reads
    doctor_id/branch_id from the booking session automatically.

    CALL THIS IMMEDIATELY AFTER A DOCTOR IS CONFIRMED - it replaces
    asking the patient which day they want before they have any idea
    when the doctor works. Patients do not know a doctor's schedule;
    asking them to name a day first means they guess, hit a day the
    doctor doesn't work or that's fully booked, and get stuck.

    This is different from `get_doctor_schedule_for_booking`, which only
    returns the doctor's GENERAL recurring weekdays with no dates and no
    guarantee anything is actually free. Every day returned here is
    confirmed to have at least one genuinely open slot, so you can show
    its date directly without any further checking.

    SHOW THE NEAREST FEW DAYS: `limit` defaults to 3. When the doctor
    genuinely has more than one day open at this branch, show them as a
    numbered list and let the patient pick - they can then choose the
    day that actually suits them in ONE message instead of rejecting a
    single offered date and waiting for the next one. When only one day
    is open, that single date is shown on its own and the patient is
    simply asked whether it suits them.

    ONE DATE PER WEEKDAY. The days returned are always DIFFERENT days of
    the week - the doctor's actual working days, each at its own soonest
    date. A weekly clinic can never come back as "Monday 24/08, Monday
    31/08, Monday 07/09": that is one option printed three times, and it
    is filtered out here rather than left for you to notice. So a doctor
    who works only Mondays returns exactly ONE day, and you should
    present it as the soonest available date rather than as a list.

    If none of them suit ("مش مناسب", "معاد أبعد", "في مواعيد تانية؟")
    call this AGAIN with the result's own `next_offset` to show the
    following few. Never invent or calculate a date yourself, and never
    dump the whole window on the first reply.

    `has_more` in the response tells you whether further days exist
    beyond the ones returned, so you can say so honestly instead of
    implying these are the only dates the doctor has.

    Returns:
    {"status": "found", "days": [{"date": "2026-08-11", "weekday_name": "Tuesday",
      "weekday_display": "الثلاثاء", "date_display": "11/08/2026", "slotCount": 7,
      "firstTime": "10:15 صباحًا", "lastTime": "11:45 صباحًا",
      "from_date": ..., "to_date": ...}, ...],
     "has_more": true, "total_available_days": 9, "next_offset": 3}
    {"status": "not_found"}  # this doctor has no open slot at all in the booking window
    {"status": "no_more_days"}  # `offset` is past the last available day
    {"status": "missing_doctor"}
    {"status": "missing_branch", "branches": [{"name": ...}, ...]}
        # This doctor works at MORE THAN ONE branch, so which one they
        # want must be settled first - their days and times differ per
        # branch. `branches` lists that doctor's real branches: show
        # those names, ask which one, confirm it with
        # `match_entity_for_booking(entity_type="branch")`, then call
        # this again. Never name a branch that isn't in this list.
        # A doctor working at only ONE branch never returns this - the
        # branch is confirmed silently and the days come back directly.
    {"status": "not_configured"} / {"status": "error"}

    Pass a chosen day's `from_date`/`to_date` VERBATIM into
    `get_available_slots_for_booking` - never retype or recompute them."""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("list_available_days_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    if not branch_id:
        # Same auto-disambiguation resolve_available_day does: if this
        # doctor works at exactly one branch, there is no real choice to
        # make, so don't manufacture a question about it.
        #
        # `effective_date`/`include_future` for the same reason as every
        # other schedule lookup in this file: without them, a branch
        # whose assignment has already LAPSED can still be the "only"
        # branch found here and get auto-confirmed, when she does not
        # currently work there at all.
        auto_confirm_effective_date = None
        try:
            timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
            auto_confirm_effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            logger.exception("list_available_days_for_booking: failed to compute today's date for branch auto-confirm")

        schedule_result = api.get_doctor_schedule(
            base_url, doctor_ids=[doctor_id],
            effective_date=auto_confirm_effective_date, include_future=True,
            language=conversation_language(state),
        )
        if schedule_result["success"]:
            schedule_items = (schedule_result["data"] or {}).get("items", [])
            branch_ids = {s.get("branchId") for s in schedule_items if s.get("branchId")}
            if len(branch_ids) == 1:
                branch_id = next(iter(branch_ids))
                session["branch_id"] = branch_id
                # Record the NAME too, not just the id. The booking
                # confirmation message prints the branch from
                # branch_display_name, so auto-confirming the id alone
                # left it blank - confirmed real production failure: a
                # confirmation went out reading "🏥 الفرع:" with nothing
                # after it, while every other line was filled in.
                if not session.get("branch_display_name"):
                    display_name = next(
                        (s.get("branchName") for s in schedule_items if s.get("branchId") == branch_id and s.get("branchName")),
                        None,
                    )
                    try:
                        branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
                        if branches_result["success"]:
                            match = next(
                                (b for b in (branches_result["data"] or {}).get("items", []) if b.get("id") == branch_id),
                                None,
                            )
                            if match:
                                display_name = _arabic_preferred_name(match) or display_name
                    except Exception:
                        logger.exception("list_available_days_for_booking: failed to enrich auto-confirmed branch name")
                    session["branch_display_name"] = display_name
                logger.info(
                    "list_available_days_for_booking: auto-confirmed single branch_id=%s (%s) for doctor_id=%s",
                    branch_id, session.get("branch_display_name"), doctor_id,
                )

    if not branch_id:
        # Hand back the actual branches this doctor works at, not just a
        # bare "missing_branch". Without them the model has to either ask
        # a vague "أي فرع تفضل؟" or - worse - name branches from its own
        # memory that this doctor may not work at.
        branch_options = []
        try:
            if schedule_result and schedule_result.get("success"):
                branches_lookup = {}
                branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
                if branches_result["success"]:
                    branches_lookup = {b.get("id"): b for b in (branches_result["data"] or {}).get("items", []) if b.get("id")}
                seen = set()
                future_branch_ids = set()
                for item in (schedule_result["data"] or {}).get("items", []):
                    item_branch_id = item.get("branchId")
                    if not item_branch_id:
                        continue

                    # A branch can have MULTIPLE schedule rows (e.g. an
                    # expiring Thursday and a future-starting Monday).
                    # This has to be checked on EVERY row, not just the
                    # first one seen per branch - otherwise a branch
                    # whose only future rota appears on its second row
                    # would never be exempted below.
                    starts_on = _schedule_row_effective_from(item)
                    if starts_on and starts_on > date.today():
                        future_branch_ids.add(item_branch_id)

                    if item_branch_id in seen:
                        continue
                    seen.add(item_branch_id)
                    match = branches_lookup.get(item_branch_id)
                    name = (_arabic_preferred_name(match) if match else None) or item.get("branchName")
                    if name:
                        # THE id IS NOT OPTIONAL.
                        #
                        # These options were previously {"name": ...}
                        # only, and the list was never remembered - so
                        # when the patient answered with its NUMBER, there
                        # was nothing to resolve the position against.
                        # Confirmed in production: the patient typed "١",
                        # then "٢", and both times got "عذرًا، ما قدرت
                        # أتعرف على الفرع اللي اخترته" followed by the
                        # SAME list again - a dead end they could only
                        # escape by typing the branch name.
                        branch_options.append({"id": item_branch_id, "name": name})

                # MARK WHICH ONES ARE ACTUALLY BOOKABLE.
                #
                # A schedule row means the doctor is ROSTERED at that
                # branch, not that anything is open there. Each candidate
                # is checked with the same availability query the next
                # step will run, and the full ones are FLAGGED - not
                # removed. See _branches_with_real_availability for why
                # removing them was worse than offering them.
                branch_options = _branches_with_real_availability(
                    state, base_url, doctor_id, branch_options,
                    future_branch_ids=future_branch_ids,
                )
        except Exception:
            logger.exception("list_available_days_for_booking: failed to build branch options for missing_branch")

        if branch_options:
            logger.info(
                "list_available_days_for_booking: missing_branch for doctor_id=%s -> %d bookable branch option(s)",
                doctor_id, len(branch_options),
            )
            # Remembered so a bare "١"/"2" reply resolves by position
            # against the SAME ordering the patient was shown.
            _remember_list(state, "branch", branch_options)
            return {"status": "missing_branch", "branches": branch_options}

        logger.info(
            "list_available_days_for_booking: doctor_id=%s is rostered at branches but none "
            "have bookable availability right now",
            doctor_id,
        )
        return {"status": "missing_branch"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now = datetime.now(tz)
    horizon_days = 42  # same booking window resolve_available_day uses
    # Naive for the same reason as resolve_available_day's - it is
    # compared against wall-clock slot times.
    lead_time = now.replace(tzinfo=None) + timedelta(hours=12)  # same 12h minimum advance lead

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=now.isoformat(), to_date=(now + timedelta(days=horizon_days)).isoformat(),
        is_booked=False, page_size=1000,
     language=conversation_language(state),)

    if not result["success"]:
        logger.error("list_available_days_for_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    if len(items) >= 1000:
        # The sweep hit its page cap, so this is a TRUNCATED view of the
        # doctor's availability - days beyond the cut-off simply won't
        # appear. Worth knowing about when availability looks wrong.
        logger.warning(
            "list_available_days_for_booking: page cap reached (%d items) for doctor_id=%s - "
            "the 42-day availability sweep is truncated",
            len(items), doctor_id,
        )

    by_date: Dict[str, list] = {}

    for item in items:
        if item.get("isBooked"):
            continue

        slot_start_local = to_local_wallclock(item.get("slotStart"), timezone_name)
        if not slot_start_local:
            continue

        try:
            dt = datetime.fromisoformat(slot_start_local)
        except ValueError:
            continue

        if dt <= lead_time:
            continue

        by_date.setdefault(dt.date().isoformat(), []).append(dt)

    logger.info(
        "list_available_days_for_booking: doctor_id=%s branch_id=%s api_returned=%d bookable_days=%d",
        doctor_id, branch_id, len(items), len(by_date),
    )

    if not by_date:
        return {"status": "not_found"}

    english_weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    language = conversation_language(state)
    days = []

    all_dates = sorted(by_date)
    offset = max(0, offset)
    limit = max(1, limit)

    if offset >= len(all_dates):
        # The patient has already been shown every available day and
        # asked for more - say so plainly rather than silently repeating
        # the same list.
        return {"status": "no_more_days", "total_available_days": len(all_dates)}

    # VERIFY EACH DAY BEFORE OFFERING IT.
    #
    # The wide 42-day sweep above and the per-day lookup the next step
    # runs (`get_available_slots_for_booking`) are two different queries
    # against the same endpoint, and they have been observed to
    # DISAGREE: a day derived from the sweep was offered to a patient,
    # who accepted it, and the per-day lookup for that exact date then
    # returned zero slots - leaving the agent contradicting itself one
    # message after promising a date. The sweep is also capped at
    # page_size, so on a busy doctor it is a truncated view to begin
    # with.
    #
    # Rather than trusting the sweep, each candidate day is confirmed
    # with the SAME query the next step will run. A day that fails is
    # skipped, not shown. With `limit` at 3 this is normally three extra
    # calls (capped at MAX_VERIFY_CALLS either way), and it makes "here
    # is your appointment" something the next turn can actually honour.
    verified_calls = 0
    MAX_VERIFY_CALLS = 8
    consumed = 0

    # ONE DATE PER WEEKDAY.
    #
    # A doctor with a weekly clinic has the same appointment repeated
    # every seven days, so the nearest three dates are routinely
    # "الاثنين 24/08, الاثنين 31/08, الاثنين 07/09" - the same day of
    # the week three times over. That is not a choice; it is one option
    # printed three times, and it pushes the genuinely different days
    # the doctor works off the bottom of the list.
    #
    # Confirmed directly: this exact list was produced in production and
    # rejected. So a weekday already represented is skipped, and the
    # next DIFFERENT one is looked for instead. The patient sees the
    # doctor's actual working days ("الاثنين، الثلاثاء، السبت"), each at
    # its own soonest date, which is what they need in order to pick.
    #
    # A doctor who genuinely only works one weekday still yields exactly
    # one entry, and the single-day block is used - the same outcome the
    # old `limit=1` produced, arrived at because it is true rather than
    # by capping the list.
    seen_weekdays = set()

    for date_iso in all_dates[offset:]:
        if len(days) >= limit:
            break

        slot_times = sorted(by_date[date_iso])
        first = slot_times[0]

        if first.weekday() in seen_weekdays:
            consumed += 1
            continue

        day_start = datetime.combine(first.date(), datetime.min.time(), tzinfo=first.tzinfo)
        day_end = datetime.combine(first.date(), datetime.max.time().replace(microsecond=0), tzinfo=first.tzinfo)

        consumed += 1

        if verified_calls < MAX_VERIFY_CALLS:
            verified_calls += 1
            real_slots = _open_slots_on_day(
                state, base_url, doctor_id, branch_id,
                day_start.isoformat(), day_end.isoformat(), timezone_name,
            )

            if real_slots is not None and not real_slots:
                logger.warning(
                    "list_available_days_for_booking: day %s looked available in the 42-day sweep "
                    "but the per-day lookup returned no open slots - skipping it rather than "
                    "offering a date that cannot be booked",
                    date_iso,
                )
                continue

            if real_slots:
                # Describe the day from the per-day lookup, which is what
                # the patient will actually be shown next.
                slot_times = real_slots
                first = slot_times[0]

        seen_weekdays.add(first.weekday())

        days.append({
            "date": date_iso,
            "weekday_name": english_weekday_names[first.weekday()],
            "weekday_display": _display_weekday(first.isoformat(), language),
            "date_display": _display_date(first.isoformat()),
            "slotCount": len(slot_times),
            "firstTime": _display_time_12h(first.isoformat(), language),
            "lastTime": _display_time_12h(slot_times[-1].isoformat(), language),
            "from_date": day_start.isoformat(),
            "to_date": day_end.isoformat(),
        })

    if not days:
        logger.warning(
            "list_available_days_for_booking: none of the %d candidate day(s) from offset=%d had "
            "open slots when checked individually",
            len(all_dates) - offset, offset,
        )
        return {"status": "not_found" if offset == 0 else "no_more_days"}

    shown_through = offset + consumed

    # REMEMBER THE DAYS, exactly as they are about to be shown.
    #
    # Every other list this project displays is remembered so a bare
    # number resolves by POSITION - doctors, branches, services, slots.
    # Days were the one omission, so "1" had nothing to resolve against
    # and the model matched it from memory instead.
    #
    # CONFIRMED REAL PRODUCTION FAILURE: the patient rejected Tuesday,
    # was shown "1️⃣ الأحد 30/08 2️⃣ الاثنين 31/08 3️⃣ الثلاثاء 01/09",
    # replied "1" - and the booking was confirmed for الثلاثاء
    # 01/09/2026, the third option and the very day they had just
    # turned down.
    _remember_list(state, "day", days)

    return {
        "status": "found",
        "days": days,
        "total_available_days": len(all_dates),
        "has_more": shown_through < len(all_dates),
        "next_offset": shown_through,
    }


@tool
def create_new_booking(
    state: Annotated[AgentState, InjectedState],
    slot_start: str,
    slot_end: str,
    patient_full_name: str,
    mobile_number: str,
    email: str = "",
) -> dict:
    """Create a brand new appointment booking. Reads the confirmed
    doctor_id/branch_id from the booking session automatically - you
    never pass an ID. `slot_start`/`slot_end` MUST be the EXACT values
    from a `resolve_available_day` + slot-lookup step in THIS
    conversation - never modified, recomputed, or invented.

    CRITICAL SAFETY CHECK (always performed automatically, you don't
    need to do anything extra): before creating the booking, this tool
    RE-VERIFIES the exact requested slot is still genuinely available
    right now (someone else may have booked it in the meantime) - this
    is not optional and cannot be skipped.

    A doctor AND branch must both already be confirmed (via
    `match_entity_for_booking`) before calling this. Returns:
    {"status": "success", "booking_ref": "GBN-..."}
    {"status": "slot_unavailable"}  # the requested slot is no longer free - tell the user and offer to pick again
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "invalid_details", "rejected": [{"field": ..., "message": ...}]}
        # the booking system REFUSED one of the patient's own details
        # (e.g. field "MobileNumber" -> "Mobile Number Not Valid"). This
        # is NOT a technical fault and NOT worth retrying: tell the
        # patient plainly which detail wasn't accepted, ask for a
        # corrected one, and try the booking again with it. Never
        # describe this as a temporary technical problem.
    {"status": "not_configured"} / {"status": "error"}
    {"status": "phone_not_verified"}  # mobile_number isn't the channel
        # identity and hasn't been verified in this conversation (no
        # successful compare_phone match, no successful verify_otp). Go
        # complete that verification for this exact number BEFORE
        # calling this tool again - never retry as-is.
    {"status": "missing_patient_name"}  # patient_full_name is empty or
        # doesn't look like a real full name (at least two name parts).
        # Go ask the patient for their full name FIRST - never call
        # this tool with a placeholder, a single word, or an empty
        # string just to see what the API says."""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}
    if not branch_id:
        return {"status": "missing_branch"}

    # SERVER-SIDE ENFORCEMENT, NOT JUST A PROMPT RULE. STEP NB6 already
    # instructs asking for the patient's full name (at least two parts)
    # before ever reaching STEP NB7/this tool - but nothing here
    # actually checked that happened. CONFIRMED REAL PRODUCTION
    # FAILURE: this tool was called with an empty patient_full_name,
    # the API rejected it ("PatientFullName Required"), and the model
    # only THEN went back to ask for the name - a wasted round trip to
    # a real external booking API for a value that was never going to
    # be accepted. Catching it here, before the API call, is faster and
    # keeps the failure entirely within this project's own validation
    # rather than depending on the booking system's error message.
    name_parts = re.findall(r"[^\W\d_]{2,}", patient_full_name or "", re.UNICODE)
    if len(name_parts) < 2:
        logger.warning(
            "create_new_booking: refusing to book with patient_full_name=%r "
            "(session_id=%s) - not a real full name (need at least 2 parts)",
            patient_full_name, session_id,
        )
        return {"status": "missing_patient_name"}

    # SERVER-SIDE ENFORCEMENT, NOT JUST A PROMPT RULE - same reasoning
    # as `lookup_appointment`'s equivalent check. STEP NB6 in the
    # prompt already instructs compare_phone/send_otp+verify_otp for
    # any number that isn't the channel identity, but nothing in this
    # tool previously checked that actually happened before creating a
    # real appointment under that number - a prompt-following slip here
    # would book (and hand a real reference number for) an appointment
    # under a phone number nobody ever proved belonged to the person
    # messaging.
    if not _phone_is_verified(state, mobile_number):
        logger.warning(
            "create_new_booking: refusing to book for an unverified mobile_number "
            "(session_id=%s) - compare_phone/verify_otp must succeed first",
            session_id,
        )
        return {"status": "phone_not_verified"}


    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("create_new_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    # Re-verify: query the EXACT requested slot's own narrow time window
    # and confirm it's still isBooked=false right now, immediately before
    # creating the booking - someone else may have taken it since it was
    # first shown to the user. Never skip this and never trust an older
    # lookup from earlier in the conversation.
    #
    # IMPORTANT: query the FULL DAY containing the slot, not just the
    # slot's own narrow start/end window - a too-narrow range risks the
    # API's own date-range boundary filtering excluding the exact slot
    # (e.g. an inclusive/exclusive edge mismatch), even though it's a
    # real, bookable slot (confirmed suspicious in production: a slot
    # that was successfully booked via the website was reported
    # "unavailable" by this exact check). The precise timestamp match
    # below already correctly isolates the one exact slot regardless of
    # how many others come back in a wider window.
    try:
        requested_start_dt = datetime.fromisoformat(slot_start)
        day_start = requested_start_dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        day_end = requested_start_dt.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    except ValueError:
        logger.warning("create_new_booking: unparsable slot_start=%r", slot_start)
        return {"status": "error"}

    slots_result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=day_start, to_date=day_end, is_booked=False, page_size=200,
     language=conversation_language(state),)

    if not slots_result["success"]:
        logger.error("create_new_booking: re-verification API call failed: status_code=%s error=%s", slots_result.get("status_code"), slots_result.get("error"))
        return {"status": "error"}

    try:
        requested_ms = requested_start_dt.timestamp()
    except ValueError:
        logger.warning("create_new_booking: unparsable slot_start=%r", slot_start)
        return {"status": "error"}

    raw_items = (slots_result["data"] or {}).get("items", [])
    logger.info(
        "create_new_booking: re-verification doctor_id=%s branch_id=%s day_range=[%s, %s] requested_slot_start=%s api_returned=%d",
        doctor_id, branch_id, day_start, day_end, slot_start, len(raw_items),
    )

    matched_slot = None
    for item in raw_items:
        if item.get("isBooked"):
            continue
        try:
            item_ms = datetime.fromisoformat(item["slotStart"].replace("Z", "+00:00")).timestamp()
        except (ValueError, KeyError, AttributeError):
            continue
        if abs(item_ms - requested_ms) < 1:  # same instant
            matched_slot = item
            break

    if not matched_slot:
        logger.warning(
            "create_new_booking: requested slot %s not found or already booked (doctor_id=%s branch_id=%s). Raw slotStarts returned: %s",
            slot_start, doctor_id, branch_id, [i.get("slotStart") for i in raw_items][:20],
        )
        return {"status": "slot_unavailable"}

    # Normalize to E.164 at the API boundary. The channel identity
    # (WhatsApp's wa_id) arrives as bare digits - "201158877175" - and
    # the booking API rejects that shape outright with
    # MobileNumber -> "Mobile Number Not Valid", even though the number
    # itself is perfectly valid and the patient can do nothing about it.
    # Confirmed real production failure: a booking collected all the way
    # to confirmation, then failed at the final step for exactly this,
    # and the patient was told to try again later. Everything the user
    # sees already goes through normalize_phone_number; this is the one
    # place that was still passing the raw form through.
    normalized_mobile = normalize_phone_number(mobile_number, state) or mobile_number
    if normalized_mobile != mobile_number:
        logger.info(
            "create_new_booking: normalized mobile_number %r -> %r for the booking API",
            mobile_number, normalized_mobile,
        )

    result = api.create_booking(
        base_url,
        patient_full_name=patient_full_name,
        mobile_number=normalized_mobile,
        branch_id=matched_slot.get("branchId") or branch_id,
        doctor_id=matched_slot.get("doctorId") or doctor_id,
        service_id=matched_slot.get("serviceId"),
        service_price=matched_slot.get("servicePrice"),
        booking_time_from=slot_start,
        booking_time_to=slot_end,
        specialty_id=matched_slot.get("specialtyId"),
        doctor_schedule_id=matched_slot.get("scheduleId"),
        space_id=matched_slot.get("spaceId"),
        email=email,
    )

    if not result["success"]:
        details = result.get("details") or []
        logger.error(
            "create_new_booking API call failed: status_code=%s error=%s rejected_fields=%s",
            result.get("status_code"), result.get("error"),
            [d.get("field") for d in details] or "unknown",
        )
        # A field-level rejection (bad phone format, missing email, ...)
        # is NOT a transient technical fault: retrying later changes
        # nothing, and telling the patient to try again wastes their
        # time on something they could fix in one message. Pass the
        # rejected field(s) back so the reply can name what needs
        # correcting instead of blaming a generic outage.
        if result.get("error") == "validation_error" and details:
            return {"status": "invalid_details", "rejected": details}
        return {"status": "error"}

    new_booking_id = result["data"]
    booking_ref = None

    if new_booking_id:
        lookup_result = api.get_booking_by_id(base_url, new_booking_id)
        if lookup_result["success"]:
            booking_ref = (lookup_result["data"] or {}).get("bookingRefNum")

    # Booking complete - clear the session so a subsequent NEW booking
    # in the same conversation starts clean, matching the confirmed
    # production behavior (session auto-cleans on success).
    _BOOKING_SESSIONS.pop(session_id, None)

    return {"status": "success", "booking_ref": booking_ref}


@tool
def get_doctor_schedule_for_booking(
    state: Annotated[AgentState, InjectedState],
    target_date: str = "",
) -> dict:
    """For a NEW BOOKING: get the confirmed doctor's general recurring
    schedule (which weekdays they work, daily hours, and which branch
    each applies to). Reads doctor_id from the booking session
    automatically - a doctor must already be confirmed via
    `match_entity_for_booking` first.

    If a branch is ALSO already confirmed in the session, the schedule
    is automatically narrowed to that branch only. If no branch is
    confirmed yet, the schedule spans EVERY branch the doctor works at -
    group your reply by branch in that case (see the booking flow's own
    display instructions).

    `target_date` (format "YYYY-MM-DD"), if given, filters to only
    currently-effective schedule rows on that date; defaults to today.
    Returns:
    {"status": "found", "schedules": [{"recurringDaysNames": [...], "fromDateTime": ..., "toDateTime": ..., "branchName": ..., "doctorName": ...}, ...]}
    {"status": "not_found"} / {"status": "missing_doctor"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")

    if not doctor_id:
        return {"status": "missing_doctor"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_doctor_schedule_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    if target_date:
        effective_date = target_date
    else:
        try:
            effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            effective_date = None

    branch_id = session.get("branch_id")
    result = api.get_doctor_schedule(
        base_url, doctor_ids=[doctor_id],
        branch_ids=[branch_id] if branch_id else None,
        effective_date=effective_date,
        # Only a question about ONE specific date should be restricted to
        # rotas already in effect. A general "when does this doctor
        # work?" must include a rota the clinic has published for a later
        # period - see api.get_doctor_schedule.
        include_future=not target_date,
        language=conversation_language(state),)

    if not result["success"]:
        logger.error("get_doctor_schedule_for_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    # DIAGNOSTIC - SETTLES WHETHER include_future ACTUALLY WORKS.
    #
    # CONFIRMED REAL PRODUCTION DISCREPANCY: a doctor's admin-panel
    # schedule shows a genuine, currently-open Monday rota at a branch
    # (Effective From a date weeks away), alongside a Thursday rota
    # already in effect. Only the Thursday row showed up in this
    # function's result; Monday never appeared, in a call made with
    # `include_future=True` specifically so that it would.
    #
    # Either `include_future` isn't doing what `api.get_doctor_schedule`
    # believes it does, or something between the API and here drops the
    # row. This can only be settled by seeing the RAW response - which
    # this line makes visible in the log, once per call, at INFO level
    # (not a rare-condition WARNING, since the whole point is to see it
    # on every request until the question is closed).
    logger.info(
        "get_doctor_schedule_for_booking: doctor_id=%s effective_date=%s include_future=%s -> "
        "api returned %d raw row(s): %s",
        doctor_id, effective_date, not target_date, len(items),
        [
            {
                "branchId": it.get("branchId"), "branchName": it.get("branchName"),
                "recurringDaysNames": it.get("recurringDaysNames"),
                "fromDateTime": it.get("fromDateTime"), "toDateTime": it.get("toDateTime"),
            }
            for it in items
        ],
    )

    if not items:
        return {"status": "not_found"}

    # Auto-confirm the branch when every schedule row shares the SAME
    # single branch - a code-level fix, since the prose instruction to
    # "silently confirm and don't ask" was repeatedly not followed in
    # production (the model kept asking about branch even when there
    # was genuinely only one). This removes the LLM's role in the
    # decision entirely.
    if not session.get("branch_id"):
        distinct_branch_ids = {item.get("branchId") for item in items if item.get("branchId")}
        if len(distinct_branch_ids) == 1:
            only_branch_id = next(iter(distinct_branch_ids))
            only_branch_name = next((item.get("branchName") for item in items if item.get("branchId") == only_branch_id), None)
            session["branch_id"] = only_branch_id
            # Prefer a real Arabic altName if we can fetch it; fall back
            # to whatever name the schedule endpoint itself provided.
            try:
                branches_result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
                if branches_result["success"]:
                    match = next((b for b in (branches_result["data"] or {}).get("items", []) if b.get("id") == only_branch_id), None)
                    if match:
                        session["branch_display_name"] = _arabic_preferred_name(match)
                    else:
                        logger.warning(
                            "get_doctor_schedule_for_booking: branch_id=%s not found in the Branches list - "
                            "falling back to the schedule row's own branchName %r, which may be English",
                            only_branch_id, only_branch_name,
                        )
                else:
                    logger.warning(
                        "get_doctor_schedule_for_booking: get_branches failed (status_code=%s error=%s) - "
                        "falling back to the schedule row's own branchName %r, which may be English",
                        branches_result.get("status_code"), branches_result.get("error"), only_branch_name,
                    )
            except Exception:
                logger.exception("get_doctor_schedule_for_booking: failed to enrich auto-confirmed branch name")
            if not session.get("branch_display_name"):
                session["branch_display_name"] = only_branch_name
            if conversation_language(state) != "en" and not _looks_arabic_text(session.get("branch_display_name") or ""):
                # Not fatal, but it is exactly the shape that used to
                # destroy an entire reply via graph.py's mixed-language
                # greeting guard, and it reads as unprofessional either
                # way - so make it visible rather than silent.
                logger.warning(
                    "get_doctor_schedule_for_booking: branch_display_name=%r has no Arabic characters "
                    "while this conversation is in Arabic (branch_id=%s)",
                    session.get("branch_display_name"), only_branch_id,
                )
            logger.info("get_doctor_schedule_for_booking: auto-confirmed single branch_id=%s (%s) for doctor_id=%s", only_branch_id, session.get("branch_display_name"), doctor_id)

    doctor_display_name = session.get("doctor_display_name")
    branch_display_name = session.get("branch_display_name")

    schedules = [
        {
            "recurringDaysNames": item.get("recurringDaysNames"),
            "fromDateTime": to_local_wallclock(item.get("fromDateTime"), timezone_name),
            "toDateTime": to_local_wallclock(item.get("toDateTime"), timezone_name),
            "branchName": branch_display_name if (branch_display_name and item.get("branchId") == session.get("branch_id")) else item.get("branchName"),
            "branchId": item.get("branchId"),
            "doctorName": doctor_display_name or item.get("doctorName"),
            # The service NAME only ("كشف رمد") - never its price. Fees
            # stay private until `get_doctor_fees` is called on an
            # explicit request; see prompts.py's FEES rule.
            "serviceName": _service_name(item, conversation_language(state)),
            # Kept so the availability check below can group by branch
            # and tell a not-yet-started rota from a full one.
            # Resolved with _schedule_row_effective_from on the RAW item
            # (which still has every candidate field name available),
            # not a single guessed key - see that helper for why. Stored
            # under this one canonical name so the later fully-booked
            # check (which reads "effectiveFrom" first) finds it.
            "effectiveFrom": (lambda d: d.isoformat() if d else None)(_schedule_row_effective_from(item)),
        }
        for item in items
    ]

    # A roster entry is not availability. Rows whose weekday has nothing
    # open left are flagged, so the reply says "fully booked" instead of
    # walking the patient into a day they cannot book - see
    # _mark_fully_booked_schedule_days.
    try:
        schedules = _mark_fully_booked_schedule_days(
            state, base_url, doctor_id, schedules, timezone_name,
        )
    except Exception:
        logger.exception(
            "get_doctor_schedule_for_booking: availability marking failed - returning the "
            "schedule unmarked rather than failing the whole lookup"
        )

    return {"status": "found", "schedules": schedules}


@tool
def get_available_slots_for_booking(
    state: Annotated[AgentState, InjectedState],
    from_date: str,
    to_date: str,
) -> dict:
    """For a NEW BOOKING: get the confirmed doctor's ACTUAL open time
    slots (not just working hours) within [from_date, to_date] - both
    ISO format, e.g. "2026-05-01T09:00:00+03:00". Typically called with
    the exact from_date/to_date returned by `resolve_available_day`.
    Reads doctor_id AND branch_id from the booking session automatically
    - both must already be confirmed. Only genuinely available (not
    already booked) slots are returned. Returns:
    {"status": "found", "slots": [{"slotStart": ..., "slotEnd": ..., "date_display": ..., "time_display": ..., "serviceName": ...}, ...]}
    {"status": "not_found"}  # no open slots in this range
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}
    if not branch_id:
        return {"status": "missing_branch"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_available_slots_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    # Safety net: swap an inverted range - confirmed real production bug
    # for the same underlying endpoint (see get_available_reschedule_slots).
    try:
        if from_date and to_date and datetime.fromisoformat(from_date) > datetime.fromisoformat(to_date):
            logger.warning("get_available_slots_for_booking: from_date=%r was AFTER to_date=%r - swapping them", from_date, to_date)
            from_date, to_date = to_date, from_date
    except ValueError:
        pass

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=from_date, to_date=to_date, is_booked=False, page_size=200,
     language=conversation_language(state),)

    if not result["success"]:
        logger.error("get_available_slots_for_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if items:
        # Record that this booking has genuinely reached the "here are
        # the times" stage. get_patient_info refuses to run before this,
        # so the patient can never be asked for their phone number and
        # name for an appointment whose time doesn't exist yet.
        session["slots_shown"] = True
    logger.info(
        "get_available_slots_for_booking: doctor_id=%s branch_id=%s from_date=%s to_date=%s api_returned=%d",
        doctor_id, branch_id, from_date, to_date, len(items),
    )
    if not items:
        logger.info("get_available_slots_for_booking: not_found - API returned zero items for this range")
        return {"status": "not_found"}

    items_before_filter = len(items)
    items = [i for i in items if i.get("isBooked") is not True]
    if not items:
        logger.info("get_available_slots_for_booking: not_found - all %d item(s) were isBooked=True", items_before_filter)
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    language = conversation_language(state)
    slots = []
    for item in items:
        slot_start = to_local_wallclock(item.get("slotStart"), timezone_name)
        slot_end = to_local_wallclock(item.get("slotEnd"), timezone_name)
        slots.append({
            "slotStart": slot_start,
            "slotEnd": slot_end,
            "date_display": _display_date(slot_start),
            "weekday_display": _display_weekday(slot_start, language),
            "time_display": _display_time_12h(slot_start, language),
            "serviceId": item.get("serviceId"),
            "serviceName": _service_name(item, language),
            # servicePrice deliberately omitted - see the equivalent
            # comment in get_available_reschedule_slots.
        })

    # Exclude past slots, dedupe, sort, cap - same safeguards as the
    # reschedule flow's equivalent (all confirmed real production issues).
    try:
        now_local = _local_now_naive(timezone_name)
        slots = [s for s in slots if s["slotStart"] and datetime.fromisoformat(s["slotStart"]) > now_local]
    except Exception:
        logger.exception("get_available_slots_for_booking: failed to filter past slots, showing all")

    if not slots:
        logger.info("get_available_slots_for_booking: not_found - all slots were in the past relative to now")
        return {"status": "not_found"}

    slots.sort(key=lambda s: s["slotStart"] or "")

    seen_starts = set()
    deduped = []
    for s in slots:
        if s["slotStart"] in seen_starts:
            continue
        seen_starts.add(s["slotStart"])
        deduped.append(s)
    slots = deduped

    MAX_SLOTS_TO_SHOW = 20
    if len(slots) > MAX_SLOTS_TO_SHOW:
        slots = slots[:MAX_SLOTS_TO_SHOW]

    # REMEMBERED, THE SAME WAY DOCTOR AND BRANCH LISTS ALREADY ARE.
    #
    # This was the one remaining numbered list in the whole booking flow
    # left ENTIRELY to the model's own memory of the conversation - no
    # code ever recorded which slot corresponded to which number. Doctor
    # and branch picks had exactly this problem (passes 22-23: a
    # patient's numbered answer resolved against the wrong stored order,
    # or against nothing at all) before being fixed the same way.
    #
    # CONFIRMED REAL PRODUCTION FAILURE this enables fixing: a patient
    # picked slot "2", was asked to confirm their WhatsApp number, said
    # "yes" - and was then asked to give the appointment time again, as
    # if the selection had never happened. It hadn't been recorded
    # anywhere; it only ever existed as something the model had to recall
    # across an intervening phone-confirmation turn, and that recall
    # failed. See `select_appointment_slot` and
    # graph._build_selected_slot_directive for the two halves of the fix.
    _remember_list(state, "slot", slots)

    return {"status": "found", "slots": slots}


@tool
def select_appointment_slot(state: Annotated[AgentState, InjectedState], user_input: str) -> dict:
    """For a NEW BOOKING: resolve the patient's reply to ONE exact slot
    from the list `get_available_slots_for_booking` just showed, and
    LOCK IT IN for this booking - CALL THIS instead of matching the
    slot yourself from memory.

    `user_input`: the patient's raw reply - a bare number ("2", "٢"),
    or the exact time they typed back ("11:00", "11 الصبح").

    WHY THIS EXISTS: doctor and branch picks are resolved this same way,
    in code, against the exact list just shown - this was the one
    remaining numbered list left entirely to your own memory of the
    conversation. CONFIRMED REAL PRODUCTION FAILURE: a patient picked
    slot "2", was asked to confirm their WhatsApp number, said "yes" -
    and was then asked to give the time again, because nothing had
    actually recorded which slot "2" was; it only existed as something
    to recall several turns later, and that recall failed. Once this
    tool resolves a slot, it is saved on the booking session and stays
    there - you never need to re-derive it, including across the phone
    number question, and a directive will remind you of the exact
    values when it's time to call `create_new_booking`.

    Returns:
    {"status": "selected", "slot": {"slotStart", "slotEnd", "date_display",
     "weekday_display", "time_display", "serviceName"}}
        -> confirm it back in ONE short line and move on to STEP NB6 -
           never re-ask for the time now that this succeeded.
    {"status": "no_list_shown"}  # no slot list is remembered for this
        session - call `get_available_slots_for_booking` first, never
        guess a time.
    {"status": "out_of_range", "list_size": N}  # a number outside the
        list that was shown - tell them the valid range, don't guess.
    {"status": "not_matched"}  # their reply matches no remembered slot
        by position or by time - show the list again, or ask them to
        pick from it, never invent a slot to fill the gap."""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    last_list = session.get("last_list")

    if not last_list or last_list.get("entity_type") != "slot":
        logger.warning(
            "select_appointment_slot: no slot list is remembered for session_id=%s",
            session_id,
        )
        return {"status": "no_list_shown"}

    slots = last_list.get("items") or []

    position = _extract_selection_number(user_input)
    if position is not None:
        if not (1 <= position <= len(slots)):
            logger.warning(
                "select_appointment_slot: position %d out of range for %d remembered slot(s)",
                position, len(slots),
            )
            return {"status": "out_of_range", "list_size": len(slots)}
        chosen = slots[position - 1]
    else:
        # Not a number - try to match the exact time they typed against
        # each remembered slot's own displayed time. Folded the same way
        # every other Arabic comparison in this file is, so digit style
        # and minor spacing differences don't cause a false miss.
        folded_input = _normalize_arabic((user_input or "").strip())
        chosen = None
        for slot in slots:
            folded_time = _normalize_arabic(str(slot.get("time_display") or ""))
            if folded_time and (folded_time in folded_input or folded_input in folded_time):
                chosen = slot
                break

        if chosen is None:
            logger.info(
                "select_appointment_slot: %r matched no remembered slot by position or time (session_id=%s)",
                user_input, session_id,
            )
            return {"status": "not_matched"}

    # LOCKED IN. This is the one place the rest of the booking flow reads
    # the chosen time from - never the model's own recollection of the
    # conversation. See graph._build_selected_slot_directive, which
    # reinforces these exact values in the prompt for as long as this
    # booking is in progress.
    session["selected_slot"] = dict(chosen)

    logger.info(
        "select_appointment_slot: session_id=%s locked in slotStart=%s (%s %s)",
        session_id, chosen.get("slotStart"), chosen.get("date_display"), chosen.get("time_display"),
    )

    return {"status": "selected", "slot": chosen}


@tool
def find_best_doctor_in_specialty(
    state: Annotated[AgentState, InjectedState],
    specialty_ids: list,
    criteria: str = "soonest",
) -> dict:
    """Among ALL doctors across one or more specialties, find either the
    one with the SOONEST available appointment, or the one with the
    CHEAPEST fee - use this when the user says they don't care which
    specific doctor they see and just want the earliest opening, or
    explicitly ask for the cheapest option (e.g. after seeing a list of
    doctors for a specialty and asking "who's soonest?" or "who's
    cheapest?").

    `specialty_ids` must come from `list_specialties`'s own response -
    IMPORTANT: pass ALL plausibly-matching specialty ids together as a
    list, same as `find_available_doctors` - a general specialty and
    its more specific sub-specialty (e.g. "Ophthalmology" AND
    "Vitreoretinal Surgery") can both be relevant to the same complaint,
    and passing only one risks missing doctors who are only filed under
    the other. `criteria`: "soonest" (default) or "cheapest".
    Returns:
    {"status": "found", "doctor": {...}, "slot": {...}}  # for "soonest" - present the doctor and when
    {"status": "found", "doctor": {...}, "price": ..., "service": ...}  # for "cheapest"
    {"status": "not_found"}  # no doctors in these specialties currently qualify
    {"status": "not_configured"} / {"status": "error"}"""

    criteria = (criteria or "soonest").strip().lower()
    if criteria not in ("soonest", "cheapest"):
        criteria = "soonest"

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("find_best_doctor_in_specialty called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    doctors_result = api.get_doctors(base_url, specialty_ids=specialty_ids, page_size=200, language=conversation_language(state))
    if not doctors_result["success"]:
        logger.error("find_best_doctor_in_specialty: get_doctors failed: status_code=%s error=%s", doctors_result.get("status_code"), doctors_result.get("error"))
        return {"status": "error"}

    raw_doctors = (doctors_result["data"] or {}).get("items", [])
    doctors = [d for d in raw_doctors if d.get("hasSlots") is not False]
    logger.info(
        "find_best_doctor_in_specialty: specialty_ids=%s criteria=%s api_returned=%d after_hasSlots_filter=%d",
        specialty_ids, criteria, len(raw_doctors), len(doctors),
    )
    if not doctors:
        logger.info("find_best_doctor_in_specialty: not_found - no doctors matched specialty_ids=%s (or all filtered out by hasSlots=False)", specialty_ids)
        return {"status": "not_found"}

    doctor_ids = [d.get("id") for d in doctors if d.get("id")]
    doctors_by_id = {d.get("id"): d for d in doctors}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    if criteria == "soonest":
        now = datetime.now(tz)
        horizon = now + timedelta(days=30)

        slots_result = api.get_doctor_schedule_slots(
            base_url, doctor_ids=doctor_ids,
            from_date=now.isoformat(), to_date=horizon.isoformat(),
            is_booked=False, page_size=1000,
         language=conversation_language(state),)
        if not slots_result["success"]:
            logger.error("find_best_doctor_in_specialty: get_doctor_schedule_slots failed: status_code=%s error=%s", slots_result.get("status_code"), slots_result.get("error"))
            return {"status": "error"}

        raw_slot_items = (slots_result["data"] or {}).get("items", [])
        logger.info(
            "find_best_doctor_in_specialty (soonest): doctor_ids=%s from=%s to=%s api_returned=%d",
            doctor_ids, now.isoformat(), horizon.isoformat(), len(raw_slot_items),
        )

        best = None
        for item in raw_slot_items:
            if item.get("isBooked"):
                continue
            slot_start = to_local_wallclock(item.get("slotStart"), timezone_name)
            if not slot_start:
                continue
            try:
                dt = datetime.fromisoformat(slot_start)
            except ValueError:
                continue
            # `slot_start` is wall-clock (naive); `now` is aware. Compare
            # like with like - see the lead_time note above.
            if dt <= now.replace(tzinfo=None):
                continue
            if best is None or dt < best[0]:
                best = (dt, item)

        if not best:
            logger.info("find_best_doctor_in_specialty (soonest): not_found - none of %d raw item(s) qualified after filtering", len(raw_slot_items))
            return {"status": "not_found"}

        dt, item = best
        doctor = doctors_by_id.get(item.get("doctorId"), {})

        return {
            "status": "found",
            "doctor": {
                "id": item.get("doctorId"),
                "formatedName": _preferred_name(doctor, conversation_language(state)) or item.get("doctorName"),
                "degreeName": doctor.get("degreeName"),
            },
            "slot": {
                "slotStart": dt.isoformat(),
                "slotEnd": to_local_wallclock(item.get("slotEnd"), timezone_name),
                "date_display": _display_date(dt.isoformat()),
                "weekday_display": _display_weekday(dt.isoformat(), conversation_language(state)),
                "time_display": _display_time_12h(dt.isoformat(), conversation_language(state)),
                "branchId": item.get("branchId"),
                "branchName": item.get("branchName"),
            },
        }

    # criteria == "cheapest" - queried per-doctor (confirmed request
    # shape only supports one doctor's fees at a time reliably; the
    # roster for a single specialty is small enough that this is fine).
    best_price = None
    best_doctor_id = None
    best_service = None

    for doctor_id in doctor_ids:
        fees_result = api.get_doctor_fees(base_url, doctor_ids=[doctor_id], language=conversation_language(state))
        if not fees_result["success"]:
            continue
        for item in (fees_result["data"] or {}).get("items", []):
            price = item.get("price")
            if price is None:
                continue
            if best_price is None or price < best_price:
                best_price = price
                best_doctor_id = doctor_id
                best_service = item.get("serviceName")

    if best_doctor_id is None:
        return {"status": "not_found"}

    doctor = doctors_by_id.get(best_doctor_id, {})
    return {
        "status": "found",
        "doctor": {
            "id": best_doctor_id,
            "formatedName": doctor.get("formatedName"),
            "degreeName": doctor.get("degreeName"),
        },
        "price": best_price,
        "service": best_service,
    }


# ==========================================================
# Complaint Agent (collect a complaint, email it via SMTP)
# ==========================================================

# EXPLICIT-CONFIRMATION GATE for send_complaint_email (STEP C6/C7).
#
# The field-presence check below (`missing`) was added after a
# CONFIRMED REAL PRODUCTION FAILURE where the tool was called in the
# same turn the patient first described their problem, with no
# follow-up questions, no name, and no confirmation. It stops a
# complaint with genuinely BLANK fields - but it does NOT stop a
# complaint whose fields are all technically non-empty because they
# were silently carried over from an EARLIER, unrelated part of the
# same session (e.g. a patient's name captured minutes earlier while
# booking an appointment), with STEP C6's summary-and-confirm question
# never actually asked for THIS complaint.
#
# CONFIRMED REAL PRODUCTION FAILURE (medtown, 2026-08-30): the patient
# said "الدواء اللي اتوصفلي غلط" (the medication I was prescribed was
# wrong) and `send_complaint_email` fired in that SAME turn - no
# question about which doctor, no phone confirmation (or offer to use
# a different number), no branch question, and critically no "تأكيد
# إرسال الشكوى بهذا الشكل؟" confirmation - because `patient_name` had
# already been captured during an earlier booking attempt in the same
# conversation and so was never "missing".
#
# This gate requires the LAST thing the assistant said before this
# call to be recognizably STEP C6's confirmation question, and the
# patient's own latest message to be a genuine affirmative answer to
# it - not just any non-empty field values.
_COMPLAINT_CONFIRMATION_QUESTION_RE = re.compile(
    r"تأكيد\s*(?:ال)?إرسال|تأكيد\s*(?:ال)?ارسال|أأكد\s*(?:ال)?إرسال|"
    r"موافق\s*ع(?:لى)?\s*(?:ال)?إرسال|هل\s*(?:ال)?بيانات\s*صحيح|"
    r"confirm\s*(?:the\s*)?(?:sending\s*(?:the\s*)?)?complaint|"
    r"shall\s*i\s*send\s*(?:this|the)\s*complaint"
)

_COMPLAINT_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:نعم|ايوه|أيوه|ايوة|آيوه|اه|آه|ايه|تمام|أكيد|اكيد|ماشي|"
    r"موافق|موافقه|موافقة|صح|تم|ok|okay|yes|sure|confirm(?:ed)?)\b",
    re.IGNORECASE,
)


def _complaint_explicitly_confirmed(state: AgentState) -> bool:
    """True only when the assistant's own immediately-preceding message
    reads as STEP C6's confirmation question AND the patient's latest
    message is a genuine affirmative reply to it."""

    messages = list(state.get("messages") or [])

    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            last_human_idx = i
            break

    if last_human_idx is None:
        return False

    last_human_text = str(getattr(messages[last_human_idx], "content", "") or "").strip()
    if not _COMPLAINT_AFFIRMATIVE_RE.match(last_human_text):
        return False

    for i in range(last_human_idx - 1, -1, -1):
        m = messages[i]
        if getattr(m, "type", None) != "ai":
            continue
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            # An AI message with tool_calls but no text content - keep
            # looking further back for the last one that actually said
            # something to the patient.
            continue
        return bool(_COMPLAINT_CONFIRMATION_QUESTION_RE.search(content))

    return False


@tool
def send_complaint_email(
    state: Annotated[AgentState, InjectedState],
    patient_name: str,
    phone: str,
    branch: str,
    category: str,
    details: str,
) -> dict:
    """Send a hospital complaint once ALL required details have been
    collected AND confirmed with the user (category, details,
    patient_name, phone, branch). Call this only ONE time per
    complaint, right before telling the user their complaint was
    submitted. Delivery goes via an n8n webhook when configured
    (preferred), or direct SMTP otherwise - this is decided
    automatically by server configuration, not something you control.

    `details` must faithfully reflect exactly what the user actually
    said - never a vague generic paraphrase (e.g. never reduce a
    specific complaint to something like "customer service issue").
    Use one bullet point per distinct issue if there are several.
    `branch` can be an empty string / "غير محدد" if not applicable.

    Every other field is genuinely required and is checked before
    anything is sent: an incomplete complaint comes back as
    "incomplete" and is NOT delivered. Never call this in the same turn
    the patient first describes their problem - a complaint can't be
    recalled or amended once it reaches the quality team, so collect
    the details, name, and phone first and confirm them.

    Returns:
    {"status": "sent"}
    {"status": "incomplete", "missing": [...], "reason": ...}
        # Required details are missing or too thin to act on, OR the
        # patient's own explicit confirmation to STEP C6's "تأكيد إرسال
        # الشكوى بهذا الشكل؟" question was not the immediately-preceding
        # exchange (missing item "explicit_confirmation") - NOTHING was
        # sent either way. Go back and collect what's listed in
        # `missing` (one question per message), show the full summary,
        # get an explicit yes to THAT summary, then call this again.
        # Do NOT tell them the complaint was submitted.
    {"status": "not_configured"}  # this clinic has no complaint recipient email(s) set up
    {"status": "error"}  # sending failed (webhook or SMTP error)"""

    # Deterministic completeness check, BEFORE any delivery attempt.
    # The flow in prompts.py already spells out collecting these one at
    # a time, but a prose instruction is not a guarantee: confirmed real
    # production failure - `send_complaint_email` was called in the very
    # same turn the patient first said "المستشفى مش نضيفة", with no
    # follow-up questions, no name, and no confirmation step. Only the
    # clinic's missing email configuration stopped a half-empty
    # complaint from reaching the quality team. A complaint sent
    # incomplete cannot be recalled or amended, so the check lives here
    # where it cannot be skipped.
    missing = []

    normalized_details = (details or "").strip()
    if not normalized_details:
        missing.append("details")
    elif len(normalized_details) < 10:
        # Long enough to be an actual description rather than a stray
        # word, a single emoji, or a fragment like "وحش".
        missing.append("details")

    if not (patient_name or "").strip():
        missing.append("patient_name")

    if not (phone or "").strip():
        missing.append("phone")

    if not (category or "").strip():
        missing.append("category")

    # `branch` is deliberately NOT required - a complaint about the
    # clinic as a whole legitimately has no branch, and demanding one
    # would force exactly the irrelevant question the complaint flow is
    # meant to avoid.

    if not missing and not _complaint_explicitly_confirmed(state):
        # Every field is technically present, but STEP C6's own
        # summarize-and-confirm question was never asked (or the
        # patient's latest message isn't a genuine "yes" to it) - see
        # the CONFIRMED REAL PRODUCTION FAILURE note above. Refuse to
        # send rather than trust that fields being non-empty means the
        # patient actually agreed to submit THIS complaint.
        missing.append("explicit_confirmation")

    if missing:
        logger.warning(
            "send_complaint_email: REFUSED to send an incomplete complaint for client_id=%s session_id=%s - missing/insufficient: %s",
            state.get("client_id"), state.get("session_id"), missing,
        )
        return {
            "status": "incomplete",
            "missing": missing,
            "reason": "collect and confirm these with the patient first, then call again",
        }

    templates = state.get("templates") or {}
    to_emails_raw = templates.get("_complaint_email_to", "")
    to_emails = [e.strip() for e in to_emails_raw.split(",") if e.strip()]

    if not to_emails:
        logger.warning("send_complaint_email called but no complaint_email_to is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    timezone_name = templates.get("_timezone", DEFAULT_TIMEZONE)
    try:
        now_local = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        now_local = datetime.now(timezone.utc)
    submitted_on = now_local.strftime("%d %B %Y  %I:%M %p")

    subject = f"شكوى جديدة - {category} - {patient_name}"
    body = (
        "Complaint Details\n"
        f"- Patient Name: {patient_name}\n"
        f"- Mobile Number: {phone}\n"
        f"- Branch: {branch or 'غير محدد'}\n"
        f"- Complaint Type: {category}\n\n"
        "Complaint Details:\n"
        f"{details}\n\n"
        "Submitted On:\n"
        f"{submitted_on}\n\n"
        "Action Required:\n"
        "يرجى متابعة الشكوى مع المريض واتخاذ الإجراءات اللازمة."
    )

    # Two independent delivery paths, tried in order. Previously a
    # webhook failure returned "error" immediately WITHOUT ever trying
    # SMTP, and a missing configuration was indistinguishable in the
    # reply from a genuine send failure - between them, complaints could
    # silently never arrive at the configured mailbox while the code
    # looked like it had "a webhook and an SMTP fallback".
    attempts: list = []

    if COMPLAINT_WEBHOOK_URL:
        # Preferred path: hand off the actual sending to n8n (its own
        # network path is already confirmed working for this system),
        # rather than connecting to SMTP directly from this backend -
        # confirmed necessary after a direct SMTP connection attempt
        # timed out in production.
        try:
            response = requests.post(
                COMPLAINT_WEBHOOK_URL,
                json={
                    "to": to_emails,
                    "from": SMTP_FROM_EMAIL,
                    "subject": subject,
                    "body": body,
                    "patientName": patient_name,
                    "phone": phone,
                    "branch": branch or "غير محدد",
                    "category": category,
                    "details": details,
                    "submittedOn": submitted_on,
                },
                timeout=15,
            )
            response.raise_for_status()
        except Exception as exc:
            attempts.append(f"webhook: {type(exc).__name__}: {exc}")
            logger.exception("send_complaint_email: failed to POST to COMPLAINT_WEBHOOK_URL - falling back to SMTP if configured")
        else:
            logger.info("send_complaint_email: sent complaint via webhook for patient_name=%r category=%r to=%s", patient_name, category, to_emails)
            return {"status": "sent", "via": "webhook"}
    else:
        attempts.append("webhook: COMPLAINT_WEBHOOK_URL not set")

    missing_smtp = [
        name for name, value in (
            ("SMTP_HOST", SMTP_HOST),
            ("SMTP_USERNAME", SMTP_USERNAME),
            ("SMTP_PASSWORD", SMTP_PASSWORD),
        ) if not value
    ]

    if missing_smtp:
        attempts.append("smtp: missing " + ", ".join(missing_smtp))
        logger.error(
            "send_complaint_email: complaint NOT delivered for client_id=%s - no working transport. Attempts: %s. "
            "Set COMPLAINT_WEBHOOK_URL (recommended) or the SMTP_* variables.",
            state.get("client_id"), " | ".join(attempts),
        )
        return {"status": "error", "reason": "no_transport_configured", "attempts": attempts}

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = ", ".join(to_emails)

    try:
        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
        try:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, to_emails, msg.as_string())
        finally:
            server.quit()
    except Exception as exc:
        attempts.append(f"smtp: {type(exc).__name__}: {exc}")
        logger.exception(
            "send_complaint_email: complaint NOT delivered for client_id=%s. Attempts: %s",
            state.get("client_id"), " | ".join(attempts),
        )
        return {"status": "error", "reason": "send_failed", "attempts": attempts}

    logger.info("send_complaint_email: sent complaint email via SMTP for patient_name=%r category=%r to=%s", patient_name, category, to_emails)
    return {"status": "sent", "via": "smtp"}


@tool
def request_human_handoff(
    state: Annotated[AgentState, InjectedState],
    reason: str,
    patient_agreed: bool,
) -> dict:
    """Signal the surrounding system (n8n) to hand this patient off to a
    human staff member RIGHT NOW.

    A handoff ENDS the patient's conversation with you and puts them in a
    queue, so it only ever happens with the patient's own say-so. There
    are exactly two ways to get that, and `patient_agreed` must be True
    for both:
      - They ASKED for a person themselves ("موظف", "عايز أتكلم مع حد",
        "حد يرد عليا", "human agent"), or
      - You OFFERED a handoff in an earlier turn (e.g. after a real
        failure you couldn't work around) and they said yes to that
        offer in this turn.

    Frustration is NOT agreement. A patient complaining that this isn't
    working, insulting you, or saying "انت مش بتعرف تعمل حاجة" is telling
    you they're upset - not asking to be transferred. In that situation
    do NOT call this tool: apologize, and ASK whether they'd like you to
    transfer them to a staff member. Then call it only after they say
    yes. Confirmed real production failure: a frustrated patient who
    never asked for anyone was transferred out of the conversation
    immediately, with the reason logged as "patient frustrated,
    requested human agent" when no such request had been made.

    WANTING TO FILE A COMPLAINT IS ALSO NOT AGREEMENT. "شكوى"/"شكوي"/"عاوزه
    اعمل شكوه"/"I have a complaint" states a TOPIC, not a request for a
    person - filing a complaint has its own flow (ask what happened,
    which doctor/branch if relevant, then call `send_complaint_email`)
    and stays with you unless the patient separately, explicitly asks
    for a human. Confirmed real production failure: the patient typed
    "شكوي" alone and was immediately transferred with
    reason="patient asked for staff" - they had said nothing of the
    kind, and never got the chance to actually describe the complaint
    at all. The word "complaint"/"شكوى" appearing anywhere in the
    message is never, by itself, grounds to call this tool.

    Pass `patient_agreed=False` if you are unsure whether they actually
    agreed - the handoff is then NOT raised, and you should ask them
    instead. Never set it True to describe a handoff you are about to
    offer but they haven't accepted yet.

    This tool does NOT contact anyone itself and returns no
    patient-facing text - it only raises a flag that n8n reads from this
    turn's response and acts on. It does NOT replace your own reply:
    still say the clinic's own handoff-confirmation line to the patient
    in this same turn, exactly as usual.

    `reason` is for logs only, never shown to the patient - one short
    phrase describing what they actually said (e.g. "patient asked for
    staff", "patient accepted handoff offer after booking API failure").

    Returns {"status": "handoff_requested"} when raised, or
    {"status": "not_requested", "reason": "patient_has_not_agreed"} when
    `patient_agreed` was False - in which case ask them first.

    HARD GUARD (enforced here, not left to this docstring alone - see
    the module-level comment above `_COMPLAINT_ROOTS_FOR_HANDOFF_GUARD`):
    if the patient's own latest message names a complaint ("شكوى"/
    "اشتكي"/"complaint" or similar) and does NOT also separately name a
    person/staff/representative, the call is downgraded to
    "not_requested" regardless of what `patient_agreed` was passed as."""

    latest_text = _latest_human_text_for_handoff_guard(state)
    has_complaint_word = any(root in latest_text for root in _COMPLAINT_ROOTS_FOR_HANDOFF_GUARD)
    has_explicit_human_request = any(root in latest_text for root in _EXPLICIT_HUMAN_REQUEST_ROOTS)

    if patient_agreed and has_complaint_word and not has_explicit_human_request:
        logger.warning(
            "request_human_handoff: BLOCKED BY HARD GUARD - patient's latest message %r "
            "names a complaint with no separate, explicit request for a person. Overriding "
            "patient_agreed=True -> not_requested instead of raising a handoff, to stop this "
            "from repeating the confirmed production failure (a bare 'شكوي' was previously "
            "transferred with reason='patient asked for staff'). session_id=%s client_id=%s "
            "original_reason=%r",
            latest_text, state.get("session_id"), state.get("client_id"), reason,
        )
        return {
            "status": "not_requested",
            "reason": "complaint_word_without_explicit_human_request",
        }

    # GENERAL CONSENT GATE (not tied to any one specific wording): the
    # complaint-word guard above exists because a documented prose rule
    # ("frustration is not agreement") was still not enough on its own
    # once - the same reasoning applies to every OTHER way this tool
    # could be called with patient_agreed=True on an inference rather
    # than a real yes. Require the agreement to be grounded in
    # something the code can actually see:
    #   - the patient's own latest message explicitly names a person
    #     ("موظف", "خدمة العملاء", "human agent"...), OR
    #   - the assistant's OWN previous turn actually said one of those
    #     same words - i.e. a handoff was genuinely offered, and this
    #     turn's short "yes" is answering THAT offer.
    # This is a heuristic, not a perfect parse of intent - it can still
    # ask an extra confirming question in a genuinely-agreed edge case
    # phrased outside these words, which is the safe direction to be
    # wrong in for something that ends a patient's conversation.
    if patient_agreed and not has_explicit_human_request:
        latest_ai_text = _latest_ai_text_before_handoff_guard(state)
        had_prior_offer = any(root in latest_ai_text for root in _EXPLICIT_HUMAN_REQUEST_ROOTS)
        if not had_prior_offer:
            logger.warning(
                "request_human_handoff: BLOCKED BY GENERAL CONSENT GATE - patient_agreed=True "
                "but the patient's latest message %r names no person explicitly, and the "
                "assistant's own last turn %r did not offer a staff handoff either. Overriding "
                "to not_requested rather than trusting an inferred consent. session_id=%s "
                "client_id=%s original_reason=%r",
                latest_text, latest_ai_text, state.get("session_id"), state.get("client_id"), reason,
            )
            return {
                "status": "not_requested",
                "reason": "consent_not_grounded_in_conversation",
            }

    if not patient_agreed:
        # Fail closed: an unconfirmed handoff silently drops rather than
        # ending someone's conversation on an inference about their mood.
        logger.info(
            "request_human_handoff: NOT raised (patient has not agreed) session_id=%s client_id=%s reason=%r",
            state.get("session_id"), state.get("client_id"), reason,
        )
        return {"status": "not_requested", "reason": "patient_has_not_agreed"}

    logger.info(
        "request_human_handoff: session_id=%s client_id=%s reason=%r",
        state.get("session_id"), state.get("client_id"), reason,
    )
    return {"status": "handoff_requested"}


# Cues that the patient is genuinely asking WHERE a branch is / how to
# get there - as opposed to simply naming it (answering a branch
# question during booking, a complaint, or anywhere else). Deliberately
# covers both "address" wording and "how do I get there" wording, in
# Arabic and English.
_LOCATION_REQUEST_CUE_RE = re.compile(
    r"عنوان|فين|وين|أين|اين|موقع|لوكيشن|خريط[هة]|كيف\s*(?:أ|ا)وصل|"
    r"ازاي\s*(?:أ|ا)روح|إزاي\s*(?:أ|ا)روح|طريقه\s*(?:ال)?وصول|"
    r"location|address|map|direction|how\s*(?:do\s*i|to)\s*get\s*there|"
    r"where\s*is"
)


@tool
def share_branch_location(
    state: Annotated[AgentState, InjectedState],
    branch_name: str,
) -> dict:
    """Signal the surrounding system (n8n) to send this branch's map
    location/pin to the patient.

    Call this ONLY when BOTH of these are true this turn:
    1. The patient explicitly asked for the branch's location, address,
       or how to get there (not just named the branch, and not just
       had it confirmed/selected as part of booking or anything else).
    2. `match_entity_info` (entity_type="branch") has ACTUALLY matched a
       real branch and you are telling the patient its address this
       turn - never call this with a branch name you have not just
       confirmed exists via that tool, and never guess or invent one.

    Simply mentioning, confirming, or picking a branch (e.g. during the
    booking flow, or the patient just typing a branch's name with no
    question attached) is NOT a location request - do not call this
    tool in that case, even if the branch's address happens to be
    matched. `branch_name` must be exactly the `name` field
    `match_entity_info` returned for that branch (not the patient's raw
    typed text, not a translation of it).

    This tool does not send anything itself and returns no patient-
    facing text - it only raises a flag (with the branch name) that n8n
    reads from this turn's response, looks up that branch's coordinates,
    and sends the actual map pin. Still give the patient the address in
    text in this same reply, exactly as usual.

    Returns {"status": "location_requested", "branch_name": branch_name}
    when the patient's own latest message genuinely asked for the
    location/address/directions.
    {"status": "not_requested", "reason": "no_explicit_location_request"}
    otherwise - NOTHING is signalled to n8n and no map pin is sent. This
    is enforced here, not left to the docstring above alone.

    CONFIRMED REAL PRODUCTION FAILURE: during the COMPLAINT flow's STEP
    C5, the patient was asked "هل في فرع محدد حابة تسجلي الشكوى عليه؟"
    and answered simply "فرع المنار" (naming the branch, no location
    question at all) - and a map pin of that branch was sent anyway."""

    latest_text = ""
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            latest_text = content if isinstance(content, str) else str(content or "")
            break

    if not _LOCATION_REQUEST_CUE_RE.search(latest_text):
        logger.warning(
            "share_branch_location: REFUSED for client_id=%s session_id=%s branch_name=%r - "
            "the patient's latest message %r does not actually ask for a location/address, "
            "so no map pin flag was raised",
            state.get("client_id"), state.get("session_id"), branch_name, latest_text,
        )
        return {"status": "not_requested", "reason": "no_explicit_location_request"}

    logger.info(
        "share_branch_location: session_id=%s client_id=%s branch_name=%r",
        state.get("session_id"), state.get("client_id"), branch_name,
    )
    return {"status": "location_requested", "branch_name": branch_name}


ALL_TOOLS = [
    validate_phone_format,
    compare_phone,
    lookup_appointment,
    check_booking_status,
    cancel_appointment,
    send_otp,
    verify_otp,
    list_specialties,
    find_available_doctors,
    list_branches_for_specialty,
    get_next_weekday_date,
    get_doctor_schedule,
    get_available_reschedule_slots,
    reschedule_appointment,
    answer_hospital_faq,
    list_hospital_services,
    list_branch_services,
    find_branches_offering_service,
    match_entity_info,
    reset_booking_session,
    match_entity_for_booking,
    get_doctor_fees,
    get_patient_info,
    resolve_available_day,
    list_available_days_for_booking,
    create_new_booking,
    get_doctor_schedule_for_booking,
    get_available_slots_for_booking,
    select_appointment_slot,
    find_best_doctor_in_specialty,
    send_complaint_email,
    request_human_handoff,
    share_branch_location,
]
