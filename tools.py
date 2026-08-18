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


def _client_default_country_code(state=None) -> str:
    """The country code to assume for a BARE LOCAL number (one written
    with a leading 0, or with no country code at all), for this client.

    Read from the client's own config (`country_codes_hint`, e.g.
    "+966, +20" -> "966") so a Saudi clinic assumes Saudi and an
    Egyptian one assumes Egypt, instead of every tenant sharing one
    hardcoded country. Falls back to DEFAULT_COUNTRY_CODE when the
    client hasn't configured a hint."""

    hint = ((state or {}).get("templates") or {}).get("_country_codes_hint")
    if hint:
        match = re.search(r"\+?(\d{1,4})", str(hint))
        if match:
            return match.group(1)

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

    local_from = to_riyadh(item.get("bookingTimeFrom"), timezone_name)
    local_to = to_riyadh(item.get("bookingTimeTo"), timezone_name)

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
    Appointment fields: ref, doctorName, branchName, serviceName,
    specialtyName, statusName, date_display, time_display, patientFullName,
    mobileNumber, email, id."""

    if use_channel_identity:
        channel_phone = state.get("channel_phone")
        logger.info("lookup_appointment: use_channel_identity=True, channel_phone=%r", channel_phone)
        if not channel_phone:
            return {"status": "no_channel_identity"}
        phone = channel_phone

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
    logger.info("OTP sent for %s (test otp=%s)", normalized, TEST_OTP)
    return {"status": "otp_sent"}


@tool
def verify_otp(phone: str, otp: str) -> dict:
    """Verify a user-entered OTP code against the one sent to `phone`.
    Returns {"status": "otp_valid"} or {"status": "otp_invalid"}."""

    normalized = normalize_phone_number(phone)

    if OTP_PROVIDER == "authentica":
        result = api.authentica_verify_otp(normalized, otp)
        return {"status": "otp_valid" if result["success"] else "otp_invalid"}

    record = _otp_storage.get(normalized)

    if not record:
        return {"status": "otp_invalid"}

    if time.time() - record["created_at"] > OTP_TTL_SECONDS:
        return {"status": "otp_invalid"}

    if str(otp).strip() == str(record["otp"]):
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


def _get_booking_session(session_id: str) -> dict:
    return _BOOKING_SESSIONS.setdefault(session_id, {
        "doctor_id": None, "branch_id": None, "service_id": None,
        "last_list": None,  # {"entity_type": "doctor"/"branch", "items": [shaped items]}
        "specialty_ids": None,  # remembered so later steps reuse the same specialties
    })


def _remember_list(state: AgentState, entity_type: str, items: list) -> None:
    """Record the list of doctors/branches just returned to the model, so
    a later bare-number reply can be resolved against the SAME ordering
    the user actually saw. Every tool that returns a user-facing list
    must call this - see the comment above _BOOKING_SESSIONS."""

    session_id = state.get("session_id")
    if not session_id:
        return

    session = _get_booking_session(session_id)
    session["last_list"] = {"entity_type": entity_type, "items": list(items)}

    logger.info(
        "_remember_list: session_id=%s entity_type=%s count=%d",
        session_id, entity_type, len(items),
    )


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


def _doctors_at_branch(state: AgentState, base_url: str, branch_id: str) -> list:
    """Fetch the available doctors at one branch, narrowed to the
    specialties this booking is already about (when known), and remember
    the list for positional selection.

    Exists because "which doctors are here" changes the moment a branch
    is confirmed - not every doctor works at every branch. Without this,
    the model would re-display doctor names it had shown BEFORE the
    branch was picked, which is both wrong (some of them don't work
    there) and unselectable (the remembered list at that point is the
    BRANCH list, so a reply of "2" resolves to nothing)."""

    session = _get_booking_session(state.get("session_id"))
    specialty_ids = session.get("specialty_ids") or None

    now = datetime.utcnow()
    result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids,
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

    if doctors:
        _remember_list(state, "doctor", doctors)

    logger.info(
        "_doctors_at_branch: branch_id=%s specialty_ids=%s -> %d doctor(s)",
        branch_id, specialty_ids, len(doctors),
    )

    return doctors


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

    if specialty_ids:
        session["specialty_ids"] = list(specialty_ids)

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
    schedule_result = api.get_doctor_schedule(
        base_url, doctor_ids=list(doctors_by_id.keys()), page_size=500,
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

    logger.info(
        "list_branches_for_specialty: specialty_ids=%s broadened=%s -> %d branch(es), %d doctor(s)",
        specialty_ids, broadened, len(branches), len(doctors),
    )

    if broadened:
        return {"status": "found_broader_search", "branches": branches}

    return {"status": "found", "branches": branches}


@tool
def find_available_doctors(
    state: Annotated[AgentState, InjectedState],
    specialty_ids: list,
    days_ahead: int = DOCTOR_AVAILABILITY_WINDOW_DAYS,
    branch_name: str = "",
    allow_broader_search: bool = True,
) -> dict:
    """Find doctors who currently have a bookable service AND an available
    schedule slot within the next `days_ahead` days, across one or more
    specialties. ALWAYS call `list_specialties` first to get correct ids
    - never guess or invent one.

    `branch_name`: optional. Pass the user's raw branch text when they've
    said which branch they want (e.g. "الدقي", "فرع زايد") - the branch
    is resolved and CONFIRMED into the booking session automatically, and
    only doctors working at that branch are returned. Leave it empty when
    the user hasn't picked a branch (or said they don't mind). If the
    user doesn't know which branches exist, call
    `list_branches_for_specialty` instead of guessing.

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

    # Remember which specialties this search used, so later steps
    # (list_branches_for_specialty, "who's soonest?") reuse exactly the
    # same set instead of the model having to re-derive them.
    if specialty_ids:
        session["specialty_ids"] = list(specialty_ids)

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

    elif session.get("branch_id"):
        # A branch was already confirmed earlier in this booking - keep
        # the doctor list consistent with it.
        branch_ids = [session["branch_id"]]

    now = datetime.utcnow()
    intersection_start = now.isoformat() + "Z"
    intersection_end = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids,
        branch_ids=branch_ids,
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


_WEEKDAY_NAMES = {
    # English (case-insensitive)
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    # Arabic
    "الاثنين": 0, "الإثنين": 0, "اثنين": 0,
    "الثلاثاء": 1, "ثلاثاء": 1,
    "الأربعاء": 2, "الاربعاء": 2, "أربعاء": 2, "اربعاء": 2,
    "الخميس": 3, "خميس": 3,
    "الجمعة": 4, "جمعة": 4,
    "السبت": 5, "سبت": 5,
    "الأحد": 6, "الاحد": 6, "أحد": 6, "احد": 6,
}


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

    key = (weekday_name or "").strip().lower()
    target_weekday = _WEEKDAY_NAMES.get(key)

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

    result = api.get_doctor_schedule(base_url, doctor_ids=[resolved["doctor_id"]], effective_date=effective_date, language=conversation_language(state))

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
            "fromDateTime": to_riyadh(item.get("fromDateTime"), timezone_name),
            "toDateTime": to_riyadh(item.get("toDateTime"), timezone_name),
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
        slot_start = to_riyadh(item.get("slotStart"), timezone_name)
        slot_end = to_riyadh(item.get("slotEnd"), timezone_name)
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
        now_local = datetime.now(ZoneInfo(timezone_name))
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

    return {"status": "success"}


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

    scored = []
    for item in candidates:
        best_score = 0.0
        for key in name_keys:
            value = item.get(key)
            if not value:
                continue
            normalized_value = _normalize_arabic(value)
            if normalized_input == normalized_value:
                best_score = max(best_score, 1.0)
            elif normalized_input in normalized_value or normalized_value in normalized_input:
                best_score = max(best_score, 0.96)
            else:
                ratio = difflib.SequenceMatcher(None, normalized_input, normalized_value).ratio()
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
    {"status": "ambiguous", "candidates": [...]}  # show each candidate's
        name and ask the user which one they meant
    {"status": "not_matched"}
    {"status": "not_configured"}  # no doctors_base_url set up for this client
    {"status": "error"}

    Doctor fields: formatedName, altName, degreeName, specialtyName,
    defaultServiceName (serviceName). Fees are NOT included here - use
    `get_doctor_fees` if (and only if) the user explicitly asks a price.
    Branch fields: name, altName, address, cityName, countryName,
    stateName, email, mobile."""

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
                    "formatedName": i.get("formatedName") or i.get("name"),
                    "altName": i.get("altName"),
                    "specialtyName": i.get("specialtyName"),
                    "degreeName": i.get("degreeName"),
                }
                for i in items
            ]
        else:
            shaped = [
                {
                    "name": i.get("name") or i.get("formatedName"),
                    "altName": i.get("altName"),
                    "address": i.get("address"),
                    "cityName": i.get("cityName"),
                }
                for i in items
            ]
        if not shaped:
            return {"status": "not_matched"}
        return {"status": "list", "items": shaped}

    match_result = _fuzzy_match(user_input, items, name_keys)
    logger.info(
        "match_entity_info: entity_type=%s user_input=%r api_returned=%d result=%s%s",
        entity_type, user_input, len(items), match_result["result"],
        f" score={match_result.get('score')}" if match_result["result"] == "matched" else "",
    )

    if match_result["result"] == "not_matched":
        return {"status": "not_matched"}

    def _shape_doctor(i):
        return {
            "formatedName": i.get("formatedName") or i.get("name"),
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
        return {
            "name": i.get("name") or i.get("formatedName"),
            "altName": i.get("altName"),
            "address": i.get("address"),
            "cityName": i.get("cityName"),
            "countryName": i.get("countryName"),
            "stateName": i.get("stateName"),
            "email": i.get("email"),
            "mobile": i.get("mobile"),
        }

    shape_fn = _shape_doctor if entity_type == "doctor" else _shape_branch

    if match_result["result"] == "matched":
        return {"status": "matched", "item": shape_fn(match_result["item"])}

    return {"status": "ambiguous", "candidates": [shape_fn(i) for i in match_result["items"]]}


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

    cache_key = (
        base_url,
        tuple(sorted(specialty_ids)) if specialty_ids else None,
        tuple(sorted(branch_ids)) if branch_ids else None,
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
            "(specialty_ids=%s branch_ids=%s)",
            elapsed, len(items), specialty_ids, branch_ids,
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
           When entity_type="branch", this ALSO carries
           "doctorsAtBranch": [...] - the doctors who actually work at
           that branch (narrowed to this booking's specialty when known)
           and already remembered for numeric selection. Show THAT list,
           numbered - never re-show doctor names from before the branch
           was chosen, because not every doctor works at every branch.
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
        result = _fetch_doctors_for_booking(state, base_url, session.get("specialty_ids"), branch_filter)

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
        schedule_result = api.get_doctor_schedule(base_url, doctor_ids=[session["doctor_id"]], language=conversation_language(state))
        if not schedule_result["success"]:
            logger.error("match_entity_for_booking (branch, doctor-filtered): get_doctor_schedule failed: status_code=%s error=%s", schedule_result.get("status_code"), schedule_result.get("error"))
            return {"matched": False, "ambiguous": False, "status": "error"}

        schedule_items = (schedule_result["data"] or {}).get("items", [])
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
        result = {"success": True, "data": {"items": [b for b in all_branch_items if b.get("id") in doctor_branch_ids]}, "error": None}
        name_keys = ["name", "altName", "formatedName", "cityName", "_configAliases"]
    else:
        result = api.get_branches(base_url, page_size=200, language=conversation_language(state))
        name_keys = ["name", "altName", "formatedName", "cityName", "_configAliases"]

    if not result["success"]:
        logger.error("match_entity_for_booking API call failed: entity_type=%s status_code=%s error=%s", entity_type, result.get("status_code"), result.get("error"))
        return {"matched": False, "ambiguous": False, "status": "error"}

    items = (result["data"] or {}).get("items", [])
    if entity_type == "branch":
        # Same bilingual-name bridge as match_entity_info above.
        items = _with_branch_aliases(items, state)

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
                "degreeName": i.get("degreeName"),
                "specialtyName": i.get("specialtyName"),
                "branchId": i.get("branchId"),
                "branchName": i.get("branchName"),
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

    if entity_type == "branch" and not needs_confirmation and shaped.get("id"):
        doctors_here = _doctors_at_branch(state, base_url, shaped["id"])
        response["doctorsAtBranch"] = doctors_here
        if not doctors_here:
            # Explicit flag rather than just an empty list: confirmed
            # real failure - with an empty list the reply still said
            # "هنا قائمة الدكاترة المتاحين في الفرع" and then listed
            # nobody, leaving the patient with a confirmed branch and no
            # way forward.
            response["noDoctorsAtBranch"] = True

    return response


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
    return (shaped_entity.get("formatedName") or shaped_entity.get("name") or "").strip()


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
    {"status": "not_found"}  # not registered - collect name/email fresh
    {"status": "not_configured"} / {"status": "error"}"""

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
    {"status": "found", "date": "YYYY-MM-DD", "weekday_name": "Thursday", "from_date": ..., "to_date": ...}
    {"status": "not_found"}  # no available slot for that weekday within the booking window
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}

    key = (weekday_name or "").strip().lower()
    target_weekday = _WEEKDAY_NAMES.get(key)
    if target_weekday is None:
        logger.warning("resolve_available_day: unrecognized weekday_name=%r", weekday_name)
        return {"status": "error"}

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
        schedule_result = api.get_doctor_schedule(base_url, doctor_ids=[doctor_id], language=conversation_language(state))
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

    lead_time = now + timedelta(hours=12)  # 12h minimum advance booking lead, matches production
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
        slot_start_local = to_riyadh(item.get("slotStart"), timezone_name)
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
        return {"status": "not_found"}

    candidates.sort()
    chosen_dt = candidates[0]
    chosen_date = chosen_dt.date()
    english_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_weekday]
    logger.info("resolve_available_day: found date=%s (weekday=%s) from %d candidate(s)", chosen_date.isoformat(), english_name, len(candidates))

    day_start = datetime.combine(chosen_date, datetime.min.time(), tzinfo=chosen_dt.tzinfo)
    day_end = datetime.combine(chosen_date, datetime.max.time().replace(microsecond=0), tzinfo=chosen_dt.tzinfo)

    return {
        "status": "found",
        "date": chosen_date.isoformat(),
        "weekday_name": english_name,
        "from_date": day_start.isoformat(),
        "to_date": day_end.isoformat(),
    }


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

    try:
        now_local = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        now_local = datetime.now(ZoneInfo(DEFAULT_TIMEZONE))

    starts = []
    for item in (result["data"] or {}).get("items", []):
        if item.get("isBooked") is True:
            continue
        local = to_riyadh(item.get("slotStart"), timezone_name)
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
    limit: int = 1,
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

SHOW THE SOONEST AVAILABLE DAY ONLY: `limit` defaults to 1 on
    purpose. A doctor with a weekly clinic produces the same appointment
    repeated at different dates ("Saturday 22/08, Saturday 29/08,
    Saturday 05/09..."), which is noise, not a choice - the patient
    almost always wants the earliest one. Offer that single date and ask
    if it suits them.

    Only when the patient asks for something else ("مش مناسب", "معاد
    أبعد", "في مواعيد تانية؟") call this AGAIN with `offset` advanced
    past what you already showed, and only then may you raise `limit`
    (e.g. limit=3) to show a few alternatives. Never dump the whole
    window on the first reply.

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
        schedule_result = api.get_doctor_schedule(base_url, doctor_ids=[doctor_id], language=conversation_language(state))
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
                for item in (schedule_result["data"] or {}).get("items", []):
                    item_branch_id = item.get("branchId")
                    if not item_branch_id or item_branch_id in seen:
                        continue
                    seen.add(item_branch_id)
                    match = branches_lookup.get(item_branch_id)
                    name = (_arabic_preferred_name(match) if match else None) or item.get("branchName")
                    if name:
                        branch_options.append({"name": name})
        except Exception:
            logger.exception("list_available_days_for_booking: failed to build branch options for missing_branch")

        if branch_options:
            logger.info(
                "list_available_days_for_booking: missing_branch for doctor_id=%s -> %d branch option(s)",
                doctor_id, len(branch_options),
            )
            return {"status": "missing_branch", "branches": branch_options}

        return {"status": "missing_branch"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now = datetime.now(tz)
    horizon_days = 42  # same booking window resolve_available_day uses
    lead_time = now + timedelta(hours=12)  # same 12h minimum advance lead

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

        slot_start_local = to_riyadh(item.get("slotStart"), timezone_name)
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
    # skipped, not shown. Because `limit` defaults to 1 this is normally
    # a single extra call, and it makes "here is your appointment"
    # something the next turn can actually honour.
    verified_calls = 0
    MAX_VERIFY_CALLS = 8
    consumed = 0

    for date_iso in all_dates[offset:]:
        if len(days) >= limit:
            break

        slot_times = sorted(by_date[date_iso])
        first = slot_times[0]

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
     language=conversation_language(state),)

    if not result["success"]:
        logger.error("get_doctor_schedule_for_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
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
            except Exception:
                logger.exception("get_doctor_schedule_for_booking: failed to enrich auto-confirmed branch name")
            if not session.get("branch_display_name"):
                session["branch_display_name"] = only_branch_name
            logger.info("get_doctor_schedule_for_booking: auto-confirmed single branch_id=%s (%s) for doctor_id=%s", only_branch_id, session.get("branch_display_name"), doctor_id)

    doctor_display_name = session.get("doctor_display_name")
    branch_display_name = session.get("branch_display_name")

    schedules = [
        {
            "recurringDaysNames": item.get("recurringDaysNames"),
            "fromDateTime": to_riyadh(item.get("fromDateTime"), timezone_name),
            "toDateTime": to_riyadh(item.get("toDateTime"), timezone_name),
            "branchName": branch_display_name if (branch_display_name and item.get("branchId") == session.get("branch_id")) else item.get("branchName"),
            "branchId": item.get("branchId"),
            "doctorName": doctor_display_name or item.get("doctorName"),
        }
        for item in items
    ]

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
        slot_start = to_riyadh(item.get("slotStart"), timezone_name)
        slot_end = to_riyadh(item.get("slotEnd"), timezone_name)
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
        now_local = datetime.now(ZoneInfo(timezone_name))
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

    return {"status": "found", "slots": slots}


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
            slot_start = to_riyadh(item.get("slotStart"), timezone_name)
            if not slot_start:
                continue
            try:
                dt = datetime.fromisoformat(slot_start)
            except ValueError:
                continue
            if dt <= now:
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
                "slotEnd": to_riyadh(item.get("slotEnd"), timezone_name),
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
        # Required details are missing or too thin to act on - NOTHING
        # was sent. Go back and collect what's listed in `missing` (one
        # question per message), confirm it with the patient, then call
        # this again. Do NOT tell them the complaint was submitted.
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
    `patient_agreed` was False - in which case ask them first."""

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


@tool
def share_branch_location(
    state: Annotated[AgentState, InjectedState],
    branch_name: str,
) -> dict:
    """Signal the surrounding system (n8n) to send this branch's map
    location/pin to the patient.

    Call this ONLY right after `match_entity_info` (entity_type=
    "branch") has ACTUALLY matched a real branch and you are telling the
    patient its address this turn - never call this with a branch name
    you have not just confirmed exists via that tool, and never guess or
    invent one. `branch_name` must be exactly the `name` field
    `match_entity_info` returned for that branch (not the patient's raw
    typed text, not a translation of it).

    This tool does not send anything itself and returns no patient-
    facing text - it only raises a flag (with the branch name) that n8n
    reads from this turn's response, looks up that branch's coordinates,
    and sends the actual map pin. Still give the patient the address in
    text in this same reply, exactly as usual.

    Returns {"status": "location_requested", "branch_name": branch_name}."""

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
    find_best_doctor_in_specialty,
    send_complaint_email,
    request_human_handoff,
    share_branch_location,
]
