"""
Raw HTTP client layer.

Every external HTTP call in the system lives here, one function per call,
mirroring the n8n HTTP Request nodes 1:1:

  - GuestBookings/GetList (by ref)    <- f_lookup_appointment.json "HTTP Request"
                                         f_cancel_appointment.json "HTTP Request"
  - GuestBookings/GetList (by phone)  <- f_lookup_appointment.json "HTTP Request2"
                                         f_cancel_appointment.json "HTTP Request2"
  - GuestBookings/Cancel/{id}         <- f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"/"HTTP Request4"
  - Authentica send-otp / verify-otp  <- langchain_cancellation.json "send_otp5"/"verify_otp5"

No business logic (filtering, selection, formatting) lives here - that's
tools.py's job. Every function catches network failures itself and
returns a structured result rather than raising, so graph nodes never
need a try/except around a tool call.
"""

import logging
import time
from typing import Optional

import requests

from config import (
    AUTHENTICA_API_KEY,
    AUTHENTICA_BASE_URL,
    AUTHENTICA_FALLBACK_EMAIL,
    AUTHENTICA_TEMPLATE_ID,
    CLIENT_ID_HEADER,
    DOCTORS_API_MAX_RETRIES,
    DOCTORS_API_RETRY_BACKOFF_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Result helper
# ==========================================================

def _result(success: bool, status_code: Optional[int] = None, data=None, error: Optional[str] = None,
            details: Optional[list] = None) -> dict:
    return {"success": success, "status_code": status_code, "data": data, "error": error,
            "details": details or []}


def _validation_details(response) -> list:
    """Pull field-level validation complaints out of a 4xx body.

    The API reports these as {"messages": [{"prop": "MobileNumber",
    "message": "Mobile Number Not Valid"}], ...}. Without surfacing them,
    every 400 collapses into one opaque "validation_error" and the
    patient is told there's a "technical problem" they should retry later
    - when in fact something specific and fixable was rejected (a phone
    number in the wrong format, a missing email) that retrying will
    never resolve. Confirmed real production failure: a booking was
    refused because the patient's mobile number wasn't accepted, and the
    reply blamed a technical fault instead of mentioning the number.

    Returns [{"field": ..., "message": ...}, ...], or [] if the body
    isn't in that shape."""

    try:
        body = response.json()
    except ValueError:
        return []

    if not isinstance(body, dict):
        return []

    details = []
    for entry in body.get("messages") or []:
        if not isinstance(entry, dict):
            continue
        message = entry.get("message") or entry.get("Message")
        if not message:
            continue
        details.append({
            "field": entry.get("prop") or entry.get("Prop") or "",
            "message": str(message),
        })

    return details


def _headers(client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if client_id:
        headers[CLIENT_ID_HEADER] = client_id
    if language:
        headers["accept-language"] = language
    return headers


# ==========================================================
# Guest Bookings API
# ==========================================================

def _request_with_retry(method: str, url: str, **kwargs):
    """Shared retry loop for every outbound call in this module (POST/PUT
    alike), so a slow-but-eventually-alive upstream gets the same patience
    everywhere - not just on the Doctors/Specialties endpoint that
    originally motivated this.

    Retries ONLY cover failure modes that are plausibly transient
    (timeout, connection error, 5xx) - see the DOCTORS_API_MAX_RETRIES
    comment in config.py. A 4xx is a real, reproducible problem with THIS
    request and retrying it would just get the same 4xx back slower, so
    those are returned immediately with no retry loop involved.

    Returns (response_or_None, last_was_timeout, last_exception).
    Callers treat `response is None` as "never got a response at all"
    and fall back to the existing timeout/exception handling; a non-None
    response (even a 4xx/5xx) is handled by the caller's normal
    status-code branches exactly as before.
    """

    max_attempts = max(1, DOCTORS_API_MAX_RETRIES + 1)
    response = None
    last_timeout = False
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        last_timeout = False
        last_exc = None
        try:
            response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
        except requests.Timeout:
            last_timeout = True
            response = None
        except requests.RequestException as exc:
            last_exc = exc
            response = None

        if response is not None and response.status_code < 500:
            break

        is_last_attempt = attempt == max_attempts
        if response is not None:
            logger.error(
                "%s %s server error status=%s (attempt %d/%d%s)",
                method.upper(), url, response.status_code, attempt, max_attempts,
                "" if is_last_attempt else ", retrying",
            )
        elif last_timeout:
            logger.warning(
                "Request timed out: %s %s (attempt %d/%d%s)",
                method.upper(), url, attempt, max_attempts,
                "" if is_last_attempt else ", retrying",
            )
        else:
            logger.warning(
                "Request failed: %s %s error=%s (attempt %d/%d%s)",
                method.upper(), url, last_exc, attempt, max_attempts,
                "" if is_last_attempt else ", retrying",
            )

        if is_last_attempt:
            break

        time.sleep(DOCTORS_API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    return response, last_timeout, last_exc


def get_bookings_by_ref(base_url: str, ref_number: str, language: Optional[str] = None, client_id: Optional[str] = None) -> dict:
    """POST {base_url}/api/GuestBookings/GetList with bookingRefNum.

    Mirrors f_lookup_appointment.json "HTTP Request" / f_cancel_appointment.json "HTTP Request".
    """

    url = f"{base_url}/api/GuestBookings/GetList"
    payload = {"bookingRefNum": ref_number}

    return _post_bookings(url, payload, language, client_id)


def get_bookings_by_phone(
    base_url: str,
    phone: str,
    language: Optional[str] = None,
    client_id: Optional[str] = None,
    page_size: int = 1000,
    status_list: Optional[list] = None,
) -> dict:
    """POST {base_url}/api/GuestBookings/GetList with mobileNumber + pageSize.

    Mirrors f_lookup_appointment.json "HTTP Request2" (pageSize: 1000).

    `status_list`, when given, is sent as the API's own "statusList"
    filter field (confirmed from the Booking API's documented request
    schema) - e.g. [1, 2] for New+Confirmed only. This lets the server
    do the active-status filtering directly. tools.py's own client-side
    filtering (_filter_active) still runs afterward as a second,
    defense-in-depth layer regardless of whether this is used.
    """

    url = f"{base_url}/api/GuestBookings/GetList"
    payload = {"mobileNumber": phone, "pageSize": page_size}

    if status_list:
        payload["statusList"] = status_list

    return _post_bookings(url, payload, language, client_id)


def _post_bookings(url: str, payload: dict, language: Optional[str], client_id: Optional[str]) -> dict:
    logger.debug("POST %s payload=%s", url, payload)

    response, last_timeout, last_exc = _request_with_retry(
        "post", url, json=payload, headers=_headers(client_id=client_id, language=language),
    )

    if response is None:
        if last_timeout:
            logger.warning("Booking lookup timed out: %s", url)
            return _result(False, error="timeout")
        logger.exception("Booking lookup request failed: %s", url)
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("GuestBookings API server error: %s status=%s body=%s", url, response.status_code, response.text[:500])
        return _result(False, response.status_code, error="server_error")

    if response.status_code in (401, 403):
        logger.error(
            "GuestBookings API AUTHENTICATION/AUTHORIZATION error (%s) - this is a credentials/access "
            "problem on the API server itself, not a request-content problem: %s body=%s",
            response.status_code, url, response.text[:500],
        )
        return _result(False, response.status_code, error="authentication_error")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error(
            "GuestBookings API validation error: %s status=%s body=%s rejected_fields=%s",
            url, response.status_code, response.text[:500],
            [d["field"] for d in details] or "unknown",
        )
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def cancel_booking_by_guid(base_url: str, booking_guid: str, client_id: Optional[str] = None) -> dict:
    """PUT {base_url}/api/GuestBookings/Cancel/{booking_guid}.

    Mirrors f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"
    (onError: continueErrorOutput -> here, a structured failure result
    instead of a raised exception achieves the same thing).
    """

    url = f"{base_url}/api/GuestBookings/Cancel/{booking_guid}"

    logger.debug("PUT %s", url)

    response, last_timeout, last_exc = _request_with_retry(
        "put", url, headers=_headers(client_id=client_id),
    )

    if response is None:
        if last_timeout:
            logger.warning("Cancel request timed out: %s", url)
            return _result(False, error="timeout")
        logger.exception("Cancel request failed: %s", url)
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        return _result(False, response.status_code, error="server_error")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="validation_error")

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body)


# ==========================================================
# Authentica OTP API (real provider - langchain_cancellation.json
# "send_otp5" / "verify_otp5"). Only used when config.OTP_PROVIDER ==
# "authentica"; see services in tools.py for the dummy alternative.
# ==========================================================

def authentica_send_otp(phone: str) -> dict:
    url = f"{AUTHENTICA_BASE_URL}/send-otp"

    payload = {
        "method": "sms",
        "template_id": AUTHENTICA_TEMPLATE_ID,
        "fallback_email": AUTHENTICA_FALLBACK_EMAIL,
        "phone": phone,
    }
    headers = {"Accept": "application/json", "X-Authorization": AUTHENTICA_API_KEY}

    response, last_timeout, last_exc = _request_with_retry("post", url, headers=headers, data=payload)

    if response is None:
        if last_timeout:
            return _result(False, error="timeout")
        logger.exception("Authentica send_otp failed")
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="send_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    return _result(True, response.status_code, data=body)


def authentica_verify_otp(phone: str, otp: str, email: str = "") -> dict:
    url = f"{AUTHENTICA_BASE_URL}/verify-otp"

    payload = {"otp": otp, "email": email, "phone": phone}
    headers = {"Accept": "application/json", "X-Authorization": AUTHENTICA_API_KEY}

    response, last_timeout, last_exc = _request_with_retry("post", url, headers=headers, data=payload)

    if response is None:
        if last_timeout:
            return _result(False, error="timeout")
        logger.exception("Authentica verify_otp failed")
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="verify_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    verified = bool(body.get("isSuccess") or body.get("success") or body.get("verified"))

    return _result(verified, response.status_code, data=body)


# ==========================================================
# Doctors / Specialties API (Medical Concierge feature)
# ==========================================================
#
# Separate service from GuestBookings, confirmed on a different port
# (1102 vs 1101). Response shape (confirmed directly from the API's own
# Swagger "Execute" output): {"data": {"items": [...], ...},
# "statusCode": 200, "isSuccess": true, "messages": [...]} - handled the
# same way _post_bookings already handles GuestBookings' identical
# response envelope.

def _post_json(url: str, payload: dict, client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    """Generic POST + envelope handling, shared by get_specialties/
    get_doctors. Mirrors _post_bookings' error handling exactly
    (timeout/5xx/4xx/empty/invalid JSON/isSuccess check), kept as a
    separate function so GuestBookings' own _post_bookings is untouched.

    `language` ("ar"/"en") is sent as the accept-language header, exactly
    as _post_bookings already does for the GuestBookings endpoints. The
    Doctors/Specialties/Branches endpoints honour it too and return
    their doctor, branch, specialty and SERVICE names already localized -
    which is the only reliable way to get e.g. "فحص نظر" instead of
    "Eye Vision Check" for a service, since unlike doctors and branches
    a service has no altName field to fall back on."""

    logger.debug("POST %s payload=%s", url, payload)

    # Retries ONLY cover failure modes that are plausibly transient on
    # this specific endpoint (timeout, connection error, 5xx) - see the
    # DOCTORS_API_MAX_RETRIES comment in config.py. A 4xx is a real,
    # reproducible problem with THIS request (bad payload, bad auth,
    # wrong path) and retrying it would just get the same 4xx back
    # slower, so those still fall straight through to the existing
    # handling below with no retry loop involved.
    max_attempts = max(1, DOCTORS_API_MAX_RETRIES + 1)
    response = None
    last_timeout = False
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        last_timeout = False
        last_exc = None
        try:
            response = requests.post(
                url,
                json=payload,
                headers=_headers(client_id=client_id, language=language),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            last_timeout = True
            response = None
        except requests.RequestException as exc:
            last_exc = exc
            response = None

        if response is not None and response.status_code < 500:
            # Success or a non-retryable 4xx - stop retrying either way.
            break

        is_last_attempt = attempt == max_attempts
        if response is not None:
            logger.error(
                "Doctors/Specialties API server error: %s status=%s body=%s (attempt %d/%d%s)",
                url, response.status_code, response.text[:1000], attempt, max_attempts,
                "" if is_last_attempt else ", retrying",
            )
        elif last_timeout:
            logger.warning(
                "Request timed out: %s (attempt %d/%d%s)",
                url, attempt, max_attempts, "" if is_last_attempt else ", retrying",
            )
        else:
            logger.warning(
                "Request failed: %s error=%s (attempt %d/%d%s)",
                url, last_exc, attempt, max_attempts, "" if is_last_attempt else ", retrying",
            )

        if is_last_attempt:
            break

        # Exponential backoff (0.5s, 1s, 2s, ...) - a short pause is
        # enough to ride out the sub-second blip this endpoint is known
        # to have, without holding up the patient for long if it's a
        # real outage.
        time.sleep(DOCTORS_API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    if response is None:
        if last_timeout:
            return _result(False, error="timeout")
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("Doctors/Specialties API server error: %s status=%s body=%s - giving up after %d attempt(s)", url, response.status_code, response.text[:1000], max_attempts)
        return _result(False, response.status_code, error="server_error")

    if response.status_code == 404:
        # A wrong endpoint PATH, not a bad request - this is a bug in our
        # own URL construction (or a changed API), never something the
        # user can fix by "trying again later". Called out separately so
        # it can't hide behind a generic "validation_error" again.
        logger.error(
            "Doctors/Specialties API endpoint NOT FOUND (404) - check the URL path is correct: %s body=%s",
            url, response.text[:500],
        )
        return _result(False, response.status_code, error="endpoint_not_found")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error(
            "Doctors/Specialties API validation error: %s status=%s body=%s rejected_fields=%s",
            url, response.status_code, response.text[:1000],
            [d["field"] for d in details] or "unknown",
        )
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def get_specialties(base_url: str, page_size: int = 200, client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    """POST {base_url}/api/Specialties/GetList.

    NOTE ON THE PATH: the Swagger UI labels this operation
    "GetSpecialtiesPagedList", but that's the operation ID, NOT the HTTP
    path - the actual path is /api/Specialties/GetList. This was
    originally coded as /api/Specialties/GetSpecialtiesPagedList, which
    returned 404 and surfaced to the user as a vague "technical problem"
    (any 4xx was being reported as "validation_error"). Confirmed by the
    same pattern on the Doctors endpoint, whose Swagger operation ID is
    "GetDoctorsPagedList" but whose real path is /api/Doctors/GetList.

    Returns every specialty this clinic offers (scoped by base_url alone,
    confirmed directly - no separate organizationId/branchId needed)."""

    url = f"{base_url}/api/Specialties/GetList"
    # NOTE: pageNumber must be 1 or above, NOT 0 - confirmed directly
    # from the API's own error response ("PageNumber should be above
    # one", thrown by PagingOptions.set_PageNumber). The Swagger UI's
    # example body shows "pageNumber": 0, but that's just a placeholder
    # default and is rejected at runtime with a 500.
    payload = {"pageNumber": 1, "pageSize": page_size}

    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctors(
    base_url: str,
    specialty_ids: Optional[list] = None,
    branch_ids: Optional[list] = None,
    service_ids: Optional[list] = None,
    has_published_service: bool = True,
    has_service_schedule: bool = True,
    intersection_start: Optional[str] = None,
    intersection_end: Optional[str] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Doctors/GetList.

    `has_published_service`/`has_service_schedule`/`intersection_start`/
    `intersection_end` are REQUEST filter fields (confirmed directly from
    the API's own request schema) - they narrow results to doctors who
    are actually bookable with an available schedule intersecting the
    given time window. The response itself then includes `hasSlots` per
    doctor reflecting that same filter.

    `branch_ids` filters to doctors who work at any of the given
    branches - confirmed as a real request field, used by the New
    Booking flow's branch-first selection path."""

    url = f"{base_url}/api/Doctors/GetList"
    payload = {
        # Must be 1 or above, not 0 - see the note in get_specialties()
        "pageNumber": 1,
        "pageSize": page_size,
        "hasPublishedService": has_published_service,
        "hasServiceSchedule": has_service_schedule,
    }

    if specialty_ids:
        payload["specialtyIds"] = specialty_ids
    if branch_ids:
        payload["branchIds"] = branch_ids
    if service_ids:
        # Confirmed request field: narrows to doctors who actually
        # provide the given service(s). Used when the patient picked a
        # SERVICE first ("فحص النظر") - the doctors for that service are
        # the answer, and re-asking "specialty or doctor?" throws the
        # choice they already made away.
        payload["serviceIds"] = service_ids
    if intersection_start:
        payload["intersectionStart"] = intersection_start
    if intersection_end:
        payload["intersectionEnd"] = intersection_end

    return _post_json(url, payload, client_id=client_id, language=language)


# ==========================================================
# Doctor Schedule / Reschedule (Reschedule Appointment feature)
# ==========================================================
#
# All three endpoints confirmed directly from the API's own Swagger
# "Execute" output, same demo server/port as Doctors/Specialties.

def get_branches(
    base_url: str,
    search_query: Optional[str] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Branches/GetList.

    Returns this clinic's branch list - name/altName/address/city/
    country/contact info per branch (confirmed directly from the API's
    real response). `search_query` is optional server-side filtering;
    the caller may also just fetch all and match client-side."""

    url = f"{base_url}/api/Branches/GetList"
    payload = {"pageNumber": 1, "pageSize": page_size}

    if search_query:
        payload["searchQuery"] = search_query

    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctor_schedule(
    base_url: str,
    doctor_ids: list,
    branch_ids: Optional[list] = None,
    effective_date: Optional[str] = None,
    page_size: int = 50,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
    include_future: bool = False,
) -> dict:
    """POST {base_url}/api/DoctorSchedules/GetList.

    Returns the doctor's GENERAL RECURRING schedule (which weekdays they
    work, and their daily start/end times, and the date range this
    schedule is valid for) - NOT specific available time slots. Each
    item has recurringDaysNames/fromDateTime/toDateTime among other
    fields (confirmed directly from the API's real response).

    `branch_ids`, when given, narrows to that specific branch's schedule
    only - used by the New Booking flow once a branch is confirmed
    (otherwise the schedule spans every branch the doctor works at).

    `effective_date` (e.g. "2026-07-30"), when given, excludes EXPIRED
    schedule rows - the row's own validity END must be on or after this
    date (`toDateTimeFrom`).

    `include_future=False` (the default) ALSO requires the row to have
    already started (`fromDateTimeTo`), i.e. it must be valid on exactly
    that date. Correct when asking about one specific day.

    `include_future=True` drops that second condition, so a rota the
    clinic has published for a LATER period is returned too. Use it for
    any general "when does this doctor work?" question - clinics publish
    the next season's rota in advance so patients can book into it, and
    hiding it makes the doctor look less available than they are."""

    url = f"{base_url}/api/DoctorSchedules/GetList"
    payload = {"pageNumber": 1, "pageSize": page_size, "doctorIds": doctor_ids}

    if branch_ids:
        payload["branchIds"] = branch_ids

    if effective_date:
        # `toDateTimeFrom` excludes EXPIRED rows: the schedule's validity
        # must end on or after this date. That is the part worth
        # filtering - a lapsed schedule is not bookable.
        payload["toDateTimeFrom"] = effective_date

        # `fromDateTimeTo` would additionally require the schedule to
        # have ALREADY STARTED, and that is only correct when asking
        # about one specific date.
        #
        # For a general "when does this doctor work?" it is wrong: a
        # clinic publishes next season's rota in advance precisely so
        # patients can book into it. Confirmed in production - a doctor
        # had Thursdays (valid until 01/09) and Mondays (valid from
        # 01/10), and the Mondays were invisible, so the assistant
        # reported the doctor works only Thursdays while the clinic had
        # deliberately opened Monday bookings.
        if not include_future:
            payload["fromDateTimeTo"] = effective_date

    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctor_schedule_slots(
    base_url: str,
    doctor_ids: list,
    from_date: str,
    to_date: str,
    is_booked: bool = False,
    branch_ids: Optional[list] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Doctors/GetDoctorScheduleSlots.

    Returns SPECIFIC time slots within [from_date, to_date] - the actual
    bookable times, not just working days. `is_booked=False` (default)
    filters to only slots that are NOT already taken - i.e. genuinely
    available ones. `branch_ids` additionally narrows to a specific
    branch (confirmed real request field) - needed for the New Booking
    flow once both a doctor AND branch are confirmed. Each item has
    slotStart/slotEnd/isBooked among other fields (confirmed directly
    from the API's real response)."""

    url = f"{base_url}/api/Doctors/GetDoctorScheduleSlots"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "fromDate": from_date,
        "toDate": to_date,
        "isBooked": is_booked,
        "doctorIds": doctor_ids,
    }

    if branch_ids:
        payload["branchIds"] = branch_ids

    return _post_json(url, payload, client_id=client_id, language=language)


def get_doctor_fees(
    base_url: str,
    doctor_ids: list,
    is_published: bool = True,
    page_size: int = 1000,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/DoctorServices/GetList.

    Returns a doctor's published services and prices - confirmed
    directly from a real production n8n workflow's request/response
    handling (extracts serviceName/price per item)."""

    url = f"{base_url}/api/DoctorServices/GetList"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "isPublished": is_published,
        "doctorIds": doctor_ids,
    }

    return _post_json(url, payload, client_id=client_id, language=language)


def get_services(
    base_url: str,
    branch_ids: Optional[list] = None,
    is_published: bool = True,
    page_size: int = 500,
    client_id: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Services/GetList.

    The clinic's real SERVICE CATALOGUE, straight from the system -
    optionally narrowed to the branches that actually provide each
    service via `branch_ids`, and to published services only via
    `is_published`.

    NOT the same thing as the services section of the knowledge base
    file: that one is marketing copy describing the hospital's service
    lines as a whole, with no per-branch information at all. When the
    question is "what services does THIS BRANCH provide?", only this
    endpoint can answer it - see tools.list_branch_services.

    `pageNumber` must be 1 or above, not 0 - same as every other
    paged endpoint here (see get_specialties()'s note)."""

    url = f"{base_url}/api/Services/GetList"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "isPublished": is_published,
    }

    if branch_ids:
        payload["branchIds"] = branch_ids

    return _post_json(url, payload, client_id=client_id, language=language)


def get_patient_info(
    base_url: str,
    mobile_number: str,
    page_size: int = 1000,
    client_id: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/GuestPatients/GetList.

    Looks up whether a patient is already registered by phone number -
    confirmed directly from a real production n8n workflow. Returns
    items with patientFullName/mobileNumber/email when found; an empty
    result (totalCount=0) means this phone number is not registered
    yet, so the caller should collect name/email fresh."""

    url = f"{base_url}/api/GuestPatients/GetList"
    payload = {
        "pageNumber": 1,
        "pageSize": page_size,
        "mobileNumber": mobile_number,
    }

    return _post_json(url, payload, client_id=client_id)


def _put_json(url: str, payload: dict, client_id: Optional[str] = None) -> dict:
    """Generic PUT + envelope handling, mirroring _post_json exactly but
    for the one confirmed PUT endpoint (GuestBookings/Update)."""

    logger.info("PUT %s payload=%s", url, payload)

    response, last_timeout, last_exc = _request_with_retry(
        "put", url, json=payload, headers=_headers(client_id=client_id),
    )

    if response is None:
        if last_timeout:
            logger.warning("Request timed out: %s", url)
            return _result(False, error="timeout")
        logger.exception("Request failed: %s", url)
        return _result(False, error=str(last_exc) if last_exc else "request_failed")

    if response.status_code >= 500:
        logger.error("GuestBookings/Update server error: %s status=%s payload=%s body=%s", url, response.status_code, payload, response.text[:1000])
        return _result(False, response.status_code, error="server_error")

    if response.status_code == 404:
        logger.error("GuestBookings/Update endpoint NOT FOUND (404): %s payload=%s body=%s", url, payload, response.text[:500])
        return _result(False, response.status_code, error="endpoint_not_found")

    if response.status_code >= 400:
        details = _validation_details(response)
        logger.error(
            "GuestBookings/Update validation error: %s status=%s payload=%s body=%r headers=%s rejected_fields=%s",
            url, response.status_code, payload, response.text[:1000], dict(response.headers),
            [d["field"] for d in details] or "unknown",
        )
        return _result(False, response.status_code, error="validation_error", details=details)

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def reschedule_booking(
    base_url: str,
    booking_id: str,
    new_from: str,
    new_to: str,
    client_id: Optional[str] = None,
) -> dict:
    """PUT {base_url}/api/GuestBookings/Update.

    Changes an EXISTING booking's time. `booking_id` is the booking's own
    GUID `id` field (NOT the human-readable bookingRefNum) - confirmed
    directly from the API's real request schema: {"id", "fromBookingTime",
    "toBookingTime"}."""

    url = f"{base_url}/api/GuestBookings/Update"
    payload = {
        "id": booking_id,
        "fromBookingTime": new_from,
        "toBookingTime": new_to,
    }

    return _put_json(url, payload, client_id=client_id)


def create_booking(
    base_url: str,
    patient_full_name: str,
    mobile_number: str,
    branch_id: str,
    doctor_id: str,
    service_id: str,
    service_price,
    booking_time_from: str,
    booking_time_to: str,
    specialty_id: str,
    doctor_schedule_id: str,
    space_id: str,
    email: str = "",
    client_id: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/GuestBookings/Reservation.

    Creates a brand new booking - confirmed directly from a real
    production n8n workflow's exact field list. ALL the id fields
    (branchId, doctorId, serviceId, servicePrice, specialtyId,
    doctorScheduleId, spaceId) must come from a slot the caller just
    re-verified is still available (via get_doctor_schedule_slots) -
    never invented or reused from an earlier, potentially-stale lookup.
    Returns the raw API response - `data` is the new booking's own GUID
    id (pass this to get_booking_by_id to retrieve its bookingRefNum)."""

    url = f"{base_url}/api/GuestBookings/Reservation"
    payload = {
        "patientFullName": patient_full_name,
        "mobileNumber": mobile_number,
        "email": email,
        "branchId": branch_id,
        "doctorId": doctor_id,
        "serviceId": service_id,
        "servicePrice": service_price,
        "bookingTimeFrom": booking_time_from,
        "bookingTimeTo": booking_time_to,
        "specialtyId": specialty_id,
        "doctorScheduleId": doctor_schedule_id,
        "spaceId": space_id,
    }

    return _post_json(url, payload, client_id=client_id)


def get_booking_by_id(base_url: str, booking_id: str, client_id: Optional[str] = None) -> dict:
    """POST {base_url}/api/GuestBookings/Get.

    Fetches a single booking's full details by its own GUID id (as
    opposed to get_bookings_by_ref, which looks up by the human-readable
    bookingRefNum/phone) - confirmed directly from the production n8n
    reference. Used right after create_booking succeeds, to read back
    the new booking's bookingRefNum to show the user."""

    url = f"{base_url}/api/GuestBookings/Get"
    payload = {"id": booking_id}

    return _post_json(url, payload, client_id=client_id)
