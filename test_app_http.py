"""
HTTP-level tests for the rewritten app.py. Same FakeLLM approach as
test_agent_graph.py, driven through the actual FastAPI TestClient.

Run with:
    python3 test_app_http.py
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import graph
from app import app

client = TestClient(app)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        return self._responses.pop(0)


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_health():
    section("Health check")
    r = client.get("/health")
    print(r.status_code, r.json())
    assert r.status_code == 200 and r.json()["status"] == "ok"
    print("PASSED")


def test_full_conversation_over_http():
    section("HTTP: greeting -> phone -> OTP mismatch -> confirm -> cancelled")

    booking = {
        "id": "GUID-P1", "bookingRefNum": "GBN-P1", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-09-05T15:00:00",
        "mobileNumber": "+201099999999",
    }

    graph._llm_with_tools = FakeLLM([
        AIMessage(content="Would you like to cancel using your booking reference or your phone number?"),
        _tool_call("validate_phone_format", {"phone": "+201003365691"}, "c1"),
        _tool_call("compare_phone", {"provided_phone": "+201003365691", "channel_phone": "+201111111111"}, "c2"),
        _tool_call("send_otp", {"phone": "+201099999999"}, "c3"),
        AIMessage(content="An OTP was sent to the number on file. Please enter it."),
        _tool_call("verify_otp", {"phone": "+201099999999", "otp": "123456"}, "c4"),
        _tool_call("lookup_appointment", {"ref_number": "", "phone": "+201003365691"}, "c5"),
        AIMessage(content="Found your booking with Dr. Omar on 05/09/2026. Shall I cancel it?"),
        _tool_call("check_booking_status", {"ref_number": "GBN-P1"}, "c6"),
        _tool_call("cancel_appointment", {"booking_id": "GUID-P1"}, "c7"),
        AIMessage(content="Done - your appointment has been cancelled."),
    ])

    with patch("api.get_bookings_by_phone", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}):

        r = client.post("/chat", json={"session_id": "http-1", "client_id": "Dar El Oyoun-demo", "message": "I want to cancel"})
        print(1, r.status_code, r.json())
        assert r.status_code == 200

        r = client.post("/chat", json={
            "session_id": "http-1", "client_id": "Dar El Oyoun-demo",
            "message": "+201003365691", "channel_phone": "+201111111111",
        })
        print(2, r.status_code, r.json())
        assert "otp" in r.json()["reply"].lower()

        r = client.post("/chat", json={"session_id": "http-1", "client_id": "Dar El Oyoun-demo", "message": "123456"})
        print(3, r.status_code, r.json())
        assert "cancel" in r.json()["reply"].lower()

        r = client.post("/chat", json={"session_id": "http-1", "client_id": "Dar El Oyoun-demo", "message": "yes"})
        print(4, r.status_code, r.json())
        assert "cancelled" in r.json()["reply"].lower() or "cancel" in r.json()["reply"].lower()

    print("PASSED")


def test_validation_error():
    section("HTTP: missing required field -> 422")
    r = client.post("/chat", json={"session_id": "x", "message": "hi"})
    print(r.status_code, r.json())
    assert r.status_code == 422
    print("PASSED")


def test_internal_error_returns_500():
    section("HTTP: unexpected exception -> 500 JSON, not a stack trace")

    def boom(messages):
        raise RuntimeError("boom")

    graph._llm_with_tools = FakeLLM([])
    graph._llm_with_tools.invoke = boom

    r = client.post("/chat", json={"session_id": "http-err-1", "client_id": "Dar El Oyoun-demo", "message": "hi"})
    print(r.status_code, r.json())
    assert r.status_code == 500

    print("PASSED")


if __name__ == "__main__":
    test_health()
    test_full_conversation_over_http()
    test_validation_error()
    test_internal_error_returns_500()
    print("\nALL HTTP TESTS PASSED\n")
