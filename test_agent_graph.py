"""
Tests for the rewritten LLM-tool-calling graph.

Since this architecture makes EVERY decision through the LLM, testing it
end-to-end requires scripting what the LLM would decide at each step of
a realistic conversation - so these tests replace graph._llm_with_tools
with a small FakeLLM that returns a pre-scripted sequence of AIMessages
(some with tool_calls, some without), while every actual tool call still
runs for real against a MOCKED api.py (never real network). This
verifies the agent<->tools loop, InjectedState wiring, and MemorySaver
checkpointing all work together correctly - not just that the code
imports.

Run with:
    python3 test_agent_graph.py
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage

import config
import graph
import main as agent
import tools


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


class FakeLLM:
    """Pops one scripted AIMessage per .invoke() call, in order."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)

    def remaining(self):
        return len(self._responses)


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_reference_cancel_full_conversation():
    section("Reference cancel: ask method -> lookup -> confirm -> check_status -> cancel -> success")

    booking = {
        "id": "GUID-1", "bookingRefNum": "GBN-2026-06-20-151", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201001255864",
    }

    fake = FakeLLM([
        # Turn 1: user says "I want to cancel my appointment" -> ask which method
        AIMessage(content="Would you like to cancel using your booking reference or phone number?"),
        # Turn 2: user gives the reference -> LLM calls lookup_appointment
        _tool_call("lookup_appointment", {"ref_number": "GBN-2026-06-20-151", "phone": ""}, "call_1"),
        # ...then, seeing found_one, presents it and asks to confirm
        AIMessage(content="I found booking GBN-2026-06-20-151 with Dr. Omar at Downtown on 20/08/2026 1:00 PM. Cancel it?"),
        # Turn 3: user says "yes" -> LLM calls check_booking_status
        _tool_call("check_booking_status", {"ref_number": "GBN-2026-06-20-151"}, "call_2"),
        # ...sees "active" -> calls cancel_appointment with the id
        _tool_call("cancel_appointment", {"booking_id": "GUID-1"}, "call_3"),
        # ...sees "success" -> final natural-language confirmation
        AIMessage(content="Your appointment has been cancelled successfully."),
    ])

    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}) as mock_lookup, \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}) as mock_cancel:

        r1 = agent.send_message("Dar El Oyoun-demo", "sess-ref-1", "I want to cancel my appointment")
        print("Turn 1:", r1)
        assert "reference" in r1.lower() or "phone" in r1.lower()

        r2 = agent.send_message("Dar El Oyoun-demo", "sess-ref-1", "GBN-2026-06-20-151")
        print("Turn 2:", r2)
        assert "GBN-2026-06-20-151" in r2
        assert mock_lookup.call_count == 1

        r3 = agent.send_message("Dar El Oyoun-demo", "sess-ref-1", "yes")
        print("Turn 3:", r3)
        assert "cancel" in r3.lower()
        assert mock_lookup.call_count == 2, "check_booking_status re-fetches by ref, calling get_bookings_by_ref again"
        assert mock_cancel.call_count == 1
        assert mock_cancel.call_args.args[1] == "GUID-1" or mock_cancel.call_args.kwargs.get("booking_guid") == "GUID-1"

    assert fake.remaining() == 0, "every scripted LLM response should have been consumed exactly once"
    print("PASSED")


def test_checkpointer_persists_chat_history_across_turns():
    section("Checkpointer: chat history accumulates across separate send_message() calls")

    fake = FakeLLM([
        AIMessage(content="Hi! How can I help?"),
        AIMessage(content="Sure, go ahead."),
    ])
    graph._llm_with_tools = fake

    agent.send_message("Dar El Oyoun-demo", "sess-persist-1", "hello")
    agent.send_message("Dar El Oyoun-demo", "sess-persist-1", "I want to cancel")

    snapshot = graph.graph.get_state(agent._config_for("sess-persist-1"))
    history = snapshot.values["messages"]
    print("Accumulated messages:", [(m.type, m.content) for m in history])

    # 2 human + 2 AI = 4 messages accumulated in one thread's history
    assert len(history) == 4
    assert history[0].type == "human" and history[0].content == "hello"
    assert history[2].type == "human" and history[2].content == "I want to cancel"

    print("PASSED")


def test_load_config_reloads_every_turn_so_prompt_updates_apply_immediately():
    section("load_config reloads CSVs/rebuilds the system prompt every turn (no stale per-thread caching)")

    fake = FakeLLM([AIMessage(content="ok"), AIMessage(content="ok again")])
    graph._llm_with_tools = fake

    with patch("config.get_messages", wraps=__import__("config").get_messages) as spy:
        agent.send_message("Dar El Oyoun-demo", "sess-cfg-1", "hi")
        agent.send_message("Dar El Oyoun-demo", "sess-cfg-1", "hi again")
        print("config.get_messages call count across 2 turns:", spy.call_count)
        assert spy.call_count == 2, (
            "templates/system_prompt must be rebuilt every turn, not cached per-thread - "
            "otherwise a prompts.py update never reaches a conversation already in progress"
        )

    print("PASSED")


def test_injected_state_hides_base_url_from_llm_schema():
    section("Tool schemas never expose base_url/state to the LLM (prevents URL hallucination)")

    import tools
    for t in tools.ALL_TOOLS:
        assert "state" not in t.args, f"{t.name} leaks 'state' into its LLM-visible schema"
        assert "base_url" not in t.args, f"{t.name} leaks 'base_url' into its LLM-visible schema"

    print("PASSED")


def test_channel_identity_auto_lookup_skips_otp_and_phone_prompt():
    section("NEW: phone cancel using wa_id automatically - no 'type your number' question, no OTP")

    booking = {
        "id": "GUID-WA", "bookingRefNum": "GBN-WA", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201155611045",
    }

    fake = FakeLLM([
        AIMessage(content="Would you like to cancel using your booking reference or phone number?"),
        _tool_call("lookup_appointment", {"use_channel_identity": True, "language": "en"}, "call_1"),
        AIMessage(content="Found your booking with Dr. Omar. Shall I cancel it?"),
        _tool_call("check_booking_status", {"ref_number": "GBN-WA", "language": "en"}, "call_2"),
        _tool_call("cancel_appointment", {"booking_id": "GUID-WA"}, "call_3"),
        AIMessage(content="Your appointment has been cancelled successfully."),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_phone", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}), \
         patch("tools.send_otp") as mock_send_otp:

        r1 = agent.send_message("Dar El Oyoun-demo", "sess-wa-1", "cancel my booking", channel_phone="+201155611045")
        print("Turn 1:", r1)

        r2 = agent.send_message("Dar El Oyoun-demo", "sess-wa-1", "phone")
        print("Turn 2:", r2)
        assert "Omar" in r2

        r3 = agent.send_message("Dar El Oyoun-demo", "sess-wa-1", "yes")
        print("Turn 3:", r3)
        assert "cancel" in r3.lower()
        assert mock_send_otp.call_count == 0, "OTP must never be sent when the booking was found via the user's own verified channel identity"

    print("PASSED")


def test_no_channel_identity_falls_back_to_asking_for_phone():
    section("use_channel_identity with no channel_phone available -> 'no_channel_identity' status")

    fake_state = {
        "client_id": "x", "session_id": "s", "channel_phone": None,
        "templates": {}, "system_prompt": "", "messages": [], "greeted": False,
    }
    result = tools.lookup_appointment.invoke({"state": fake_state, "use_channel_identity": True})
    print(result)
    assert result["status"] == "no_channel_identity"

    print("PASSED")


def test_inactivity_timeout_starts_fresh_conversation():
    section("Session timeout: a long gap between messages starts a brand new conversation")

    fake = FakeLLM([AIMessage(content="Hi, first conversation"), AIMessage(content="Hi, brand new conversation")])
    graph._llm_with_tools = fake

    with patch("main._now", side_effect=[1000.0, 1000.0 + config.SESSION_TIMEOUT_SECONDS + 60]):
        agent.send_message("Dar El Oyoun-demo", "sess-timeout-1", "hello")
        gen_before = agent._generation.get("sess-timeout-1", 0)

        agent.send_message("Dar El Oyoun-demo", "sess-timeout-1", "hello again much later")
        gen_after = agent._generation.get("sess-timeout-1", 0)

    print("generation before:", gen_before, "-> after:", gen_after)
    assert gen_after == gen_before + 1

    thread_id = f"guest-cancel:sess-timeout-1:{gen_after}"
    snapshot = graph.graph.get_state({"configurable": {"thread_id": thread_id}})
    print("messages in the new (post-timeout) thread:", [m.content for m in snapshot.values["messages"]])
    assert len(snapshot.values["messages"]) == 2, "the fresh thread should only have this turn's messages, not the old conversation's"

    print("PASSED")


def test_memory_stays_after_quick_followup_but_resets_after_post_success_silence():
    section("No immediate reset after success; quick follow-up continues same chat; 10+min silence after success resets")

    booking = {
        "id": "GUID-RS", "bookingRefNum": "GBN-RS", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201001255864",
    }

    fake = FakeLLM([
        AIMessage(content="Would you like to cancel using your booking reference or phone number?"),
        _tool_call("lookup_appointment", {"ref_number": "GBN-RS"}, "c1"),
        AIMessage(content="Found it, confirm?"),
        _tool_call("check_booking_status", {"ref_number": "GBN-RS"}, "c2"),
        _tool_call("cancel_appointment", {"booking_id": "GUID-RS"}, "c3"),
        AIMessage(content="Cancelled successfully!"),
        AIMessage(content="Sure, here's that other info (same conversation, no greeting)"),
        AIMessage(content="Hello again! (fresh conversation, greeting shown)"),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}):

        t = [1000.0]
        with patch("main._now", side_effect=lambda: t[0]):
            agent.send_message("Dar El Oyoun-demo", "sess-reset-2", "I want to cancel")
            agent.send_message("Dar El Oyoun-demo", "sess-reset-2", "GBN-RS")

            reply = agent.send_message("Dar El Oyoun-demo", "sess-reset-2", "yes")
            gen_right_after_success = agent._generation.get("sess-reset-2", 0)
            print("reply:", reply, "| generation immediately after success (must NOT bump yet):", gen_right_after_success)
            assert gen_right_after_success == 0

            t[0] += 120  # 2 minutes later - well within the 10-minute grace window
            reply2 = agent.send_message("Dar El Oyoun-demo", "sess-reset-2", "thanks, one more question")
            gen_after_quick_followup = agent._generation.get("sess-reset-2", 0)
            print("quick follow-up reply:", reply2, "| generation (must still be SAME):", gen_after_quick_followup)
            assert gen_after_quick_followup == 0

            t[0] += config.POST_SUCCESS_TIMEOUT_SECONDS + 30  # now let 10+ minutes of silence pass after THAT last message... 
            # NOTE: the grace marker was already cleared by the quick follow-up above, so from here it's governed
            # by the general SESSION_TIMEOUT_SECONDS, not POST_SUCCESS_TIMEOUT_SECONDS - confirm no reset yet since
            # POST_SUCCESS_TIMEOUT_SECONDS (10min) < SESSION_TIMEOUT_SECONDS (1hr).
            reply3 = agent.send_message("Dar El Oyoun-demo", "sess-reset-2", "still here?")
            gen_after_general_gap = agent._generation.get("sess-reset-2", 0)
            print("after 10+min gap (but under 1hr general timeout), generation (must still be SAME):", gen_after_general_gap)
            assert gen_after_general_gap == 0

    print("PASSED")


def test_resets_after_post_success_silence_without_followup():
    section("A genuine 10+ minute silence right after success (no follow-up at all) DOES reset")

    booking = {
        "id": "GUID-RS2", "bookingRefNum": "GBN-RS2", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201001255864",
    }

    fake = FakeLLM([
        _tool_call("check_booking_status", {"ref_number": "GBN-RS2"}, "c1"),
        _tool_call("cancel_appointment", {"booking_id": "GUID-RS2"}, "c2"),
        AIMessage(content="Cancelled successfully!"),
        AIMessage(content="Hello! (fresh conversation, greeting shown)"),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}):

        t = [1000.0]
        with patch("main._now", side_effect=lambda: t[0]):
            agent.send_message("Dar El Oyoun-demo", "sess-reset-3", "yes")
            gen_before = agent._generation.get("sess-reset-3", 0)

            t[0] += config.POST_SUCCESS_TIMEOUT_SECONDS + 30
            reply = agent.send_message("Dar El Oyoun-demo", "sess-reset-3", "good morning")
            gen_after = agent._generation.get("sess-reset-3", 0)
            print("reply:", reply, "| generation before/after post-success silence:", gen_before, "->", gen_after)
            assert gen_after == gen_before + 1

    print("PASSED")


def test_greeting_prepended_deterministically_and_only_once():
    section("Deterministic greeting: guaranteed on turn 1, never repeated, no duplicate if LLM already said it")

    fake = FakeLLM([
        AIMessage(content="ممكن أعرف أساعدك إزاي؟"),
        AIMessage(content="Sure, tell me the reference number."),
    ])
    graph._llm_with_tools = fake

    reply1 = agent.send_message("Dar El Oyoun-demo", "sess-greet-1", "مرحبا")
    print("turn 1:", reply1)
    assert "أنا لطيفة" in reply1 and "ممكن أعرف أساعدك إزاي؟" in reply1

    reply2 = agent.send_message("Dar El Oyoun-demo", "sess-greet-1", "عايز ألغي")
    print("turn 2:", reply2)
    assert "أنا لطيفة" not in reply2, "greeting must not repeat on later turns of the same conversation"

    print("PASSED")


def test_fresh_thread_first_turn_going_straight_to_a_tool_call_does_not_crash():
    section("REGRESSION: a brand-new thread whose first turn calls a tool immediately (no plain-text reply first) must not crash")

    booking = {
        "id": "GUID-FIRST", "bookingRefNum": "GBN-FIRST", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201001255864",
    }

    fake = FakeLLM([
        _tool_call("lookup_appointment", {"ref_number": "GBN-FIRST"}, "c1"),
        AIMessage(content="Found it, confirm?"),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}):
        # This is the FIRST EVER message on this thread, and the LLM's
        # first action is a tool call (not a plain-text reply) - this is
        # exactly the scenario that broke before the fix: InjectedState's
        # strict validation failed because "greeted" was entirely absent
        # from state at the point the tool call executed.
        reply = agent.send_message("Dar El Oyoun-demo", "sess-firstcall-1", "GBN-FIRST")
        print("reply:", reply)
        assert "confirm" in reply.lower()

    print("PASSED")


def test_time_of_day_salutation_and_no_premature_ref_phone_question():
    section("Bare greeting: time-aware salutation swapped in, no jump ahead to ref/phone question")

    fake = FakeLLM([AIMessage(content="")])
    graph._llm_with_tools = fake

    reply = agent.send_message("Dar El Oyoun-demo", "sess-salut-1", "صباح الخير")
    print(reply)
    assert reply.startswith("صباح النور")
    assert "أنا لطيفة" in reply
    assert "رقم الحجز" not in reply and "رقم الموبايل" not in reply, (
        "must not jump ahead to asking for ref/phone before the user states cancellation intent"
    )

    print("PASSED")


def test_default_salutation_when_no_time_of_day_cue():
    section("No time-of-day cue in the message -> falls back to the template's own default opening line")

    fake = FakeLLM([AIMessage(content="")])
    graph._llm_with_tools = fake

    reply = agent.send_message("Dar El Oyoun-demo", "sess-salut-2", "أهلا")
    print(reply)
    assert reply.startswith("أهلاً بيك")

    print("PASSED")


def test_filter_active_matches_confirmed_dashboard_status_vocabulary():
    section("_filter_active matches the real dashboard status vocabulary (جديد/تم التأكيد/وصل/لم يحضر/مكتمل)")

    items = [
        {"id": "new", "statusName": "جديد", "status": 1, "bookingTimeFrom": "2026-12-01T10:00:00"},
        {"id": "confirmed", "statusName": "تم التأكيد", "status": 2, "bookingTimeFrom": "2026-12-01T10:00:00"},
        {"id": "arrived", "statusName": "وصل", "status": 3, "bookingTimeFrom": "2026-12-01T10:00:00"},
        {"id": "noshow", "statusName": "لم يحضر", "status": 4, "bookingTimeFrom": "2026-12-01T10:00:00"},
        {"id": "completed", "statusName": "مكتمل", "status": 5, "bookingTimeFrom": "2026-12-01T10:00:00"},
    ]
    result = tools._filter_active(items)
    ids = [i["id"] for i in result]
    print("remaining:", ids)
    assert ids == ["new", "confirmed"], "only New and Confirmed should remain cancellable"

    print("PASSED")


def test_pure_english_conversation_never_gets_arabic_greeting_or_leak():
    section("REGRESSION: a purely English conversation must never get the Arabic CSV greeting forced in")

    fake = FakeLLM([AIMessage(content="Hi! I'm Latifa from Dar El Oyoun Hospitals. How can I help you today?")])
    graph._llm_with_tools = fake

    reply = agent.send_message("Dar El Oyoun-demo", "sess-english-1", "I want to cancel my appointment")
    print(reply)
    assert not graph._looks_arabic(reply), "an English-first conversation must not contain any Arabic text"

    print("PASSED")


def test_english_conversation_gets_deterministic_english_greeting_template():
    section("English conversation gets the deterministic English greeting template (not an LLM-improvised one)")

    fake = FakeLLM([AIMessage(content="")])
    graph._llm_with_tools = fake

    reply = agent.send_message("Dar El Oyoun-demo", "sess-eng-template-1", "I want to cancel my appointment")
    print(reply)
    assert reply.startswith("Hi there!")
    assert "Latifa" in reply and "Dar El Oyoun Hospitals" in reply
    assert "Booking a new appointment" in reply
    assert not graph._looks_arabic(reply)

    print("PASSED")


def test_send_otp_safeguard_when_llm_skips_compare_phone():
    section("REGRESSION (found in production logs): LLM calls send_otp directly for a number matching channel_phone, skipping compare_phone")

    booking = {
        "id": "GUID-SAFE", "bookingRefNum": "GBN-SAFE", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201155611045",
    }

    # Deliberately skips compare_phone entirely - exactly what was observed
    # in production logs - to prove the send_otp-level safeguard catches it.
    fake = FakeLLM([
        _tool_call("send_otp", {"phone": "+201155611045"}, "c1"),
        _tool_call("lookup_appointment", {"phone": "+201155611045", "language": "en"}, "c2"),
        AIMessage(content="Found it, confirm cancellation?"),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_phone", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}), \
         patch("tools._otp_storage", {}):

        reply = agent.send_message(
            "Dar El Oyoun-demo", "sess-safeguard-1",
            "Cancel appointment for these +201155611045",
            channel_phone="+201155611045",
        )
        print(reply)
        assert "+201155611045" not in tools._otp_storage, "no OTP should actually have been generated/stored"
        assert "confirm" in reply.lower()

    print("PASSED")


def test_filter_active_falls_back_to_string_matching_when_status_code_missing():
    section("_filter_active falls back to string matching only when the numeric status code is entirely missing")

    items = [
        {"id": "new-no-code", "statusName": "New", "status": None, "bookingTimeFrom": "2026-12-01T10:00:00"},
        {"id": "cancelled-no-code", "statusName": "Cancelled", "status": None, "bookingTimeFrom": "2026-12-01T10:00:00"},
    ]
    result = tools._filter_active(items)
    ids = [i["id"] for i in result]
    print("remaining:", ids)
    assert ids == ["new-no-code"]

    print("PASSED")


def test_lookup_appointment_requests_cancellable_statuses_from_api():
    section("lookup_appointment asks the Booking API to filter by statusList=[1, 2] directly")

    fake_state = {
        "client_id": "x", "session_id": "s", "channel_phone": None,
        "templates": {"_base_url": "https://demo.catalystsystems.io:1102"},
        "system_prompt": "", "messages": [], "greeted": False,
    }

    with patch("api.get_bookings_by_phone", return_value={"success": True, "status_code": 200, "data": {"items": []}, "error": None}) as mock_call:
        tools.lookup_appointment.invoke({"state": fake_state, "phone": "+201155611045", "language": "en"})
        _, kwargs = mock_call.call_args
        print("status_list requested:", kwargs.get("status_list"))
        assert kwargs.get("status_list") == [1, 2]

    print("PASSED")


def test_medical_concierge_matched_specialty_with_available_doctor():
    section("Medical Concierge: symptom matches an offered specialty, finds an available doctor")

    specialties_data = {"items": [
        {"id": "sp-psych-1", "name": "Psychiatry", "statusName": "Active"},
    ]}
    doctors_data = {"items": [
        {"id": "doc-1", "name": "Dr. Omar", "specialtyName": "Psychiatry", "degreeName": "Consultant", "hasSlots": True},
    ]}

    fake = FakeLLM([
        _tool_call("list_specialties", {}, "c1"),
        _tool_call("find_available_doctors", {"specialty_id": "sp-psych-1"}, "c2"),
        AIMessage(content="It sounds like a psychiatry consultation could help. Dr. Omar (Consultant) has availability - would you like to proceed?"),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_specialties", return_value={"success": True, "status_code": 200, "data": specialties_data, "error": None}), \
         patch("api.get_doctors", return_value={"success": True, "status_code": 200, "data": doctors_data, "error": None}):

        reply = agent.send_message("tanasuq-saudi", "sess-medconcierge-1", "I've been feeling really anxious lately")
        print(reply)
        assert "Omar" in reply

    print("PASSED")


def test_medical_concierge_specialty_not_offered_at_this_clinic():
    section("Medical Concierge: symptom would need a specialty this clinic does NOT offer")

    specialties_data = {"items": [
        {"id": "sp-psych-1", "name": "Psychiatry", "statusName": "Active"},
    ]}

    fake = FakeLLM([
        _tool_call("list_specialties", {}, "c1"),
        AIMessage(content="This sounds like it might need an internal medicine specialist, but that isn't something we offer here at Tanasuq Medical Center - I'd recommend looking into that specialty elsewhere."),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_specialties", return_value={"success": True, "status_code": 200, "data": specialties_data, "error": None}):
        reply = agent.send_message("tanasuq-saudi", "sess-medconcierge-2", "I have really bad stomach pain and indigestion")
        print(reply)
        assert "tanasuq" in reply.lower() or "here" in reply.lower()

    print("PASSED")


def test_ordinary_anxiety_mention_is_not_treated_as_crisis():
    section("REGRESSION: plain 'عندي قلق' at an eye clinic -> normal specialty-mismatch, NOT crisis/human-handoff")

    specialties_data = {"items": [
        {"id": "sp-eye-1", "name": "Ophthalmology", "statusName": "Active"},
    ]}

    fake = FakeLLM([
        _tool_call("list_specialties", {}, "c1"),
        AIMessage(content="It sounds like this might need a psychiatrist, but that isn't something we offer here at Dar El Oyoun Hospitals - I'd recommend looking into that specialty elsewhere."),
    ])
    graph._llm_with_tools = fake

    with patch("api.get_specialties", return_value={"success": True, "status_code": 200, "data": specialties_data, "error": None}):
        reply = agent.send_message("Dar El Oyoun-demo", "sess-anxiety-1", "عندي قلق")
        print(reply)
        assert "customer service" not in reply.lower() and "staff" not in reply.lower(), (
            "an ordinary anxiety mention must not immediately trigger human handoff"
        )

    print("PASSED")


if __name__ == "__main__":
    test_reference_cancel_full_conversation()
    test_checkpointer_persists_chat_history_across_turns()
    test_load_config_reloads_every_turn_so_prompt_updates_apply_immediately()
    test_injected_state_hides_base_url_from_llm_schema()
    test_channel_identity_auto_lookup_skips_otp_and_phone_prompt()
    test_no_channel_identity_falls_back_to_asking_for_phone()
    test_inactivity_timeout_starts_fresh_conversation()
    test_memory_stays_after_quick_followup_but_resets_after_post_success_silence()
    test_resets_after_post_success_silence_without_followup()
    test_greeting_prepended_deterministically_and_only_once()
    test_fresh_thread_first_turn_going_straight_to_a_tool_call_does_not_crash()
    test_time_of_day_salutation_and_no_premature_ref_phone_question()
    test_default_salutation_when_no_time_of_day_cue()
    test_filter_active_matches_confirmed_dashboard_status_vocabulary()
    test_pure_english_conversation_never_gets_arabic_greeting_or_leak()
    test_english_conversation_gets_deterministic_english_greeting_template()
    test_send_otp_safeguard_when_llm_skips_compare_phone()
    test_filter_active_falls_back_to_string_matching_when_status_code_missing()
    test_lookup_appointment_requests_cancellable_statuses_from_api()
    test_medical_concierge_matched_specialty_with_available_doctor()
    test_medical_concierge_specialty_not_offered_at_this_clinic()
    test_ordinary_anxiety_mention_is_not_treated_as_crisis()
    print("\nALL AGENT GRAPH TESTS PASSED\n")
