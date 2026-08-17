"""
Tests for the multi-agent layer.

Complements (does not replace) test_agent_graph.py and test_app_http.py,
which still cover the agent<->tools loop, InjectedState wiring and
checkpointing end to end. Those two must keep passing untouched - that
is the actual proof this refactor didn't break anything - so nothing
here modifies them.

What this file proves:
  1. ROUTING       - the right specialist owns the right message.
  2. STICKINESS    - a mid-flow "نعم"/OTP/phone stays with its owner.
  3. RELEASE       - a completed flow lets go of the conversation.
  4. PROMPT SCOPE  - each specialist gets its own flow + the shared core,
                     and NOT the other flows.
  5. TOOL SCOPE    - each specialist is bound to its own tools only,
                     and the ToolNode can still execute anything.
  6. CONTRACT      - every specialist's reply is normalized identically.
  7. SAFETY        - the kill switch and the fail-safe fallbacks work.

Run with:
    python3 test_multiagent.py
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents
import config
import graph
import main as agent_entrypoint
import prompts
import tools
from agents import registry, response_contract, router as router_module


PASSES = []
FAILURES = []


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(label, condition, detail=""):
    if condition:
        PASSES.append(label)
        print(f"  ok    {label}")
    else:
        FAILURES.append(f"{label} {detail}".strip())
        print(f"  FAIL  {label} {detail}")


def human(text):
    return [HumanMessage(content=text)]


# ==========================================================
# 1. ROUTING
# ==========================================================

def test_routing_picks_the_right_specialist():
    section("Routing: a fresh message goes to the specialist that owns it")

    cases = [
        # --- Arabic, several dialects -------------------------------
        ("عايز ألغي الحجز بتاعي",                      "cancel"),
        ("أبغى ألغي موعدي",                            "cancel"),
        ("ممكن إلغاء الحجز؟",                          "cancel"),
        ("عاوز أأجل معادي لبكرة",                      "reschedule"),
        ("ابغى اغير موعد الكشف",                       "reschedule"),
        ("ممكن تعديل الحجز؟",                          "reschedule"),
        ("عايز أحجز كشف عند دكتور",                    "booking"),
        ("ابغى احجز موعد جديد",                        "booking"),
        ("عندي صداع شديد من امبارح",                   "medical"),
        ("بحس بألم في صدري",                           "medical"),
        ("أي دكتور يناسب حالتي؟",                      "medical"),
        ("عندي شكوى على الخدمة",                       "complaint"),
        ("حابب أقدم اقتراح",                           "complaint"),
        ("فين عنوان الفرع؟",                           "faq"),
        ("ايه الخدمات اللي عندكم؟",                    "faq"),
        ("عايز أكلم موظف",                             "concierge"),
        # --- English ------------------------------------------------
        ("I want to cancel my appointment",            "cancel"),
        ("can I reschedule my booking?",               "reschedule"),
        ("I need to book a new appointment",           "booking"),
        ("I have a terrible headache",                 "medical"),
        ("I want to file a complaint",                 "complaint"),
        ("what are your opening hours?",               "faq"),
        ("let me speak to a human",                    "concierge"),
    ]

    for message, expected in cases:
        chosen, reason = agents.route_turn(human(message), active_agent=None)
        check(
            f"{message[:34]:<36} -> {expected}",
            chosen == expected,
            f"(got {chosen!r}: {reason})",
        )


def test_bare_greeting_opens_with_the_concierge():
    section("Routing: a message with no intent yet must not guess a flow")

    for greeting in ("مرحبا", "صباح الخير", "hi", "hello there", "السلام عليكم"):
        chosen, _ = agents.route_turn(human(greeting), active_agent=None)
        check(f"{greeting!r} -> concierge", chosen == agents.CONCIERGE, f"(got {chosen!r})")


# ==========================================================
# 2. STICKINESS
# ==========================================================

def test_mid_flow_messages_stay_with_their_owner():
    section("Stickiness: content-free mid-flow replies keep the same specialist")

    # Exactly the messages that make up most of a real conversation.
    mid_flow = ["نعم", "أيوه", "yes", "١", "3", "123456", "+201001234567",
                "الخميس", "ok", "تمام", "Dr. Omar", "لطفي"]

    for owner in ("cancel", "booking", "reschedule", "complaint", "medical"):
        for message in mid_flow:
            chosen, reason = agents.route_turn(human(message), active_agent=owner)
            check(
                f"{owner} keeps {message!r}",
                chosen == owner,
                f"(got {chosen!r}: {reason})",
            )


def test_a_deliberate_change_of_subject_does_switch():
    section("Stickiness is not stubbornness: a clear new intent still switches")

    cases = [
        ("booking",    "لأ خلاص، عايز ألغي الحجز بتاعي بدل كده", "cancel"),
        ("cancel",     "طيب ممكن أأجل الموعد بدل ما ألغيه؟",     "reschedule"),
        ("faq",        "عايز أحجز موعد جديد",                    "booking"),
        ("medical",    "عندي شكوى على الاستقبال",                "complaint"),
        ("booking",    "I want to speak to a human",             "concierge"),
    ]

    for owner, message, expected in cases:
        chosen, reason = agents.route_turn(human(message), active_agent=owner)
        check(
            f"{owner} -> {expected}",
            chosen == expected,
            f"(got {chosen!r}: {reason})",
        )


def test_completed_flow_is_released():
    section("Release: a finished flow hands the conversation back")

    # A cancellation that actually succeeded, then a neutral follow-up.
    messages = [
        HumanMessage(content="الغي الحجز"),
        AIMessage(content=""),
        ToolMessage(content='{"status": "success"}', name="cancel_appointment", tool_call_id="c1"),
        AIMessage(content="تم الإلغاء"),
        HumanMessage(content="شكرا"),
    ]

    chosen, reason = agents.route_turn(messages, active_agent="cancel")
    check("cancel released after success", chosen == agents.CONCIERGE, f"(got {chosen!r}: {reason})")

    # ...but an UNfinished flow is not released by a neutral message.
    unfinished = [
        HumanMessage(content="الغي الحجز"),
        AIMessage(content="تأكيد الإلغاء؟"),
        HumanMessage(content="ثانية واحدة"),
    ]
    chosen, _ = agents.route_turn(unfinished, active_agent="cancel")
    check("unfinished cancel is NOT released", chosen == "cancel", f"(got {chosen!r})")


def test_routing_is_reproducible():
    section("Determinism: the same message always routes the same way")

    message = "عايز أحجز كشف عند دكتور عيون"
    results = {agents.route_turn(human(message), None)[0] for _ in range(50)}
    check("50 identical routings", len(results) == 1, f"(got {results})")


def test_router_costs_no_llm_call():
    section("Cost: routing never touches the LLM in the default mode")

    called = {"n": 0}

    class ExplodingLLM:
        def invoke(self, *_args, **_kwargs):
            called["n"] += 1
            raise AssertionError("the deterministic router must not call the LLM")

    original = graph._llm
    graph._llm = ExplodingLLM()
    try:
        for message in ("مرحبا", "الغي الحجز", "عندي وجع", "asdfghjkl", ""):
            agents.route_turn(human(message), None)
    finally:
        graph._llm = original

    check("zero LLM calls during routing", called["n"] == 0, f"(got {called['n']})")


# ==========================================================
# 3. PROMPT SCOPING
# ==========================================================

_SHARED_MARKERS = (
    "LANGUAGE & DIALECT",
    "RESPONSE FORMAT CONTRACT",
    "REFERENCE PHRASES",
    "FIXED TEMPLATES",
    "GLOBAL HARD RULES",
    "EVERYTHING THIS ASSISTANT CAN DO",
)

_FLOW_BANNER = {
    "cancel":     "CONVERSATION FLOW",
    "reschedule": "RESCHEDULE FLOW",
    "booking":    "NEW BOOKING FLOW",
    "medical":    "MEDICAL GUIDANCE FLOW",
    "faq":        "GENERAL HOSPITAL INFO",
    "complaint":  "COMPLAINT FLOW",
}


def _banner_present(prompt, banner):
    """A banner counts as present only as an actual ==== heading.

    Checks EVERY occurrence, not just the first: a specialist's own
    "YOUR JOB" text legitimately mentions its flow by name in prose
    ("Follow the\nMEDICAL GUIDANCE FLOW below"), and that mention comes
    before the real heading.
    """

    marker = f"\n{banner}"
    position = prompt.find(marker)

    while position != -1:
        preceding = prompt[max(0, position - 70):position]
        if "=" * 40 in preceding:
            return True
        position = prompt.find(marker, position + 1)

    return False


def test_every_specialist_gets_the_shared_core():
    section("Prompt scope: the shared core reaches every specialist identically")

    templates = config.get_messages("Dar El Oyoun-demo")

    for name in agents.AGENT_NAMES:
        prompt = prompts.build_agent_system_prompt(templates, name)
        for marker in _SHARED_MARKERS:
            check(f"{name} has {marker}", marker in prompt)

    # The contract must be the SAME text everywhere, not a paraphrase.
    contracts = {
        prompts.build_agent_system_prompt(templates, name).count(
            response_contract.RESPONSE_FORMAT_CONTRACT.strip()
        )
        for name in agents.AGENT_NAMES
    }
    check("the contract text is byte-identical for all", contracts == {1}, f"(counts {contracts})")


def test_specialists_do_not_see_other_flows():
    section("Prompt scope: a specialist is not handed its teammates' flows")

    templates = config.get_messages("Dar El Oyoun-demo")

    for name, own_banner in _FLOW_BANNER.items():
        prompt = prompts.build_agent_system_prompt(templates, name)

        check(f"{name} has its own {own_banner}", _banner_present(prompt, own_banner))

        for other, other_banner in _FLOW_BANNER.items():
            if other == name:
                continue
            # Documented exception: reschedule genuinely reuses
            # cancellation's STEP 1-2 to identify and verify the booking.
            if name == "reschedule" and other == "cancel":
                continue
            check(
                f"{name} does NOT have {other_banner}",
                not _banner_present(prompt, other_banner),
            )


def test_concierge_is_the_full_legacy_agent():
    section("Safety net: the fallback specialist is the old agent, unchanged")

    templates = config.get_messages("Dar El Oyoun-demo")
    prompt = prompts.build_agent_system_prompt(templates, agents.CONCIERGE)

    for banner in _FLOW_BANNER.values():
        check(f"concierge has {banner}", _banner_present(prompt, banner))

    check(
        "concierge keeps every tool",
        len(agents.tools_for(agents.CONCIERGE)) == len(tools.ALL_TOOLS),
    )


def test_scoping_actually_shrinks_the_prompt():
    section("The point of all this: specialists carry far less prompt")

    templates = config.get_messages("Dar El Oyoun-demo")
    full = len(prompts.build_system_prompt(templates))

    for name in _FLOW_BANNER:
        scoped = len(prompts.build_agent_system_prompt(templates, name))
        check(
            f"{name} prompt is smaller than the old one "
            f"({scoped / 1024:.0f}K vs {full / 1024:.0f}K)",
            scoped < full,
        )


def test_unknown_agent_falls_back_instead_of_raising():
    section("Fail-safe: an unknown specialist name degrades to the concierge")

    templates = config.get_messages("Dar El Oyoun-demo")

    spec = registry.get_spec("does-not-exist")
    check("unknown name -> concierge spec", spec.name == agents.CONCIERGE)

    prompt = prompts.build_agent_system_prompt(templates, "does-not-exist")
    check("unknown name still yields a usable prompt", len(prompt) > 10_000)

    broken = prompts.build_agent_system_prompt(templates, None)
    check("None still yields a usable prompt", len(broken) > 10_000)


def test_missing_sections_fall_back_to_the_whole_prompt():
    section("Fail-safe: a renamed heading in prompts.py must not lose instructions")

    from agents import sections as sections_module

    pretend_renamed = {
        sections_module.PREAMBLE_KEY: "You are the assistant.",
        "language": "=" * 60 + "\nLANGUAGE\n" + "=" * 60 + "\nspeak their language",
        "__order__": f"{sections_module.PREAMBLE_KEY}\nlanguage",
    }

    result = registry.build_agent_prompt(pretend_renamed, "cancel")
    check("incomplete split -> everything is kept", "speak their language" in result)
    check("incomplete split -> preamble is kept", "You are the assistant." in result)


# ==========================================================
# 4. TOOL SCOPING
# ==========================================================

def test_tool_subsets_are_correct():
    section("Tool scope: specialists only see the tools their flow needs")

    def names(agent_name):
        return {t.name for t in agents.tools_for(agent_name)}

    booking = names("booking")
    check("booking can create a booking", "create_new_booking" in booking)
    check(
        "booking canNOT reach existing bookings (the real production bug)",
        "lookup_appointment" not in booking and "check_booking_status" not in booking,
    )
    check("booking cannot cancel", "cancel_appointment" not in booking)

    cancel = names("cancel")
    check("cancel can cancel", "cancel_appointment" in cancel)
    check("cancel must check status first", "check_booking_status" in cancel)
    check("cancel cannot create bookings", "create_new_booking" not in cancel)

    reschedule = names("reschedule")
    check("reschedule can reschedule", "reschedule_appointment" in reschedule)
    check("reschedule can look the booking up", "lookup_appointment" in reschedule)

    complaint = names("complaint")
    check("complaint can send the email", "send_complaint_email" in complaint)
    check("complaint cannot touch bookings", "create_new_booking" not in complaint)
    check("complaint is a small surface", len(complaint) <= 6, f"({len(complaint)} tools)")

    medical = names("medical")
    check("medical can find doctors", "find_available_doctors" in medical)
    check("medical cannot book", "create_new_booking" not in medical)

    for name in agents.AGENT_NAMES:
        check(f"{name} has at least one tool", len(names(name)) >= 1)


def test_toolnode_can_still_execute_anything():
    section("Safety net: scoping is at binding time, so no call can 404")

    node_tools = {t.name for t in tools.ALL_TOOLS}
    for name in agents.AGENT_NAMES:
        for tool in agents.tools_for(name):
            check(
                f"{name}:{tool.name} is executable by the shared ToolNode",
                tool.name in node_tools,
            )


def test_tool_scoping_can_be_switched_off():
    section("Escape hatch: AGENT_TOOL_SCOPING=false gives every tool back")

    original = config.AGENT_TOOL_SCOPING
    config.AGENT_TOOL_SCOPING = False
    try:
        for name in agents.AGENT_NAMES:
            check(
                f"{name} gets all tools when scoping is off",
                len(agents.tools_for(name)) == len(tools.ALL_TOOLS),
            )
    finally:
        config.AGENT_TOOL_SCOPING = original


# ==========================================================
# 5. THE OUTPUT CONTRACT
# ==========================================================

def test_filler_openers_are_removed():
    section("Contract: replies start with content, never with an acknowledgement")

    cases = [
        ("تمام! رقم الحجز إيه؟",            "رقم الحجز إيه؟"),
        ("طيب، ابعتلي رقم الموبايل",         "ابعتلي رقم الموبايل"),
        ("Sure, go ahead.",                  "Go ahead."),
        ("Of course! Here are the times.",   "Here are the times."),
        ("أبشر، تم الإلغاء",                 "تم الإلغاء"),
    ]

    for raw, expected in cases:
        result, changed = response_contract.normalize_reply(raw)
        check(f"{raw[:26]!r} -> {expected[:26]!r}", result == expected, f"(got {result!r})")
        check(f"{raw[:20]!r} reported as changed", changed)


def test_normalization_never_empties_a_reply():
    section("Contract: a filler-only reply is left alone rather than blanked")

    for raw in ("ok", "تمام", "Sure", "", "   "):
        result, _ = response_contract.normalize_reply(raw)
        check(f"{raw!r} survives", result == raw, f"(got {result!r})")


def test_meta_narration_is_removed():
    section("Contract: no 'let me check that for you'")

    cases = [
        "Let me check that for you. Your appointment is on Thursday.",
        "لحظة أتأكد. موعدك يوم الخميس.",
        "One moment. Your appointment is on Thursday.",
    ]

    for raw in cases:
        result, _ = response_contract.normalize_reply(raw)
        check(f"narration stripped from {raw[:28]!r}", len(result) < len(raw), f"(got {result!r})")


def test_routing_language_can_never_leak():
    section("Contract: the patient must never learn there are several agents")

    leaks = [
        "I'm transferring you to our booking agent. What day suits you?",
        "Handing you over to our complaints module. What happened?",
        "هحولك للوكيل المختص. تحب إمتى؟",
    ]

    for raw in leaks:
        result, _ = response_contract.normalize_reply(raw)
        for word in ("transferring", "Handing you over", "module", "الوكيل المختص"):
            check(f"{word!r} removed", word not in result, f"(got {result!r})")


def test_persona_is_not_reintroduced_mid_conversation():
    section("Contract: one voice - no re-introduction after the greeting")

    greeting = "أهلاً بيك 👋\nأنا لطيفة، المساعدة الافتراضية في مستشفى دار العيون"
    raw = "أهلاً بيك 👋\nأنا لطيفة، المساعدة الافتراضية في مستشفى دار العيون\nموعدك يوم الخميس."

    result, changed = response_contract.normalize_reply(raw, greeting)
    check("re-introduction removed", result == "موعدك يوم الخميس.", f"(got {result!r})")
    check("reported as changed", changed)

    untouched, _ = response_contract.normalize_reply("موعدك يوم الخميس.", greeting)
    check("a normal reply is untouched", untouched == "موعدك يوم الخميس.")


def test_vertical_rhythm_is_identical():
    section("Contract: same spacing in every message")

    result, _ = response_contract.normalize_reply("سطر  \n\n\n\nسطر تاني   \n\n")
    check("blank runs collapsed and trimmed", result == "سطر\n\nسطر تاني", f"(got {result!r})")


def test_the_contract_is_applied_to_every_specialist():
    section("Contract: enforced in the graph, for whoever owns the turn")

    class FakeLLM:
        def __init__(self, reply):
            self.reply = reply

        def invoke(self, _messages):
            return AIMessage(content=self.reply)

    original = graph._llm_with_tools
    try:
        for index, (message, owner) in enumerate([
            ("عايز ألغي الحجز",   "cancel"),
            ("عايز أحجز موعد",    "booking"),
            ("عندي شكوى",         "complaint"),
            ("عندي صداع",         "medical"),
        ]):
            graph._llm_with_tools = FakeLLM("تمام! Let me check that for you. تحب إيه؟")

            reply = agent_entrypoint.send_message(
                "Dar El Oyoun-demo", f"sess-contract-{index}", message,
            )

            check(f"{owner}: filler removed", "تمام!" not in reply, f"(got {reply[-60:]!r})")
            check(f"{owner}: narration removed", "Let me check" not in reply)
    finally:
        graph._llm_with_tools = original


def test_the_specialist_that_answered_is_recorded():
    section("Observability: the routing decision is on the state, not in the reply")

    class FakeLLM:
        def invoke(self, _messages):
            return AIMessage(content="تحب إيه؟")

    original = graph._llm_with_tools
    graph._llm_with_tools = FakeLLM()
    try:
        agent_entrypoint.send_message("Dar El Oyoun-demo", "sess-observe-1", "عايز ألغي الحجز")
        snapshot = graph.graph.get_state(agent_entrypoint._config_for("sess-observe-1"))

        check("active_agent persisted", snapshot.values.get("active_agent") == "cancel",
              f"(got {snapshot.values.get('active_agent')!r})")
        check("routing reason recorded", bool(snapshot.values.get("routing_reason")))

        # ...and it survives to the next turn, which is what stickiness
        # is actually built on.
        graph._llm_with_tools = FakeLLM()
        agent_entrypoint.send_message("Dar El Oyoun-demo", "sess-observe-1", "نعم")
        snapshot = graph.graph.get_state(agent_entrypoint._config_for("sess-observe-1"))
        check("still cancel after a bare 'نعم'", snapshot.values.get("active_agent") == "cancel",
              f"(got {snapshot.values.get('active_agent')!r})")
    finally:
        graph._llm_with_tools = original


# ==========================================================
# 6. SAFETY / ROLLBACK
# ==========================================================

def test_graph_shape():
    section("Graph shape: supervisor in front, one node per specialist")

    nodes = set(graph.graph.get_graph().nodes)

    check("router node exists", "router" in nodes)
    check("load_config still exists", "load_config" in nodes)
    check("tools node still exists", "tools" in nodes)

    for name in agents.AGENT_NAMES:
        check(f"agent_{name} node exists", f"agent_{name}" in nodes)


def test_kill_switch_rebuilds_the_old_graph():
    section("Rollback: MULTI_AGENT_ENABLED=false restores the original graph")

    import importlib

    original = config.MULTI_AGENT_ENABLED
    config.MULTI_AGENT_ENABLED = False
    try:
        reloaded = importlib.reload(graph)
        nodes = set(reloaded.graph.get_graph().nodes)

        check("single 'agent' node is back", "agent" in nodes)
        check("no router", "router" not in nodes)
        check("no specialist nodes", not any(n.startswith("agent_") for n in nodes))
    finally:
        config.MULTI_AGENT_ENABLED = original
        importlib.reload(graph)


def test_public_api_is_unchanged():
    section("Compatibility: everything the old graph exported still exists")

    for name in ("graph", "agent", "load_config", "route_after_agent",
                 "checkpointer", "_llm_with_tools", "_looks_arabic",
                 "_detect_target_language", "_build_greeting"):
        check(f"graph.{name} still exists", hasattr(graph, name))

    check("prompts.build_system_prompt untouched", callable(prompts.build_system_prompt))
    check("tools.ALL_TOOLS untouched", len(tools.ALL_TOOLS) == 28, f"({len(tools.ALL_TOOLS)})")


def test_the_old_entrypoint_still_works():
    section("Compatibility: main.send_message() behaves exactly as before")

    class FakeLLM:
        def invoke(self, _messages):
            return AIMessage(content="ممكن أعرف أساعدك إزاي؟")

    original = graph._llm_with_tools
    graph._llm_with_tools = FakeLLM()
    try:
        reply = agent_entrypoint.send_message("Dar El Oyoun-demo", "sess-compat-1", "مرحبا")
        check("still returns a string", isinstance(reply, str) and bool(reply.strip()))
        check("greeting still guaranteed on turn 1", "أنا لطيفة" in reply, f"(got {reply[:60]!r})")
    finally:
        graph._llm_with_tools = original


# ==========================================================
# 7. INTERIM "PLEASE WAIT" MESSAGES
# ==========================================================

def test_progress_is_off_by_default():
    section("Progress: does nothing until explicitly switched on")

    check("PROGRESS_ENABLED defaults to false", config.PROGRESS_ENABLED is False)

    import progress
    progress.begin_turn("off-1")
    progress.schedule("off-1", "c", ["find_available_doctors"], "ar")
    check("nothing armed while disabled", "off-1" not in progress._timers)
    progress.end_turn("off-1")


def test_progress_message_matches_the_work():
    section("Progress: the line says what is actually happening")

    import progress

    cases = [
        (["find_available_doctors"],           "الأطباء"),
        (["list_available_days_for_booking"],  "المواعيد"),
        (["lookup_appointment"],               "الحجز"),
        (["create_new_booking"],               "تأكيد"),
        (["cancel_appointment"],               "الإلغاء"),
        (["send_otp"],                         "التحقق"),
        (["send_complaint_email"],             "الشكوى"),
    ]

    for tool_names, expected in cases:
        text = progress.message_for(tool_names, "ar")
        check(f"{tool_names[0]} -> mentions {expected!r}", expected in text, f"(got {text!r})")

    english = progress.message_for(["find_available_doctors"], "en")
    check("English conversation gets an English line", "doctors" in english.lower(), f"(got {english!r})")

    # An unknown tool must still produce something sane, not a KeyError.
    fallback = progress.message_for(["some_future_tool"], "ar")
    check("unknown tool falls back to the generic line", bool(fallback.strip()))


def test_progress_priority_when_several_tools_run():
    section("Progress: the most significant action wins")

    import progress

    text = progress.message_for(["find_available_doctors", "create_new_booking"], "ar")
    check("booking outranks searching", "تأكيد" in text, f"(got {text!r})")


def test_progress_can_be_overridden_per_tenant():
    section("Progress: a clinic can supply its own wording via CSV")

    import progress

    templates = {"msg_progress_searching_doctors": "ثانية واحدة، بدور لك على الدكاترة 🔎"}
    text = progress.message_for(["find_available_doctors"], "ar", templates)
    check("CSV override wins", text == "ثانية واحدة، بدور لك على الدكاترة 🔎", f"(got {text!r})")


def test_progress_fires_only_on_a_slow_turn():
    section("Progress: a fast turn stays one message, a slow turn gets two")

    import time
    import progress

    original_enabled = config.PROGRESS_ENABLED
    original_mode = config.PROGRESS_MODE
    original_delay = config.PROGRESS_DELAY_SECONDS

    config.PROGRESS_ENABLED = True
    config.PROGRESS_MODE = "log"
    config.PROGRESS_DELAY_SECONDS = 0.3

    try:
        booking = {
            "id": "G1", "bookingRefNum": "GBN-P", "statusName": "New", "status": 1,
            "doctorName": "Dr. Omar", "branchName": "Downtown",
            "bookingTimeFrom": "2026-09-05T13:00:00", "mobileNumber": "+201001255864",
        }

        def slow_api(*_args, **_kwargs):
            time.sleep(1.0)
            return {"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}

        class ScriptedLLM:
            def __init__(self, replies):
                self.replies = list(replies)

            def invoke(self, _messages):
                return self.replies.pop(0)

        def tool_call(name, args, call_id):
            return AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": call_id, "type": "tool_call"},
            ])

        original_llm = graph._llm_with_tools

        # --- slow turn ---
        progress.last_delivered.pop("prog-slow", None)
        graph._llm_with_tools = ScriptedLLM([
            tool_call("lookup_appointment", {"ref_number": "GBN-P"}, "c1"),
            AIMessage(content="لقيت الحجز، تحب تلغيه؟"),
        ])
        with patch("api.get_bookings_by_ref", side_effect=slow_api):
            agent_entrypoint.send_message("Dar El Oyoun-demo", "prog-slow", "عايز أشوف حجزي")

        sent = progress.last_delivered.get("prog-slow")
        check("slow turn told the patient to wait", bool(sent), f"(got {sent!r})")
        check("and the line was Arabic", sent and "لحظة" in sent, f"(got {sent!r})")

        # --- fast turn ---
        progress.last_delivered.pop("prog-fast", None)
        graph._llm_with_tools = ScriptedLLM([AIMessage(content="تحب إيه؟")])
        agent_entrypoint.send_message("Dar El Oyoun-demo", "prog-fast", "مرحبا")

        check(
            "fast turn stayed silent",
            progress.last_delivered.get("prog-fast") is None,
            f"(got {progress.last_delivered.get('prog-fast')!r})",
        )

        # --- no timers left behind either way ---
        check("no timer leaked", not progress._timers, f"(got {list(progress._timers)})")

    finally:
        graph._llm_with_tools = original_llm
        config.PROGRESS_ENABLED = original_enabled
        config.PROGRESS_MODE = original_mode
        config.PROGRESS_DELAY_SECONDS = original_delay


def test_progress_only_once_per_turn():
    section("Progress: six tool calls in one turn is still one 'please wait'")

    import progress

    original = config.PROGRESS_ENABLED
    original_mode = config.PROGRESS_MODE
    config.PROGRESS_ENABLED = True
    config.PROGRESS_MODE = "log"
    try:
        progress.begin_turn("once-1")
        progress._deliver("once-1", "c", "first")

        progress.schedule("once-1", "c", ["create_new_booking"], "ar")
        check("second schedule suppressed", "once-1" not in progress._timers)
        check("first message stands", progress.last_delivered.get("once-1") == "first")
        progress.end_turn("once-1")
    finally:
        config.PROGRESS_ENABLED = original
        config.PROGRESS_MODE = original_mode


def test_progress_failure_cannot_break_a_turn():
    section("Progress: a broken webhook must never affect the real reply")

    import progress

    original = config.PROGRESS_ENABLED
    original_mode = config.PROGRESS_MODE
    original_url = config.PROGRESS_WEBHOOK_URL

    config.PROGRESS_ENABLED = True
    config.PROGRESS_MODE = "webhook"
    config.PROGRESS_WEBHOOK_URL = "http://127.0.0.1:9/does-not-exist"
    try:
        progress.begin_turn("boom-1")
        # Delivering straight to a dead URL must swallow the error.
        progress._deliver("boom-1", "c", "test")
        check("delivery failure swallowed", True)
        progress.end_turn("boom-1")

        # And a bad tool list must not raise out of schedule().
        progress.schedule("boom-2", "c", None, "ar")
        check("bad input swallowed", True)
    finally:
        config.PROGRESS_ENABLED = original
        config.PROGRESS_MODE = original_mode
        config.PROGRESS_WEBHOOK_URL = original_url


# ==========================================================

def main():
    config.configure_logging()

    test_routing_picks_the_right_specialist()
    test_bare_greeting_opens_with_the_concierge()
    test_mid_flow_messages_stay_with_their_owner()
    test_a_deliberate_change_of_subject_does_switch()
    test_completed_flow_is_released()
    test_routing_is_reproducible()
    test_router_costs_no_llm_call()

    test_every_specialist_gets_the_shared_core()
    test_specialists_do_not_see_other_flows()
    test_concierge_is_the_full_legacy_agent()
    test_scoping_actually_shrinks_the_prompt()
    test_unknown_agent_falls_back_instead_of_raising()
    test_missing_sections_fall_back_to_the_whole_prompt()

    test_tool_subsets_are_correct()
    test_toolnode_can_still_execute_anything()
    test_tool_scoping_can_be_switched_off()

    test_filler_openers_are_removed()
    test_normalization_never_empties_a_reply()
    test_meta_narration_is_removed()
    test_routing_language_can_never_leak()
    test_persona_is_not_reintroduced_mid_conversation()
    test_vertical_rhythm_is_identical()
    test_the_contract_is_applied_to_every_specialist()
    test_the_specialist_that_answered_is_recorded()

    test_progress_is_off_by_default()
    test_progress_message_matches_the_work()
    test_progress_priority_when_several_tools_run()
    test_progress_can_be_overridden_per_tenant()
    test_progress_fires_only_on_a_slow_turn()
    test_progress_only_once_per_turn()
    test_progress_failure_cannot_break_a_turn()

    test_graph_shape()
    test_public_api_is_unchanged()
    test_the_old_entrypoint_still_works()
    test_kill_switch_rebuilds_the_old_graph()

    print("\n" + "=" * 70)
    print(f"{len(PASSES)} checks passed, {len(FAILURES)} failed")
    print("=" * 70)

    if FAILURES:
        for failure in FAILURES:
            print("  FAILED:", failure)
        raise SystemExit(1)

    print("\nALL MULTI-AGENT TESTS PASSED\n")


if __name__ == "__main__":
    main()
