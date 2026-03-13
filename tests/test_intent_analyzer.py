"""
Comprehensive tests for persona_engine/planner/intent_analyzer.py

Targets 95%+ coverage of the analyze_intent function across all four sections:
  1. Interaction Mode inference (keyword matching, sorted order, passthrough)
  2. Conversation Goal inference (question vs non-question branches, passthrough)
  3. User Intent classification (ask, request, challenge, share, clarify)
  4. Needs Clarification detection (short, multi-?, ambiguous, vague question)

Plus citation recording in TraceContext and edge cases.
"""

import pytest

from persona_engine.planner.intent_analyzer import analyze_intent
from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode


@pytest.fixture
def ctx() -> TraceContext:
    """Fresh TraceContext for each test."""
    return TraceContext()


# ============================================================================
# Section 1: Interaction Mode Detection
# ============================================================================


class TestInteractionModeKeywords:
    """Each keyword for every mode triggers the correct InteractionMode."""

    @pytest.mark.parametrize("keyword", ["brainstorm", "ideas", "creative", "explore"])
    def test_brainstorm_keywords(self, ctx: TraceContext, keyword: str) -> None:
        mode, _, _, _ = analyze_intent(
            f"Let us {keyword} new things and discuss them at length this week",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.BRAINSTORM

    @pytest.mark.parametrize("keyword", ["coach", "guidance", "advice", "mentor", "grow"])
    def test_coaching_keywords(self, ctx: TraceContext, keyword: str) -> None:
        mode, _, _, _ = analyze_intent(
            f"I need {keyword} with my latest skill assessment plan this week",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.COACHING

    @pytest.mark.parametrize("keyword", ["help", "support", "issue", "problem", "fix", "broken"])
    def test_customer_support_keywords(self, ctx: TraceContext, keyword: str) -> None:
        mode, _, _, _ = analyze_intent(
            f"My system has a {keyword} that needs dealing with this week",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.CUSTOMER_SUPPORT

    @pytest.mark.parametrize("keyword", ["debate", "disagree", "argue", "counter", "wrong"])
    def test_debate_keywords(self, ctx: TraceContext, keyword: str) -> None:
        mode, _, _, _ = analyze_intent(
            f"I want to {keyword} the latest claim being made by the panel",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.DEBATE

    @pytest.mark.parametrize("keyword", ["interview", "hiring", "candidate", "position", "job"])
    def test_interview_keywords(self, ctx: TraceContext, keyword: str) -> None:
        mode, _, _, _ = analyze_intent(
            f"We have a {keyword} scheduled at the building this week next",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.INTERVIEW

    @pytest.mark.parametrize("keyword", ["survey", "questionnaire", "poll", "rating"])
    def test_survey_keywords(self, ctx: TraceContext, keyword: str) -> None:
        mode, _, _, _ = analyze_intent(
            f"Please fill in this {keyword} and submit it when finished today",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.SURVEY

    def test_default_casual_chat_when_no_keywords_match(self, ctx: TraceContext) -> None:
        """No matching keywords at all falls back to CASUAL_CHAT."""
        mode, _, _, _ = analyze_intent(
            "I had a great day at the beach and the sun was shining all day",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.CASUAL_CHAT


class TestInteractionModeCaseInsensitivity:
    """Keyword matching is case-insensitive via .lower()."""

    def test_all_uppercase(self, ctx: TraceContext) -> None:
        mode, _, _, _ = analyze_intent(
            "Let us BRAINSTORM NEW THINGS and discuss them all at length today",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.BRAINSTORM

    def test_mixed_case(self, ctx: TraceContext) -> None:
        mode, _, _, _ = analyze_intent(
            "I need Guidance with my latest assessment plan and skills review",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.COACHING

    def test_title_case_debate(self, ctx: TraceContext) -> None:
        mode, _, _, _ = analyze_intent(
            "I want to Debate the claim and all the evidence given by the team",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.DEBATE


class TestInteractionModeSortedPriority:
    """Modes checked in sorted enum-value order; first match wins."""

    def test_brainstorm_before_customer_support(self, ctx: TraceContext) -> None:
        """'explore' (BRAINSTORM) matched before 'help' (CUSTOMER_SUPPORT)."""
        mode, _, _, _ = analyze_intent(
            "Can you help me explore these new design insights at length this week",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.BRAINSTORM

    def test_coaching_before_customer_support(self, ctx: TraceContext) -> None:
        """'coach' (COACHING) matched before 'support' (CUSTOMER_SUPPORT)."""
        mode, _, _, _ = analyze_intent(
            "I need support and coaching tips at the same time this week next",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.COACHING

    def test_customer_support_before_debate(self, ctx: TraceContext) -> None:
        """'fix' (CUSTOMER_SUPPORT) matched before 'wrong' (DEBATE)."""
        mode, _, _, _ = analyze_intent(
            "Something is wrong and I need a fix quite urgently this week now",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.CUSTOMER_SUPPORT

    def test_debate_before_interview(self, ctx: TraceContext) -> None:
        """'argue' (DEBATE) matched before 'job' (INTERVIEW)."""
        mode, _, _, _ = analyze_intent(
            "I want to argue that this job listing is misleading to all candidates",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.DEBATE

    def test_interview_before_survey(self, ctx: TraceContext) -> None:
        """'job' (INTERVIEW) matched before 'poll' (SURVEY)."""
        mode, _, _, _ = analyze_intent(
            "Please take this poll about the new job listings and submit it today",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.INTERVIEW


class TestInteractionModePassthrough:
    """When current_mode is provided, it is used as-is (no inference)."""

    def test_mode_passthrough_ignores_keywords(self, ctx: TraceContext) -> None:
        mode, _, _, _ = analyze_intent(
            "I want to brainstorm new ideas and plan the creative session",
            InteractionMode.INTERVIEW,
            ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.INTERVIEW

    @pytest.mark.parametrize("given_mode", list(InteractionMode))
    def test_all_modes_pass_through(self, given_mode: InteractionMode) -> None:
        ctx = TraceContext()
        mode, _, _, _ = analyze_intent(
            "Plain text with nothing special at all happening today in the lab",
            given_mode, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == given_mode


# ============================================================================
# Section 2: Conversation Goal Inference
# ============================================================================


class TestGoalQuestionBranch:
    """Goal inference when input contains '?'."""

    @pytest.mark.parametrize("keyword", ["why", "how", "explain", "what", "tell me", "describe"])
    def test_question_with_educate_keyword(self, ctx: TraceContext, keyword: str) -> None:
        """? + educate keyword -> EDUCATE."""
        _, goal, _, _ = analyze_intent(
            f"Can you {keyword} this system and all its design principles?",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.EDUCATE

    @pytest.mark.parametrize("keyword", ["should", "recommend", "suggest", "best"])
    def test_question_with_persuade_keyword(self, ctx: TraceContext, keyword: str) -> None:
        """? + persuade keyword -> PERSUADE."""
        _, goal, _, _ = analyze_intent(
            f"Can you {keyword} the right path ahead?",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.PERSUADE

    def test_question_without_special_keywords_gives_gather_info(self, ctx: TraceContext) -> None:
        """? without educate/persuade keywords -> GATHER_INFO."""
        _, goal, _, _ = analyze_intent(
            "Is the system running at the expected level right this instant?",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.GATHER_INFO

    def test_educate_takes_priority_over_persuade_in_question(self, ctx: TraceContext) -> None:
        """EDUCATE keywords checked before PERSUADE keywords in ? branch."""
        _, goal, _, _ = analyze_intent(
            "What should I recommend in my next big presentation?",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.EDUCATE


class TestGoalNoQuestionBranch:
    """Goal inference when input does NOT contain '?'."""

    @pytest.mark.parametrize("keyword", ["fix", "solve", "resolve", "help with", "deal with"])
    def test_resolve_issue_keywords(self, ctx: TraceContext, keyword: str) -> None:
        _, goal, _, _ = analyze_intent(
            f"I need to {keyword} the bug and get the system back up again now",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.RESOLVE_ISSUE

    @pytest.mark.parametrize("keyword", ["think", "feel", "believe", "opinion", "view"])
    def test_build_rapport_keywords(self, ctx: TraceContext, keyword: str) -> None:
        _, goal, _, _ = analyze_intent(
            f"I {keyword} that the new design is quite nice and well made indeed",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.BUILD_RAPPORT

    @pytest.mark.parametrize("keyword", ["learn", "teach", "understand", "know about"])
    def test_educate_keywords_no_question(self, ctx: TraceContext, keyword: str) -> None:
        _, goal, _, _ = analyze_intent(
            f"I want to {keyword} the new developments in science and technology",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.EDUCATE

    @pytest.mark.parametrize("keyword", ["explore", "consider", "discuss", "talk about"])
    def test_explore_ideas_keywords(self, ctx: TraceContext, keyword: str) -> None:
        _, goal, _, _ = analyze_intent(
            f"Let us {keyword} the new insights and latest findings at length",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.EXPLORE_IDEAS

    def test_default_goal_is_build_rapport(self, ctx: TraceContext) -> None:
        """No matching keywords at all -> default BUILD_RAPPORT."""
        _, goal, _, _ = analyze_intent(
            "I had a nice day at the beach and the sun was shining all day",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.BUILD_RAPPORT


class TestGoalPriority:
    """Goal branches checked in order; first match wins."""

    def test_question_branch_overrides_no_question_keywords(self, ctx: TraceContext) -> None:
        """'?' + 'how' -> EDUCATE, even when 'fix' (RESOLVE_ISSUE) is present."""
        _, goal, _, _ = analyze_intent(
            "How can I fix this and get it back up and running?",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.EDUCATE

    def test_resolve_issue_before_build_rapport(self, ctx: TraceContext) -> None:
        """RESOLVE_ISSUE ('fix') checked before BUILD_RAPPORT ('think')."""
        _, goal, _, _ = analyze_intent(
            "I think I need to fix this thing and get it back up again",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.RESOLVE_ISSUE

    def test_build_rapport_before_educate_no_question(self, ctx: TraceContext) -> None:
        """BUILD_RAPPORT ('think') checked before EDUCATE ('learn') outside ? branch."""
        _, goal, _, _ = analyze_intent(
            "I think I need to learn new things in the field and expand",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.BUILD_RAPPORT

    def test_educate_before_explore_no_question(self, ctx: TraceContext) -> None:
        """EDUCATE ('learn') checked before EXPLORE_IDEAS ('explore')."""
        _, goal, _, _ = analyze_intent(
            "I want to learn and then we can discuss the new landscape at length",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        assert goal == ConversationGoal.EDUCATE


class TestGoalPassthrough:
    """When current_goal is provided, it is used as-is (no inference)."""

    def test_goal_passthrough_ignores_keywords(self, ctx: TraceContext) -> None:
        _, goal, _, _ = analyze_intent(
            "How does this system actually maintain itself?",
            InteractionMode.CASUAL_CHAT,
            ConversationGoal.PERSUADE,
            ctx,
        )
        assert goal == ConversationGoal.PERSUADE

    @pytest.mark.parametrize("given_goal", list(ConversationGoal))
    def test_all_goals_pass_through(self, given_goal: ConversationGoal) -> None:
        ctx = TraceContext()
        _, goal, _, _ = analyze_intent(
            "Plain text with nothing special at all happening in the lab",
            InteractionMode.CASUAL_CHAT, given_goal, ctx,
        )
        assert goal == given_goal


# ============================================================================
# Section 3: User Intent Classification
# ============================================================================


class TestUserIntentAsk:
    """'?' in input -> intent 'ask'."""

    def test_simple_question(self, ctx: TraceContext) -> None:
        _, _, intent, _ = analyze_intent(
            "What is the meaning and significance?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"

    def test_ask_takes_priority_over_request(self, ctx: TraceContext) -> None:
        """'?' is checked before 'please'/'could you'."""
        _, _, intent, _ = analyze_intent(
            "Could you please tell me all the details?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"

    def test_ask_takes_priority_over_challenge(self, ctx: TraceContext) -> None:
        """'?' is checked before 'wrong'."""
        _, _, intent, _ = analyze_intent(
            "Is this actually wrong?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"

    def test_ask_takes_priority_over_share(self, ctx: TraceContext) -> None:
        """'?' is checked before 'i think'."""
        _, _, intent, _ = analyze_intent(
            "I think this is nice but is it real?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"

    def test_ask_takes_priority_over_clarify(self, ctx: TraceContext) -> None:
        """'?' is checked before 'confused'."""
        _, _, intent, _ = analyze_intent(
            "I am confused, is this the latest data?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"


class TestUserIntentRequest:
    """'please'/'could you'/'would you'/'can you' -> intent 'request'."""

    @pytest.mark.parametrize("phrase", ["please", "could you", "would you", "can you"])
    def test_request_keywords(self, ctx: TraceContext, phrase: str) -> None:
        _, _, intent, _ = analyze_intent(
            f"{phrase.capitalize()} send me the latest analysis data and insights",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "request"

    def test_request_overrides_challenge(self, ctx: TraceContext) -> None:
        """'please' checked before 'wrong'."""
        _, _, intent, _ = analyze_intent(
            "Please tell me what went wrong with the data and the latest analysis",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "request"

    def test_request_overrides_share(self, ctx: TraceContext) -> None:
        """'please' checked before 'i think'."""
        _, _, intent, _ = analyze_intent(
            "I think you should please send me the latest full status update",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "request"


class TestUserIntentChallenge:
    """'wrong'/'disagree'/'not true'/'actually'/'no,' -> intent 'challenge'."""

    @pytest.mark.parametrize("phrase", ["wrong", "disagree", "not true", "actually"])
    def test_challenge_keywords(self, ctx: TraceContext, phrase: str) -> None:
        _, _, intent, _ = analyze_intent(
            f"That is just {phrase} and I have the data to back up my claim fully",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "challenge"

    def test_challenge_no_comma(self, ctx: TraceContext) -> None:
        """'no,' (with trailing comma) triggers challenge."""
        _, _, intent, _ = analyze_intent(
            "No, that is a bad take and I have all the evidence right at hand",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "challenge"

    def test_challenge_overrides_share(self, ctx: TraceContext) -> None:
        """'actually' (challenge) checked before 'i think' (share)."""
        _, _, intent, _ = analyze_intent(
            "I think that is actually quite a bad assessment made by the panel",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "challenge"


class TestUserIntentShare:
    """'i think'/'i feel'/'my experience'/'in my view' -> intent 'share'."""

    @pytest.mark.parametrize("phrase", ["I think", "I feel", "my experience", "in my view"])
    def test_share_keywords(self, ctx: TraceContext, phrase: str) -> None:
        _, _, intent, _ = analyze_intent(
            f"{phrase} is that this new design path is quite nice and well made",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "share"

    def test_default_intent_is_share(self, ctx: TraceContext) -> None:
        """No matching keywords at all -> default 'share'."""
        _, _, intent, _ = analyze_intent(
            "The day was pleasant and the wind was gentle by the lake all day",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "share"


class TestUserIntentClarify:
    """'what do you mean'/'clarify'/'explain that'/'confused' -> intent 'clarify'."""

    def test_clarify_what_do_you_mean(self, ctx: TraceContext) -> None:
        _, _, intent, _ = analyze_intent(
            "What do you mean by that statement and the findings in the analysis",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "clarify"

    def test_clarify_keyword(self, ctx: TraceContext) -> None:
        _, _, intent, _ = analyze_intent(
            "I need to clarify this main claim and the data cited in the study",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "clarify"

    def test_clarify_explain_that(self, ctx: TraceContext) -> None:
        _, _, intent, _ = analyze_intent(
            "The team needs to explain that in simpler detail and with examples",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "clarify"

    def test_clarify_confused(self, ctx: TraceContext) -> None:
        _, _, intent, _ = analyze_intent(
            "I am really confused by the latest data and all the claims made",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "clarify"


# ============================================================================
# Section 4: Needs Clarification Detection
# ============================================================================


class TestNeedsClarificationTrue:
    """Cases where needs_clarification should be True."""

    def test_very_short_input_four_words(self, ctx: TraceContext) -> None:
        """4 words >= 3 threshold, so does NOT trigger clarification anymore."""
        _, _, _, needs = analyze_intent(
            "Tell me right away",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_very_short_input_two_words(self, ctx: TraceContext) -> None:
        _, _, _, needs = analyze_intent(
            "Hi there",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is True

    def test_very_short_input_one_word(self, ctx: TraceContext) -> None:
        _, _, _, needs = analyze_intent(
            "Hello",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is True

    def test_multiple_question_marks_four(self, ctx: TraceContext) -> None:
        """More than 2 '?' triggers needs clarification."""
        _, _, _, needs = analyze_intent(
            "Is this it? Really? But is that all? And when is it due?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is True

    def test_multiple_question_marks_three(self, ctx: TraceContext) -> None:
        """Exactly 3 <= 3 threshold, so does NOT trigger clarification anymore."""
        _, _, _, needs = analyze_intent(
            "What is happening? Is it bad? And will it get well?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_ambiguous_word_or_no_longer_triggers(self, ctx: TraceContext) -> None:
        """'or' in normal sentences no longer triggers clarification (Fix 6)."""
        _, _, _, needs = analyze_intent(
            "I want red or blue as the main design theme picked this time",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_ambiguous_word_either_no_longer_triggers(self, ctx: TraceContext) -> None:
        _, _, _, needs = analyze_intent(
            "I like either the new plan submitted next week and the backup plan",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_ambiguous_word_maybe_no_longer_triggers(self, ctx: TraceContext) -> None:
        _, _, _, needs = analyze_intent(
            "I was thinking maybe the new timeline is a bad call at all times",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_ambiguous_word_not_sure_only_in_phrase(self, ctx: TraceContext) -> None:
        """Standalone 'not sure' no longer triggers; only 'not sure what' does."""
        _, _, _, needs = analyze_intent(
            "I am not sure if the latest design is the right call at all",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_ambiguous_word_kind_of_no_longer_triggers(self, ctx: TraceContext) -> None:
        _, _, _, needs = analyze_intent(
            "This seems kind of like a big deal and I am quite invested in it",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_vague_question_no_longer_triggers(self, ctx: TraceContext) -> None:
        """Short questions (< 8 words) no longer trigger clarification (Fix 6)."""
        _, _, intent, needs = analyze_intent(
            "What exactly is the right plan ahead?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"
        assert needs is False

    def test_or_as_substring_no_longer_triggers(self, ctx: TraceContext) -> None:
        """Substring 'or' inside 'informatics' no longer triggers (Fix 6)."""
        _, _, _, needs = analyze_intent(
            "I am studying informatics at the lab all week and all night",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_or_in_common_word_for_no_longer_triggers(self, ctx: TraceContext) -> None:
        """Substring 'or' inside 'for' no longer triggers (Fix 6)."""
        _, _, _, needs = analyze_intent(
            "I am studying new things at the lab all week and waiting for results",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False


class TestNeedsClarificationFalse:
    """Cases where needs_clarification should be False."""

    def test_long_clear_statement(self, ctx: TraceContext) -> None:
        """Long statement with no ambiguity markers at all."""
        _, _, _, needs = analyze_intent(
            "I have been studying advanced mathematics and physics all week at the university level",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        # 14 words, no ?, no 'or' substring, intent != "ask"
        assert needs is False

    def test_long_question_without_ambiguity(self, ctx: TraceContext) -> None:
        """Long question (>= 8 words) without ambiguity words."""
        _, _, intent, needs = analyze_intent(
            "What exactly is the mechanism behind the latest quantum computing advancement?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"
        # 10 words, 1 ?, no ambiguous substrings, ask but 10 >= 8
        assert needs is False

    def test_exactly_five_words_no_ambiguity(self, ctx: TraceContext) -> None:
        """5 words is NOT < 5, so the short-input check is False."""
        _, _, _, needs = analyze_intent(
            "I like this new plan",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        # 5 words, no ?, no ambiguous substrings, intent != "ask"
        assert needs is False

    def test_exactly_two_question_marks_is_not_multiple(self, ctx: TraceContext) -> None:
        """2 '?' is NOT > 2, so multi-question check is False."""
        _, _, _, needs = analyze_intent(
            "Is this happening? And is the new patch being deployed?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        # 10 words, 2 ?, no ambiguous substrings, ask + 10 >= 8
        assert needs is False

    def test_ask_with_eight_or_more_words(self, ctx: TraceContext) -> None:
        """intent='ask' + >= 8 words does NOT trigger vague question check."""
        _, _, intent, needs = analyze_intent(
            "What is the plan that the team has been making lately?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"
        # 11 words >= 8, so last condition is False
        assert needs is False

    def test_ask_exactly_eight_words(self, ctx: TraceContext) -> None:
        """intent='ask' + exactly 8 words: 8 is NOT < 8."""
        _, _, intent, needs = analyze_intent(
            "What is the plan that was made today?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "ask"
        # 8 words, 1 ?, no ambiguous substrings, 8 not < 8
        assert needs is False


# ============================================================================
# Section 5: Citation Recording in TraceContext
# ============================================================================


class TestCitationRecording:
    """TraceContext citations are properly recorded during analysis."""

    def test_mode_inference_adds_citation(self, ctx: TraceContext) -> None:
        """Inferring mode (current_mode=None) records a citation."""
        analyze_intent(
            "I had a great day at the beach and it was pleasant all day",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        mode_citations = [
            c for c in ctx.citations if c.source_id == "mode_inference"
        ]
        assert len(mode_citations) == 1
        c = mode_citations[0]
        assert c.source_type == "rule"
        assert c.target_field == "conversation_frame.interaction_mode"
        assert c.operation == "set"
        assert c.value_before == "none"
        assert c.value_after == InteractionMode.CASUAL_CHAT.value
        assert c.weight == 1.0
        assert c.reason == "Early intent analysis"

    def test_mode_citation_matches_detected_mode(self, ctx: TraceContext) -> None:
        """Citation value_after reflects the actual detected mode."""
        analyze_intent(
            "Let us brainstorm new things and discuss them all at length today",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        mode_citations = [
            c for c in ctx.citations if c.source_id == "mode_inference"
        ]
        assert len(mode_citations) == 1
        assert mode_citations[0].value_after == InteractionMode.BRAINSTORM.value

    def test_mode_citation_effect_contains_mode_name(self, ctx: TraceContext) -> None:
        """Citation effect message mentions the detected mode value."""
        analyze_intent(
            "I want to debate the claim and all the evidence given by the team",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        mode_citations = [
            c for c in ctx.citations if c.source_id == "mode_inference"
        ]
        assert "debate" in mode_citations[0].effect

    def test_goal_inference_adds_citation(self, ctx: TraceContext) -> None:
        """Inferring goal (current_goal=None) records a citation."""
        analyze_intent(
            "I had a great day at the beach and it was pleasant all day",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        goal_citations = [
            c for c in ctx.citations if c.source_id == "goal_inference"
        ]
        assert len(goal_citations) == 1
        c = goal_citations[0]
        assert c.source_type == "rule"
        assert c.target_field == "conversation_frame.goal"
        assert c.operation == "set"
        assert c.value_before == "none"
        assert c.weight == 1.0
        assert c.reason == "Input pattern analysis"

    def test_goal_citation_matches_detected_goal(self, ctx: TraceContext) -> None:
        """Citation value_after reflects the actual inferred goal."""
        analyze_intent(
            "How does this mechanism actually function?",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        goal_citations = [
            c for c in ctx.citations if c.source_id == "goal_inference"
        ]
        assert len(goal_citations) == 1
        assert goal_citations[0].value_after == ConversationGoal.EDUCATE.value

    def test_goal_citation_effect_contains_goal_name(self, ctx: TraceContext) -> None:
        """Citation effect message mentions the inferred goal value."""
        analyze_intent(
            "I want to learn new things in the field and expand my skill set",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        goal_citations = [
            c for c in ctx.citations if c.source_id == "goal_inference"
        ]
        assert "educate" in goal_citations[0].effect

    def test_both_inferred_produces_two_citations(self, ctx: TraceContext) -> None:
        """When both mode and goal are None, exactly 2 citations are recorded."""
        analyze_intent(
            "I had a nice day at the beach and the sun was shining all day",
            None, None, ctx,
        )
        assert len(ctx.citations) == 2
        source_ids = {c.source_id for c in ctx.citations}
        assert source_ids == {"mode_inference", "goal_inference"}

    def test_no_mode_citation_when_mode_provided(self, ctx: TraceContext) -> None:
        """Providing current_mode skips mode inference citation."""
        analyze_intent(
            "I had a nice day at the beach and it was pleasant all day",
            InteractionMode.CASUAL_CHAT, None, ctx,
        )
        mode_citations = [
            c for c in ctx.citations if c.source_id == "mode_inference"
        ]
        assert len(mode_citations) == 0

    def test_no_goal_citation_when_goal_provided(self, ctx: TraceContext) -> None:
        """Providing current_goal skips goal inference citation."""
        analyze_intent(
            "I had a nice day at the beach and it was pleasant all day",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        goal_citations = [
            c for c in ctx.citations if c.source_id == "goal_inference"
        ]
        assert len(goal_citations) == 0

    def test_no_citations_when_both_provided(self, ctx: TraceContext) -> None:
        """Providing both mode and goal produces zero citations."""
        analyze_intent(
            "Plain text with nothing special at all happening in the lab",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert len(ctx.citations) == 0


# ============================================================================
# Section 6: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge cases: empty input, single word, whitespace, long input, etc."""

    def test_empty_input(self, ctx: TraceContext) -> None:
        """Empty string: all defaults, needs_clarification=True."""
        mode, goal, intent, needs = analyze_intent("", None, None, ctx)
        assert mode == InteractionMode.CASUAL_CHAT
        assert goal == ConversationGoal.BUILD_RAPPORT
        assert intent == "share"
        assert needs is True  # 0 words < 5

    def test_single_word_no_keywords(self, ctx: TraceContext) -> None:
        """Single word with no matching keywords."""
        mode, goal, intent, needs = analyze_intent("Hello", None, None, ctx)
        assert mode == InteractionMode.CASUAL_CHAT
        assert goal == ConversationGoal.BUILD_RAPPORT
        assert intent == "share"
        assert needs is True  # 1 word < 5

    def test_single_question_mark(self, ctx: TraceContext) -> None:
        """Just '?': intent=ask, needs_clarification=True."""
        _, _, intent, needs = analyze_intent("?", None, None, ctx)
        assert intent == "ask"
        assert needs is True  # 1 word < 5

    def test_whitespace_only_input(self, ctx: TraceContext) -> None:
        """Whitespace-only: split() returns [], so < 5 words."""
        _, _, _, needs = analyze_intent(
            "     ",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is True

    def test_very_long_input_no_ambiguity(self, ctx: TraceContext) -> None:
        """Very long input with no ambiguity triggers."""
        long_text = (
            "I have been studying advanced mathematics and physics all week "
            "at the university level and I find it all quite stimulating "
            "and intellectually engaging and I plan to keep studying this "
            "particular discipline until I achieve a high level and mastery"
        )
        _, _, _, needs = analyze_intent(
            long_text,
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert needs is False

    def test_return_value_is_four_tuple(self, ctx: TraceContext) -> None:
        """Function returns a 4-tuple of correct types."""
        result = analyze_intent(
            "I like the new design quite a bit and it is well made",
            None, None, ctx,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        mode, goal, intent, needs = result
        assert isinstance(mode, InteractionMode)
        assert isinstance(goal, ConversationGoal)
        assert isinstance(intent, str)
        assert isinstance(needs, bool)

    def test_keyword_embedded_in_longer_word(self, ctx: TraceContext) -> None:
        """'job' embedded in 'jobless' still matches INTERVIEW (substring match)."""
        mode, _, _, _ = analyze_intent(
            "Being jobless is quite challenging and takes a mental health hit",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.INTERVIEW

    def test_creative_embedded_in_uncreative(self, ctx: TraceContext) -> None:
        """'creative' in 'uncreative' matches BRAINSTORM (substring match)."""
        mode, _, _, _ = analyze_intent(
            "Being uncreative is the distant opposite and we need a new plan change",
            None, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert mode == InteractionMode.BRAINSTORM

    def test_integration_all_four_outputs(self, ctx: TraceContext) -> None:
        """All four outputs are consistent for a complex input."""
        mode, goal, intent, needs = analyze_intent(
            "How can I brainstorm new design insights?",
            None, None, ctx,
        )
        assert mode == InteractionMode.BRAINSTORM  # "brainstorm" keyword
        assert goal == ConversationGoal.EDUCATE    # "?" + "how"
        assert intent == "ask"                     # "?"
        # Substring 'or' no longer triggers clarification (Fix 6)
        assert needs is False

    def test_multiple_clarification_triggers_at_once(self, ctx: TraceContext) -> None:
        """Short + ambiguous + vague question all fire simultaneously."""
        _, _, _, needs = analyze_intent(
            "Maybe this?",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        # 2 words < 5, "maybe" in input, intent="ask" and 2 < 8
        assert needs is True

    def test_input_with_only_punctuation(self, ctx: TraceContext) -> None:
        """Punctuation-only input (no letters)."""
        _, _, intent, needs = analyze_intent(
            "!!!",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        assert intent == "share"  # no matching keywords
        assert needs is True      # 1 word < 5

    def test_no_without_comma_is_not_challenge(self, ctx: TraceContext) -> None:
        """'no' without comma does NOT trigger challenge (keyword is 'no,')."""
        _, _, intent, _ = analyze_intent(
            "I say no to that assessment and I have the data backing my side up",
            InteractionMode.CASUAL_CHAT, ConversationGoal.BUILD_RAPPORT, ctx,
        )
        # "no" without comma; no other challenge keywords present
        # (checking: "wrong"? no. "disagree"? no. "not true"? no. "actually"? no.)
        # Falls through to share: "i say" does not contain "i think"/"i feel"/etc.
        # Default intent is "share"
        assert intent == "share"
