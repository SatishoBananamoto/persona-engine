"""
Intent Analyzer - Early Phase Analysis

Deterministic rule-based intent detection that runs early in the pipeline
to infer interaction mode and conversation goal when not explicitly provided.

Outputs:
- interaction_mode (if not already set)
- conversation_goal (inferred from user input)
- user_intent classification
- needs_clarification flag
"""


from persona_engine.planner.trace_context import TraceContext
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode


def analyze_intent(
    user_input: str,
    current_mode: InteractionMode | None,
    current_goal: ConversationGoal | None,
    ctx: TraceContext
) -> tuple[InteractionMode, ConversationGoal, str, bool]:
    """
    Analyze user intent (deterministic, rule-based).

    This runs EARLY in the pipeline (before stance generation) so that
    interaction_mode and goal can influence behavioral decisions.

    Args:
        user_input: Raw user input text
        current_mode: Already-set mode (None = infer it)
        current_goal: Already-set goal (None = infer it)
        ctx: TraceContext for recording decisions

    Returns:
        (interaction_mode, goal, user_intent, needs_clarification)
    """

    user_lower = user_input.lower()

    # ========================================================================
    # 1. INTERACTION MODE (only infer if not already set)
    # ========================================================================

    if current_mode is None:
        # Deterministic keyword matching (sorted for consistency)
        mode_keywords = {
            InteractionMode.SURVEY: ["survey", "questionnaire", "poll", "rating"],
            InteractionMode.CUSTOMER_SUPPORT: ["help", "support", "issue", "problem", "fix", "broken"],
            InteractionMode.INTERVIEW: ["interview", "hiring", "candidate", "position", "job"],
            InteractionMode.DEBATE: ["debate", "disagree", "argue", "counter", "wrong"],
            InteractionMode.COACHING: ["coach", "guidance", "advice", "mentor", "grow"],
            InteractionMode.BRAINSTORM: ["brainstorm", "ideas", "creative", "explore"],
        }

        # Check in deterministic order (enum definition order)
        detected_mode = InteractionMode.CASUAL_CHAT  # Default

        for mode, keywords in sorted(mode_keywords.items(), key=lambda x: x[0].value):
            if any(kw in user_lower for kw in keywords):
                detected_mode = mode
                break

        # Record mode inference
        ctx.enum(
            source_type="rule",
            source_id="mode_inference",
            target_field="conversation_frame.interaction_mode",
            operation="set",
            before="none",
            after=detected_mode.value,
            effect=f"Inferred mode from keywords: {detected_mode.value}",
            weight=1.0,
            reason="Early intent analysis"
        )

        mode = detected_mode
    else:
        mode = current_mode

    # ========================================================================
    # 2. CONVERSATION GOAL
    # ========================================================================

    if current_goal is None:
        # Infer goal from input patterns

        # Question detection
        if "?" in user_input:
            if any(word in user_lower for word in ["why", "how", "explain", "what", "tell me", "describe"]):
                goal = ConversationGoal.EDUCATE
            elif any(word in user_lower for word in ["should", "recommend", "suggest", "best"]):
                goal = ConversationGoal.PERSUADE
            else:
                goal = ConversationGoal.GATHER_INFO

        # Problem-solving
        elif any(word in user_lower for word in ["fix", "solve", "resolve", "help with", "deal with"]):
            goal = ConversationGoal.RESOLVE_ISSUE

        # Opinion/feeling
        elif any(word in user_lower for word in ["think", "feel", "believe", "opinion", "view"]):
            goal = ConversationGoal.BUILD_RAPPORT

        # Teaching/explaining
        elif any(word in user_lower for word in ["learn", "teach", "understand", "know about"]):
            goal = ConversationGoal.EDUCATE

        # Exploration
        elif any(word in user_lower for word in ["explore", "consider", "discuss", "talk about"]):
            goal = ConversationGoal.EXPLORE_IDEAS

        else:
            goal = ConversationGoal.BUILD_RAPPORT  # Default

        # Record goal inference
        ctx.enum(
            source_type="rule",
            source_id="goal_inference",
            target_field="conversation_frame.goal",
            operation="set",
            before="none",
            after=goal.value,
            effect=f"Inferred goal: {goal.value}",
            weight=1.0,
            reason="Input pattern analysis"
        )
    else:
        goal = current_goal

    # ========================================================================
    # 3. USER INTENT CLASSIFICATION
    # ========================================================================

    if "?" in user_input:
        user_intent = "ask"
    elif any(word in user_lower for word in ["please", "could you", "would you", "can you"]):
        user_intent = "request"
    elif any(word in user_lower for word in ["wrong", "disagree", "not true", "actually", "no,"]):
        user_intent = "challenge"
    elif any(word in user_lower for word in ["i think", "i feel", "my experience", "in my view"]):
        user_intent = "share"
    elif any(word in user_lower for word in ["what do you mean", "clarify", "explain that", "confused"]):
        user_intent = "clarify"
    else:
        user_intent = "share"

    # ========================================================================
    # 4. NEEDS CLARIFICATION DETECTION
    # ========================================================================

    needs_clarification = (
        len(user_input.split()) < 5  # Very short
        or user_input.count("?") > 2  # Multiple questions
        or any(word in user_lower for word in ["or", "either", "maybe", "not sure", "kind of"])  # Ambiguous
        or (user_intent == "ask" and len(user_input.split()) < 8)  # Vague question
    )

    return mode, goal, user_intent, needs_clarification
