"""
IR Schema Usage Example

Demonstrates how to create and validate Intermediate Representation structures.
This has been moved out of the schema module to keep schemas focused on data.
"""

from persona_engine.schema.ir_schema import (
    IntermediateRepresentation,
    ConversationFrame,
    ResponseStructure,
    CommunicationStyle,
    KnowledgeAndDisclosure,
    Citation,
    InteractionMode,
    ConversationGoal,
    Verbosity,
    Tone,
    UncertaintyAction,
    KnowledgeClaimType,
)


def example_casual_conversation():
    """Example: Creating an IR for a casual conversation"""
    ir = IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CASUAL_CHAT,
            goal=ConversationGoal.EXPLORE_IDEAS
        ),
        response_structure=ResponseStructure(
            intent="Share thoughts on remote work",
            stance="Strong supporter of remote work flexibility",
            rationale="Improved work-life balance (goal) + 3 years positive remote experience",
            elasticity=0.4,  # Fairly strong opinion
            confidence=0.85
        ),
        communication_style=CommunicationStyle(
            tone=Tone.WARM_CONFIDENT,
            verbosity=Verbosity.MEDIUM,
            formality=0.3,  # Casual conversation
            directness=0.7
        ),
        knowledge_disclosure=KnowledgeAndDisclosure(
            disclosure_level=0.7,  # Comfortable sharing in casual chat
            uncertainty_action=UncertaintyAction.ANSWER,
            knowledge_claim_type=KnowledgeClaimType.PERSONAL_EXPERIENCE
        ),
        citations=[
            Citation(
                source_type="goal",  # Now properly categorized as goal, not value
                source_id="work_life_balance",
                effect="High work-life balance priority → pro-remote stance",
                weight=0.9
            ),
            Citation(
                source_type="trait",
                source_id="extraversion",
                effect="Moderate extraversion (0.45) → acknowledges social trade-offs but still positive",
                weight=0.6
            )
        ],
        turn_id="example_turn_001",
        seed=123
    )
    
    return ir


def example_customer_support():
    """Example: IR for customer support interaction"""
    ir = IntermediateRepresentation(
        conversation_frame=ConversationFrame(
            interaction_mode=InteractionMode.CUSTOMER_SUPPORT,
            goal=ConversationGoal.RESOLVE_ISSUE,
            success_criteria=[
                "Understand customer's main concern",
                "Provide clear solution",
                "Confirm next steps"
            ]
        ),
        response_structure=ResponseStructure(
            intent="Provide clear, professional explanation",
            stance="Product feature is designed for X use case",
            rationale="Based on product knowledge (domain_expert)",
            elasticity=0.3,  # More firm in support context
            confidence=0.9
        ),
        communication_style=CommunicationStyle(
            tone=Tone.PROFESSIONAL_COMPOSED,
            verbosity=Verbosity.MEDIUM,
            formality=0.7,
            directness=0.8
        ),
        knowledge_disclosure=KnowledgeAndDisclosure(
            disclosure_level=0.5,  # Professional, less personal
            uncertainty_action=UncertaintyAction.ANSWER,
            knowledge_claim_type=KnowledgeClaimType.DOMAIN_EXPERT
        ),
        citations=[
            Citation(
                source_type="rule",
                source_id="social_role_at_work",
                effect="Work social role → formality=0.7, directness=0.8",
                weight=1.0
            ),
            Citation(
                source_type="trait",
                source_id="conscientiousness",
                effect="High conscientiousness (0.82) → structured, clear explanation",
                weight=0.9
            )
        ],
        turn_id="support_001",
        seed=456
    )
    
    return ir


if __name__ == "__main__":
    print("Creating example IRs...\n")
    print("=" * 60)
    
    # Casual conversation
    casual_ir = example_casual_conversation()
    print("✓ Casual conversation IR created")
    print(f"  Mode: {casual_ir.conversation_frame.interaction_mode}")
    print(f"  Tone: {casual_ir.communication_style.tone}")
    print(f"  Citations: {len(casual_ir.citations)}")
    
    # Customer support
    support_ir = example_customer_support()
    print("\n✓ Customer support IR created")
    print(f"  Mode: {support_ir.conversation_frame.interaction_mode}")
    print(f"  Tone: {support_ir.communication_style.tone}")
    print(f"  Success criteria: {len(support_ir.conversation_frame.success_criteria or [])}")
    
    print("\n" + "=" * 60)
    print("✓ All IR examples valid!")
    
    # Show JSON serialization
    print("\nCasual IR JSON (first 300 chars):")
    print(casual_ir.model_dump_json(indent=2)[:300] + "...")
