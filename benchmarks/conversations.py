"""
Benchmark Conversations — standardized scripts for validating persona behavior.

Four interaction modes with 5 turns each. Used to verify:
- Personas produce valid IR for all interaction types
- Twin personas diverge on trait-sensitive prompts
- No validation failures across the persona library
"""

CASUAL_CHAT = [
    "Hey, how's it going?",
    "What have you been up to lately?",
    "Do you have any hobbies or interests?",
    "What's your take on work-life balance?",
    "Any plans for the weekend?",
]

INTERVIEW = [
    "Can you tell me about your professional background?",
    "What would you say is your greatest strength?",
    "How do you handle pressure or tight deadlines?",
    "Where do you see yourself in five years?",
    "What's the most challenging problem you've solved?",
]

CUSTOMER_SUPPORT = [
    "I'm having trouble understanding how this works.",
    "Can you explain the process step by step?",
    "What if the standard approach doesn't work for my situation?",
    "I tried what you suggested but it didn't help.",
    "Is there someone else who might be able to help with this?",
]

SURVEY = [
    "On a scale of 1-10, how satisfied are you with remote work?",
    "What do you think about AI replacing jobs?",
    "Do you believe climate change is the most urgent issue today?",
    "Should companies be required to share salary information?",
    "What's one thing you'd change about your industry?",
]

ALL_BENCHMARKS = {
    "casual_chat": CASUAL_CHAT,
    "interview": INTERVIEW,
    "customer_support": CUSTOMER_SUPPORT,
    "survey": SURVEY,
}
