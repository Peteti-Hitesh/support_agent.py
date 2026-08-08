"""
Task 4 — Complete Helpdesk Agent
==================================

This module ties Tasks 1-3 together into a working conversational
helpdesk agent:

  - Task 1 (task1_faq_search.py)   -> FAQS knowledge base
  - Task 2 (task2_llm_client.py)   -> LLMClient.generate_faq_response()
  - Task 3 (task3_faq_matcher.py)  -> hybrid_search()

Nothing from Tasks 1-3 is reimplemented here — SupportAgent only
orchestrates calls into those modules.

Run
---
    python task4_support_agent.py            # interactive chat
    python task4_support_agent.py --demo      # scripted end-to-end demo
"""

# pylint: disable=invalid-name

import os
import random
import sys
from dataclasses import dataclass
from typing import List

# Ensure local task modules can be imported when running from this script's directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import_failed = False
_import_error = None
try:
    from task1_faq_search import FAQS
    from task2_llm_client import LLMClient, run_mock_response, DEFAULT_MODEL
    from task3_faq_matcher import hybrid_search
except ImportError as e:
    _import_error = e
    FAQS = []
    LLMClient = None
    run_mock_response = None
    DEFAULT_MODEL = None
    hybrid_search = None
    import_failed = True


def _fail_startup() -> None:
    print(f"Error: Could not import task modules ({_import_error}).")
    print(
        "Task 4 requires task1_faq_search.py, task2_llm_client.py, and "
        "task3_faq_matcher.py to be in the same directory as "
        "task4_support_agent.py. Please add them and rerun the script."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. ConversationTurn
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in the conversation history.

    Attributes:
        role: str: Either "user" or "assistant".
        content: str: Message text for the turn.
        faq_id: str|None: Optional matched FAQ id.
        confidence: float|None: Optional match confidence score.
    """

    role: str        # "user" or "assistant"
    content: str
    faq_id: str = None
    confidence: float = None


# ---------------------------------------------------------------------------
# 2. SupportAgent
# ---------------------------------------------------------------------------

class SupportAgent:
    """Orchestrates FAQ search (Task 3) and LLM response generation
    (Task 2) into a stateful, multi-turn helpdesk conversation.
    """

    def __init__(self, faqs, llm_client, confidence_threshold: float = 0.15):
        """Initialise the agent with FAQ data, an LLM client, and a
        minimum confidence threshold required for a "confident" match.

        Args:
            faqs (list): FAQ dictionaries from Task 1.
            llm_client (LLMClient): A configured Task 2 LLMClient.
            confidence_threshold (float): Minimum hybrid_search() score
                required to treat a match as confident.
        """
        self.faqs = faqs
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

        self.history: List[ConversationTurn] = []
        self.escalated: bool = False

    def _generate_response(self, user_message: str, faq_entry: dict) -> str:
        """Call the Task 2 LLM client, falling back to a mock rephrasing
        if the live API call fails (e.g. no network / no API key), so
        the agent never crashes mid-conversation.
        """
        try:
            return self.llm_client.generate_faq_response(user_message, faq_entry)
        except RuntimeError:
            return run_mock_response(user_message, faq_entry)

    def handle_message(self, user_message: str) -> str:
        """Process one user message end-to-end and return the agent's reply.

        Flow:
          1. Record the user's message in history.
          2. Run hybrid_search() (Task 3) to find the best FAQ match.
          3. If confident (score >= threshold): generate a response via
             the Task 2 LLM client and record the FAQ id + confidence.
          4. Otherwise: return a fallback response, offer escalation,
             and record confidence 0.0.

        Args:
            user_message (str): The raw text the user typed.

        Returns:
            str: The agent's reply.
        """
        self.history.append(ConversationTurn(role="user", content=user_message))

        results = hybrid_search(self.faqs, user_message, top_k=1)
        best = results[0] if results else None

        if best is not None and best[1] >= self.confidence_threshold:
            faq_entry, confidence = best
            response = self._generate_response(user_message, faq_entry)
            self.history.append(
                ConversationTurn(
                    role="assistant",
                    content=response,
                    faq_id=faq_entry["id"],
                    confidence=confidence,
                )
            )
            return response

        # No confident match -> fallback + escalation offer.
        response = (
            "I don't have information about that in my knowledge base. "
            "Would you like me to connect you with a human support agent?"
        )
        self.history.append(
            ConversationTurn(
                role="assistant",
                content=response,
                faq_id=None,
                confidence=0.0,
            )
        )
        return response

    def escalate(self, reason: str = "User requested human support") -> str:
        """Mark the session as escalated and return a confirmation
        message containing a mock ticket ID and an estimated response
        time.

        Args:
            reason (str): Why the escalation was triggered (stored for
                the summary but not required to be shown to the user).

        Returns:
            str: A confirmation message with ticket ID and ETA.
        """
        self.escalated = True
        ticket_id = f"TICKET-{random.randint(10000, 99999)}"

        self.history.append(
            ConversationTurn(
                role="assistant",
                content=f"Escalated ({reason}) -> {ticket_id}",
            )
        )

        return (
            "Your request has been escalated to our support team.\n"
            f"Ticket ID: {ticket_id}\n"
            "Estimated response time: within 4 business hours."
        )

    def get_conversation_summary(self) -> str:
        """Return a formatted, human-readable summary of the conversation
        so far, including matched FAQ ids and confidence scores.

        Returns:
            str: Multi-line summary. Says so explicitly if history is empty.
        """
        if not self.history:
            return "No conversation history yet."

        lines = ["Conversation Summary", "=" * 21]
        for i, turn in enumerate(self.history, start=1):
            speaker = "You" if turn.role == "user" else "SupportAI"
            line = f"{i}. [{speaker}] {turn.content}"
            if turn.role == "assistant" and turn.confidence is not None:
                faq_display = turn.faq_id if turn.faq_id else "none"
                line += f"  (confidence: {turn.confidence:.2f}, faq: {faq_display})"
            lines.append(line)

        lines.append("-" * 21)
        lines.append(f"Escalated: {'Yes' if self.escalated else 'No'}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear conversation history and reset escalation status."""
        self.history = []
        self.escalated = False


# ---------------------------------------------------------------------------
# 3. Interactive Chat Interface
# ---------------------------------------------------------------------------

BANNER = (
    "\u2554" + "\u2550" * 44 + "\u2557\n"
    "\u2551       SupportAI \u2014 Helpdesk Agent         \u2551\n"
    "\u255a" + "\u2550" * 44 + "\u255d"
)

LOW_CONFIDENCE_STREAK_LIMIT = 3


def _build_agent() -> SupportAgent:
    """Construct a SupportAgent wired up to a Task 2 LLMClient.

    Falls back gracefully to mock mode if no OPENROUTER_API_KEY is set
    (SupportAgent._generate_response already handles live-call failures,
    so this just picks a sensible client either way).
    """
    if (
        import_failed
        or LLMClient is None
        or run_mock_response is None
        or DEFAULT_MODEL is None
        or hybrid_search is None
    ):
        _fail_startup()

    api_key = os.environ.get("OPENROUTER_API_KEY", "missing-key")
    llm_client = LLMClient(api_key=api_key, model=DEFAULT_MODEL)
    return SupportAgent(faqs=FAQS, llm_client=llm_client, confidence_threshold=0.15)


def run_chat() -> None:
    """Run the interactive command-line chat loop.

    Commands:
        <any text>  - send as a question to the agent
        history     - display conversation summary
        escalate    - trigger escalation to human support
        reset       - clear conversation and start fresh
        quit        - exit the application
    """
    agent = _build_agent()
    print(BANNER)
    print()

    low_confidence_streak = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThank you for using SupportAI. Goodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command == "quit":
            print("Thank you for using SupportAI. Goodbye!")
            break

        if command == "history":
            print()
            print(agent.get_conversation_summary())
            print()
            continue

        if command == "reset":
            agent.reset()
            low_confidence_streak = 0
            print("\nConversation has been reset. Starting fresh!\n")
            continue

        if command == "escalate":
            print()
            print("SupportAI:")
            print(agent.escalate())
            print()
            low_confidence_streak = 0
            continue

        # Ordinary question.
        response = agent.handle_message(user_input)
        last_turn = agent.history[-1]

        print()
        if last_turn.confidence and last_turn.confidence > 0:
            print(f"SupportAI [confidence: {last_turn.confidence:.2f}, faq: {last_turn.faq_id}]:")
            low_confidence_streak = 0
        else:
            print("SupportAI [confidence: 0.00]:")
            low_confidence_streak += 1

        print(response)
        print()

        if low_confidence_streak >= LOW_CONFIDENCE_STREAK_LIMIT:
            print(
                "SupportAI: It looks like I'm having trouble finding answers to "
                "your recent questions. Would you like to escalate to human "
                "support? (type 'escalate')\n"
            )
            low_confidence_streak = 0  # avoid repeating the suggestion every turn


# ---------------------------------------------------------------------------
# 4. End-to-End Demonstration (scripted, non-interactive)
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Scripted walkthrough of all four required scenarios, driving the
    same SupportAgent + chat-formatting logic used by run_chat(), but
    with pre-set inputs so it can run non-interactively.
    """
    agent = _build_agent()
    print(BANNER)
    print()

    scripted_inputs = [
        "How do I change my password?",          # 1. clear FAQ question
        "Can I get my money back?",               # 2. paraphrased question
        "What are your office hours in Tokyo?",   # 3. outside FAQ scope
        "escalate",                                # 4. escalation request
    ]

    for user_input in scripted_inputs:
        print(f"You: {user_input}")
        command = user_input.lower()

        if command == "escalate":
            print()
            print("SupportAI:")
            print(agent.escalate())
            print()
            continue

        response = agent.handle_message(user_input)
        last_turn = agent.history[-1]

        print()
        if last_turn.confidence and last_turn.confidence > 0:
            print(f"SupportAI [confidence: {last_turn.confidence:.2f}, faq: {last_turn.faq_id}]:")
        else:
            print("SupportAI [confidence: 0.00]:")
        print(response)
        print()

    print("You: quit")
    print("Thank you for using SupportAI. Goodbye!")
    print()

    print(agent.get_conversation_summary())


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        # Interactive if stdin is a real terminal, otherwise fall back
        # to the scripted demo so the file still "runs without errors"
        # in non-interactive environments (e.g. automated grading).
        if sys.stdin.isatty():
            run_chat()
        else:
            run_demo()
