# SupportAI — Conversational Helpdesk Agent

A four-part project that builds a working helpdesk chatbot from the ground up: a
searchable FAQ knowledge base, an LLM layer that rephrases answers naturally, an
intelligent hybrid matcher that understands paraphrased questions, and a stateful
agent that ties it all together into a multi-turn chat with escalation to human
support.

## Project structure

```
task1_faq_search.py     Task 1 — FAQ data (FAQS) and keyword-based search
task2_llm_client.py      Task 2 — LLMClient, rephrases FAQ answers via OpenRouter
task3_faq_matcher.py     Task 3 — FAQMatcher (TF-IDF + cosine similarity) and hybrid_search()
task4_support_agent.py   Task 4 — SupportAgent orchestration, chat interface, demo
```

All four files must live in the same directory — Task 4 imports directly from
Tasks 1–3 and does not reimplement any of their logic.

## Setup

```bash
pip install requests scikit-learn
```

Optional — enable live LLM-generated responses instead of mock rephrasing:

```bash
export OPENROUTER_API_KEY="sk-or-..."      # macOS/Linux
setx OPENROUTER_API_KEY "sk-or-..."        # Windows
```

Without a key (or without network access), the project automatically falls
back to a clearly-labelled `[MOCK RESPONSE ...]` mode so every module still
runs end-to-end offline.

## Running the project

```bash
python task4_support_agent.py            # interactive chat (in a real terminal)
python task4_support_agent.py --demo     # scripted 4-scenario demo, non-interactive
```

Each task module can also be run individually to see its own demo:

```bash
python task1_faq_search.py
python task2_llm_client.py
python task3_faq_matcher.py
```

## Task 1 — FAQ Knowledge Base

`task1_faq_search.py` defines:

- **`FAQS`** — 7 FAQ entries across Account, Billing, Technical, and Shipping
  categories, each with an `id`, `category`, `question`, `answer`, and a list
  of `keywords`.
- **`search_by_keyword(faqs, query)`** — case-insensitive substring matching
  against each FAQ's category/question/keywords, ranked by hit count.
- **`get_faq_by_id(faqs, faq_id)`** — direct lookup by id.
- **`get_faqs_by_category(faqs, category)`** — filter by category.

This is plain Python with no external dependencies or LLM calls.

## Task 2 — LLM Response Generation

`task2_llm_client.py` adds an LLM layer on top of Task 1:

- **`LLMClient`** — a thin wrapper around OpenRouter's chat completions API
  (`openai/gpt-4o-mini` by default).
  - `generate(prompt, system_message=None, max_tokens=512)` — general-purpose
    chat completion.
  - `generate_faq_response(user_question, faq_entry)` — rephrases a matched
    FAQ's official answer into a natural, conversational reply, grounded
    strictly in the FAQ content (the system prompt explicitly forbids
    inventing facts, policies, numbers, or steps not present in the FAQ).
- **`run_mock_response(user_question, faq_entry)`** — offline fallback used
  when no API key/network is available, so demos never crash.

Requires `requests`.

## Task 3 — Intelligent FAQ Matching

`task3_faq_matcher.py` adds semantic matching on top of Task 1's exact
keyword search:

- **`FAQMatcher`** — builds a TF-IDF index over each FAQ's question +
  keywords, and scores a query against every FAQ via cosine similarity.
  - `match(query, top_k=3)` — top-k matches with similarity scores.
  - `best_match(query, threshold=0.15)` — single best match if it clears the
    confidence threshold, else `None`.
  - `explain_match(query)` — human-readable ranked explanation for debugging.
- **`hybrid_search(faqs, query, top_k=3)`** — merges Task 1's keyword search
  (base score 0.5) with Task 3's TF-IDF score, keeping the **higher** of the
  two per FAQ, so paraphrases are still found even when they share few exact
  words with the FAQ text.

Requires `scikit-learn`.

## Task 4 — Complete Helpdesk Agent

`task4_support_agent.py` orchestrates Tasks 1–3 into a stateful agent —
it only *calls* `hybrid_search()` and `llm_client.generate_faq_response()`;
it does not reimplement their logic.

### `ConversationTurn`

```python
@dataclass
class ConversationTurn:
    role: str        # "user" or "assistant"
    content: str
    faq_id: str = None
    confidence: float = None
```

### `SupportAgent`

| Method | Behaviour |
|---|---|
| `__init__(faqs, llm_client, confidence_threshold=0.15)` | Stores FAQ data, LLM client, and the minimum `hybrid_search()` score required for a confident match. |
| `handle_message(user_message)` | Records the user turn, runs `hybrid_search()`, and either returns an LLM-phrased answer (confident match) or a fallback + escalation offer (score 0.0, no match). |
| `escalate(reason=...)` | Marks the session escalated and returns a confirmation with a mock ticket ID (`TICKET-XXXXX`) and ETA. |
| `get_conversation_summary()` | Formats the full turn-by-turn history with FAQ IDs, confidence scores, and escalation status. |
| `reset()` | Clears history and escalation status for a fresh session. |

If the live OpenRouter call fails (no key/network), `handle_message()`
gracefully falls back to Task 2's `run_mock_response()` instead of raising an
error.

### Interactive chat interface

| Input | Action |
|---|---|
| Any text | Send as a question to the agent |
| `history` | Display the conversation summary |
| `escalate` | Trigger escalation to human support |
| `reset` | Clear conversation and start fresh |
| `quit` | Exit the application |

- Displays a welcome banner on startup.
- Shows the confidence score and matched FAQ ID after every response.
- Tracks a low-confidence streak, reset on any confident answer; after 3
  consecutive low-confidence turns, the agent proactively suggests
  escalation.

### End-to-end demo

`python task4_support_agent.py --demo` scripts all four required scenarios:

| # | Scenario | Expected behaviour |
|---|---|---|
| 1 | Clear FAQ question | Agent answers with high confidence |
| 2 | Paraphrased question | Agent matches via `hybrid_search()` and answers |
| 3 | Question outside FAQ scope | Agent responds with fallback and offers escalation |
| 4 | Escalation request | Agent creates a mock ticket and confirms |

Sample output:

```
You: How do I change my password?

SupportAI [confidence: 0.60, faq: faq-001]:
No worries! Head to the login page and click "Forgot Password." ...

You: Can I get my money back?

SupportAI [confidence: 0.50, faq: faq-002]:
Yes! We offer full refunds within 30 days of purchase...

You: What are your office hours in Tokyo?

SupportAI [confidence: 0.00]:
I don't have information about that in my knowledge base.
Would you like me to connect you with a human support agent?

You: escalate

SupportAI:
Your request has been escalated to our support team.
Ticket ID: TICKET-48291
Estimated response time: within 4 business hours.

You: quit
Thank you for using SupportAI. Goodbye!
```

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'task1_faq_search'` | All four `task*.py` files must be in the same folder, and you must run Task 4 from that folder. |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn requests`. |
| `[MOCK RESPONSE ...]` in every reply | Expected without `OPENROUTER_API_KEY` / network access — not an error. Set the key to get live LLM phrasing. |
| Interactive mode exits straight to the demo | By design: piped/redirected stdin isn't a real terminal, so the script falls back to `run_demo()`. Run in an actual terminal for the interactive chat, or pass `--demo` explicitly when scripting/CI. |
