"""Agent circuit breaker — stop a hallucination before it cascades.

A multi-step agent passes each step's output to the next. When step 3
hallucinates a figure that isn't in the source, an unguarded agent would
carry it into its final BUY recommendation. `verify_step()` is a circuit
breaker: it halts the chain the moment a step stops being grounded.

Run with: python examples/agent_circuit_breaker.py
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys

# Keep the demo output clean — silence the ML stack's load reports / progress
# bars before anything imports it.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import structlog  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from athena_verify import verify_step  # noqa: E402  (configure logging first)

# ANSI colors (no dependency); disabled when output isn't a TTY.
_TTY = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


DIM, BOLD = "2", "1"
GREEN, RED, YELLOW = "32", "31", "33"

# What the agent is allowed to rely on — retrieved from a 10-Q filing.
EVIDENCE = [
    "Acme Corp reported Q3 revenue of $2.4 billion, up 12% year over year.",
    "Operating income was $530 million for the quarter.",
    "Net profit margin for Q3 was 22%, in line with the prior quarter.",
    "The company reaffirmed full-year guidance and declared a $0.15 dividend.",
]

# Each reasoning step the agent produces, fed forward to the next.
STEPS = [
    "Acme's Q3 revenue was $2.4 billion, up 12% year over year.",
    "Operating income for the quarter came in at $530 million.",
    "Net margin expanded sharply to 35%, a major profitability breakout.",
    "Given the margin breakout, raise the price target and recommend BUY.",
]


def main() -> None:
    print(c("\n  Financial research agent — 4 reasoning steps", BOLD))
    print(c(f"  Grounding on {len(EVIDENCE)} passages from Acme's 10-Q", DIM))

    # Load the NLI model up front (and quietly) so the steps below stream
    # without a pause or stray library output.
    print(c("  loading grounding model…\n", DIM), flush=True)
    with contextlib.redirect_stderr(io.StringIO()):
        verify_step(claim="warm up", evidence=["warm up"])

    for i, claim in enumerate(STEPS, 1):
        step = verify_step(claim=claim, evidence=EVIDENCE, threshold=0.5)
        badge = (
            c(" PASS ", f"{BOLD};{GREEN}")
            if step.passed
            else c(" HALT ", f"{BOLD};{RED}")
        )
        print(f"  {badge} step {i}  {c(f'trust={step.trust_score:.2f}', DIM)}")
        print(f"         {claim}")

        if step.action == "halt":
            print()
            print(c("  ⛔ circuit breaker tripped — ungrounded claim blocked", f"{BOLD};{RED}"))
            print(c("     the agent never reached step 4, so the BUY call built", YELLOW))
            print(c("     on a hallucinated 35% margin was never made.", YELLOW))
            print(c("\n     Source says: net profit margin for Q3 was 22%.", DIM))
            return
        print()

    print(c("  ✓ all steps grounded — recommendation cleared to proceed", GREEN))


if __name__ == "__main__":
    main()
