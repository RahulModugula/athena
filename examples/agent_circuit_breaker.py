"""Agent circuit breaker demo using verify_step().

Demonstrates a 3-step agent where intermediate claims are verified.
Steps 1-2 pass verification; step 3 contains a fabricated claim that
halts the chain before it can propagate.
"""

from athena_verify import verify_step

EVIDENCE = [
    "The Eiffel Tower is located in Paris, France.",
    "It was constructed between 1887 and 1889.",
    "The tower stands 330 metres tall.",
]

STEPS = [
    ("The Eiffel Tower is in Paris, France.", True),  # supported
    ("The tower is 330 metres tall.", True),  # supported
    ("The Eiffel Tower was built in 1650.", False),  # fabricated — halts
]


def run_agent() -> None:
    """Run a 3-step agent with circuit breaker logic."""
    print("Agent Circuit Breaker Demo")
    print("=" * 60)
    print(f"Evidence: {len(EVIDENCE)} documents\n")

    for i, (claim, _expected_pass) in enumerate(STEPS, 1):
        result = verify_step(claim=claim, evidence=EVIDENCE)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"Step {i}: {status}")
        print(f"  Claim: '{claim}'")
        print(f"  Trust Score: {result.trust_score:.3f}")
        print(f"  Action: {result.action}")

        if result.action == "halt":
            print("\n  [CIRCUIT BREAKER] Halting agent — fabricated claim detected.")
            break
        print()
    else:
        print("All steps passed. Agent completed successfully.")


if __name__ == "__main__":
    run_agent()
