"""Command-line interface for athena-verify."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from athena_verify import verify


def color_score(score: float) -> str:
    """Return ANSI color code based on trust score."""
    if score >= 0.8:
        return "\033[92m"  # green
    elif score >= 0.5:
        return "\033[93m"  # yellow
    else:
        return "\033[91m"  # red


def reset_color() -> str:
    """Return ANSI reset code."""
    return "\033[0m"


def format_trust_score(score: float, width: int = 6) -> str:
    """Format trust score with color."""
    return f"{color_score(score)}{score:.2f}{reset_color()}"


def print_table(result) -> None:
    """Print colored sentence-by-sentence trust score table."""
    print()
    print("Verification Results")
    print("=" * 100)
    print(f"Overall Trust Score: {format_trust_score(result.trust_score)}")
    print(f"Status: {'✓ PASSED' if result.verification_passed else '✗ FAILED'}")
    print("=" * 100)
    print()

    # Header
    print(
        f"{'#':<3} {'Trust':<8} {'Status':<12} {'Sentence':<70}",
    )
    print("-" * 100)

    # Rows
    for sentence in result.sentences:
        idx = sentence.index + 1
        trust = format_trust_score(sentence.trust_score, width=6)
        status = sentence.support_status
        text = sentence.text[:67] + "..." if len(sentence.text) > 70 else sentence.text

        print(f"{idx:<3} {trust:<8} {status:<12} {text:<70}")

    print("-" * 100)
    print()

    # Summary
    if result.unsupported:
        print(f"⚠ Unsupported sentences ({len(result.unsupported)}):")
        for sentence in result.unsupported:
            print(f"  • {sentence.text}")
        print()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="athena-verify",
        description="Verify RAG answers against context for hallucinations",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify an answer against context")
    verify_parser.add_argument(
        "--answer",
        required=True,
        help="The answer text to verify",
    )
    verify_parser.add_argument(
        "--context",
        required=True,
        help="Path to context file (text file or JSON)",
    )
    verify_parser.add_argument(
        "--question",
        default="",
        help="Optional question (for context)",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (machine-readable)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load context from file
    context_path = Path(args.context)
    if not context_path.exists():
        print(f"Error: Context file not found: {args.context}", file=sys.stderr)
        sys.exit(1)

    context_text = context_path.read_text()

    # Try to parse as JSON, fall back to plain text
    try:
        context_data = json.loads(context_text)
        if isinstance(context_data, list):
            context = context_data
        elif isinstance(context_data, dict) and "context" in context_data:
            context = context_data["context"]
        else:
            context = [context_data]
    except json.JSONDecodeError:
        context = [context_text]

    # Run verification
    result = verify(
        question=args.question,
        answer=args.answer,
        context=context,
    )

    # Output
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print_table(result)


if __name__ == "__main__":
    main()
