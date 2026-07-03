"""athena-verify live demo — Hugging Face Space (Gradio).

Paste a question, an LLM answer, and your retrieved context chunks (one per
line). You get back a per-sentence trust score, the unsupported sentences, and
the exact source span that did or did not support each claim.

Runs the real library, zero-shot, on CPU. No API keys, nothing leaves the Space.

Local run:
    pip install -r demo/requirements.txt
    python demo/app.py
"""

from __future__ import annotations

import contextlib
import html

import gradio as gr

from athena_verify import verify

# --- Colour scheme by support status (readable in light and dark) -------------
STATUS_STYLE = {
    "SUPPORTED": ("#0f5132", "#d1e7dd", "✓"),
    "PARTIAL": ("#664d03", "#fff3cd", "~"),
    "UNSUPPORTED": ("#842029", "#f8d7da", "✗"),
    "CONTRADICTED": ("#842029", "#f5c2c7", "✗"),
}

# --- Examples: each is (question, answer, context) ----------------------------
# The legal-contract example is the money demo: NLI alone is weak at number
# substitution, so a hallucinated "$1M" (context says "$2M") landing in red is
# the exact failure mode people don't expect a 25ms local model to catch.
EXAMPLES = [
    [
        "What are the key terms of the contract?",
        "The liability cap is $1,000,000 per incident. "
        "The agreement can be terminated with 30 days notice. "
        "It was signed on January 15, 2024.",
        "The indemnification clause in Section 12.1 states that the liability cap is "
        "$2,000,000 per incident, with an aggregate annual cap of $5,000,000.\n"
        "The agreement was signed on January 15, 2024, by both parties.\n"
        "Either party may terminate this agreement with 90 days written notice.",
    ],
    [
        "What is the recommended dosage?",
        "The recommended dose is 400mg taken twice daily with food. "
        "It should not be taken with alcohol.",
        "The recommended dosage is 200mg administered twice daily.\n"
        "Patients should take the medication with food to reduce stomach upset.",
    ],
    [
        "Who founded the company and when?",
        "Acme Corp was founded in 2011 by Jane Doe and John Smith in Austin, Texas.",
        "Acme Corp was founded in 2011 by Jane Doe.\n"
        "The company is headquartered in Austin, Texas.",
    ],
]


def _render(result) -> str:  # noqa: ANN001 - VerificationResult
    passed = result.verification_passed
    header_colour = "#0f5132" if passed else "#842029"
    header_bg = "#d1e7dd" if passed else "#f8d7da"
    verdict = "PASSED" if passed else "UNSUPPORTED CLAIMS FOUND"

    parts = [
        '<div style="font-family:system-ui,sans-serif;line-height:1.5">',
        f'<div style="padding:10px 14px;border-radius:8px;background:{header_bg};'
        f'color:{header_colour};font-weight:600;margin-bottom:14px">'
        f"Overall trust: {result.trust_score:.0%} &nbsp;·&nbsp; {verdict} "
        f"&nbsp;·&nbsp; {len(result.unsupported)} of {len(result.sentences)} "
        f"sentence(s) flagged</div>",
    ]

    for s in result.sentences:
        fg, bg, mark = STATUS_STYLE.get(s.support_status, ("#333", "#eee", "?"))
        parts.append(
            f'<div style="border-left:4px solid {fg};background:{bg};'
            f'padding:8px 12px;margin:6px 0;border-radius:4px">'
            f'<div style="color:{fg};font-weight:600">'
            f"{mark} [{s.support_status}] &nbsp;trust {s.trust_score:.0%}</div>"
            f'<div style="color:#111;margin-top:2px">{html.escape(s.text)}</div>'
        )
        if s.supporting_spans:
            for span in s.supporting_spans:
                parts.append(
                    f'<div style="color:#333;font-size:0.85em;margin-top:4px;'
                    f'padding-left:10px;border-left:2px solid #bbb">'
                    f"← chunk[{span.chunk_idx}] supports this: "
                    f"<i>{html.escape(span.text)}</i></div>"
                )
        elif s.best_matching_context:
            snippet = s.best_matching_context[:160]
            parts.append(
                f'<div style="color:#666;font-size:0.85em;margin-top:4px;'
                f'padding-left:10px;border-left:2px solid #bbb">'
                f"closest context (not sufficient): <i>{html.escape(snippet)}</i></div>"
            )
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def run_verify(question: str, answer: str, context_text: str) -> str:
    if not answer.strip():
        return '<div style="color:#842029">Please enter an answer to verify.</div>'
    context = [line.strip() for line in context_text.splitlines() if line.strip()]
    if not context:
        return (
            '<div style="color:#842029">Please enter at least one context chunk '
            "(one per line).</div>"
        )
    result = verify(question=question.strip(), answer=answer.strip(), context=context)
    return _render(result)


with gr.Blocks(title="athena-verify — RAG hallucination detector") as demo:
    gr.Markdown(
        "# athena-verify\n"
        "**Local, zero-shot RAG hallucination detection.** Paste an LLM answer and the "
        "context you retrieved; each sentence is scored against the context and the "
        "unsupported ones are flagged, with the source span that did (or didn't) support it. "
        "Runs on a small NLI model — no API keys, nothing leaves this Space.\n\n"
        "[GitHub](https://github.com/RahulModugula/athena) · `pip install athena-verify` · MIT"
    )
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Question", lines=1)
            answer = gr.Textbox(label="LLM answer (the thing to verify)", lines=4)
            context = gr.Textbox(
                label="Retrieved context — one chunk per line", lines=6
            )
            btn = gr.Button("Verify", variant="primary")
        with gr.Column():
            output = gr.HTML(label="Result")

    gr.Examples(examples=EXAMPLES, inputs=[question, answer, context])
    btn.click(run_verify, inputs=[question, answer, context], outputs=output)


if __name__ == "__main__":
    # Pre-warm the NLI model so the first real request is fast (best-effort).
    with contextlib.suppress(Exception):
        verify(question="warmup", answer="The sky is blue.", context=["The sky is blue."])
    demo.launch()
