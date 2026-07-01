"""Check-worthiness filtering.

Not every answer sentence is a verifiable factual claim. Questions, hedges, and
meta-statements about the retrieval itself ("the passages do not mention X",
"unable to answer") are *not* hallucinations — but an NLI/grounding model scores
them as unsupported because they genuinely aren't entailed by the context.
Flagging them inflates false positives (on RAGTruth QA, ~17% of clean responses
contain such a sentence). We skip them before flagging.

The filter is deliberately conservative — it only excludes clear non-claims, per
the check-worthiness literature's warning that aggressive relevance filtering
removes genuinely verifiable claims (FActScore, VERISCORE, ClaimBuster).
"""

from __future__ import annotations

import re

# Meta-statements about the context / the model's ability to answer. These are
# often the *correct, honest* response and must never count as hallucinations.
_META_RE = re.compile(
    r"\b("
    r"unable to answer|cannot answer|can'?t answer|"
    r"(do(es)?\s+not|don'?t|cannot|can'?t)\s+"
    r"(mention|provide|specify|state|address|contain|say|include|indicate|determine)|"
    r"no\s+(information|mention|details|answer|indication|reference)|"
    r"there\s+(is|are)\s+no\s+(information|mention|details|indication)|"
    r"based\s+on\s+(the|given|provided)\s+(passages?|context|information)|"
    r"the\s+(passages?|context|text|document)s?\s+(do(es)?\s+not|don'?t)|"
    r"i\s+(cannot|can'?t|am\s+unable|do\s+not\s+have)|"
    r"insufficient\s+(information|context|data)|"
    r"it\s+is\s+(unclear|not\s+(clear|possible|specified|mentioned|stated))"
    r")\b",
    re.I,
)

_QUESTION_WORDS = (
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
)


def is_question(sentence: str) -> bool:
    """True if the sentence is interrogative."""
    s = sentence.strip()
    if not s:
        return False
    if s.endswith("?"):
        return True
    first = s.split(maxsplit=1)[0].lower().strip(",.;:\"'")
    return first in _QUESTION_WORDS and "?" in s


def is_checkworthy(sentence: str) -> bool:
    """True if the sentence is a verifiable factual claim worth grounding.

    Returns False for questions and meta/refusal statements about the context,
    which would otherwise be mislabelled as unsupported.
    """
    s = sentence.strip()
    if not s:
        return False
    if is_question(s):
        return False
    return not _META_RE.search(s)
