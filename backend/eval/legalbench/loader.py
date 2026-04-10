"""Load and ingest LegalBench-RAG dataset for evaluation."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_legalbench_qa(dataset_path: str | Path | None = None) -> list[dict]:
    """Load LegalBench-RAG QA pairs from a JSONL file.

    Expected format:
    [
        {
            "question": "...",
            "answer": "...",
            "source_files": ["file1.pdf", ...] (optional)
        }
    ]

    Args:
        dataset_path: Path to JSONL file with QA pairs.
                     If None, looks for eval/data/legalbench.jsonl

    Returns:
        List of QA pair dicts
    """
    if dataset_path is None:
        dataset_path = Path(__file__).parent.parent / "data" / "legalbench.jsonl"

    if not Path(dataset_path).exists():
        logger.warning(f"dataset not found: {dataset_path}")
        return []

    qa_pairs = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    qa_pairs.append(obj)
                except json.JSONDecodeError:
                    logger.warning("failed to parse line")
                    continue

    logger.info(f"loaded {len(qa_pairs)} QA pairs from {dataset_path}")
    return qa_pairs


def load_legalbench_corpus(corpus_path: str | Path | None = None) -> dict:
    """Load source documents for LegalBench evaluation.

    Expected: a directory with contract PDFs/TXTs.

    Args:
        corpus_path: Path to directory containing source documents.
                    If None, looks for eval/data/legalbench_corpus/

    Returns:
        Dict mapping filename -> content
    """
    if corpus_path is None:
        corpus_path = Path(__file__).parent.parent / "data" / "legalbench_corpus"

    corpus = {}
    corpus_dir = Path(corpus_path)

    if not corpus_dir.exists():
        logger.warning(f"corpus directory not found: {corpus_path}")
        return corpus

    for file_path in corpus_dir.glob("**/*"):
        if file_path.is_file() and file_path.suffix in (".txt", ".pdf"):
            try:
                # For MVP, just support TXT files
                if file_path.suffix == ".txt":
                    with open(file_path, encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    corpus[file_path.name] = content
            except Exception as e:
                logger.warning(f"failed to load {file_path}: {e}")

    logger.info(f"loaded {len(corpus)} documents from {corpus_path}")
    return corpus
