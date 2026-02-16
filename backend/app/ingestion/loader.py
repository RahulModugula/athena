import io
from dataclasses import dataclass

import structlog
from pypdf import PdfReader

logger = structlog.get_logger()


@dataclass
class Page:
    content: str
    page_number: int
    metadata: dict


def load_pdf(file_bytes: bytes, filename: str) -> list[Page]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(Page(content=text, page_number=i + 1, metadata={"source": filename}))
    logger.info("loaded pdf", filename=filename, pages=len(pages))
    return pages


def load_text(file_bytes: bytes, filename: str) -> list[Page]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [Page(content=text, page_number=1, metadata={"source": filename})]


def load_markdown(file_bytes: bytes, filename: str) -> list[Page]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [Page(content=text, page_number=1, metadata={"source": filename})]


LOADER_MAP = {
    "application/pdf": load_pdf,
    "text/plain": load_text,
    "text/markdown": load_markdown,
}


def load_document(file_bytes: bytes, filename: str, mime_type: str) -> list[Page]:
    loader = LOADER_MAP.get(mime_type)
    if loader is None:
        raise ValueError(f"unsupported mime type: {mime_type}")
    return loader(file_bytes, filename)
