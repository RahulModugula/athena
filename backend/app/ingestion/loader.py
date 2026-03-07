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


def load_html(file_bytes: bytes, filename: str) -> list[Page]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(file_bytes, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    logger.info("loaded html", filename=filename)
    return [Page(content=text, page_number=1, metadata={"source": filename})]


def load_docx(file_bytes: bytes, filename: str) -> list[Page]:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    logger.info("loaded docx", filename=filename, paragraphs=len(paragraphs))
    return [Page(content=text, page_number=1, metadata={"source": filename})]


LOADER_MAP = {
    "application/pdf": load_pdf,
    "text/plain": load_text,
    "text/markdown": load_markdown,
    "text/html": load_html,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": load_docx,
}


def load_document(file_bytes: bytes, filename: str, mime_type: str) -> list[Page]:
    loader = LOADER_MAP.get(mime_type)
    if loader is None:
        raise ValueError(f"unsupported mime type: {mime_type}")
    return loader(file_bytes, filename)
