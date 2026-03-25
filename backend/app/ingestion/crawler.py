"""Web crawler for documentation sites.

Fetches pages from a documentation site via sitemap.xml or recursive
link-following, returning each page as raw bytes + metadata suitable for
the existing ingestion pipeline.
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import structlog

logger = structlog.get_logger()

_DEFAULT_UA = "Athena-DocsCrawler/1.0"


@dataclass
class CrawledPage:
    url: str
    content_bytes: bytes
    mime_type: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass
class CrawlResult:
    pages: list[CrawledPage] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    pages_skipped: int = 0


async def crawl_sitemap(
    base_url: str,
    *,
    sitemap_url: str | None = None,
    url_pattern: str | None = None,
    max_pages: int = 500,
    request_delay: float = 0.25,
    existing_etags: dict[str, str] | None = None,
) -> CrawlResult:
    """Crawl a documentation site by reading its sitemap.xml.

    Parameters
    ----------
    base_url:
        Root URL of the docs site (e.g. ``https://docs.example.com``).
    sitemap_url:
        Explicit sitemap URL.  Defaults to ``{base_url}/sitemap.xml``.
    url_pattern:
        Optional regex to filter which URLs to crawl.
    max_pages:
        Stop after fetching this many pages.
    request_delay:
        Seconds to wait between HTTP requests.
    existing_etags:
        Mapping of URL → ETag from previous crawl.  Pages with matching
        ETags are skipped (incremental re-index).
    """
    result = CrawlResult()
    existing_etags = existing_etags or {}
    url_re = re.compile(url_pattern) if url_pattern else None

    sitemap_url = sitemap_url or f"{base_url.rstrip('/')}/sitemap.xml"

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": _DEFAULT_UA},
    ) as client:
        # ---- Check robots.txt ----
        robots = RobotFileParser()
        robots_url = f"{base_url.rstrip('/')}/robots.txt"
        try:
            resp = await client.get(robots_url)
            if resp.status_code == 200:
                robots.parse(resp.text.splitlines())
        except Exception:
            pass  # if robots.txt fails, proceed without restrictions

        # ---- Fetch sitemap ----
        try:
            resp = await client.get(sitemap_url)
            resp.raise_for_status()
        except Exception as exc:
            result.errors.append({"url": sitemap_url, "error": str(exc)})
            return result

        urls = _parse_sitemap(resp.text, base_url)
        logger.info("sitemap parsed", url_count=len(urls), sitemap=sitemap_url)

        # ---- Crawl pages ----
        fetched = 0
        for page_url in urls:
            if fetched >= max_pages:
                break

            if url_re and not url_re.search(page_url):
                continue

            if not robots.can_fetch(_DEFAULT_UA, page_url):
                result.pages_skipped += 1
                continue

            try:
                headers: dict[str, str] = {}
                old_etag = existing_etags.get(page_url)
                if old_etag:
                    headers["If-None-Match"] = old_etag

                resp = await client.get(page_url, headers=headers)

                if resp.status_code == 304:
                    result.pages_skipped += 1
                    continue

                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "text/html")
                mime = content_type.split(";")[0].strip()

                result.pages.append(CrawledPage(
                    url=page_url,
                    content_bytes=resp.content,
                    mime_type=mime,
                    etag=resp.headers.get("etag"),
                    last_modified=resp.headers.get("last-modified"),
                ))
                fetched += 1

            except Exception as exc:
                result.errors.append({"url": page_url, "error": str(exc)})

            if request_delay > 0:
                await asyncio.sleep(request_delay)

    logger.info(
        "crawl complete",
        fetched=len(result.pages),
        skipped=result.pages_skipped,
        errors=len(result.errors),
    )
    return result


def _parse_sitemap(xml_text: str, base_url: str) -> list[str]:
    """Extract URLs from a sitemap XML document."""
    urls: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return urls

    # Handle namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Handle sitemap index (contains other sitemaps)
    for sitemap_tag in root.findall(f"{ns}sitemap"):
        loc = sitemap_tag.find(f"{ns}loc")
        if loc is not None and loc.text:
            urls.append(loc.text.strip())

    # Handle urlset (contains actual pages)
    for url_tag in root.findall(f"{ns}url"):
        loc = url_tag.find(f"{ns}loc")
        if loc is not None and loc.text:
            url = loc.text.strip()
            # Only include URLs from the same domain
            if urlparse(url).netloc == urlparse(base_url).netloc:
                urls.append(url)

    return urls
