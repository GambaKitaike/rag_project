from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import RecursiveUrlLoader
except ImportError:
    from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; HyperMeshRAG/1.0; +https://github.com/)"


def _fmt_duration(sec: float) -> str:
    if sec >= 3600:
        return f"{int(sec // 3600)}h {int((sec % 3600) // 60)}m {sec % 60:.0f}s"
    if sec >= 60:
        return f"{int(sec // 60)}m {sec % 60:.1f}s"
    return f"{sec:.2f}s"


@dataclass
class AltairManualLoaderConfig:
    start_url: str
    max_depth: int = 8
    allowed_domain: str = "altair.com"
    common_link_ratio_threshold: float = 0.7
    max_links_per_page: int = 300
    timeout_sec: int = 20
    user_agent: str = DEFAULT_USER_AGENT


def _is_allowed_domain(url: str, allowed_domain: str) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    allowed = allowed_domain.lower()
    return netloc == allowed or netloc.endswith(f".{allowed}")


def _normalize_url(base_url: str, href: str) -> str | None:
    if not href:
        return None
    lowered = href.strip().lower()
    if lowered.startswith(("javascript:", "mailto:", "tel:")):
        return None

    joined = urljoin(base_url, href.strip())
    clean, _fragment = urldefrag(joined)
    parsed = urlparse(clean)
    if parsed.scheme not in {"http", "https"}:
        return None
    return clean


def _is_navigation_area(anchor_tag) -> bool:
    for parent in anchor_tag.parents:
        if parent is None:
            continue
        tag_name = (parent.name or "").lower()
        if tag_name in {"header", "footer", "nav", "aside"}:
            return True
        attrs = " ".join(
            str(parent.get(key, "")) for key in ("id", "class", "role", "aria-label")
        ).lower()
        if any(token in attrs for token in ("nav", "menu", "breadcrumb", "header", "footer")):
            return True
    return False


def _clean_page_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_links_for_graph(
    soup: BeautifulSoup, base_url: str, allowed_domain: str
) -> tuple[list[str], list[str]]:
    all_internal_links: list[str] = []
    candidate_links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        normalized = _normalize_url(base_url, anchor.get("href", ""))
        if not normalized or not _is_allowed_domain(normalized, allowed_domain):
            continue

        all_internal_links.append(normalized)
        if not _is_navigation_area(anchor):
            candidate_links.append(normalized)

    dedup_all = list(dict.fromkeys(all_internal_links))
    dedup_candidate = list(dict.fromkeys(candidate_links))
    return dedup_candidate, dedup_all


def _filter_common_links(
    page_links: dict[str, list[str]], threshold_ratio: float, total_pages: int
) -> tuple[dict[str, list[str]], set[str]]:
    if total_pages <= 1:
        return page_links, set()

    presence_counter = Counter()
    for links in page_links.values():
        presence_counter.update(set(links))

    threshold_count = max(2, int(total_pages * threshold_ratio))
    common_links = {url for url, count in presence_counter.items() if count >= threshold_count}

    filtered = {
        url: [link for link in links if link not in common_links]
        for url, links in page_links.items()
    }
    return filtered, common_links


def _load_recursive_docs(config: AltairManualLoaderConfig) -> list[Document]:
    loader = RecursiveUrlLoader(
        url=config.start_url,
        max_depth=config.max_depth,
        prevent_outside=True,
        timeout=config.timeout_sec,
        check_response_status=True,
        continue_on_failure=True,
        headers={"User-Agent": config.user_agent},
    )
    docs = loader.load()
    return [
        doc
        for doc in docs
        if _is_allowed_domain(doc.metadata.get("source", ""), config.allowed_domain)
    ]


def load_altair_manual_documents(
    start_url: str,
    max_depth: int = 8,
    allowed_domain: str = "altair.com",
    common_link_ratio_threshold: float = 0.7,
    max_links_per_page: int = 300,
    timeout_sec: int = 20,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[Document]:
    """
    Recursively crawl manual HTML with RecursiveUrlLoader, extract body text and
    internal links with BeautifulSoup (single parse per page).

    Each Document has metadata["source_url"] and metadata["links"] (nav-filtered,
    common-link filtered). metadata["all_links"] lists all *.allowed_domain links.
    """
    config = AltairManualLoaderConfig(
        start_url=start_url,
        max_depth=max_depth,
        allowed_domain=allowed_domain,
        common_link_ratio_threshold=common_link_ratio_threshold,
        max_links_per_page=max_links_per_page,
        timeout_sec=timeout_sec,
        user_agent=user_agent,
    )

    t0 = time.perf_counter()
    raw_docs = _load_recursive_docs(config)
    t_fetch = time.perf_counter() - t0
    print(
        f"[タイミング] 再帰HTTP取得 (RecursiveUrlLoader): {_fmt_duration(t_fetch)} "
        f"（生HTML {len(raw_docs)} ページ）"
    )

    page_candidates: dict[str, list[str]] = {}
    page_all_internal_links: dict[str, list[str]] = {}
    parsed_docs: list[Document] = []

    t1 = time.perf_counter()
    for i, doc in enumerate(raw_docs, start=1):
        source_url = doc.metadata.get("source", "")
        if not source_url:
            continue

        try:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            clean_text = _clean_page_text(soup)
            candidate_links, all_links = _extract_links_for_graph(
                soup=soup,
                base_url=source_url,
                allowed_domain=config.allowed_domain,
            )
        except Exception:
            logger.exception("Failed to parse page: %s", source_url)
            continue

        page_candidates[source_url] = candidate_links[: config.max_links_per_page]
        page_all_internal_links[source_url] = all_links[: config.max_links_per_page]
        parsed_docs.append(
            Document(
                page_content=clean_text,
                metadata={
                    **doc.metadata,
                    "source": source_url,
                    "source_url": source_url,
                },
            )
        )
        if i % 50 == 0 or i == len(raw_docs):
            elapsed = time.perf_counter() - t1
            print(
                f"[進捗] HTML解析・リンク抽出 {i}/{len(raw_docs)} ページ "
                f"（経過 {_fmt_duration(elapsed)}）"
            )

    t_parse = time.perf_counter() - t1
    print(
        f"[タイミング] HTML解析・リンク抽出 (BeautifulSoup): {_fmt_duration(t_parse)} "
        f"（{len(parsed_docs)} ページ処理）"
    )

    t2 = time.perf_counter()
    filtered_links, common_links = _filter_common_links(
        page_links=page_candidates,
        threshold_ratio=config.common_link_ratio_threshold,
        total_pages=len(parsed_docs),
    )

    finalized_docs: list[Document] = []
    for doc in parsed_docs:
        source_url = doc.metadata["source_url"]
        page_links = filtered_links.get(source_url, [])
        all_links = page_all_internal_links.get(source_url, [])

        updated_metadata = {
            **doc.metadata,
            "links": page_links,
            "all_links": all_links,
            "all_links_count": len(all_links),
            "links_count": len(page_links),
            "excluded_links_count": max(0, len(all_links) - len(page_links)),
            "common_links_threshold": config.common_link_ratio_threshold,
            "common_links_detected_total": len(common_links),
        }
        finalized_docs.append(Document(page_content=doc.page_content, metadata=updated_metadata))

    t_post = time.perf_counter() - t2
    print(f"[タイミング] リンク後処理（共通リンク除去・metadata確定）: {_fmt_duration(t_post)}")

    total = time.perf_counter() - t0
    print(
        f"[タイミング] ローダー合計: {_fmt_duration(total)} "
        f"（取得 {_fmt_duration(t_fetch)} / 解析 {_fmt_duration(t_parse)} / 後処理 {_fmt_duration(t_post)}）"
    )
    print(
        "✅ Recursive loader 完了 "
        f"(pages={len(finalized_docs)}, common_links={len(common_links)}, depth={config.max_depth})"
    )
    return finalized_docs


def build_hyperlink_graph(documents: List[Document]) -> Dict[str, List[str]]:
    """Edges use metadata['links']; endpoints restricted to crawled page URLs."""
    nodes = {
        (doc.metadata.get("source_url") or doc.metadata.get("source") or "")
        for doc in documents
    }
    nodes.discard("")

    graph: Dict[str, List[str]] = {}
    for doc in documents:
        source = doc.metadata.get("source_url") or doc.metadata.get("source", "")
        links = doc.metadata.get("links", [])
        graph[source] = [link for link in links if link in nodes]
    return graph
