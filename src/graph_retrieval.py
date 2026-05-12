from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


def build_adjacency_from_chunks(chunks: list[Document]) -> dict[str, list[str]]:
    """One adjacency list per page URL (first chunk wins; metadata is identical per page)."""
    adj: dict[str, list[str]] = {}
    for d in chunks:
        url = d.metadata.get("source_url") or d.metadata.get("source")
        if not url or url in adj:
            continue
        links = d.metadata.get("links")
        if isinstance(links, list):
            adj[url] = [str(x) for x in links]
        else:
            adj[url] = []
    return adj


def index_chunks_by_url(chunks: list[Document]) -> dict[str, list[Document]]:
    by_url: dict[str, list[Document]] = defaultdict(list)
    for d in chunks:
        url = d.metadata.get("source_url") or d.metadata.get("source")
        if url:
            by_url[str(url)].append(d)
    return dict(by_url)


def graph_expand(
    seed_docs: list[Document],
    adjacency: dict[str, list[str]],
    chunks_by_url: dict[str, list[Document]],
    max_extra_chunks: int,
    chunks_per_neighbor: int,
) -> list[Document]:
    """Append chunks from linked pages (1-hop) not already represented in seed_docs."""
    if max_extra_chunks <= 0:
        return list(seed_docs)

    in_index = set(chunks_by_url)
    seen_chunk_ids: set[int] = set()
    seen_urls: set[str] = set()
    out: list[Document] = []

    for d in seed_docs:
        cid = id(d)
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            out.append(d)
            u = d.metadata.get("source_url") or d.metadata.get("source")
            if u:
                seen_urls.add(str(u))

    extra_added = 0
    for u in list(seen_urls):
        if extra_added >= max_extra_chunks:
            break
        for nbr in adjacency.get(u, []):
            if extra_added >= max_extra_chunks:
                break
            if nbr in seen_urls or nbr not in in_index:
                continue
            seen_urls.add(nbr)
            for ch in chunks_by_url.get(nbr, [])[:chunks_per_neighbor]:
                if extra_added >= max_extra_chunks:
                    break
                ch_id = id(ch)
                if ch_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(ch_id)
                out.append(ch)
                extra_added += 1
    return out


class GraphExpandedRetriever(BaseRetriever):
    """Runs a base retriever, then adds chunks from hyperlink-adjacent pages (same index)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any = Field(description="Typically EnsembleRetriever (vector + BM25).")
    adjacency: dict[str, list[str]] = Field(default_factory=dict)
    chunks_by_url: dict[str, list[Document]] = Field(default_factory=dict)
    graph_max_extra: int = Field(default=6, ge=0)
    chunks_per_neighbor: int = Field(default=1, ge=1)
    recall_cap: int = Field(
        default=24,
        ge=1,
        description="Max documents taken from base retriever before graph expansion.",
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = self.base_retriever.invoke(query)
        docs = docs[: self.recall_cap]
        return graph_expand(
            docs,
            self.adjacency,
            self.chunks_by_url,
            self.graph_max_extra,
            self.chunks_per_neighbor,
        )
