"""Module-level refs set during lifespan for MCP tool access."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.store import GraphStore

graph_store: GraphStore | None = None
