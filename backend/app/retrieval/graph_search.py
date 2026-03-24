from app.graph.store import GraphStore


async def graph_context_search(query: str, store: GraphStore) -> str:
    """Query the knowledge graph and return context as a formatted string."""
    if not store or not store.is_connected:
        return ""
    # Extract entity names from query (simple word extraction, skip stopwords)
    words = [w for w in query.split() if len(w) > 4]
    if not words:
        return ""
    subgraph = await store.get_entity_context(words[:5])
    if not subgraph.entities:
        return ""
    lines = ["## Knowledge Graph Context"]
    for e in subgraph.entities[:10]:
        lines.append(f"- {e.name} ({e.type}): {e.description}")
    for r in subgraph.relationships[:10]:
        lines.append(f"  → {r.source_id} {r.type} {r.target_id}")
    return "\n".join(lines)
