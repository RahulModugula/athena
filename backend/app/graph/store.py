from typing import Any

import structlog

from app.graph.models import Entity, Relationship, Subgraph

logger = structlog.get_logger()


class GraphStore:
    def __init__(self, uri: str, auth: tuple[str, str] | None = None) -> None:
        self._uri = uri
        self._auth = auth
        self._driver: Any = None

    async def connect(self) -> None:
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(self._uri, auth=self._auth)
            await self._driver.verify_connectivity()
            logger.info("neo4j connected", uri=self._uri)
        except Exception as exc:
            logger.warning("neo4j connection failed", error=str(exc))
            self._driver = None

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    async def upsert_entities(self, entities: list[Entity]) -> None:
        if self._driver is None:
            return
        try:
            async with self._driver.session() as session:
                for entity in entities:
                    await session.run(
                        """
                        MERGE (e:Entity {id: $id})
                        SET e.name = $name,
                            e.type = $type,
                            e.description = $description,
                            e.properties = $properties
                        """,
                        id=entity.id,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        properties=str(entity.properties),
                    )
        except Exception as exc:
            logger.warning("upsert_entities failed", error=str(exc))

    async def upsert_relationships(self, rels: list[Relationship]) -> None:
        if self._driver is None:
            return
        try:
            async with self._driver.session() as session:
                for rel in rels:
                    await session.run(
                        """
                        MATCH (s:Entity {id: $source_id})
                        MATCH (t:Entity {id: $target_id})
                        MERGE (s)-[r:RELATES {type: $type}]->(t)
                        SET r.description = $description
                        """,
                        source_id=rel.source_id,
                        target_id=rel.target_id,
                        type=rel.type,
                        description=rel.description,
                    )
        except Exception as exc:
            logger.warning("upsert_relationships failed", error=str(exc))

    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        if self._driver is None:
            return []
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name =~ $pattern
                    RETURN e
                    LIMIT $limit
                    """,
                    pattern=f"(?i).*{query}.*",
                    limit=limit,
                )
                records = await result.data()
                return [
                    Entity(
                        id=r["e"]["id"],
                        name=r["e"]["name"],
                        type=r["e"]["type"],
                        description=r["e"].get("description", ""),
                    )
                    for r in records
                ]
        except Exception as exc:
            logger.warning("search_entities failed", error=str(exc))
            return []

    async def get_entity_context(self, entity_names: list[str]) -> Subgraph:
        if self._driver is None:
            return Subgraph()
        try:
            async with self._driver.session() as session:
                patterns = [f"(?i).*{name}.*" for name in entity_names]
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE any(p IN $patterns WHERE e.name =~ p)
                    OPTIONAL MATCH (e)-[r:RELATES]-(neighbor:Entity)
                    RETURN e, r, neighbor
                    """,
                    patterns=patterns,
                )
                records = await result.data()

                seen_entities: dict[str, Entity] = {}
                relationships: list[Relationship] = []

                for rec in records:
                    e_data = rec.get("e")
                    if e_data and e_data["id"] not in seen_entities:
                        seen_entities[e_data["id"]] = Entity(
                            id=e_data["id"],
                            name=e_data["name"],
                            type=e_data["type"],
                            description=e_data.get("description", ""),
                        )
                    neighbor = rec.get("neighbor")
                    rel = rec.get("r")
                    if neighbor and neighbor["id"] not in seen_entities:
                        seen_entities[neighbor["id"]] = Entity(
                            id=neighbor["id"],
                            name=neighbor["name"],
                            type=neighbor["type"],
                            description=neighbor.get("description", ""),
                        )
                    if rel and e_data and neighbor:
                        relationships.append(
                            Relationship(
                                source_id=e_data["id"],
                                target_id=neighbor["id"],
                                type=rel.get("type", "RELATES"),
                                description=rel.get("description", ""),
                            )
                        )

                return Subgraph(
                    entities=list(seen_entities.values()),
                    relationships=relationships,
                )
        except Exception as exc:
            logger.warning("get_entity_context failed", error=str(exc))
            return Subgraph()
