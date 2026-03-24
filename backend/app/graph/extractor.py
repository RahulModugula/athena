import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.models import Entity, Relationship

logger = structlog.get_logger()

EXTRACT_SYSTEM = """You are a knowledge graph extraction assistant. Extract named entities and relationships from the provided text.

Return ONLY valid JSON with this exact schema:
{
  "entities": [
    {"id": "unique_slug", "name": "Entity Name", "type": "PERSON|ORG|CONCEPT|PLACE|EVENT|TECHNOLOGY|OTHER", "description": "brief description"}
  ],
  "relationships": [
    {"source_id": "slug1", "target_id": "slug2", "type": "RELATION_TYPE", "description": "brief description"}
  ]
}

Rules:
- id must be a lowercase underscore slug derived from the name
- Extract 3-10 entities and up to 10 relationships
- Only include relationships between entities you extracted
- Return valid JSON only, no markdown or explanation"""


async def extract_entities(text: str, llm) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from text using an LLM."""
    try:
        json_llm = llm.bind(response_format={"type": "json_object"})
        messages = [
            SystemMessage(content=EXTRACT_SYSTEM),
            HumanMessage(content=f"Extract entities and relationships from:\n\n{text[:3000]}"),
        ]
        response = await json_llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        data = json.loads(content)

        entities = [Entity(**e) for e in data.get("entities", [])]
        relationships = [Relationship(**r) for r in data.get("relationships", [])]
        logger.info("entities extracted", entities=len(entities), relationships=len(relationships))
        return entities, relationships
    except Exception as exc:
        logger.warning("entity extraction failed", error=str(exc))
        return [], []
