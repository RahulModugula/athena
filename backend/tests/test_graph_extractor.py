"""Tests for graph entity extractor with mocked LLM."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph.extractor import extract_entities
from app.graph.models import Entity, Relationship


def make_mock_llm(response_content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = response_content

    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(return_value=mock_response)

    mock_llm = MagicMock()
    mock_llm.bind = MagicMock(return_value=mock_bound)
    return mock_llm


@pytest.mark.asyncio
async def test_extract_entities_returns_entities_and_relationships() -> None:
    payload = {
        "entities": [
            {"id": "alan_turing", "name": "Alan Turing", "type": "PERSON", "description": "mathematician"},
            {"id": "enigma", "name": "Enigma", "type": "TECHNOLOGY", "description": "cipher machine"},
        ],
        "relationships": [
            {"source_id": "alan_turing", "target_id": "enigma", "type": "DECRYPTED", "description": "broke the code"},
        ],
    }
    llm = make_mock_llm(json.dumps(payload))
    entities, relationships = await extract_entities("Alan Turing broke Enigma.", llm)

    assert len(entities) == 2
    assert isinstance(entities[0], Entity)
    assert entities[0].id == "alan_turing"
    assert entities[1].type == "TECHNOLOGY"

    assert len(relationships) == 1
    assert isinstance(relationships[0], Relationship)
    assert relationships[0].type == "DECRYPTED"


@pytest.mark.asyncio
async def test_extract_entities_empty_on_invalid_json() -> None:
    llm = make_mock_llm("not valid json at all")
    entities, relationships = await extract_entities("some text", llm)
    assert entities == []
    assert relationships == []


@pytest.mark.asyncio
async def test_extract_entities_empty_on_llm_error() -> None:
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    mock_llm = MagicMock()
    mock_llm.bind = MagicMock(return_value=mock_bound)

    entities, relationships = await extract_entities("some text", mock_llm)
    assert entities == []
    assert relationships == []


@pytest.mark.asyncio
async def test_extract_entities_empty_lists_in_response() -> None:
    llm = make_mock_llm(json.dumps({"entities": [], "relationships": []}))
    entities, relationships = await extract_entities("nothing to extract", llm)
    assert entities == []
    assert relationships == []
