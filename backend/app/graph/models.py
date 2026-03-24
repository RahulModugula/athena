from pydantic import BaseModel


class Entity(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""
    properties: dict = {}


class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    description: str = ""


class Subgraph(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    context_text: str = ""
