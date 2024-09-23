from dataclasses import dataclass

@dataclass
class EmbeddingRecord:
    id: int
    chat_id: str
    text: str
    embedding: list[float]
    success: bool
    error: str | None
    