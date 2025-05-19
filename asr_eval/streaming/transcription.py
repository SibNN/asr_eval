from __future__ import annotations

from dataclasses import dataclass, field
import uuid


@dataclass(kw_only=True)
class PartialTranscription:
    """
    A chunk returned by a streaming ASR model, may contain any text and any ID. If the model
    wants to edit the previous chunk, it can yield the same ID with another text. Example:
    
    PartialTranscription.join([
        PartialTranscription(text='a'),
        PartialTranscription(id=1, text='b'),
        PartialTranscription(id=2, text='c'),
        PartialTranscription(id=1, text='b2'),
    ]) == 'a b2 c'
    """
    id: int = field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    
    @classmethod
    def join(cls, transcriptions: list[PartialTranscription]) -> str:
        parts: dict[int, str] = {}
        for t in transcriptions:
            parts[t.id] = t.text  # add or edit a text, keeps the order
        return ' '.join(parts.values())