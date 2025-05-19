from __future__ import annotations

from dataclasses import dataclass, field
import uuid


@dataclass(kw_only=True)
class PartialTranscription:
    """
    A chunk returned by a streaming ASR model, may contain any text and any ID. If the model
    wants to edit the previous chunk, it can yield the same ID with another text, or refer to
    the last chunk with ID == 'latest'. Example:
    
    PartialTranscription.join([
        PartialTranscription(text='a'),
        PartialTranscription(id='latest', text='a2'),
        PartialTranscription(id=1, text='b'),
        PartialTranscription(id=2, text='c'),
        PartialTranscription(id=1, text='b2 b3'),
    ]) == 'a2 b2 b3 c'
    """
    id: int | str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    
    @classmethod
    def join(cls, transcriptions: list[PartialTranscription]) -> str:
        parts: dict[int | str, str] = {}
        
        latest_id = None
        for t in transcriptions:
            if t.id == 'latest':
                # edit the lastest chunk
                if latest_id is None:
                    # the very first chunk has "latest" ID, assiging an ID
                    latest_id = '<initial>'
                parts[latest_id] = t.text
            elif t.id in parts:
                # edit one of the previous chunks
                parts[t.id] = t.text
            else:
                # add a new chunk, set as latest
                parts[t.id] = t.text
                latest_id = t.id
        return ' '.join(parts.values())