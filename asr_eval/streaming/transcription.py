from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import uuid

from .base import Signal


LATEST = '__latest__'  # a special symbol to refer to the latest transcription chunk


@dataclass(kw_only=True)
class PartialTranscription:
    """
    A chunk returned by a streaming ASR model, may contain any text and any ID. If the model
    wants to edit the previous chunk, it can yield the same ID with another text, or refer to
    the last chunk with ID == LATEST. Example:
    
    PartialTranscription.join([
        PartialTranscription(text='a'),               # append a new chunk with text 'a' without an explicit id to refer
        PartialTranscription(id=LATEST, text='a2'),   # edit the latest chunk: 'a' -> 'a2'
        PartialTranscription(id=1, text='b'),         # append a new chunk with text 'b', 2 chunks in total: 'a', 'b'[id=1]
        PartialTranscription(id=2, text='c'),         # append a new chunk with text 'c', 3 chunks in total: 'a', 'b'[id=1], 'c'[id=2]
        PartialTranscription(id=1, text='b2 b3'),     # edit the chunk with id=1: 'a', 'b2 b3'[id=1], 'c'[id=2]
    ]) == 'a2 b2 b3 c'
    
    The argument final=True in PartialTranscription indicates that this chunk is final and
    will not be changed later.
    """
    id: int | str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    final: bool = False
    
    @classmethod
    def join(cls, transcriptions: list[PartialTranscription | Literal[Signal.FINISH]]) -> str:
        parts: dict[int | str, str] = {}
        final_ids: set[int | str] = set()
        
        for t in transcriptions:
            if t is Signal.FINISH:
                continue
            if t.id == LATEST:
                # edit the lastest chunk
                current_id = list(parts)[-1] if len(parts) else '<initial>'
            else:
                # edit one of the previous chunks or add a new chunk, set as latest
                current_id = t.id
            
            assert current_id not in final_ids, 'trying to rewrite chunk marked as final'
            if t.final:
                final_ids.add(current_id)
            
            parts[current_id] = t.text
            
        return ' '.join(parts.values())