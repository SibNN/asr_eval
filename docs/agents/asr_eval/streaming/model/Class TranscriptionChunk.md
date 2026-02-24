# Class TranscriptionChunk (defined in asr_eval/streaming/model.py at lines 642-703)

@dataclasses.dataclass(kw_only=True)
class TranscriptionChunk:
    """ A chunk returned by a
    :class:`~asr_eval.streaming.model.StreamingASR` model.

    Contain a text and optional ID. If we want to override the
    previously emitted partial transcription, we should emit a new chunk
    with the same ID. It will not be treated as a new text to append,
    but as a correction for the previous chunk with this ID.

    Note:
        A :code:`TranscriptionChunk` id is not the same concept as audio
        recording ID.

    Example:
        >>> from asr_eval.streaming.model import TranscriptionChunk
        >>> chunks = []
        >>> # append a new chunk without an explicit uid to refer
        >>> # without ID we cannot correct this chunk later
        >>> chunks.append(TranscriptionChunk(text='word1'))
        >>> # append a new chunk with id 1
        >>> chunks.append(TranscriptionChunk(uid=1, text='word2'))
        >>> # append a new chunk with id 2
        >>> chunks.append(TranscriptionChunk(uid=2, text='word3'))
        >>> print(TranscriptionChunk.join(chunks))
        word1 word2 word3
        >>> # correct a chunk with id 1
        >>> chunks.append(TranscriptionChunk(uid=1, text='word2a word2b'))
        >>> # remove a chunk with id 2
        >>> chunks.append(TranscriptionChunk(uid=2, text=''))
        >>> print(TranscriptionChunk.join(chunks))
        word1 word2a word2b
    """
    ...

    uid: int | str = field(default_factory=new_uid)

    text: str

    @classmethod
    def join(
        cls,
        transcriptions: (
            typing.Sequence[asr_eval.streaming.model.TranscriptionChunk]
            | typing.Sequence[asr_eval.streaming.model.OutputChunk | typing.Literal[Signal.FINISH]]
        ),
    ) -> str:
        """Join transcription chunks. If the :code:`transcriptions`
        are :code:`OutputChunk` instances, extracts a transcription
        chunks from each output chunk.

        See example in the
        :class:`~asr_eval.streaming.model.TranscriptionChunk` docs.
        """
        ...