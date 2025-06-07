import pytest

from asr_eval.streaming.model import LATEST, TranscriptionChunk


def test_join_transcriptions():
    assert TranscriptionChunk.join([
        TranscriptionChunk(text='a'),
        TranscriptionChunk(id=LATEST, text='a2'),
        TranscriptionChunk(id=1, text='b'),
        TranscriptionChunk(id=2, text='c'),
        TranscriptionChunk(id=1, text='b2 b3'),
    ]) == 'a2 b2 b3 c'
    
    assert TranscriptionChunk.join([
        TranscriptionChunk(text='a'),
        TranscriptionChunk(id=1, text='b'),
        TranscriptionChunk(id=2, text='c'),
        TranscriptionChunk(text='x'),
        TranscriptionChunk(id=1, text='b2'),
        TranscriptionChunk(id=LATEST, text='!@<>'),
        TranscriptionChunk(id=1, text='b2 b3'),
        TranscriptionChunk(id=2, text='c2'),
    ]) == 'a b2 b3 c2 !@<>'
    
    assert TranscriptionChunk.join([
        TranscriptionChunk(id=LATEST, text='a'),
        TranscriptionChunk(id=LATEST, text='b'),
    ]) == 'b'
    
    
def test_final():
    with pytest.raises(AssertionError):
        TranscriptionChunk.join([
            TranscriptionChunk(id=2, text='a'),
            TranscriptionChunk(id=1, text='a', final=True),
            TranscriptionChunk(id=2, text='b'),
            TranscriptionChunk(id=1, text='a'),
        ])
    
    with pytest.raises(AssertionError):
        TranscriptionChunk.join([
            TranscriptionChunk(id=LATEST, text='a', final=True),
            TranscriptionChunk(id=LATEST, text='b'),
        ])