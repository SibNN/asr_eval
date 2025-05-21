import pytest

from asr_eval.streaming.model import LATEST, PartialTranscription


def test_join_transcriptions():
    assert PartialTranscription.join([
        PartialTranscription(text='a'),
        PartialTranscription(id=LATEST, text='a2'),
        PartialTranscription(id=1, text='b'),
        PartialTranscription(id=2, text='c'),
        PartialTranscription(id=1, text='b2 b3'),
    ]) == 'a2 b2 b3 c'
    
    assert PartialTranscription.join([
        PartialTranscription(text='a'),
        PartialTranscription(id=1, text='b'),
        PartialTranscription(id=2, text='c'),
        PartialTranscription(text='x'),
        PartialTranscription(id=1, text='b2'),
        PartialTranscription(id=LATEST, text='!@<>'),
        PartialTranscription(id=1, text='b2 b3'),
        PartialTranscription(id=2, text='c2'),
    ]) == 'a b2 b3 c2 !@<>'
    
    assert PartialTranscription.join([
        PartialTranscription(id=LATEST, text='a'),
        PartialTranscription(id=LATEST, text='b'),
    ]) == 'b'
    
    
def test_final():
    with pytest.raises(AssertionError):
        PartialTranscription.join([
            PartialTranscription(id=2, text='a'),
            PartialTranscription(id=1, text='a', final=True),
            PartialTranscription(id=2, text='b'),
            PartialTranscription(id=1, text='a'),
        ])
    
    with pytest.raises(AssertionError):
        PartialTranscription.join([
            PartialTranscription(id=LATEST, text='a', final=True),
            PartialTranscription(id=LATEST, text='b'),
        ])