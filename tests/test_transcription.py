from asr_eval.streaming.model import LATEST, TranscriptionChunk


def test_join_transcriptions():
    assert TranscriptionChunk.join([
        TranscriptionChunk(text='a'),
        TranscriptionChunk(ref=LATEST, text='a2'),
        TranscriptionChunk(ref=1, text='b'),
        TranscriptionChunk(ref=2, text='c'),
        TranscriptionChunk(ref=1, text='b2 b3'),
    ]) == 'a2 b2 b3 c'
    
    assert TranscriptionChunk.join([
        TranscriptionChunk(text='a'),
        TranscriptionChunk(ref=1, text='b'),
        TranscriptionChunk(ref=2, text='c'),
        TranscriptionChunk(text='x'),
        TranscriptionChunk(ref=1, text='b2'),
        TranscriptionChunk(ref=LATEST, text='!@<>'),
        TranscriptionChunk(ref=1, text='b2 b3'),
        TranscriptionChunk(ref=2, text='c2'),
    ]) == 'a b2 b3 c2 !@<>'
    
    assert TranscriptionChunk.join([
        TranscriptionChunk(ref=LATEST, text='a'),
        TranscriptionChunk(ref=LATEST, text='b'),
    ]) == 'b'