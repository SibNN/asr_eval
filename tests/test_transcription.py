from asr_eval.streaming.model import TranscriptionChunk


def test_join_transcriptions():
    assert TranscriptionChunk.join([
        TranscriptionChunk(text='a'),
        TranscriptionChunk(uid=1, text='b'),
        TranscriptionChunk(uid=2, text='c'),
        TranscriptionChunk(uid=1, text='b2 b3'),
    ]) == 'a b2 b3 c'
    
    assert TranscriptionChunk.join([
        TranscriptionChunk(text='a'),
        TranscriptionChunk(uid=1, text='b'),
        TranscriptionChunk(uid=2, text='c'),
        TranscriptionChunk(text='x'),
        TranscriptionChunk(uid=1, text='b2'),
        TranscriptionChunk(uid=1, text='b2 b3'),
        TranscriptionChunk(uid=2, text='c2'),
    ]) == 'a b2 b3 c2 x'