from asr_eval.streaming.transcription import PartialTranscription


def test_join_transcriptions():
    assert PartialTranscription.join([
        PartialTranscription(text='a'),
        PartialTranscription(id='latest', text='a2'),
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
        PartialTranscription(id='latest', text='!@<>'),
        PartialTranscription(id=1, text='b2 b3'),
        PartialTranscription(id=2, text='c2'),
    ]) == 'a b2 b3 c2 !@<>'
    
    assert PartialTranscription.join([
        PartialTranscription(id='latest', text='a'),
        PartialTranscription(id='latest', text='b'),
    ]) == 'b'