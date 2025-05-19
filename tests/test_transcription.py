from asr_eval.streaming.transcription import PartialTranscription


def test_join_transcriptions():
    assert PartialTranscription.join([
        PartialTranscription(text='a'),
        PartialTranscription(id=1, text='b'),
        PartialTranscription(id=2, text='c'),
        PartialTranscription(text='x'),
        PartialTranscription(id=1, text='b2'),
        PartialTranscription(text='xx'),
        PartialTranscription(id=1, text='b3'),
        PartialTranscription(id=2, text='c2'),
    ]) == 'a b3 c2 x xx'