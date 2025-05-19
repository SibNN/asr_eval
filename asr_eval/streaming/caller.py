from collections import defaultdict
from asr_eval.streaming.model import RECORDING_ID_TYPE, Signal, StreamingBlackBoxASR
from asr_eval.streaming.transcription import PartialTranscription


def wait_for_transcribing(
    asr: StreamingBlackBoxASR,
    ids: list[RECORDING_ID_TYPE]
) -> dict[RECORDING_ID_TYPE, list[PartialTranscription]]:
    '''
    Blocks and listen for outputs from the StreamingBlackBoxASR until Signal.FINISH received
    for all the specified ids. Then returns all the received outputs.
    
    Note that in general the returned dict may contain other IDs not specified in `ids` argument
    if StreamingBlackBoxASR outputs these IDs.
    '''
    results: dict[RECORDING_ID_TYPE, list[PartialTranscription]] = defaultdict(list)
    finished: dict[RECORDING_ID_TYPE, bool] = {id: False for id in ids}
    
    while True:
        id, output = asr.output_buffer.get()
        if output is Signal.FINISH:
            finished[id] = True
            if all(finished.values()):
                return results
        else:
            results[id].append(output)