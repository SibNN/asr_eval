from concurrent.futures import ThreadPoolExecutor
import textwrap
from asr_eval.streaming.buffer import ID_TYPE

from .model import Signal, StreamingBlackBoxASR
from .transcription import PartialTranscription
from .sender import BaseStreamingAudioSender, StreamingAudioSender


def receive_full_transcription(
    asr: StreamingBlackBoxASR,
    id: ID_TYPE,
    sender: BaseStreamingAudioSender | None = None,
) -> list[PartialTranscription]:
    '''
    Blocks and waits until the full transcription (ended with Signal.FINISH) received for the given ID.
    If sender is provided, runs .start_sending() on it before.
    '''
    if sender:
        assert sender.id == id
        sender.start_sending(send_to=asr.input_buffer)
    results: list[PartialTranscription] = []
    while True:
        output_chunk, _id = asr.output_buffer.get(id=id)
        if output_chunk.data is Signal.FINISH:
            return results
        else:
            results.append(output_chunk.data)

def transribe_parallel(
    asr: StreamingBlackBoxASR,
    senders: list[StreamingAudioSender],
    n_threads: int,
) -> dict[ID_TYPE, list[PartialTranscription]]:
    '''
    Transcribes the senders in parallel, no more than `n_threads` senders simultaneously. Sender is
    considered to be transcribed when Signal.FINISH received for its ID.
    
    Call asr.start_thread() before calling this method, and asr.stop_thread() after.
    '''
    def process_sender(sender: StreamingAudioSender) -> tuple[ID_TYPE, list[PartialTranscription]]:
        print(f'Transcribing {sender.id}')
        transcription = receive_full_transcription(asr=asr, sender=sender, id=sender.id)
        print(f'Transcribed {sender.id}: {textwrap.shorten(PartialTranscription.join(transcription), width=80)}', flush=True)
        return sender.id, transcription
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        return dict(executor.map(process_sender, senders))