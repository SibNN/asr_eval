from concurrent.futures import ThreadPoolExecutor
from textwrap import shorten
from asr_eval.streaming.buffer import ID_TYPE

from .model import OutputChunk, Signal, StreamingASR, TranscriptionChunk
from .sender import BaseStreamingAudioSender, StreamingAudioSender


def receive_full_transcription(
    asr: StreamingASR,
    id: ID_TYPE,
    sender: BaseStreamingAudioSender | None = None,
    send_all_without_delays: bool = False,
) -> list[OutputChunk]:
    '''
    Blocks and waits until the full transcription (ended with Signal.FINISH) received for the given ID.
    If sender is provided, runs .start_sending() on it before.
    '''
    if sender:
        print(f'Transcribing {sender.id}', flush=True)
        assert sender.id == id
        if send_all_without_delays:
            sender.send_all_without_delays(send_to=asr.input_buffer)
        else:
            sender.start_sending(send_to=asr.input_buffer)
    results: list[OutputChunk] = []
    while True:
        output_chunk, _id = asr.output_buffer.get(id=id)
        results.append(output_chunk)
        if output_chunk.data is Signal.FINISH:
            print(f'Transcribed {id}: {shorten(TranscriptionChunk.join(results), width=80)}', flush=True)
            return results


def transсribe_parallel(
    asr: StreamingASR,
    senders: list[StreamingAudioSender],
    n_threads: int,
    send_all_without_delays: bool = False,
) -> dict[ID_TYPE, list[OutputChunk]]:
    '''
    Transcribes the senders in parallel, no more than `n_threads` senders simultaneously. Sender is
    considered to be transcribed when Signal.FINISH received for its ID.
    
    Call asr.start_thread() before calling this method, and asr.stop_thread() after.
    '''
    def process_sender(sender: StreamingAudioSender) -> tuple[ID_TYPE, list[OutputChunk]]:
        chunks = receive_full_transcription(
            asr=asr,
            sender=sender,
            id=sender.id,
            send_all_without_delays=send_all_without_delays,
        )
        return sender.id, chunks
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        return dict(executor.map(process_sender, senders))