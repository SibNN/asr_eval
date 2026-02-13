from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ThreadPoolExecutor

from asr_eval.streaming.buffer import ID_TYPE
from asr_eval.streaming.model import OutputChunk, Signal, StreamingASR
from asr_eval.streaming.sender import StreamingSender, get_uniform_cutoffs
from asr_eval.utils.misc import new_uid
from asr_eval.utils.types import FLOATS


__all__ = [
    'receive_transcription',
    'transcribe_parallel',
]


def receive_transcription(
    asr: StreamingASR, id: ID_TYPE
) -> Iterable[OutputChunk]:
    """Blocks and waits until the full transcription (ended with
    :code:`Signal.FINISH`) received for the given ID.
    """

    while True:
        output_chunk, _id = asr.output_buffer.get(id=id)
        yield output_chunk
        if output_chunk.data is Signal.FINISH:
            return


def transcribe_parallel(
    asr: StreamingASR,
    waveforms: list[FLOATS],
    n_threads: int,
    send_all_without_delays: bool = False,
    real_time_interval_sec: float = 1 / 25,
    speed_multiplier: float = 1,
) -> dict[ID_TYPE, list[OutputChunk]]:
    """Transcribes the waveforms in parallel, but with no more than
    :code:`n_threads` simultaneous senders.
    
    Call :code:`asr.start_thread()` before calling this method, and
    :code:`asr.stop_thread()` after.
    """
    
    assert asr.is_thread_started()
    def process_sender(
        waveform: FLOATS
    ) -> tuple[ID_TYPE, list[OutputChunk]]:
        cutoffs = get_uniform_cutoffs(
            waveform=waveform,
            real_time_interval_sec=real_time_interval_sec,
            speed_multiplier=speed_multiplier,
        )
        sender = StreamingSender(
            cutoffs=cutoffs,
            waveform=waveform,
            asr=asr,
            id=new_uid(),
        )
        sender.start_sending(
            without_delays=send_all_without_delays,
        )
        chunks = list(receive_transcription(asr=asr, id=sender.id))
        return sender.id, chunks
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        return dict(executor.map(process_sender, waveforms))