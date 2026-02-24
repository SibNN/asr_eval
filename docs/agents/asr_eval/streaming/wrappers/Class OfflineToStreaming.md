# Class OfflineToStreaming (defined in asr_eval/streaming/wrappers.py at lines 69-137)

class OfflineToStreaming(asr_eval.streaming.model.StreamingASR):
    """Converts non-streaming (offline) ASR model into a streaming one.
    Calls the offline model with the given :code:`interval` (at the
    audio timescale).

    For example, let the audio be 3 seconds long, and
    :code:`interval=1`. Will call the offline model:

    1. On the waveform slice from 0 to 1 second (when enough data
       received)
    2. On the waveform slice from 0 to 2 seconds (when enough data
       received)
    3. On the waveform slice from 0 to 3 seconds (when enough data
       received)

    Each time completely overwrites the old transcription with the new
    one (this is achieved by sending a new
    :class:`~asr_eval.streaming.model.TranscriptionChunk` with the same
    id).

    TODO support longform audios somehow (with or without VAD).

    TODO support batching?

    TODO set also a real-time minimal interval between model calls.

    TODO add :code:`keep=True` arg to :code:`.get_with_rechunking()`
    instead making another buffer.
    """
    ...

    @property
    @typing.override
    def audio_type(self) -> typing.Literal['float']:
    ...