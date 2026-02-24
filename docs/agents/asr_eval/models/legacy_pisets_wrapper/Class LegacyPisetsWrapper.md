# Class LegacyPisetsWrapper (defined in asr_eval/models/legacy_pisets_wrapper.py at lines 15-74)

class LegacyPisetsWrapper(asr_eval.models.base.interfaces.TimedTranscriber):
    '''
    A Pisets transcriber from https://github.com/bond005/pisets

    Commit hash e095ae626bbd18bb4490b9745d0acc34006c4eb8

    Requires a manual cloning into the `repo_dir` before instantiating.
    '''
    ...

    @typing.override
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
    ...