# Class PyannoteDiarizationWrapper (defined in asr_eval/models/pyannote_diarization.py at lines 22-168)

class PyannoteDiarizationWrapper:
    """A wrapper for Pyannote diarization.

    Requires :code:`pyannote>=4.0.0`. To use, you first need to accept
    conditions here
    https://huggingface.co/pyannote/speaker-diarization-community-1 ,
    then specify your HF_TOKEN in the environmental variable.
    """
    ...

    pipeline: SpeakerDiarization

    model_name: str