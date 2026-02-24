# Class OutputBeam (defined in asr_eval/ctc/lm.py at lines 25-40)

@dataclasses.dataclass
class OutputBeam:
    """ Outputs of BeamSearchDecoderCTC.decode_beams as dataclass.

    Is needed for 0.5.0 version, but not for the last version from repo.
    In asr_eval we use the 0.5.0 version.
    """
    ...

    text: str

    last_lm_state: kenlm.State | list[kenlm.State] # type: ignore

    text_frames: list[WordFrames]

    logit_score: float

    lm_score: float