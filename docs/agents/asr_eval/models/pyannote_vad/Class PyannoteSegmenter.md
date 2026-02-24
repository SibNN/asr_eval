# Class PyannoteSegmenter (defined in asr_eval/models/pyannote_vad.py at lines 22-67)

class PyannoteSegmenter(asr_eval.models.base.interfaces.Segmenter):
    '''VAD-based audio segmenter based on Pyannote. With default params
    is equivalent to :code:`gigaam.vad_utils.segment_audio`.

    Requires :code:`pyannote>=4.0.0`. Based on
    https://github.com/salute-developers/GigaAM/blob/main/gigaam/vad_utils.py .
    This segmenter does NOT require gigaam package to be installed,
    because all the required functions are copied from the gigaam
    package. The model is cached in PYANNOTE_CACHE dir, by default:
    ~/.cache/torch/pyannote.
    '''
    ...