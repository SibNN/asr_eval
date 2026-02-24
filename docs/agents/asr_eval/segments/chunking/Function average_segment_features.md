# Function average_segment_features (defined in asr_eval/segments/chunking.py at lines 222-285)

def average_segment_features(
    segments: list[asr_eval.segments.segment.AudioSegment],
    features: list[asr_eval.utils.types.FLOATS] | list[asr_eval.utils.types.INTS],
    feature_tick_size: float,
    averaging_weights: typing.Literal['beta', 'uniform'] = 'beta',
) -> asr_eval.utils.types.FLOATS:
    """Given audio features calculated on the given audio chunking,
    averages them. The chunks (:code:`segments`) may overlap.

    Args:
        segments: A list of segments. Typically obtained by a uniform
            chunking using
            :func:`~asr_eval.segments.chunking.chunk_audio`, but may
            also be non-uniform.
        features: 2D feature array for each segment.
        feature_tick_size: A time interval between consecutive positions
            in :code:`features`.
        averaging_weights: May be "uniform" (flat) or "beta" (decaying
            at time edges of each feature array in :code:`features`). Is
            used to weight features.
    """
    ...