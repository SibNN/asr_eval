# Function chunk_audio (defined in asr_eval/segments/chunking.py at lines 17-80)

def chunk_audio(
    length: float,
    segment_length: float,
    segment_shift: float,
    last_chunk_mode: typing.Literal['same_length', 'same_shift'] = 'same_length',
) -> list[asr_eval.segments.segment.AudioSegment]:
    """Chunks the audio uniformly.

    Args:
        length: A total audio length.
        segment_length: The desired length of each segment.
        segment_shift: The desired shift between conecutive segments.

    If :code:`length < segment_length`, returns a single chunk from 0 to
    length. Otherwise calculates how much chunks with the given
    :code:`segment_length` and :code:`segment_shift` fit into the
    :code:`length`. If the length does not accommodate an integer number
    of shifts, adds a single additional chunk:

    - If :code:`last_chunk_mode='same_length'`: from
      :code:`length - segment_length` to :code:`length`
    - If :code:`last_chunk_mode='same_shift'`: from
      :code:`<last_chunk_end> + segment_shift` to :code:`length`

    .. code-block:: none

        <---->  segment_shift
        <----------------------->  segment_length
        <--------------------------------------->  length
        =========================                |  
              ==========================         |  
                    ===========================  |  
                      ===========================|  # an additional chunk

    Example:
        >>> chunk_audio(length=41, segment_length=30, segment_shift=5) # doctest: +NORMALIZE_WHITESPACE
        [AudioSegment(start_time=0.0, end_time=30.0),
        AudioSegment(start_time=5.0, end_time=35.0),
        AudioSegment(start_time=10.0, end_time=40.0),
        AudioSegment(start_time=11, end_time=41)]
    """
    ...