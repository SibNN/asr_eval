# Function remap_time (defined in asr_eval/streaming/evaluation.py at lines 568-637)

def remap_time(
    cutoffs: list[asr_eval.streaming.sender.Cutoff],
    input_chunks: list[asr_eval.streaming.model.InputChunk],
    output_chunks: list[asr_eval.streaming.model.OutputChunk]
) -> tuple[list[asr_eval.streaming.model.InputChunk], list[asr_eval.streaming.model.OutputChunk]]:
    """Remapping is an optional mechanism that eliminates time spans
    where both the sender waits (due to its schedule) and the model
    waits (because it already processed the chunk and waits for the
    next). This makes evaluation faster than real time with the same
    results. Using remapping is meaningful when input chunks was sent
    with :code:`without_delays=True`.

    Technically, :code:`remap_time` adds artificial delays in some
    places, shifting put timestamps and get timestamps forward for both
    input and output chuks. More concretely, it iterates chunks from
    the first to the last and finds input chunks that were
    taken from the input buffer until they should be placed in the
    buffer according to the :code:`cutoffs` schedule. When such a
    situation is found, all the put and get timestamps starting from
    this time are shifted forwards by the calculated time delta.

    In the end, this allows to imitate a chunk history as it would have
    looked if :code:`without_delays=False` in senders.

    Note:
        This is not applicable (would work incorrectly) for
        :class:`~asr_eval.streaming.model.StreamingASR` that start
        another threads from its main beckground thread (where
        :attr:`~asr_eval.streaming.model.StreamingASR.is_multithreaded`
        is True).
    """
    ...