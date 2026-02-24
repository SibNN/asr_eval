# Function average_wer (defined in asr_eval/align/metrics.py at lines 89-122)

def average_wer(
    samples: list[asr_eval.align.metrics.Metrics], mode: typing.Literal['plain', 'concat']
) -> float:
    """Averages WER value for a list of samples.

    Two alternative averaging methods are implemented:

    1. In "plain" method we calculate WER for each sample, clipping it
    from 0 to 1, and then average all the values.
    2. In "concat" method we sum up each counter (replacements,
    deletions, insertions and ground truth lengths) for all samples, and
    then calculate the WER value from the resulting counters. This is
    roughly equivalent to concatenating all the predictions and ground
    truth before calculating WER. Also, if ground truth length > 0 for
    all samples, this is equal to averaging WER (with
    :code:`clip=False`) for all samples, taking their ground truth
    lengths as averaging weights.

    Thus, in "concat" method long samples have larger effect on the
    overall metric, which is reasonable. The "plain" mode is also
    reasonable, because different samples may represent different
    conditions (acoustical, lexical etc.) and can be viewed as many
    different classes (or clusters), and "plain" mode is similar to
    macro-averaging metrics for these classes.
    """
    ...