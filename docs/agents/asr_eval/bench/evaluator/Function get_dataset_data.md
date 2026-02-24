# Function get_dataset_data (defined in asr_eval/bench/evaluator.py at lines 280-416)

def get_dataset_data(
    multiple_alignments: dict[int, asr_eval.align.alignment.MultipleAlignment],
    count_absorbed_insertions: bool = True,
    max_consecutive_insertions: int | None = None,
    wer_averaging_mode: typing.Literal['plain', 'concat'] = 'concat',
    exclude_samples_with_digits: bool = False,
    max_samples_to_render: int | None = None,
) -> asr_eval.bench.evaluator.DatasetData:
    """Takes raw multiple alignments (usually from
    :meth`~asr_eval.bench.loader.PredictionLoader.get_multiple_alignments`)
    and 1) renders multiple alignments in a displayable form, 2) averages
    metrics across all samples.

    Acts as a main utility for the ASR dashboard data model.

    See Also:
        More details and examples in the user guide
        :doc:`/guide_alignment_wer`.

    Args:
        multiple_alignments: multiple alignments for several sample ids
            in some dataset. All the multiple alignments should NOT
            necessary contain the same set of pipelines.
        count_absorbed_insertions: a parameter for
            :meth:`~asr_eval.align.alignment.Alignment.error_listing`
            when calculating metrics.
        max_consecutive_insertions: a parameter for
            :meth:`~asr_eval.align.alignment.Alignment.error_listing`
            when calculating metrics.
        wer_averaging_mode: a parameter for
            :meth:`~asr_eval.align.metrics.DatasetMetric.from_samples`
            when averagint metrics.
        exclude_samples_with_digits: if True, when averagint metrics,
            excludes all samples where a digit is found either in the
            ground truth transcription, or in some of the pipeline
            predictions. This acts as a "poor man's solution" to avoid
            issues with normalization of numericals.
        max_samples_to_render: if not None, don't render multiple
            alignments for all samples except the specified number of
            samples.

    Returns:
        See the :class:`~asr_eval.bench.evaluator.DatasetData` docs.
    """
    ...