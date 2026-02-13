"""Tools for string alignment to evaluate speech recognition quality.

Some main utilities:

- :class:`~asr_eval.align.parsing.Parser` and
  :const:`~asr_eval.align.parsing.DEFAULT_PARSER` to parse
  predictions and annotations (possibly multivariant), which
  may include normalization, lowercase and diactritic conversion,
  splitting into words and punctuation removal.
- :class:`~asr_eval.align.transcription.Transcription` as a format for
  parsed transcription.
- :class:`~asr_eval.align.alignment.Alignment` to align predictions
  against multivariant annotations and calculate metrics.
- :func:`~asr_eval.align.solvers.dynprog.solve_optimal_alignment` as
  the underlying optimal alignment algorithm via dynamic
  programming.
- :class:`~asr_eval.align.metrics.DatasetMetric` and
  :func:`~asr_eval.align.metrics.bootstrap` to calculate average
  metrics with uncertainty via bootstrapping.
- :meth:`~asr_eval.align.alignment.Alignment.error_listing` to
  perform fine-grained error analysis.
- :class:`~asr_eval.align.alignment.MultipleAlignment` to visually
  compare multiple predictions for the same ground truth.
- :func:`~asr_eval.align.char_aligner.char_align` to align strings on a
  character level.
- :func:`~asr_eval.align.timings.fill_word_timings_inplace` to fill
  word timings via forced alignment using
  :class:`~asr_eval.models.base.interfaces.CTC` models.
"""