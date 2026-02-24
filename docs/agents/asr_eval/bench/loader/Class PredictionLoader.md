# Class PredictionLoader (defined in asr_eval/bench/loader.py at lines 49-451)

class PredictionLoader:
    """Loads and aligns predictions saved with
    :func:`~asr_eval.bench.run.run_pipeline`.

    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.

    Args:
        storage: A storage where the predictions were saved, typically a
            :class:`~asr_eval.utils.storage.shelf_storage.ShelfStorage`.
        cache: A cache to store alignments and other data to cache, may
            be initially filled or empty. The cache is reusable.
        pipelines: A list of pipelines names or patterns to load. By
            default loads all pipelines.
        dataset_specs: A list of dataset names, patterns or specs to
            load. By default loads all datasets. In simple case just
            use dataset name, such as :code:`dataset_specs=['fleurs']`.
            For more complex case, see the example below.

    Dataset specs (specificators with semicolons) allow to specify
    augmentors, parsers or sample count to load, see
    :doc:`/guide_evaluation_dashboard` for details. Examples:

    Example:
        :code:`PredictionLoader(dataset_specs=["fleurs:n=100!"])` will
        search for the fleurs dataset in the storage. For every "key"
        consisting of (pipeline + augmentor + parser) it will try to
        load exactly 100 first samples of the fleurs dataset. Will drop
        keys that have not all of these samples. Will drop all other
        samples. This ensures that for all the "keys" exactly the same
        sample set is loaded, which allows to compare them on the same
        data.
    """
    ...

    grouped_loaded_predictions: dict[asr_eval.bench.loader.GroupKey, dict[int, asr_eval.bench.loader.SamplePrediction]]
    """A public attribute that exposes a mapping. The keys are
    combinations of dataset + pipeline + augmentor + parser. The values
    are mapping from sample id to a prediction that keeps the predicted
    text and the inference time.
    """

    def get_multiple_alignments(
        self,
        dataset_name: str,
        augmentor_name: str = 'none',
        parser_name: str = 'default',
        pipeline_patterns: collections.abc.Sequence[str] = ('*',),
    ) -> dict[int, asr_eval.align.alignment.MultipleAlignment]:
        """Compares multiple pipelines on a dataset.

        See Also:
            More details and examples in the user guide
            :doc:`/guide_evaluation_dashboard`.

        Given a list of :code:`pipeline_patterns`, searches for all keys
        in the
        :attr:`~asr_eval.bench.loader.PredictionLoader.grouped_loaded_predictions`
        that match the given pipeline, dataset, augmentor and parser.
        Since we can only compare pipelines with the same augmentor
        and parser, this provides all the results we have: pipelines,
        and their predictons on sample ids. Importantly, different
        pipelines may have different sets of sample ids: say, we run
        the first pipeline on 100 samples and the second pipeline
        only on 10 samples. Let we have pipelines P_1, ..., P_N and
        their sets of sample ids S_1, ..., S_N. The current function
        returns a dict where keys are union(S_1, ..., S_N), and for
        each sample id a
        :class:`~asr_eval.align.alignment.MultipleAlignment` is
        provided, with all pipelines that have this id predicted. In our
        example, the function returns a dict of all sample ids, and
        for 10 of them the :code:`MultipleAlignment` has 2 pipelines,
        while for the remaining 90 ids the :code:`MultipleAlignment` has
        only 1 pipeline. Further we can: 1) visualize all the
        alignments, 2) call :func:`~asr_eval.bench.evaluator.get_dataset_data`
        function that averages metrics.
        """
        ...

    def get_ordered_sample_ids(self, dataset_name: str) -> list[int]:
        """For a given registered dataset, returns a sequence of
        sample ids in the standard (shuffled) version, that can be
        obtained by :code:`get_dataset(dataset_name, shuffle=True)`.
        """
        ...

    def get_annotation(
        self,
        dataset_name: str,
        parser_name: str | typing.Literal['default'],
        sample_id: int
    ) -> asr_eval.align.transcription.Transcription:
        """Get a parsed annotation for the given dataset, parser name
        and sample id. If not in cache, retrieves the annotation by
        instantiating this dataset.
        """
        ...