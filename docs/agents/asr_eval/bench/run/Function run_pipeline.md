# Function run_pipeline (defined in asr_eval/bench/run.py at lines 25-151)

def run_pipeline(
    storage: asr_eval.utils.storage.BaseStorage,
    pipeline_name: str,
    dataset_specs: collections.abc.Sequence[str | asr_eval.bench.datasets.dataset_spec.DatasetSpec],
    print_transcriptions: bool = False,
    overwrite_existing: bool = False,
    suffix: str | None = None,
    keep: list[str] | None = None,
):
    '''
    Runs a pipeline on a list of datasets.

    Has also a CLI version, see
    :code:`python -m asr_eval.bench.run --help`

    See Also:
        More details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.

    Args:
        storage: Storage to save the results, such as
            :class:`~asr_eval.utils.storage.shelf_storage.ShelfStorage`.
        pipeline_name: Pipeline name to run.
        dataset_specs: List of dataset names, patterns or specs.
        print_transcriptions: Print transcriptions at runtime.
        overwrite_existing: Overwrite existing results, instead of
            skipping them.
        suffix: If not None, add the suffix to the pipeline name when
            saving to storage. Useful for versioning.
        keep: If not empty, keeps only the specified fields in the
            outputs of
            :class:`~asr_eval.bench.pipelines.TranscriberPipeline`.
            Can be used if the storage (e.g. .csv files) does not
            support data types for other fields.
    '''
    ...