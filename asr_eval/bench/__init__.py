"""Tools for reproducible ASR evaluation and dashboard visualizations.

The package includes a registry for several types of components:

- **Datasets** for ASR evaluation, in Hugging Face
  `Dataset <https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset>`__
  format:
  
    - **Storage**:
      :data:`~asr_eval.bench.datasets.datasets_registry`
    - **Registering**:
      :deco:`~asr_eval.bench.datasets.register_dataset`
    - **Retrieving**:
      :func:`~asr_eval.bench.datasets.get_dataset`
      
- **Pipelines** for offline and streaming ASR in a unified
  format:
  
    - **Storage**:
      :data:`~asr_eval.bench.pipelines.pipelines_registry`
    - **Registering**:
      :class:`~asr_eval.bench.pipelines.TranscriberPipeline`
    - **Retrieving**:
      :func:`~asr_eval.bench.pipelines.get_pipeline`
      
- **Parsers** to define text tokenization and normalization scheme:

    - **Storage**:
      :data:`~asr_eval.bench.parsers.parsers_registry`
    - **Registering**:
      :func:`~asr_eval.bench.parsers.register_parsers`
    - **Retrieving**:
      :func:`~asr_eval.bench.parsers.get_parser`
      
- **Augmentors** to define audio processing methods:

    - **Storage**:
      :data:`~asr_eval.bench.augmentors.augmentors_registry`
    - **Registering**:
      :class:`~asr_eval.bench.augmentors.AudioAugmentor`
    - **Retrieving**:
      :func:`~asr_eval.bench.augmentors.get_augmentor`

The package offers several command line tools:

- :mod:`python -m asr_eval.bench.check <asr_eval.bench.check>` to run a
  quick check that a pipeline works.
- :mod:`python -m asr_eval.bench.run <asr_eval.bench.run>` to run
  pipelines on datasets, supports incremental runs, allows to specify
  augmentors and sample counts, and store predictions, as well as input
  and output chunk history for streaming pipelines.
- :mod:`python -m asr_eval.bench.dashboard.run <asr_eval.bench.dashboard.run>`
  to load the results and run an interactive dashboard that displays
  metrics and results on individual samples, and enables fune-grained
  comparison.
- :mod:`python -m asr_eval.bench.streaming.make_plots <asr_eval.bench.streaming.make_plots>`
  to analyze input and output chunk history, make various diagrams
  for streaming pipelines and save the into a folder.
- :mod:`python -m asr_eval.bench.streaming.show_plots <asr_eval.bench.streaming.show_plots>`
  to show the obtained streaming diagrams vis web interface.

Internally these tools use an abstract class
:class:`~asr_eval.utils.storage.BaseStorage` to store the results. It
has a concrete
implementation :class:`~asr_eval.utils.storage.ShelfStorage` (but it is
possible to quickly adapt to a new storage type such as SQLite or
wandb). The :class:`~asr_eval.bench.loader.PredictionLoader` loads the
saved predictions and performs string alignment, saving the alignments
in cache. The helper functions
:func:`~asr_eval.bench.evaluator.get_dataset_data` and
:func:`~asr_eval.bench.evaluator.compare_pipelines` calculate metrics
with bootstrap confidence and run a fine-grained comparison.
"""