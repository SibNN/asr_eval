Research framework
###############################

An experimental framework :code:`asr_eval.bench` is designed to store experimental results
to analyze, reuse and reproduce them. The bench keeps of a list of named datasets, consisting of
samples, and a list of named pipelines.

Basically, it works as follows: we apply a pipeline to a dataset sample and store the
predictions in a separate file. Another script loads all the available predictions and
calculate metrics and/or visualize the predictions.

Organizing datasets
=======================

Each speech recognition test set has a unique name and a loading function without arguments
that returns a list of samples. Each sample contain an :code:`'audio'` field with sampling
rate 16 000 and a :code:`'transcription'` field - a ground truth transcription, possibly
multivariant. Usually a loading function takes a dataset from HuggingFace and shuffles it
with seed 0 before returning. Shuffling allows to get the first N samples to obtain
a representative set (often HF datasets are not shuffled). Example:

.. code-block:: python

    from typing import cast
    from asr_eval.bench.datasets import AudioSample, get_dataset

    dataset = get_dataset('podlodka-full')()
    sample = cast(AudioSample, dataset[0])

Organizing pipelines
=======================

A pipeline is a runnable that processes a dataset sample and writes any files into
:code:`{pipeline}/{dataset}/{sample_idx}` directory. A pipeline consists of a unique name,
instantiattion code and code to run on a single sample. Usually a pipeline writes a file
:code:`transcription.json`. This json contains either a transcription or a timed transcription,
saved by :code:`save_to_json` utility.

However, in general a pipeline may output arbitrary files, and pipelines may use outputs of
another pipelines. This is useful for milti-stage processing, when we can reuse the outputs
of each stage.

Pipelines are never parametrized. Parametrization would introduce many complexities with
1) tracking if the results are already calculated or not, 2) specifying the same parameters
to reproduce the results, 3) adding hyperparameters that were previously hardcoded.

Thus that if you want to try multiple hyperparameter values, you need to create multiple
pipelines. Example:

.. code-block:: python

    from asr_eval.bench.pipelines import TranscriberPipeline, get_pipeline
    from asr_eval.models.whisper_wrapper import WhisperLongformWrapper

    class _(TranscriberPipeline, register_as='whisper-temp=0'):
        def init(self):
            return WhisperLongformWrapper(temperature=0.0)

    class _(TranscriberPipeline, register_as='whisper-temp=0.5'):
        def init(self):
            return WhisperLongformWrapper(temperature=0.5)

    pipeline = get_pipeline('whisper-temp=0.5')()

Running pipelines
=======================

Using code:

.. code-block:: python

    from asr_eval.bench.run import run_pipeline

    run_pipeline(
        pipeline_name='whisper-tiny',
        dataset_names=['resd', 'podlodka'],
        root_dir='outputs',
        max_samples=100,
    )

Or, equally, using command line interface:

.. code-block:: bash

    python -m asr_eval.bench.run -p whisper-tiny -d resd podlodka -r outputs -m 100

Evaluator
=======================

An evaluator is a component that searches for :code:`*/*/*/transcription.json` files, loads them,
determines if this is timed transcription or not and aligns with ground truth for the corresponding
dataset sample. Since aligning requires some time, caches the alignment results into :code:`alignment.pkl`
near the :code:`transcription.json`. Evaluator stores all the results in a dataframe.

Dashboard
=======================

A dashboard is a script that runs evaluator on the output dir and starts a web server to visualize
the alignments and metrics. When calculating metrics, the dashboard  takes into account that different
sample sets could be processed for different pipelines (for example, pipeline A processed the whole dataset
and pipeline B processed only a half of the dataset), and compares the pipelines carefully.

.. code-block:: bash

    python -m asr_eval.bench.dashboard