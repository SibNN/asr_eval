The bench is designed for storing all the results to analyze and reuse them.

The bench keeps of a list of named datasets, consisting of samples, and a list of named
pipelines.

A **pipeline** is a runnable that processes a sample and writes any files into
`{pipeline}/{dataset}/{sample_idx}` directory. A pipeline consists of:

1. a unique name
2. instantiate method
3. call on dataset sample method

Usually a pipeline writes a file `{pipeline}/{dataset}/{sample_idx}/transcription.json`.
This json contains either a transcription or a timed transcription, saved by `save_to_json`
utility. However, a pipeline may output any information, and pipelines may use outputs of
another pipelines. This is useful for milti-stage processing, when each stage is a pipeline,
and thus you can save and load the results of each stage.

A separate **eval-server** script searches for `*/*/*/transcription.json` files, loads them,
determines if this is timed transcription or not, aligns with ground truth for the
corresponding sample and saves the results as `alignment.pkl` near the `transcription.json`.
If this file already exists, loads it instead. Then the script displays and/or saves averaged
WER metric for each pipeline and dataset.

The eval-server script also takes into account that different sample sets could be processed
for different pipelines, and compares the pipelines carefully.