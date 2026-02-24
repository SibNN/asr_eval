# Function visualize_speaker_embeddings (defined in asr_eval/utils/deduplicate.py at lines 273-383)

def visualize_speaker_embeddings(
    splits: dict[str, Dataset] | DatasetDict,
    split_colors: dict[str, str] | None = None,
    max_samples_per_split: int | None = None,
    save_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> tuple[asr_eval.utils.types.FLOATS, asr_eval.utils.types.FLOATS]:
    """Performs speaker embedding analysis via UMAP projection into a 2D
    plot. Draws the plot and saves to the :code:`save_path`. Returns
    speaker embeddings, both original and after UMAP.

    Requires :code:`pip install torch umap-learn pyannote.audio`
    """
    ...