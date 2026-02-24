# Function load_from_json (defined in asr_eval/utils/serializing.py at lines 70-83)

def load_from_json(path: str | pathlib.Path) -> typing.Any:
    """Loads a data structure that was saved with
    :func:`~asr_eval.utils.serializing.save_to_json`. If the .json file
    does not contain any `_target_` fields, will act equally to
    :code:`json.loads(path.read_text())`.
    """
    ...