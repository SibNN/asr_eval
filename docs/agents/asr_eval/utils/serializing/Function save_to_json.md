# Function save_to_json (defined in asr_eval/utils/serializing.py at lines 42-68)

def save_to_json(obj: typing.Any, path: str | pathlib.Path, indent: int = 4):
    """Serializes an hierarchical structure of dataclasses/lists/dicts
    to a json-compatible dict and then saves to a .json file. Can be
    loaded back with :func:`~asr_eval.utils.serializing.load_from_json`.

    If an exception or keyboard interrupt happens during saving, the
    file will not br created.
    """
    ...