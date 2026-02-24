# Function deserialize_object (defined in asr_eval/utils/serializing.py at lines 155-198)

def deserialize_object(serialized: typing.Any, ignore_errors: bool = False) -> typing.Any:
    """Deserializes an object serialized with
    :func:`~asr_eval.utils.serializing.serialize_object`.

    If no :code:'_target_' fields found, returns the input data without
    changes.
    """
    ...