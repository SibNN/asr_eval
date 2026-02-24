# Function serialize_object (defined in asr_eval/utils/serializing.py at lines 96-153)

def serialize_object(obj: typing.Any) -> typing.Any:
    """ Serializes an hierarchical structure of dataclasses, lists,
    dicts or enums into a json-compatible dict.

    This includes converting dataclasses into dicts (omitting fields
    where the value is None and the default value is also None). The
    class full name is written to the additional :code:`_target_`
    field to construct the object back with
    :func:`~asr_eval.utils.serializing.deserialize_object`.

    Besides dataclasses, can serialize
    :func:`~asr_eval.utils.serializing.SerializableToDict` objects. This
    is useful for custom classes that are not dataclasses, but we want
    to be able to save (to json or yaml) and load them.
    """
    ...