# Class SerializableToDict (defined in asr_eval/utils/serializing.py at lines 25-40)

class SerializableToDict(abc.ABC):
    """An interface to to serialize an object into a json-compatibl
    dict with
    :func:`~asr_eval.utils.serializing.serialize_object` and load back
    with :func:`~asr_eval.utils.serializing.deserialize_object`.

    Is not needed for dataclasses, only for objects with custom
    (de)serialization logic.
    """
    ...

    @abc.abstractmethod
    def serialize_to_dict(self) -> dict[str, typing.Any]:
        """Returns a dict to write into json. The resulting dict can be
        passed back into the class constructor to restore an equal
        object.
        """
        ...