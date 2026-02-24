# Class ShelfStorage (defined in asr_eval/utils/storage/dict_storage.py at lines 90-113)

class ShelfStorage(asr_eval.utils.storage.dict_storage.DictStorage):
    """ An implementation of
    :class:`~asr_eval.utils.storage.BaseStorage` based on Python's
    :code:`shelf`.

    With :code:`read_only=True` you can open the same file multiple times
    simultaneously.

    Note:
        Methods :meth:`~asr_eval.utils.storage.BaseStorage.list_all` or
        :meth:`~asr_eval.utils.storage.BaseStorage.delete_all` iterate
        over all the rows, which may be slow in this implementation.
    """
    ...

    @typing.override
    def close(self):
    ...