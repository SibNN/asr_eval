# Class CSVStorage (defined in asr_eval/utils/storage/csv_storage.py at lines 16-373)

class CSVStorage(asr_eval.utils.storage.base_storage.BaseStorage):
    """A csv-based :code:`BaseStorage` implementation.

    Note that :code:`BaseStorage` can use int/float/str/bool as key
    types.

    Warning:
        Gemini 3.0 LLM code!

    Note:
        While :code:`BaseStorage` interface is flexible and values
        can be of any pickleable type, CSV format is very limited
        and not typed. In this implementation, we try to serialize
        objects such as timed text segments into json and back, but this
        may cause unexpected behaviour or simply not work in some cases.
        Also note that row deletion/modification operation is extremely
        inefficient since it requires to rewrite the whole file.
        Finally, not that simultaneous modifications to the same file
        should not be done, or this may cause errors.
    """
    ...

    @typing.override
    def has_row(self, **keys: asr_eval.utils.storage.base_storage.VALUE) -> bool:
    ...

    @typing.override
    def add_row(self, value: typing.Any, overwrite: bool = True, **keys: asr_eval.utils.storage.base_storage.VALUE):
    ...

    @typing.override
    def get_row(self, **keys: asr_eval.utils.storage.base_storage.VALUE) -> typing.Any:
    ...

    @typing.override
    def delete_row(self, missing_ok: bool = False, **keys: asr_eval.utils.storage.base_storage.VALUE):
    ...

    @typing.override
    def list_all(
        self,
        load_values: bool = False,
        **keys: asr_eval.utils.storage.base_storage.VALUE,
    ) -> pl.DataFrame:
    ...

    @typing.override
    def iter_rows(
        self,
        load_values: bool = False,
        **keys: asr_eval.utils.storage.base_storage.VALUE,
    ) -> collections.abc.Iterator[dict[str, typing.Any]]:

    ...

    @typing.override
    def delete_all(self, **keys: asr_eval.utils.storage.base_storage.VALUE):
    ...

    @typing.override
    def close(self):
        # No open file handles are maintained persistently.
        ...