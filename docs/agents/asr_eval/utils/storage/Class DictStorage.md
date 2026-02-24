# Class DictStorage (defined in asr_eval/utils/storage/dict_storage.py at lines 13-88)

class DictStorage(asr_eval.utils.storage.base_storage.BaseStorage):
    """A dict-based in-memory :code:`BaseStorage` implementation."""
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
    ...