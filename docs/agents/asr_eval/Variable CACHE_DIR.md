# Variable CACHE_DIR (defined in asr_eval/__init__.py at lines 51-61)

CACHE_DIR = Path(os.environ.get(
    'ASR_EVAL_CACHE', appdirs.user_cache_dir('asr_eval') # type: ignore
))
"""A cache dir for asr_eval.

Default ~/.cache/asr_eval/ on Linux. May be overridden by setting the
environmental variable :code:`ASR_EVAL_CACHE`.

:meta hide-value:
"""