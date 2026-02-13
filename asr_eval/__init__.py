import os
from pathlib import Path
from dotenv import load_dotenv

import appdirs

try:
    from datasets import Dataset # pyright: ignore[reportUnusedImport]
    has_datasets_package = True
except ImportError:
    has_datasets_package = False

def check_datasets_package():
    if not has_datasets_package:
        raise RuntimeError(
            'Package `datasets` missing'
            ', please install asr_eval[datasets].'
        )

try:
    import asr_eval_extras # type: ignore
    has_extras_package = True
except ImportError:
    has_extras_package = False
    

__all__ = [
    'ROOT_DIR',
    'CACHE_DIR',
    '__version__',
]


__version__ = "0.3.0"


# load env variables from .env file in same directory
# as python script or searches for it incrementally higher up.
load_dotenv()


ROOT_DIR = Path(__file__).parent
"""The root directory for the asr_eval package, where its
:code:`__init__.py` lives.

:meta hide-value:
"""


# Default value: ~/.cache/asr_eval/' on Linux
CACHE_DIR = Path(os.environ.get(
    'ASR_EVAL_CACHE', appdirs.user_cache_dir('asr_eval') # type: ignore
))
"""A cache dir for asr_eval.

Default ~/.cache/asr_eval/ on Linux. May be overridden by setting the
environmental variable :code:`ASR_EVAL_CACHE`.

:meta hide-value:
"""


_autodoc_type_aliases = {
    'FLOATS': ':type:`~asr_eval.utils.types.FLOATS`',
    'INTS': ':type:`~asr_eval.utils.types.INTS`',
    'VALUE': ':type:`~asr_eval.utils.storage.VALUE`',
    'WORD_ERROR_TYPE': 'asr_eval.align.alignment.WORD_ERROR_TYPE',
}