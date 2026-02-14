import argparse
from itertools import islice
import re
from typing import Any

from asr_eval.utils.storage import make_storage


description = """A command line tool to show several lines from the
given disk storage (see :func:`~asr_eval.utils.storage.ShelfStorage`).
"""


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-s',
    '--storage',
    required=True,
    help='Path of the storage file to probe',
    metavar='PATH',
)
parser.add_argument(
    '-n',
    type=int,
    default=10,
    help='Number of lines to show',
)


# --- Code for building docs ---
cli_block_for_docs = re.sub(
    r'usage: .*?.py',
    'usage: <strong>python -m asr_eval.utils.storage.probe</strong>',
    parser.format_help()
)
__doc__ = (
    # need a literal block (like .. code-block::) but with word wrap
    f'{description}\n\n.. raw:: html\n\n\t'
    + '<pre style="white-space: pre-wrap">'
    + cli_block_for_docs.replace('\n', '<br>')
    + '</pre>'
)
parser.description = description
# --- End code for building docs ---


if __name__ == '__main__':
    import polars as pl

    args = parser.parse_args()
    
    storage = make_storage(args.storage)
    
    row_iterator = storage.iter_rows(load_values=True)
    
    rows: list[dict[str, Any]] = []
    for row in islice(row_iterator, args.n):
        row['value'] = str(row['value'])[:100]
        rows.append(row)
    
    df = pl.DataFrame(rows)
    df = df.select(pl.selectors.exclude('value'), 'value')
    df = df.fill_null('')
    
    with pl.Config(tbl_rows=args.n) as cfg:
        print(df)