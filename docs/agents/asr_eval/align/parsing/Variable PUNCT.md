# Variable PUNCT (defined in asr_eval/align/parsing.py at lines 24-34)

PUNCT = re.escape(r""".,!?:;…-‑–—'"‘“”«»()[]{}""")
r"""
A default set of punctuation characters to exclude from words
:code:`.,!?:;…-‑–—'"‘“”«»()[]{}`. Note that this does not affect parsing
multivariant syntax. To override, create a
:class:`~asr_eval.align.parsing.Parser` with custom
:attr:`~asr_eval.align.parsing.Parser.tokenizing` field.

:meta hide-value:
"""