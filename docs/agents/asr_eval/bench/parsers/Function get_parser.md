# Function get_parser (defined in asr_eval/bench/parsers/_registry.py at lines 44-55)

def get_parser(name: str, type: typing.Literal['true', 'pred']):
    """Retrieve a registered parser for annotation (:code:`type='true'`)
    of prediction (:code:`type='pred'`). Will instantiate this parser
    and return it on all subsequent calls (useful for parsers containing
    neural text normalizers).
    """
    ...