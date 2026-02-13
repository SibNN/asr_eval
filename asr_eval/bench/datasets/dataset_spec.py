from dataclasses import dataclass
from typing import Literal, Self


@dataclass
class DatasetSpec:
    """Represents an extended syntax for specifying datasets when
    running pipelines and dashboard. Allows to specify the required
    samples count, augmentor and parser.
    
    The dataset spec is understanded and used by two utilities:
    
    1. :mod:`python -m asr_eval.bench.run <asr_eval.bench.run>`
    2. :mod:`python -m asr_eval.bench.dashboard.run <asr_eval.bench.dashboard.run>`
    
    A dataset spec has a string representation as a semicolon-separated
    string. The first value is a name pattern, other values are
    modifiers in form :code:`<key>=<value>`.
    
    The "a" modifier specifies the augmentor to use (see
    :class:`~asr_eval.bench.augmentors.AudioAugmentor`). Has a special
    value "all" (a default value) - when running pipelines it is treated
    as "run without augmentor", and when running dashboard it is treated
    as "load the results will all augmentors available in
    storage".
    
    The "p" modifier specifies the parser to use (see
    :func:`~asr_eval.bench.parsers.get_parser`). It is ignored when
    running pipelines, and when running dashboard the specified parser
    will be used. By default uses a "default" parser
    (:data:`~asr_eval.align.parsing.DEFAULT_PARSER`).
    
    The "n" modifier specifies the number of samples. If may be either
    "all" or integer, where "all" means all the samples in the dataset.
    The value may also have exclamation mark as suffix (example:
    "n=20!") - when running pipelines it is ignored, and when running
    a dashboard it will drop all the pipeline with not enough samples.
    For example, if "n=all!", then all the pipelines with partial
    results will not be displayed in the dashboard.

    See Also:
        See details and examples in the user guide
        :doc:`/guide_evaluation_dashboard`.
    
    Example:
    
        >>> from asr_eval.bench.datasets import DatasetSpec
        >>> DatasetSpec.from_string('fleurs-*:p=ru-norm:n=50!') # doctest: +SKIP
        DatasetSpec(
            name_pattern='fleurs-*',
            augmentor='all',
            parser='ru-norm',
            n_samples=50,
            n_samples_mode='exactly'
        )
    """
    name_pattern: str
    augmentor: str | Literal['none', 'all'] = 'all'
    parser: str | Literal['default'] = 'default'
    n_samples: int | Literal['all'] = 'all'
    
    # exclamation mark suffix for n_samples:
    n_samples_mode: Literal['up_to', 'exactly'] = 'up_to'

    def to_string(self) -> str:
        result = self.name_pattern
        if self.augmentor != 'all':
            # if non-default value
            result += f':a={self.augmentor}'
            # if non-default value
        if self.parser != 'none':
            result += f':p={self.parser}'
        if self.n_samples != 'all' or self.n_samples_mode != 'up_to':
            # if non-default value
            result += f':n={self.n_samples}'
            if self.n_samples_mode == 'exactly':
                result += '!'
        return result

    @classmethod
    def from_string(cls, string: str) -> Self:
        # example input: gigaam-*|!*-lm-*:n=1000!:a=all
        name_pattern, *spec_expressions = string.split(':')
        
        # for the our example, here:
        # name_pattern == "gigaam-*|!*-lm-*"
        # spec_expressions == ["n=1000!", "a=all"]
        
        spec_dict: dict[str, str] = {}
        for expr in spec_expressions:
            if not '=' in expr:
                raise ValueError(
                    f'Expression "{expr}" does not contain'
                    f' "=" symbol in the dataset spec {string}'
                )
            k, v = expr.split('=', 1)
            if len(k) == 0 or len(v) == 0:
                raise ValueError(
                    f'Empty key or value in the expression {expr}'
                    f' in the dataset spec {string}'
                )
            if k in spec_dict:
                raise ValueError(
                    f'Key "{k}" repeats multiple times'
                    f' in the dataset spec {string}'
                )
            spec_dict[k] = v
            
        # for the our example, here:
        # spec_pairs == {"n": "1000!", "a": "all"}
        
        obj = cls(name_pattern=name_pattern)
        for k, v in spec_dict.items():
            match k:
                case 'a':
                    obj.augmentor = v
                case 'p':
                    obj.parser = v
                case 'n':
                    if v.endswith('!'):
                        obj.n_samples_mode = 'exactly'
                        v = v.removesuffix('!')
                    if v != 'all':
                        try:
                            v = int(v)
                        except ValueError as e:
                            raise ValueError(
                                f'Value of "{k}" should be integer'
                                f' or "all" in the dataset spec {string}'
                            ) from e
                    obj.n_samples = v
                case _:
                    raise ValueError(
                        f'Unknown key "{k}" in the dataset spec {string}'
                    )
        
        return obj