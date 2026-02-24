# Class DatasetSpec (defined in asr_eval/bench/datasets/dataset_spec.py at lines 5-138)

@dataclasses.dataclass
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
    ...

    name_pattern: str

    augmentor: str | typing.Literal['none', 'all'] = 'all'

    parser: str | typing.Literal['default'] = 'default'

    n_samples: int | typing.Literal['all'] = 'all'

    n_samples_mode: typing.Literal['up_to', 'exactly'] = 'up_to'

    def to_string(self) -> str:
    ...

    @classmethod
    def from_string(cls, string: str) -> typing.Self:
        # example input: gigaam-*|!*-lm-*:n=1000!:a=all
        ...