from collections.abc import Iterable
from itertools import chain
import re
from typing import cast
from collections import Counter

from asr_eval.bench.datasets import AudioSample, get_dataset


en_words: list[str] = []

for sample in cast(Iterable[AudioSample], chain(
    get_dataset('speech-massive-ru'), 
    get_dataset('multivariant-v1-200'), 
    get_dataset('fleurs-ru'), 
)):
    text = sample['transcription'].lower()
    en_words += re.findall(r'[a-z]+', text)

print(Counter(en_words).most_common(100))