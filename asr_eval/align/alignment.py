from dataclasses import dataclass

from .matching import solve_optimal_alignment
from .parsing import parse_multivariant_string, parse_single_variant_string
from .transcription import MultiVariantTranscription, SingleVariantTranscription


@dataclass
class Alignment:
    '''
    TODO docstring
    '''
    truth: MultiVariantTranscription | SingleVariantTranscription
    pred: SingleVariantTranscription
    multivariant_indices: list[int]
    buckets: list[list[str]]
    
    def __init__(
        self,
        truth: str | MultiVariantTranscription | SingleVariantTranscription,
        pred: str | SingleVariantTranscription,
    ):
        if isinstance(truth, str):
            truth = parse_multivariant_string(truth)
        if isinstance(pred, str):
            pred = parse_single_variant_string(pred)
        
        matches, selected_multivariant_blocks = solve_optimal_alignment(truth.tokens, pred.tokens)
        
        ... # TODO