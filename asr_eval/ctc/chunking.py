from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy
from tqdm.auto import tqdm


@dataclass(init=False)
class LogProbsWindow:
    clipped_start_ticks: int
    clipped_end_ticks: int
    log_probs: npt.NDArray[np.floating[Any]]
    weights: npt.NDArray[np.floating[Any]]
    
    def __init__(
        self,
        start_time: float,  # can be negative
        end_time: float,
        waveform: npt.NDArray[np.floating[Any]],
        ctc_model: Callable[[npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]],
        model_tick_size_sec: float,
        sampling_rate: int = 16_000,
    ):
        total_len_sec = len(waveform) / sampling_rate
        
        clipped_start_time = np.clip(start_time, 0, total_len_sec)
        clipped_end_time = np.clip(end_time, 0, total_len_sec)
        
        self.clipped_start_ticks = int(clipped_start_time / model_tick_size_sec)
        clipped_end_ticks = int(clipped_end_time / model_tick_size_sec)
        
        clipped_start_pos = int(self.clipped_start_ticks * model_tick_size_sec * sampling_rate)
        clipped_end_pos = int(clipped_end_ticks * model_tick_size_sec * sampling_rate)
        
        waveform_chunk = waveform[clipped_start_pos : clipped_end_pos]
        self.log_probs = ctc_model(waveform_chunk)
        
        expected_ticks = clipped_end_ticks - self.clipped_start_ticks
        actual_ticks = len(self.log_probs)
        assert np.allclose(actual_ticks, expected_ticks, atol=1.1), (
            f'Expected {expected_ticks} ticks from the audio length,'
            f' got {actual_ticks} ticks from the model. Incorrect {model_tick_size_sec=}?'
        )
        self.clipped_end_ticks = self.clipped_start_ticks + len(self.log_probs)
        
        clip_ratio_start = (clipped_start_time - start_time) / (end_time - start_time)
        clip_ratio_end = (clipped_end_time - start_time) / (end_time - start_time)
        
        self.weights = scipy.stats.beta.pdf(
            np.linspace(clip_ratio_start, clip_ratio_end, num=len(self.log_probs)), a=5, b=5
        )
        self.weights /= self.weights.max()


def average_logp_windows(windows: list[LogProbsWindow]) -> npt.NDArray[np.floating[Any]]:
    max_ticks = max(window.clipped_end_ticks for window in windows)
    
    sum_weights = np.zeros(max_ticks)
    for window in windows:
        sum_weights[window.clipped_start_ticks:window.clipped_end_ticks] += window.weights

    averaged_log_probs = np.zeros((max_ticks, windows[0].log_probs.shape[1]))
    for window in windows:
        span = slice(window.clipped_start_ticks, window.clipped_end_ticks)
        averaged_log_probs[span] += (
            window.log_probs * (window.weights / sum_weights[span])[:, None]
        )
    
    return averaged_log_probs


def chunked_ctc_prediction(
    waveform: npt.NDArray[np.floating[Any]],
    ctc_model: Callable[[npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]],
    model_tick_size_sec: float,
    segment_size_sec: float = 30,
    segment_shift_sec: float = 6,
    sampling_rate: int = 16_000,
) -> list[LogProbsWindow]:
    total_len_sec = len(waveform) / sampling_rate
    # total_ticks = int(total_len_sec / model_tick_size_sec)
    # print(f'{total_len_sec = }, {total_ticks = }')

    windows: list[LogProbsWindow] = []
    for center in np.arange(0, total_len_sec, step=segment_shift_sec):
        start = center - segment_size_sec / 2  # can be negative
        end = center + segment_size_sec / 2
        windows.append(LogProbsWindow(
            start_time=start,
            end_time=end,
            waveform=waveform,
            ctc_model=ctc_model,
            model_tick_size_sec=model_tick_size_sec,
            sampling_rate=sampling_rate,
        ))

    return windows