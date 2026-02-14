from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dash import dash_table
    from dash.html import Div
    from dash.development.base_component import Component

from asr_eval.align.metrics import (
    dataset_metric_to_dataframe, plot_dataset_metric
)
from asr_eval.bench.datasets import get_dataset_sample_by_id
from asr_eval.bench.pipelines import get_pipeline_index
from asr_eval.bench.evaluator import DatasetData, SampleData

try:
    # need to import like this, or the error occurs:
    # "Component library was imported during callback."
    import dash
    from dash.html import Div, Audio, P
    from dash_extensions import Purify
except ImportError:
    pass



def render_samples(
    dataset_data: DatasetData,
    audio_urls: dict[int, str] | None = None,
    max_string_length: int | None = 250,
) -> Div:

    blocks: list[Component] = []
    for sample in dataset_data.samples:
        rendered_html_lines = render_sample_as_html_lines(sample)
        if rendered_html_lines is not None:
            # rendered_html_lines is None if skipping due to max_samples limit
            if audio_urls is not None:
                audio_url_path = audio_urls[sample.sample_id]
                blocks.append(Audio( # pyright: ignore[reportPossiblyUnboundVariable]
                    src=dash.get_asset_url(audio_url_path), # type: ignore
                    autoPlay=False,
                    controls=True,
                    preload='none',
                    style={'display': 'block', 'margin-top': '10px'},
                ))
            if max_string_length is not None:
                pass
                # TODO decide what to to with html tags
                # wrapped_blocks: list[list[str]] = [rendered_html_lines]
                # while True:
                #     last_block = wrapped_blocks[-1]
                #     max_line_len = max(len(line) for line in last_block)
                #     if max_line_len <= max_string_length:
                #         break
                #     # find the best split pos
                #     for char_pos in range(max_string_length, max(-1, \
                #             max_string_length - 20), -1):
                #         if all(
                #             len(line) <= char_pos or line[char_pos] == ' '
                #             for line in last_block
                #         )
                #     break
                # rendered_html_lines = list(chain(*wrapped_blocks))
            blocks.append(P( # pyright: ignore[reportPossiblyUnboundVariable]
                Purify(html='<br/>'.join(rendered_html_lines)), # pyright: ignore[reportPossiblyUnboundVariable]
                style={'margin-bottom': '0px'},
            ))
    
    return Div(blocks, style={
        'font-family': '"Consolas", "Ubuntu Mono", "Monaco", monospace',
        'white-space': 'pre',
        'font-stretch': 'condensed',
    })


def render_metric_plots(dataset_data: DatasetData) -> list[Component]:
    from dash.html import Img

    dataset_metric = dataset_data.dataset_metric
    wer_base64 = plot_dataset_metric(
        dataset_metric, what='wer', show=False
    )
    n_replacements_base64 = plot_dataset_metric(
        dataset_metric, what='n_replacements', show=False
    )
    n_insertions_base64 = plot_dataset_metric(
        dataset_metric, what='n_insertions', show=False
    )
    n_deletions_base64 = plot_dataset_metric(
        dataset_metric, what='n_deletions', show=False
    )
    
    BASE64_HEADER = 'data:image/svg+xml;base64,'
    STYLE = {'flex': '1 1 auto', 'max-width': '24.5%'}
    return [
        Img(src=BASE64_HEADER + wer_base64, style=STYLE),
        Img(src=BASE64_HEADER + n_replacements_base64, style=STYLE),
        Img(src=BASE64_HEADER + n_insertions_base64, style=STYLE),
        Img(src=BASE64_HEADER + n_deletions_base64, style=STYLE),
    ]
    
def _df_to_html(df: pd.DataFrame, **kwargs: Any) -> dash_table.DataTable:
    from dash import dash_table

    return dash_table.DataTable(
        df.round(3).to_dict('records'), # type: ignore
        [{"name": i, "id": i} for i in cast(Sequence[str], df.columns)],
        **kwargs,
    )
    
def render_metric_tables(dataset_data: DatasetData) -> list[Component]:
    dataset_metric = dataset_data.dataset_metric
    wer_df = dataset_metric_to_dataframe(
        dataset_metric, what='wer'
    )
    n_replacements_df = dataset_metric_to_dataframe(
        dataset_metric, what='n_replacements'
    )
    n_insertions_df = dataset_metric_to_dataframe(
        dataset_metric, what='n_insertions'
    )
    n_deletions_df = dataset_metric_to_dataframe(
        dataset_metric, what='n_deletions'
    )
    
    return [
        _df_to_html(wer_df),
        _df_to_html(n_replacements_df),
        _df_to_html(n_insertions_df),
        _df_to_html(n_deletions_df),
    ]


def render_sample_as_html_lines(sample: SampleData) -> list[str] | None:
        if sample.baseline_transcription_html is None:
            return None  # skip rendering this sample due to max_samples limit
        display_rows: list[tuple[str, ...]] = []
        if sample.baseline_is_ground_truth:
            # labeled dataset
            display_rows.append((
                'time', # elapsed_time
                'E',  # n_errors
                'R',  # n_replacements
                'D',  # n_deletions
                'I',  # n_insertions
                'True', # pipeline_name
                sample.baseline_transcription_html
            ))
            for pipeline_name, pipeline_data in sorted(
                sample.pipelines.items(),
                key=lambda item: get_pipeline_index(item[0]),
            ):
                display_rows.append((
                    (
                        f'{pipeline_data.elapsed_time:.2f}'
                        if not np.isnan(pipeline_data.elapsed_time)
                        else '?'
                    ),
                    str(pipeline_data.metrics.n_errors),
                    str(pipeline_data.metrics.n_replacements),
                    str(pipeline_data.metrics.n_deletions),
                    str(pipeline_data.metrics.n_insertions),
                    pipeline_name,
                    cast(str, pipeline_data.transcription_html),
                ))
        else:
            # unlabeled dataset
            display_rows.append((
                'time', # elapsed_time
                sample.baseline_name,
                sample.baseline_transcription_html
            ))
            for pipeline_name, pipeline_data in sample.pipelines.items():
                display_rows.append((
                    f'{pipeline_data.elapsed_time:.2f}',
                    pipeline_name,
                    cast(str, pipeline_data.transcription_html),
                ))
            
        df = pd.DataFrame(display_rows)
        # the last column (transcription) contans <span> tags, so .to_string()
        # shows it incorrectly
        display_lines = df.iloc[:, :-1].to_string( # type: ignore
            # col_space=5,
            index=False,
            header=False,
        ).split('\n')
        
        # we can use <br/> or \n for white-space: pre mode
        return [
            f'<span style="font-size: 0.7em">{sample.sample_id}</span>'
        ] + [
            line + ' |' + transcription
            for line, transcription in zip(display_lines, df.iloc[:, -1]) # type: ignore
        ]
    

def get_audio_assets_url(
    dataset_name: str,
    augmentor_name: str,
    sample_id: int,
    root_dir: Path,
    do_export: bool = True,
) -> str:
    audio_url_path = f'{dataset_name}/{sample_id}.mp3'
    audio_path = root_dir / audio_url_path
    if do_export and not audio_path.is_file():
        import soundfile as sf
        
        waveform = get_dataset_sample_by_id(
            dataset_name,
            split='test',
            sample_id=sample_id,
            augmentor_name=augmentor_name,
        )['audio']['array']
        print('Exporting', audio_path)
        audio_path.parent.mkdir(exist_ok=True, parents=True)
        sf.write(audio_path, waveform, samplerate=16_000) # type: ignore
    return audio_url_path