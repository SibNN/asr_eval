from pathlib import Path
from tempfile import TemporaryDirectory
from asr_eval.segments.segment import TimedText
from asr_eval.utils.storage.dict_storage import ShelfStorage


def test_shelf_storage():
    with TemporaryDirectory(delete=True) as tmpdirname:
        st = ShelfStorage(Path(tmpdirname) / 'test_storage.db')
        
        st.add_row(value='qwe', dataset='fleurs', sample=0, what='ground_truth')
        st.add_row(value=TimedText(0, 1, 'Hi'), dataset='fleurs', model='whisper', sample=0, what='pred', x=True)
        st.add_row(value=TimedText(0, 1, 'Ho'), dataset='fleurs', model='tuned', steps=100, sample=0, what='pred')
        
        st.delete_all(steps=100.0)
        df = st.list_all(load_values=False)
        assert set([
            tuple(sorted(row.items()))
            for row in df.iter_rows(named=True)
        ]) == {
            (
                ('dataset', 'fleurs'),
                ('model', 'tuned'),
                ('sample', 0),
                ('steps', 100),
                ('what', 'pred'),
                ('x', None)
            ),
            (
                ('dataset', 'fleurs'),
                ('model', None),
                ('sample', 0),
                ('steps', None),
                ('what', 'ground_truth'),
                ('x', None)
            ),
            (
                ('dataset', 'fleurs'),
                ('model', 'whisper'),
                ('sample', 0),
                ('steps', None),
                ('what', 'pred'),
                ('x', True)
            )
        }
        
        st.delete_all(steps=100)
        df = st.list_all(load_values=False)
        assert set([
            tuple(sorted(row.items()))
            for row in df.iter_rows(named=True)
        ]) == {
            (
                ('dataset', 'fleurs'),
                ('model', None),
                ('sample', 0),
                ('what', 'ground_truth'),
                ('x', None)
            ),
            (
                ('dataset', 'fleurs'),
                ('model', 'whisper'),
                ('sample', 0),
                ('what', 'pred'),
                ('x', True)
            )
        }
        
        st.close()