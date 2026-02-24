import shutil
import subprocess
import textwrap

from asr_eval.utils.docutils import extract_definitions
from pathlib import Path

import tiktoken

def num_tokens(string: str, encoding_name: str = 'gpt-4') -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(string, disallowed_special=())
    return len(tokens)

# inputs
repo_dir = Path('.')
files = sorted(Path('asr_eval').rglob('*.py'))

# outputs
generated_docs_dir = Path('docs/agents')
generated_tree_file = (
    generated_docs_dir.parent / (generated_docs_dir.stem + '_tree.txt')
)
shutil.rmtree(generated_docs_dir, ignore_errors=True)

# generating docs
for path in files:
    path = path.relative_to(repo_dir)

    defs = extract_definitions(repo_dir, path)

    if defs:
        module = defs[0].module

        top_defs = [d for d in defs if '.' not in d.name]
        sub_defs = [d for d in defs if '.' in d.name]  # class fields and methods

        output_dir = generated_docs_dir / Path(module.replace('.', '/'))
        output_dir.mkdir(exist_ok=True, parents=True)
        
        files_for_top_defs: dict[str, str] = {}

        top_name_types = {
            d.name: d.human_readable_type
            for d in top_defs
        }

        for d in top_defs:
            files_for_top_defs[top_name_types[d.name] + ' ' + d.name] = (
                f'# {d.human_readable_type} {d.name}'
                + f' ({d.defined_at(skip_default_path='')})'
                + '\n\n'
                + d.definition_ellipsis
            )

        for d in sub_defs:
            top_name, subname = d.name.split('.', 1)
            files_for_top_defs[top_name_types[top_name] + ' '+ top_name] += (
                '\n\n' + textwrap.indent(d.definition_ellipsis, '    ')
            )

        for stem, content in files_for_top_defs.items():
            (output_dir / f'{stem}.md').write_text(content)

# saving tree
tree_cmd = ["tree", str(generated_docs_dir)]
tree = subprocess.check_output(tree_cmd).decode()
generated_tree_file.parent.mkdir(exist_ok=True, parents=True)
generated_tree_file.write_text(tree)

# stats
total_code_tokens = sum([
    num_tokens(path.read_text())
    for path in files
])
print(f'Total .py tokens: {total_code_tokens}')

total_docs_tokens = sum([
    num_tokens(path.read_text())
    for path in generated_docs_dir.rglob('*')
    if path.is_file()
])
print(f'Total docs tokens: {total_docs_tokens}')


total_tree_tokens = num_tokens(tree)
print(f'Total tree tokens: {total_tree_tokens}')

print('Tree command:', " ".join(tree_cmd))
