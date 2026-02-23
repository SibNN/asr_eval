from collections import defaultdict

from asr_eval.utils.docutils import extract_definitions, pretty_format_module_defs
from pathlib import Path

import tiktoken

def num_tokens(string: str, encoding_name: str = 'gpt-4') -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(string, disallowed_special=())
    return len(tokens)

repo_dir = Path('.')
generated_docs_dir = Path('docs/build/generated')
files = sorted(Path('asr_eval').rglob('*.py'))

total_tokens = 0
total_tokens_per_dir: dict[str, int] = defaultdict(int)

for path in files:
    path = path.relative_to(repo_dir)

    defs = extract_definitions(repo_dir, path)
    if defs:
        module = defs[0].module
        content = (
            f'# Docs for module `{module}` (`{path}`)'
            + '\n\n'
            + pretty_format_module_defs(defs, str(path))
        )

        out_file = generated_docs_dir / path.with_suffix('.md')
        out_file.parent.mkdir(exist_ok=True, parents=True)
        out_file.write_text(content)
        
        n_tokens = num_tokens(content)
        total_tokens += n_tokens

        submodule = '.'.join(module.split('.')[:2])
        total_tokens_per_dir[submodule] += n_tokens

        print(f'{path}: {n_tokens} tokens')

total_code_tokens = sum([
    num_tokens(path.read_text()) for path in files
])
print(f'Total .py tokens: {total_code_tokens}')

print(f'Total .md tokens: {total_tokens}')
for dir_name, n_tokens in total_tokens_per_dir.items():
    print(f'Total .md tokens for {dir_name}: {n_tokens}')
