from pathlib import Path
import re


header = '''\
Reference documentation
#######################


'''


if __name__ == '__main__':
    output_rst = header

    for path in sorted(Path('docs/source').glob('asr_eval.*.rst')):
        content = path.read_text()
        
        module_name_header = content.split('\n\n', 1)[0].replace(' package', '')
        
        output_rst += f'{module_name_header}\n\n\n'
        for block in re.finditer(r'\.\. automodule[^\n]*(\n +[^\n]+)+', content, flags=re.DOTALL):
            output_rst += f'{block.group()}\n\n\n'
        
        path.unlink()
        
    Path('docs/source/asr_eval.rst').unlink()
    Path('docs/source/reference.rst').write_text(output_rst)