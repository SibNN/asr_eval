# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# type: ignore

import tomllib
from datetime import datetime
from pathlib import Path
import warnings

from sphinx.ext import autodoc
from sphinx.ext.autodoc import _generate
from sphinx.deprecation import RemovedInSphinx10Warning

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)



# Monkey patching to disable "Bases: object" or "Bases: _PrivateCls"
# tested for sphinx 9.0.4, will not work for sphinx<9

wrapped = _generate._add_directive_lines
def _add_directive_lines_wrapper(
    *, more_content, is_final, config, indent, options, props, result, source_name,
) -> None:
    def filter_base(base: str) -> bool:
        base = base.removeprefix(':py:class:`').removesuffix('`')
        if base == 'object':
            return False
        base = base.split('.')[-1]
        if base.startswith('_'):  # base class name starts from underscore
            return False
        return True
    if hasattr(props, '_obj_bases'):
        props._obj_bases = tuple(b for b in props._obj_bases if filter_base(b))
        if not props._obj_bases:
            options = options.copy()
            setattr(options, 'show_inheritance', False)
    return wrapped(
        more_content=more_content,
        is_final=is_final,
        config=config,
        indent=indent,
        options=options,
        props=props,
        result=result,
        source_name=source_name,
    )
_generate._add_directive_lines = _add_directive_lines_wrapper

# Monkey patching to disable "Bases: object" or "Bases: _PrivateCls"
# for sphinx<9

# class MockedClassDocumenter(autodoc.ClassDocumenter):
#     def add_line(self, line: str, source: str, *lineno: int) -> None:
#         print(f'{line.strip()=}')
#         if line.strip() == 'Bases:':
#             return
#         super().add_line(line, source, *lineno)
# autodoc.ClassDocumenter = MockedClassDocumenter
# def setup(app):
#     def handle_bases(app, name, obj, options, bases: list):
#         bases[:] = [
#             _cls for _cls in bases
#             if _cls is not object
#             and not _cls.__name__.startswith('_')
#         ]
#     app.connect('autodoc-process-bases', handle_bases)


with open(Path(__file__).parent.parent.parent / 'pyproject.toml', 'rb') as f:
    project_data = tomllib.load(f)

import asr_eval

url = project_data['project']['urls']['Repository']
project = project_data['project']['name']
release = asr_eval.__version__
author = project_data['project']['authors'][0]['name']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    # https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'bysource'  # sort by order in __all__
autoclass_content = 'both'  # include __init__ docs into class docs

napoleon_preprocess_types = True
napoleon_use_param = True

autodoc_default_options = {
    # 'special-members': '__init__',
}

autodoc_typehints = 'description'  # show types in description, not in signature
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True

intersphinx_resolve_self = 'asr_eval'

# a global variable `autodoc_type_aliases` is recognied by Sphinx
# from asr_eval import _autodoc_type_aliases as autodoc_type_aliases

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx13'
html_theme_path = ['_themes']
html_css_files = ['sphinx13.css', 'tweaks.css']
html_static_path = ['_static']