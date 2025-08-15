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

with open(Path(__file__).parent.parent.parent / 'pyproject.toml', 'rb') as f:
    project_data = tomllib.load(f)

project = project_data['project']['name']
release = project_data['project']['version']
author = project_data['project']['authors'][0]['name']
copyright = f'{datetime.now().year}, {author} & Siberian Neuronets LLC'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

def setup(app):
    app.add_css_file('my_theme.css')