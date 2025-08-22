# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "SingleOrigin"
copyright = "2025, Stephen D. Funni"
author = "Stephen D. Funni"

release = '3.0'
version = '3.0b2'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- AutoAPI information
import os
autoapi_dirs = [os.path.abspath('../.') + '/SingleOrigin/']
autoapi_type = "python"