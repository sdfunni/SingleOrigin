# Configuration file for the Sphinx documentation builder.
import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))

# -- Project information

project = "SingleOrigin"
copyright = "2025, Stephen D. Funni"
author = "Stephen D. Funni"

release = '3.0'
version = '3.0b2'

# -- General configuration

exclude_patterns = [
    'Examples',
]

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
#    'sphinx.ext.autodoc',
#    'sphinx.ext.autosummary',
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

# html_theme = 'pydata_sphinx_theme'

# Theme options for pydata_sphinx_theme:
# html_theme_options = {
#     "show_nav_level": 2,  # Adjust sidebar depth
#     "header_links_before_dropdown": 4, # Show top links before the dropdown menu
# }

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- AutoAPI information
autoapi_dirs = [os.path.abspath('../.') + '/SingleOrigin/']
autoapi_type = "python"
autoapi_add_toctree_entry = False
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
#    'show-module-summary',
]
autoapi_python_class_content = 'both'
autoapi_member_order = 'bysource'
