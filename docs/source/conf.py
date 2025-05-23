# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import os
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder

__PROJECT_ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

project = "pyquickbench"
author = "Gabriel Fougeron"
project_copyright = "2024, Gabriel Fougeron"
version = "0.2.6"

# sys.path.append(os.path.abspath("./_pygments"))
# from style import PythonVSMintedStyle
# pygments_style = PythonVSMintedStyle.__qualname__

language = "en"

extensions = [
    "sphinx.ext.duration"           ,
    "sphinx.ext.doctest"            ,
    "sphinx.ext.autodoc"            ,
    "sphinx.ext.viewcode"           ,
    "sphinx.ext.todo"               ,
    "sphinx.ext.autosummary"        ,
    "sphinx.ext.mathjax"            ,
    "sphinx.ext.napoleon"           ,
    "sphinx.ext.intersphinx"        ,
    "sphinx.ext.githubpages"        ,
    "sphinx_gallery.gen_gallery"    ,
    "sphinx_needs"                  ,
    "sphinxcontrib.test_reports"    ,
    "sphinxcontrib.plantuml"        ,
    "myst_parser"                   ,
    "sphinxext.rediraffe"           ,
    "sphinx_sitemap"                ,
    "sphinx_design"                 ,
    'sphinx_copybutton'             ,
]

# The suffix of source filenames.
source_suffix = ".rst"

master_doc = "index"

# The encoding of source files.
source_encoding = "utf-8"

add_module_names = False

autosummary_generate = True

templates_path = ["templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python"        : ("https://docs.python.org/3"                      , None),
    "sphinx"        : ("https://www.sphinx-doc.org/en/master/"          , None),
    'numpy'         : ('https://numpy.org/doc/stable/'                  , None),
    "scipy"         : ("http://docs.scipy.org/doc/scipy/reference/"     , None),
    "matplotlib"    : ("https://matplotlib.org/stable/"                 , None), 
    # 'Pillow'        : ('https://pillow.readthedocs.io/en/stable/'       , None),
    # 'cycler'        : ('https://matplotlib.org/cycler/'                 , None),
    # 'dateutil'      : ('https://dateutil.readthedocs.io/en/stable/'     , None),
    # 'ipykernel'     : ('https://ipykernel.readthedocs.io/en/latest/'    , None),
    # 'pandas'        : ('https://pandas.pydata.org/pandas-docs/stable/'  , None),
    # 'pytest'        : ('https://pytest.org/en/stable/'                  , None),
    # 'tornado'       : ('https://www.tornadoweb.org/en/stable/'          , None),
    # 'xarray'        : ('https://docs.xarray.dev/en/stable/'             , None),
    # 'meson-python'  : ('https://meson-python.readthedocs.io/en/stable/' , None),
    # 'pip'           : ('https://pip.pypa.io/en/stable/'                 , None),
}

intersphinx_disabled_reftypes = ["*"]
intersphinx_cache_limit = -1
intersphinx_timeout = 1

rediraffe_redirects = {
    "gallery": "_build/auto_examples/index"             ,
    "tutorial": "_build/auto_examples/tutorial/index"   ,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_logo_abs = os.path.join(__PROJECT_ROOT__, "docs", "source", "_static", "img", "plot_icon.png")
html_logo_rel = "_static/img/plot_icon.png"
html_logo = html_logo_rel
html_favicon = html_logo_rel
html_baseurl = "https://gabrielfougeron.github.io/pyquickbench/"
html_show_sourcelink = True

html_theme_options = {
    # "collapse_navigation": True,
    # "navigation_depth": 3,
    # "sidebar_includehidden" : True,
    # "search_bar_text" : "Search the docs ...",
    # "search_bar_position" : "sidebar",
    # "show_toc_level" : 0 ,
    "show_nav_level" : 2,
    "show_prev_next": False,
    "header_links_before_dropdown": 20,
    "use_edit_page_button": True,
    "pygments_light_style": "tango",
    "pygments_dark_style":  "monokai",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gabrielfougeron/pyquickbench",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyquickbench/",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "Anaconda",
            "url": "https://anaconda.org/conda-forge/pyquickbench",
            "icon": "fa-custom fa-anaconda",
        },
    ],
    "logo": {
        "text": "pyquickbench",
        "alt_text": "pyquickbench",
    },
    "footer_start" : "",
    "footer_end" : "",
}

# Add / remove things from left sidebar
html_sidebars = {
    "api": [
        "navbar-logo"       ,
        "sidebar-nav-bs"    ,
    ],
    "_build/auto_examples/**": [
        "navbar-logo"       ,
        "sidebar-nav-bs"    ,
    ],
}

html_context = {
    "display_github"    : True              , # Integrate GitHub
    "github_user"       : "gabrielfougeron" , # Username
    "github_repo"       : "pyquickbench"    , # Repo name
    "github_version"    : "main"            , # Version
    "version"           : "main"            , # Version
    "conf_py_path"      : "docs/source/"    , # Path in the checkout to the docs root
    "doc_path"          : "docs/source/"    ,
    "default_mode"      : "light"           ,
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_js_files = [
    "js/custom-icon.js",
]

tr_report_template = "./test-report/test_report_template.txt"

# sphinx-gallery configuration

sphinx_gallery_conf = {
    # path to your examples scripts
    "filename_pattern"          : "/"                       ,
    "ignore_pattern"            : r"NOTREADY"               ,
    "examples_dirs"             : "../../examples/"         ,
    "gallery_dirs"              : "_build/auto_examples/"   ,
    "subsection_order"          : ExplicitOrder([
        "../../examples/tutorial"  ,
        "../../examples/benchmarks",
    ])                                                      ,
    "within_subsection_order"   : "FileNameSortKey"         ,
    "backreferences_dir"        : "_build/generated"        ,
    "image_scrapers"            : ("matplotlib",)           ,
    "default_thumb_file"        : html_logo_abs             ,
    "plot_gallery"              : True                      ,
    "matplotlib_animations"     : True                      ,
    "nested_sections"           : True                      ,
    "reference_url"             : {"sphinx_gallery": None,} ,
    "min_reported_time"         : 10000                     ,
}


#############
# Latex PDF #
#############
latex_engine = "pdflatex"

# latex_documents = [("startdocname", "targetname", "title", "author", "theme", "toctree_only")]
latex_documents = [
    (master_doc, "pyquickbench.tex", "Pyquickbench documentation", "Gabriel Fougeron", "manual"),
]

latex_elements = {
    "preamble":r"\usepackage{xfrac}",
    'sphinxsetup': "verbatimforcewraps, verbatimmaxunderfull=0",
}

latex_use_latex_multicolumn = False
latex_show_urls = "footnote"

latex_theme = "manual"
# latex_theme = "howto"

##################
# Math rendering #
##################
# 
# mathjax3_config = {
#   "loader": {"load": ["[tex]/xfrac"]},
#   "tex": {"packages": {"[+]": ["xfrac"]}},
# }


#####################
# Napoleon settings #
#####################

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

autodoc_typehints = "description"
autosummary_generate = True

##########################
# Sitemap and robots.txt #
##########################

sitemap_url_scheme = "{link}"
html_extra_path = [
    "./robots.txt"    ,
]