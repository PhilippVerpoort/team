import os

import plotly.io as pio
from itables import init_notebook_mode, show

# Set up plotly.
pio.renderers.default = "notebook_connected"
pio.templates["docs_template"] = pio.templates["simple_white"].update(
    layout=dict(
        dragmode=False,
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
)
pio.templates.default = "docs_template"


# Set up itables.
init_notebook_mode(connected=True)


# Silence POSTED warnings when executing for mkdocs.
if IS_DOCS:
    from warnings import filterwarnings

    filterwarnings("ignore", category=POSTEDWarning)
