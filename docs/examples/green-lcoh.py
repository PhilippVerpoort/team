# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python (docs)
#     language: python
#     name: docs
# ---

# %% [markdown]
# # Calculating LCOH

# %% [markdown]
# Here we calculate the Levelised Cost of Hydrogen (LCOH) for electrolytic hydrogen.

# %% [markdown]
# ## Prepare notebook.

# %%
import numpy as np
import pandas as pd
import plotly

pd.options.plotting.backend = "plotly"

from cet_units import Q
from posted import TEDF
from team.tools import calc_LCOX

# %% [markdown]
# ## Obtain data

# %%
tech_ely = TEDF.load("Tech|Electrolysis").aggregate(
    subtech=["AEL", "PEM"],
    size="100 MW",
    append_references=True,
    agg=["source", "subtech"],
)
display(tech_ely)

# %% [markdown]
# ## Choose assumptions

# %%
assumptions = pd.concat([
    pd.DataFrame().from_records([
        {"variable": "Price|Water", "value": 10.0, "unit": "EUR_2024/t_H2O"},
        {"variable": "OCF", "value": 50.0, "unit": "percent"},
    ]),
    pd.DataFrame({"price_case": ["low", "high"], "variable": "Price|Electricity", "value": [30.0, 60.0], "unit": "EUR_2024/MWh"}),
], ignore_index=True).team.sort_columns()
display(assumptions)

# %% [markdown]
# ## Calculate LCOH for different electricity prices

# %%
df = (
    tech_ely
    .team.perform(calc_LCOX, reference="Output|Hydrogen", using=assumptions, interest_rate=0.08, book_lifetime="20 years", only_new=True)
    .team.unit_to("EUR_2024/MWh_H2_LHV")
    .sort_values(by="price_case", key=lambda col: col.apply(["low", "high"].index))
)
display(df)

display(
    df
    .plot.bar(
        x="price_case",
        y="value",
        color="variable",
    )
    .update_layout(
        xaxis_title="Electricity price case",
        yaxis_title=f"LCOH  ( {df.unit.iloc[0]} )",
    )
)

# %% [markdown]
# ## Calculate LCOH for different capacity factors

# %%
assumptions = pd.concat([
    pd.DataFrame().from_records([
        {"variable": "Price|Electricity", "value": 50.0, "unit": "EUR_2024/MWh"},
        {"variable": "Price|Water", "value": 10.0, "unit": "EUR_2024/t_H2O"},
    ]),
    pd.DataFrame({"ocf": pd.Series(range(10, 100)), "variable": "OCF", "value": np.arange(10.0, 100.0), "unit": "percent"}),
])

# %%
df = (
    tech_ely
    .team.perform(calc_LCOX, reference="Output|Hydrogen", using=assumptions, interest_rate=0.08, book_lifetime="20 years", only_new=True)
    .team.unit_to("EUR_2024/kg_H2")
    .team.sum_over("variable")
)
display(df)

display(
    df
    .plot.line(
        x="ocf",
        y="value",
    )
    .update_layout(
        xaxis_title="Operational Capacity Factor  (%)",
        yaxis_title=f"LCOH  ( {df.unit.iloc[0]} )",
    )
)
