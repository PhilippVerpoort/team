"""TEAM: Techno-economic assessment and manipulation framework.

TEAM is a framework for performing techno-economic assessments in the context
of energy and climate-mitigation studies.
"""

import pint_pandas
from cet_units import ureg

from ._accessor import TEAMAccessor as team_accessor

# Set the unit registry for pint_pandas.
pint_pandas.PintType.ureg = ureg

__all__ = [
    "team_accessor",
]
