import pint_pandas

from cet_units import ureg

from .accessor import TEAMAccessor


# Set the unit registry for pint_pandas.
pint_pandas.PintType.ureg = ureg
