from warnings import warn
from re import match

import pandas as pd

from cet_units import Q


def calc_emissions(data: dict[str, float | Q | pd.Series] | pd.DataFrame):
    # Define return dict.
    ret = {}

    # Determine Emissions from Inputs and Outputs.
    unused = []
    for io in data:
        # Loop only over Inputs/Outputs.
        if not match(r"^(Input|Output)\|.*", io):
            continue
        # Split variable up into type (Input/Output) and flow (H2, CH4, etc).
        io_type, io_flow = io.split("|", 2)
        # Get emission factors.
        emi_fac = data.get("Emission Factor|" + io_flow)
        if not emi_fac:
            unused.append(io)
            continue
        # Inputs are counted as positive, outputs are counted as negative.
        emi_sign = +1 if io_type == "Input" else -1
        # Combine type, word, and flow into new name.
        emi_name = f"GHG Emissions|{io_flow}"
        ret[emi_name] = emi_sign * data[io] * emi_fac

    # Warn about unknown emission factors.
    if unused:
        warn(
            "There is no emission factors specified for the following "
            "inputs/outputs: " + ", ".join(unused)
        )

    return ret


def calc_GHGI(
    data: dict[str, float | Q | pd.Series] | pd.DataFrame,
    reference: str,
    rescale: float = 1.0,
):
    # Get reference.
    ref = data.get(reference)

    # Raise exception if reference is zero or cannot be found in the data.
    if reference and ref is None:
        raise Exception("Reference is not provided in data: " + reference)

    # Define return dict.
    ret = {}

    # Get Emissions.
    for emi in data:
        # Loop only over GHG Emissions.
        if not match(r"^Emission\|.*", emi):
            continue
        # Split emissions type (CO2, CH4, N2O, etc) off from variable name.
        emi_type = emi.split("|", 2)[1]
        # Combine emi type into new name.
        emi_name = f"GHGI|{emi_type}"
        ret[emi_name] = data[emi]

    # Divide by reference and rescale by factor before returning.
    return {key: value / ref * rescale for key, value in ret.items()}
