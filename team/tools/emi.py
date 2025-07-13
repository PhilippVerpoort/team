from typing import Callable
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
        if emi_fac is None:
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
            "There are no emission factors specified for the following "
            "inputs/outputs: " + ", ".join(unused)
        )

    return ret


def calc_GHGI(
    data: dict[str, float | Q | pd.Series] | pd.DataFrame,
    name: str | None = None,
    reference: str | None = None,
):
    # Get reference.
    ref = data.get(reference) if reference else 1.0

    # Raise exception if reference is zero or cannot be found in the data.
    if reference and ref is None:
        raise Exception("Reference is not provided in data: " + reference)

    # Define return dict.
    ret = {}

    # Get Emissions.
    for emi in data:
        m = match(r"^GHG Emissions\|(.*)", emi)
        if m:
            ret[m.group(1)] = data[emi]

    # Define prefix.
    prefix = "GHGI|"
    if name is not None:
        prefix += str(name) + "|"

    # Divide by reference before returning.
    return {(prefix+key): value / ref for key, value in ret.items()}


def calc_GHGI_pc(
    data: dict[str, float | Q | pd.Series] | pd.DataFrame,
    name: str | None = None,
    reference: str | None = None,
    pat: Callable | None = None,
    **kwargs,
):
    # Get functional unit.
    if reference is not None:
        if name is not None:
            fu = data.get(f"Functional Unit|{name}|{reference}")
        else:
            fu = data.get(f"Functional Unit|{reference}")
        if fu is None:
            raise Exception(
                "Functional unit could not be found in data: " + str(reference)
            )
    else:
        fu = None

    # Set variable name prefix.
    prefix_scale = "Scaling|"
    if name is not None:
        prefix_scale += str(name) + "|"

    # Get list of processes with scaling.
    processes = [
        c.removeprefix(prefix_scale)
        for c in data
        if c.startswith(prefix_scale)
    ]

    # Raise exception if no scalings are found.
    if not processes:
        raise Exception("Scaling factors for processes could not be found.")

    # Set default pattern if not provide.
    if pat is None:
        pat = "Tech|{proc}|".format

    # Loop over processes and determine LCOX. Rescale according to computed
    # scalings. Divide by functional unit if specified. Then return.
    ret = {}
    for proc in processes:
        prefix_tech = pat(proc=proc)
        ret_proc = calc_GHGI(
            data.rename(columns=lambda c: c.removeprefix(prefix_tech)),
            **kwargs,
        )
        factor = data[prefix_scale + proc]
        if fu is not None:
            factor /= fu
        if name is not None:
            prefix_new = f"GHGI|{name}|{proc}|"
        else:
            prefix_new = f"GHGI|{proc}|"
        ret |= {
            (prefix_new + var_name.removeprefix("GHGI|")): var_value * factor
            for var_name, var_value in ret_proc.items()
        }
    return ret
