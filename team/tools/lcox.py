from typing import Optional, Callable
from warnings import warn
from re import match, sub

import pandas as pd

from cet_units import ureg, Q


@ureg.wraps(
    ureg.dimensionless / ureg.year,
    (ureg.dimensionless, ureg.year)
)
def annuity_factor(ir: Q, bl: Q):
    return ir * (1 + ir) ** bl / ((1 + ir) ** bl - 1) / Q("1 a")


def calc_LCOX(
    data: dict[str, float | Q | pd.Series] | pd.DataFrame,
    name: str | None = None,
    reference: str | None = None,
    interest_rate: Optional[float | Q] = None,
    book_lifetime: Optional[Q] = None,
):
    # Get all relevant variables.
    ref = data.get(reference) if reference else 1.0
    interest_rate = interest_rate or data.get("Interest Rate")
    book_lifetime = (
        book_lifetime or data.get("Book Lifetime", data.get("Lifetime"))
    )
    capex = data.get("CAPEX")
    fopex = data.get("OPEX Fixed")
    vopex = data.get("OPEX Variable")
    ocf = data.get("OCF", 1.0)
    ghg_price = data.get("GHG Price")

    # Raise exception if reference is set, but data is undefined or zero.
    if reference is not None and ref is None:
        raise Exception("Reference is not provided in data: " + reference)

    # Define return dict.
    ret = {}

    # Process CAPEX and Fixed OPEX if provided.
    if capex is not None or fopex is not None:
        # Get reference capacity.
        cap_ref = cap_ref_io = None
        name_cap_ref = None
        for c in data:
            if match(r"^(Input|Output) Capacity\|.*$", c):
                name_cap_ref = c
                cap_ref = data[c]
                name_cap_ref_io = sub(
                    r"(Input|Output) Capacity",
                    r"\1",
                    name_cap_ref,
                )
                cap_ref_io = data.get(name_cap_ref_io)
                if cap_ref_io is not None:
                    continue
        if cap_ref is None:
            warn(
                "Could not find a reference capacity for CAPEX and/or Fixed "
                "OPEX."
            )
        elif cap_ref_io is None:
            warn(
                "Found reference capacity for CAPEX and/or Fixed "
                "OPEX, but no corresponding reference Input/Output: "
                + name_cap_ref
            )

        # Calculate Capital Cost and Fixed OM Cost.
        if cap_ref is not None and cap_ref_io is not None:
            if capex is not None:
                if interest_rate is not None and book_lifetime is not None:
                    if isinstance(interest_rate, float | int | str):
                        interest_rate = Q(interest_rate)
                    if isinstance(book_lifetime, str):
                        book_lifetime = Q(book_lifetime)
                    elif isinstance(book_lifetime, float | int):
                        book_lifetime = book_lifetime * Q("year")
                    anf = annuity_factor(interest_rate, book_lifetime)
                    ret["Capital"] = anf * capex / ocf / cap_ref * cap_ref_io
                else:
                    warn(
                        "Could not calculate capital cost due to missing "
                        "interest rate and/or book lifetime."
                    )

            if fopex is not None:
                ret["OM Fixed"] = fopex / ocf / cap_ref * cap_ref_io

    # Add variable OPEX.
    if vopex is not None:
        ret["OM Variable"] = vopex

    # Calculate Input Costs and Output Revenues.
    unused = []
    for io in data:
        # Loop only over Inputs/Outputs.
        if not match(r"^(Input|Output)\|.*", io):
            continue
        # Split variable up into type (Input/Output) and flow (Elec, H2, CH4, etc).
        io_type, io_flow = io.split("|", 2)
        # Get corresponding price for flow. If price not given, report as unused.
        io_price = data.get("Price|" + io_flow)
        if io_price is None:
            if io != reference:
                unused.append(io)
            continue
        # Inputs are counted as costs, outputs are counted as revenues.
        io_sign = +1 if io_type == "Input" else -1
        io_word = "Cost" if io_type == "Input" else "Revenue"
        # Combine type, word, and flow into new name.
        io_name = f"{io_type} {io_word}|{io_flow}"
        ret[io_name] = io_sign * data[io] * io_price

    # Warn about unused IO variables.
    if unused:
        warn(
            "The following inputs/outputs are not used in LCOX, "
            "because they are neither the reference nor is an "
            "associated price given: " + ", ".join(unused)
        )

    # Add Costs from CO2 pricing if provided.
    if ghg_price is not None:
        for emi in data:
            # Loop only over GHG Emissions.
            if not match(r"^GHG Emission\|.*", emi):
                continue
            # Split emissions type (CO2, CH4, N2O, etc) off from variable name.
            emi_type = emi.split("|", 2)[1]
            # Combine emi type into new name.
            emi_name = f"GHG Pricing|{emi_type}"
            ret[emi_name] = ghg_price * data[emi]

    # Add name to prefix.
    prefix = "LCOX|"
    if name is not None:
        prefix += name + "|"

    # Divide by reference before returning.
    return {(prefix+key): value / ref for key, value in ret.items()}


def calc_LCOX_pc(
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
        ret_proc = calc_LCOX(
            data.rename(columns=lambda c: c.removeprefix(prefix_tech)),
            **kwargs,
        )
        factor = data[prefix_scale + proc]
        if fu is not None:
            factor /= fu
        if name is not None:
            prefix_new = f"LCOX|{name}|{proc}|"
        else:
            prefix_new = f"LCOX|{proc}|"
        ret |= {
            (prefix_new + var_name.removeprefix("LCOX|")): var_value * factor
            for var_name, var_value in ret_proc.items()
        }
    return ret
