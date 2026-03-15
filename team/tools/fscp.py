from re import match
from warnings import warn


def calc_FSCP(
    data,
    options: list[str] | None = None,
    var_cost: str = "LCOX",
    var_ghgi: str = "GHGI",
    var_out: str = "FSCP",
):
    ret = {}

    if options is None:
        options_cost = [m.group(1) for var in data if (m:=match(rf"{var_cost}\|(.*)", var))]
        options_ghgi = [m.group(1) for var in data if (m:=match(rf"{var_ghgi}\|(.*)", var))]

        options_cost_notin_ghgi = [o for o in options_cost if o not in options_ghgi]
        options_ghgi_notin_cost = [o for o in options_ghgi if o not in options_cost]

        if options_cost_notin_ghgi:
            warn(
                "The following options were found as cost but not as GHGI: "
                + ", ".join(options_cost_notin_ghgi)
            )
        if options_ghgi_notin_cost:
            warn(
                "The following options were found as GHGI but not as cost: "
                + ", ".join(options_ghgi_notin_cost)
            )

        options = [o for o in options_cost if o in options_ghgi]


    for id_x, fuel_x in enumerate(options):
        for id_y, fuel_y in enumerate(options):
            if id_x < id_y:
                ret[f"{var_out}|{fuel_x} to {fuel_y}"] = (
                    (data[f"{var_cost}|{fuel_y}"] - data[f"{var_cost}|{fuel_x}"])
                  / (data[f"{var_ghgi}|{fuel_x}"] - data[f"{var_ghgi}|{fuel_y}"])
                )

    return ret
