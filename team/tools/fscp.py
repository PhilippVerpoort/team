def calc_FSCP(
    data,
    options: list[str],
    var_cost: str = "LCOX",
    var_emi: str = "GHGI",
    var_out: str = "FSCP",
):
    ret = {}

    for id_x, fuel_x in enumerate(options):
        for id_y, fuel_y in enumerate(options):
            if id_x < id_y:
                ret[f"{var_out}|{fuel_x} to {fuel_y}"] = (
                    data[f"{var_cost}|{fuel_y}"] - data[f"{var_cost}|{fuel_x}"]
                ) / (data[f"{var_emi}|{fuel_x}"] - data[f"{var_emi}|{fuel_y}"])

    return ret
