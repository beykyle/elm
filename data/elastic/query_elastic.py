import exfor_tools

# settings consistent across data set
energy_range = [10, 200]


def reattempt_parse(failed_parse, parsing_kwargs):
    return exfor_tools.ExforEntryAngularDistribution(
        entry=failed_parse.entry,
        target=failed_parse.target,
        projectile=failed_parse.projectile,
        quantity=failed_parse.quantity,
        Einc_range=energy_range,
        vocal=True,
        parsing_kwargs=parsing_kwargs
    )


def print_failed_parses(failed_parses):
    for k, v in failed_parses.items():
        print(f"Entry: {k}")
        print(v.failed_parses[k][1])


def query_elastic_data(target):
    print("\n========================================================")
    print("Parsing (p,p) ...")
    print("========================================================")
    entries_pp, failed_parses_pp = exfor_tools.query_for_entries(
        target=target,
        projectile=(1, 1),
        quantity="dXS/dA",
        Einc_range=energy_range,
        vocal=True,
        filter_subentries=exfor_tools.filter_out_lab_angle,
    )
    print("\n========================================================")
    print(f"Succesfully parsed {len(entries_pp.keys())} entries for (p,p)")
    print(f"Failed to parse {len(failed_parses_pp.keys())} entries")
    print("========================================================\n\n")

    print("\n========================================================")
    print("Parsing (p,p) ratio ...")
    print("========================================================")
    entries_ppr, failed_parses_ppr = exfor_tools.query_for_entries(
        target=target,
        projectile=(1, 1),
        quantity="dXS/dRuth",
        Einc_range=energy_range,
        vocal=True,
        filter_subentries=exfor_tools.filter_out_lab_angle,
    )
    print("\n========================================================")
    print(f"Succesfully parsed {len(entries_ppr.keys())} entries for (p,p) ratio")
    print(f"Failed to parse {len(failed_parses_ppr.keys())} entries")
    print("========================================================\n\n")

    print("\n========================================================")
    print("Parsing (n,n)...")
    print("========================================================")
    entries_nn, failed_parses_nn = exfor_tools.query_for_entries(
        target=target,
        projectile=(1, 0),
        quantity="dXS/dA",
        Einc_range=energy_range,
        vocal=True,
        filter_subentries=exfor_tools.filter_out_lab_angle,
    )
    print("\n========================================================")
    print(f"Succesfully parsed {len(entries_nn.keys())} entries for (n,n)")
    print(f"Failed to parse {len(failed_parses_nn.keys())} entries")
    print("========================================================\n\n")

    return (
        (entries_pp, failed_parses_pp),
        (entries_ppr, failed_parses_ppr),
        (entries_nn, failed_parses_nn),
    )
