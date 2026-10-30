"""Reproduce every result in the paper, block by block."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from analysis import config as C
from analysis import figures as F
from analysis import tables as T
from analysis.build import load_papers
from analysis.estimation import run_event_studies
from analysis.sample import (collab_band_subsets, estimation_sample,
                             team_subset)

ALL_RESULTS = []


def banner(txt):
    print("\n" + "=" * 72)
    print(txt)
    print("=" * 72)


def collect(results, **meta):
    for es in results.values():
        if es is not None:
            ALL_RESULTS.append({**es.row(), **meta})


def block0(rebuild=False):
    banner("BLOCK 0  data and estimation sample")
    C.print_inputs()
    papers = load_papers(force=rebuild)
    d, controls, labels = estimation_sample(papers)
    return d, controls, labels


def block1(d):
    banner("BLOCK 1  sample composition")

    print("\n-- Table 1: summary statistics")
    tab1 = T.summary_table(d)
    print(tab1.to_string(index=False))

    print("\n-- Figure A.1: publications and authors by year")
    comp = F.sample_composition(d)
    print(comp.round(2).to_string(index=False))

    print("\n-- Table A.1: papers by primary domain and year")
    tabA1 = T.year_cross_table(
        d, "paper_primary_domain", "tableA1_domain_by_year",
        "Distribution of papers by primary domain and year.",
        "tab:domains_by_year")

    print("\n-- Table A.2: papers by primary field and year")
    tabA2 = T.year_cross_table(
        d, C.FE_COL, "tableA2_field_by_year",
        "Distribution of papers by primary field and year.",
        "tab:fields_by_year")

    print("\n-- Figure: team size over time")
    ts = F.team_size_over_time(d)
    print(ts.pivot_table(index="year", columns="group_label",
                         values="mean").round(3).to_string())

    return {"table1": tab1, "tableA1": tabA1, "tableA2": tabA2,
            "composition": comp, "team_size": ts}


def block2(d, controls):
    banner("BLOCK 2  H1 task participation and H2 portfolio specialisation")

    stage_specs = {k: [(k, g, C.STAGE_LABELS[g]) for g in C.STAGE_KEYS]
                   for k in ("share", "count")}
    metric_specs = [("metric", col, lab) for col, lab in C.METRICS]

    es_h1 = {}
    for kind in ("share", "count"):
        print(f"\n-- H1 {C.OUTCOME_NAME[kind]}, all teams")
        res = run_event_studies(d, stage_specs[kind], controls)
        es_h1[kind] = res
        collect(res, group="all", block="H1")
        for view in ("levels", "detrended"):
            F.grid_figure(res, stage_specs[kind], view,
                          f"H1_{kind}_all_{view}.png", ncol=3)

    print("\n-- H2 portfolio specialisation, all teams")
    res = es_h2 = run_event_studies(d, metric_specs, controls)
    collect(res, group="all", block="H2")
    for view in ("levels", "detrended"):
        F.grid_figure(res, metric_specs, view, f"H2_metrics_all_{view}.png",
                      ncol=2, panel_w=6.75, panel_h=5.4)

    print("\n-- Table B.1: static post-LLM estimates by task domain")
    tb1 = T.static_table(d, controls, "tableB1_static_by_stage",
                         r"\textbf{Static baseline estimates of post-LLM "
                         r"shifts in task participation.}",
                         "tab:static_stage")
    print(tb1.round(4).to_string(index=False))

    return {"tableB1": tb1, "H1": es_h1, "H2": es_h2}


def block3(d, controls):
    banner("BLOCK 3  H3 team history and the reallocation of participation")

    stage_specs = {k: [(k, g, C.STAGE_LABELS[g]) for g in C.STAGE_KEYS]
                   for k in ("share", "count")}
    metric_specs = [("metric", col, lab) for col, lab in C.METRICS]

    es_h3 = {}
    for tag, types, label in C.TEAM_GROUPS:
        if tag == "all":
            continue
        sub = team_subset(d, types)
        es_h3[tag] = {}
        print(f"\n########## {label}  (n={len(sub):,}) ##########")

        for kind in ("share", "count"):
            print(f"\n-- {C.OUTCOME_NAME[kind]}")
            res = run_event_studies(sub, stage_specs[kind], controls)
            es_h3[tag][kind] = res
            collect(res, group=tag, block="H3")

        print("\n-- portfolio specialisation")
        res = es_h3[tag]["metrics"] = run_event_studies(
            sub, metric_specs, controls)
        collect(res, group=tag, block="H3")


    lab = {t: l for t, _ty, l in C.TEAM_GROUPS}
    n = {t: len(team_subset(d, ty)) for t, ty, _l in C.TEAM_GROUPS}
    title_a = f"Panel A. {lab['existing']} (n={n['existing']:,})"
    title_b = f"Panel B. {lab['new']} (n={n['new']:,})"

    for kind in ("share", "count"):
        for view in ("levels", "detrended"):
            F.panel_ab_figure(es_h3["existing"][kind], es_h3["new"][kind],
                              stage_specs[kind], view,
                              f"H3_{kind}_teamtype_{view}.png",
                              title_a, title_b, ncol=3,
                              col_in=4.8, row_in=3.2, ms=18)
    for view in ("levels", "detrended"):
        F.panel_ab_figure(es_h3["existing"]["metrics"],
                          es_h3["new"]["metrics"], metric_specs, view,
                          f"H3_metrics_teamtype_{view}.png",
                          title_a, title_b)

    print("\n-- Table B.2: participation, Post x New")
    tb2 = T.static_table(d, controls, "tableB2_participation_interactions",
                         r"\textbf{Post-LLM changes in task participation "
                         r"by team type.}",
                         "tab:participations_interactions", with_new=True)
    print(tb2.round(4).to_string(index=False))

    print("\n-- Table B.3: specialisation, Post x New")
    tb3 = T.specialisation_interaction_table(
        d, controls, "tableB3_specialisation_interactions",
        r"\textbf{Post-LLM changes in task portfolios and overlap by team "
        r"type.}", "tab:specialization_interactions")
    print(tb3.round(4).to_string(index=False))

    return {"tableB2": tb2, "tableB3": tb3, "by_group": es_h3}


def block4(d, controls):
    banner("BLOCK 4  the 14 CRediT contributor roles")

    specs_a = [("share14", r, C.ROLE_LABELS[r]) for r in C.ROLE_KEYS]
    specs_b = [("metric", col, lab) for col, lab in C.METRICS14]

    out = {}
    for tag, types, label in C.TEAM_GROUPS:
        sub = team_subset(d, types)
        print(f"\n########## {label}  (n={len(sub):,}) ##########")
        print("\n-- Panel A: the 14 roles, contributor share")
        res_a = run_event_studies(sub, specs_a, controls)
        collect(res_a, group=tag, block="14roles")
        print("\n-- Panel B: portfolio specialisation over the 14 roles")
        res_b = run_event_studies(sub, specs_b, controls)
        collect(res_b, group=tag, block="14roles")
        out[tag] = {"roles": res_a, "metrics": res_b}
        for view in ("levels", "detrended"):
            F.tasks14_figure(res_a, specs_a, res_b, specs_b, view,
                             f"B7_tasks14_{tag}_{view}.png")

    return out


def block5(d, controls):
    banner("BLOCK 5  departures from pre-LLM trends across fields")
    tab = T.field_break_table(d, controls)
    cols = ["field", "N"] + [f"{s}_mark" for s in C.STAGE_KEYS] \
        + [f"{c}_mark" for c, _l in C.METRICS]
    print(tab[cols].to_string(index=False))
    return tab


def save_results():
    if not ALL_RESULTS:
        print("nothing to save yet - run some blocks first")
        return None
    out = C.OUT_DIR / "event_studies.csv"
    log = pd.DataFrame(ALL_RESULTS)
    log.to_csv(out, index=False)
    print(f"saved {out}  ({len(log):,} event studies)")
    return log


def block6(d, controls):
    banner("BLOCK 6  newly formed teams by prior-collaboration band")

    try:
        bands = collab_band_subsets(d)
    except KeyError as e:
        print(f"    !! {e}")
        return None

    stage_specs = [("share", g, C.STAGE_LABELS[g]) for g in C.STAGE_KEYS]
    metric_specs = [("metric", col, lab) for col, lab in C.METRICS]

    labels = [lab for lab, _sub in bands]
    ns = {lab: len(sub) for lab, sub in bands}
    res_a, res_b = {}, {}
    for lab, sub in bands:
        print(f"\n########## {lab}  (n={len(sub):,}) ##########")
        print("\n-- task participation")
        res_a[lab] = run_event_studies(sub, stage_specs, controls)
        collect(res_a[lab], group=f"band {lab}", block="C")
        print("\n-- portfolio specialisation")
        res_b[lab] = run_event_studies(sub, metric_specs, controls)
        collect(res_b[lab], group=f"band {lab}", block="C")

    F.bands_figure(res_a, stage_specs, "C1_bands_participation.png",
                   labels, ns, ncol=3)
    F.bands_figure(res_b, metric_specs, "C2_bands_specialisation.png",
                   labels, ns, ncol=2)
    return {"bands": bands, "participation": res_a, "specialisation": res_b}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="*", type=int,
                    default=[1, 2, 3, 4, 5, 6],
                    help="which blocks to run (default: all)")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-derive the cached data from the raw files")
    a = ap.parse_args()

    t0 = time.time()
    d, controls, labels = block0(rebuild=a.rebuild)

    if 1 in a.blocks:
        block1(d)
    if 2 in a.blocks:
        block2(d, controls)
    if 3 in a.blocks:
        block3(d, controls)
    if 4 in a.blocks:
        block4(d, controls)
    if 5 in a.blocks:
        block5(d, controls)
    if 6 in a.blocks:
        block6(d, controls)

    save_results()

    banner(f"done in {time.time() - t0:.0f}s   "
           f"figures -> {C.FIG_DIR}   tables -> {C.TAB_DIR}")


if __name__ == "__main__":
    main()
