"""Controls, the estimation sample, and the subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def build_controls(d, verbose=True):
    d = d.copy()
    used, labels = [], []
    for sp in C.CONTROL_SPEC:
        src = next((c for c in sp["src"] if c in d.columns), None)
        if src is None:
            msg = f"control '{sp['name']}': none of {sp['src']} present"
            if sp.get("required", False):
                raise KeyError(msg)
            if verbose:
                print(f"  !! {msg} - skipped")
            continue
        fn = C.TF[sp["tf"]] if isinstance(sp["tf"], str) else sp["tf"]
        x = pd.to_numeric(d[src], errors="coerce").astype(float)
        d[sp["name"]] = (pd.Series(fn(x), index=d.index)
                         .replace([np.inf, -np.inf], np.nan))
        if sp.get("use", True):
            used.append(sp["name"])
            labels.append(sp.get("label", sp["name"]))
    return d, used, labels


def controls_formula(used):
    return " + ".join(used) if used else "1"


def controls_spec(used):
    lab = {sp["name"]: sp.get("label", sp["name"]) for sp in C.CONTROL_SPEC}
    return " + ".join(lab.get(c, c) for c in used) if used else "1"


def controls_text(labels):
    return ", ".join(labels) if labels else "field fixed effects only"


def estimation_sample(papers, verbose=True, check=True):
    def say(*a):
        if verbose:
            print(*a)

    d = papers.copy()
    say(f"papers                            : {len(d):,}")

    n = len(d)
    d = d[d["doi_norm"].notna()].drop_duplicates("doi_norm")
    say(f"  1. duplicate / missing DOIs     : {n - len(d):,} -> {len(d):,}")

    d["seed_year"] = pd.to_numeric(d["seed_year"],
                                   errors="coerce").astype("Int64")
    n = len(d)
    d = d[d["seed_year"].isin(C.YEARS)].copy()
    d["seed_year"] = d["seed_year"].astype(int)
    say(f"  2. outside {C.YEARS[0]}-{C.YEARS[-1]}            : "
        f"{n - len(d):,} -> {len(d):,}")

    if "n_authors_bq" in d.columns:
        n = len(d)
        d = d[d["n_authors_bq"].notna()
              & (d["n_authors_bq"].astype(float) > 0)].copy()
        say(f"  3. no matched author list       : {n - len(d):,} "
            f"-> {len(d):,}")
    else:
        say("  3. no matched author list       : SKIPPED "
            "('n_authors_bq' absent)")

    d, used, labels = build_controls(d, verbose=False)
    for c in used:
        say(f"     {c:20s} missing for {int(d[c].isna().sum()):,}")
    n = len(d)
    d = d[d[used].notna().all(axis=1)].copy()
    say(f"  4. incomplete controls          : {n - len(d):,} -> {len(d):,}")

    if C.FE_COL in d.columns:
        n = len(d)
        d = d[d[C.FE_COL].notna()].copy()
        say(f"  5. no primary field             : {n - len(d):,} "
            f"-> {len(d):,}")

    say(f"\nESTIMATION SAMPLE : {len(d):,} papers")
    if check and C.EXPECTED_N:
        if len(d) == C.EXPECTED_N:
            say(f"  = the {C.EXPECTED_N:,} papers of the paper")
        else:
            say(f"  !! expected {C.EXPECTED_N:,}, got {len(d):,} "
                f"({len(d) - C.EXPECTED_N:+,}) - see the ladder above")
    say(f"  team_type : "
        f"{d['team_type'].value_counts(dropna=False).to_dict()}")
    say(f"  controls  : {controls_text(labels)}")
    return d, used, labels


def team_subset(d, types):
    return d if types is None else d[d["team_type"].isin(types)]


def outcome(d, kind, key):
    if kind == "share":
        return pd.to_numeric(d[f"share_grp_{key}"],
                             errors="coerce").astype(float)
    if kind == "count":
        return np.log(pd.to_numeric(d[f"n_grp_{key}"],
                                    errors="coerce").astype(float) + 1)
    if kind == "share14":
        return pd.to_numeric(d[f"share_{key}"],
                             errors="coerce").astype(float)
    if kind == "count14":
        return np.log(pd.to_numeric(d[f"n_{key}"],
                                    errors="coerce").astype(float) + 1)
    if kind == "metric":
        return pd.to_numeric(d[key], errors="coerce").astype(float)
    raise ValueError(f"unknown outcome kind {kind!r}")


def team_key(d, ids_col="author_ids"):
    from .build import parse_id_list
    if ids_col not in d.columns:
        raise KeyError(
            f"'{ids_col}' is not on this table. It is written by "
            "tools/prepare_dataset.py - rerun that, then "
            "load_papers(force=True).")
    return d[ids_col].map(
        lambda v: "|".join(sorted(set(parse_id_list(v)))) if v else "")


def spanning_teams(d, ids_col="author_ids", split_year=None, verbose=True):
    split_year = C.SHADE_FROM if split_year is None else split_year

    m = d[[C.YEAR_COL]].copy()
    m["team"] = team_key(d, ids_col)
    m = m[m["team"] != ""]
    m["is_pre"] = m[C.YEAR_COL] < split_year

    g = m.groupby("team")["is_pre"]
    t = pd.DataFrame({"papers": g.size(), "papers_pre": g.sum()})
    t["papers_post"] = t["papers"] - t["papers_pre"]
    t["n_authors"] = [len(k.split("|")) for k in t.index]

    both = (t["papers_pre"] > 0) & (t["papers_post"] > 0)
    s = dict(
        papers_with_ids=int(len(m)),
        papers_without_ids=int(len(d) - len(m)),
        distinct_teams=int(len(t)),
        teams_pre_only=int(((t["papers_pre"] > 0) & (t["papers_post"] == 0)).sum()),
        teams_post_only=int(((t["papers_pre"] == 0) & (t["papers_post"] > 0)).sum()),
        teams_both_sides=int(both.sum()),
        papers_in_spanning_teams=int(t.loc[both, "papers"].sum()),
        split_year=int(split_year),
    )
    if verbose:
        print(f"papers with author ids      : {s['papers_with_ids']:,}"
              f"   (without: {s['papers_without_ids']:,})")
        print(f"distinct team compositions  : {s['distinct_teams']:,}")
        print(f"  only before {split_year}        : {s['teams_pre_only']:,}")
        print(f"  only {split_year} onwards       : {s['teams_post_only']:,}")
        print(f"  on BOTH sides of {split_year}   : {s['teams_both_sides']:,}"
              f"   ({s['papers_in_spanning_teams']:,} papers)")
    return s, t.sort_values("papers", ascending=False)


def collab_band_subsets(d, pure_new=None, verbose=True):
    col = C.COLLAB_COL
    if col not in d.columns:
        raise KeyError(
            f"'{col}' is not on the table. It comes from team_types_flex "
            "via tools/prepare_dataset.py - rerun that, then "
            "load_papers(force=True).")
    pure_new = C.PURE_NEW if pure_new is None else pure_new
    if pure_new not in ("first", "separate", "drop"):
        raise ValueError(f"pure_new={pure_new!r}")

    r = pd.to_numeric(d[col], errors="coerce")
    hy = d["team_type"].eq("hybrid")
    pure = d["team_type"].eq("new")

    out = []
    if pure_new == "separate":
        out.append(("Pure new", d[pure]))
    for i, (lo, hi, label) in enumerate(C.COLLAB_BANDS):
        m = hy & r.between(lo, hi, inclusive="left").fillna(False)
        if pure_new == "first" and i == 0:
            m = m | pure
        out.append((label, d[m]))

    if verbose:
        hi_max = max(h for _lo, h, _l in C.COLLAB_BANDS)
        n_above = int((hy & (r >= hi_max)).sum())
        n_nan = int((hy & r.isna()).sum())
        print(f"hybrid teams {int(hy.sum()):,} | purely new "
              f"{int(pure.sum()):,} ({pure_new})")
        for label, sub in out:
            print(f"  {label:10s} n = {len(sub):,}")
        print(f"  excluded: {n_above:,} hybrids at {hi_max:.0%} or more, "
              f"{n_nan:,} without a ratio")
    return out
