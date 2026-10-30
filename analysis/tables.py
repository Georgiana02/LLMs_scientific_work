"""Every table in the paper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from .estimation import (event_study, static_model, stars)
from .sample import build_controls, team_subset


def write_tex(lines, name, verbose=True):
    path = C.TAB_DIR / name
    path.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print(f"    saved {path.name}")
    return path


def esc(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _ms(x, dec):
    v = pd.to_numeric(x, errors="coerce").dropna()
    if len(v) == 0:
        return "--"
    return f"{v.mean():.{dec}f} ({v.std(ddof=1):.{dec}f})"


def _m(x, dec):
    v = pd.to_numeric(x, errors="coerce").dropna()
    return "--" if len(v) == 0 else f"{v.mean():.{dec}f}"


def summary_table(d, name="table1_summary_stats", verbose=True):
    panels = [(f"Panel A: Pre-LLM Baseline Period "
               f"({C.PANEL_PRE[0]}--{C.PANEL_PRE[-1]})", C.PANEL_PRE),
              (f"Panel B: Post-LLM Diffusion Period "
               f"({C.PANEL_POST[0]}--{C.PANEL_POST[-1]})", C.PANEL_POST)]
    ctrl_rows = [("Team Size", "team_size", 2),
                 ("Intl. Collaboration", "iic", 2),
                 ("Prior Pubs", "avg_pubs_per_author", 2),
                 ("Academic Age", "avg_academic_age", 2)]

    rows = []
    for title, yrs in panels:
        dp = d[d[C.YEAR_COL].isin(yrs)]
        rows.append(("PANEL", title, None))
        rows.append(("SECTION", r"Task Participation Outcomes ($P_{kpt}$)",
                     None))
        for g in C.STAGE_KEYS:
            cells = []
            for _tag, types, _lab in C.TEAM_GROUPS:
                s = team_subset(dp, types)
                cells += [_ms(s[f"n_grp_{g}"], 2),
                          _m(s[f"share_grp_{g}"], 3)]
            rows.append(("ROW", C.STAGE_LABELS[g], cells))

        rows.append(("SECTION", "Portfolio Specialisation Outcomes", None))
        for col, lab in C.METRICS:
            cells = []
            for _tag, types, _l in C.TEAM_GROUPS:
                s = team_subset(dp, types)
                cells += [_ms(s[col], 3), "--"]
            rows.append(("ROW", lab, cells))

        rows.append(("SECTION",
                     r"Control Variables ($\mathbf{X}_{pt}$)", None))
        for lab, col, dec in ctrl_rows:
            if col not in dp.columns:
                continue
            cells = []
            for _tag, types, _l in C.TEAM_GROUPS:
                s = team_subset(dp, types)
                cells += [_ms(s[col], dec), "--"]
            rows.append(("ROW", lab, cells))

        cells = []
        for _tag, types, _l in C.TEAM_GROUPS:
            cells += [f"{len(team_subset(dp, types)):,}", "--"]
        rows.append(("ROW", r"Observations ($N$)", cells))

    flat = []
    for kind, lab, cells in rows:
        flat.append([lab] + (cells if cells else [""] * 6))
    cols = ["Variable"] + [f"{g} {w}" for _t, _ty, g in C.TEAM_GROUPS
                           for w in ("Mean (SD)", "Share")]
    tab = pd.DataFrame(flat, columns=cols)
    tab.to_csv(C.TAB_DIR / f"{name}.csv", index=False)

    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{8pt}",
         r"\renewcommand{\arraystretch}{1.25}",
         r"\begin{tabular}{l cc cc cc}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}}"
                            for _t, _ty, lab in C.TEAM_GROUPS) + r" \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         "Variable & " + " & ".join(["Mean (SD) & Share"] * 3) + r" \\",
         r"\midrule"]
    first_panel = True
    for kind, lab, cells in rows:
        if kind == "PANEL":
            if not first_panel:
                L.append(r"\addlinespace[1.0em]")
            first_panel = False
            L.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{lab}}}}} \\")
        elif kind == "SECTION":
            L.append(r"\addlinespace[0.65em]")
            L.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{lab}}}}} \\")
            L.append(r"\addlinespace[0.25em]")
        else:
            if lab.startswith("Observations"):
                L.append(r"\addlinespace[0.55em]")
                cells = [cells[0], "", cells[2], "", cells[4], ""]
            L.append(r"\quad " + lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Summary statistics across team types and time "
          r"periods.}}", r"\label{tab:summary_stats}", r"\end{table}"]
    write_tex(L, f"{name}.tex", verbose)
    return tab


def year_cross_table(d, col, name, caption, label, verbose=True):
    if col not in d.columns:
        print(f"    !! '{col}' absent - {name} skipped")
        return None
    t = pd.crosstab(d[col], d[C.YEAR_COL]).reindex(columns=C.YEARS,
                                                   fill_value=0)
    t["Total"] = t.sum(axis=1)
    t = t.sort_values("Total", ascending=False)
    t.loc["Total"] = t.sum()
    t.to_csv(C.TAB_DIR / f"{name}.csv")

    L = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{5pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "r" * (len(C.YEARS) + 1) + "}",
         r"\toprule",
         "Field & " + " & ".join(str(y) for y in C.YEARS) + r" & Total \\",
         r"\midrule"]
    for ix, row in t.iterrows():
        if ix == "Total":
            L.append(r"\midrule")
        L.append(esc(ix) + " & "
                 + " & ".join(f"{int(v):,}" for v in row.values) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", "}",
          rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    write_tex(L, f"{name}.tex", verbose)
    return t


def _r2cell(v, dec=4):
    return "" if v is None or pd.isna(v) else f"{v:.{dec}f}"


def _cell(v, dec=3):
    if v is None:
        return "---"
    b, _se, p = v
    mark = stars(p)
    s = f"{b:.{dec}f}"
    return f"{s}{mark}" if mark else s


def static_table(d, controls, name, caption, label, with_new=False,
                 verbose=True):
    samples = [C.PERIODS_FULL, C.PERIODS_CLEAN]
    outcomes = [("share", r"\textbf{Share}"), ("count", r"\textbf{Count}")]
    terms = ("post", "new", "post_new") if with_new else ("post",)
    term_lab = {"post": r"$\text{Post}$", "new": r"$\text{New}$",
                "post_new": r"$\text{Post} \times \text{New}$"}

    res, nobs, r2, rows = {}, {}, {}, []
    for s in samples:
        ds = d[d[C.YEAR_COL].isin(s["years"])]
        for kind, _ol in outcomes:
            for g in C.STAGE_KEYS:
                r = static_model(ds, kind, g, controls, s["post_from"],
                                 terms=terms)
                if r is None:
                    continue
                res[(s["label"], kind, g)] = r["est"]
                nobs[(s["label"], kind)] = r["N"]
                r2[(s["label"], kind, g)] = r.get("r2", np.nan)
                for t, (b, se, p) in r["est"].items():
                    rows.append(dict(sample=s["label"], outcome=kind,
                                     stage=g, label=C.STAGE_LABELS[g],
                                     term=t, beta=b, se=se, p=p,
                                     mark=stars(p, tex=False), N=r["N"],
                                     r2=r.get("r2", np.nan)))
    tab = pd.DataFrame(rows)
    tab.to_csv(C.TAB_DIR / f"{name}.csv", index=False)

    ncol = len(samples) * len(outcomes)
    L = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{8pt}",
         r"\renewcommand{\arraystretch}{1.3}",
         r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{s['label']}}}}}"
                            for s in samples) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}"
                  for i in range(len(samples))),
         r"\textbf{Task Domain} & "
         + " & ".join(ol for _s in samples for _o, ol in outcomes) + r" \\",
         r"\midrule"]

    for bi, g in enumerate(C.STAGE_KEYS):
        if with_new:
            if bi:
                L.append(r"\addlinespace")
            L.append(rf"\multicolumn{{{ncol + 1}}}{{l}}"
                     rf"{{\textbf{{{C.STAGE_LABELS[g]}}}}} \\")
            for t in terms:
                cells = [_cell(res.get((s["label"], k, g), {}).get(t))
                         for s in samples for k, _ol in outcomes]
                L.append(r"\hspace{1em}" + term_lab[t] + " & "
                         + " & ".join(cells) + r" \\")
            L.append(r"\hspace{1em}$R^2$ & "
                     + " & ".join(_r2cell(r2.get((s["label"], k, g)))
                                  for s in samples for k, _ol in outcomes)
                     + r" \\")
        else:
            cells = [_cell(res.get((s["label"], k, g), {}).get("post"))
                     for s in samples for k, _ol in outcomes]
            L.append(C.STAGE_LABELS[g] + " & " + " & ".join(cells) + r" \\")
            L.append(r"\hspace{1em}\textit{$R^2$} & "
                     + " & ".join(_r2cell(r2.get((s["label"], k, g)))
                                  for s in samples for k, _ol in outcomes)
                     + r" \\")

    L.append(r"\midrule")
    L.append("Controls & " + " & ".join(["Yes"] * ncol) + r" \\")
    L.append("Field fixed effects & " + " & ".join(["Yes"] * ncol) + r" \\")
    L.append("Observations & "
             + " & ".join(f"{nobs.get((s['label'], k), 0):,}"
                          for s in samples for k, _ol in outcomes) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption} {C.SIG_KEY_TEX}}}",
          rf"\label{{{label}}}", r"\end{table}"]
    write_tex(L, f"{name}.tex", verbose)
    return tab


def specialisation_interaction_table(d, controls, name, caption, label,
                                     verbose=True):
    samples = [C.PERIODS_FULL, C.PERIODS_CLEAN]
    terms = ("post", "new", "post_new")
    term_lab = {"post": "Post", "new": "New",
                "post_new": r"Post $\times$ New"}
    res, nobs, r2, rows = {}, {}, {}, []
    for s in samples:
        ds = d[d[C.YEAR_COL].isin(s["years"])]
        for col, lab in C.METRICS:
            r = static_model(ds, "metric", col, controls, s["post_from"],
                             terms=terms)
            if r is None:
                continue
            res[(s["label"], col)] = r["est"]
            nobs[(s["label"], col)] = r["N"]
            r2[(s["label"], col)] = r.get("r2", np.nan)
            for t, (b, se, p) in r["est"].items():
                rows.append(dict(sample=s["label"], metric=col, label=lab,
                                 term=t, beta=b, se=se, p=p,
                                 mark=stars(p, tex=False), N=r["N"],
                                 r2=r.get("r2", np.nan)))
    tab = pd.DataFrame(rows)
    tab.to_csv(C.TAB_DIR / f"{name}.csv", index=False)

    cols = [(s, col, lab) for s in samples for col, lab in C.METRICS]
    ncol = len(cols)
    L = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
         r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{s['label']}}}"
                            for s in samples) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + i * 2}-{3 + i * 2}}}"
                  for i in range(len(samples))),
         "Term & " + " & ".join(lab for _s, _c, lab in cols) + r" \\",
         r"\midrule"]
    for t in terms:
        vals = [res.get((s["label"], col), {}).get(t) for s, col, _l in cols]
        L.append(term_lab[t] + " & "
                 + " & ".join(_cell(v, 4) for v in vals) + r" \\")
        L.append(" & " + " & ".join("" if v is None else f"({v[1]:.4f})"
                                    for v in vals) + r" \\")
    L.append(r"\midrule")
    L.append("Controls & " + " & ".join(["Yes"] * ncol) + r" \\")
    L.append("Field fixed effects & " + " & ".join(["Yes"] * ncol) + r" \\")
    L.append("Observations & "
             + " & ".join(f"{nobs.get((s['label'], col), 0):,}"
                          for s, col, _l in cols) + r" \\")
    L.append("$R^2$ & " + " & ".join(_r2cell(r2.get((s["label"], col)))
                                     for s, col, _l in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{{caption} {C.SIG_KEY_TEX}}}",
          rf"\label{{{label}}}", r"\end{table}"]
    write_tex(L, f"{name}.tex", verbose)
    return tab


def field_break_table(d, controls, name="table2_field_breaks",
                      min_papers=500, min_per_year=10, verbose=True):
    fields = []
    for f, g in d.groupby(C.FE_COL):
        if len(g) < min_papers:
            continue
        per_year = g.groupby(C.YEAR_COL).size().reindex(C.YEARS).fillna(0)
        if per_year.min() < min_per_year:
            continue
        fields.append((f, len(g)))
    fields.sort(key=lambda t: -t[1])
    if verbose:
        print(f"    fields kept: {len(fields)} "
              f"(>= {min_papers} papers, >= {min_per_year}/year)")

    rows = []
    for f, n in fields:
        g = d[d[C.FE_COL] == f]
        rec = {"field": f, "N": n}
        for st in C.STAGE_KEYS:
            ps = []
            for kind in ("share", "count"):
                es = event_study(g, kind, st, C.STAGE_LABELS[st], controls,
                                 verbose=False)
                ps.append(es.break_p if es else np.nan)
                rec[f"{st}_{kind}_p"] = ps[-1]
            best = np.nanmin(ps) if not all(pd.isna(p) for p in ps) else np.nan
            rec[f"{st}_p"] = best
            rec[f"{st}_mark"] = stars(best, tex=False)
        for col, lab in C.METRICS:
            es = event_study(g, "metric", col, lab, controls, verbose=False)
            p = es.break_p if es else np.nan
            rec[f"{col}_p"] = p
            rec[f"{col}_mark"] = stars(p, tex=False)
        rows.append(rec)
        if verbose:
            marks = "".join("+" if rec.get(f"{s}_p", 1) < .05 else
                            ("." if rec.get(f"{s}_p", 1) < C.P_DAGGER else " ")
                            for s in C.STAGE_KEYS)
            print(f"      {f[:42]:42s} n={n:>7,}  [{marks}]")

    tab = pd.DataFrame(rows)
    tab.to_csv(C.TAB_DIR / f"{name}.csv", index=False)

    def mark(p):
        if pd.isna(p):
            return ""
        if p < .05:
            return r"\checkmark"
        if p < C.P_DAGGER:
            return r"$\dagger$"
        return ""

    short = {"ideation": "Ideation", "data_and_coding": "Data",
             "execution": "Exec.", "communication": "Writing",
             "coordination": "Coord.", "funding_resources": "Funding"}
    ncol = len(C.STAGE_KEYS) + len(C.METRICS)
    L = [r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{6pt}",
         r"\renewcommand{\arraystretch}{1.15}",
         r"\begin{tabular}{lr" + "c" * len(C.STAGE_KEYS)
         + " " + "c" * len(C.METRICS) + "}", r"\toprule",
         rf" & $N$ & \multicolumn{{{len(C.STAGE_KEYS)}}}{{c}}"
         rf"{{A. Within-stage participation}} & "
         rf"\multicolumn{{{len(C.METRICS)}}}{{c}}"
         rf"{{B. Portfolio specialisation}} \\",
         rf"\cmidrule(lr){{3-{2 + len(C.STAGE_KEYS)}}} "
         rf"\cmidrule(lr){{{3 + len(C.STAGE_KEYS)}-{2 + ncol}}}",
         "Field & & " + " & ".join([short[s] for s in C.STAGE_KEYS]
                                   + [lab for _c, lab in C.METRICS])
         + r" \\", r"\midrule"]
    for _, r in tab.iterrows():
        cells = [mark(r[f"{s}_p"]) for s in C.STAGE_KEYS] + \
                [mark(r[f"{c}_p"]) for c, _l in C.METRICS]
        L.append(f"{esc(r['field'])} & {int(r['N']):,} & "
                 + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Significant post-LLM trend departures by "
          r"field.} A checkmark (\checkmark) indicates rejection at the 5\% "
          r"level, and a dagger ($\dagger$) rejection at the 10\% level but "
          r"not at 5\%, of the joint null that the 2023--2025 estimates "
          r"follow the extrapolated linear 2018--2021 trend. In Panel A a "
          r"mark requires a departure in the contributor share or in the "
          r"log contributor count; either one suffices. Tests are "
          r"two-sided.}",
          r"\label{tab:field_breaks}", r"\end{table}"]
    write_tex(L, f"{name}.tex", verbose)
    return tab


def coef_table(log, block=None, kind=None, group=None, years=None,
               pct=False, se=True, name=None, caption=None, label=None,
               verbose=True):
    d = log.copy()
    for col, val in (("block", block), ("kind", kind), ("group", group)):
        if val is not None and col in d.columns:
            d = d[d[col] == val]
    if d.empty:
        print("    !! nothing in the log matches that selection")
        return None

    years = [y for y in (years or C.YEARS) if y != C.REF_YEAR
             and f"b_{y}" in d.columns]
    k = 100.0 if pct else 1.0

    out = pd.DataFrame({"outcome": d["label"].values})
    if group is None and "group" in d.columns:
        out["group"] = d["group"].values
    out["N"] = d["N"].values

    for y in years:
        b = d[f"b_{y}"].values * k
        s = d[f"se_{y}"].values * k
        p = 2 * norm.sf(np.abs(np.divide(b, s, out=np.full_like(b, np.nan),
                                         where=s > 0)))
        out[f"{y}"] = [f"{v:.3f}{stars(pv, tex=False)}"
                       for v, pv in zip(b, p)]
        if se:
            out[f"({y})"] = [f"({v:.3f})" for v in s]

    if f"dev_{max(years)}" in d.columns:
        out[f"dev {max(years)}"] = (d[f"dev_{max(years)}"].values * k).round(3)
    out["pre p"] = d["pre_p"].values.round(4)
    out["post p"] = d["post_p"].values.round(4)
    out["break p"] = d["break_p"].values.round(4)

    if name:
        out.to_csv(C.TAB_DIR / f"{name}.csv", index=False)
        head = [c for c in out.columns]
        L = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}",
             r"\toprule",
             " & ".join(esc(h) for h in head) + r" \\", r"\midrule"]
        for _i, row in out.iterrows():
            L.append(" & ".join(esc(str(v)) for v in row.values) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        if caption:
            L.append(rf"\caption{{{caption}}}")
        if label:
            L.append(rf"\label{{{label}}}")
        L.append(r"\end{table}")
        write_tex(L, f"{name}.tex", verbose)
    return out


def headline_table(log, block=None, kind=None, group=None, year=None,
                   pct=None, dec=None, name=None, verbose=True):
    d = log.copy()
    for col, val in (("block", block), ("kind", kind), ("group", group)):
        if val is not None and col in d.columns:
            d = d[d[col] == val]
    if d.empty:
        print("    !! nothing in the log matches that selection")
        return None

    year = year or max(C.POST_YEARS)
    if pct is None:
        pct = bool(kind) and kind != "metric"
    k, lab_b, lab_d = ((100.0, "pp_vs_2022", "pp_vs_trend") if pct
                       else (1.0, f"b_{year}", f"dev_{year}"))


    dec = (1 if pct else 3) if dec is None else dec

    out = pd.DataFrame({"label": d["label"].values})
    if group is None and "group" in d.columns:
        out["group"] = d["group"].values
    out["N"] = d["N"].values
    out[lab_b] = (d[f"b_{year}"].values * k).round(dec)
    out[lab_d] = (d[f"dev_{year}"].values * k).round(dec)
    out["pre_p"] = d["pre_p"].values.round(4)
    out["post_p"] = d["post_p"].values.round(4)
    out["break_p"] = d["break_p"].values.round(4)
    out[f"p_dev_{year}"] = d[f"p_dev_{year}"].values.round(4)

    if name:
        out.to_csv(C.TAB_DIR / f"{name}.csv", index=False)
        if verbose:
            print(f"    saved {name}.csv")
    return out.reset_index(drop=True)
