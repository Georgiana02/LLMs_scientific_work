"""Every figure in the paper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from . import config as C
from .estimation import pstr, stars

matplotlib.rcParams.update(C.RCPARAMS)

STAT_SIZE = 6.6
STAT_X, STAT_Y = 0.022, 0.030
STAT_SEP = " · "


def colour_for(kind):
    if kind.startswith("share"):
        return C.COL_SHARE
    if kind.startswith("count"):
        return C.COL_COUNT
    return C.COL_SPEC


def _frame(ax):
    ax.axvspan(C.SHADE_FROM - 0.5, max(C.YEARS) + 0.6, color=C.SHADE,
               lw=0, zorder=0)
    ax.axvline(C.SHADE_FROM - 0.5, ls=(0, (2, 2)), color="0.55", lw=1.0,
               zorder=1)
    ax.axhline(0, color=C.ZERO, lw=1.0, zorder=1)
    ax.set_xticks(C.YEARS)
    ax.set_xlim(min(C.YEARS) - 0.6, max(C.YEARS) + 0.6)
    ax.tick_params(axis="x", rotation=45)


def _ylim(es, view, stats_here):
    if view == "levels":
        lo = [es.coefs["low"].min()]
        hi = [es.coefs["high"].max()]
        if es.brk:
            pv = np.array([es.brk["proj"][y] for y in C.YEARS])
            lo.append(pv.min())
            hi.append(pv.max())
    else:
        dv = es.brk["dev"]
        lo = [(dv["dev"] - 1.96 * dv["se"]).min()]
        hi = [(dv["dev"] + 1.96 * dv["se"]).max()]
    a, b = min(lo), max(hi)
    rng = max(b - a, 1e-9)
    pad_lo = (0.17 if stats_here else 0.08) * rng
    return a - pad_lo, b + 0.07 * rng


def _panel_stats(ax, es, col, size=STAT_SIZE):
    bits = [f"pre {pstr(es.pre_p)}{stars(es.pre_p)}",
            f"post {pstr(es.post_p)}{stars(es.post_p)}"]
    if es.brk:
        bits.append(f"break {pstr(es.break_p)}{stars(es.break_p)}")
    ax.text(STAT_X, STAT_Y, STAT_SEP.join(bits), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=size, color=col["main"],
            zorder=8)


def draw_panel(ax, es, view, col=None, title=None, ms=30, stat_size=STAT_SIZE):
    if es is None or (view == "detrended" and es.brk is None):
        ax.axis("off")
        return
    col = col or colour_for(es.kind)
    _frame(ax)

    if view == "levels":
        if es.brk:
            pv = np.array([es.brk["proj"][y] for y in C.YEARS], float)
            obs = es.coefs.set_index("year")["estimate"].reindex(C.YEARS).values
            msk = np.array([y in C.DEV_YEARS for y in C.YEARS])
            ax.fill_between(np.asarray(C.YEARS)[msk], pv[msk], obs[msk],
                            color=col["wedge"], alpha=C.WEDGE_ALPHA, lw=0,
                            zorder=2)
            fit_hi = max(es.brk["proj_years"])
            solid = np.asarray(C.YEARS, float) <= fit_hi
            ax.plot(np.asarray(C.YEARS, float)[solid], pv[solid], "-",
                    lw=1.3, color=C.CFCOL, alpha=.85, zorder=3)
            ax.plot(np.asarray(C.YEARS, float)[~solid], pv[~solid],
                    ls=(0, (5, 2.5)), lw=1.3, color=C.CFCOL, alpha=.85,
                    zorder=3)
        x = es.coefs["year"].values.astype(float)
        y = es.coefs["estimate"].values
        err = [es.coefs["estimate"] - es.coefs["low"],
               es.coefs["high"] - es.coefs["estimate"]]
    else:
        dv = es.brk["dev"]
        x = dv["year"].values.astype(float)
        y = dv["dev"].values
        err = 1.96 * dv["se"].values

    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=col["ci"], elinewidth=1.1,
                capsize=2.4, capthick=1.0, alpha=C.ERR_ALPHA, zorder=4)
    ax.plot(x, y, "-", lw=1.6, color=col["main"], zorder=5)
    ax.scatter(x, y, color=col["main"], s=ms, marker="o", zorder=6,
               edgecolor="white", linewidth=0.6)

    stats_here = view in C.STATS_ON
    ax.set_ylim(*_ylim(es, view, stats_here))
    if stats_here:
        _panel_stats(ax, es, col, stat_size)
    if title:
        ax.set_title(title, fontweight="bold")


def legend_handles(col, view, label=None):
    if view == "levels":
        return [
            plt.Line2D([], [], color=col["main"], marker="o", lw=1.6,
                       label=label or "Estimates, 95% CI"),
            plt.Line2D([], [], color=C.CFCOL, ls=(0, (5, 2.5)), lw=1.3,
                       label=f"Linear pre-trend ({C.PROJ_YEARS[0]}-"
                             f"{C.PROJ_YEARS[-1]}), extrapolated"),
            mpatches.Patch(color=col["wedge"], alpha=C.WEDGE_ALPHA,
                           label=f"Departure tested ({C.DEV_YEARS[0]}-"
                                 f"{C.DEV_YEARS[-1]})"),
            mpatches.Patch(color=C.SHADE,
                           label=f"Post-LLM diffusion ({C.SHADE_FROM}-"
                                 f"{max(C.YEARS)})"),
        ]
    return [
        plt.Line2D([], [], color=col["main"], marker="o", lw=1.6,
                   label=label or "Estimates net of the pre-trend, 95% CI"),
        mpatches.Patch(color=C.SHADE,
                       label=f"Post-LLM diffusion ({C.SHADE_FROM}-"
                             f"{max(C.YEARS)})"),
    ]


SHOW = False


def show_figures(on=True):
    global SHOW
    SHOW = bool(on)


def _save(fig, outfile, verbose=True):
    path = C.FIG_DIR / outfile
    fig.savefig(path, dpi=C.DPI, bbox_inches="tight")
    if SHOW:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
    plt.close(fig)
    if verbose and not SHOW:
        print(f"    saved {path.name}")
    return path


def grid_figure(results, specs, view, outfile, ncol=3, panel_w=5.0,
                panel_h=4.25, suptitle=None, verbose=True):
    drawn = [(k, key, lab) for k, key, lab in specs
             if results.get((k, key)) is not None]
    if not drawn:
        print(f"    !! nothing estimable for {outfile}")
        return None
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(panel_w * ncol, panel_h * nrow),
                             squeeze=False)
    flat = list(axes.flat)
    for ax, (kind, key, label) in zip(flat, specs):
        draw_panel(ax, results.get((kind, key)), view, title=label)
    for ax in flat[len(specs):]:
        ax.axis("off")

    col = colour_for(drawn[0][0])
    h = legend_handles(col, view)
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5,
               columnspacing=1.4, handletextpad=0.6)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold", fontsize=13.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94 if suptitle else 0.99])
    return _save(fig, outfile, verbose)


NCOL_A = 4
COL_IN, ROW_IN, ROW_B_IN = 4.8, 3.2, 2.4
TOP_IN, GAP_IN, BTITLE_IN, TITLE_GAP_IN = 0.90, 0.50, 0.80, 0.46
LEG_IN, LEG_PAD_IN = 1.00, 0.62
HSPACE, WSPACE = 0.45, 0.25
PANEL_TITLE_SIZE, PANEL_TITLE_X = 9, 0.055
STAT_SIZE_A, STAT_SIZE_B = 5.6, 6.6


def tasks14_figure(res_a, specs_a, res_b, specs_b, view, outfile,
                   verbose=True):
    nrowA = int(np.ceil(len(specs_a) / NCOL_A))
    H = (TOP_IN + nrowA * ROW_IN + LEG_IN + GAP_IN + BTITLE_IN
         + ROW_B_IN + LEG_IN)
    W = NCOL_A * COL_IN
    fig = plt.figure(figsize=(W, H))

    a_top = 1 - TOP_IN / H
    a_bot = 1 - (TOP_IN + nrowA * ROW_IN) / H
    b_top = 1 - (TOP_IN + nrowA * ROW_IN + LEG_IN + GAP_IN + BTITLE_IN) / H
    b_bot = LEG_IN / H

    gsA = fig.add_gridspec(nrowA, NCOL_A, top=a_top, bottom=a_bot,
                           left=0.055, right=0.985,
                           hspace=HSPACE, wspace=WSPACE)
    gsB = fig.add_gridspec(1, NCOL_A, top=b_top, bottom=b_bot,
                           left=0.055, right=0.985, wspace=WSPACE)

    colA = colour_for(specs_a[0][0])
    for i, (kind, key, label) in enumerate(specs_a):
        ax = fig.add_subplot(gsA[i // NCOL_A, i % NCOL_A])
        draw_panel(ax, res_a.get((kind, key)), view, col=colA, title=label,
                   ms=16, stat_size=STAT_SIZE_A)

    colB = C.COL_SPEC
    start = (NCOL_A - len(specs_b)) // 2
    first_ax = None
    for j, (kind, key, label) in enumerate(specs_b):
        ax = fig.add_subplot(gsB[0, start + j])
        first_ax = first_ax or ax
        draw_panel(ax, res_b.get((kind, key)), view, col=colB, title=label,
                   ms=26, stat_size=STAT_SIZE_B)

    fig.text(PANEL_TITLE_X, 1 - 0.18 / H, "Panel A. Task Participation",
             ha="left", va="top", fontweight="bold",
             fontsize=PANEL_TITLE_SIZE)
    bx = first_ax.get_position().x0 if first_ax is not None else PANEL_TITLE_X
    fig.text(bx, b_top + TITLE_GAP_IN / H, "Panel B. Portfolio Specialisation",
             ha="left", va="bottom", fontweight="bold",
             fontsize=PANEL_TITLE_SIZE)

    for col, y in ((colA, a_bot), (colB, b_bot)):
        h = legend_handles(col, view)
        fig.legend(handles=h, loc="upper center",
                   bbox_to_anchor=(0.5, y - LEG_PAD_IN / H),
                   ncol=len(h), frameon=False, fontsize=9.5,
                   columnspacing=1.6, handletextpad=0.6)
    return _save(fig, outfile, verbose)


AB_COL_IN, AB_ROW_IN = 6.0, 3.6
AB_TOP_IN, AB_GAP_IN, AB_BTITLE_IN = 0.90, 0.50, 0.80
AB_TITLE_GAP_IN = 0.46


def panel_ab_figure(res_a, res_b, specs, view, outfile,
                    label_a="Panel A", label_b="Panel B",
                    col=None, ms=26, ncol=None, col_in=None, row_in=None,
                    verbose=True):
    ncol = len(specs) if ncol is None else ncol
    nrow = int(np.ceil(len(specs) / ncol))
    col_in = AB_COL_IN if col_in is None else col_in
    row_in = AB_ROW_IN if row_in is None else row_in
    H = (AB_TOP_IN + nrow * row_in + LEG_IN + AB_GAP_IN + AB_BTITLE_IN
         + nrow * row_in + LEG_IN)
    W = ncol * col_in
    fig = plt.figure(figsize=(W, H))

    a_top = 1 - AB_TOP_IN / H
    a_bot = 1 - (AB_TOP_IN + nrow * row_in) / H
    b_top = 1 - (AB_TOP_IN + nrow * row_in + LEG_IN + AB_GAP_IN
                 + AB_BTITLE_IN) / H
    b_bot = LEG_IN / H

    left, right = 0.075, 0.985
    gsA = fig.add_gridspec(nrow, ncol, top=a_top, bottom=a_bot,
                           left=left, right=right,
                           wspace=WSPACE, hspace=HSPACE)
    gsB = fig.add_gridspec(nrow, ncol, top=b_top, bottom=b_bot,
                           left=left, right=right,
                           wspace=WSPACE, hspace=HSPACE)

    colr = col or colour_for(specs[0][0])
    first_a = first_b = None
    stat_size = STAT_SIZE_B if len(specs) <= 2 else STAT_SIZE_A
    for j, (kind, key, label) in enumerate(specs):
        r, c = divmod(j, ncol)
        ax = fig.add_subplot(gsA[r, c])
        first_a = first_a or ax
        draw_panel(ax, res_a.get((kind, key)), view, col=colr, title=label,
                   ms=ms, stat_size=stat_size)
        ax = fig.add_subplot(gsB[r, c])
        first_b = first_b or ax
        draw_panel(ax, res_b.get((kind, key)), view, col=colr, title=label,
                   ms=ms, stat_size=stat_size)

    ax_x = first_a.get_position().x0 if first_a is not None else PANEL_TITLE_X
    fig.text(ax_x, 1 - 0.18 / H, label_a, ha="left", va="top",
             fontweight="bold", fontsize=PANEL_TITLE_SIZE)
    bx = first_b.get_position().x0 if first_b is not None else ax_x
    fig.text(bx, b_top + AB_TITLE_GAP_IN / H, label_b, ha="left",
             va="bottom", fontweight="bold", fontsize=PANEL_TITLE_SIZE)

    for y in (a_bot, b_bot):
        h = legend_handles(colr, view)
        fig.legend(handles=h, loc="upper center",
                   bbox_to_anchor=(0.5, y - LEG_PAD_IN / H),
                   ncol=len(h), frameon=False, fontsize=9.5,
                   columnspacing=1.6, handletextpad=0.6)
    return _save(fig, outfile, verbose)


def draw_bands_panel(ax, per_band, labels, colors, title=None,
                     stat_size=4.8):
    drawn = [(lab, es, c) for lab, es, c in zip(labels, per_band, colors)
             if es is not None]
    if not drawn:
        ax.axis("off")
        return
    _frame(ax)

    lo, hi = [], []
    for i, (lab, es, c) in enumerate(drawn):
        off = (i - (len(drawn) - 1) / 2) * C.BAND_OFFSET
        x = es.coefs["year"].values.astype(float) + off
        y = es.coefs["estimate"].values
        ax.fill_between(x, es.coefs["low"].values, es.coefs["high"].values,
                        color=c, alpha=C.BAND_ALPHA, lw=0, zorder=3)
        ax.plot(x, y, "-", lw=1.4, color=c, zorder=4 + i)
        ax.scatter(x, y, s=14, color=c, zorder=5 + i, marker="o",
                   edgecolor="white", linewidth=0.5)
        lo.append(es.coefs["low"].min())
        hi.append(es.coefs["high"].max())

    a, b = min(lo), max(hi)
    rng = max(b - a, 1e-9)
    ax.set_ylim(a - (0.10 + 0.055 * len(drawn)) * rng, b + 0.07 * rng)

    for i, (lab, es, c) in enumerate(drawn):
        bits = [f"pre {pstr(es.pre_p)}{stars(es.pre_p)}",
                f"post {pstr(es.post_p)}{stars(es.post_p)}"]
        if es.brk:
            bits.append(f"break {pstr(es.break_p)}{stars(es.break_p)}")
        ax.text(STAT_X, STAT_Y + (len(drawn) - 1 - i) * 0.052,
                f"{lab}: " + STAT_SEP.join(bits), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=stat_size, color=c,
                zorder=8)
    if title:
        ax.set_title(title, fontweight="bold")


def bands_figure(results, specs, outfile, labels, ns, ncol=3,
                 panel_w=5.0, panel_h=4.25, verbose=True):
    colors = C.BAND_COLORS[:len(labels)]
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(panel_w * ncol, panel_h * nrow),
                             squeeze=False)
    flat = list(axes.flat)
    for ax, (kind, key, label) in zip(flat, specs):
        per_band = [results[lab].get((kind, key)) for lab in labels]
        draw_bands_panel(ax, per_band, labels, colors, title=label)
    for ax in flat[len(specs):]:
        ax.axis("off")

    h = [plt.Line2D([], [], color=c, marker="o", lw=1.5, ms=4,
                    label=f"{lab}  (n={ns[lab]:,})")
         for lab, c in zip(labels, colors)]
    h.append(mpatches.Patch(color=C.SHADE,
                            label=f"Post-LLM diffusion ({C.SHADE_FROM}-"
                                  f"{max(C.YEARS)})"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5,
               columnspacing=1.4, handletextpad=0.6)
    fig.tight_layout(rect=[0, 0.06, 1, 0.99])
    return _save(fig, outfile, verbose)


def sample_composition(d, outfile="A1_sample_composition.png",
                       author_mode="distinct", id_col="author_ids",
                       verbose=True):
    from .build import parse_id_list

    years = C.YEARS
    pubs = (d.groupby(C.YEAR_COL).size().reindex(years)
            .fillna(0).astype(int))
    assign = (d.groupby(C.YEAR_COL)["team_size"].sum().reindex(years)
              .fillna(0).astype(int))

    distinct = pd.Series(np.nan, index=years, dtype=float)
    if id_col in d.columns:
        m = d[[C.YEAR_COL, id_col]].copy()
        m["_ids"] = m[id_col].map(parse_id_list)
        distinct = (m.explode("_ids").dropna(subset=["_ids"])
                    .groupby(C.YEAR_COL)["_ids"].nunique()
                    .reindex(years).fillna(0).astype(float))
        allids = set().union(*[set(x) for x in m["_ids"] if x]) \
            if len(m) else set()
        if verbose:
            print(f"    distinct authors over the window: {len(allids):,}")
    elif verbose:
        print(f"    !! '{id_col}' absent - distinct count unavailable")

    if author_mode == "distinct" and distinct.isna().all():
        if verbose:
            print("    -> falling back to author_mode='assignments'")
        author_mode = "assignments"
    authors = (distinct if author_mode == "distinct" else assign).astype(int)

    tab = pd.DataFrame({"year": years, "publications": pubs.values,
                        "authors_distinct": distinct.values,
                        "author_paper_assignments": assign.values})
    tab["mean_team_size"] = tab["author_paper_assignments"] / tab["publications"]

    x = np.arange(len(years), dtype=float)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax2 = ax.twinx()
    ax.yaxis.grid(True, color="0.90", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    b1 = ax.bar(x - 0.19, pubs.values, 0.38, color="#7f7f7f", zorder=3,
                label="Number of Publications")
    b2 = ax2.bar(x + 0.19, authors.values, 0.38, color="#5ba3d0", zorder=3,
                 label="Number of Authors")
    for rect, v in zip(b1, pubs.values):
        ax.annotate(f"{v:,}", (rect.get_x() + rect.get_width() / 2,
                               rect.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, zorder=4)
    for rect, v in zip(b2, authors.values):
        ax2.annotate(f"{v:,}", (rect.get_x() + rect.get_width() / 2,
                                rect.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlim(-0.6, len(years) - 0.4)
    lo, hi = pubs.min(), pubs.max()
    pad = 0.12 * (hi - lo if hi > lo else hi)
    ax.set_ylim(lo - 3.2 * pad, hi + 1.2 * pad)
    ax2.set_ylim(0, authors.max() * 1.12)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax.legend(handles=[b1, b2], loc="upper right", frameon=True,
              framealpha=1.0, edgecolor="0.8", fontsize=12)
    fig.tight_layout()
    _save(fig, outfile, verbose)
    return tab


def team_size_over_time(d, outfile="team_size_by_teamtype.png",
                        ma_window=2, ma_center=False, ma_min_periods=1,
                        verbose=True):
    colors = {"all": "#1f3b63", "existing": "#7f9db9", "new": "#b0452b"}
    years = C.YEARS
    stats, rows = {}, []
    for tag, types, label in C.TEAM_GROUPS:
        sub = d if types is None else d[d["team_type"].isin(types)]
        g = sub.groupby(C.YEAR_COL)["team_size"]
        s = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(),
                          "median": g.median()}).reindex(years)
        s["se"] = s["sd"] / np.sqrt(s["n"].where(s["n"] > 0))
        s["lo"] = s["mean"] - 1.96 * s["se"]
        s["hi"] = s["mean"] + 1.96 * s["se"]
        s["ma"] = s["mean"].rolling(ma_window, center=ma_center,
                                    min_periods=ma_min_periods).mean()
        stats[tag] = s
        for y in years:
            if pd.isna(s.loc[y, "n"]):
                continue
            rows.append(dict(group=tag, group_label=label, year=y,
                             n=int(s.loc[y, "n"]), mean=s.loc[y, "mean"],
                             sd=s.loc[y, "sd"], median=s.loc[y, "median"],
                             ci_low=s.loc[y, "lo"], ci_high=s.loc[y, "hi"],
                             ma=s.loc[y, "ma"]))

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axvspan(C.SHADE_FROM - 0.5, max(years) + 0.4, color=C.SHADE, lw=0,
               zorder=0)
    ax.axvline(C.SHADE_FROM - 0.5, ls=(0, (2, 2)), color="0.55", lw=1.0,
               zorder=1)
    for tag, _t, label in C.TEAM_GROUPS:
        s, col = stats[tag], colors[tag]
        ax.fill_between(years, s["lo"].values, s["hi"].values, color=col,
                        alpha=0.16, lw=0, zorder=2)
        ax.plot(years, s["mean"].values, "-", lw=1.0, color=col, alpha=0.45,
                zorder=3)
        ax.plot(years, s["mean"].values, "o", ms=4.2, color=col, alpha=0.45,
                zorder=4)
        ax.plot(years, s["ma"].values, "-", lw=1.9, color=col, zorder=5,
                label=f"{label}  (n={int(s['n'].sum()):,})")
    ax.set_xticks(years)
    ax.set_xlim(min(years) - 0.4, max(years) + 0.4)
    ax.tick_params(axis="x", rotation=45)

    h, l = ax.get_legend_handles_labels()
    if ma_window > 1:
        h.append(plt.Line2D([], [], color="0.45", lw=1.9))
        l.append(f"{ma_window}-year moving average")
        h.append(plt.Line2D([], [], color="0.45", lw=1.0, marker="o",
                            ms=4.2, alpha=0.45))
        l.append("Yearly mean")
    h.append(plt.Rectangle((0, 0), 1, 1, color=C.SHADE))
    l.append(f"Post-LLM diffusion ({C.SHADE_FROM}-{max(years)})")
    fig.legend(handles=h, labels=l, loc="lower center", ncol=len(h),
               frameon=False, bbox_to_anchor=(0.5, 0.004), fontsize=8.5,
               columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.10, 1, 0.99])
    _save(fig, outfile, verbose)
    return pd.DataFrame(rows)
