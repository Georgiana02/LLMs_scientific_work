"""Event studies, joint Wald tests, trend breaks, static models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _field

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from . import config as C
from .sample import controls_spec, outcome

try:
    import pyfixest as pf
except ImportError as e:
    raise ImportError(
        "pyfixest is required - every estimate in the paper comes from it.\n"
        "    pip install pyfixest") from e


def stars(p, tex=True):
    if p is None or pd.isna(p):
        return ""
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    if p < C.P_DAGGER:
        return C.DAGGER if tex else C.DAGGER_TXT
    return ""


def pstr(p):
    if p is None or pd.isna(p):
        return "p n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def ptex(p):
    if p is None or pd.isna(p):
        return "n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def fit(d, y_col, x_cols, fe_col=None, year_dummies=True, ref_year=None):
    fe_col = fe_col or C.FE_COL
    ref_year = C.REF_YEAR if ref_year is None else ref_year
    rhs = " + ".join(x_cols) if x_cols else "1"
    lhs = (f"i({C.YEAR_COL}, ref={ref_year}) + {rhs}"
           if year_dummies else rhs)
    fml = f"{y_col} ~ {lhs} | {fe_col}"
    try:
        return pf.feols(fml, data=d, vcov=C.VCOV)
    except Exception as e:
        print(f"     !! pyfixest failed ({type(e).__name__}: {e})")
        return None


def vcov(m):
    for a in ("_vcov", "vcov_matrix"):
        if hasattr(m, a) and not callable(getattr(m, a)):
            return np.asarray(getattr(m, a))
    try:
        return np.asarray(m.vcov())
    except Exception:
        return None


def coef_table(m):
    t = m.tidy().reset_index()
    if "Coefficient" not in t.columns:
        t = t.rename(columns={t.columns[0]: "Coefficient"})
    t = t[t["Coefficient"].astype(str).str.contains(C.YEAR_COL)].copy()
    if t.empty:
        return None
    t["year"] = (t["Coefficient"].astype(str)
                 .str.extract(r"(\d{4})")[0].astype(int))
    res = pd.DataFrame({"year": t["year"].values,
                        "estimate": t["Estimate"].values,
                        "low": t["2.5%"].values,
                        "high": t["97.5%"].values})
    res = pd.concat([res, pd.DataFrame([{"year": C.REF_YEAR, "estimate": 0.0,
                                         "low": 0.0, "high": 0.0}])])
    return res.sort_values("year").reset_index(drop=True)


def joint_wald(m, years):
    b = m.coef()
    names, V = list(b.index), vcov(m)
    sel = [i for i, n in enumerate(names)
           if C.YEAR_COL in n and re.search(r"(\d{4})", n)
           and int(re.search(r"(\d{4})", n).group(1)) in years]
    if V is None or not sel:
        return np.nan
    bb, VV = b.values[sel], V[np.ix_(sel, sel)]
    try:
        return float(chi2.sf(float(bb @ np.linalg.solve(VV, bb)), len(sel)))
    except np.linalg.LinAlgError:
        return np.nan


def year_index(m):
    b = m.coef()
    idx = {}
    for i, n in enumerate(list(b.index)):
        if C.YEAR_COL in n:
            mo = re.search(r"(\d{4})", n)
            if mo:
                idx[int(mo.group(1))] = i
    return b.values, vcov(m), idx


def _proj_c(idx, n_coef, P, h):
    t = np.asarray(P, float)
    tbar = t.mean()
    Stt = float(((t - tbar) ** 2).sum())
    c = np.zeros(n_coef)
    for y in P:
        w = 1.0 / len(P) + (y - tbar) * (h - tbar) / Stt
        if y in idx:
            c[idx[y]] += w
    return c


def _dev_c(idx, n_coef, P, h):
    c = -_proj_c(idx, n_coef, P, h)
    if h in idx:
        c[idx[h]] += 1.0
    return c


def trend_break(b, V, idx):
    if V is None:
        return None
    P = list(C.PROJ_YEARS)
    if sum(1 for y in P if y in idx) < 2:
        return None
    n = len(b)
    t = np.asarray(P, float)
    tbar = t.mean()
    Stt = float(((t - tbar) ** 2).sum())

    cs = np.zeros(n)
    for y in P:
        if y in idx:
            cs[idx[y]] = (y - tbar) / Stt
    slope = float(cs @ b)
    se_s = float(np.sqrt(max(cs @ V @ cs, 0)))

    dev = []
    for h in C.YEARS:
        c = _dev_c(idx, n, P, h)
        e = float(c @ b)
        se = float(np.sqrt(max(c @ V @ c, 0)))
        dev.append(dict(year=h, dev=e, se=se,
                        p=2 * norm.sf(abs(e / se)) if se > 0 else np.nan))
    dev = pd.DataFrame(dev)

    R = np.asarray([_dev_c(idx, n, P, h) for h in C.DEV_YEARS])
    Rb, Vs = R @ b, R @ V @ R.T
    try:
        W = float(Rb @ np.linalg.solve(Vs, Rb))
    except np.linalg.LinAlgError:
        W = float(Rb @ np.linalg.pinv(Vs) @ Rb)

    return dict(slope=slope, se_slope=se_s,
                p_slope=2 * norm.sf(abs(slope / se_s)) if se_s > 0 else np.nan,
                dev=dev, joint_p=float(chi2.sf(W, R.shape[0])),
                proj={h: float(_proj_c(idx, n, P, h) @ b) for h in C.YEARS},
                proj_years=P)


@dataclass
class EventStudy:
    key: str
    label: str
    kind: str
    coefs: pd.DataFrame
    pre_p: float
    post_p: float
    brk: dict | None
    n: int
    meta: dict = _field(default_factory=dict)

    def level(self, year):
        s = self.coefs.set_index("year")["estimate"]
        return float(s.get(year, np.nan))

    def deviation(self, year):
        if not self.brk:
            return np.nan
        s = self.brk["dev"].set_index("year")["dev"]
        return float(s.get(year, np.nan))

    def se(self, year):
        c = self.coefs.set_index("year")
        if year not in c.index:
            return np.nan
        return float((c.loc[year, "high"] - c.loc[year, "low"]) / (2 * 1.96))

    def dev_p(self, year):
        if not self.brk:
            return np.nan
        s = self.brk["dev"].set_index("year")["p"]
        return float(s.get(year, np.nan))

    @property
    def break_p(self):
        return self.brk["joint_p"] if self.brk else np.nan

    def row(self):
        r = dict(key=self.key, label=self.label, kind=self.kind, N=self.n,
                 pre_p=self.pre_p, post_p=self.post_p,
                 break_p=self.break_p,
                 post_mark=stars(self.post_p, tex=False),
                 break_mark=stars(self.break_p, tex=False),
                 **self.meta)
        if self.brk:
            r.update(slope=self.brk["slope"], p_slope=self.brk["p_slope"])


        for y in C.YEARS:
            if y == C.REF_YEAR:
                continue
            r[f"b_{y}"] = self.level(y)
            r[f"se_{y}"] = self.se(y)
        for y in C.POST_YEARS:
            r[f"dev_{y}"] = self.deviation(y)
            r[f"p_dev_{y}"] = self.dev_p(y)
        return r


_SPEC_SHOWN = set()


def event_study(d, kind, key, label, controls, verbose=True, meta=None):
    dd = d.copy()
    dd["y_out"] = outcome(dd, kind, key)
    dd = dd[dd["y_out"].notna()]
    if len(dd) < C.MIN_GROUP_N:
        if verbose:
            print(f"     !! {kind}/{key}: {len(dd):,} usable rows, skipped")
        return None

    if verbose and kind not in _SPEC_SHOWN:
        print(f"\n  SPEC [{kind}]  y ~ i(year, ref={C.REF_YEAR}) + "
              f"{controls_spec(controls)} | {C.FE_COL}")
        print(f"        pyfixest, HC1; pre-trend on {C.PROJ_YEARS}, "
              f"break on {C.DEV_YEARS}\n")
        _SPEC_SHOWN.add(kind)

    m = fit(dd, "y_out", controls)
    if m is None:
        return None
    coefs = coef_table(m)
    if coefs is None:
        return None
    b, V, idx = year_index(m)
    return EventStudy(key=key, label=label, kind=kind, coefs=coefs,
                      pre_p=joint_wald(m, C.PRE_YEARS),
                      post_p=joint_wald(m, C.POST_YEARS),
                      brk=trend_break(b, V, idx),
                      n=int(getattr(m, "_N", len(dd))),
                      meta=dict(meta or {}))


def run_event_studies(d, specs, controls, verbose=True, meta=None):
    out = {}
    for kind, key, label in specs:
        es = event_study(d, kind, key, label, controls, verbose=verbose,
                         meta=meta)
        out[(kind, key)] = es
        if es is None or verbose is False:
            continue
        print(f"     {label:26s} pre {pstr(es.pre_p)}"
              f"{stars(es.pre_p, tex=False)}  "
              f"post {pstr(es.post_p)}{stars(es.post_p, tex=False)}  "
              f"break {pstr(es.break_p)}{stars(es.break_p, tex=False)}  "
              f"N {es.n:,}")
    return out


def static_model(d, kind, key, controls, post_from, terms=("post",),
                 new_types=("new", "hybrid"), base_types=("existing",)):
    dd = d.copy()
    dd["y_out"] = outcome(dd, kind, key)
    dd = dd[dd["y_out"].notna()].copy()

    if "new" in terms or "post_new" in terms:
        keep = list(new_types) + list(base_types)
        dd = dd[dd["team_type"].isin(keep)].copy()
        dd["new"] = dd["team_type"].isin(new_types).astype(float)
    dd["post"] = (dd[C.YEAR_COL] >= post_from).astype(float)
    if "post_new" in terms:
        dd["post_new"] = dd["post"] * dd["new"]
    if len(dd) < C.MIN_GROUP_N:
        return None

    x_cols = list(terms) + list(controls)
    m = fit(dd, "y_out", x_cols, year_dummies=False)
    if m is None:
        return None

    b = m.coef()
    V = vcov(m)
    names = list(b.index)
    est = {}
    for t in terms:
        if t not in names:
            continue
        i = names.index(t)
        beta = float(b.values[i])
        se = float(np.sqrt(max(V[i, i], 0.0))) if V is not None else np.nan
        p = 2 * norm.sf(abs(beta / se)) if se and se > 0 else np.nan
        est[t] = (beta, se, p)
    r2 = getattr(m, "_r2", np.nan)
    return dict(est=est, N=int(getattr(m, "_N", len(dd))),
                r2=float(r2) if r2 is not None else np.nan,
                r2_within=float(getattr(m, "_r2_within", np.nan)))
