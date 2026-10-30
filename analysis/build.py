"""plos_prepared.parquet -> clean -> papers."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


_OBJ_COLS = ["author_contributions", "contrib_count", "share_dict",
             "group_count", "group_share"]


def normalize_doi(doi):
    if doi is None or (isinstance(doi, float) and pd.isna(doi)):
        return None
    d = str(doi).strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = d.strip(" .;,\"'/\\")
    return d if "/" in d and len(d) >= 6 else None


def robust_parse_dict(x):
    if isinstance(x, dict):
        return x
    if not isinstance(x, str) or x.strip() in ("", "{}", "nan", "NaN"):
        return {}
    try:
        r = ast.literal_eval(x)
        if isinstance(r, dict):
            return r
    except Exception:
        pass
    try:
        fixed = re.sub(r"(\w)'(\w)", r"\1__APOS__\2", x)
        r = ast.literal_eval(fixed)
        if isinstance(r, dict):
            return {(k.replace("__APOS__", "'") if isinstance(k, str) else k): v
                    for k, v in r.items()}
    except Exception:
        pass
    try:
        r = json.loads(x)
        if isinstance(r, dict):
            return r
    except Exception:
        pass
    return {}


_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def parse_id_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i).strip() for i in x if str(i).strip()]
    if not isinstance(x, str):
        return []
    s = x.strip()
    toks = [a or b for a, b in _QUOTED.findall(s)]
    if not toks:
        toks = re.split(r"[,\s]+", s.strip("[]()"))
    return [t.strip() for t in toks if t.strip()]


def _normalize_role(r):
    if not isinstance(r, str):
        return None
    r = r.lower().strip().replace("–", "-").replace("—", "-")
    r = r.replace("&;", "&")
    r = re.sub(r"[^\w\s\-&]", "", r)
    return re.sub(r"\s+", " ", r).strip()


def map_role(r):
    rc = _normalize_role(r)
    if not rc:
        return None
    if rc in C.ROLE_MAP_RAW:
        return C.ROLE_MAP_RAW[rc]
    if "writing" in rc and "original" in rc:
        return "writing_original_draft"
    if "writing" in rc and "review" in rc:
        return "writing_review_editing"
    return rc


def clean_contrib_roles(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for author, roles_val in d.items():
        if isinstance(roles_val, list):
            out[author] = roles_val
        elif isinstance(roles_val, str):
            roles_val = roles_val.replace("Roles ", "").replace("roles ", "")
            roles = [r.strip() for r in roles_val.split(",") if r.strip()]
            out[author] = list({m for m in (map_role(r) for r in roles) if m})
    return out


def _is_na_value(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().upper() in ("", "N/A", "NA")
    if isinstance(v, list):
        return len(v) == 0
    return False


_BITS = {k: 1 << i for i, k in enumerate(C.ROLE_KEYS)}
_POPCNT = np.array([bin(i).count("1") for i in range(1 << len(C.ROLE_KEYS))],
                   dtype=np.int16)
_STAGE_BITS = {g: 1 << i for i, g in enumerate(C.STAGE_KEYS)}


def _role_mask(roles):
    m = 0
    for r in set(roles or []):
        m |= _BITS.get(r, 0)
    return m


def _stage_mask(roles):
    m = 0
    for r in set(roles or []):
        g = C.ROLE_TO_STAGE.get(r)
        if g:
            m |= _STAGE_BITS[g]
    return m


def role_counts(ac):
    counts = {k: 0 for k in C.ROLE_KEYS}
    if not isinstance(ac, dict):
        return counts
    for roles in ac.values():
        if not isinstance(roles, list):
            continue
        for r in set(roles):
            if r in counts:
                counts[r] += 1
    return counts


def stage_counts(ac):
    counts = {g: 0 for g in C.STAGE_KEYS}
    if not isinstance(ac, dict):
        return counts
    for roles in ac.values():
        if not isinstance(roles, list):
            continue
        for g in {C.ROLE_TO_STAGE[r] for r in set(roles)
                  if r in C.ROLE_TO_STAGE}:
            counts[g] += 1
    return counts


def _shares(counts, team_size):
    if not isinstance(counts, dict) or not team_size or pd.isna(team_size):
        return {k: np.nan for k in counts} if isinstance(counts, dict) else {}
    return {k: v / team_size for k, v in counts.items()}


def _mean_set_size(ac, maskfn):
    if not isinstance(ac, dict) or not ac:
        return np.nan
    sizes = [_POPCNT[maskfn(r)] for r in ac.values()
             if isinstance(r, list) and maskfn(r)]
    return float(np.mean(sizes)) if sizes else np.nan


def _mean_jaccard(ac, maskfn):
    if not isinstance(ac, dict) or len(ac) < C.JACCARD_MIN_AUTHORS:
        return np.nan
    masks = [maskfn(r) for r in ac.values() if isinstance(r, list)]
    masks = [m for m in masks if m]
    if len(masks) < C.JACCARD_MIN_AUTHORS:
        return np.nan
    arr = np.asarray(masks, dtype=np.int64)
    iu = np.triu_indices(len(arr), 1)
    inter = _POPCNT[(arr[:, None] & arr[None, :])[iu]].astype(float)
    union = _POPCNT[(arr[:, None] | arr[None, :])[iu]].astype(float)
    ok = union > 0
    return float(np.mean(inter[ok] / union[ok])) if ok.any() else np.nan


def avg_tasks_per_author(ac):
    return _mean_set_size(ac, _stage_mask)


def avg_jaccard(ac):
    return _mean_jaccard(ac, _stage_mask)


def avg_tasks_per_author14(ac):
    return _mean_set_size(ac, _role_mask)


def avg_jaccard14(ac):
    return _mean_jaccard(ac, _role_mask)


def build_clean(verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    if not Path(C.PREPARED_PARQUET).exists():
        raise FileNotFoundError(
            f"{C.PREPARED_PARQUET} does not exist.\n"
            "The merges now live in their own script - run it once:\n"
            "    python tools/prepare_dataset.py")
    df_old = pd.read_parquet(C.PREPARED_PARQUET)
    say(f"prepared table : {len(df_old):,} rows, {df_old.shape[1]} columns")
    if "doi_norm" not in df_old.columns:
        df_old["doi_norm"] = df_old["seed_doi"].map(normalize_doi)

    dd = df_old["contribs_new_str"].map(robust_parse_dict)
    is_empty = dd.map(lambda x: not isinstance(x, dict) or len(x) == 0)
    any_na = dd.map(lambda x: isinstance(x, dict) and len(x) > 0
                    and any(_is_na_value(v) for v in x.values()))
    say(f"  empty dict            : {int(is_empty.sum()):,}")
    say(f"  >= 1 author N/A       : {int(any_na.sum()):,}")

    clean = df_old[~(is_empty | any_na)].copy()
    say(f"  kept                  : {len(clean):,}")


    clean["author_contributions"] = (clean["contribs_new_str"]
                                     .map(robust_parse_dict)
                                     .map(clean_contrib_roles))
    clean["team_size"] = clean["author_contributions"].map(
        lambda x: len(x) if isinstance(x, dict) else 0).astype(float)

    stray = Counter()
    for d in clean["author_contributions"]:
        for roles in (d or {}).values():
            if isinstance(roles, list):
                stray.update(r for r in set(roles) if r not in _BITS)
    if stray:
        say(f"  !! role codes outside the 14, ignored: "
            f"{dict(stray.most_common(8))}")

    clean["contrib_count"] = clean["author_contributions"].map(role_counts)
    clean["share_dict"] = [
        _shares(c, t) for c, t in zip(clean["contrib_count"],
                                      clean["team_size"])]
    clean["group_count"] = clean["author_contributions"].map(stage_counts)
    clean["group_share"] = [
        _shares(c, t) for c, t in zip(clean["group_count"],
                                      clean["team_size"])]


    clean["avg_tasks_per_author"] = clean["author_contributions"].map(
        avg_tasks_per_author)
    clean["avg_jaccard"] = clean["author_contributions"].map(avg_jaccard)
    clean["avg_tasks_per_author14"] = clean["author_contributions"].map(
        avg_tasks_per_author14)
    clean["avg_jaccard14"] = clean["author_contributions"].map(avg_jaccard14)

    clean["doi_norm"] = clean["seed_doi"].map(normalize_doi)
    clean["seed_year"] = pd.to_numeric(clean["seed_year"],
                                       errors="coerce").astype("Int64")
    return clean


def save_clean(clean, path=None, verbose=True):
    path = path or C.CLEAN_CACHE
    out = clean.copy()
    for c in _OBJ_COLS:
        if c in out.columns:
            out[c] = out[c].map(lambda v: json.dumps(v, ensure_ascii=False)
                                if isinstance(v, (dict, list)) else None)
    for c in out.columns:
        if out[c].dtype == object and c not in _OBJ_COLS:
            out[c] = out[c].astype(str).where(out[c].notna(), None)
    out.to_parquet(path, index=False, compression="zstd")
    if verbose:
        print(f"cached clean -> {path} "
              f"({path.stat().st_size / 1e6:.0f} MB, {len(out):,} rows)")


def load_clean(path=None, verbose=True):
    path = path or C.CLEAN_CACHE
    d = pd.read_parquet(path)
    for c in _OBJ_COLS:
        if c in d.columns:
            d[c] = d[c].map(lambda s: json.loads(s)
                            if isinstance(s, str) and s else {})
    if verbose:
        print(f"loaded clean <- {path} ({len(d):,} rows)")
    return d


def build_papers(clean, verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    d = clean
    out = pd.DataFrame(index=d.index)
    keep = ["seed_doi", "doi_norm", "seed_year", "publication_date",
            "team_type", "team_size", "paper_primary_field",
            "paper_primary_domain", "avg_tasks_per_author", "avg_jaccard",
            "avg_tasks_per_author14", "avg_jaccard14", "author_ids"]
    for c in keep:
        if c in d.columns:
            out[c] = d[c]

    for k in C.ROLE_KEYS:
        out[f"n_{k}"] = [c.get(k, 0) for c in d["contrib_count"]]
        out[f"share_{k}"] = [s.get(k, np.nan) for s in d["share_dict"]]
    for g in C.STAGE_KEYS:
        out[f"n_grp_{g}"] = [c.get(g, 0) for c in d["group_count"]]
        out[f"share_grp_{g}"] = [s.get(g, np.nan) for s in d["group_share"]]

    for c in ("n_authors_bq", "avg_pubs_per_author", "iic",
              "avg_academic_age", "n_authors_info", "collab_ratio"):
        if c in d.columns:
            out[c] = d[c]
    missing = [c for c in ("n_authors_bq", "avg_pubs_per_author", "iic",
                           "avg_academic_age")
               if c not in out.columns]
    if missing:
        say(f"  !! not on the prepared table: {missing} - rerun "
            "tools/prepare_dataset.py")


    if "paper_primary_field" in out.columns:
        vc = out["paper_primary_field"].value_counts()
        rare = vc[vc < C.RARE_FIELD_N].index
        out["paper_primary_field"] = (out["paper_primary_field"]
                                      .where(~out["paper_primary_field"]
                                             .isin(rare), "Other"))
        say(f"  field levels after collapsing rare (<{C.RARE_FIELD_N}): "
            f"{out['paper_primary_field'].nunique()}")

    out["seed_year"] = pd.to_numeric(out["seed_year"],
                                     errors="coerce").astype("Int64")
    out["team_size"] = pd.to_numeric(out["team_size"], errors="coerce")
    say(f"papers table : {len(out):,} rows x {out.shape[1]} cols")
    return out


def load_papers(force=False, verbose=True):
    if C.PAPERS_CACHE.exists() and not force:
        d = pd.read_parquet(C.PAPERS_CACHE)
        if verbose:
            print(f"loaded papers <- {C.PAPERS_CACHE} ({len(d):,} rows)")
        return d

    if C.CLEAN_CACHE.exists() and not force:
        clean = load_clean(verbose=verbose)
    else:
        clean = build_clean(verbose=verbose)
        save_clean(clean, verbose=verbose)

    papers = build_papers(clean, verbose=verbose)
    papers.to_parquet(C.PAPERS_CACHE, index=False, compression="zstd")
    if verbose:
        print(f"cached papers -> {C.PAPERS_CACHE}")
    return papers


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild from the raw files, ignoring the caches")
    a = ap.parse_args()
    load_papers(force=a.force)
    sys.exit(0)
