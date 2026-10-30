"""Constants: paths, window, taxonomy, controls."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
CACHE_DIR   = PROJECT_DIR / "cache"
OUT_DIR     = PROJECT_DIR / "outputs"
FIG_DIR     = OUT_DIR / "figures"
TAB_DIR     = OUT_DIR / "tables"


PREPARED_PARQUET = DATA_DIR / "plos_prepared.parquet"


CLEAN_CACHE  = CACHE_DIR / "clean.parquet"
PAPERS_CACHE = CACHE_DIR / "papers.parquet"

for _d in (DATA_DIR, CACHE_DIR, OUT_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def print_inputs():
    exists = Path(PREPARED_PARQUET).exists()
    print("analysis input")
    print(f"  prepared table {'OK  ' if exists else 'MISSING'}  "
          f"{PREPARED_PARQUET}")
    if not exists:
        print("     -> run  python tools/prepare_dataset.py  first")


YEARS      = list(range(2018, 2026))
REF_YEAR   = 2022
SHADE_FROM = 2023

PRE_YEARS  = [2018, 2019, 2020, 2021]
POST_YEARS = [2023, 2024, 2025]
PROJ_YEARS = [2018, 2019, 2020, 2021]
DEV_YEARS  = [2023, 2024, 2025]

PRE_TREND_ORDER = 1


PERIODS_FULL  = dict(label="Full Period (2018--2025)",
                     years=list(range(2018, 2026)), post_from=2023)
PERIODS_CLEAN = dict(label="No Contamination (2018--19/2024--25)",
                     years=[2018, 2019, 2024, 2025], post_from=2024)


PANEL_PRE  = [2018, 2019]
PANEL_POST = [2024, 2025]


EXPECTED_N   = 117_211
RARE_FIELD_N = 50
MIN_GROUP_N  = 100


TEAM_GROUPS = [
    ("all",      None,              "All Teams"),
    ("existing", ["existing"],      "Established Teams"),
    ("new",      ["new", "hybrid"], "Newly Formed Teams"),
]


ROLE_LABELS = {
    "conceptualization":      "Conceptualization",
    "methodology":            "Methodology",
    "software":               "Software",
    "validation":             "Validation",
    "formal_analysis":        "Formal Analysis",
    "investigation":          "Investigation",
    "resources":              "Resources",
    "data_curation":          "Data Curation",
    "writing_original_draft": "Writing - Original Draft",
    "writing_review_editing": "Writing - Review & Editing",
    "visualization":          "Visualization",
    "supervision":            "Supervision",
    "project_administration": "Project Administration",
    "funding_acquisition":    "Funding Acquisition",
}
ROLE_KEYS = list(ROLE_LABELS)


ROLE_MAP_RAW = {
    "conceptualization": "conceptualization",
    "conceptualizations": "conceptualization",
    "conceptualiza": "conceptualization",
    "data curation": "data_curation",
    "ata curation": "data_curation",
    "data-curation": "data_curation",
    "data analysis": "formal_analysis",
    "formal analysis": "formal_analysis",
    "investigation": "investigation",
    "methodology": "methodology",
    "software": "software",
    "supervisione": "supervision",
    "validation": "validation",
    "visualization": "visualization",
    "visualizationg": "visualization",
    "resources": "resources",
    "project administration": "project_administration",
    "supervision": "supervision",
    "funding": "funding_acquisition",
    "funding acquisition": "funding_acquisition",
    "funding-acquisition": "funding_acquisition",
    "writing - original draft": "writing_original_draft",
    "writing - original-draft": "writing_original_draft",
    "writing & original draft": "writing_original_draft",
    "writing - original draft preparation": "writing_original_draft",
    "writing- original draft": "writing_original_draft",
    "writing - review & editing": "writing_review_editing",
    "writing - review editing": "writing_review_editing",
    "writing - review original draft": "writing_review_editing",
    "writing - review - editing": "writing_review_editing",
    "riting - review & editing": "writing_review_editing",
    "writing & review - editing": "writing_review_editing",
    "writing- review & editing": "writing_review_editing",
}


TASK_GROUPS = {
    "ideation":          {"conceptualization", "methodology"},
    "data_and_coding":   {"data_curation", "software"},
    "execution":         {"investigation", "formal_analysis",
                          "validation", "visualization"},
    "communication":     {"writing_original_draft", "writing_review_editing"},
    "coordination":      {"supervision", "project_administration"},
    "funding_resources": {"resources", "funding_acquisition"},
}
STAGE_KEYS = list(TASK_GROUPS)
STAGE_LABELS = {
    "ideation":          "Ideation",
    "data_and_coding":   "Data and Coding",
    "execution":         "Execution",
    "communication":     "Writing",
    "coordination":      "Coordination",
    "funding_resources": "Funding and Resources",
}
ROLE_TO_STAGE = {r: g for g, rs in TASK_GROUPS.items() for r in rs}


METRICS = [
    ("avg_tasks_per_author", "Tasks per Author"),
    ("avg_jaccard",          "Task Overlap"),
]
METRIC_UNITS = {
    "avg_tasks_per_author": "mean number of the six task domains per author",
    "avg_jaccard": "mean pairwise Jaccard overlap of the authors' task sets",
}
JACCARD_MIN_AUTHORS = 2


METRICS14 = [
    ("avg_tasks_per_author14", "Tasks per Author"),
    ("avg_jaccard14",          "Task Overlap"),
]


import numpy as np

TF = {
    "raw":   lambda x: x,
    "log1p": lambda x: np.log(x + 1),
    "log":   lambda x: np.log(x.mask(x <= 0)),
    "inv":   lambda x: 1.0 / x.mask(x <= 0),
}
TF_LABEL = {"raw": "{s}", "log1p": "log({s} + 1)", "log": "log({s})",
            "inv": "1/{s}"}

CONTROL_SPEC = [
    dict(name="log_size", src=["team_size"], tf="log", required=True,
         use=True, label="log(team size)"),
    dict(name="log_iic", src=["iic"], tf="log1p", required=True,
         use=True, label="log(IIC + 1)"),
    dict(name="log_avg_pubs_auth",
         src=["avg_prior_pubs_clean", "avg_pubs_per_author",
              "avg_prior_pubs"],
         tf="log1p", required=True, use=True,
         label="log(prior publications per author + 1)"),
    dict(name="log_academic_age",
         src=["avg_academic_age_clean", "avg_academic_age"],
         tf="log1p", required=True, use=True,
         label="log(academic age + 1)"),
]
FE_COL   = "paper_primary_field"
YEAR_COL = "seed_year"
VCOV     = "hetero"


P_DAGGER   = 0.10
DAGGER     = r"$^{\dagger}$"
DAGGER_TXT = "†"
DAGGER_TEX = r"\dagger"
SIG_KEY_TXT = (f"{DAGGER_TXT} p<{P_DAGGER:.2f}   * p<0.05   "
               f"** p<0.01   *** p<0.001")
SIG_KEY_TEX = (r"$^{\dagger}p<0.10$, $^{*}p<0.05$, "
               r"$^{**}p<0.01$, $^{***}p<0.001$")


COL_SHARE = dict(main="#1f3b63", ci="#9fb3c8", wedge="#8fa4bd")
COL_COUNT = dict(main="#1c6b5b", ci="#a9c9c2", wedge="#b5d2cc")
COL_SPEC  = dict(main="#5b3a70", ci="#c0aecb", wedge="#ab93ba")
CFCOL     = "#b0452b"
SHADE     = "#eceff2"
ZERO      = "#2b2b2b"
ERR_ALPHA = 0.60
WEDGE_ALPHA = 0.28


COLLAB_COL   = "collab_ratio"
COLLAB_BANDS = [(0.00, 0.25, "0%\u201325%"),
                (0.25, 0.50, "25%\u201350%"),
                (0.50, 0.75, "50%\u201375%")]
BAND_COLORS  = ["#a8c0d8", "#6f93b8", "#3d6b99", "#12263f"]
BAND_OFFSET  = 0.10
BAND_ALPHA   = 0.10
PURE_NEW     = "first"


OUTCOME_COLOR = {"share": COL_SHARE, "count": COL_COUNT, "metric": COL_SPEC}
OUTCOME_NAME  = {"share": "Contributor share",
                 "count": "Log contributor count",
                 "metric": "Portfolio specialisation"}
VIEW_NAME = {"levels": f"coefficients relative to {REF_YEAR}",
             "detrended": "net of the extrapolated pre-trend"}

STATS_ON = {"levels"}


LEGEND_FONT      = 9.5
BAND_LEGEND_FONT = 9
LEGEND_PAD_IN    = 0.34
LEGEND_ROW_IN    = 0.24
DPI = 300

RCPARAMS = {
    "font.size": 10.5, "axes.titlesize": 11.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.default": "regular",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
}