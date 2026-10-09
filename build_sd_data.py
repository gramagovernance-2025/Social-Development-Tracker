"""
build_sd_data.py - builds the Social Development Tracker's JSON data
(data/state.json, data/manifest.json, data/districts/*.json,
data/blocks/*.json, data/export_data.json) from:

  - 2_Data/Processed/sd_tracker/*.pkl   (LokOS + VPRP pre-aggregates, see build_sd_inputs.py)
  - 2_Data/Raw Files/Didi Ki Nursery     (nursery monthly + nursery-wise reports)
  - 2_Data/Raw Files/VanMitra Plantation (live plants / distribution / requests, 24-25 and 25-26)
  - 2_Data/Raw Files/DAK Specta          (DAK alternate services, CSC/VLE transactions)
  - Dropbox/DAK_MIS_Scrape               (DAK cases, DAK / coordinator / Sakshma Didi lists)
  - 5_VRF_Analysis/2_Data/2_Clean_Data/vo_vrf_final.dta

Unit of analysis is the BLOCK (534-block universe shared with the DAK
Dashboard and the Synthesised CLF Tracker, block_universe_534.csv).
District and state values are always recomputed from summed numerators
and denominators, never averaged from block values.

Scoring (agreed KPI list, "Social Development Tracker KPIs" doc, Oct 2026):
  - "target" KPIs score as % of target, capped at 100.
  - "pctl" KPIs score as a state percentile among peers (blocks among
    blocks, districts among districts), inclusive (<=) convention, same
    as the CLF and DAK trackers. Direction-aware.
  - Component score = mean of that component's available KPI scores.
  - Overall score = weighted mean of available component scores (default
    weights DAK 30, VRF 10, VPRP 20, Nursery 5, Plantation 5, Disability 10,
    Second Chance 20; the page lets the viewer re-weight components).
  - A component a block doesn't have (no DAK, no data) is left OUT of the
    overall, not scored 0. Second Chance has no data yet and is never scored.
"""
import json, re, glob, os, difflib, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                          # .../6_LokOS_Analysis
JEEVIKA = BASE.parent                           # .../3_Jeevika
DROPBOX = JEEVIKA.parents[1]                    # .../Dropbox
RAW = BASE / "2_Data" / "Raw Files"
PROC = BASE / "2_Data" / "Processed" / "sd_tracker"
DAK_SRC = DROPBOX / "DAK_MIS_Scrape"
VRF_FILE = JEEVIKA / "5_VRF_Analysis" / "2_Data" / "2_Clean_Data" / "vo_vrf_final.dta"
SYNTH = BASE / "3_Output" / "Synthesised CLF Tracker"
OUT = HERE / "data"

pd.set_option("future.no_silent_downcasting", True)

# ============================================================================ block universe + matching
UNIV = pd.read_csv(SYNTH / "block_universe_534.csv")
UNIV["key"] = UNIV.block_name.map(lambda b: re.sub(r"[^A-Z]", "", str(b).upper()))
DISTRICTS = sorted(UNIV.district_norm.unique())

DIST_ALIAS = {
    "AURANAGABAD": "AURANGABAD", "KAIMUR ALIAS BHABUA": "KAIMUR (BHABUA)", "KAIMUR": "KAIMUR (BHABUA)",
    "KAIMUR(BHABUA)": "KAIMUR (BHABUA)", "KAIMUR BHABUA": "KAIMUR (BHABUA)", "EAST CHAMPARAN": "PURBI CHAMPARAN",
    "WEST CHAMPARAN": "PASHCHIM CHAMPARAN", "PURVI CHAMPARAN": "PURBI CHAMPARAN", "SARHASA": "SAHARSA",
    "PURNEA": "PURNIA", "MUNGHYR": "MUNGER", "SHEIKPURA": "SHEIKHPURA",
}
def norm_district(d):
    d = re.sub(r"\s+", " ", str(d).strip().upper())
    d = DIST_ALIAS.get(d, d)
    if d in DISTRICTS:
        return d
    m = difflib.get_close_matches(d, DISTRICTS, 1, 0.8)
    return m[0] if m else None

def block_key(b):
    b = str(b).upper()
    b = re.sub(r"\b(SADAR|BLOCK|C\.?\s*D\.?|PRAKHAND)\b", " ", b)
    return re.sub(r"[^A-Z]", "", b)

# Names that are genuinely different (not just spelling) across sources:
# (district_norm, block_key) -> universe block_name
BLOCK_ALIAS = {
    ("MADHUBANI", "RAHIKA"): "Madhubani",                    # Rahika is the Madhubani sadar block
    ("PATNA", "DANAPUR"): "Dinapur-Cum-Khagaul",
    ("PATNA", "MOKAMA"): "Mokameh",
    ("PURBI CHAMPARAN", "CHHAURADANONARKATIA"): "Narkatia",  # nursery portal: 'Chhauradano(Narkatia)'
    ("PURBI CHAMPARAN", "CHAWRADANO"): "Narkatia",
}
MATCH_LOG = {}
_cache = {}
def match_block(district, block, source):
    """(raw district, raw block) -> block_id, or None. Exact letters-only key
    within district, then prefix (truncated names such as 'Dinapur-Cum-'),
    then fuzzy (difflib >= 0.78) within the same district."""
    ck = (district, block)
    if ck in _cache:
        bid = _cache[ck]
    else:
        d = norm_district(district)
        bid = None
        if d:
            cand = UNIV[UNIV.district_norm == d]
            k = block_key(block)
            if (d, k) in BLOCK_ALIAS:
                k = block_key(BLOCK_ALIAS[(d, k)])
            hit = cand[cand.key == k]
            if len(hit) == 0 and len(k) >= 5:
                hit = cand[cand.key.str.startswith(k) | cand.key.map(lambda x: k.startswith(x) and len(x) >= 5)]
            if len(hit) == 1:
                bid = hit.block_id.iloc[0]
            elif len(hit) == 0 and k:
                m = difflib.get_close_matches(k, cand.key.tolist(), 1, 0.78)
                if m:
                    bid = cand[cand.key == m[0]].block_id.iloc[0]
        _cache[ck] = bid
    lg = MATCH_LOG.setdefault(source, {"matched": 0, "unmatched": set()})
    if bid:
        lg["matched"] += 1
    else:
        lg["unmatched"].add(f"{district} / {block}")
    return bid

def attach_block(df, dcol, bcol, source):
    pairs = df[[dcol, bcol]].drop_duplicates()
    pairs["block_id"] = [match_block(d, b, source) for d, b in zip(pairs[dcol], pairs[bcol])]
    out = df.merge(pairs, on=[dcol, bcol], how="left")
    return out[out.block_id.notna()].copy()

def by_block(df, cols, how="sum"):
    return df.groupby("block_id")[cols].agg(how)

# ============================================================================ LokOS base tables
print("LokOS pre-aggregates")
mem = attach_block(pd.read_pickle(PROC / "members_block.pkl"), "district", "block", "LokOS members")
MEM = by_block(mem, ["members", "active_members", "scst_members", "dis_self", "dis_family", "in_pwd_members", "dis_in_pwd"])
vob = attach_block(pd.read_pickle(PROC / "vo_block.pkl"), "district", "block", "LokOS VOs")
VOS = by_block(vob, ["active_vos"])
shg = attach_block(pd.read_pickle(PROC / "shg_block.pkl"), "district", "block", "LokOS SHGs")
mdis = attach_block(pd.read_pickle(PROC / "members_disability.pkl"), "district", "block", "LokOS members")
pwdf = attach_block(pd.read_pickle(PROC / "pwd_formation.pkl"), "district", "block", "LokOS SHGs")
edu = attach_block(pd.read_pickle(PROC / "members_edu_age.pkl"), "district", "block", "LokOS members")

# ============================================================================ DAK
print("DAK")
def dak_csv(name):
    return pd.read_csv(DAK_SRC / name, low_memory=False)
dak_master = attach_block(dak_csv("dak_master_list.csv"), "District", "Block", "DAK master")
HAS_DAK = set(dak_master.block_id)
cases = []
for kind, f in [("ent", "entitlement_case_master_list.csv"), ("gbv", "gender_case_master_list.csv")]:
    c = dak_csv(f); c["kind"] = kind; cases.append(c)
cases = attach_block(pd.concat(cases, ignore_index=True), "District", "Block", "DAK cases")
cases["app"] = pd.to_datetime(cases.Application_Date, errors="coerce")
det = pd.concat([dak_csv("entitlement_case_resolved_detail.csv"), dak_csv("gender_case_resolved_detail.csv")])
det["res_date"] = pd.to_datetime(det.Date, errors="coerce")
det["Case_ID"] = det.Case_ID.astype(str)
cases["Case_ID"] = cases.Case_ID.astype(str)
cases = cases.merge(det.drop_duplicates("Case_ID")[["Case_ID", "res_date"]], on="Case_ID", how="left")
DATA_DATE = cases.app.max()
cases["resolved"] = cases.Case_Status.eq("Resolved")
cases["days"] = (cases.res_date - cases.app).dt.days.where(cases.resolved)
cases.loc[cases.days < 0, "days"] = np.nan
cases["pend_age"] = (DATA_DATE - cases.app).dt.days.where(~cases.resolved)
HAS_DAK |= set(cases.block_id)
ENT_TYPES = [("Ration card", r"ration|rashan|rasan"), ("Caste / income / residence certificate", r"jati|jaati|caste|income|aay praman|awasiya|niwas|residen"),
             ("Pension", r"pension|pention|pensan|vridha|viklang|vidhwa"), ("Aadhaar", r"aadhar|aadhaar|adhar"),
             ("Job card / MGNREGA", r"job ?card|nrega|manrega|mgnrega"), ("Housing", r"awas|aawas|housing"),
             ("Birth / death certificate", r"janm|jnam|birth|death|mrityu|mirtu"), ("Ayushman card", r"ayushman|aayushman|golden"),
             ("Labour card", r"shram|labou?r card|majdur")]
desc = cases.Case_Description.astype(str).str.lower()
cases["ent_type"] = "Other"
for lab, pat in reversed(ENT_TYPES):
    cases.loc[(cases.kind == "ent") & desc.str.contains(pat, regex=True), "ent_type"] = lab
coord = attach_block(dak_csv("dak_coordinator_master_list.csv"), "District", "Block", "DAK coordinators")
sdl = attach_block(dak_csv("sakshma_didi_master_list.csv"), "District", "Block", "Sakshma Didis")

# Alternate services (DAK Specta / CSC) - latest 12 months; only months that
# were actually scraped count toward the denominator.
sp = pd.concat([pd.read_csv(f) for f in glob.glob(str(RAW / "DAK Specta" / "*.csv"))], ignore_index=True)
sp["Date"] = pd.to_datetime(sp.Date, errors="coerce")
sp["month"] = sp.Date.dt.to_period("M")
ALT_END = sp.month.max()
ALT_WINDOW = pd.period_range(ALT_END - 11, ALT_END, freq="M")
ALT_AVAIL = sorted(set(sp.month.dropna()) & set(ALT_WINDOW))
sp = sp[sp.Date.dt.year >= 2025]   # same window as the DAK tracker; periods filter further
ALT_EXPECTED = pd.period_range(pd.Period("2025-01", freq="M"), ALT_END, freq="M")
ALT_ALL = sorted(set(sp.month.dropna()))
sp = attach_block(sp, "District", "Sub District", "DAK alternate services")
HAS_DAK_FINAL = HAS_DAK

# ============================================================================ VRF
print("VRF")
vrf = pd.read_stata(VRF_FILE, convert_categoricals=False)
vrf = attach_block(vrf, "district_name", "block_name", "VRF")
vrf["w"] = vrf.totalshgmembers.fillna(0)
vrf["full_cov"] = 1 - vrf.incomplete_vrf_coverage
vrf["recv_pos"] = vrf.totalvrfreceived > 0
vrf["zero_int_recv"] = vrf.zero_interest_vo.fillna(0) * vrf.recv_pos
for c in ["savings_discipline_rate", "corpus_multiplier", "interest_yield", "full_cov"]:
    v = vrf[c]
    vrf[c + "_wx"] = (v * vrf.w).where(v.notna())
    vrf[c + "_w"] = vrf.w.where(v.notna())

# ============================================================================ VPRP
print("VPRP")
vvo = attach_block(pd.read_pickle(PROC / "vprp_vo.pkl"), "district", "block", "VPRP")
vent = attach_block(pd.read_pickle(PROC / "vprp_ent_block.pkl"), "district", "block", "VPRP")
vsch = attach_block(pd.read_pickle(PROC / "vprp_scheme.pkl"), "district", "block", "VPRP")
vpg = attach_block(pd.read_pickle(PROC / "vprp_pgsrd.pkl"), "district", "block", "VPRP")
_sdp = pd.read_pickle(PROC / "vprp_sdp.pkl")
vsdp = attach_block(_sdp["issues"], "district", "block", "VPRP")
vdep = attach_block(_sdp["departments"], "district", "block", "VPRP")
for d in (vvo, vent, vsch, vpg, vsdp, vdep):
    d["year"] = d.year.astype(int)
VPRP_YEARS = sorted(vvo.year.unique())
VPRP_LATEST = max(VPRP_YEARS)
# SC/ST reach needs the requester's social category, which is almost entirely
# missing in the 2025 entitlement export (404 of 473k requesters) - use the
# latest year where at least half of requesters have a category recorded.
_cat = vent.groupby("year")[["requesters", "requesters_cat_known"]].sum()
SCST_YEAR = int(max(y for y, r in _cat.iterrows() if r.requesters and r.requesters_cat_known / r.requesters >= 0.5))
CENTRAL = {"state-specific": "State-specific schemes", "pmayg": "PMAY-G (housing)", "mgnregs-job-card": "MGNREGS job card",
           "healthcard": "Health card", "ujjwala": "Ujjwala (LPG)", "pmjjby": "PMJJBY (life insurance)",
           "pmsby": "PMSBY (accident insurance)", "rationcard": "Ration card", "pmsbhgy": "PMSBHGY",
           "widow-pension": "Widow pension", "disability-pension": "Disability pension",
           "rationcard-add": "Ration card (add member)", "old-age-pension": "Old-age pension"}
vsch["central"] = vsch.scheme_type.isin(CENTRAL)
PGSRD_LABEL = {"pgsrd-public-goods": "Public goods", "pgsrd-resources": "Resources", "pgsrd-services": "Services"}

# ============================================================================ Nursery
print("Nursery")
NUR = RAW / "Didi Ki Nursery"
nm = pd.concat([pd.read_csv(f) for f in glob.glob(str(NUR / "DidiKiNursery_*.csv"))], ignore_index=True)
nm = nm[nm.iloc[:, 1].astype(str).str.upper() != "TOTAL"]
nm.columns = ["sl", "block", "nurs", "mgnrega", "forest", "fruit", "timber", "bio", "total", "sold_m", "sold_y",
              "sold_cum", "sale_val", "recv", "due", "district", "period", "month", "year"]
nm["date"] = pd.to_datetime(nm.period, format="%B-%Y")
nm["ym"] = nm.date.dt.strftime("%Y-%m")
nm = attach_block(nm, "district", "block", "Nursery monthly")
NUR_FY = ("2025-04", "2026-03")
NUR_LATEST = nm.ym.max()
NUR_MONTHS = sorted(nm.ym.unique())
# Money entries with a sale value but NO plants sold and (almost) NO payment are
# treated as entry errors and left out of the money figures (Oct 2026):
# 14 rows, including a Rs 3 crore entry for Akorhi Gola (Rohtas), May 2025,
# which alone was ~45% of all unpaid money statewide. Plants and stock in
# those rows are kept.
_bad = (nm.sale_val > 0) & (nm.sold_m == 0) & (nm.recv < 0.01 * nm.sale_val)   # "no payment" = under 1% paid (Akorhi Gola records Rs 1)
# Also left out:
#  - the SAME sale value re-entered by a block in a later month: a running total
#    carried forward, not a new sale (e.g. Gaya blocks re-enter Rs 4,80,000
#    against 50, 100, 200 or 900 plants; Birpur re-enters Rs 2,53,440). Only the
#    first entry is kept.
#  - more than Rs 5,000 per plant, far beyond any sapling price (e.g. 1 plant
#    for Rs 4.8 lakh). Higher-priced but real sales (Rs 500-2,000/plant) stay.
NUR_PRICE_CAP = 5000
_order = nm.sort_values("date").index
_repeat = pd.Series(False, index=nm.index)
_pos = nm.loc[_order]
_repeat.loc[_order] = (_pos.sale_val > 0) & _pos.duplicated(subset=["district", "block", "sale_val"], keep="first")
_bad = _bad | _repeat | ((nm.sale_val > 0) & (nm.sold_m > 0) & (nm.sale_val / nm.sold_m.where(nm.sold_m > 0) > NUR_PRICE_CAP))
_r_noplants = (nm.sale_val > 0) & (nm.sold_m == 0) & (nm.recv < 0.01 * nm.sale_val)
_r_price = (nm.sale_val > 0) & (nm.sold_m > 0) & (nm.sale_val / nm.sold_m.where(nm.sold_m > 0) > NUR_PRICE_CAP)
NUR_FLAG_ROWS = nm[_bad].assign(reason=np.select(
    [_r_noplants[_bad], _repeat[_bad], _r_price[_bad]],
    ["Sale value with no plants sold and almost nothing paid",
     "Same sale value as an earlier month (running total carried forward)",
     f"More than Rs {NUR_PRICE_CAP:,} per plant"], "Implausible money entry"))[
    ["district", "block", "block_id", "period", "sold_m", "sale_val", "recv", "reason"]].copy()
# sold more plants in a month than the nursery had in stock the month before
_nm_sorted = nm.sort_values("date")
_prev = _nm_sorted.groupby("block_id").total.shift(1)
NUR_OVERSOLD = _nm_sorted[(_prev > 0) & (_nm_sorted.sold_m > _prev)].assign(prev_stock=_prev)[
    ["district", "block", "block_id", "period", "sold_m", "prev_stock"]].copy()
NUR_EXCLUDED = int(_bad.sum()); NUR_EXCLUDED_VALUE = float(nm.loc[_bad, "sale_val"].sum())
nm.loc[_bad, ["sale_val", "recv", "due"]] = 0   # drop the whole money entry, not just part of it
nmf = nm[(nm.ym >= NUR_FY[0]) & (nm.ym <= NUR_FY[1])]

nw = pd.read_csv(NUR / "NurseryWiseReport.csv")
nw.columns = ["sl", "dist", "block", "gp", "estyr", "dept", "name", "linked", "plant", "plant_en"] + \
             [f"c{i}" for i in range(20)] + ["scraped"]
nw = attach_block(nw, "dist", "block", "Nursery-wise report")
H_SOLD = {"c12": "Over 5 ft", "c13": "3–5 ft", "c14": "1.5–3 ft", "c15": "Under 1.5 ft"}
nw["sold"] = nw[list(H_SOLD)].sum(axis=1)
nw["avail"] = nw[[f"c{i}" for i in range(16, 20)]].sum(axis=1)

# ============================================================================ Plantation
print("Plantation")
PL = RAW / "VanMitra Plantation"
def load_pl(folder, kind):
    out = []
    for f in glob.glob(str(PL / folder / "*.xlsx")):
        d = re.match(r"Block\w+?_(.+)_\d{4}\.xlsx", os.path.basename(f)).group(1)
        x = pd.read_excel(f)
        if kind == "live":
            x = x.iloc[:, [1, 2]]; x.columns = ["block", "live"]
        else:
            x = x.drop(columns="SlNo").rename(columns={"Block": "block"})
        x["district"] = d
        out.append(x)
    x = pd.concat(out, ignore_index=True)
    x = x[x.block.notna() & (x.block.astype(str).str.upper() != "TOTAL")]
    return attach_block(x, "district", "block", "VanMitra plantation")
PLY = {"2024-25": "2425", "2025-26": "2526"}
PL_LIVE, PL_DIST, PL_REQ = {}, {}, {}
for lab, y in PLY.items():
    PL_LIVE[lab] = load_pl(f"Live_Plants_{y}", "live")
    PL_DIST[lab] = load_pl(f"Plant_Distribution_{y}", "d")
    PL_REQ[lab] = load_pl(f"Plant_Requests_{y}", "d")
SPECIES = [c for c in PL_DIST["2025-26"].columns if c not in ("block", "district", "block_id", "Total")]
FRUIT = ["Guava", "Awala", "Katahal", "Lichi", "Aam", "Jaamun", "Bel", "Neemboo", "Sahajan", "Shareepha"]
SACRED = ["Peepal", "Baragad"]
SPECIES_LABEL = {"Awala": "Amla", "Katahal": "Jackfruit", "Aam": "Mango", "Jaamun": "Jamun", "Neemboo": "Lemon",
                 "Sahajan": "Drumstick", "Shareepha": "Custard apple", "Saagavaan": "Teak", "Baragad": "Banyan",
                 "Sheesham": "Shisham", "Others": "Other species"}
PL_CUR, PL_PREV = "2025-26", "2024-25"

# ============================================================================ helpers
def nz(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x
def r1(x, nd=1):
    x = nz(x if x is None else float(x))
    return None if x is None else round(x, nd)
def ratio(a, b, mult=100.0, nd=1):
    if b is None or b == 0 or a is None or (isinstance(b, float) and math.isnan(b)):
        return None
    return r1(a / b * mult, nd)
def toint(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(round(float(x)))
def mix(series, top=None, other_label="Other"):
    s = series[series > 0].sort_values(ascending=False)
    tot = s.sum()
    if tot == 0:
        return []
    if top and len(s) > top:
        s = pd.concat([s.iloc[:top], pd.Series({other_label: s.iloc[top:].sum()})])
    return [{"label": str(k), "n": toint(v), "pct": r1(v / tot * 100)} for k, v in s.items()]

def sub(df, ids):
    return df[df.block_id.isin(ids)]

# ============================================================================ per-unit metrics
def compute_unit(ids):
    """All display metrics + KPI values for a set of block_ids (one block,
    a district's blocks, or the whole state)."""
    ids = set(ids)
    out = {}
    m = MEM.reindex(list(ids)).sum()
    active_members = float(m.active_members)

    # ---------------- DAK
    has_dak = bool(ids & HAS_DAK_FINAL)
    c = sub(PER["cases"], ids)
    dak = {"has_dak": has_dak, "n_daks": int(sub(dak_master, ids).DAK_Name.nunique()) if has_dak else 0}
    for kind in ("ent", "gbv"):
        k = c[c.kind == kind]
        n, res = len(k), int(k.resolved.sum())
        dak[kind] = {"received": n, "resolved": res, "pending": n - res,
                     "resolution_rate": ratio(res, n),
                     "median_days": r1(k.days.median()) if k.days.notna().any() else None,
                     "pending_age": r1(k.pend_age.mean()) if (~k.resolved).any() else (0.0 if n else None),
                     "pending_90": int(((~k.resolved) & (k.pend_age > 90)).sum())}
    dak["source_mix"] = mix(c.Case_Reach.value_counts(), top=6)
    dak["ent_type_mix"] = mix(c[c.kind == "ent"].ent_type.value_counts().drop("Other", errors="ignore"), top=6)
    dak["violence_mix"] = mix(c[c.kind == "gbv"].Violence_Type.value_counts(), top=6)
    dak["coordinators"] = int(sub(coord, ids).Coordinator_ID.nunique())
    dak["sakshma_didis"] = int(sub(sdl, ids).SD_ID.nunique())
    # reach is per member of blocks that HAVE a DAK - for a district, counting
    # members of DAK-less blocks would understate how far the DAKs reach
    dak_members = float(MEM.reindex(list(ids & HAS_DAK_FINAL)).active_members.sum())
    dak["cases_per_1000"] = ratio(len(c), dak_members, 1000) if has_dak else None
    a = sub(PER["sp"], ids)
    dak["alt"] = {"transactions": int(a["Total Txn"].sum()), "amount": r1(a["Total Amount"].sum(), 0),
                  "active_vles": int(a["CSC Id"].nunique()), "active_months": int(a.month.nunique()),
                  "n_services": int(a.loc[a["Total Txn"] > 0, "Service"].nunique()),
                  "months_available": len(PER["alt_months"]),
                  "service_mix": mix(a.groupby("Service")["Total Txn"].sum(), top=6)}
    # Scored KPI (agreed with Mohan, Oct 2026): number of distinct alternate
    # services the DAK offered in the last 12 months - simpler than counting
    # active months, and ranks blocks almost identically.
    dak["alt_services"] = dak["alt"]["n_services"] if has_dak else None
    out["dak"] = dak

    # ---------------- VRF
    v = sub(vrf, ids)
    if len(v):
        out["vrf"] = {
            "n_vos": len(v), "received": r1(v.totalvrfreceived.sum(), 0), "corpus": r1(v.totalvrfcorpus.sum(), 0),
            "savings": r1(v.vrfsavings.sum(), 0), "interest": r1(v.vrfinterestamount.sum(), 0),
            "incomplete_pct": ratio(v.incomplete_vrf_coverage.sum(), len(v)),
            "coverage_gap": r1(v.vrf_coverage_gap.fillna(0).sum(), 0),
            "zero_interest_pct": ratio(v.zero_int_recv.sum(), v.recv_pos.sum()),
            "loan_rotation": ratio(v.loans_given.sum(), v.totalvrfreceived.sum(), 1, 2),
            "fsf_pct": ratio(v.fsf_eligibile.sum(), len(v)),
            "savings_discipline": ratio(v.savings_discipline_rate_wx.sum(), v.savings_discipline_rate_w.sum(), 1),
            "corpus_multiplier": ratio(v.corpus_multiplier_wx.sum(), v.corpus_multiplier_w.sum(), 1, 2),
            "interest_yield": ratio(v.interest_yield_wx.sum(), v.interest_yield_w.sum(), 1, 2),
            "full_coverage": ratio(v.full_cov_wx.sum(), v.full_cov_w.sum(), 100),
        }
    else:
        out["vrf"] = None

    # ---------------- VPRP
    vv = sub(vvo, ids)
    active_vos = float(VOS.reindex(list(ids)).active_vos.sum())
    by_year = []
    for y in VPRP_YEARS:
        g = vv[vv.year == y]
        sets = {p: set(zip(g[g.part == p].block_id, g[g.part == p].vo_name)) for p in ("ent", "pgsrd", "sdp")}
        anyv = sets["ent"] | sets["pgsrd"] | sets["sdp"]
        by_year.append({"year": int(y), "any": len(anyv), "ent": len(sets["ent"]), "pgsrd": len(sets["pgsrd"]),
                        "sdp": len(sets["sdp"]), "all3": len(sets["ent"] & sets["pgsrd"] & sets["sdp"])})
    cur = next(b for b in by_year if b["year"] == PER["vprp_year"])
    e_all = sub(vent, ids)
    e = e_all[e_all.year == PER["vprp_year"]]
    e_sc = e_all[e_all.year == PER["scst_year"]]
    sc = sub(vsch, ids); sc = sc[sc.year == PER["vprp_year"]]
    pg = sub(vpg, ids); pg = pg[pg.year == PER["vprp_year"]]
    sd = sub(vsdp, ids); sd = sd[sd.year == PER["vprp_year"]]
    dp = sub(vdep, ids); dp = dp[dp.year == PER["vprp_year"]]
    requests = float(e.requests.sum())
    out["vprp"] = {
        "year": int(PER["vprp_year"]), "active_vos": toint(active_vos), "by_year": by_year,
        "requests": toint(requests), "requests_per_100": ratio(requests, active_members, 100),
        "central_mix": mix(sc[sc.central].groupby("scheme_type").n.sum().rename(index=CENTRAL), top=10),
        "state_mix": mix(sc[~sc.central].groupby("scheme_type").n.sum(), top=5),
        "pgsrd_n": int(pg.n.sum()),
        "pgsrd_type_mix": mix(pg.groupby("pgsrd_type").n.sum().rename(index=PGSRD_LABEL)),
        "pgsrd_items": mix(pg.groupby("item_demanded").n.sum(), top=10),
        "sdp_n": int(sd.n.sum()),
        "sdp_sector_mix": mix(sd.groupby("sector").n.sum()),
        "sdp_issues": mix(sd.groupby("social_issue").n.sum(), top=10),
        "departments": mix(dp.groupby("department").n.sum(), top=10),
        "coverage": min(100.0, ratio(cur["any"], active_vos)) if ratio(cur["any"], active_vos) is not None else None,
        "sdp_filed": min(100.0, ratio(cur["sdp"], active_vos)) if ratio(cur["sdp"], active_vos) is not None else None,
        "scst_reach": ratio(e_sc.scst_requesters.sum(), m.scst_members), "scst_year": PER["scst_year"],
        "scst_requesters": toint(e_sc.scst_requesters.sum()), "scst_members": toint(m.scst_members),
    }

    # ---------------- Nursery
    n_all = sub(PER["nm"], ids)                 # rows in this period
    n_latest = n_all[n_all.ym == PER["nur_latest"]]
    n_fy = n_all
    n_full = sub(nm, ids)                       # every month, for the trend chart
    nurs = int(n_latest.nurs.sum()); mg = int(n_latest.mgnrega.sum()); fo = int(n_latest.forest.sum())
    n_blocks = len(ids)
    monthly = n_full.groupby("ym")[["sold_m", "fruit", "timber", "bio", "total", "sale_val", "recv"]].sum()
    _pm = n_all.groupby("ym")[["fruit", "timber", "bio", "total"]].sum()
    peak_ym = _pm.total.idxmax() if len(_pm) else None
    nwb = sub(nw, ids)
    sold_fy = float(n_fy.sold_m.sum())
    nlist = []
    if n_blocks == 1:
        g = nwb.groupby(["gp", "name", "estyr", "dept"]).agg(species=("plant_en", "nunique"), sold=("sold", "sum")).reset_index()
        nlist = [{"name": r["name"].replace("_DIDI_KI_NURSERY", "").replace("_", " ").title(), "gp": str(r.gp).title(),
                  "year": r.estyr, "dept": {"MGNREGA": "MGNREGA", "FOREST DEPARTMENT": "Forest Dept", "JEEViKA": "JEEViKA"}.get(str(r.dept).strip(), "Not specified"),
                  "species": int(r.species), "sold": int(r.sold)} for _, r in g.iterrows()]
    out["nursery"] = {
        "nurseries": nurs, "mgnrega": mg, "forest": fo, "unlinked": max(0, nurs - mg - fo),
        "target": 3 * n_blocks, "gap": int(sum(max(0, 3 - x) for x in n_latest.groupby("block_id").nurs.sum().reindex(list(ids)).fillna(0))),
        "blocks_meeting": int((n_latest.groupby("block_id").nurs.sum() >= 3).sum()), "n_blocks": n_blocks,
        "stock_latest": {"month": PER["nur_latest"], "fruit": toint(n_latest.fruit.sum()), "timber": toint(n_latest.timber.sum()),
                         "bio": toint(n_latest.bio.sum()), "total": toint(n_latest.total.sum())},
        "stock_peak": None if peak_ym is None else {"month": peak_ym, "fruit": toint(_pm.loc[peak_ym, "fruit"]),
                       "timber": toint(_pm.loc[peak_ym, "timber"]), "bio": toint(_pm.loc[peak_ym, "bio"]),
                       "total": toint(_pm.loc[peak_ym, "total"])},
        "monthly": {"months": list(monthly.index), "sold": [toint(x) for x in monthly.sold_m],
                    "stock": [toint(x) for x in monthly.total]},
        "sold_fy": toint(sold_fy), "sale_value": r1(n_fy.sale_val.sum(), 0), "received": r1(n_fy.recv.sum(), 0),
        "pending": r1(n_fy.due.sum(), 0),
        "species_sold": mix(nwb.groupby("plant_en").sold.sum(), top=10),
        "height_mix": mix(nwb[list(H_SOLD)].sum().rename(index=H_SOLD)),
        "nurserywise_n": int(nwb.groupby(["gp", "name"]).ngroups),
        "list": nlist,
        "target_pct": min(100.0, ratio(nurs, 3 * n_blocks)) if n_blocks else None,
        "sales_per_nursery": r1(sold_fy / nurs) if nurs else 0.0,
        "paid_rate": ratio(n_fy.recv.sum(), n_fy.sale_val.sum()),
    }

    # ---------------- Plantation
    pl = {"years": {}}
    for y in PLY:
        lv = sub(PL_LIVE[y], ids).live.sum(); dt = sub(PL_DIST[y], ids); rq = sub(PL_REQ[y], ids)
        dist = float(dt.Total.sum())
        pl["years"][y] = {"distributed": toint(dist), "alive": toint(lv), "requested": toint(rq.Total.sum()) if len(rq) else None,
                          "survival": min(100.0, ratio(lv, dist)) if ratio(lv, dist) is not None else None,
                          "has_data": bool(len(dt))}
    dt = sub(PL_DIST[PER["pl_cur"]], ids)[SPECIES].sum()
    rq = sub(PL_REQ[PER["pl_cur"]], ids)[SPECIES].sum() if len(sub(PL_REQ[PER["pl_cur"]], ids)) else pd.Series(0, index=SPECIES)
    tot = dt.sum()
    pl["group_mix"] = mix(pd.Series({"Fruit": dt[FRUIT].sum(), "Timber / shade": dt[[s for s in SPECIES if s not in FRUIT + SACRED + ["Others"]]].sum(),
                                     "Sacred (peepal, banyan)": dt[SACRED].sum(), "Other species": dt.get("Others", 0)}))
    pl["species"] = mix(dt.rename(index=lambda s: SPECIES_LABEL.get(s, s)), top=10)
    rq_tot = rq.sum()
    pl["req_vs_dist"] = [{"label": SPECIES_LABEL.get(s, s), "requested_pct": r1(rq[s] / rq_tot * 100) if rq_tot else None,
                          "distributed_pct": r1(dt[s] / tot * 100) if tot else None}
                         for s in rq.sort_values(ascending=False).index[:10] if s != "Others"]
    cs = pl["years"][PER["pl_cur"]]["survival"]
    ps = pl["years"][PER["pl_prev"]]["survival"] if PER["pl_prev"] else None
    pl["survival_change"] = r1(cs - ps) if cs is not None and ps is not None else None
    pl["survival"] = cs
    pl["reach"] = ratio(pl["years"][PER["pl_cur"]]["distributed"], active_members, 100) if pl["years"][PER["pl_cur"]]["has_data"] else None
    pl["has_data"] = pl["years"][PER["pl_cur"]]["has_data"]
    out["plantation"] = pl

    # ---------------- Disability + other special SHGs
    sh = sub(shg, ids).groupby("type")[["n", "vo_linked", "bank", "ccl", "members", "formed_2025plus"]].sum()
    def srow(t):
        if t not in sh.index:
            return {"n": 0, "vo_pct": None, "bank_pct": None, "ccl_pct": None, "members": 0, "avg_members": None}
        r = sh.loc[t]
        return {"n": int(r.n), "vo_pct": ratio(r.vo_linked, r.n), "bank_pct": ratio(r.bank, r.n), "ccl_pct": ratio(r.ccl, r.n),
                "members": int(r.members), "avg_members": r1(r.members / r.n) if r.n else None}
    pwd = srow("PWD")
    target = int(math.ceil(active_vos / 2)) if active_vos else 0
    md = sub(mdis, ids)
    form = sub(pwdf, ids).groupby("form_year").n.sum()
    form = form[form.index >= 2014]
    out["disability"] = {
        "pwd": pwd, "regular": srow("REGULAR"),
        "others": [{"type": t, **srow(t)} for t in ["PVTG", "TRANSGENDER", "ELDERLY", "OTHER"]],
        "formed_by_year": [{"year": int(k), "n": int(v)} for k, v in form.items()],
        "dis_self": toint(m.dis_self), "dis_family": toint(m.dis_family), "members": toint(m.members),
        "dis_pct": ratio(m.dis_self, m.members, 100, 2),
        "type_mix": mix(md.groupby("dtype").n.sum()),
        "soc_mix": mix(md.groupby("soc").n.sum()),
        "dis_in_pwd": toint(m.dis_in_pwd), "dis_in_regular": toint(m.dis_self - m.dis_in_pwd),
        "inclusion_pct": ratio(m.dis_in_pwd, m.dis_self),
        "active_vos": toint(active_vos), "target": target,
        "target_pct": min(100.0, ratio(pwd["n"], target)) if target else None,
    }

    # ---------------- Second Chance (target population only; programme data awaited)
    ed = sub(edu, ids)
    def tp(bands, cat=None):
        x = ed[ed.edu.isin(SC_EDU) & ed.band.isin(bands)]
        if cat:
            x = x[x.cat == cat]
        return toint(x.n.sum())
    out["secondchance"] = {"tp35": tp(["lt25", "25_30", "30_35"]), "tp30": tp(["lt25", "25_30"]), "tp25": tp(["lt25"]),
                           "tp35_scst": tp(["lt25", "25_30", "30_35"], "SCST"), "tp35_obc": tp(["lt25", "25_30", "30_35"], "OBC"),
                           "enrolled": None, "appeared": None, "passed": None}
    out["active_members"] = toint(active_members)
    out["pl_season"] = PER["pl_cur"]
    return out

# ============================================================================ periods
# Calendar years, like the DAK tracker: "cumulative" (everything), 2026, 2025.
# Sources that do not change over time (VRF, disability, Second Chance) look
# the same in every period. VPRP plans and VanMitra seasons are yearly: a
# calendar year shows that year's plan / the season planted that year, and the
# latest available when the year has none yet (labelled on the page).
YEARS = [2026, 2025]
PERIODS = ["cumulative"] + [str(y) for y in YEARS]
PL_SEASONS = list(PLY)                          # ["2024-25", "2025-26"]

def make_period(p):
    if p == "cumulative":
        cs, s_, n_ = cases, sp, nm
        vy, plc = VPRP_LATEST, PL_CUR
    else:
        y = int(p)
        cs, s_, n_ = cases[cases.app.dt.year == y], sp[sp.Date.dt.year == y], nm[nm.date.dt.year == y]
        vy = y if y in VPRP_YEARS else VPRP_LATEST
        season = f"{y}-{str(y + 1)[2:]}"
        plc = season if season in PL_SEASONS else PL_CUR
    i = PL_SEASONS.index(plc)
    return {"key": p, "cases": cs, "sp": s_, "nm": n_, "nur_latest": n_.ym.max() if len(n_) else NUR_LATEST,
            "alt_months": sorted(set(s_.month.dropna())), "vprp_year": int(vy), "scst_year": int(min(SCST_YEAR, vy)),
            "pl_cur": plc, "pl_prev": PL_SEASONS[i - 1] if i > 0 else None}

PER = make_period("cumulative")

def period_notes(p):
    per = make_period(p)
    latest_year = max(YEARS)
    so_far = " (so far)" if p == str(latest_year) else ""
    lm = pd.Period(per["nur_latest"]).strftime("%b %Y")
    n = {}
    if p == "cumulative":
        n["dak"] = f"All cases registered up to {DATA_DATE:%d %b %Y}; alternate services since Jan 2025 ({len(ALT_ALL)} of {len(ALT_EXPECTED)} months scraped)"
        n["nursery"] = f"Sales and payments Jan 2025 – {pd.Period(NUR_LATEST).strftime('%b %Y')}; stock as of {lm}"
    else:
        n["dak"] = f"Cases registered in {p}{so_far}; alternate services in {p} ({len(per['alt_months'])} months with data)"
        n["nursery"] = f"Sales and payments in {p}{so_far}; stock as of {lm}"
    vy = per["vprp_year"]
    n["vprp"] = f"Plan year {vy}" + (f" (no {p} plans yet)" if p != "cumulative" and int(p) != vy else "") + f"; SC/ST reach uses {per['scst_year']}"
    n["plantation"] = f"{per['pl_cur']} planting season" + (f" (no {p}-{str(int(p) + 1)[2:]} data yet)" if p != "cumulative" and per["pl_cur"] != f"{p}-{str(int(p) + 1)[2:]}" else "") + \
        (f"; compared with {per['pl_prev']}" if per["pl_prev"] else "")
    n["vrf"] = "Current position of every VO (no history: the same in every period)"
    n["disability"] = "Current SHGs and members from LokOS (no history: the same in every period)"
    n["secondchance"] = "Target population from LokOS member education and age; programme data awaited"
    n["budget"] = "District budget allocated and used; data pending"
    return n

MONTH_KEYS = [str(m) for m in pd.period_range("2025-01", max(pd.Period(NUR_LATEST), ALT_END, pd.Period(DATA_DATE, "M")), freq="M")]
def monthly_for(ids):
    """Month-by-month DAK and nursery figures (the only sources that report monthly)."""
    ids = set(ids)
    c = sub(cases, ids); c = c[c.app.dt.year >= 2025]
    gc = c.assign(ym=c.app.dt.strftime("%Y-%m")).groupby(["ym", "kind"]).agg(rec=("Case_ID", "size"), res=("resolved", "sum"))
    a = sub(sp, ids)
    ga = a.assign(ym=a.month.astype(str)).groupby("ym").agg(txn=("Total Txn", "sum"), amt=("Total Amount", "sum"), svc=("Service", "nunique"))
    n_ = sub(nm, ids)
    gn = n_.groupby("ym").agg(sold=("sold_m", "sum"), val=("sale_val", "sum"), recv=("recv", "sum"), due=("due", "sum"), stock=("total", "sum"))
    out = {}
    for ym in MONTH_KEYS:
        def cv(kind, col):
            return int(gc.loc[(ym, kind), col]) if (ym, kind) in gc.index else 0
        out[ym] = {"ent": [cv("ent", "rec"), cv("ent", "res")], "gbv": [cv("gbv", "rec"), cv("gbv", "res")],
                   "alt": [int(ga.loc[ym, "txn"]), r1(ga.loc[ym, "amt"], 0), int(ga.loc[ym, "svc"])] if ym in ga.index else ([0, 0, 0] if pd.Period(ym) in set(ALT_ALL) else None),
                   "nur": [int(gn.loc[ym, "sold"]), r1(gn.loc[ym, "val"], 0), r1(gn.loc[ym, "recv"], 0), r1(gn.loc[ym, "due"], 0), int(gn.loc[ym, "stock"])] if ym in gn.index else None}
    return out

# Second Chance: dropouts = members whose highest class is primary or middle
# school (CLASS - 5 / 7 / 8), the same definition as BBOSE_LokOS_Targets.xlsx
# (verified against its statewide totals when the build runs).
SC_EDU = ["CLASS - 5", "CLASS - 7", "CLASS - 8"]

# ============================================================================ KPI registry + scoring
COMPONENTS = [
    # Default weights agreed Oct 2026. Weights are relative: a component a unit
    # has no score for (no DAK, Second Chance until its data arrives) drops out
    # and the remaining weights are rescaled to sum to 100%.
    {"key": "dak", "label": "Gender (DAK)", "weight": 30},
    {"key": "vrf", "label": "VRF", "weight": 10},
    {"key": "vprp", "label": "VPRP", "weight": 20},
    {"key": "nursery", "label": "Nursery", "weight": 5},
    {"key": "plantation", "label": "Plantation", "weight": 5},
    {"key": "disability", "label": "Disability SHGs", "weight": 10},
    {"key": "secondchance", "label": "Second Chance", "weight": 20},
    # District budget allocated vs used: data pending, so not scored yet and
    # weight 0 until a weight is agreed.
    {"key": "budget", "label": "Budget", "weight": 0},
]
# key, component, label, unit, scoring type, higher_is_better, getter, description
KPIS = [
    ("dak_ent", "dak", "Entitlement case score", "score", "composite", True, None,
     "Average of three state percentiles: resolution rate, median days to resolve (lower is better), average age of pending cases (lower is better)"),
    ("dak_gbv", "dak", "Gender-violence case score", "score", "composite", True, None,
     "The same three measures, for gender-based-violence cases"),
    ("dak_alt", "dak", "Alternate services offered", "num", "pctl", True, lambda u: u["dak"]["alt_services"],
     "Number of different alternate (CSC) services the DAK provided in the period"),
    ("dak_reach", "dak", "Case reach", "per1000", "pctl", True, lambda u: u["dak"]["cases_per_1000"],
     "Total DAK cases per 1,000 active SHG members (in blocks with a DAK)"),
    ("vrf_disc", "vrf", "Savings discipline", "pct", "pctl", True, lambda u: u["vrf"] and u["vrf"]["savings_discipline"],
     "Actual VRF savings ÷ savings expected if every member saved every month (member-weighted)"),
    ("vrf_mult", "vrf", "Corpus multiplier", "x", "pctl", True, lambda u: u["vrf"] and u["vrf"]["corpus_multiplier"],
     "Total VRF corpus ÷ VRF received (member-weighted)"),
    ("vrf_yield", "vrf", "Interest yield", "pct", "pctl", True, lambda u: u["vrf"] and u["vrf"]["interest_yield"],
     "Interest earned ÷ total VRF corpus (member-weighted)"),
    ("vrf_cov", "vrf", "Full VRF coverage", "pct", "pctl", True, lambda u: u["vrf"] and u["vrf"]["full_coverage"],
     "% of VOs that have received their full VRF (member-weighted)"),
    ("vprp_cov", "vprp", "VPRP coverage", "pct", "target", True, lambda u: u["vprp"]["coverage"],
     "VOs that filed any part of the VPRP ÷ active VOs (target 100%)"),
    ("vprp_sdp", "vprp", "Social issues filed", "pct", "target", True, lambda u: u["vprp"]["sdp_filed"],
     "VOs that filed the social issues (SDP) part ÷ active VOs (target 100%)"),
    ("vprp_scst", "vprp", "SC/ST reach", "pct", "pctl", True, lambda u: u["vprp"]["scst_reach"],
     "SC/ST SHG members with at least one entitlement request ÷ all SC/ST SHG members (2024 plans: social category is missing in the 2025 export)"),
    ("nur_target", "nursery", "Nursery target", "pct", "target", True, lambda u: u["nursery"]["target_pct"],
     "Nurseries ÷ 3 per block (capped at 100%)"),
    ("nur_sales", "nursery", "Plants sold per nursery", "num", "pctl", True, lambda u: u["nursery"]["sales_per_nursery"],
     "Plants sold in the period ÷ number of nurseries (0 if the block has no nursery)"),
    ("nur_paid", "nursery", "Payment received rate", "pct", "pctl", True, lambda u: u["nursery"]["paid_rate"],
     "Payment received ÷ value of plants sold in the period"),
    ("pl_surv", "plantation", "Survival rate", "pct", "pctl", True, lambda u: u["plantation"]["survival"],
     "Live plants ÷ plants distributed, 2025-26 (capped at 100%)"),
    ("pl_reach", "plantation", "Plantation reach", "per100", "pctl", True, lambda u: u["plantation"]["reach"],
     "Plants distributed per 100 active SHG members, 2025-26"),
    ("pwd_target", "disability", "Disability SHG target", "pct", "target", True, lambda u: u["disability"]["target_pct"],
     "Disability SHGs ÷ (active VOs ÷ 2, rounded up), capped at 100%"),
    ("pwd_bank", "disability", "Bank account", "pct", "pctl", True, lambda u: u["disability"]["pwd"]["bank_pct"],
     "Disability SHGs with an SHG bank account ÷ all disability SHGs"),
    ("pwd_fed", "disability", "Federation rate", "pct", "pctl", True, lambda u: u["disability"]["pwd"]["vo_pct"],
     "Disability SHGs linked to a VO ÷ all disability SHGs"),
]
KPI_META = [{"key": k, "component": c, "label": l, "unit": un, "type": t, "higher": h, "desc": d} for k, c, l, un, t, h, _, d in KPIS]

def pctl(series, higher=True):
    s = series.astype(float)
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if valid.sum():
        # ties share the LOWEST rank in the tie, so a large block of
        # identical values (e.g. hundreds of blocks with zero nursery sales)
        # doesn't all inherit a high percentile
        out[valid] = (s[valid] if higher else -s[valid]).rank(method="min") / valid.sum() * 100
    return out.round(1)

def score_units(units):
    """units: dict id -> unit metrics. Adds kpi values/scores, component
    scores and overall score, with percentiles taken among these peers."""
    ids = list(units)
    vals = pd.DataFrame(index=ids)
    for k, c, l, un, t, h, get, d in KPIS:
        if get:
            vals[k] = [get(units[i]) if units[i].get(c) is not None else None for i in ids]
    sc = pd.DataFrame(index=ids)
    # DAK category scores: percentile of each case metric among units with that case type
    for kind, key in (("ent", "dak_ent"), ("gbv", "dak_gbv")):
        rr = pd.Series({i: units[i]["dak"][kind]["resolution_rate"] if units[i]["dak"]["has_dak"] and units[i]["dak"][kind]["received"] else None for i in ids}, dtype=float)
        md = pd.Series({i: units[i]["dak"][kind]["median_days"] if units[i]["dak"]["has_dak"] and units[i]["dak"][kind]["received"] else None for i in ids}, dtype=float)
        pa = pd.Series({i: units[i]["dak"][kind]["pending_age"] if units[i]["dak"]["has_dak"] and units[i]["dak"][kind]["received"] else None for i in ids}, dtype=float)
        comp = pd.concat([pctl(rr), pctl(md, False), pctl(pa, False)], axis=1).mean(axis=1, skipna=True).round(1)
        vals[key] = comp
        sc[key] = comp
    for k, c, l, un, t, h, get, d in KPIS:
        if t == "pctl":
            sc[k] = pctl(vals[k], h)
        elif t == "target":
            sc[k] = vals[k].astype(float).clip(upper=100).round(1)
    # DAK KPIs only count where the unit has a DAK
    for k in ("dak_ent", "dak_gbv", "dak_alt", "dak_reach"):
        sc.loc[[i for i in ids if not units[i]["dak"]["has_dak"]], k] = np.nan
    comp_scores = pd.DataFrame(index=ids)
    for comp in COMPONENTS:
        ks = [k for k, c, *_ in KPIS if c == comp["key"]]
        comp_scores[comp["key"]] = sc[ks].mean(axis=1, skipna=True) if ks else np.nan
    w = pd.Series({c["key"]: c["weight"] for c in COMPONENTS})
    present = comp_scores.notna()
    overall_raw = (comp_scores.fillna(0) * w).sum(axis=1) / present.mul(w).sum(axis=1).replace(0, np.nan)
    # No shared ranks: rank on the UNROUNDED score; exact ties go to the higher
    # Gender (DAK) score (largest weight), then alphabetical order.
    names = pd.Series({i: str(units[i].get("name", i)) for i in ids})
    order = pd.DataFrame({"o": overall_raw, "d": comp_scores["dak"].fillna(-1), "n": names})         .sort_values(["o", "d", "n"], ascending=[False, False, True], na_position="last")
    rank = pd.Series(np.nan, index=ids)
    scored_ids = [i for i in order.index if pd.notna(order.at[i, "o"])]
    rank[scored_ids] = np.arange(1, len(scored_ids) + 1)
    # sort key used by the page (keeps the same order as these ranks)
    sort_key = pd.Series({i: (len(scored_ids) - (scored_ids.index(i))) for i in scored_ids})
    overall = overall_raw.round(1)
    comp_scores = comp_scores.round(1)
    for i in ids:
        u = units[i]
        u["kpi"] = {k: {"value": nz(vals.at[i, k]) if k in vals else None, "score": nz(sc.at[i, k])} for k, *_ in KPIS}
        for k in u["kpi"]:
            v = u["kpi"][k]["value"]
            u["kpi"][k]["value"] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            s_ = u["kpi"][k]["score"]
            u["kpi"][k]["score"] = None if s_ is None or (isinstance(s_, float) and math.isnan(s_)) else float(s_)
        u["comp_scores"] = {c["key"]: nz(float(comp_scores.at[i, c["key"]])) for c in COMPONENTS}
        u["overall_score"] = nz(float(overall.at[i]))
        u["overall_state_rank"] = None if math.isnan(rank.at[i]) else int(rank.at[i])
        u["overall_sort"] = int(sort_key[i]) if i in sort_key.index else None
        u["overall_state_n"] = int(overall.notna().sum())

def rank_row(u, extra):
    row = {**extra, "overall_score": u["overall_score"], "overall_state_rank": u["overall_state_rank"], "overall_sort": u.get("overall_sort")}
    for c in COMPONENTS:
        row["comp_" + c["key"]] = u["comp_scores"][c["key"]]
    for k, *_ in KPIS:
        row[k] = u["kpi"][k]["value"]
        row[k + "__score"] = u["kpi"][k]["score"]
    return row

def slugify(s):
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", str(s).lower()))

# ============================================================================ build
# ============================================================================ data checks
# Two kinds of flag:
#   "error": a value that cannot be right (more plants alive than given, a
#            case resolved before it was filed, ...). Flagged wherever it occurs.
#   "check": a value far from what other places report. Measured with a robust
#            z-score (distance from the median in units of the median absolute
#            deviation), both ends, |z| > 3.5. Unlike a fixed "top/bottom 1%",
#            this flags nothing when the data is clean and everything when it
#            is not, and the outliers themselves don't distort it.
CHECK_Z = 3.5
SKEWED_UNITS = {"num", "per100", "per1000", "x"}
# Only metrics where an extreme value suggests a recording problem. Bounded
# performance rates (VO linkage, bank accounts, plan coverage, ...) are left
# out: a block at 100% there is doing well, not entering data wrongly.
STAT_KPIS = ["dak_reach", "vrf_disc", "vrf_mult", "vrf_yield", "nur_sales", "pl_surv", "pl_reach"]

def build_checks(units, level):
    """units: dict id -> computed unit (blocks or districts). Returns flags."""
    flags = []
    def add(uid, comp, rule, sev, detail, value=None):
        u = units[uid]
        flags.append({"level": level, "id": uid, "district": u["district"] if level == "block" else u["name"],
                      "block": u["name"] if level == "block" else None, "c": comp, "rule": rule, "sev": sev,
                      "detail": detail, "value": value})
    ids = list(units)
    # ---- statistical checks on metric values
    for k in STAT_KPIS:
        meta = next(m for m in KPI_META if m["key"] == k)
        v = pd.Series({i: units[i]["kpi"][k]["value"] for i in ids}, dtype=float).dropna()
        if len(v) < 10:
            continue
        t = np.log1p(v.clip(lower=0)) if meta["unit"] in SKEWED_UNITS else v
        med = t.median(); mad = (t - med).abs().median() * 1.4826
        if not mad or np.isnan(mad):
            continue
        z = (t - med) / mad
        lo, hi = v.quantile(0.1), v.quantile(0.9)
        fmt = (lambda a: f"{a:.1f}%") if meta["unit"] == "pct" else (lambda a: f"{a:,.1f}")
        for i in z[z.abs() > CHECK_Z].index:
            add(i, meta["component"], f"Unusually {'high' if z[i] > 0 else 'low'}: {meta['label']}", "check",
                f"{meta['label']} is {fmt(v[i])}; most {level}s are between {fmt(lo)} and {fmt(hi)}", round(float(v[i]), 2))
    return flags

def block_rule_flags(blocks):
    flags = []
    def add(bid, comp, rule, sev, detail, value=None):
        u = blocks[bid]
        flags.append({"level": "block", "id": bid, "district": u["district"], "block": u["name"], "c": comp,
                      "rule": rule, "sev": sev, "detail": detail, "value": value})
    # DAK: resolved before filed; resolved cases with no resolution details
    neg = cases[(cases.res_date - cases.app).dt.days < 0].groupby("block_id").size()
    for bid, n in neg.items():
        add(bid, "dak", "Case resolved before it was filed", "error", f"{n} resolved case{'s' if n > 1 else ''} with a resolution date earlier than the application date", int(n))
    nodet = cases[cases.resolved & cases.res_date.isna()].groupby("block_id").size()
    for bid, n in nodet.items():
        add(bid, "dak", "Resolved case with no resolution details", "check", f"{n} resolved case{'s' if n > 1 else ''} with no resolution date or outcome on the portal", int(n))
    for bid in blocks:
        if bid in HAS_DAK and not (cases.block_id == bid).any():
            add(bid, "dak", "DAK with no cases", "check", "The block has a DAK but no cases are recorded")
    # VRF: corpus below what was received; savings far above the expected amount
    g = vrf.assign(low=(vrf.totalvrfreceived > 0) & (vrf.totalvrfcorpus < vrf.totalvrfreceived),
                   high=vrf.savings_discipline_rate > 300).groupby("block_id")[["low", "high"]].sum()
    for bid, r in g.iterrows():
        if r.low:
            add(bid, "vrf", "VRF corpus smaller than VRF received", "error", f"{int(r.low)} VO{'s' if r.low > 1 else ''} report a corpus below the grant received (savings and interest should only add to it)", int(r.low))
        if r.high:
            add(bid, "vrf", "Savings more than 3 times the expected amount", "check", f"{int(r.high)} VO{'s' if r.high > 1 else ''} report VRF savings over 300% of what full monthly saving would give", int(r.high))
    # VPRP: more VOs filed than there are active VOs
    for bid, u in blocks.items():
        v = u["vprp"]; cur = next((b for b in v["by_year"] if b["year"] == v["year"]), None)
        if cur and v["active_vos"] and cur["any"] > v["active_vos"]:
            add(bid, "vprp", "More VOs filed than active VOs", "check", f"{cur['any']} VOs filed a plan in {v['year']} but LokOS lists {v['active_vos']} active VOs (VO names may not match)", cur["any"])
    # Nursery: implausible money entries and overselling
    for _, r in NUR_FLAG_ROWS.iterrows():
        add(r.block_id, "nursery", r.reason, "error", f"{r.period}: {int(r.sold_m):,} plants, value Rs {int(r.sale_val):,}, received Rs {int(r.recv):,}. Left out of the money figures.", int(r.sale_val))
    for _, r in NUR_OVERSOLD.iterrows():
        add(r.block_id, "nursery", "Sold more plants than were in stock", "check", f"{r.period}: {int(r.sold_m):,} plants sold, but only {int(r.prev_stock):,} in stock the month before", int(r.sold_m))
    # Plantation: more alive than given; species not adding up to the total
    for y in PLY:
        lv = PL_LIVE[y].groupby("block_id").live.sum(); dt = PL_DIST[y].groupby("block_id").Total.sum()
        both = pd.concat([lv, dt], axis=1).dropna()
        for bid, r in both[both.live > both.Total].iterrows():
            add(bid, "plantation", "More plants alive than were given", "error", f"{y}: {int(r.live):,} alive but {int(r.Total):,} given", int(r.live))
        sp = PL_DIST[y].groupby("block_id")[SPECIES].sum().sum(axis=1)
        diff = pd.concat([sp.rename("sp"), dt.rename("tot")], axis=1).dropna()
        for bid, r in diff[(diff.sp - diff.tot).abs() > 0.01 * diff.tot.clip(lower=1)].iterrows():
            add(bid, "plantation", "Species don't add up to the total", "check", f"{y}: species add up to {int(r.sp):,} but the total says {int(r.tot):,}", int(r.tot))
    # Disability: disability SHGs but no members with a disability
    for bid, u in blocks.items():
        d = u["disability"]
        if d["pwd"]["n"] > 0 and not d["dis_self"]:
            add(bid, "disability", "Disability SHGs but no disabled members recorded", "check", f"{d['pwd']['n']} disability SHGs, but no member in the block is recorded as having a disability", d["pwd"]["n"])
    return flags

def source_issues():
    return [
        {"c": "vprp", "sev": "check", "issue": f"{VPRP_LATEST} entitlement export looks incomplete",
         "detail": f"About a quarter as many requests per VO as 2023, and social category is missing for almost all {VPRP_LATEST} requesters. SC/ST reach uses {SCST_YEAR} instead."},
        {"c": "nursery", "sev": "check", "issue": "Nursery-wise report is missing nurseries",
         "detail": "It lists 636 of the 893 nurseries in the monthly report, and its 'dried plants' columns are empty everywhere."},
        {"c": "nursery", "sev": "check", "issue": "Payments are only recorded in the month of sale",
         "detail": "Sale value always equals received + due for that month, so payments that arrive later never appear."},
        {"c": "dak", "sev": "check", "issue": f"Alternate services cover {len(ALT_ALL)} of the {len(ALT_EXPECTED)} months since Jan 2025",
         "detail": "Some monthly CSC exports were never scraped, and only 37 blocks have any activity."},
        {"c": "disability", "sev": "check", "issue": "Disability type defaults to 'Sight'",
         "detail": "LokOS records 'Sight' as the disability type for most members who have no disability, so type is only used for members flagged as disabled."},
    ]

def source_dates():
    def newest(paths):
        ts = [os.path.getmtime(f) for f in paths if os.path.exists(f)]
        return pd.Timestamp.fromtimestamp(max(ts)).strftime("%d %b %Y") if ts else None
    CLEAN = BASE / "2_Data" / "Cleaned"
    return [
        {"c": "dak", "source": "DAK cases (DAK MIS)", "updated": newest(glob.glob(str(DAK_SRC / "*case_master_list.csv"))), "covers": f"Cases up to {DATA_DATE:%d %b %Y}"},
        {"c": "dak", "source": "DAK alternate services (CSC)", "updated": newest(glob.glob(str(RAW / "DAK Specta" / "*.csv"))), "covers": f"Up to {ALT_END.strftime('%b %Y')}"},
        {"c": "vrf", "source": "VRF", "updated": newest([str(VRF_FILE)]), "covers": "Current position of each VO"},
        {"c": "vprp", "source": "VPRP (LokOS)", "updated": newest(glob.glob(str(CLEAN / "clf_*.dta"))), "covers": f"Plan years {min(VPRP_YEARS)}–{VPRP_LATEST}"},
        {"c": "nursery", "source": "Didi ki Nursery", "updated": newest(glob.glob(str(RAW / "Didi Ki Nursery" / "*.csv"))), "covers": f"Up to {pd.Period(NUR_LATEST).strftime('%b %Y')}"},
        {"c": "plantation", "source": "VanMitra plantation", "updated": newest(glob.glob(str(RAW / "VanMitra Plantation" / "*" / "*.xlsx"))), "covers": "2024-25 and 2025-26"},
        {"c": "disability", "source": "LokOS SHG and member profiles", "updated": newest(glob.glob(str(BASE / "2_Data" / "Raw Files" / "Member-Level Profile" / "*.csv")) + [str(CLEAN / "lokos_shg_profiles.dta")]), "covers": "Current SHGs and members"},
    ]

def dist(vals):
    s_ = pd.Series(vals, dtype=float).dropna()
    if not len(s_):
        return None
    return {"median": round(float(s_.median()), 1), "n": int(len(s_)), "green": int((s_ >= 66).sum()),
            "yellow": int(((s_ >= 33) & (s_ < 66)).sum()), "red": int((s_ < 33).sum())}

def run_period(p):
    """Compute and score every block, district and the state for one period."""
    global PER
    PER = make_period(p)
    print(f"[{p}] blocks")
    blocks = {}
    for _, b in UNIV.iterrows():
        u = compute_unit([b.block_id])
        u.update({"level": "block", "block_id": b.block_id, "name": b.block_name, "district": b.district_name.title()})
        blocks[b.block_id] = u
    score_units(blocks)
    print(f"[{p}] districts")
    districts = {}
    for d in DISTRICTS:
        ids = UNIV[UNIV.district_norm == d].block_id.tolist()
        u = compute_unit(ids)
        u.update({"level": "district", "name": d.title(), "district_geo_id": d, "n_blocks": len(ids)})
        districts[d] = u
    score_units(districts)
    for d in DISTRICTS:   # within-district block ranks, no shared places
        ids = UNIV[UNIV.district_norm == d].block_id.tolist()
        ov = pd.Series({i: blocks[i]["overall_score"] for i in ids}, dtype=float)
        srt = pd.Series({i: blocks[i]["overall_sort"] for i in ids}, dtype=float)
        rk = srt.rank(ascending=False, method="first")
        for i in ids:
            blocks[i]["overall_district_rank"] = None if math.isnan(rk[i]) else int(rk[i])
            blocks[i]["overall_district_n"] = int(ov.notna().sum())
    print(f"[{p}] state")
    state = compute_unit(UNIV.block_id.tolist())
    state.update({"level": "state", "name": "Bihar"})
    # The state has no peers, so its scores describe the TYPICAL BLOCK:
    #  - "target" metrics: the state's own value against the target
    #  - "percentile" metrics, component scores and the SD Index: the median
    #    block score, plus how many blocks fall in each colour band.
    state["kpi"], state["dist"] = {}, {}
    for k, c, l, un, t, h, get, d in KPIS:
        val = get(state) if get and state.get(c) is not None else None
        dd = dist([blocks[i]["kpi"][k]["score"] for i in blocks])
        state["dist"][k] = dd
        score = (min(100.0, round(float(val), 1)) if val is not None else None) if t == "target" else (dd["median"] if dd else None)
        state["kpi"][k] = {"value": None if val is None else float(val), "score": score}
    state["comp_scores"] = {}
    for c in COMPONENTS:
        dd = dist([blocks[i]["comp_scores"][c["key"]] for i in blocks])
        state["dist"]["comp_" + c["key"]] = dd
        state["comp_scores"][c["key"]] = dd["median"] if dd else None
    dd = dist([blocks[i]["overall_score"] for i in blocks])
    state["dist"]["overall"] = dd
    state["overall_score"] = dd["median"] if dd else None
    return blocks, districts, state

# parts of a unit that change between periods; everything else is stored once
VARY = ["dak", "nursery", "plantation", "vprp", "kpi", "comp_scores", "overall_score", "overall_state_rank",
        "overall_state_n", "overall_sort", "overall_district_rank", "overall_district_n", "active_members", "dist", "pl_season"]

def main():
    RES = {p: run_period(p) for p in PERIODS}
    blocks, districts, state = RES["cumulative"]
    global PER
    PER = make_period("cumulative")

    meta = {
        "components": COMPONENTS, "kpis": KPI_META,
        "period_keys": PERIODS, "months": MONTH_KEYS,
        "period_notes": {p_: period_notes(p_) for p_ in PERIODS},
        "periods": {
            "dak": f"All cases registered up to {DATA_DATE:%d %b %Y}; alternate services since Jan 2025 ({len(ALT_ALL)} of {len(ALT_EXPECTED)} months scraped)",
            "vrf": "VRF position of every VO, from the latest VRF data (same source as the VRF CLF tracker)",
            "vprp": f"Latest plan year {VPRP_LATEST} (trend {min(VPRP_YEARS)}–{VPRP_LATEST}); SC/ST reach uses {SCST_YEAR}",
            "nursery": f"Sales and payments Jan 2025 – {pd.Period(NUR_LATEST).strftime('%b %Y')}; stock as of {pd.Period(NUR_LATEST).strftime('%b %Y')}",
            "nursery_excluded": f"{NUR_EXCLUDED} implausible money entries (worth Rs {NUR_EXCLUDED_VALUE/1e7:.2f} crore) are left out of the money figures: a sale value with no plants sold and almost nothing paid (including Rs 3 crore for Akorhi Gola, May 2025), the same value re-entered in a later month (a running total carried forward), or more than Rs {NUR_PRICE_CAP:,} per plant. Plants sold still count. The portal only records payment in the month of sale, so payments that arrive later never show up.",
            "plantation": "Survival and reach 2025-26; comparison with 2024-25",
            "budget": "District budget allocated and used; data pending",
            "disability": "LokOS SHG and member profiles, scraped July 2026",
            "secondchance": "Target population from LokOS member education and age; programme data awaited",
        },
        "built": pd.Timestamp.now().strftime("%d %b %Y"),
        "match_log": {k: {"matched_rows": v["matched"], "unmatched": sorted(v["unmatched"])[:50], "n_unmatched": len(v["unmatched"])} for k, v in MATCH_LOG.items()},
    }

    # Clear old outputs file-by-file: Dropbox often holds a lock on the
    # folder itself, so removing the folder outright fails on Windows.
    OUT.mkdir(exist_ok=True)
    for sub_ in ("blocks", "districts"):
        (OUT / sub_).mkdir(exist_ok=True)
        for f in (OUT / sub_).glob("*.json"):
            f.unlink()
    (OUT / "geo").mkdir(exist_ok=True)
    for f in (SYNTH / "data" / "geo").glob("*.geojson"):
        shutil.copy2(f, OUT / "geo" / f.name)

    def dump(path, obj):
        with open(path, "w", encoding="utf8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=lambda o: None)

    def bslug(i):
        return slugify(f"{UNIV.set_index('block_id').district_norm[i]}-{RES['cumulative'][0][i]['name']}-{i}")
    def over(u):
        return {k: u[k] for k in VARY if k in u}
    def lists_for(p):
        # district rows, block rows per district, all block rows, top/bottom for period p
        bl, di, _ = RES[p]
        drows, brows_by_d = [], {}
        for d in DISTRICTS:
            bids = UNIV[UNIV.district_norm == d].sort_values("block_name").block_id.tolist()
            brows_by_d[d] = [rank_row(bl[i], {"name": bl[i]["name"], "block_id": i, "slug": bslug(i),
                                              "overall_district_rank": bl[i]["overall_district_rank"]}) for i in bids]
            drows.append(rank_row(di[d], {"name": di[d]["name"], "slug": slugify(d), "district_geo_id": d}))
        allb = [rank_row(bl[i], {"name": bl[i]["name"], "district": bl[i]["district"], "block_id": i, "slug": bslug(i)}) for i in bl]
        sc_ = sorted([r for r in allb if r["overall_score"] is not None], key=lambda r: -(r["overall_sort"] or 0))
        return drows, brows_by_d, allb, sc_[:10], sc_[::-1][:10]
    LISTS = {p: lists_for(p) for p in PERIODS}
    manifest = {"districts": []}
    for d in DISTRICTS:
        dslug = slugify(d)
        brows = LISTS["cumulative"][1][d]
        dfile = dict(districts[d]); dfile["blocks"] = brows
        dfile["periods"] = {p: {**over(RES[p][1][d]), "blocks": LISTS[p][1][d]} for p in PERIODS if p != "cumulative"}
        dfile["monthly"] = monthly_for(UNIV[UNIV.district_norm == d].block_id.tolist())
        dump(OUT / "districts" / f"{dslug}.json", dfile)
        manifest["districts"].append({"slug": dslug, "name": districts[d]["name"],
                                      "blocks": [{"slug": r["slug"], "name": r["name"], "block_id": r["block_id"]} for r in brows]})
        for r in brows:
            i = r["block_id"]
            bfile = dict(blocks[i])
            bfile["periods"] = {p: over(RES[p][0][i]) for p in PERIODS if p != "cumulative"}
            bfile["monthly"] = monthly_for([i])
            dump(OUT / "blocks" / f"{r['slug']}.json", bfile)
    district_rows, _, all_blocks, top_, bot_ = LISTS["cumulative"]
    state["districts"] = district_rows
    state["top_blocks"], state["bottom_blocks"] = top_, bot_
    state["periods"] = {p: {**over(RES[p][2]), "districts": LISTS[p][0], "top_blocks": LISTS[p][3], "bottom_blocks": LISTS[p][4]}
                        for p in PERIODS if p != "cumulative"}
    state["monthly"] = monthly_for(UNIV.block_id.tolist())
    meta["sources"] = source_dates()
    _upd = [pd.Timestamp(s_["updated"]) for s_ in meta["sources"] if s_["updated"]]
    meta["last_updated"] = max(_upd).strftime("%d %b %Y") if _upd else None
    state["meta"] = meta
    dump(OUT / "state.json", state)
    flags = block_rule_flags(blocks) + build_checks(blocks, "block") + build_checks(districts, "district")
    for f_ in flags:
        if f_["level"] == "block":
            f_["slug"] = slugify(f"{blocks[f_['id']]['district']}-{blocks[f_['id']]['name']}-{f_['id']}")
            f_["dslug"] = slugify(UNIV.set_index("block_id").district_norm[f_["id"]])
        else:
            f_["dslug"] = slugify(f_["id"])
    dump(OUT / "checks.json", {"flags": flags, "source_issues": source_issues(), "z": CHECK_Z})
    print(f"Data checks: {len(flags)} flags ({sum(f_['sev'] == 'error' for f_ in flags)} errors)")
    dump(OUT / "manifest.json", manifest)
    dump(OUT / "scoring_summary.json", {"periods": {p: {
        "blocks": [{"id": r["block_id"], "name": r["name"], "district": r["district"], "slug": r["slug"],
                    "comp": {c["key"]: r["comp_" + c["key"]] for c in COMPONENTS}} for r in LISTS[p][2]],
        "districts": [{"id": r["district_geo_id"], "name": r["name"],
                       "comp": {c["key"]: r["comp_" + c["key"]] for c in COMPONENTS}} for r in LISTS[p][0]]} for p in PERIODS}})
    lab = {k: l for k, c, l, *_ in KPIS}
    def flat(r, keep):
        o = {kk: r.get(kk) for kk in keep}
        o["SD Index"] = r["overall_score"]
        for c in COMPONENTS:
            o[c["label"] + " score"] = r["comp_" + c["key"]]
        for k, *_ in KPIS:
            o[lab[k]] = r[k]; o[lab[k] + " (score)"] = r[k + "__score"]
        return o
    dump(OUT / "export_data.json", {"periods": {p: {
        "blocks": [flat(r, ["district", "name", "block_id", "overall_state_rank"]) for r in LISTS[p][2]],
        "districts": [flat(r, ["name", "overall_state_rank"]) for r in LISTS[p][0]]} for p in PERIODS}})

    # ---------------- offline bundle
    # Browsers block fetch() for pages opened straight from disk (file://),
    # but still run <script src> files. bundle.js carries every data file
    # keyed by its relative path, and index.html loads it only when opened
    # from disk, so double-clicking index.html works without a server.
    parts = []
    for f in sorted(list(OUT.rglob("*.json")) + list(OUT.rglob("*.geojson"))):
        rel = "data/" + f.relative_to(OUT).as_posix()
        parts.append(json.dumps(rel) + ":" + f.read_text(encoding="utf8"))
    (OUT / "bundle.js").write_text("window.SD_BUNDLE={" + ",".join(parts) + "};", encoding="utf8")

    # ---------------- checks printed for the build log
    print("\nMatching:")
    for k, v in meta["match_log"].items():
        print(f"  {k:28s} unmatched name pairs: {v['n_unmatched']}  e.g. {v['unmatched'][:6]}")
    print("\nSecond Chance check vs BBOSE_LokOS_Targets (state tp35 should be ~1,353,667):", state["secondchance"]["tp35"])
    for k, *_ in KPIS:
        s_ = pd.Series([blocks[i]["kpi"][k]["score"] for i in blocks], dtype=float)
        print(f"  {k:12s} blocks scored {s_.notna().sum():3d}  median score {s_.median():.1f}")
    ov = pd.Series([blocks[i]["overall_score"] for i in blocks], dtype=float)
    print(f"\nOverall score: {ov.notna().sum()} blocks, median {ov.median():.1f}")
    print("Done ->", OUT)

if __name__ == "__main__":
    main()
