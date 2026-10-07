"""
build_sd_inputs.py - heavy pre-aggregation step for the Social Development
Tracker. Reads the large LokOS / VPRP .dta files ONCE and writes small
block-level tables to 2_Data/Processed/sd_tracker/, which build_sd_data.py
then reads (in seconds) to produce the tracker JSON.

Run this only when the underlying LokOS or VPRP data is refreshed - it
takes roughly 20-30 minutes (the VPRP entitlements file alone is 3.8 GB).

Everything here is keyed on the RAW (district, block) names of each source.
Matching to the tracker's 534-block universe happens in build_sd_data.py,
so a matching fix never requires re-running this slow step.

Outputs (all pandas pickles):
  members_block.pkl      member counts, SC/ST, disability, PWD-SHG inclusion
  members_disability.pkl disabled members by type and by social category
  members_edu_age.pkl    Second Chance target population building blocks
  shg_block.pkl          SHG counts / VO link / bank account / credit link, by special type
  pwd_formation.pkl      PWD SHGs by formation year
  vo_block.pkl           active VOs per block
  vprp_vo.pkl            one row per (district, block, year, vo_name, part)
  vprp_ent_block.pkl     entitlement requests + unique requesters (SC/ST split), per block-year
  vprp_scheme.pkl        entitlement requests by scheme, per block-year
  vprp_pgsrd.pkl         PGSRD demands by type and item, per block-year
  vprp_sdp.pkl           SDP issues by sector / issue / department, per block-year
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]          # .../6_LokOS_Analysis
CLEAN = BASE / "2_Data" / "Cleaned"
OUT = BASE / "2_Data" / "Processed" / "sd_tracker"
OUT.mkdir(parents=True, exist_ok=True)
AGE_DATE = pd.Timestamp("2026-09-01")
T0 = time.time()

def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)

def s(x):
    return x.astype(str).str.strip()

def num(x):
    return pd.to_numeric(x.astype(str).str.strip(), errors="coerce")

# ---------------------------------------------------------------- SHGs
log("SHG profiles")
shg = pd.read_stata(CLEAN / "lokos_shg_profiles.dta",
                    columns=["district", "block", "shg_code", "special_shg_type", "parent_vo",
                             "shg_account_type_1", "shg_account_type_2", "shg_active",
                             "shg_active_members", "shg_formation_date"])
shg["type"] = s(shg.special_shg_type).replace({"": "REGULAR", "nan": "REGULAR"})
shg["vo_linked"] = s(shg.parent_vo).ne("") & s(shg.parent_vo).ne("nan")
shg["bank"] = s(shg.shg_account_type_1).ne("") & s(shg.shg_account_type_1).ne("nan")
shg["ccl"] = shg.shg_account_type_1.isin(["CCL (Default)", "Combo (Default)"]) | \
             s(shg.shg_account_type_2).str.contains("CCL|Combo", regex=True)
shg["form_year"] = shg.shg_formation_date.dt.year
g = shg.groupby(["district", "block", "type"])
shg_block = g.agg(n=("shg_code", "size"), vo_linked=("vo_linked", "sum"), bank=("bank", "sum"),
                  ccl=("ccl", "sum"), members=("shg_active_members", "sum"),
                  formed_2025plus=("form_year", lambda y: (y >= 2025).sum())).reset_index()
shg_block.to_pickle(OUT / "shg_block.pkl")
pwd = shg[shg.type == "PWD"]
pwd.groupby(["district", "block", "form_year"]).size().rename("n").reset_index().to_pickle(OUT / "pwd_formation.pkl")
pwd_codes = set(pwd.shg_code)
log(f"  {len(shg):,} SHGs, {len(pwd_codes):,} PWD")
del shg

# ---------------------------------------------------------------- VOs
log("VO profiles")
vo = pd.read_stata(CLEAN / "lokos_vo_profiles.dta", columns=["district", "block", "vo_code", "vo_active"])
vo[vo.vo_active == 1].groupby(["district", "block"]).size().rename("active_vos").reset_index().to_pickle(OUT / "vo_block.pkl")

# ---------------------------------------------------------------- members
log("Members (chunked)")
cols = ["shg_code", "district", "block", "member_active", "member_disability_self",
        "member_disability_family", "member_disability_type", "member_social_category",
        "member_education", "member_dob"]
it = pd.read_stata(CLEAN / "lokos_members_clean.dta", columns=cols, chunksize=2_000_000, convert_categoricals=True)
with pd.io.stata.StataReader(CLEAN / "lokos_members_clean.dta") as r:
    vlab = r.value_labels()
lab_soc = vlab.get("lbl_socialcategory", {})
lab_edu = vlab.get("lbl_education", {})
lab_dtype = next((v for k, v in vlab.items() if "disab" in k.lower() and "type" in k.lower()), {})

blk, dis, edu = [], [], []
for ch in it:
    soc = ch.member_social_category.map(lab_soc) if lab_soc and ch.member_social_category.dtype.kind in "if" else s(ch.member_social_category)
    eduv = ch.member_education.map(lab_edu) if lab_edu and ch.member_education.dtype.kind in "if" else s(ch.member_education)
    dtyp = ch.member_disability_type.map(lab_dtype) if lab_dtype and ch.member_disability_type.dtype.kind in "if" else s(ch.member_disability_type)
    d = pd.DataFrame({
        "district": ch.district, "block": ch.block,
        "active": num(ch.member_active) == 1,
        "scst": soc.isin(["SC", "ST"]),
        "dis": num(ch.member_disability_self) == 1,
        "dis_fam": num(ch.member_disability_family) == 1,
        "in_pwd": ch.shg_code.isin(pwd_codes),
    })
    d["dis_in_pwd"] = d.dis & d.in_pwd
    blk.append(d.groupby(["district", "block"]).agg(
        members=("active", "size"), active_members=("active", "sum"), scst_members=("scst", "sum"),
        dis_self=("dis", "sum"), dis_family=("dis_fam", "sum"), in_pwd_members=("in_pwd", "sum"),
        dis_in_pwd=("dis_in_pwd", "sum")))
    # disabled members by type / social category
    dd = pd.DataFrame({"district": ch.district, "block": ch.block, "dtype": dtyp, "soc": soc})[d.dis.values]
    dis.append(dd.groupby(["district", "block", "dtype", "soc"]).size().rename("n"))
    # Second Chance building blocks: education x age band x category
    age = (AGE_DATE - pd.to_datetime(ch.member_dob, errors="coerce")).dt.days / 365.25
    band = pd.cut(age, [0, 25, 30, 35, 200], right=False, labels=["lt25", "25_30", "30_35", "35plus"])
    cat = np.where(soc.isin(["SC", "ST"]), "SCST", np.where(soc == "OBC", "OBC", "OTHER"))
    ee = pd.DataFrame({"district": ch.district, "block": ch.block, "edu": eduv, "band": band.astype(str), "cat": cat})
    ee = ee[ee.edu.isin(["CLASS - 2", "CLASS - 5", "CLASS - 7", "CLASS - 8"])]
    edu.append(ee.groupby(["district", "block", "edu", "band", "cat"]).size().rename("n"))
    log(f"  chunk done ({sum(len(x) for x in blk)} block rows so far)")

pd.concat(blk).groupby(level=[0, 1]).sum().reset_index().to_pickle(OUT / "members_block.pkl")
pd.concat(dis).groupby(level=[0, 1, 2, 3]).sum().reset_index().to_pickle(OUT / "members_disability.pkl")
pd.concat(edu).groupby(level=[0, 1, 2, 3, 4]).sum().reset_index().to_pickle(OUT / "members_edu_age.pkl")
log("Members done")

# ---------------------------------------------------------------- VPRP: PGSRD + SDP (small)
log("VPRP PGSRD / SDP")
pg = pd.read_stata(CLEAN / "clf_pgsrd_requests.dta", columns=["year", "district", "block", "vo_name", "pgsrd_type", "item_demanded"])
sd = pd.read_stata(CLEAN / "clf_sdp.dta")
pg.groupby(["district", "block", "year", "pgsrd_type", "item_demanded"]).size().rename("n").reset_index().to_pickle(OUT / "vprp_pgsrd.pkl")
dep_cols = [c for c in sd.columns if c.startswith("department_")]
sdl = sd.melt(id_vars=["district", "block", "year", "sector", "social_issue"], value_vars=dep_cols, value_name="department")
sdl = sdl[s(sdl.department).ne("") & sdl.department.notna()]
sdp_out = {
    "issues": sd.groupby(["district", "block", "year", "sector", "social_issue"]).size().rename("n").reset_index(),
    "departments": sdl.groupby(["district", "block", "year", "department"]).size().rename("n").reset_index(),
}
pd.to_pickle(sdp_out, OUT / "vprp_sdp.pkl")
vo_parts = [pg[["district", "block", "year", "vo_name"]].assign(part="pgsrd"),
            sd[["district", "block", "year", "vo_name"]].assign(part="sdp")]

# ---------------------------------------------------------------- VPRP: entitlements (3.8 GB)
log("VPRP entitlements (chunked)")
ecols = ["year", "district", "block", "vo_name", "shg_name", "shg_membername", "social_category", "scheme_type", "state_scheme"]
it = pd.read_stata(CLEAN / "clf_vprp_entitlements.dta", columns=ecols, chunksize=1_500_000, convert_categoricals=True)
with pd.io.stata.StataReader(CLEAN / "clf_vprp_entitlements.dta") as r:
    evl = r.value_labels()
def lab(col, ser):
    for k, v in evl.items():
        if col.split("_")[0] in k.lower() and ser.dtype.kind in "if":
            return ser.map(v)
    return ser
sch, req, vos = [], [], []
for ch in it:
    for c in ["social_category", "scheme_type", "state_scheme"]:
        if ch[c].dtype.kind in "if":
            ch[c] = lab(c, ch[c])
    sch.append(ch.groupby(["district", "block", "year", "scheme_type"]).size().rename("n"))
    st = ch[s(ch.state_scheme).ne("") & ch.state_scheme.notna()]
    sch.append(st.groupby(["district", "block", "year", "state_scheme"]).size().rename("n").rename_axis(index={"state_scheme": "scheme_type"}))
    # vprp_cleaning.do codes social_category 1=GENERAL 2=SC 3=ST 4=OBC 5=Other/Minority
    ch["scst"] = num(ch.social_category).isin([2, 3]) | s(ch.social_category).str.upper().isin(["SC", "ST"])
    ch["cat_known"] = num(ch.social_category).notna()
    req.append(ch[["district", "block", "year", "vo_name", "shg_name", "shg_membername", "scst", "cat_known"]].drop_duplicates())
    vos.append(ch[["district", "block", "year", "vo_name"]].drop_duplicates())
    log(f"  chunk ({len(ch):,} rows)")
pd.concat(sch).groupby(level=[0, 1, 2, 3]).sum().reset_index().to_pickle(OUT / "vprp_scheme.pkl")
rq = pd.concat(req).drop_duplicates(subset=["district", "block", "year", "vo_name", "shg_name", "shg_membername"])
n_req = pd.concat(sch[0::2]).groupby(level=[0, 1, 2]).sum().rename("requests")
ent = rq.groupby(["district", "block", "year"]).agg(requesters=("scst", "size"), scst_requesters=("scst", "sum"),
                                                     requesters_cat_known=("cat_known", "sum")).join(n_req).reset_index()
ent.to_pickle(OUT / "vprp_ent_block.pkl")
vo_parts.append(pd.concat(vos).drop_duplicates().assign(part="ent"))
pd.concat(vo_parts).drop_duplicates().to_pickle(OUT / "vprp_vo.pkl")
log("All done")
