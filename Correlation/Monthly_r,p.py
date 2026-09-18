
"""
Created on Tue Nov  4 02:38:54 2025

@author: Jams
"""


#Calculates Pearson correlation (r, p) between monthly regional climate variable for each station and multiple SST indices (Nino1+2, Nino3, Nino3.4, Nino4, DMI, MEI.v2).
#Now includes smart matching to correctly merge Lat/Lon for all stations.

import pandas as pd
from scipy.stats import pearsonr
from difflib import get_close_matches

# === Input files ===
tmax_file = "tmin_monthly.xlsx"
sst_file = "Nino_3_3.4_4_(HadISST)_from 1982.xlsx"
coord_file = "Lat and Lon of stations.xlsx"

# === Load datasets ===
tmax_df = pd.read_excel(tmax_file)
sst_df = pd.read_excel(sst_file)
coord_df = pd.read_excel(coord_file)

# === Clean coordinate station names ===
coord_df["Station"] = coord_df["Station"].str.strip().str.replace("_", " ").str.replace("-", " ")

# === Convert wide Tmax to long ===
tmax_long = tmax_df.melt(id_vars=["Year", "Month"], var_name="Station", value_name="Tmax")

# Clean Tmax station names (remove suffixes and normalize)
tmax_long["Station"] = (
    tmax_long["Station"]
    .str.replace("_merged_data", "", regex=False)
    .str.replace("_", " ")
    .str.strip()
)

# === Attempt direct merge first ===
merged_tmax = pd.merge(tmax_long, coord_df, on="Station", how="left")

# === Fuzzy match any missing coordinates ===
missing = merged_tmax[merged_tmax["Lat"].isna()]["Station"].unique()

if len(missing) > 0:
    print("\n⚠️ Some stations missing coordinates, applying fuzzy matching...")
    for st in missing:
        match = get_close_matches(st, coord_df["Station"], n=1, cutoff=0.7)
        if match:
            lat = coord_df.loc[coord_df["Station"] == match[0], "Lat"].values[0]
            lon = coord_df.loc[coord_df["Station"] == match[0], "Lon"].values[0]
            merged_tmax.loc[merged_tmax["Station"] == st, ["Lat", "Lon"]] = [lat, lon]
            print(f"   Matched '{st}' → '{match[0]}'")
        else:
            print(f"   ⚠️ No close match found for '{st}'")

# === Merge with SST data ===
merged = pd.merge(merged_tmax, sst_df, on=["Year", "Month"], how="inner")

# === Identify SST columns ===
sst_cols = [c for c in sst_df.columns if c not in ["Year", "Month"]]

# === Calculate correlations ===
stations = merged["Station"].unique()

for sst in sst_cols:
    results = []
    for st in stations:
        sub = merged[merged["Station"] == st].dropna(subset=["Tmax", sst])
        if len(sub) > 2:
            r, p = pearsonr(sub["Tmax"].values, sub[sst].values)
            lat = sub["Lat"].iloc[0]
            lon = sub["Lon"].iloc[0]
            results.append({"Station": st, "Lat": lat, "Lon": lon, "r_value": r, "p_value": p})

    results_df = pd.DataFrame(results)
    out_name = f"Pearson_r_p_{sst}.xlsx"
    results_df.to_excel(out_name, index=False)
    print(f"✅ Saved: {out_name}")

print("\n🎯 All SST correlation files created successfully!")
