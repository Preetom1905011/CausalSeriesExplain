import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

import scipy.stats as stats
from test import run_sim
from test_epw_mod import add_events, num_to_str, read_and_modify_epw, read_epw_to_df, save_df_to_csv_and_epw, EPW_COLS, CSV_HEADER_LABELS

def apply_events_to_epw_column(df, sel_inds, col_num_name, col_str_name,
                               peak_center_rel=0.3,
                               peak_amp=5.0,
                               peak_width=5,
                               plateau_start_rel=0.6,
                               plateau_value_increase=10.0):
    """
    Apply add_events to the numeric column `col_num_name` for rows in `sel_inds`.
    Then write back the numeric results into the string column `col_str_name` using num_to_str.
    - df: DataFrame returned by read_epw_to_df (has both string cols like 'DryBulb_C' and numeric 'DryBulb_C_num')
    - sel_inds: list / Index / slice of DataFrame indices to modify (can be DatetimeIndex or integer positions if df index numeric)
    - col_num_name: e.g. 'DryBulb_C_num'
    - col_str_name: e.g. 'DryBulb_C'
    Returns modified df (in-place) for convenience.
    """
    # Extract base numeric series for selected indices
    base_series = df.loc[sel_inds, col_num_name].astype(float).values.copy()

    # Generate modified series using add_events
    modified = add_events(base_series,
                          peak_center_rel=peak_center_rel,
                          peak_amp=peak_amp,
                          peak_width=peak_width,
                          plateau_start_rel=plateau_start_rel,
                          plateau_value_increase=plateau_value_increase)

    # Write numeric modifications back
    df.loc[sel_inds, col_num_name] = modified

    # Update string column values using the num_to_str formatter
    orig_strings = df.loc[sel_inds, col_str_name].astype(str).values
    new_strings = [num_to_str(orig, nv) for orig, nv in zip(orig_strings, modified)]
    df.loc[sel_inds, col_str_name] = new_strings

    return df


# ---------- EDIT THESE ----------
IDF_PATH = r"C:/EnergyPlusV25-2-0/TestResults/UnitHeater.idf"       # or your chosen IDF
EPW_PATH = r"C:/EnergyPlusV25-2-0/WeatherData/USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw"
MOD_EPW = r"C:/EnergyPlusV25-2-0/TestResults/washington_dulles_modified.epw"   # output modified EPW
OUT_DIR = r"C:/EnergyPlusV25-2-0/TestResults/Run1"          # EnergyPlus output dir

TARGET_YEAR = 1997                 # example: EPW year in file; actual EPW year used for indexing
TARGET_MONTH = 1                   # July
START_HOUR_IN_MONTH = 0            # 0-based offset into month (0 => first hour of month)
N_HOURS = 100

USE_SYNTH = False
# --------------------------------

# Use your read_epw_to_df function you already have:
# from your_module import read_epw_to_df, save_df_to_csv_and_epw
# If those funcs are in the same script, they're available directly.

header, df = read_epw_to_df(EPW_PATH)   # returns header_lines, df with EPW_COLS and numeric columns '<col>_num'

# 2) find target month/year rows and choose start index within month
month_mask = (df['Year_num'] == TARGET_YEAR) & (df['Month_num'] == TARGET_MONTH)
month_inds = df.index[month_mask]
if len(month_inds) == 0:
    raise RuntimeError("No rows found for requested month/year. Check EPW Year/Month values.")
# ensure we have enough hours in month
if START_HOUR_IN_MONTH + N_HOURS > len(month_inds):
    raise ValueError("Requested window exceeds month length. Reduce N_HOURS or START_HOUR_IN_MONTH.")
sel_inds = month_inds[START_HOUR_IN_MONTH : START_HOUR_IN_MONTH + N_HOURS]



# # Grab numeric series (copy to avoid mutating original)
temp = df.loc[sel_inds, "DryBulb_C_num"].astype(float).values.copy()
ghi  = df.loc[sel_inds, "GlobalHorizontalRadiation_Wh_m2_num"].astype(float).values.copy()
# relHu  = df.loc[sel_inds, "RelHumidity_pct_num"].astype(float).values.copy()
# atp  = df.loc[sel_inds, "AtmosPressure_Pa_num"].astype(float).values.copy()



# plt.plot(np.arange(N_HOURS), temp, label="DryBulb (modified)")
# plt.plot(np.arange(N_HOURS), ghi / 100.0, label="GHI/100 (scaled)")  # scale for plotting
# plt.plot(np.arange(N_HOURS), relHu, label="relHu")  # scale for plotting
# plt.plot(np.arange(N_HOURS), atp/10000, label="atp")  # scale for plotting
# plt.legend()
# plt.show()

# Produce two eventful signals
if USE_SYNTH:
    temp_mod = add_events(temp, peak_amp=6.0, peak_width=4, plateau_value_increase=3.0)
    ghi_mod  = add_events(ghi, peak_amp=500.0, peak_width=2, plateau_value_increase=300.0)
else:
    temp_mod = temp
    ghi_mod = ghi


# Write those back into the DataFrame numeric columns, and update string columns accordingly
# Apply events to Dry Bulb
if USE_SYNTH:
    df = apply_events_to_epw_column(
        df,
        sel_inds,
        col_num_name="DryBulb_C_num",
        col_str_name="DryBulb_C",
        peak_center_rel=0.35,
        peak_amp=6.0,
        peak_width=4,
        plateau_start_rel=0.65,
        plateau_value_increase=3.0
    )
    # Apply events to GHI
    df = apply_events_to_epw_column(
        df,
        sel_inds,
        col_num_name="GlobalHorizontalRadiation_Wh_m2_num",
        col_str_name="GlobalHorizontalRadiation_Wh_m2",
        peak_center_rel=0.25,
        peak_amp=500.0,
        peak_width=2,
        plateau_start_rel=0.6,
        plateau_value_increase=300.0
    )

# Save the modified EPW (include header so EnergyPlus recognizes it)
save_df_to_csv_and_epw(df, header, csv_out="mod_epw.csv", epw_out=MOD_EPW, include_header=True)


run_sim(IDF_PATH, MOD_EPW, OUT_DIR)


temp_up = df.loc[sel_inds, "DryBulb_C_num"].astype(float).values.copy()
ghi_up  = df.loc[sel_inds, "GlobalHorizontalRadiation_Wh_m2_num"].astype(float).values.copy()

csv_path = Path(OUT_DIR) / "eplusout.csv"
# Read output CSV and find facility electricity meter column
out_df = pd.read_csv(csv_path)
# find a column with 'Electricity:Facility' or similar
meter_cols = [c for c in out_df.columns if "Electricity:Facility" in c or "Electricity:Facility" in c.replace(" ", "")]
print("Found meter columns:", meter_cols[:10])


elec_col = meter_cols[0]
print(len(out_df.columns), len(out_df[elec_col]))
elec_series = out_df.loc[sel_inds, elec_col].astype(float).values.copy()

print(len(out_df.columns), len(out_df[elec_col]))

pickle_save = Path(OUT_DIR) / "baseline_engplus.p"
pickle.dump([temp_mod, ghi_mod, elec_series], open(pickle_save, "wb"))
print(">>>", len(temp_mod), len(ghi_mod), len(elec_series))


plt.plot(np.arange(N_HOURS), temp, label="DryBulb (modified)", linestyle="solid")
plt.plot(np.arange(N_HOURS), ghi / 100.0, label="GHI/100 (scaled)", linestyle="solid")  # scale for plotting
plt.plot(np.arange(N_HOURS), elec_series/10000000, label="Electricity", linestyle="solid")  # scale for plotting

plt.legend()
plt.show()

exit()

# The CSV to inspect is usually OUT_DIR/eplusout.csv (created by ReadVarsESO). If not, check OUT_DIR for eplusout.eso then run ReadVarsESO.
csv_path = Path(OUT_DIR) / "eplusout.csv"
if not csv_path.exists():
    raise FileNotFoundError("eplusout.csv not found. Check ReadVarsESO or eplusout.eso presence.")

# Read output CSV and find facility electricity meter column
out_df = pd.read_csv(csv_path)
# find a column with 'Electricity:Facility' or similar
meter_cols = [c for c in out_df.columns if "Electricity:Facility" in c or "Electricity:Facility" in c.replace(" ", "")]
print("Found meter columns:", meter_cols[:10])

if not meter_cols:
    # fallback: print first 50 cols to inspect
    print(out_df.columns[:50])
    raise ValueError("Couldn't find Electricity:Facility column automatically; inspect column names printed above.")

# Extract the facility electricity time series and align with our 100 points
elec_col = meter_cols[0]
elec_series = out_df[elec_col].astype(float).values

# eplusout.csv often contains many rows (timesteps). Choose a slicing that matches your EPW modification window.
# If readvars produced hourly and your sim uses hourly, then start at same index
elec_window = elec_series[start_idx:end_idx]

corr_temp = np.corrcoef(temp_mod, elec_window)[0,1]
corr_ghi  = np.corrcoef(ghi_mod, elec_window)[0,1]

print(f"Pearson corr temp vs electricity (100 pts): {corr_temp:.3f}")
print(f"Pearson corr GHI  vs electricity (100 pts): {corr_ghi:.3f}")

# Plot signals
plt.figure(figsize=(10,6))
plt.plot(np.arange(n_points), temp_mod, label="DryBulb (modified)")
plt.plot(np.arange(n_points), ghi_mod / 100.0, label="GHI/100 (scaled)")  # scale for plotting
plt.plot(np.arange(n_points), elec_window / np.max(elec_window) * np.max(temp_mod), label="Electricity (scaled)")
plt.legend()
plt.xlabel("Hour index (relative)")
plt.title("Inputs and Facility Electricity (scaled)")
plt.show()