import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import optuna
import pickle
import random
from pathlib import Path
import pandas as pd


from test import run_sim
from test_epw_mod import add_events, num_to_str, read_and_modify_epw, read_epw_to_df, save_df_to_csv_and_epw, EPW_COLS, CSV_HEADER_LABELS

# Seed
np.random.seed(42)


OUT_DIR = r"C:/EnergyPlusV25-2-0/TestResults/Run1"  


def update_epw(df, sel_inds, col_num_name, col_str_name, modified):
    df.loc[sel_inds, col_num_name] = modified

    # Update string column values using the num_to_str formatter
    orig_strings = df.loc[sel_inds, col_str_name].astype(str).values
    new_strings = [num_to_str(orig, nv) for orig, nv in zip(orig_strings, modified)]
    df.loc[sel_inds, col_str_name] = new_strings

    return df


def run_model(temp, ghi, target="Electricity:Facility"):
    
    # ---------- EDIT THESE ----------
    IDF_PATH = r"C:/EnergyPlusV25-2-0/TestResults/UnitHeater.idf"       # or your chosen IDF
    EPW_PATH = r"C:/EnergyPlusV25-2-0/WeatherData/USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw"
    MOD_EPW = r"C:/EnergyPlusV25-2-0/TestResults/temp_mod.epw"   # output modified EPW
    
    TARGET_YEAR = 1997                 # example: EPW year in file; actual EPW year used for indexing
    TARGET_MONTH = 1                   # July
    START_HOUR_IN_MONTH = 0            # 0-based offset into month (0 => first hour of month)
    N_HOURS = 100

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

    df = update_epw(
            df,
            sel_inds,
            col_num_name="DryBulb_C_num",
            col_str_name="DryBulb_C",
            modified=temp
    )
    
    df = update_epw(
        df,
        sel_inds,
        col_num_name="GlobalHorizontalRadiation_Wh_m2_num",
        col_str_name="GlobalHorizontalRadiation_Wh_m2",
        modified=ghi
    )
    
    save_df_to_csv_and_epw(df, header, csv_out="mod_epw.csv", epw_out=MOD_EPW, include_header=True)

    _, status = run_sim(IDF_PATH, MOD_EPW, OUT_DIR)

    if status == 0:
        csv_path = Path(OUT_DIR) / "eplusout.csv"
        # Read output CSV and find facility electricity meter column
        out_df = pd.read_csv(csv_path)
        # find a column with 'Electricity:Facility' or similar
        meter_cols = [c for c in out_df.columns if target in c or target in c.replace(" ", "")]
        print("Found meter columns:", meter_cols[:10])


        elec_col = meter_cols[0]
        print(len(out_df.columns), len(out_df[elec_col]))
        # elec_series = out_df.loc[sel_inds, elec_col].astype(float).values.copy()
        elec_series = out_df.loc[np.arange(N_HOURS), elec_col].astype(float).values.copy()
        return status, elec_series
    else:
        return status, []





# --- Plot Results ---
def plot_results(baseline_temp, baseline_ghi, baseline_infected, counterfactuals, t_span):
    times = np.arange(t_span[0], t_span[1])

    plt.figure(figsize=(16, 12))
    
    # temp
    plt.subplot(3, 1, 1)
    plt.plot(times, baseline_temp, label='Baseline temp', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(times, cf['temp'], label=f'CF {i+1} temp', alpha=0.7)
    plt.title("temp")
    plt.xlabel("Hours")
    plt.ylabel("temp")
    plt.legend()
    
    # ghi
    plt.subplot(3, 1, 2)
    plt.plot(times, baseline_ghi, label='Baseline ghi', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(times, cf['ghi'], label=f'CF {i+1} ghi', alpha=0.7)
    plt.title("ghi")
    plt.xlabel("Hours")
    plt.ylabel("ghi")
    plt.legend()
    
    # Infected curves
    plt.subplot(3, 1, 3)
    plt.plot(times, baseline_infected, label='Baseline Electricity', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(times, cf['Electricity'], label=f'CF {i+1} Electricity', alpha=0.7)
    plt.title("Electricity over time")
    plt.xlabel("Hours")
    plt.ylabel("Electricity (J)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("optuna_CFE_engplus.png")
    plt.show()


if __name__ == "__main__":
    # --- Baseline Parameters ---
    t_span = (0, 100)
    T = t_span[1]

    pickle_save = Path(OUT_DIR) / "baseline_engplus.p"
    [baseline_temp, baseline_ghi, baseline_E] = pickle.load(open(pickle_save, "rb"))
    # Global memory (only retains accepted low-MSE ones)
    archive_controls = []
    archive_mses = []

    # print(baseline_temp)
    # def objective(trial):
    #     global archive_controls, archive_mses
        
    #     temp_list = [trial.suggest_float(f"temp_{i}", 0, 1) for i in range(T)]
    #     ghi_list = [trial.suggest_float(f"ghi_{i}", 0, 1) for i in range(T)]
        

    #     unnorm_temp_list = [v * (40 - (-5)) + (-5) for v in temp_list]
    #     unnorm_ghi_list = [int(v * (1000 - (0)) + (0)) for v in ghi_list]

    #     status, sol_E = run_model(unnorm_temp_list, unnorm_ghi_list)
        
    #     if status != 0:
    #         mse = np.inf
    #     else:
    #         mse = np.mean((np.array(sol_E) - np.array(baseline_E)) ** 2)
    #         nmse = mse / np.var(baseline_E)

    #     # Loss 2: Diversity of temp and ghi from baseline
    #     diff_temp = np.mean(np.abs((np.array(unnorm_temp_list) - np.array(baseline_temp)) / np.array(baseline_temp)))
    #     diff_ghi = np.mean(np.abs((np.array(unnorm_ghi_list) - np.array(baseline_ghi)) / (1 + np.array(baseline_ghi))))
    #     diversity_from_baseline = diff_temp + diff_ghi

    #     # Loss 3: Diversity from previous trials
    #     diversity_from_others = 0
    #     if archive_controls:
    #         current = np.concatenate([temp_list, ghi_list])
    #         distances = [np.mean(np.abs(current - np.concatenate(past))) for past in archive_controls]
    #         # Use min distance to any previous as penalty (closer → more penalty)
    #         min_dist = min(distances)
    #         diversity_from_others = min_dist
    #     else:
    #         diversity_from_others = 1.0

    #     # --- Save candidate if good enough based on percentile ---
    #     archive_mses.append(nmse)
    #     threshold = np.percentile(archive_mses, 25)  # Keep only top 25% mse
    #     if nmse <= threshold:
    #         archive_controls.append((temp_list.copy(), ghi_list.copy()))


    #     return nmse - 0.1 * diversity_from_baseline - 0.1 * diversity_from_others

    def squash01(x):
        """Smooth, monotonic map [0,inf) -> [0,1)."""
        return x / (1.0 + x)

    def objective(trial):
        global archive_controls, archive_mses

        
        EPS = 1e-12

        # tunable weights (sum doesn't have to be 1, but these are reasonable starting points)
        w_mse = 0.7
        w_div_base = 0.2
        w_div_others = 0.1

        weeks = int(T // 3) + 1

        # sample (your existing code)
        week_temp_list = [trial.suggest_float(f"temp_{i}", 0, 1) for i in range(weeks)]
        week_ghi_list  = [trial.suggest_float(f"ghi_{i}", 0, 1) for i in range(weeks)]

        temp_list = list(np.repeat(np.array(week_temp_list), 3))[:T]
        ghi_list = list(np.repeat(np.array(week_ghi_list), 3))[:T]

        
        # unnormalize to physical ranges (your existing mapping)
        unnorm_temp_list = [v * (40 - (-5)) + (-5) for v in temp_list]   # in degC
        unnorm_ghi_list  = [int(v * (1000 - 0) + 0) for v in ghi_list]    # in Wh/m2

        status, sol_E = run_model(unnorm_temp_list, unnorm_ghi_list)

        if status != 0 or sol_E is None:
            # failed run -> very bad score
            return 1.0  # worst possible (since we'll keep objective small = better)
        
        # === 1) Normalized, bounded MSE ===
        mse = np.mean((np.array(sol_E) - np.array(baseline_E)) ** 2)
        var_baseline = np.var(baseline_E) + EPS
        nmse_raw = mse / var_baseline           # scale-invariant
        nmse = squash01(nmse_raw)               # now in [0,1)

        # === 2) Diversity from baseline (relative differences) ===
        # handle possible zeros in baseline by dividing by (abs(baseline)+1)
        denom_temp = np.abs(baseline_temp) + 1.0
        denom_ghi  = np.abs(baseline_ghi) + 1.0

        diff_temp_raw = np.mean(np.abs((np.array(unnorm_temp_list) - np.array(baseline_temp)) / denom_temp))
        diff_ghi_raw  = np.mean(np.abs((np.array(unnorm_ghi_list)  - np.array(baseline_ghi))  / denom_ghi))
        div_base_raw  = diff_temp_raw + diff_ghi_raw
        div_base = squash01(div_base_raw)       # bounded [0,1)

        # === 3) Diversity from previous candidates ===
        # archive_controls stores normalized lists in your earlier code (temp_list, ghi_list), so work in that space
        if archive_controls:
            current = np.concatenate([temp_list, ghi_list])
            distances = [np.mean(np.abs(current - np.concatenate(past))) for past in archive_controls]
            min_dist = float(min(distances))
            # min_dist already roughly in [0,1], but squash anyway for safety
            div_others = squash01(min_dist)
        else:
            div_others = 1.0   # encourage exploration early on

        # === Combine ===
        # We *minimize* objective. Lower is better.
        # nmse is the main loss; we subtract diversity terms (larger diversity → smaller objective)
        combined = w_mse * nmse - w_div_base * div_base - w_div_others * div_others

        # bookkeeping: store nmse for percentile selection if you want to keep that logic
        archive_mses.append(nmse)
        # optionally keep good controls by nmse threshold (as before)
        try:
            threshold = np.percentile(archive_mses, 25)
        except Exception:
            threshold = nmse
        if nmse <= threshold:
            archive_controls.append((temp_list.copy(), ghi_list.copy()))

        return combined


    # # ====== testing which variables are dependent =======
    # temp_list = [random.random() for i in range(T)]
    # ghi_list = [random.random() for i in range(T)]


    # unnorm_temp_list = [v * (40 - (-5)) + (-5) for v in temp_list]
    # unnorm_ghi_list = [int(v * (1000 - (0)) + (0)) for v in ghi_list]

    # status, sol_E = run_model(unnorm_temp_list, unnorm_ghi_list, target="Electricity:Facility")
    # temp_list = [i + 10 for i in unnorm_temp_list]
    # temp_list = [i - 50 for i in unnorm_ghi_list]
    # status, sol_E2 = run_model(temp_list, unnorm_ghi_list, target="Electricity:Facility")
    # print(">>", status)
    # plt.plot(sol_E)
    # plt.plot(sol_E2)
    # plt.show()
    # exit()

    # --- Run Optimization ---
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150)

    # --- Extract N best counterfactuals ---
    N = 30
    top_trials = sorted(study.trials, key=lambda x: x.value)[:N]

    counterfactuals = []
    for trial in top_trials:
        freq = int(T // 3) + 1
        week_temp_list = [trial.params[f"temp_{i}"] for i in range(freq)]
        week_ghi_list = [trial.params[f"ghi_{i}"] for i in range(freq)]
        
        temp_list = list(np.repeat(np.array(week_temp_list), 3))[:T]
        ghi_list = list(np.repeat(np.array(week_ghi_list), 3))[:T]

        
        unnorm_temp_list = [v * (35 - (0)) + (0) for v in temp_list]
        unnorm_ghi_list = [int(v * (500 - (0)) + (0)) for v in ghi_list]
        
        status, sol_E = run_model(unnorm_temp_list, unnorm_ghi_list)
        print(">>", status)
        if status == 0:
            mse = np.mean((np.array(sol_E) - np.array(baseline_E)) ** 2)
            nmse = mse / np.var(baseline_E)  
            counterfactuals.append({
                "temp": unnorm_temp_list,
                "ghi": unnorm_ghi_list,
                "Electricity": sol_E,
                "nmse": nmse
            })

    plt.plot([c["nmse"] for c in counterfactuals])
    plt.show()
    plot_results(baseline_temp, baseline_ghi, baseline_E, counterfactuals, t_span)
    # 3 is infected and good, 100 trials
    # 2 is infected and 300 trials, possibly good
    pickle.dump([counterfactuals, study.trials, baseline_temp, baseline_ghi, baseline_E], open("CFE_sols_150.p", "wb"))