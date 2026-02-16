import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import pickle

from test_CFE_optuna import run_model

# --- Plot Results with Clusters ---
def plot_results_clustered(baseline_phi_s1, baseline_phi_a1, baseline_infected, clustered_cfs, t_span, cluster_colors):
    n_weeks = len(baseline_phi_s1)
    weeks = np.arange(n_weeks)
    times = np.arange(t_span[0], t_span[1])


    # # keep adding generated input from LLM
    GPT_gen = {0: {"temp":  [18.4, 14.2, 10.1, 6.8, 8.9, 12.7, 16.5, 9.5, 10.7, 11.9, 13.2, 14.6, 16.1, 17.7, 12.8, 13.4, 12.9, 13.6, 13.1, 13.8, 13.3, 7.2, 6.1, 8.4, 7.0, 9.1, 7.8, 8.7, 8.6, 10.1, 11.8, 13.8, 16.0, 18.3],
                   "ghi":  [280, 220, 165, 120, 150, 205, 265, 332, 336, 334, 337, 335, 338, 336, 240, 198, 160, 126, 154, 192, 236, 145, 152, 160, 410, 168, 154, 148, 210, 224, 238, 252, 266, 244] 
                   },
               1: {"temp": [8.2, 7.9, 8.5, 9.1, 8.7, 9.4, 9.0, 12.6, 13.1, 13.8, 14.2, 13.6, 14.5, 13.9, 13.2, 12.8, 13.5, 14.0, 13.4, 12.9, 13.7, 9.6, 10.2, 9.1, 8.8, 9.7, 10.4, 9.3, 10.1, 11.0, 12.2, 13.6, 15.1, 16.7],
                   "ghi": [92, 88, 85, 90, 94, 98, 102, 176, 184, 192, 205, 198, 212, 206, 158, 150, 142, 136, 144, 152, 160, 186, 194, 208, 221, 214, 226, 218, 324, 329, 327, 331, 328, 330]
                   },
               2: {"temp":  [6.4, 6.1, 5.8, 6.0, 5.7, 6.3, 6.6, 8.1, 8.7, 9.2, 9.8, 10.1, 10.6, 11.0, 12.4, 13.1, 13.8, 14.4, 13.9, 14.7, 15.2, 9.6, 9.1, 8.7, 8.4, 9.0, 9.7, 10.3, 12.2, 12.8, 13.4, 13.9, 14.3, 14.8],
                   "ghi": [78, 74, 70, 73, 76, 80, 85, 122, 130, 138, 145, 152, 160, 168, 210, 182, 155, 132, 150, 178, 208, 172, 150, 132, 118, 136, 158, 176, 304, 307, 309, 311, 310, 308]
                   },
               3: {"temp":  [5.9, 5.1, 4.6, 4.2, 4.8, 5.5, 6.3, 8.4, 9.1, 10.2, 8.8, 11.0, 9.7, 10.6, 14.0, 12.0, 10.1, 11.3, 13.2, 15.1, 16.8, 8.5, 9.7, 10.6, 8.9, 11.2, 9.4, 10.1, 11.0, 12.2, 13.4, 14.6, 15.9, 17.1],
                   "ghi":  [180, 150, 120, 90, 110, 145, 175, 260, 230, 200, 170, 195, 225, 255, 185, 205, 195, 215, 200, 220, 210, 190, 205, 420, 210, 195, 225, 205, 230, 245, 260, 430, 255, 240]
                   },
               4: {"temp": [5.8, 4.9, 5.3, 4.4, 5.1, 5.9, 6.2, 10.4, 8.1, 6.3, 9.2, 12.6, 15.4, 17.8, 9.4, 10.8, 12.1, 8.7, 11.3, 12.9, 10.2, 13.2, 11.0, 9.1, 8.4, 10.1, 12.3, 13.8, 12.0, 13.1, 14.0, 15.5, 16.8, 18.2],
                "ghi":  [110, 92, 84, 76, 88, 101, 118, 185, 198, 430, 205, 188, 214, 196, 242, 208, 171, 149, 176, 207, 236, 95, 88, 80, 72, 78, 86, 94, 220, 240, 410, 250, 230, 245]},
                
                5: {"temp": [4.2, 3.8, 4.5, 6.0, 8.5, 11.2, 10.5, 7.9, 11.6, 9.2, 10.1, 8.8, 13.5, 14.8, 13.9, 12.6, 9.4, 12.9, 11.0, 13.7, 16.2, 14.2, 15.6, 14.7, 13.1, 10.2, 13.8, 12.4, 15.9, 18.6, 9.8, 11.4, 10.6, 5.3],
                    "ghi":  [90, 72, 84, 140, 410, 165, 170, 145, 185, 160, 430, 175, 230, 255, 240, 95, 82, 88, 180, 445, 205, 250, 270, 245, 100, 76, 92, 210, 460, 230, 220, 180, 235, 165]
                },
                6: {"temp": [5.2, 4.6, 5.8, 6.2, 8.9, 11.5, 12.4, 5.6, 4.9, 5.8, 12.8, 14.1, 13.4, 14.9, 10.8, 12.6, 5.1, 4.7, 12.2, 13.7, 12.9, 14.4, 10.9, 12.1, 5.4, 4.8, 7.2, 8.8, 10.4, 11.9, 10.7, 12.3, 5.0, 4.5],
                    "ghi":   [92, 78, 105, 140, 420, 165, 155, 88, 76, 102, 315, 338, 322, 330, 150, 182, 84, 74, 318, 334, 327, 321, 148, 176, 90, 79, 300, 308, 304, 312, 152, 178, 86, 73]
                }
              }
    
    GPT_gen = {0: GPT_gen[list(GPT_gen.keys())[-1]]}
    I_GPT = []
    for i in range(len(GPT_gen)):
        T = 100
        week_temp_list, week_ghi_list = GPT_gen[i]["temp"], GPT_gen[i]["ghi"]
        
        unnorm_temp_list = list(np.repeat(np.array(week_temp_list), 3))[:T]
        unnorm_ghi_list = list(np.repeat(np.array(week_ghi_list), 3))[:T]
        
        status, sol_GPT = run_model(unnorm_temp_list, unnorm_ghi_list)
        print(">>", status, len(sol_GPT))
        if status == 0:
            I_GPT.append(sol_GPT)
        else:
            I_GPT.append([])
        
            
        
    style = ["solid", "dotted", "dashdot", "dashed", (0, (5, 10)), (0, (3, 1, 1, 1, 1, 1)), "dashed", "dashed", "dashed"]

    plt.figure(figsize=(14, 12))
    
    # Group CFs by cluster
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for cf, cluster in clustered_cfs:
        cluster_dict[cluster].append(cf)

    # Convert baseline weekly phi rates to daily stepwise
    days = np.arange(len(baseline_phi_s1))  # Daily time axis

    clustering_context = {v: {} for v in range(len(cluster_dict))}

    # --- X_s (phi_s1) ---
    marks = ["o", "v", "s"]
    plt.subplot(3, 1, 1)
    # plt.plot(days, baseline_phi_s1_daily, label='Baseline $X_s$', color='black', linewidth=2)
    for cluster_id, cf_list in sorted(cluster_dict.items()):
        data = np.array([cf['temp'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(days, mean, color=cluster_colors[cluster_id],  marker=marks[cluster_id], linewidth=2, label=f'Cluster {cluster_id} mean', alpha=0.5)
        plt.fill_between(days, mean - std, mean + std, color=cluster_colors[cluster_id], alpha=0.2)

        clustering_context[cluster_id]["temp"] = list(mean)
    
    # GPT CFE for X_s
    for i in range(len(GPT_gen)):
        plt.plot(days, np.repeat(GPT_gen[i]["temp"], 3)[:T], color="red", linestyle=style[i], linewidth=2, label=r'LLM Generated $X_{s, final}$')
    
    # plt.title(r"$X_s$")
    # plt.xlabel("Hours", fontsize=16)
    plt.ylabel(r"$X_{d}$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc="upper left")
    
    # --- X_a (phi_a1) ---
    plt.subplot(3, 1, 2)
    # plt.plot(days, baseline_phi_a1_daily, label='Baseline $X_a$', color='black', linewidth=2)
    for cluster_id, cf_list in sorted(cluster_dict.items()):
        data = np.array([cf['ghi'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(days, mean, color=cluster_colors[cluster_id], marker=marks[cluster_id], linewidth=2, label=f'Cluster {cluster_id} mean', alpha=0.5)
        plt.fill_between(days, mean - std, mean + std, color=cluster_colors[cluster_id], alpha=0.2)
        
        clustering_context[cluster_id]["ghi"] = list(mean)
    
    # GPT CFE for X_a
    for i in range(len(GPT_gen)):
        plt.plot(days, np.repeat(GPT_gen[i]["ghi"], 3)[:T], color="red",  linewidth=3, linestyle=style[i], label=r'LLM Generated $X_{a, final}$')
    
    # plt.title(r"$X_a$", fontsize=14)
    # plt.xlabel("Hours", fontsize=16)
    plt.ylabel(r"$X_{g}$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc="upper left")
    
    # --- Y (Infected) ---
    plt.subplot(3, 1, 3)
    plt.plot(times, baseline_infected, label=r'$Y_{base}$', color='black', linewidth=3)
    for cluster_id, cf_list in sorted(cluster_dict.items()):
        data = np.array([cf['Electricity'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(times, mean, color=cluster_colors[cluster_id],  marker=marks[cluster_id], linewidth=2, label=f'Cluster {cluster_id} mean', alpha=0.5)
        plt.fill_between(times, mean - std, mean + std, color=cluster_colors[cluster_id], alpha=0.2)

        clustering_context[cluster_id]["Electricity"] = list(mean)
    
    # GPT CFE for Y
    for i in range(len(I_GPT)):
        plt.plot(times, I_GPT[i], color="red", linestyle=style[i],  linewidth=2, label=r"LLM Generated $Y_{final}$")
    
    # plt.title(r"$Y$ (Infected over time)")
    plt.xlabel("Hours", fontsize=20)
    plt.ylabel(r"$Y$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc="upper left")
    plt.tight_layout()
    plt.savefig("optuna_CFE_pysirtem_clustered_100_gpt41_new.png", dpi=300)
    plt.show()

    print(f">>>>>\n {[int(float(val) * 100) / 100 for val in I_GPT[-1]]}")
    # pickle.dump([GPT_gen, I_GPT, clustering_context], open("CFE_sols_150_gpt_gen.p", "wb"))




# seperate plots with clusters
def plot_results_clustered_sep(baseline_phi_s1, baseline_phi_a1, baseline_infected, clustered_cfs, t_span, cluster_colors):
    n_weeks = len(baseline_phi_s1)
    weeks = np.arange(n_weeks)
    times = np.arange(t_span[0], t_span[1])


    # # keep adding generated input from LLM
    GPT_gen = {0: {"temp":  [18.4, 14.2, 10.1, 6.8, 8.9, 12.7, 16.5, 9.5, 10.7, 11.9, 13.2, 14.6, 16.1, 17.7, 12.8, 13.4, 12.9, 13.6, 13.1, 13.8, 13.3, 7.2, 6.1, 8.4, 7.0, 9.1, 7.8, 8.7, 8.6, 10.1, 11.8, 13.8, 16.0, 18.3],
                   "ghi":  [280, 220, 165, 120, 150, 205, 265, 332, 336, 334, 337, 335, 338, 336, 240, 198, 160, 126, 154, 192, 236, 145, 152, 160, 410, 168, 154, 148, 210, 224, 238, 252, 266, 244] 
                   },
               1: {"temp": [8.2, 7.9, 8.5, 9.1, 8.7, 9.4, 9.0, 12.6, 13.1, 13.8, 14.2, 13.6, 14.5, 13.9, 13.2, 12.8, 13.5, 14.0, 13.4, 12.9, 13.7, 9.6, 10.2, 9.1, 8.8, 9.7, 10.4, 9.3, 10.1, 11.0, 12.2, 13.6, 15.1, 16.7],
                   "ghi": [92, 88, 85, 90, 94, 98, 102, 176, 184, 192, 205, 198, 212, 206, 158, 150, 142, 136, 144, 152, 160, 186, 194, 208, 221, 214, 226, 218, 324, 329, 327, 331, 328, 330]
                   },
               2: {"temp":  [6.4, 6.1, 5.8, 6.0, 5.7, 6.3, 6.6, 8.1, 8.7, 9.2, 9.8, 10.1, 10.6, 11.0, 12.4, 13.1, 13.8, 14.4, 13.9, 14.7, 15.2, 9.6, 9.1, 8.7, 8.4, 9.0, 9.7, 10.3, 12.2, 12.8, 13.4, 13.9, 14.3, 14.8],
                   "ghi": [78, 74, 70, 73, 76, 80, 85, 122, 130, 138, 145, 152, 160, 168, 210, 182, 155, 132, 150, 178, 208, 172, 150, 132, 118, 136, 158, 176, 304, 307, 309, 311, 310, 308]
                   },
               3: {"temp":  [5.9, 5.1, 4.6, 4.2, 4.8, 5.5, 6.3, 8.4, 9.1, 10.2, 8.8, 11.0, 9.7, 10.6, 14.0, 12.0, 10.1, 11.3, 13.2, 15.1, 16.8, 8.5, 9.7, 10.6, 8.9, 11.2, 9.4, 10.1, 11.0, 12.2, 13.4, 14.6, 15.9, 17.1],
                   "ghi":  [180, 150, 120, 90, 110, 145, 175, 260, 230, 200, 170, 195, 225, 255, 185, 205, 195, 215, 200, 220, 210, 190, 205, 420, 210, 195, 225, 205, 230, 245, 260, 430, 255, 240]
                   },
               4: {"temp": [5.8, 4.9, 5.3, 4.4, 5.1, 5.9, 6.2, 10.4, 8.1, 6.3, 9.2, 12.6, 15.4, 17.8, 9.4, 10.8, 12.1, 8.7, 11.3, 12.9, 10.2, 13.2, 11.0, 9.1, 8.4, 10.1, 12.3, 13.8, 12.0, 13.1, 14.0, 15.5, 16.8, 18.2],
                "ghi":  [110, 92, 84, 76, 88, 101, 118, 185, 198, 430, 205, 188, 214, 196, 242, 208, 171, 149, 176, 207, 236, 95, 88, 80, 72, 78, 86, 94, 220, 240, 410, 250, 230, 245]},
                
                5: {"temp": [4.2, 3.8, 4.5, 6.0, 8.5, 11.2, 10.5, 7.9, 11.6, 9.2, 10.1, 8.8, 13.5, 14.8, 13.9, 12.6, 9.4, 12.9, 11.0, 13.7, 16.2, 14.2, 15.6, 14.7, 13.1, 10.2, 13.8, 12.4, 15.9, 18.6, 9.8, 11.4, 10.6, 5.3],
                    "ghi":  [90, 72, 84, 140, 410, 165, 170, 145, 185, 160, 430, 175, 230, 255, 240, 95, 82, 88, 180, 445, 205, 250, 270, 245, 100, 76, 92, 210, 460, 230, 220, 180, 235, 165]
                },
                6: {"temp": [5.2, 4.6, 5.8, 6.2, 8.9, 11.5, 12.4, 5.6, 4.9, 5.8, 12.8, 14.1, 13.4, 14.9, 10.8, 12.6, 5.1, 4.7, 12.2, 13.7, 12.9, 14.4, 10.9, 12.1, 5.4, 4.8, 7.2, 8.8, 10.4, 11.9, 10.7, 12.3, 5.0, 4.5],
                    "ghi":   [92, 78, 105, 140, 420, 165, 155, 88, 76, 102, 315, 338, 322, 330, 150, 182, 84, 74, 318, 334, 327, 321, 148, 176, 90, 79, 300, 308, 304, 312, 152, 178, 86, 73]
                }
              }
    
    GPT_gen = {0: GPT_gen[list(GPT_gen.keys())[-1]]}
    I_GPT = []
    T = 100
    for i in range(len(GPT_gen)):
        week_temp_list, week_ghi_list = GPT_gen[i]["temp"], GPT_gen[i]["ghi"]
        
        unnorm_temp_list = list(np.repeat(np.array(week_temp_list), 3))[:T]
        unnorm_ghi_list = list(np.repeat(np.array(week_ghi_list), 3))[:T]
        
        status, sol_GPT = run_model(unnorm_temp_list, unnorm_ghi_list)
        print(">>", status, len(sol_GPT))
        if status == 0:
            I_GPT.append(sol_GPT)
        else:
            I_GPT.append([])
        
            
        
    style = ["solid", "dotted", "dashdot", "dashed", (0, (5, 10)), (0, (3, 1, 1, 1, 1, 1)), "dashed", "dashed", "dashed"]
    
    # Group CFs by cluster
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for cf, cluster in clustered_cfs:
        cluster_dict[cluster].append(cf)

    # Convert baseline weekly phi rates to daily stepwise
    days = np.arange(len(baseline_phi_s1))  # Daily time axis

    clustering_context = {v: {} for v in range(len(cluster_dict))}


    # --- X_s (phi_s1) ---
    marks = ["o", "v", "s"]
    # Increase base font sizes by +2
    label_fs = 16
    tick_fs = 16
    legend_fs = 16

    # --- X_s (phi_s1) ---
    plt.figure(figsize=(12, 5))

    for cluster_id, cf_list in cluster_dict.items():
        data = np.array([cf['temp'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(days, mean,
                color=cluster_colors[cluster_id],
                marker=marks[cluster_id],
                linewidth=2,
                label=f'Cluster {cluster_id} mean',
                alpha=0.5)
        plt.fill_between(days, mean - std, mean + std,
                        color=cluster_colors[cluster_id],
                        alpha=0.2)

    for i in range(len(GPT_gen)):
        plt.plot(days,
                list(np.repeat(GPT_gen[i]["temp"], 3))[:T],
                color="red",
                linestyle=style[i],
                linewidth=3,
                label=r'LLM Generated $X_{d, final}$')

    plt.xlabel("Days", fontsize=label_fs)
    plt.ylabel(r"$X_s$", fontsize=label_fs+4)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=legend_fs, loc="upper left")
    plt.tight_layout()
    plt.savefig("Xd_clustered_EngPlus.png", dpi=300)
    plt.close()


    # --- X_a (phi_a1) ---
    plt.figure(figsize=(12, 5))

    for cluster_id, cf_list in cluster_dict.items():
        data = np.array([cf['ghi'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(days, mean,
                color=cluster_colors[cluster_id],
                marker=marks[cluster_id],
                linewidth=2,
                label=f'Cluster {cluster_id} mean',
                alpha=0.5)
        plt.fill_between(days, mean - std, mean + std,
                        color=cluster_colors[cluster_id],
                        alpha=0.2)

    for i in range(len(GPT_gen)):
        plt.plot(days,
                list(np.repeat(GPT_gen[i]["ghi"], 3))[:T],
                color="red",
                linestyle=style[i],
                linewidth=3,
                label=r'LLM Generated $X_{g, final}$')

    plt.xlabel("Days", fontsize=label_fs)
    plt.ylabel(r"$X_a$", fontsize=label_fs+4)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=legend_fs, loc="upper right")
    plt.tight_layout()
    plt.savefig("Xg_clustered_EngPlus.png", dpi=300)
    plt.close()


    # --- Y (Infected) ---
    plt.figure(figsize=(12, 6))

    plt.plot(times,
            baseline_infected,
            label=r'$Y_{base}$',
            color='black',
            linewidth=3)

    for cluster_id, cf_list in cluster_dict.items():
        data = np.array([cf['Electricity'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(times, mean,
                color=cluster_colors[cluster_id],
                marker=marks[cluster_id],
                linewidth=2,
                label=f'Cluster {cluster_id} mean',
                alpha=0.5)
        plt.fill_between(times, mean - std, mean + std,
                        color=cluster_colors[cluster_id],
                        alpha=0.2)

    for i in range(len(I_GPT)):
        plt.plot(times,
                I_GPT[i],
                color="red",
                linestyle=style[i],
                linewidth=3,
                label=r"LLM Generated $Y_{final}$")

    plt.xlabel("Days", fontsize=label_fs)
    plt.ylabel(r"$Y$", fontsize=label_fs+4)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=legend_fs, loc="upper left")
    plt.tight_layout()
    plt.savefig("Y_clustered_EngPlus.png", dpi=300)
    plt.close()


    


# --- INPUTS ---

[counterfactuals, study_trials, baseline_phi_s1, baseline_phi_a1, baseline_I] = pickle.load(open("CFE_sols_150.p", "rb"))

# np.set_printoptions(precision=2, suppress=True, floatmode='fixed')
np.set_printoptions(suppress=True, precision=2)


[GPT_gen, I_GPT, clustering_context] = pickle.load(open("CFE_sols_150_gpt_gen.p", "rb"))
# for k, v in clustering_context.items():
#     print(f"{k} ====> \n")
#     for k1, v1 in v.items():
#         print(f"{k1}: {[int(float(val) * 100) / 100 for val in v1]} \n\n")
#         input('')

# print(f"base: {[int(float(val) * 100) / 100 for val in baseline_I]} \n\n")
# exit()

# Example: 20 counterfactuals (each phi_s1 + phi_a1 is one sample)
# Each should be a 1D list of length 2 * N_WEEKS
X = np.array([np.concatenate([c["temp"], c["ghi"]]) for c in counterfactuals])

# The mse from simulation for each control set
mses = np.array([c["nmse"] for c in counterfactuals])

# Baseline input: same dimension as X[0]
baseline = np.concatenate([baseline_phi_s1, baseline_phi_a1])

# --- Embed into 2D using UMAP  ---
X_scaled = StandardScaler().fit_transform(X)
reducer = umap.UMAP(n_components=2, random_state=42)
X_embedded = reducer.fit_transform(X_scaled)

baseline_embedded = reducer.transform([StandardScaler().fit(X).transform([baseline])[0]])[0]

# --- Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_embedded)


similarity = 1 / (mses + 1e-6)  # avoid divide by zero
sizes = 1000 * (similarity / np.max(similarity))  # normalize for plotting


plt.figure(figsize=(10, 5))
scatter = plt.scatter(
    X_embedded[:, 0], X_embedded[:, 1],
    c=clusters, cmap='cool',
    s=sizes, alpha=0.6, edgecolor='k', linewidth=0.5
)

# Add index labels to each counterfactual point
i = 0
print(X_embedded)
for (x, y) in X_embedded:
    plt.text(x, y, str(i+1), fontsize=8, ha='center', va='center', color='black')
    i += 1

# plt.title("Counterfactual Control Clustering")
plt.xlabel("UMAP Dimension 1", fontsize=16)
plt.ylabel("UMAP Dimension 2", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters_CFE_150.png", dpi=300)
# plt.show()
# exit()
# Separate counterfactuals by cluster
cf_cluster0 = [cf for i, cf in enumerate(counterfactuals) if clusters[i] == 0]
cf_cluster1 = [cf for i, cf in enumerate(counterfactuals) if clusters[i] == 1]
cf_cluster2 = [cf for i, cf in enumerate(counterfactuals) if clusters[i] == 2]

# Colors for clusters
cluster_colors = ['cyan', 'mediumslateblue', 'violet']#['dimgrey', 'grey', 'silver']  # Cluster 0: blue, Cluster 1: orange
cluster_colors = ['dimgrey', 'grey', 'silver']  # Cluster 0: blue, Cluster 1: orange

# Save clusters and colors together for plotting
clustered_cfs = [(cf, clusters[i]) for i, cf in enumerate(counterfactuals)]

t_span = (0, 100)
# Now call the updated plotting function
plot_results_clustered_sep(
    baseline_phi_s1, baseline_phi_a1, baseline_I,
    clustered_cfs, t_span,
    cluster_colors
)
