import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import pickle
from Parameters_pf import *
from DDE_pf import *

# --- Plot Results with Clusters ---
def plot_results_clustered(baseline_phi_s1, baseline_phi_a1, baseline_infected, clustered_cfs, t_span, cluster_colors):
    n_weeks = len(baseline_phi_s1)
    weeks = np.arange(n_weeks)
    times = np.arange(t_span[0], t_span[1])


    # keep adding generated input from LLM
    GPT_gen = {0: {"phi_s": [
                    0.10,  # week 1 - moderate
                    0.02,  # week 2 - low
                    0.18,  # week 3 - high
                    0.03,  # week 4 - low
                    0.15,  # week 5 - high
                    0.04,  # week 6 - low
                    0.16,  # week 7 - high
                    0.05,  # week 8 - low
                    0.14,  # week 9 - high
                    0.06,  # week 10 - low
                    0.13,  # week 11 - high
                    0.07,  # week 12 - low
                    0.12,  # week 13 - high
                    0.08,  # week 14 - low
                    0.11   # week 15 - moderate
                ]  ,
                   "phi_a": [
                                0.03,  # week 1 - low
                                0.04,  # week 2 - low
                                0.20,  # week 3 - sharp increase
                                0.22,  # week 4 - high
                                0.23,  # week 5 - high
                                0.24,  # week 6 - high
                                0.25,  # week 7 - high
                                0.26,  # week 8 - high
                                0.27,  # week 9 - high
                                0.10,  # week 10 - dip (to allow resurgence)
                                0.28,  # week 11 - rise again
                                0.29,  # week 12 - high
                                0.30,  # week 13 - high
                                0.31,  # week 14 - high
                                0.32   # week 15 - high
                            ] },
               1: {"phi_s": [
                            0.03, 0.03, 0.04,    # Initial growth (low)
                            0.06, 0.08,          # Peak (moderate)
                            0.10, 0.11, 0.12, 0.12,  # Decline (moderate, stable)
                            0.11, 0.10, 0.10,    # Plateau (moderate, not high)
                            0.07, 0.06,          # Resurgence (dip)
                            0.10                 # Post-resurgence (moderate)
                            ],
                   "phi_a": [
                        0.04, 0.04, 0.05,    # Initial growth (low)
                        0.07, 0.10,          # Peak (moderate)
                        0.13, 0.14, 0.15, 0.15,  # Decline (moderate, stable)
                        0.14, 0.13, 0.13,    # Plateau (moderate, not high)
                        0.08, 0.07,          # Resurgence (dip)
                        0.12                 # Post-resurgence (moderate)
                    ]},
               2: {"phi_s": [0.03, 0.03, 0.03, 0.10, 0.15, 0.15, 0.13, 0.11, 0.09, 0.07, 0.06, 0.07, 0.08, 0.09, 0.09],
                   "phi_a": [0.04, 0.04, 0.04, 0.06, 0.08, 0.10, 0.12, 0.13, 0.15, 0.15, 0.15, 0.08, 0.07, 0.13, 0.14]
                  },
               3: {"phi_s":  [0.03, 0.03, 0.03, 0.12, 0.18, 0.10, 0.06, 0.04, 0.04, 0.04, 0.04, 0.06, 0.06, 0.04, 0.04],
                   "phi_a": [0.04, 0.04, 0.04, 0.06, 0.08, 0.12, 0.16, 0.18, 0.18, 0.18, 0.18, 0.10, 0.08, 0.16, 0.18]
                  },
               4: {"phi_s": [
                    0.03, 0.03,         # Weeks 1-2: Initial growth (low)
                    0.03, 0.04, 0.05,   # Weeks 3-5: Peak formation (low, slight rise)
                    0.18,               # Week 6: Post-peak sharp spike
                    0.08,               # Week 7: Drop after spike
                    0.05, 0.04, 0.04,   # Weeks 8-10: Decline/plateau (low/moderate)
                    0.04, 0.04,         # Weeks 11-12: Plateau (low)
                    0.06, 0.06,         # Weeks 13-14: Resurgence (moderate)
                    0.05                # Week 15: Late plateau (low/moderate)
                ],
                "phi_a": [
                    0.04, 0.04,         # Weeks 1-2: Initial growth (low)
                    0.05, 0.06, 0.08,   # Weeks 3-5: Peak formation (slow rise)
                    0.10,               # Week 6: Post-peak (moderate)
                    0.13,               # Week 7: Continue rising
                    0.16, 0.18, 0.18,   # Weeks 8-10: Decline/plateau (high)
                    0.18, 0.10,         # Weeks 11-12: Plateau, then dip for resurgence
                    0.08, 0.16,         # Weeks 13-14: Resurgence (dip, then return high)
                    0.18                # Week 15: Late plateau (high)
                ]},
              }
    

    I_GPT = []
    for i in range(len(GPT_gen)):
        phi_s_new, phi_a_new = GPT_gen[i]["phi_s"], GPT_gen[i]["phi_a"]
        sol_GPT = runModel(init_cond, T_delay, params, t_span[1], def_inf_rate, phi_s_new, def_phi_s2, phi_a_new, def_phi_a2, def_g_rate) 
        inf = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])
        I_GPT.append(inf)
        
    style = ["solid", "dotted", "dashdot", "dashed", (0, (5, 10)), (0, (3, 1, 1, 1, 1, 1))]

    plt.figure(figsize=(16, 12))
    
    # Group CFs by cluster
    from collections import defaultdict
    cluster_dict = defaultdict(list)
    for cf, cluster in clustered_cfs:
        cluster_dict[cluster].append(cf)

    # Convert baseline weekly phi rates to daily stepwise
    baseline_phi_s1_daily = np.repeat(baseline_phi_s1, 7)
    baseline_phi_a1_daily = np.repeat(baseline_phi_a1, 7)
    days = np.arange(len(baseline_phi_s1_daily))  # Daily time axis


    # --- X_s (phi_s1) ---
    marks = ["o", "v", "s"]
    plt.subplot(3, 1, 1)
    # plt.plot(days, baseline_phi_s1_daily, label='Baseline $X_s$', color='black', linewidth=2)
    for cluster_id, cf_list in cluster_dict.items():
        data = np.array([np.repeat(cf['phi_s1'], 7) for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(days, mean, color=cluster_colors[cluster_id],  marker=marks[cluster_id], linewidth=2, label=f'Cluster {cluster_id} mean', alpha=0.5)
        plt.fill_between(days, mean - std, mean + std, color=cluster_colors[cluster_id], alpha=0.2)
    
    # GPT CFE for X_s
    for i in range(len(GPT_gen)):
        plt.plot(days, np.repeat(GPT_gen[i]["phi_s"], 7), color="red", linestyle=style[i], linewidth=2, label=r'LLM Generated $X_{s, final}$')
    
    # plt.title(r"$X_s$")
    plt.xlabel("Days", fontsize=14)
    plt.ylabel(r"$X_s$", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper left")
    
    # --- X_a (phi_a1) ---
    plt.subplot(3, 1, 2)
    # plt.plot(days, baseline_phi_a1_daily, label='Baseline $X_a$', color='black', linewidth=2)
    for cluster_id, cf_list in cluster_dict.items():
        data = np.array([np.repeat(cf['phi_a1'], 7) for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(days, mean, color=cluster_colors[cluster_id], marker=marks[cluster_id], linewidth=2, label=f'Cluster {cluster_id} mean', alpha=0.5)
        plt.fill_between(days, mean - std, mean + std, color=cluster_colors[cluster_id], alpha=0.2)
    
    # GPT CFE for X_a
    for i in range(len(GPT_gen)):
        plt.plot(days, np.repeat(GPT_gen[i]["phi_a"], 7), color="red",  linewidth=3, linestyle=style[i], label=r'LLM Generated $X_{a, final}$')
    
    # plt.title(r"$X_a$", fontsize=14)
    plt.xlabel("Days", fontsize=14)
    plt.ylabel(r"$X_a$", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper left")
    
    # --- Y (Infected) ---
    plt.subplot(3, 1, 3)
    plt.plot(times, baseline_infected, label=r'$Y_{base}$', color='black', linewidth=3)
    for cluster_id, cf_list in cluster_dict.items():
        data = np.array([cf['infected'] for cf in cf_list])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(times, mean, color=cluster_colors[cluster_id],  marker=marks[cluster_id], linewidth=2, label=f'Cluster {cluster_id} mean', alpha=0.5)
        plt.fill_between(times, mean - std, mean + std, color=cluster_colors[cluster_id], alpha=0.2)
    
    # GPT CFE for Y
    for i in range(len(I_GPT)):
        plt.plot(times, I_GPT[i], color="red", linestyle=style[i],  linewidth=2, label=r"LLM Generated $Y_{final}$")
    
    # plt.title(r"$Y$ (Infected over time)")
    plt.xlabel("Days", fontsize=14)
    plt.ylabel(r"$Y$", fontsize=14)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper left")
    plt.tight_layout()
    plt.savefig("optuna_CFE_pysirtem_clustered_100_gpt41_new.png", dpi=300)


    pickle.dump([GPT_gen, I_GPT], open("CFE_sols3_gpt_gen.p", "wb"))


# --- INPUTS ---

[counterfactuals, study_trials, baseline_phi_s1, baseline_phi_a1, baseline_I] = pickle.load(open("CFE_sols3.p", "rb"))

# Example: 20 counterfactuals (each phi_s1 + phi_a1 is one sample)
# Each should be a 1D list of length 2 * N_WEEKS
X = np.array([np.concatenate([c["phi_s1"], c["phi_a1"]]) for c in counterfactuals])

# The mse from simulation for each control set
mses = np.array([c["mse"] for c in counterfactuals])

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
for i, (x, y) in enumerate(X_embedded):
    plt.text(x, y, str(i+1), fontsize=8, ha='center', va='center', color='black')

# plt.title("Counterfactual Control Clustering")
plt.xlabel("UMAP Dimension 1", fontsize=16)
plt.ylabel("UMAP Dimension 2", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters_CFE_100_gpt41_baseline_new2.png", dpi=300)
# exit()
# Separate counterfactuals by cluster
cf_cluster0 = [cf for i, cf in enumerate(counterfactuals) if clusters[i] == 0]
cf_cluster1 = [cf for i, cf in enumerate(counterfactuals) if clusters[i] == 1]

# Colors for clusters
cluster_colors = ['cyan', 'mediumslateblue', 'violet']#['dimgrey', 'grey', 'silver']  # Cluster 0: blue, Cluster 1: orange

# Save clusters and colors together for plotting
clustered_cfs = [(cf, clusters[i]) for i, cf in enumerate(counterfactuals)]

t_span = (0, 100)
# Now call the updated plotting function
plot_results_clustered(
    baseline_phi_s1, baseline_phi_a1, baseline_I,
    clustered_cfs, t_span,
    cluster_colors
)
