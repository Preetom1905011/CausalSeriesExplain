
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import optuna
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import pickle

from test_CFE_optuna import run_model

from scipy.ndimage import gaussian_filter

# # for 3 row plot - separate x_a and x_s
# def plot_phi_and_inf_on_axes(axs, phi_s, phi_a, inf, base_phi_s, base_phi_a, base_inf, tag):
#     weeks = np.arange(1, (len(base_inf) // 7 + 1) + 1, 1)
#     style = ["solid", "dotted", "dashdot", "dashed", (0, (5, 10))]
#     days = np.arange(1, (len(base_inf) + 1), 1)
#     # phi_s
#     for i in range(len(phi_s)):
#         axs[0].plot(days, np.repeat(phi_s[i], 7)[:len(days)], color="red", linestyle=style[i], label=r'LLM Generated $X_{s}$', linewidth=3)
#     # axs[0].set_title("phi_s1 (weekly rates)")
#     axs[0].set_title(tag, fontsize=16)
#     axs[0].set_xlabel("Days", fontsize=16)
#     axs[0].set_ylabel(r"$X_{s}$", fontsize=16)
#     # plt.xticks(fontsize=16)
#     # plt.yticks(fontsize=16)
#     axs[0].set_ylim([0, 0.35])
#     axs[0].legend(fontsize=16, loc="upper left")

#     # phi_a
#     for i in range(len(phi_a)):
#         axs[1].plot(days, np.repeat(phi_a[i], 7)[:len(days)], color="red", linestyle=style[i], label=r'LLM Generated $X_{a}$', linewidth=3)
#     # axs[1].set_title("phi_a1 (weekly rates)")
#     axs[1].set_xlabel("Days", fontsize=16)
#     axs[1].set_ylabel(r"$X_{a}$", fontsize=16)
#     # plt.xticks(fontsize=16)
#     # plt.yticks(fontsize=16)
#     axs[1].set_ylim([0, 0.35])
#     axs[1].legend(fontsize=16, loc="upper left")

#     # infected
#     for i in range(len(inf)):
#         y = inf[i]
#         axs[2].plot(np.arange(1, len(base_inf)+1, 1), y, color="red", linestyle=style[i], label=r'$Y_{final}$', linewidth=3)
        
#         for j in range(20, len(y) - 1):
#             if y[j] > y[j-1] and y[j] > y[j+1]:
#                 axs[2].axvline(x=j+1, color='red', linestyle=':', linewidth=1.2)
#                 axs[2].axhline(y=y[j], color='red', linestyle=':', linewidth=1.2)
#             elif y[j] < y[j-1] and y[j] < y[j+1]:
#                 axs[2].axvline(x=j+1, color='red', linestyle=':', linewidth=1.2)
#                 axs[2].axhline(y=y[j], color='red', linestyle=':', linewidth=1.2)

#     axs[2].plot(np.arange(1, len(base_inf)+1, 1), base_inf, color="black", linestyle=style[0], label=r'$Y_{base}$', linewidth=3)
    
#     for j in range(20, len(base_inf) - 1):
#         if base_inf[j] > base_inf[j-1] and base_inf[j] > base_inf[j+1]:
#             axs[2].axvline(x=j+1, color='black', linestyle='--', linewidth=1.2)
#             axs[2].axhline(y=base_inf[j], color='black', linestyle='--', linewidth=1.2)
#         elif base_inf[j] < base_inf[j-1] and base_inf[j] < base_inf[j+1]:
#             axs[2].axvline(x=j+1, color='black', linestyle='--', linewidth=1.2)
#             axs[2].axhline(y=base_inf[j], color='black', linestyle='--', linewidth=1.2)

#     # axs[2].set_title("Infected (Daily)")
#     axs[2].set_xlabel("Days", fontsize=16)
#     axs[2].set_ylabel(r"Y", fontsize=16)
#     # plt.xticks(fontsize=16)
#     # plt.yticks(fontsize=16)
#     # axs[2].set_ylim([0, 2200000]) #new
#     axs[2].set_ylim([0, 120000]) #new 2 
#     # axs[2].set_ylim([0, 16000]) # baseline original
#     axs[2].legend(fontsize=16, loc="upper left")

def plot_phi_and_inf_on_axes(axs, phi_s, phi_a, inf, base_phi_s, base_phi_a, base_inf, tag):
    weeks = np.arange(1, (len(base_inf) // 3 + 1) + 1, 1)
    style = ["solid", "dotted", "dashdot", "dashed", (0, (5, 10))]
    days = np.arange(1, (len(base_inf) + 1), 1)


    # --- Combined phi_s and phi_a on axs[0] ---
    # for i in range(len(phi_s)):
    #     axs[0].plot(days, np.repeat(phi_s[i], 3)[:len(days)], color="tomato", linestyle="solid",
    #                 label=r'LLM Generated $X_{d}$', linewidth=2, alpha=0.75)
    # for i in range(len(phi_a)):
    #     axs[0].plot(days, np.repeat(phi_a[i], 3)[:len(days)], color="royalblue", linestyle="dashed",
    #                 label=r'LLM Generated $X_{g}$', linewidth=2, alpha=0.75)

    axs.set_title(tag, fontsize=18)
    # axs[0].set_xlabel("Days", fontsize=22)
    # axs[0].set_ylabel(r"$X_{s}, X_{a}$", fontsize=22)
    # axs[0].tick_params(axis='both', which='major', labelsize=16)
    # # axs[0].set_ylim([0, 0.35])
    # axs[0].legend(fontsize=22, loc="upper left")

    # --- Infected curve on axs[1] (formerly axs[2]) ---
    # for i in range(len(inf)):
    #     y = inf[i]
    print(inf)
    axs.plot(inf, color="red", linestyle=style[i],
                    label=r'$Y_{final}$', linewidth=2)
        
        # for j in range(20, len(y) - 1):
        #     if y[j] > y[j-1] and y[j] > y[j+1]:
        #         axs[1].axvline(x=j+1, color='red', linestyle=':', linewidth=1.2)
        #         axs[1].axhline(y=y[j], color='red', linestyle=':', linewidth=1.2)
        #     elif y[j] < y[j-1] and y[j] < y[j+1]:
        #         axs[1].axvline(x=j+1, color='red', linestyle=':', linewidth=1.2)
        #         axs[1].axhline(y=y[j], color='red', linestyle=':', linewidth=1.2)

    axs.plot(base_inf[0], color="dimgrey", linestyle=style[0],
                label=r'$Y_{base}$', linewidth=2, alpha=0.75)

    # for j in range(20, len(base_inf) - 1):
    #     if base_inf[j] > base_inf[j-1] and base_inf[j] > base_inf[j+1]:
    #         axs[1].axvline(x=j+1, color='black', linestyle='--', linewidth=1.2)
    #         axs[1].axhline(y=base_inf[j], color='black', linestyle='--', linewidth=1.2)
    #     elif base_inf[j] < base_inf[j-1] and base_inf[j] < base_inf[j+1]:
    #         axs[1].axvline(x=j+1, color='black', linestyle='--', linewidth=1.2)
    #         axs[1].axhline(y=base_inf[j], color='black', linestyle='--', linewidth=1.2)

    axs.set_xlabel("Hours", fontsize=20)
    axs.set_ylabel(r"$Y$", fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=16)
    # axs[1].set_ylim([0, 2200000]) #new
    # axs[1].set_ylim([0, 120000]) #new 2 
    # axs[1].set_ylim([0, 16000]) # baseline original
    axs.set_ylim([100000000, 510000000]) # baseline original energyplus
    axs.legend(fontsize=18, loc="upper left")


# phi_s1_new = [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
# phi_a1_new = [0.01, 0.01, 0.03, 0.06, 0.09, 0.20, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20, 0.20, 0.20, 0.20]
# phi_s1_new2 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.05, 0.05, 0.03, 0.05, 0.01, 0.01]
# phi_a1_new2 = [0.01, 0.01, 0.03, 0.06, 0.09, 0.20, 0.01, 0.01, 0.01, 0.20, 0.20, 0.05, 0.20, 0.01, 0.01]

# # wrong rule - new (final)
# phi_s1_new = [0.30, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.30, 0.30, 0.01, 0.01, 0.15, 0.15]
# phi_a1_new = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.28, 0.28, 0.28, 0.28, 0.10, 0.10, 0.10]
# # no rule - new
# phi_s1_new = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.25, 0.20, 0.15, 0.10, 0.06, 0.03, 0.01]
# phi_a1_new = [0.01, 0.02, 0.04, 0.08, 0.13, 0.18, 0.23, 0.28, 0.30, 0.28, 0.23, 0.18, 0.13, 0.08, 0.04]

# wrong rule - new2 (final)
phi_s1_new2 = [0.25, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01, 0.30, 0.01, 0.15, 0.15, 0.15]
phi_a1_new2 = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.10, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01]
# # no rule - new2
# phi_s1_new2 = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.22, 0.18, 0.14, 0.10, 0.14, 0.18, 0.22, 0.18]
# phi_a1_new2 = [0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.30, 0.26, 0.22, 0.18, 0.14, 0.10, 0.06]

# baseline_data is what it was trained on, _new and _new2 are the two variations with different inf rate, phi_a and phi_s
# [base_phi_s1, base_phi_a1, base_inf, base_inf_rate, _] = pickle.load(open("baseline_data_new2.p", "rb"))
# print(list(base_inf), max(base_phi_a1), max(base_phi_s1))
# exit()
# res = runModel(init_cond, T_delay, params, 100, base_inf_rate, phi_s1_new, def_phi_s2, phi_a1_new, def_phi_a2, def_g_rate) 
# res = runModel(init_cond, T_delay, params, 100, base_inf_rate, phi_s1_new2, def_phi_s2, phi_a1_new2, def_phi_a2, def_g_rate) 
# ------ the bottom res to generate a new baseline - update params in Parameters_pf.py
# res = runModel(init_cond, T_delay, params, 100, def_inf_rate, def_phi_s1, def_phi_s2, def_phi_a1, def_phi_a2, def_g_rate) 
# compare_and_plot_daily_counts(res, "AZ")
# sim_daily_pos, sim_daily_neg = calculate_daily_counts(res, True)
'''
print(base_inf, "\n\n", max(base_phi_s1), min(base_phi_s1), max(base_phi_a1), min(base_phi_s1))
# exit()
print(np.array(res).shape)

for t in range(np.array(res).shape[0]):
    for c in range(len(res[t])):
        if res[t][c] < 0:
            print(">>>", t, list(M_states.keys())[c], res[t][c])
    print(sum(init_cond), sum(np.array(res)[t, :]))


rows = 8
cols = 10
fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
axes = axes.flatten()

for k in list(M_states.keys()):
    i = M_states[k]
    ax = axes[i]
    y = list(np.array(res)[:, M_states[k]]) # Replace with your data
    ax.plot(y)
    ax.set_title(f"Plot {k}", fontsize=8)
# Hide any unused subplots
for j in range(76, len(axes)):
    axes[j].axis('off')


plt.tight_layout()
# plt.savefig("test_figs/test_all_delay_baseline_new.png", dpi=300)
plt.clf()
'''

# plt.figure(figsize=(16, 12))
# sol_GPT = res
# inf = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])

# plt.plot(np.arange(len(inf)), inf, label="GPT: infected")
# plt.legend()
# # plt.savefig("test_figs/test_Infected_delay_baseline_new3_for_learning", dpi=300)
# plt.clf()

# print(inf)
# fig, axs = plt.subplots(3, 1, figsize=(18, 12))

# plot_phi_and_inf_on_axes(axs, [phi_s1_new], [phi_a1_new], [inf], base_phi_s1, base_phi_a1, base_inf, "baseline_new")
# plot_phi_and_inf_on_axes(axs, [phi_s1_new2], [phi_a1_new2], [inf], base_phi_s1, base_phi_a1, base_inf, "baseline_new2")

# # wrong rule set - new
# plot_phi_and_inf_on_axes(axs, [phi_s1_new], [phi_a1_new], [inf], base_phi_s1, base_phi_a1, base_inf, "Inaccurate Ruleset")
# # empty rule set - new
# plot_phi_and_inf_on_axes(axs, [phi_s1_new], [phi_a1_new], [inf], base_phi_s1, base_phi_a1, base_inf, "Empty Ruleset")


# # wrong rule set - new2
# plot_phi_and_inf_on_axes(axs, [phi_s1_new2], [phi_a1_new2], [inf], base_phi_s1, base_phi_a1, base_inf, "Inaccurate Ruleset 2")
# # empty rule set - new
# plot_phi_and_inf_on_axes(axs, [phi_s1_new2], [phi_a1_new2], [inf], base_phi_s1, base_phi_a1, base_inf, "Empty Ruleset 2")


# plt.tight_layout()
# plt.savefig("test_figs/compare_different_rule2.png", dpi=300)
# plt.clf()

# pickle.dump([def_phi_s1, def_phi_a1, inf, def_inf_rate, res], open("baseline_data_new3_for_learning.p", "wb"))
'''
# ------------------ Experiment Empty rule set ----------------------------------

fig, axs = plt.subplots(3, 2, figsize=(18, 12))
axs = axs.T  # Transpose to get columns as individual plot blocks

# empty rule - new (final)
phi_s1_new = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.25, 0.20, 0.15, 0.10, 0.06, 0.03, 0.01]
phi_a1_new = [0.01, 0.02, 0.04, 0.08, 0.13, 0.18, 0.23, 0.28, 0.30, 0.28, 0.23, 0.18, 0.13, 0.08, 0.04]

# empty rule - new2 (final)
phi_s1_new2 = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.22, 0.18, 0.14, 0.10, 0.14, 0.18, 0.22, 0.18]
phi_a1_new2 = [0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.30, 0.26, 0.22, 0.18, 0.14, 0.10, 0.06]

# baseline empty
phi_s1_new = [0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.28, 0.30, 0.27, 0.20, 0.13, 0.08, 0.10, 0.18, 0.25]
phi_a1_new = [0.02, 0.04, 0.07, 0.12, 0.18, 0.25, 0.29, 0.28, 0.24, 0.17, 0.11, 0.07, 0.09, 0.16, 0.23]


# baseline_data is what it was trained on, _new and _new2 are the two variations with different inf rate, phi_a and phi_s
[base_phi_s1, base_phi_a1, base_inf, base_inf_rate, _] = pickle.load(open("baseline_data_new.p", "rb"))
[base_phi_s1_2, base_phi_a1_2, base_inf_2, base_inf_rate_2, _] = pickle.load(open("baseline_data_new2.p", "rb"))


baseline_sol = pickle.load(open("baseline_data.p", "rb"))
# baseline_I = list(np.array(baseline_sol)[:, M_states["IA"]] + np.array(baseline_sol)[:, M_states["IS"]])
baseline_I = params['r'] * (np.array(baseline_sol)[:, M_states['PS']] + np.array(baseline_sol)[:, M_states['PA']] + np.array(baseline_sol)[:, M_states['IA']] + np.array(baseline_sol)[:, M_states['ATN']]) + np.array(baseline_sol)[:, M_states['IS']] + np.array(baseline_sol)[:, M_states['STN']]

print(list(baseline_I))


res = runModel(init_cond, T_delay, params, 100, def_inf_rate, phi_s1_new, def_phi_s2, phi_a1_new, def_phi_a2, def_g_rate) 
res2 = runModel(init_cond, T_delay, params, 100, base_inf_rate_2, phi_s1_new2, def_phi_s2, phi_a1_new2, def_phi_a2, def_g_rate) 

sol_GPT = res
inf_1 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])

sol_GPT = res2
inf_2 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])



# Call the plotting function on each column
plot_phi_and_inf_on_axes(axs[0], [phi_s1_new], [phi_a1_new], [inf_1], base_phi_s1, base_phi_a1, list(baseline_I), "Empty ruleset - Baseline")
plot_phi_and_inf_on_axes(axs[1], [phi_s1_new2], [phi_a1_new2], [inf_2], base_phi_s1_2, base_phi_a1_2, base_inf_2, "Empty ruleset - Non-Prominent Phase Changes")

plt.tight_layout()
plt.savefig("test_figs/compare_no_rules_subplots2.png", dpi=300)
plt.clf()



# ------------------ Experiment wrong rule set ----------------------------------

fig, axs = plt.subplots(3, 2, figsize=(18, 12))
axs = axs.T  # Transpose to get columns as individual plot blocks

# wrong rule - new (final)
phi_s1_new = [0.30, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.30, 0.30, 0.01, 0.01, 0.15, 0.15]
phi_a1_new = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.28, 0.28, 0.28, 0.28, 0.10, 0.10, 0.10]

# wrong rule - new2 (final)
phi_s1_new2 = [0.25, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01, 0.30, 0.01, 0.15, 0.15, 0.15]
phi_a1_new2 = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.10, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01]


# baseline_data is what it was trained on, _new and _new2 are the two variations with different inf rate, phi_a and phi_s
[base_phi_s1, base_phi_a1, base_inf, base_inf_rate, _] = pickle.load(open("baseline_data_new.p", "rb"))
[base_phi_s1_2, base_phi_a1_2, base_inf_2, base_inf_rate_2, _] = pickle.load(open("baseline_data_new2.p", "rb"))


res = runModel(init_cond, T_delay, params, 100, base_inf_rate, phi_s1_new, def_phi_s2, phi_a1_new, def_phi_a2, def_g_rate) 
res2 = runModel(init_cond, T_delay, params, 100, base_inf_rate_2, phi_s1_new2, def_phi_s2, phi_a1_new2, def_phi_a2, def_g_rate) 

sol_GPT = res
inf_1 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])

sol_GPT = res2
inf_2 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])



# Call the plotting function on each column
plot_phi_and_inf_on_axes(axs[0], [phi_s1_new], [phi_a1_new], [inf_1], base_phi_s1, base_phi_a1, base_inf, "Inaccurate ruleset - Prominent Phase Changes")
plot_phi_and_inf_on_axes(axs[1], [phi_s1_new2], [phi_a1_new2], [inf_2], base_phi_s1_2, base_phi_a1_2, base_inf_2, "Inaccurate ruleset - Non-Prominent Phase Changes")

plt.tight_layout()
plt.savefig("test_figs/compare_wrong_rules_subplots.png", dpi=300)
plt.clf()
'''
'''
# ------------- experiment 3 generalizability ------------------------
# for baseline new
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
axs = axs.T  # Transpose to get columns as individual plot blocks

# # accurate new - final
# phi_s1_new = [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05]
# phi_a1_new = [0.01, 0.01, 0.03, 0.06, 0.09, 0.20, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20, 0.20, 0.20, 0.20]

# accurate new ruleset - final
phi_s1_new = [0.01, 0.01, 0.01,   # weeks 0-2: Low
                0.08, 0.08,         # weeks 2-4: Moderate
                0.20, 0.20,         # weeks 4-6: SharpSpike
                0.01, 0.20,         # weeks 6-8: DipThenRise
                0.01, 0.20,         # weeks 8-10: SlowRise
                0.08, 0.08, 0.08, 0.08, 0.08]
phi_a1_new = [0.01, 0.02, 0.04,   # weeks 0-2: SlowRise
                0.06, 0.08,         # weeks 2-4: SlowRise
                0.01, 0.01,         # weeks 4-6: Low
                0.01, 0.01,         # weeks 6-8: Low
                0.18, 0.18,         # weeks 8-10: HighStable
                0.01, 0.06, 0.10, 0.15, 0.20]

# wrong rule - new (final)
phi_s1_new2 = [0.30, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.30, 0.30, 0.01, 0.01, 0.15, 0.15]
phi_a1_new2 = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.28, 0.28, 0.28, 0.28, 0.10, 0.10, 0.10]

# baseline_data is what it was trained on, _new and _new2 are the two variations with different inf rate, phi_a and phi_s
[base_phi_s1, base_phi_a1, base_inf, base_inf_rate, _] = pickle.load(open("baseline_data_new.p", "rb"))
print(base_inf)
# exit()

res = runModel(init_cond, T_delay, params, 100, base_inf_rate, phi_s1_new, def_phi_s2, phi_a1_new, def_phi_a2, def_g_rate) 
res2 = runModel(init_cond, T_delay, params, 100, base_inf_rate, phi_s1_new2, def_phi_s2, phi_a1_new2, def_phi_a2, def_g_rate) 

sol_GPT = res
inf_1 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])

sol_GPT = res2
inf_2 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])



# Call the plotting function on each column
plot_phi_and_inf_on_axes(axs[0], [phi_s1_new], [phi_a1_new], [inf_1], base_phi_s1, base_phi_a1, base_inf, "Accurate Generalized Ruleset")
plot_phi_and_inf_on_axes(axs[1], [phi_s1_new2], [phi_a1_new2], [inf_2], base_phi_s1, base_phi_a1, base_inf, "Inaccurate/Swapped Ruleset")

plt.tight_layout()
plt.savefig("test_figs/compare_generalize_baseline_newrules_new.png", dpi=300)
plt.clf()
# exit()

# for baseline new 2
fig, axs = plt.subplots(2, 2, figsize=(16, 9))
axs = axs.T  # Transpose to get columns as individual plot blocks

# wrong rule - new2 (final)
phi_s1_new = [0.25, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01, 0.30, 0.01, 0.15, 0.15, 0.15]
phi_a1_new = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.10, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01]

# # accurate new2 - final
# phi_a1_new2 = [
#     0.01,  # Week 1: Initial Growth
#     0.01,  # Week 2: Initial Growth
#     0.03,  # Week 3: Peak Formation (slow rise)
#     0.05,  # Week 4: Peak Formation
#     0.08,  # Week 5: Peak Formation
#     0.15,  # Week 6: Post-Peak (moderate)
#     0.15,  # Week 7: Post-Peak
#     0.25,  # Week 8: Decline/Plateau (high, stable)
#     0.25,  # Week 9: Decline/Plateau
#     0.25,  # Week 10: Decline/Plateau
#     0.25,  # Week 11: Decline/Plateau
#     0.10,  # Week 12: Resurgence (dip)
#     0.15,  # Week 13: Resurgence (return high)
#     0.25,  # Week 14: Resurgence (high)
#     0.25   # Week 15: Resurgence (high)
# ]
# phi_s1_new2 = [
#     0.01,  # Week 1: Initial Growth
#     0.01,  # Week 2: Initial Growth
#     0.01,  # Week 3: Peak Formation (low)
#     0.01,  # Week 4: Peak Formation
#     0.01,  # Week 5: Peak Formation
#     0.30,  # Week 6: Post-Peak (sharp spike)
#     0.05,  # Week 7: Post-Peak (drop after spike)
#     0.05,  # Week 8: Decline/Plateau (low/moderate)
#     0.05,  # Week 9: Decline/Plateau
#     0.05,  # Week 10: Decline/Plateau
#     0.05,  # Week 11: Decline/Plateau
#     0.05,  # Week 12: Resurgence (low/moderate)
#     0.05,  # Week 13: Resurgence
#     0.05,  # Week 14: Resurgence
#     0.05   # Week 15: Resurgence
# ]

# accurate new ruleset - new2 - final
phi_a1_new2 = [
     0.01, 0.03, 0.05, 0.07, 0.09, 0.10,  # weeks 0–5: SlowRise
    0.12, 0.14, 0.16, 0.18, 0.20,        # weeks 6–10: SlowRise
    0.28, 0.28,                          # weeks 11–12: HighStable
    0.25, 0.30                           # weeks 13–14: SlowRise
]
phi_s1_new2 = [
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # weeks 0–5: Low
    0.15, 0.15, 0.15, 0.15, 0.15,        # weeks 6–10: Moderate
    0.10, 0.20,                          # weeks 11–12: SlowRise
    0.15, 0.15
]


# baseline_data is what it was trained on, _new and _new2 are the two variations with different inf rate, phi_a and phi_s
[base_phi_s1_2, base_phi_a1_2, base_inf_2, base_inf_rate_2, _] = pickle.load(open("baseline_data_new2.p", "rb"))


res = runModel(init_cond, T_delay, params, 100, base_inf_rate_2, phi_s1_new, def_phi_s2, phi_a1_new, def_phi_a2, def_g_rate) 
res2 = runModel(init_cond, T_delay, params, 100, base_inf_rate_2, phi_s1_new2, def_phi_s2, phi_a1_new2, def_phi_a2, def_g_rate) 

sol_GPT = res
inf_1 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])

sol_GPT = res2
inf_2 = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])



# Call the plotting function on each column
plot_phi_and_inf_on_axes(axs[1], [phi_s1_new], [phi_a1_new], [inf_1], base_phi_s1_2, base_phi_a1_2, base_inf_2, "Inaccurate/Swapped Ruleset")
plot_phi_and_inf_on_axes(axs[0], [phi_s1_new2], [phi_a1_new2], [inf_2], base_phi_s1_2, base_phi_a1_2, base_inf_2, "Accurate Generalized Ruleset")

plt.tight_layout()
plt.savefig("test_figs/compare_generalize_baseline_newrules_new2.png", dpi=300)
plt.clf()
exit()
'''
# ------------------------- experiment 2 (context vs no-context) -------------------------------------
# [GPT_gen_cont, I_GPT_cont] = pickle.load( open("CFE_sols3_gpt_gen.p", "rb"))
# [GPT_gen_cont, I_GPT_cont] = pickle.load( open("New_relearning_gpt_gen.p", "rb"))

[GPT_gen_cont, I_GPT_cont, _] = pickle.load(open("CFE_sols_150_gpt_gen.p", "rb"))
# baseline_sol = pickle.load(open("baseline_data.p", "rb"))
[counterfactuals, study_trials, baseline_phi_s1, baseline_phi_a1, baseline_I] = pickle.load(open("CFE_sols_150.p", "rb"))


# gpt context + rules
GPT_gen_cont = [GPT_gen_cont[6]]
I_GPT_cont = [list(I_GPT_cont[6])]

# gpt rules + no context
# GPT_gen_no_cont = {0: {"phi_s": [0.01, 0.01, 0.01, 0.01, 0.20, 0.05, 0.03, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05],
#                    "phi_a": [0.01, 0.01, 0.03, 0.05, 0.08, 0.15, 0.15, 0.15, 0.05, 0.10, 0.15, 0.18, 0.20, 0.20, 0.20]}
GPT_gen_no_cont = {0: {"temp": [4.2, 3.8, 4.5, 6.0, 8.5, 11.2, 10.5, 7.9, 11.6, 9.2, 10.1, 8.8, 13.5, 14.8, 13.9, 12.6, 9.4, 12.9, 11.0, 13.7, 16.2, 14.2, 15.6, 14.7, 13.1, 10.2, 13.8, 12.4, 15.9, 18.6, 9.8, 11.4, 10.6, 5.3],
                    "ghi":  [90, 72, 84, 140, 410, 165, 170, 145, 185, 160, 430, 175, 230, 255, 240, 95, 82, 88, 180, 445, 205, 250, 270, 245, 100, 76, 92, 210, 460, 230, 220, 180, 235, 165]
                }
                }
I_GPT_no_cont = []
for i in range(len(GPT_gen_no_cont)):
    T = 100
    week_temp_list, week_ghi_list = GPT_gen_no_cont[i]["temp"], GPT_gen_no_cont[i]["ghi"]
    
    unnorm_temp_list = list(np.repeat(np.array(week_temp_list), 3))[:T]
    unnorm_ghi_list = list(np.repeat(np.array(week_ghi_list), 3))[:T]
    status, sol_GPT = run_model(unnorm_temp_list, unnorm_ghi_list)
    print(">>", status, len(sol_GPT))
    if status == 0:
        I_GPT_no_cont.append(sol_GPT)
    else:
        I_GPT_no_cont.append([])
T = 100
        
# gpt no rules + no context
GPT_NO_RULES = {0: {"temp": [7.372961, 7.372961, 7.372961, 7.372961, 7.359890, 7.348605, 7.338895, 12.076484, 12.120733, 14.922297, 23.790048, 24.321201, 24.812022, 24.650043, 25.091222, 24.906465, 25.346987, 24.977278, 25.303164, 25.009598, 25.025506, 20.737457, 20.886293, 20.538672, 8.640782, 7.431561, 7.854020, 7.391039, 7.357163, 7.349572, 7.340122, 12.055943, 12.122070, 14.923153],
                    "ghi": [110.594420, 110.398356, 110.229074, 110.083419, 181.147259, 181.811002, 223.834454, 356.850727, 364.818017, 372.180323, 369.750647, 376.368332, 373.596982, 380.204799, 374.659171, 379.547453, 375.143971, 375.382584, 311.061862, 313.294395, 308.080073, 129.611724, 111.473413, 117.810304, 110.865590, 110.357442, 110.243584, 110.101834, 180.839151, 181.831051, 223.847291, 357.097203, 366.423199, 374.756399]}}
week_temp_list, week_ghi_list = GPT_NO_RULES[0]["temp"], [ int(v) for v in GPT_NO_RULES[0]["ghi"]]

unnorm_temp_list = list(np.repeat(np.array(week_temp_list), 3))[:T]
unnorm_ghi_list = list(np.repeat(np.array(week_ghi_list), 3))[:T]
I_NO_RULES = run_model(unnorm_temp_list, unnorm_ghi_list)

fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs = axs.T  # Transpose to get columns as individual plot blocks

plot_phi_and_inf_on_axes(axs[0], [list(GPT_NO_RULES[0]["temp"])], [list(GPT_NO_RULES[0]["ghi"])], I_NO_RULES[1], baseline_phi_s1, baseline_phi_a1, [list(baseline_I)], "No Rules + No Clustering Context")
plot_phi_and_inf_on_axes(axs[1], [list(GPT_gen_no_cont[0]["temp"])], [list(GPT_gen_no_cont[0]["ghi"])], I_GPT_no_cont[0], baseline_phi_s1, baseline_phi_a1, [list(baseline_I)], "Rules + No Clustering Context")
plot_phi_and_inf_on_axes(axs[2], [GPT_gen_cont[0]["temp"]], [GPT_gen_cont[0]["ghi"]], I_GPT_cont[0], baseline_phi_s1, baseline_phi_a1, [list(baseline_I)], "Rules + Clustering Context")


plt.tight_layout()
plt.savefig("compare_context_new.png", dpi=300)
plt.clf()

exit()
'''
# Seed
np.random.seed(42)


# --- Plot Results ---
def plot_results(baseline_phi_s1, baseline_phi_a1, baseline_infected, counterfactuals, t_span):
    n_weeks = len(baseline_phi_s1)
    weeks = np.arange(n_weeks)
    times = np.arange(t_span[0], t_span[1])

    plt.figure(figsize=(16, 12))
    
    # phi_s1
    plt.subplot(3, 1, 1)
    plt.plot(weeks, baseline_phi_s1, label='Baseline phi_s1', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(weeks, cf['phi_s1'], label=f'CF {i+1} phi_s1', alpha=0.7)
    plt.title("phi_s1 (weekly rates)")
    plt.xlabel("Week")
    plt.ylabel("phi_s1")
    plt.legend()
    
    # phi_a1
    plt.subplot(3, 1, 2)
    plt.plot(weeks, baseline_phi_a1, label='Baseline phi_a1', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(weeks, cf['phi_a1'], label=f'CF {i+1} phi_a1', alpha=0.7)
    plt.title("phi_a1 (weekly rates)")
    plt.xlabel("Week")
    plt.ylabel("phi_a1")
    plt.legend()
    
    # Infected curves
    plt.subplot(3, 1, 3)
    plt.plot(times, baseline_infected, label='Baseline infected', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(times, cf['infected'], label=f'CF {i+1} infected', alpha=0.7)
    plt.title("Infected over time")
    plt.xlabel("Days")
    plt.ylabel("Infected individuals")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("optuna_CFE_pysirtem2.png")

# --- Baseline Parameters ---
t_span = (0, 100)
weeks = (t_span[1] // 7) + 1
t_eval = np.arange(t_span[0], t_span[1] + 1)
initial_conditions = init_cond

baseline_sol = pickle.load(open("baseline_data.p", "rb"))
# baseline_I = list(np.array(baseline_sol)[:, M_states["IA"]] + np.array(baseline_sol)[:, M_states["IS"]])
baseline_I = params['r'] * (np.array(baseline_sol)[:, M_states['PS']] + np.array(baseline_sol)[:, M_states['PA']] + np.array(baseline_sol)[:, M_states['IA']] + np.array(baseline_sol)[:, M_states['ATN']]) + np.array(baseline_sol)[:, M_states['IS']] + np.array(baseline_sol)[:, M_states['STN']]

# --- Optimization Objective: Minimize trajectory deviation, encourage theta diversity ---
baseline_phi_s1 = np.array(def_phi_s1[:weeks])
baseline_phi_a1 = np.array(def_phi_a1[:weeks])
N_WEEKS = len(baseline_phi_s1)

# Global memory (only retains accepted low-MSE ones)
archive_controls = []
archive_mses = []

def objective(trial):
    global archive_controls, archive_mses
    
    phi_s1_list = [trial.suggest_float(f"phi_s1_{i}", 0.01, 0.25) for i in range(N_WEEKS)]
    phi_a1_list = [trial.suggest_float(f"phi_a1_{i}", 0.01, 0.25) for i in range(N_WEEKS)]

    sol = runModel(init_cond, T_delay, params, t_span[1], def_inf_rate, phi_s1_list, def_phi_s2, phi_a1_list, def_phi_a2, def_g_rate) 
    I_sim = params['r'] * (np.array(sol)[:, M_states['PS']] + np.array(sol)[:, M_states['PA']] + np.array(sol)[:, M_states['IA']] + np.array(sol)[:, M_states['ATN']]) + np.array(sol)[:, M_states['IS']] + np.array(sol)[:, M_states['STN']]

    # Loss 1: How Similar to baseline infected curve
    mse = np.mean((np.array(I_sim) - np.array(baseline_I)) ** 2)

    # Loss 2: Diversity of phi_s1 and phi_a1 from baseline
    diff_phi_s1 = np.mean(np.abs((np.array(phi_s1_list) - np.array(baseline_phi_s1)) / np.array(baseline_phi_s1)))
    diff_phi_a1 = np.mean(np.abs((np.array(phi_a1_list) - np.array(baseline_phi_a1)) / np.array(baseline_phi_a1)))
    diversity_from_baseline = diff_phi_s1 + diff_phi_a1

    # Loss 3: Diversity from previous trials
    diversity_from_others = 0
    if archive_controls:
        current = np.concatenate([phi_s1_list, phi_a1_list])
        distances = [np.mean(np.abs(current - np.concatenate(past))) for past in archive_controls]
        # Use min distance to any previous as penalty (closer → more penalty)
        min_dist = min(distances)
        diversity_from_others = min_dist
    else:
        diversity_from_others = 1.0

    # --- Save candidate if good enough based on percentile ---
    archive_mses.append(mse)
    threshold = np.percentile(archive_mses, 25)  # Keep only top 25% mse
    if mse <= threshold:
        archive_controls.append((phi_s1_list.copy(), phi_a1_list.copy()))


    return mse - 0.1 * diversity_from_baseline - 0.1 * diversity_from_others

# --- Run Optimization ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300)

# --- Extract N best counterfactuals ---
N = 40
top_trials = sorted(study.trials, key=lambda x: x.value)[:N]

counterfactuals = []
for trial in top_trials:
    phi_s1_list = [trial.params[f"phi_s1_{i}"] for i in range(N_WEEKS)]
    phi_a1_list = [trial.params[f"phi_a1_{i}"] for i in range(N_WEEKS)]
    sol = runModel(init_cond, T_delay, params, t_span[1], def_inf_rate, phi_s1_list, def_phi_s2, phi_a1_list, def_phi_a2, def_g_rate) 
    I_sim = list(params['r'] * (np.array(sol)[:, M_states['PS']] + np.array(sol)[:, M_states['PA']] + np.array(sol)[:, M_states['IA']] + np.array(sol)[:, M_states['ATN']]) + np.array(sol)[:, M_states['IS']] + np.array(sol)[:, M_states['STN']])

    mse = np.mean((np.array(I_sim) - np.array(baseline_I)) ** 2)
    counterfactuals.append({
        "phi_s1": phi_s1_list,
        "phi_a1": phi_a1_list,
        "infected": I_sim,
        "mse": mse
    })



plot_results(baseline_phi_s1, baseline_phi_a1, baseline_I, counterfactuals, t_span)
# 3 is infected and good, 100 trials
# 2 is infected and 300 trials, possibly good
pickle.dump([counterfactuals, study.trials, baseline_phi_s1, baseline_phi_a1, baseline_I], open("CFE_sols2.p", "wb"))
'''