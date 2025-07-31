
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import optuna
import pickle
import random
from Parameters_pf import *
from DDE_pf import *
from scipy.ndimage import gaussian_filter

def plot_phi_and_inf_on_axes(axs, phi_s, phi_a, inf, base_phi_s, base_phi_a, base_inf, tag):
    weeks = np.arange(1, (len(base_inf) // 7 + 1) + 1, 1)
    style = ["solid", "dotted", "dashdot", "dashed", (0, (5, 10))]
    days = np.arange(1, (len(base_inf) + 1), 1)

    # --- Combined phi_s and phi_a on axs[0] ---
    for i in range(len(phi_s)):
        axs[0].plot(days, np.repeat(phi_s[i], 7)[:len(days)], color="tomato", linestyle="solid",
                    label=r'LLM Generated $X_{s}$', linewidth=2, alpha=0.75)
    for i in range(len(phi_a)):
        axs[0].plot(days, np.repeat(phi_a[i], 7)[:len(days)], color="royalblue", linestyle="dashed",
                    label=r'LLM Generated $X_{a}$', linewidth=2, alpha=0.75)

    axs[0].set_title(tag, fontsize=16)
    axs[0].set_xlabel("Days", fontsize=16)
    axs[0].set_ylabel(r"$X_{s}, X_{a}$", fontsize=16)
    axs[0].set_ylim([0, 0.35])
    axs[0].legend(fontsize=14, loc="upper left")

    # --- Infected curve on axs[1] (formerly axs[2]) ---
    for i in range(len(inf)):
        y = inf[i]
        axs[1].plot(np.arange(1, len(base_inf)+1, 1), y, color="red", linestyle=style[i],
                    label=r'$Y_{final}$', linewidth=2)
        
        for j in range(20, len(y) - 1):
            if y[j] > y[j-1] and y[j] > y[j+1]:
                axs[1].axvline(x=j+1, color='red', linestyle=':', linewidth=1.2)
                axs[1].axhline(y=y[j], color='red', linestyle=':', linewidth=1.2)
            elif y[j] < y[j-1] and y[j] < y[j+1]:
                axs[1].axvline(x=j+1, color='red', linestyle=':', linewidth=1.2)
                axs[1].axhline(y=y[j], color='red', linestyle=':', linewidth=1.2)

    axs[1].plot(np.arange(1, len(base_inf)+1, 1), base_inf, color="dimgrey", linestyle=style[0],
                label=r'$Y_{base}$', linewidth=2, alpha=0.75)

    for j in range(20, len(base_inf) - 1):
        if base_inf[j] > base_inf[j-1] and base_inf[j] > base_inf[j+1]:
            axs[1].axvline(x=j+1, color='black', linestyle='--', linewidth=1.2)
            axs[1].axhline(y=base_inf[j], color='black', linestyle='--', linewidth=1.2)
        elif base_inf[j] < base_inf[j-1] and base_inf[j] < base_inf[j+1]:
            axs[1].axvline(x=j+1, color='black', linestyle='--', linewidth=1.2)
            axs[1].axhline(y=base_inf[j], color='black', linestyle='--', linewidth=1.2)

    axs[1].set_xlabel("Days", fontsize=16)
    axs[1].set_ylabel(r"$Y$", fontsize=16)
    # axs[1].set_ylim([0, 2200000]) #new
    axs[1].set_ylim([0, 120000]) #new 2 
    # axs[1].set_ylim([0, 16000]) # baseline original
    axs[1].legend(fontsize=14, loc="upper left")


# ------------- experiment 3 generalizability ------------------------
# for baseline new
fig, axs = plt.subplots(2, 2, figsize=(18, 8))
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
plt.savefig("compare_generalize_baseline_newrules_new.png", dpi=300)
plt.clf()
exit()
'''
# for baseline new 2
fig, axs = plt.subplots(2, 2, figsize=(18, 8))
axs = axs.T  # Transpose to get columns as individual plot blocks

# wrong rule - new2 (final)
phi_s1_new = [0.25, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01, 0.30, 0.01, 0.15, 0.15, 0.15]
phi_a1_new = [0.01, 0.01, 0.28, 0.28, 0.28, 0.28, 0.28, 0.10, 0.28, 0.28, 0.28, 0.28, 0.01, 0.01, 0.01]

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
plt.savefig("compare_generalize_baseline_newrules_new2.png", dpi=300)
plt.clf()
exit()
'''
# ------------------------- experiment 2 (context vs no-context) -------------------------------------
# [GPT_gen_cont, I_GPT_cont] = pickle.load( open("CFE_sols3_gpt_gen.p", "rb"))
[GPT_gen_cont, I_GPT_cont] = pickle.load( open("New_relearning_gpt_gen.p", "rb"))

# gpt context + rules
GPT_gen_cont = [GPT_gen_cont[6]]
I_GPT_cont = [list(I_GPT_cont[6])]

# gpt rules + no context
GPT_gen_no_cont = {0: {"phi_s": [# Weeks 0–2: Low
                                0.01, 0.01, 0.01,
                                # Weeks 2–4: Moderate
                                0.10, 0.11, 0.12,
                                # Weeks 4–6: SharpSpike
                                0.01, 0.20, 0.01,
                                # Weeks 6–8: DipThenRise
                                0.12, 0.05, 0.13,
                                # Weeks 8–10: SlowRise
                                0.08, 0.10, 0.12,
                                # Weeks 10–14: Moderate
                                0.10, 0.11, 0.12, 0.13, 0.12],
                   "phi_a": [
                                    # Weeks 0–2: SlowRise
                                    0.01, 0.02, 0.03,
                                    # Weeks 2–4: SlowRise
                                    0.04, 0.05, 0.06,
                                    # Weeks 4–6: Low
                                    0.01, 0.01, 0.01,
                                    # Weeks 6–8: Low
                                    0.01, 0.01, 0.01,
                                    # Weeks 8–10: HighStable
                                    0.19, 0.19, 0.19,
                                    # Weeks 10–14: SlowRise
                                    0.07, 0.08, 0.09, 0.10, 0.11
                                ]}
                }
I_GPT_no_cont = []
for i in range(len(GPT_gen_no_cont)):
    phi_s_new, phi_a_new = GPT_gen_no_cont[i]["phi_s"], GPT_gen_no_cont[i]["phi_a"]
    sol_GPT = runModel(init_cond, T_delay, params, 100, def_inf_rate, phi_s_new, def_phi_s2, phi_a_new, def_phi_a2, def_g_rate) 
    inf = list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])
    I_GPT_no_cont.append(list(inf))

# gpt no rules + no context
GPT_NO_RULES = {0: {"phi_s": [0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.28, 0.30, 0.27, 0.20, 0.13, 0.08, 0.10, 0.18, 0.25],
 "phi_a": [0.02, 0.04, 0.07, 0.12, 0.18, 0.25, 0.29, 0.28, 0.24, 0.17, 0.11, 0.07, 0.09, 0.16, 0.23]}}
phi_s_new, phi_a_new = GPT_NO_RULES[0]["phi_s"], GPT_NO_RULES[0]["phi_a"]
sol_GPT = runModel(init_cond, T_delay, params, 100, def_inf_rate, phi_s_new, def_phi_s2, phi_a_new, def_phi_a2, def_g_rate)  
I_NO_RULES = [list(params['r'] * (np.array(sol_GPT)[:, M_states['PS']] + np.array(sol_GPT)[:, M_states['PA']] + np.array(sol_GPT)[:, M_states['IA']] + np.array(sol_GPT)[:, M_states['ATN']]) + np.array(sol_GPT)[:, M_states['IS']] + np.array(sol_GPT)[:, M_states['STN']])]


    

baseline_sol = pickle.load(open("baseline_data.p", "rb"))
# baseline_I = list(np.array(baseline_sol)[:, M_states["IA"]] + np.array(baseline_sol)[:, M_states["IS"]])
baseline_I = params['r'] * (np.array(baseline_sol)[:, M_states['PS']] + np.array(baseline_sol)[:, M_states['PA']] + np.array(baseline_sol)[:, M_states['IA']] + np.array(baseline_sol)[:, M_states['ATN']]) + np.array(baseline_sol)[:, M_states['IS']] + np.array(baseline_sol)[:, M_states['STN']]

fig, axs = plt.subplots(2, 3, figsize=(18, 8))
axs = axs.T  # Transpose to get columns as individual plot blocks

plot_phi_and_inf_on_axes(axs[0], [list(GPT_NO_RULES[0]["phi_s"])], [list(GPT_NO_RULES[0]["phi_a"])], I_NO_RULES, base_phi_s1, base_phi_a1, list(baseline_I), "No Rules + No Clustering Context")
plot_phi_and_inf_on_axes(axs[1], [list(GPT_gen_no_cont[0]["phi_s"])], [list(GPT_gen_no_cont[0]["phi_a"])], I_GPT_no_cont, base_phi_s1, base_phi_a1, list(baseline_I), "Rules + No Clustering Context")
plot_phi_and_inf_on_axes(axs[2], [GPT_gen_cont[0]["phi_s"]], [GPT_gen_cont[0]["phi_a"]], I_GPT_cont, base_phi_s1, base_phi_a1, list(baseline_I), "Rules + Clustering Context")


plt.tight_layout()
plt.savefig("compare_context_new.png", dpi=300)
plt.clf()

