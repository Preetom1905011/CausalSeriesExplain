import numpy as np
import pickle
from scipy.signal import correlate
from scipy.ndimage import shift
import matplotlib.pyplot as plt



def compute_temporal_and_yaxis_deviation(baseline, counterfactual, max_lag=30):
    L = min(len(baseline), len(counterfactual))
    baseline = baseline[:L]
    counterfactual = np.array(counterfactual[:L])

    best_mse = float('inf')
    best_lag = 0
    best_shifted = counterfactual.copy()

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted = np.pad(counterfactual[-lag:], (0, -lag), mode='edge')
        elif lag > 0:
            shifted = np.pad(counterfactual[:-lag], (lag, 0), mode='edge')
        else:
            shifted = counterfactual

        mse = np.mean((shifted - baseline) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_lag = lag
            best_shifted = shifted.copy()

    y_dev = np.mean(np.abs(best_shifted - baseline))
    return best_lag

# Load data
[GPT_gen, I_GPT, _] = pickle.load(open("CFE_sols_150_gpt_gen.p", "rb"))
# baseline_sol = pickle.load(open("baseline_data.p", "rb"))
[counterfactuals, study_trials, baseline_phi_s1, baseline_phi_a1, baseline_I] = pickle.load(open("CFE_sols_150.p", "rb"))

temporal_devs = []
y_axis_devs = []

for cf in I_GPT[:6]:
    # Ensure same length
    L = min(len(baseline_I), len(cf))
    baseline = baseline_I[:L]
    target = np.array(cf[:L])

    lag = compute_temporal_and_yaxis_deviation(baseline_I, cf, max_lag=30)
    temporal_devs.append(lag)
    
    # y_dev = np.mean(np.abs(target[:L] - baseline[:L]))
    y_dev = np.mean((target[:L] - baseline[:L]) ** 2)
    baseline_var = np.var(baseline[:L])
    nmse = y_dev / baseline_var
    y_axis_devs.append(nmse)

# Results
print("Temporal Deviations (in days):", temporal_devs)
print("Y-Axis Deviations:", y_axis_devs)

# Example lists (replace these with your actual data)
# temporal_devs = [...]
# y_axis_devs = [...]

x = list(range(1, len(temporal_devs) + 1))

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Iteration', fontsize=16)
# ax1.plot(x, temporal_devs, marker='o', color=color1)
ax1.set_xticks(x)

# color1 = 'tab:blue'
# ax1.set_ylabel('Temporal Deviation (in days)', color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)

# ax2 = ax1.twinx()  # instantiate a second axes sharing the same x-axis
ax2 = ax1

color2 = 'tab:red'
ax2.set_ylabel(r"Distance Measure L_{out}", color=color2, fontsize=14)
ax2.plot(x, y_axis_devs, marker='s', linestyle='--', color=color2, label=r"L_{out}(Y', $Y_{base}$)")
ax2.tick_params(axis='y', labelcolor=color2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

fig.tight_layout()
# plt.title('Deviation of Counterfactual Infected Trajectories')
plt.savefig("RQ1_new.png", dpi=300)
plt.show()
