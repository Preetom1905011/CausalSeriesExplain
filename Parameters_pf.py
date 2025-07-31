import random
# ------- inputs ---------

# number of states (CONSTANT)
num_states = 76

# initializing all states to 0
init_cond = [0 for n in range(num_states)]
# Initial Susceptible
init_cond[0] = 7278717   #FL-21600000   # WI-5800000         #MN-5600000  #AZ-7278717  # Susceptible
init_cond[2] = 10       # Symptomatic Inf.
init_cond[3] = 10       # Asymptomatic Inf.

# Total Population
N = sum(init_cond)

# Simulation Duration
N_t = 308

# -------- parameters ----------

# Testing Related Parameters (Passing to Equation), 
params = {
            # Testing Related Parameters (Used for Estimated Statistic, for instance # of Tests etc.)
            "phi_a_1": 0.0,      # Testing rate of Test 1 for Asymptomatc people 
            "phi_a_2": 0.001,      # Testing rate of Test 2 for Asymptomatc people
            "phi_s_1": 0.0,      # Testing rate of Test 1 for Symptomatc people
            "phi_s_2": 0.075,      # Testing rate of Test 2 for Symptomatc people
            "phi_se": 0.01,      # Testing rate for Serology
            "beta": 0.0,         # Infection Rate
            "beta_prime": 0.0,   # Higer Infection rate (Falsely Presumed Immune)
            "M": 1,              # Mixing Rate
            "True_P_1": 0.7,     # True Positive Accuracy for test 1 (1-True_P_1) means Type I error
            "True_P_2": 0.7,     # True Positive Accuracy for test 2
            "True_N_1": 0.95,    # True Negative Accuracy for test 1 (1-True_N_1) means Type II error
            "True_N_2": 0.95,    # True Negative Accuracy for test 2
            "True_P_SE": 0.84,   # True Positive Serology Test Accuracy
            "True_N_SE": 0.97,   # True Negative Serology Test Accuracy
            "N": N,              # Total Population

            # General Parameters (Passing to Equations) 3. **** User define
            "ili":  0,                 # Pr of flu symptoms (from CDC)
            "per_a":  0.16,            # Pr of being exposed to Asymptomatic
            "per_s":  0.84,            # Pr of being exposed to symptomatic
            "kappa_s_1":  0.0088/7,    # Mortality rate in Infected Symptomatic
            "kappa_s_2":  0.0088,      # Mortality rate in Quarantine
            "lambda_a":  1/3.5,        # Pr of Unknown Recover for Infected Symptomatic 
            "lambda_s":  1/7,          # Pr of Unknown Recover for Infected ASymptomatic     
            "hos_1":  0.06,            # Pr of Hospitalization for Quarantined
            "hos_2":  0.06/7,          # Pr of Hosp. for Infected Symptomatic
            "hos_3":  0.01,            # Pr of Hosp. for Falsely Quarantined with Flu
            "kappa_h_1":  0.074,       # Mortality rate in Hospital 1
            "kappa_h_2": 0.074,        # Mortality rate in Hospital 2        
            "g_beta": 0.0,             # user define
            "r": 0.51
}

# Time & Dealy Parameter (Passing to Equations) 4. ***** User define
T_delay = {
            "eta": 3.2,          #latent time for asymptomatic
            "omega": 3.2,        #latent time for symptomatic
            "tau_1": 3,          #Response Time 1 for testing result 
            "tau_2": 3,          #Response Time 2 for testing result
            "sigma": 5,          #Time for Seriology Testing result
            "gamma": 90,         #Time for Immunity lasts
            "lambda_q": 14,      #Quarantine Time
            "lambda_H_1": 6,     #Time in Hospital (Infected, go to hospital after get tested)
            "lambda_H_2": 6,     #Time in Hospital (Infected, directly go to hospital)
            "lambda_H_3": 2      #Time in Hospital (Flu symp)
}

M_states = {'S': 0, 'E': 1, 'IA': 2, 'IS': 3, 'AT_1': 4, 'ST_1': 5, 'QAP': 6, 'QSP': 7, 'ATN': 8, 'STN': 9, 'PS': 10, 
          'KR': 11, 'D': 12, 'FT_1': 13, 'FTN': 14, 'QFS': 15, 'NT_1': 16, 'NTN': 17, 'NTP': 18, 'IM': 19, 'FPI': 20, 
          'STI': 21, 'SRE': 22, 'UR': 23, 'FPS': 24, 'AT_3': 25, 'QAP_1': 26, 'ATN_1': 27, 'NT_2': 28, 'FT_2': 29, 'AT_2': 30, 
          'AT_4': 31, 'ST_2': 32, 'H_1': 33, 'H_2': 34, 'H_3': 35, 'ST_3': 36, 'FT_3': 37, 'ST_4': 38, 'PA': 39, 
          # Fictitious States for preparing delay
          'F_NT_1': 40, 'F_NT_2': 41, 'F_NTP': 42, 'F_FT_1': 43, 'F_FT_2': 44, 'F_QFS': 45, 
          'F_AT_1': 46, 'F_AT_2': 47, 'F_QAP': 48, 'F_ST_1': 49, 'F_ST_2': 50, 'F_QSP': 51, 
          'F_AT_3': 52, 'F_AT_4': 53, 'F_QAP_1': 54, 
          'F_H1': 55, 'F_ST_4': 56, 'F_H2': 57, 'F_ST_3': 58, 'F_H3': 59, 'F_FT_3': 60, 
          'F_IM': 61, 'F_STI': 62, 'F_SRE': 63, 'F_FPS': 64, 'F_PA': 65, 'F_PS': 66, 
          'F_GT1': 67, 'F_GT2': 68, 
          'GT_1': 69, 'GT_2': 70, 'F_QGP': 71, 'GTN': 72, 'QGP': 73, 'SF': 74, 'GS': 75}

# used for piecewise dynamics calc.
n = 7
m = 7


def_inf_rate = [0.99602978, 0.847793768, 0.724539091, 0.768295519, 0.09602978, 0.082793768, 0.07481209, 0.080955482, 
            0.240040082, 0.21358817, 0.242162475, 0.239525318, 0.42381072, 0.408029527, 0.36849255, 0.374748069, 
            0.171918459, 0.183664271, 0.170267097, 0.171243521, 0.135662311, 0.116481419, 0.126002507, 0.127221244, 
            0.21126249, 0.217145694, 0.213098174, 0.215815292, 0.273219098, 0.279144043, 0.317967676, 0.322431178, 
            0.342658964, 0.356393599, 0.340463537, 0.342096561, 0.358228876, 0.363243005, 0.330476919, 0.332643822, 
            0.264258173, 0.263336395, 0.289138929, 0.291891337]
def_phi_s = [0.098539469, 0.088464225, 0.070386173, 0.076488112, 0.098539469, 0.088464225, 0.070386173, 0.076488112, 
         0.139141285, 0.164298606, 0.144555141, 0.140163213, 0.222869525, 0.26100563, 0.274393986, 0.280416065, 
         0.301830851, 0.267270776, 0.278730759, 0.277770895, 0.349926403, 0.302143118, 0.280232672, 0.288712245, 
         0.224393238, 0.255934452, 0.268608084, 0.273929796, 0.284745678, 0.28250055, 0.28697757, 0.288674448, 
         0.298297769, 0.344262075, 0.374039837, 0.384387172, 0.503024073, 0.470948659, 0.419200354, 0.428005814, 
         0.395347304, 0.362813121, 0.352110047, 0.355734449]
def_g_rate = [0.004 for i in range(44)]

# def_inf_rate = [0.5 + random.uniform(-0.25, 0.35) for i in def_inf_rate]
# def_phi_s = [0.25 for i in range(44)]

def_phi_a1 = [i * 0.5 + random.uniform(0, 0.001) for i in def_phi_s]
# def_phi_a1 = [i * random.uniform(0.25, 0.75) + random.uniform(0, 0.001) for i in def_phi_s]
def_phi_a2 = [i * 0.05 + random.uniform(0, 0.001) for i in def_phi_s]
def_phi_s1 = [i * 0.5 + random.uniform(0, 0.001) for i in def_phi_s]
# def_phi_s1 = [i * random.uniform(0.25, 0.75) + random.uniform(0, 0.001) for i in def_phi_s]
def_phi_s2 = [i * 0.075 + random.uniform(0, 0.001) for i in def_phi_s]
