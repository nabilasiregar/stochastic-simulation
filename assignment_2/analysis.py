import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.stats.power as smp

palette = {
    "green": (139, 191, 159),
    "blue": (131, 188, 255),
    "midnight": (18, 69, 89),
    "violet": (89, 52, 79),
    "crayola": (238, 32, 77)
}

normalized_palette = {key: tuple(val / 255.0 for val in value)
                      for key, value in palette.items()}

colors = list(normalized_palette.values())
color_stops = [i / (len(colors) - 1) for i in range(len(colors))]
cmap = LinearSegmentedColormap.from_list(
    "custom_gradient", list(zip(color_stops, colors)))


def statistics(filename):
    #Collecting the data from the csv
    f = pd.read_csv(filename, header=0)
    df = pd.DataFrame(f)
        
    mm1_data = df[(df["n_server"] == 1) & (df["priority"] == False)]["waiting_time"]
    mm2_data = df[(df["n_server"] == 2) & (df["priority"] == False)]["waiting_time"]
    mm4_data = df[(df["n_server"] == 4) & (df["priority"] == False)]["waiting_time"]
    
    prio_mm1_data = df[(df["n_server"] == 1) & (df["priority"] == True)]["waiting_time"]
    prio_mm2_data = df[(df["n_server"] == 2) & (df["priority"] == True)]["waiting_time"]
    prio_mm4_data = df[(df["n_server"] == 4) & (df["priority"] == True)]["waiting_time"]
    
    log_mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    log_mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    log_mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    
    hyp_mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "hyperexponential")]["waiting_time"]
    hyp_mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "hyperexponential")]["waiting_time"]
    hyp_mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "hyperexponential")]["waiting_time"]
    
    mm1_wait_mean = np.mean(mm1_data)
    mm2_wait_mean = np.mean(mm2_data)
    mm4_wait_mean = np.mean(mm4_data)
    
    prio_mm1_wait_mean = np.mean(prio_mm1_data)
    prio_mm2_wait_mean = np.mean(prio_mm2_data)
    prio_mm4_wait_mean = np.mean(prio_mm4_data)
    
    log_mm1_wait_mean = np.mean(log_mm1_data)
    log_mm2_wait_mean = np.mean(log_mm2_data)
    log_mm4_wait_mean = np.mean(log_mm4_data)
    
    hyp_mm1_wait_mean = np.mean(hyp_mm1_data)
    hyp_mm2_wait_mean = np.mean(hyp_mm2_data)
    hyp_mm4_wait_mean = np.mean(hyp_mm4_data)
    
    mm1_wait_std = np.std(mm1_data)
    mm2_wait_std = np.std(mm2_data)
    mm4_wait_std = np.std(mm4_data)
    
    prio_mm1_wait_std = np.std(prio_mm1_data)
    prio_mm2_wait_std = np.std(prio_mm2_data)
    prio_mm4_wait_std = np.std(prio_mm4_data)
    
    log_mm1_wait_std = np.std(log_mm1_data)
    log_mm2_wait_std = np.std(log_mm2_data)
    log_mm4_wait_std = np.std(log_mm4_data)
    
    hyp_mm1_wait_std = np.std(hyp_mm1_data)
    hyp_mm2_wait_std = np.std(hyp_mm2_data)
    hyp_mm4_wait_std = np.std(hyp_mm4_data)
    
    methods = ["M/M/1", "M/M/1 Priority", "M/LN/1", "M/H/1", "M/M/2", "M/M/2 Priority", "M/LN/2", "M/H/2", "M/M/4", "M/M/4 Priority", "M/LN/4", "M/H/4"]
    means = [mm1_wait_mean, prio_mm1_wait_mean, log_mm1_wait_mean, hyp_mm1_wait_mean, mm2_wait_mean, prio_mm2_wait_mean, log_mm2_wait_mean, hyp_mm2_wait_mean, mm4_wait_mean, prio_mm4_wait_mean, log_mm4_wait_mean, hyp_mm4_wait_mean]
    stds = [mm1_wait_std, prio_mm1_wait_std, log_mm1_wait_std, hyp_mm1_wait_std, mm2_wait_std, prio_mm2_wait_std, log_mm2_wait_std, hyp_mm2_wait_std, mm4_wait_std, prio_mm4_wait_std, log_mm4_wait_mean, hyp_mm4_wait_std]
    
    #Creating the general stats
    alpha = 0.01
    general_stats = df.groupby(["n_server", "dist_serve", "priority"])["waiting_time"].agg(["mean", "std", "min", "max"])
    print(general_stats.style.to_latex())
    
    #ANOVA        
    gen_statistic, gen_p_value = stats.f_oneway(mm1_data, mm2_data, mm4_data)
    print(f"P_value for General ANOVA: {gen_p_value}")
    print()
    mm1_statistic, mm1_p_value = stats.f_oneway(mm1_data, prio_mm1_data, log_mm1_data, hyp_mm1_data)
    print(f"P_value for ANOVA of all M/X/1 Variants: {mm1_p_value}")
    print()
    mm2_statistic, mm2_p_value = stats.f_oneway(mm2_data, prio_mm2_data, log_mm2_data, hyp_mm2_data)
    print(f"P_value for ANOVA of all M/X/2 Variants: {mm1_p_value}")
    print()
    mm4_statistic, mm4_p_value = stats.f_oneway(mm4_data, prio_mm4_data, log_mm4_data, hyp_mm4_data)
    print(f"P_value for ANOVA of all M/X/4 Variants: {mm1_p_value}")
    print()
    
    #Creating the confidence intervals
    
    #plt.figure(figsize=(10, 6))
    
    for i in range(len(stds)):
            standard_error = stds[i]/ 10
            df = 99
            conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
            rounded_conf_interval = tuple(round(value, 3) for value in conf_interval)
            print(f'Mean: {means[i]}')
            print(f"{methods[i]} Confidence Interval: {rounded_conf_interval}")
            plt.errorbar(x=i, y=means[i], yerr=(conf_interval[1] - means[i]), fmt='o', label=methods[i])

    plt.xticks(range(len(means)), [methods[i] for i in range(len(means))], fontsize = 12)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.title('Confidence Intervals for Expected Waiting Times', fontsize = 16)
    plt.show()

statistics("./simulation_results/results.csv")


def expected_wait_time_n(n, lam, mu):
    rho = lam/(n* mu)
    delay_probability = ((n* rho) ** n / math.factorial(n)) * (((1 - rho) *  np.sum([(n * rho) ** m / math.factorial(m) for m in range(n)]) + (n * rho) ** n / math.factorial(n)) ** -1)
    expected_wait_time_n = delay_probability / (n * mu * (1 - rho))
    
    return expected_wait_time_n

def power_analysis(lam, mu):

    mm1_wait = expected_wait_time_n(2, lam, mu)
    mm2_wait = expected_wait_time_n(2, 2*lam, mu)
    mm4_wait = expected_wait_time_n(4, 4*lam, mu)

    
    var_mm1_wait = (1/mm1_wait)**2
    var_mm2_wait = (1/mm2_wait)**2
    var_mm4_wait = (1/mm4_wait)**2
    
    
    

    waiting_time_data = np.array([mm1_wait, mm2_wait, mm4_wait])
    overall_mean = np.mean(waiting_time_data)
    ss_total = np.sum((waiting_time_data - overall_mean) ** 2)
    ss_between = np.sum([(np.mean(queue_type) - overall_mean) ** 2 for queue_type in waiting_time_data])
    eta_squared = ss_between /ss_total
    
    sample_size = smp.FTestAnovaPower().solve_power(effect_size=1, alpha=0.01, k_groups=3, power=0.8, nobs=None)
    print(f"Number of samples needed for lam = {lam} and mu = {mu}: {int(np.ceil(sample_size))}")
    
power_analysis(0.3, 0.8)
    
    
