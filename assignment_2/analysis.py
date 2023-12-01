import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import statsmodels.stats.power as smp
from statsmodels.stats.multicomp import MultiComparison

def statistics(filename):
    """
    Comprehensive function to provide all of the statistics from the generated data from all of the queues.
    Input: Path to file for csv with result data
    Output:  Latex formatted summary statistics, 
    p-values for ANOVA between M/M/n queues and for each n in the M/X/n queues and their Tukey Post-hoc results, 
    p-values for T-tests between M/M/n queues and their M/LN/X counterparts, 
    confidence intervals for each type of queue, 
    plots for the average waiting times for each queue with their confidence interval
    """
    #Collecting the data from the csv and grouping them
    f = pd.read_csv(filename, header=0)
    df = pd.DataFrame(f)
        
    mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)]["waiting_time"]
    mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)]["waiting_time"]
    mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)]["waiting_time"]
    print(len(mm2_data.values))
    prio_mm1_data = df[(df["n_server"] == 1) & (df["priority"] ==True) & (df["preempt"] == False)]["waiting_time"]
    prio_mm2_data = df[(df["n_server"] == 2) & (df["priority"] == True) & (df["preempt"] == False)]["waiting_time"]
    prio_mm4_data = df[(df["n_server"] == 4) & (df["priority"] == True) & (df["preempt"] == False)]["waiting_time"]
    
    log_mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    log_mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    log_mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    
    hyp_mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "hyperexponential")]["waiting_time"]
    hyp_mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "hyperexponential")]["waiting_time"]
    hyp_mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "hyperexponential")]["waiting_time"]
    
    preempt_mm1_data = df[(df["n_server"] == 1) & (df["preempt"] == True) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    preempt_mm2_data = df[(df["n_server"] == 2) & (df["preempt"] == True) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    preempt_mm4_data = df[(df["n_server"] == 4) & (df["preempt"] == True) & (df["dist_serve"] == "lognormal")]["waiting_time"]

    preempt_log_mm1_data = df[(df["n_server"] == 1) & (df["priority"] == True) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    preempt_log_mm2_data = df[(df["n_server"] == 2) & (df["priority"] == True) & (df["dist_serve"] == "lognormal")]["waiting_time"]
    preempt_log_mm4_data = df[(df["n_server"] == 4) & (df["priority"] == True) & (df["dist_serve"] == "lognormal")]["waiting_time"]

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
    
    preempt_mm1_wait_mean = np.mean(preempt_mm1_data)
    preempt_mm2_wait_mean = np.mean(preempt_mm2_data)
    preempt_mm4_wait_mean = np.mean(preempt_mm4_data)
    
    preempt_log_mm1_wait_mean = np.mean(preempt_log_mm1_data)
    preempt_log_mm2_wait_mean = np.mean(preempt_log_mm2_data)
    preempt_log_mm4_wait_mean = np.mean(preempt_log_mm4_data)
    
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
    
    preempt_mm1_wait_std = np.std(preempt_mm1_data)
    preempt_mm2_wait_std = np.std(preempt_mm2_data)
    preempt_mm4_wait_std = np.std(preempt_mm4_data)
    
    preempt_log_mm1_wait_std = np.std(preempt_log_mm1_data)
    preempt_log_mm2_wait_std = np.std(preempt_log_mm2_data)
    preempt_log_mm4_wait_std = np.std(preempt_log_mm4_data)
    
    methods = ["M/M/1", "M/M/1NP", "M/M/1P", "M/H/1", "M/M/2", "M/M/2NP", "M/M/2P", "M/H/2", "M/M/4", "M/M/4NP", "M/M/4P", "M/H/4"]
    means = [mm1_wait_mean, prio_mm1_wait_mean, preempt_log_mm1_wait_mean, hyp_mm1_wait_mean, mm2_wait_mean, prio_mm2_wait_mean, preempt_mm2_wait_mean, hyp_mm2_wait_mean, mm4_wait_mean, prio_mm4_wait_mean, preempt_mm4_wait_mean, hyp_mm4_wait_mean]
    stds = [mm1_wait_std, prio_mm1_wait_std, preempt_mm1_wait_std, hyp_mm1_wait_std, mm2_wait_std, prio_mm2_wait_std, preempt_mm2_wait_std, hyp_mm2_wait_std, mm4_wait_std, prio_mm4_wait_std, preempt_mm4_wait_std, hyp_mm4_wait_std]
    log_methods = ["M/M/1", "M/LN/1P", "M/LN/1", "M/M/2", "M/LN/2P", "M/LN/2","M/M/4","M/LN/4P", "M/LN/4"]
    log_means = [mm1_wait_mean, preempt_log_mm1_wait_mean, log_mm1_wait_mean, mm2_wait_mean, preempt_log_mm2_wait_mean, log_mm2_wait_mean, mm4_wait_mean, preempt_log_mm4_wait_mean, log_mm4_wait_mean]
    log_stds = [mm1_wait_std, preempt_log_mm1_wait_std, log_mm1_wait_std, mm2_wait_std, preempt_log_mm2_wait_std, log_mm2_wait_std, mm4_wait_std, preempt_log_mm4_wait_std, log_mm4_wait_std]
    
    #Creating the general stats and displaying in LaTeX 
    alpha = 0.01
    general_stats = df.groupby(["n_server", "dist_serve", "priority"])["waiting_time"].agg(["mean", "std", "min", "max"])
    print(general_stats.style.to_latex())
    
    #ANOVA for between M/M/n types and then between all M/X/n types for each n
    gen_statistic, gen_p_value = stats.f_oneway(mm1_data, mm2_data, mm4_data)
    print(f"P_value for General ANOVA: {gen_p_value}")
    print()
    mm1_statistic, mm1_p_value = stats.f_oneway(mm1_data, prio_mm1_data, preempt_mm1_data, hyp_mm1_data)
    print(f"P_value for ANOVA of M/X/1 Variants without LN: {mm1_p_value}")
    print()
    mm2_statistic, mm2_p_value = stats.f_oneway(mm2_data, prio_mm2_data, preempt_mm2_data, hyp_mm2_data)
    print(f"P_value for ANOVA of M/X/2 Variants without LN: {mm1_p_value}")
    print()
    mm4_statistic, mm4_p_value = stats.f_oneway(mm4_data, prio_mm4_data, preempt_mm4_data, hyp_mm4_data)
    print(f"P_value for ANOVA of M/X/4 Variants without LN: {mm1_p_value}")
    print()
    log_statistic, log_p_value = stats.f_oneway(log_mm1_data, log_mm2_data, log_mm4_data)
    print(f"P_value for ANOVA with LN: {log_p_value}")
    print()
    
    #Tukey Post-hoc tests for each significant ANOVA
    mc = MultiComparison(np.concatenate([mm1_data.values, mm2_data.values, mm4_data.values]),
                     groups=np.repeat(["M/M/1", "M/M/2", "M/M/4"], [len(mm1_data), len(mm2_data), len(mm4_data)]))
    print("Tukey for General Comparison")
    print(mc.tukeyhsd())
    print()
    
    mc = MultiComparison(np.concatenate([mm1_data.values, prio_mm1_data.values, preempt_mm1_data.values, hyp_mm1_data.values]),
                     groups=np.repeat(["M/M/1", "M/M/1NP", "M/M/1P", "M/H/1"], [len(mm1_data), len(prio_mm1_data), len(preempt_mm1_data), len(hyp_mm1_data)]))
    print("Tukey for M/X/1 without LN")
    print(mc.tukeyhsd())
    print()
    
    mc = MultiComparison(np.concatenate([mm2_data.values, prio_mm2_data.values, preempt_mm2_data.values, hyp_mm2_data.values]),
                     groups=np.repeat(["M/M/2", "M/M/2NP", "M/M/2P", "M/H/2"], [len(mm2_data), len(prio_mm2_data), len(preempt_mm2_data), len(hyp_mm2_data)]))
    print("Tukey for M/X/2 without LN")
    print(mc.tukeyhsd())
    print()
    
    mc = MultiComparison(np.concatenate([mm4_data.values, prio_mm4_data.values, preempt_mm4_data.values, hyp_mm4_data.values]),
                     groups=np.repeat(["M/M/4", "M/M/4NP", "M/M/4P", "M/H/4"], [len(mm4_data), len(prio_mm4_data), len(preempt_mm4_data), len(hyp_mm4_data)]))
    print("Tukey for M/X/4 without LN")
    print(mc.tukeyhsd())
    print()
    
    #Independent T-Tests between general M/M/n queues and corresponding M/LN/n queues
    t_statistic_1, p_value_1 = stats.ttest_ind(mm1_data, log_mm1_data)
    print(f"P-value for Independent T-test between M/M/1 and M/LN/1: {p_value_1}")
    t_statistic_2, p_value_2 = stats.ttest_ind(mm2_data, log_mm2_data)
    print(f"P-value for Independent T-test between M/M/2 and M/LN/2: {p_value_2}")
    t_statistic_4, p_value_4 = stats.ttest_ind(mm4_data, log_mm4_data)
    print(f"P-value for Independent T-test between M/M/4 and M/LN/4: {p_value_4}")
    print()

    
    #Creating the confidence intervals for M/X/n queues exclusing M/LN/n queues
    
    method_colors = {
    "M/M/1": "#8BBF9F",
    "M/M/1P": "#83BCFF",
    "M/LN/1": "#124559",
    "M/H/1": "#EE204D",
    "M/M/2": "#8BBF9F",
    "M/M/2P": "#83BCFF",
    "M/LN/2": "#124559",
    "M/H/2": "#EE204D",
    "M/M/4": "#8BBF9F",
    "M/M/4P": "#83BCFF",
    "M/LN/4": "#124559",
    "M/H/4": "#EE204D",
    }
    
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
            standard_error = stds[i]/ 10
            df = 99
            conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
            rounded_conf_interval = tuple(round(value, 3) for value in conf_interval)
            print(f"{methods[i]} Confidence Interval: {rounded_conf_interval}")
            color = to_rgba(method_colors[method])
            plt.bar(i, means[i], yerr=standard_error, capsize=5, color=color, label=method)
            
    plt.xticks(range(len(means)), [methods[i] for i in range(len(means))], fontsize = 12)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.title('Expected Waiting Times with Confidence Intervals', fontsize = 16)
    plt.show()
    print()
    
    #Creating confidence intervals for M/M/n queues against M/LN/n queues
    plt.figure(figsize=(10, 6))
    for i, log_method in enumerate(log_methods):
            log_standard_error = log_stds[i]/ 10
            df = 99
            log_conf_interval = stats.t.interval(1-alpha, df, log_means[i], scale=log_standard_error)
            log_rounded_conf_interval = tuple(round(value, 3) for value in log_conf_interval)
            print(f"{log_methods[i]} Confidence Interval for LN: {log_rounded_conf_interval}")
            color = to_rgba(method_colors[log_method])
            plt.bar(i, log_means[i], yerr=log_standard_error, capsize=5, color=color, label=log_method)

    plt.xticks(range(len(log_means)), [log_methods[i] for i in range(len(log_means))], fontsize = 12)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.title('Expected Waiting Times with Confidence Intervals', fontsize = 16)
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

    
