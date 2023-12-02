import numpy as np
import pandas as pd
import pingouin as pg
import math
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import statsmodels.stats.power as smp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
        
    mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)]["avg_waiting_time"]
    mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)]["avg_waiting_time"]
    mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "expovariate") & (df["priority"] == False)]["avg_waiting_time"]
    
    prio_mm1_data = df[(df["n_server"] == 1) & (df["priority"] ==True) & (df["preempt"] == False)]["avg_waiting_time"]
    prio_mm2_data = df[(df["n_server"] == 2) & (df["priority"] == True) & (df["preempt"] == False)]["avg_waiting_time"]
    prio_mm4_data = df[(df["n_server"] == 4) & (df["priority"] == True) & (df["preempt"] == False)]["avg_waiting_time"]
    
    log_mm1_data = df[(df["n_server"] == 1) & (df["priority"] == False) & (df["dist_serve"] == "lognormal")]["avg_waiting_time"]
    log_mm2_data = df[(df["n_server"] == 2) & (df["priority"] == False) & (df["dist_serve"] == "lognormal")]["avg_waiting_time"]
    log_mm4_data = df[(df["n_server"] == 4) & (df["priority"] == False) & (df["dist_serve"] == "lognormal")]["avg_waiting_time"]
    
    hyp_mm1_data = df[(df["n_server"] == 1) & (df["dist_serve"] == "hyperexponential")]["avg_waiting_time"]
    hyp_mm2_data = df[(df["n_server"] == 2) & (df["dist_serve"] == "hyperexponential")]["avg_waiting_time"]
    hyp_mm4_data = df[(df["n_server"] == 4) & (df["dist_serve"] == "hyperexponential")]["avg_waiting_time"]
    
    preempt_mm1_data = df[(df["n_server"] == 1) & (df["preempt"] == True) & (df["dist_serve"] == "expovariate")]["avg_waiting_time"]
    preempt_mm2_data = df[(df["n_server"] == 2) & (df["preempt"] == True) & (df["dist_serve"] == "expovariate")]["avg_waiting_time"]
    preempt_mm4_data = df[(df["n_server"] == 4) & (df["preempt"] == True) & (df["dist_serve"] == "expovariate")]["avg_waiting_time"]

    preempt_log_mm1_data = df[(df["n_server"] == 1) & (df["priority"] == True) & (df["dist_serve"] == "lognormal")]["avg_waiting_time"]
    preempt_log_mm2_data = df[(df["n_server"] == 2) & (df["priority"] == True) & (df["dist_serve"] == "lognormal")]["avg_waiting_time"]
    preempt_log_mm4_data = df[(df["n_server"] == 4) & (df["priority"] == True) & (df["dist_serve"] == "lognormal")]["avg_waiting_time"]

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
    
    log_methods = ["M/M/1P", "M/LN/1P", "M/LN/1", "M/M/2P", "M/LN/2P", "M/LN/2","M/M/4P","M/LN/4P", "M/LN/4"]
    log_means = [prio_mm1_wait_mean, preempt_log_mm1_wait_mean, log_mm1_wait_mean, prio_mm2_wait_mean, preempt_log_mm2_wait_mean, log_mm2_wait_mean, prio_mm4_wait_mean, preempt_log_mm4_wait_mean, log_mm4_wait_mean]
    log_stds = [prio_mm1_wait_std, preempt_log_mm1_wait_std, log_mm1_wait_std, prio_mm2_wait_std, preempt_log_mm2_wait_std, log_mm2_wait_std, prio_mm4_wait_std, preempt_log_mm4_wait_std, log_mm4_wait_std]

    
    #Creating the general stats and displaying in LaTeX 
    general_stats = df.groupby(["n_server", "dist_serve", "priority", "preempt"])["avg_waiting_time"].agg(["mean", "std"])
    print(general_stats.style.to_latex())
    

    #Kruskal-Wallis tests 
    gen_kw_statistic, gen_kw_p_value = stats.kruskal(mm1_data, mm2_data, mm4_data)
    print(f"P_value for General Kruskal-Wallis test: {gen_kw_p_value}")
    print()

    mm1_kw_statistic, mm1_kw_p_value = stats.kruskal(mm1_data, prio_mm1_data, preempt_mm1_data, hyp_mm1_data)
    print(f"P_value for Kruskal-Wallis test of M/X/1 Variants without LN: {mm1_kw_p_value}")
    print()

    mm2_kw_statistic, mm2_kw_p_value = stats.kruskal(mm2_data, prio_mm2_data, preempt_mm2_data, hyp_mm2_data)
    print(f"P_value for Kruskal-Wallis test of M/X/2 Variants without LN: {mm2_kw_p_value}")
    print()

    mm4_kw_statistic, mm4_kw_p_value = stats.kruskal(mm4_data, prio_mm4_data, preempt_mm4_data, hyp_mm4_data)
    print(f"P_value for Kruskal-Wallis test of M/X/4 Variants without LN: {mm4_kw_p_value}")
    print()

    log_kw_statistic, log_kw_p_value = stats.kruskal(log_mm1_data, log_mm2_data, log_mm4_data)
    print(f"P_value for Kruskal-Wallis test with LN: {log_kw_p_value}")
    print()
    
    #Tukey Post-hoc tests for each significant ANOVA
    tukey = pairwise_tukeyhsd(np.concatenate([mm1_data.values, mm2_data.values, mm4_data.values]),
                     groups=np.repeat(["M/M/1", "M/M/2", "M/M/4"], [len(mm1_data), len(mm2_data), len(mm4_data)]))
    print("Tukey for General Comparison")
    print(tukey)
    print()
    
    tukey = pairwise_tukeyhsd(np.concatenate([mm1_data.values, prio_mm1_data.values, preempt_mm1_data.values, hyp_mm1_data.values]),
                     groups=np.repeat(["M/M/1", "M/M/1NP", "M/M/1P", "M/H/1"], [len(mm1_data), len(prio_mm1_data), len(preempt_mm1_data), len(hyp_mm1_data)]))
    print("Tukey for M/X/1 without LN")
    print(tukey)
    print()
    
    tukey = pairwise_tukeyhsd(np.concatenate([mm2_data.values, prio_mm2_data.values, preempt_mm2_data.values, hyp_mm2_data.values]),
                     groups=np.repeat(["M/M/2", "M/M/2NP", "M/M/2P", "M/H/2"], [len(mm2_data), len(prio_mm2_data), len(preempt_mm2_data), len(hyp_mm2_data)]))
    print("Tukey for M/X/2 without LN")
    print(tukey)
    print()
    
    tukey = pairwise_tukeyhsd(np.concatenate([mm4_data.values, prio_mm4_data.values, preempt_mm4_data.values, hyp_mm4_data.values]),
                     groups=np.repeat(["M/M/4", "M/M/4NP", "M/M/4P", "M/H/4"], [len(mm4_data), len(prio_mm4_data), len(preempt_mm4_data), len(hyp_mm4_data)]))
    print("Tukey for M/X/4 without LN")
    print(tukey)
    print()
    
    #Shapiro-Wilk Tests for Normality
    shapiro_stat_group1, shapiro_pvalue_group1 = stats.shapiro(mm1_data)
    shapiro_stat_group2, shapiro_pvalue_group2 = stats.shapiro(mm2_data)
    shapiro_stat_group3, shapiro_pvalue_group3 = stats.shapiro(mm4_data)
    shapiro_stat_group4, shapiro_pvalue_group4 = stats.shapiro(prio_mm1_data)
    shapiro_stat_group5, shapiro_pvalue_group5 = stats.shapiro(prio_mm2_data)
    shapiro_stat_group6, shapiro_pvalue_group6 = stats.shapiro(prio_mm4_data)
    shapiro_stat_group7, shapiro_pvalue_group7 = stats.shapiro(preempt_mm1_data)
    shapiro_stat_group8, shapiro_pvalue_group8 = stats.shapiro(preempt_mm2_data)
    shapiro_stat_group9, shapiro_pvalue_group9 = stats.shapiro(preempt_mm4_data)
    shapiro_stat_group10, shapiro_pvalue_group10 = stats.shapiro(log_mm1_data)
    shapiro_stat_group11, shapiro_pvalue_group11 = stats.shapiro(log_mm2_data)
    shapiro_stat_group12, shapiro_pvalue_group12 = stats.shapiro(log_mm4_data)
    shapiro_stat_group13, shapiro_pvalue_group13 = stats.shapiro(preempt_log_mm1_data)
    shapiro_stat_group14, shapiro_pvalue_group14 = stats.shapiro(preempt_log_mm2_data)
    shapiro_stat_group15, shapiro_pvalue_group15 = stats.shapiro(preempt_log_mm4_data)
    shapiro_stat_group16, shapiro_pvalue_group16 = stats.shapiro(hyp_mm1_data)
    shapiro_stat_group17, shapiro_pvalue_group17 = stats.shapiro(hyp_mm2_data)
    shapiro_stat_group18, shapiro_pvalue_group18 = stats.shapiro(hyp_mm4_data)
    

    print(
        f"p-value: {shapiro_pvalue_group1}")
    print(
        f"p-value: {shapiro_pvalue_group2}")
    print(
        f"p-value: {shapiro_pvalue_group3}")
    print(
        f"p-value: {shapiro_pvalue_group4}")
    print(
        f"p-value: {shapiro_pvalue_group5}")
    print(
        f"p-value: {shapiro_pvalue_group6}")
    print(
        f"p-value: {shapiro_pvalue_group7}")
    print(
        f"p-value: {shapiro_pvalue_group8}")
    print(
        f"p-value: {shapiro_pvalue_group9}")
    print(
        f"p-value: {shapiro_pvalue_group10}")
    print(
        f"p-value: {shapiro_pvalue_group12}")
    print(
        f"p-value: {shapiro_pvalue_group12}")
    print(
        f"p-value: {shapiro_pvalue_group13}")
    print(
        f"p-value: {shapiro_pvalue_group14}")
    print(
        f"p-value: {shapiro_pvalue_group15}")
    print(
        f"p-value: {shapiro_pvalue_group16}")
    print(
        f"p-value: {shapiro_pvalue_group17}")
    print(
        f"p-value: {shapiro_pvalue_group18}")

    
    #Creating the confidence intervals for M/X/n queues exclusing M/LN/n queues
    
    method_colors = {
    "M/M/1": "#8BBF9F",
    "M/M/1NP": "#83BCFF",
    "M/M/1P": "#59344F",
    "M/LN/1": "#124559",
    "M/LN/1P": "#FFA500",
    "M/H/1": "#EE204D",
    "M/M/2": "#8BBF9F",
    "M/M/2NP": "#83BCFF",
    "M/M/2P": "#59344F",
    "M/LN/2": "#124559",
    "M/LN/2P": "#FFA500",
    "M/H/2": "#EE204D",
    "M/M/4": "#8BBF9F",
    "M/M/4NP": "#83BCFF",
    "M/M/4P": "#59344F",
    "M/LN/4": "#124559",
    "M/LN/4P": "#FFA500",
    "M/H/4": "#EE204D",
    }
    
    plt.figure(figsize=(12, 6))
    for i, method in enumerate(methods):
            alpha = 0.05
            standard_error = stds[i]/ 10
            df = 99
            conf_interval = stats.t.interval(1-alpha, df, means[i], scale=standard_error)
            rounded_conf_interval = tuple(round(value, 3) for value in conf_interval)
            print(f"{methods[i]} Confidence Interval: {rounded_conf_interval}")
            color = to_rgba(method_colors[method])
            plt.bar(i, means[i], yerr=standard_error, capsize=5, color=color, label=method)
            
    plt.xticks([1.5,5.5,9.5], ["n = 1", "n = 2", "n = 4"], fontsize = 14)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.legend(["M/M/n", "M/M/n Nonpreemptive", "M/M/n Preemptive", "M/H/n"], loc='upper right', bbox_to_anchor=(1.395, 1.02), fontsize = 14)
    plt.title('Expected Waiting Times with Confidence Intervals', fontsize = 16)
    plt.tight_layout()
    plt.show()
    print()
    
    #Creating confidence intervals for M/M/n queues against M/LN/n queues
    plt.figure(figsize=(12, 6))
    for i, log_method in enumerate(log_methods):
            alpha = 0.05
            log_standard_error = log_stds[i]/ 10
            df = 99
            log_conf_interval = stats.t.interval(1-alpha, df, log_means[i], scale=log_standard_error)
            log_rounded_conf_interval = tuple(round(value, 3) for value in log_conf_interval)
            print(f"{log_methods[i]} Confidence Interval for LN: {log_rounded_conf_interval}")
            color = to_rgba(method_colors[log_method])
            plt.bar(i, log_means[i], yerr=log_standard_error, capsize=5, color=color, label=log_method)

    plt.xticks([1, 4, 7], ["n = 1", "n = 2", "n = 4"], fontsize = 14)
    plt.ylabel('Expected Waiting Time', fontsize = 16)
    plt.yticks(fontsize = 14)
    plt.legend(["M/M/n Preemptive", "M/LN/n Preemptive", "M/LN/n"], loc='upper right', bbox_to_anchor=(1.35, 1.02), fontsize = 14)
    plt.title('Expected Waiting Times with Confidence Intervals', fontsize = 16)
    plt.tight_layout()
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

    
