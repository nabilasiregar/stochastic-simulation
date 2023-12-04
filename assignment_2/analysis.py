"""
Comprehensive file to provide all of the statistics from the generated data from all of the queues.
Input: Path to file for csv with result data
Output:  Latex formatted summary statistics, 
p-values for ANOVA between M/M/n queues and for each n in the M/X/n queues and their Tukey Post-hoc results, 
p-values for T-tests between M/M/n queues and their M/LN/X counterparts, 
confidence intervals for each type of queue, 
plots for the average waiting times for each queue with their confidence interval
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#Collecting the data from the csv and grouping them
f = pd.read_csv("./simulation_results/results.csv", header=0)
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


#Creating the general stats and displaying in LaTeX 
general_stats = df.groupby(["n_server", "dist_serve", "priority", "preempt"])["avg_waiting_time"].agg(["mean", "std"])
print(general_stats.style.to_latex())


#Kruskal-Wallis tests 
mm1_kw_statistic, mm1_kw_p_value = stats.kruskal(mm1_data, prio_mm1_data, preempt_mm1_data, hyp_mm1_data, log_mm1_data, preempt_log_mm1_data)
print(f"P_value for Kruskal-Wallis test of M/X/1 Variants without LN: {mm1_kw_p_value}")

mm2_kw_statistic, mm2_kw_p_value = stats.kruskal(mm2_data, prio_mm2_data, preempt_mm2_data, hyp_mm2_data, log_mm2_data, preempt_log_mm2_data)
print(f"P_value for Kruskal-Wallis test of M/X/2 Variants without LN: {mm2_kw_p_value}")

mm4_kw_statistic, mm4_kw_p_value = stats.kruskal(mm4_data, prio_mm4_data, preempt_mm4_data, hyp_mm4_data, log_mm4_data, preempt_log_mm4_data)
print(f"P_value for Kruskal-Wallis test of M/X/4 Variants without LN: {mm4_kw_p_value}")
print()


#Tukey Post-hoc tests for each significant Kruskall-Wallis
tukey = pairwise_tukeyhsd(np.concatenate([mm1_data.values, prio_mm1_data.values, preempt_mm1_data.values, hyp_mm1_data.values, log_mm1_data.values, preempt_log_mm1_data.values]),
                groups=np.repeat(["M/M/1", "M/M/1NP", "M/M/1P", "M/H/1", "M/LN/1", "M/LN/1P"], [len(mm1_data), len(prio_mm1_data), len(preempt_mm1_data), len(hyp_mm1_data), len(log_mm1_data), len(preempt_log_mm1_data)]))
print("Tukey for M/X/1 without LN")
print(tukey)
print()

tukey = pairwise_tukeyhsd(np.concatenate([mm2_data.values, prio_mm2_data.values, preempt_mm2_data.values, hyp_mm2_data.values, log_mm2_data.values, preempt_log_mm2_data.values]),
                groups=np.repeat(["M/M/2", "M/M/2NP", "M/M/2P", "M/H/2", "M/LN/2", "M/LN/2P"], [len(mm2_data), len(prio_mm2_data), len(preempt_mm2_data), len(hyp_mm2_data), len(log_mm2_data), len(preempt_log_mm2_data)]))
print("Tukey for M/X/2 without LN")
print(tukey)
print()

tukey = pairwise_tukeyhsd(np.concatenate([mm4_data.values, prio_mm4_data.values, preempt_mm4_data.values, hyp_mm4_data.values, log_mm4_data.values, preempt_log_mm4_data.values]),
                groups=np.repeat(["M/M/4", "M/M/4NP", "M/M/4P", "M/H/4", "M/LN/4", "M/LN/4P"], [len(mm4_data), len(prio_mm4_data), len(preempt_mm4_data), len(hyp_mm4_data), len(log_mm4_data), len(preempt_log_mm4_data)]))
print("Tukey for M/X/4 without LN")
print(tukey)
print()


#Shapiro-Wilk Tests for Normality
data_arrays = [mm1_data, mm2_data, mm4_data, prio_mm1_data, prio_mm2_data, prio_mm4_data,
        preempt_mm1_data, preempt_mm2_data, preempt_mm4_data,
        log_mm1_data, log_mm2_data, log_mm4_data,
        preempt_log_mm1_data, preempt_log_mm2_data, preempt_log_mm4_data,
        hyp_mm1_data, hyp_mm2_data, hyp_mm4_data]

data_names = ["mm1", "mm2", "mm4", "prio_mm1", "prio_mm2", "prio_mm4",
        "preempt_mm1", "preempt_mm2", "preempt_mm4",
        "log_mm1", "log_mm2", "log_mm4",
        "preempt_log_mm1", "preempt_log_mm2", "preempt_log_mm4",
        "hyp_mm1", "hyp_mm2", "hyp_mm4"]

shapiro_stats = []
shapiro_pvalues = []
for data in data_arrays:
        shapiro_stat, shapiro_pvalue = stats.shapiro(data)
        shapiro_stats.append(shapiro_stat)
        shapiro_pvalues.append(shapiro_pvalue)

for i, p_value in enumerate(shapiro_pvalues):
        print(f"Shapiro-Wilk p-value for {data_names[i]}: {p_value}")
        print()


#Create and display the confidence intervals
all_methods = ["M/M/1", "M/M/1NP", "M/M/1P", "M/LN/1P", "M/H/1", "M/M/2", "M/M/2NP", "M/M/2P", "M/LN/2P", "M/H/2", "M/M/4", "M/M/4NP", "M/M/4P", "M/LN/4P", "M/H/4", "M/LN/1", "M/LN/2", "M/LN/4"]
all_means = [mm1_wait_mean, prio_mm1_wait_mean, preempt_mm1_wait_mean, preempt_log_mm1_wait_mean, hyp_mm1_wait_mean,
        mm2_wait_mean, prio_mm2_wait_mean, preempt_mm2_wait_mean, preempt_log_mm2_wait_mean, hyp_mm2_wait_mean,
        mm4_wait_mean, prio_mm4_wait_mean, preempt_mm4_wait_mean, preempt_log_mm4_wait_mean, hyp_mm4_wait_mean, log_mm1_wait_mean, log_mm2_wait_mean, log_mm4_wait_mean]
all_stds = [mm1_wait_std, prio_mm1_wait_std, preempt_mm1_wait_std, preempt_log_mm1_wait_std, hyp_mm1_wait_std, 
        mm2_wait_std, prio_mm2_wait_std, preempt_mm2_wait_std, preempt_log_mm2_wait_std, hyp_mm2_wait_std, 
        mm4_wait_std, prio_mm4_wait_std, preempt_mm4_wait_std, preempt_log_mm4_wait_std, hyp_mm4_wait_std, log_mm1_wait_std, log_mm2_wait_std, log_mm4_wait_std]

for i, method in enumerate(all_methods):
        alpha = 0.01
        standard_error = all_stds[i]/ np.sqrt(len(mm1_data))
        df = len(mm1_data) - 1
        conf_interval = stats.t.interval(1-alpha, df, all_means[i], scale=standard_error)
        rounded_conf_interval = tuple(round(value, 3) for value in conf_interval)
        print(f"{all_methods[i]} Confidence Interval: {rounded_conf_interval}")


#One-Sample T-Tests for general M/M/n distributions
analytical_mean_mm1 = (0.98)*(1/(1 - 0.98))
analytical_mean_mm2 = ((2*(0.98)**2)/(1 + (0.98)))*(1/(1 - 0.98))*(1/2)
analytical_mean_mm4 = ((32*(0.98)**4)/(8*(0.98)**3 + 12*(0.98)**2 + 9*(0.98) + 3))*(1/(1 - 0.98))*(1/4)

test_statistic1, p_value1 = stats.ttest_1samp(mm1_data.values, analytical_mean_mm1, alternative="two-sided")
print()
print(f"P-value 1-Sample T-test for MM1 data: {p_value1}")

test_statistic2, p_value2 = stats.ttest_1samp(mm2_data.values, analytical_mean_mm2, alternative="two-sided")
print(f"P-value 1-Sample T-test for MM2 data: {p_value2}")

test_statistic3, p_value4 = stats.ttest_1samp(mm4_data.values, analytical_mean_mm4, alternative="two-sided")
print(f"P-value 1-Sample T-test for MM4 data: {p_value4}")


#Creating bar charts for M/X/n queues excluding M/LN/n queues
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

methods = ["M/M/1NP", "M/M/1P", "M/LN/1P", "M/H/1", "M/M/2NP", "M/M/2P", "M/LN/2P", "M/H/2", "M/M/4NP", "M/M/4P", "M/LN/4P", "M/H/4"]
means = [prio_mm1_wait_mean, preempt_mm1_wait_mean, preempt_log_mm1_wait_mean, hyp_mm1_wait_mean, 
        prio_mm2_wait_mean, preempt_mm2_wait_mean, preempt_log_mm2_wait_mean, hyp_mm2_wait_mean, 
        prio_mm4_wait_mean, preempt_mm4_wait_mean, preempt_log_mm4_wait_mean, hyp_mm4_wait_mean]
stds = [prio_mm1_wait_std, preempt_mm1_wait_std, preempt_log_mm1_wait_std, hyp_mm1_wait_std, 
        prio_mm2_wait_std, preempt_mm2_wait_std, preempt_log_mm2_wait_std, hyp_mm2_wait_std, 
        prio_mm4_wait_std, preempt_mm4_wait_std, preempt_log_mm4_wait_std, hyp_mm4_wait_std]

log_methods = ["M/M/1", "M/LN/1", "M/M/2", "M/LN/2","M/M/4", "M/LN/4"]
log_means = [mm1_wait_mean, log_mm1_wait_mean, mm2_wait_mean, log_mm2_wait_mean, mm4_wait_mean, log_mm4_wait_mean]
log_stds = [mm1_wait_std, log_mm1_wait_std, mm2_wait_std, log_mm2_wait_std, mm4_wait_std, log_mm4_wait_std]

plt.figure(figsize=(12, 6))
for i, method in enumerate(methods):
        alpha = 0.05
        standard_error = stds[i]/ np.sqrt(20)
        df = 19
        color = to_rgba(method_colors[method])
        plt.bar(i, means[i], yerr=standard_error, capsize=5, color=color, label=method)
        
plt.xticks([1.5, 5.5, 9.5], ["n = 1", "n = 2", "n = 4"], fontsize = 14)
plt.ylabel('Expected Waiting Time', fontsize = 16)
plt.yticks(fontsize = 14)
plt.legend(["M/M/n Nonpreemptive Priority", "M/M/n Preemptive Priority", "M/LN/n Preemptive Priority", "M/H/n"], loc='upper right', fontsize = 14)
plt.title('Expected Waiting Times for Varying Queue Types', fontsize = 16)
plt.tight_layout()
plt.show()
print()


#Creating Bar Charts for M/M/n queues against M/LN/n queues
plt.figure(figsize=(12, 6))
for i, log_method in enumerate(log_methods):
        alpha = 0.05
        log_standard_error = log_stds[i]/ np.sqrt(20)
        color = to_rgba(method_colors[log_method])
        plt.bar(i, log_means[i], yerr=log_standard_error, capsize=5, color=color, label=log_method)

plt.xticks([0.5, 2.5, 4.5], ["n = 1", "n = 2", "n = 4"], fontsize = 14)
plt.ylabel('Expected Waiting Time', fontsize = 16)
plt.yticks(fontsize = 14)
plt.legend(["M/M/n", "M/LN/n"], loc='upper right', fontsize = 14)
plt.title('Expected Waiting Times for Varying Queue Types - Comparing M/M/n to M/LN/n', fontsize = 16)
plt.tight_layout()
plt.show()
