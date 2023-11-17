import pandas as pd
import scipy.stats as stats

f = pd.read_csv("mandelbrot_sample_size_comparison.csv", header=0)
df = pd.DataFrame(f)
alpha = 0.01

# Descriptive stats
point_estimated = df.groupby(["method", "sample_size"]).aggregate(
    ["mean", "std", "min", "max"])["area"]
point_estimated["Confidence Interval"] = [stats.t.interval(
    1-alpha, 99, mean, std/33) for mean, std in zip(point_estimated["mean"], point_estimated["std"])]
# print(point_estimated.to_latex(float_format="%.3f"))
print(point_estimated.head(100))

array_like = df.groupby("method")["area"].apply(
    list).reset_index()["area"].values

levene_stat, levene_pvalue = stats.levene(*array_like)
print(f"Levene's Test - Statistic: {levene_stat}, p-value: {levene_pvalue}")

bartlett_stat, bartlett_pvalue = stats.bartlett(*array_like)
print(
    f"Bartlett's Test - Statistic: {bartlett_stat}, p-value: {bartlett_pvalue}")

shapiro_stat_group1, shapiro_pvalue_group1 = stats.shapiro(array_like[0])
shapiro_stat_group2, shapiro_pvalue_group2 = stats.shapiro(array_like[1])
shapiro_stat_group3, shapiro_pvalue_group3 = stats.shapiro(array_like[2])
shapiro_stat_group4, shapiro_pvalue_group4 = stats.shapiro(array_like[3])

print(
    f"Shapiro-Wilk Test - Group 1: Statistic: {shapiro_stat_group1}, p-value: {shapiro_pvalue_group1}")
print(
    f"Shapiro-Wilk Test - Group 2: Statistic: {shapiro_stat_group2}, p-value: {shapiro_pvalue_group2}")
print(
    f"Shapiro-Wilk Test - Group 3: Statistic: {shapiro_stat_group3}, p-value: {shapiro_pvalue_group3}")
print(
    f"Shapiro-Wilk Test - Group 4: Statistic: {shapiro_stat_group4}, p-value: {shapiro_pvalue_group4}")
