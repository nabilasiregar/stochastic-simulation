from Engels_14947994_Sahrani_12661651_Siregar_1486305 import uniform_square, latin_hypercube, orthogonal, return_cdf
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
from numpy import var, linspace

n = 7**2
params = [0, 1, n]

uniform_variances = []
amateur_lhs_variances = []
amateur_orth_variances = []
pro_lhs_variances = []
pro_ortho_variances = []

for i in range(100):
    uniform_variances.append(var(uniform_square(*params)))
    amateur_lhs_variances.append(var(latin_hypercube(*params)))
    amateur_orth_variances.append(var(orthogonal(*params)))
    pro_lhs_variances.append(var(LatinHypercube(2).random(n)))
    pro_ortho_variances.append(var(LatinHypercube(2, strength=2).random(n)))

domain = list(range(100))
palette = {
    "green": (139, 191, 159),
    "blue": (131, 188, 255),
    "midnight": (18, 69, 89),
    "violet": (89, 52, 79),
    "crayola": (238, 32, 77)
}

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Comparison of Sampling Methods")

# Subplot 1
axs[0].set_title("Variance of the Sampling methods")
axs[0].plot(domain, uniform_variances, label='Uniform Sampling', color='green')
axs[0].plot(domain, amateur_lhs_variances,
            label='Amateur LHS Sampling', color='blue')
axs[0].plot(domain, amateur_orth_variances,
            label='Amateur Orthogonal Sampling', color='midnightblue')
axs[0].plot(domain, pro_lhs_variances,
            label='Pro LHS Sampling', color='violet')
axs[0].plot(domain, pro_ortho_variances,
            label='Pro Orthogonal Sampling', color='crimson')
axs[0].legend()

# Subplot 2
axs[1].set_title("Cumulative Distribution Functions")
axs[1].plot(linspace(*params), return_cdf(uniform_square(*params))
            [0], label='Uniform Sampling', color='green')
axs[1].plot(linspace(*params), return_cdf(latin_hypercube(*params))
            [0], label='Amateur LHS Sampling', color='blue')
axs[1].plot(linspace(*params), return_cdf(orthogonal(*params))[0],
            label='Amateur Orthogonal Sampling', color='midnightblue')
axs[1].plot(linspace(*params), return_cdf(LatinHypercube(2).random(n))
            [0], label='Pro LHS Sampling', color='violet')
axs[1].plot(linspace(*params), return_cdf(LatinHypercube(2, strength=2).random(n))
            [0], label='Pro Orthogonal Sampling', color='crimson')
axs[1].plot(linspace(*params), linspace(0, 1, n), color="gray")
axs[1].legend()

plt.show()

fig.show()
