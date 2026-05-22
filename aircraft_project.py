import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest

data = 
{
    7907: [194, 209, 250, 279, 312, 493],
    7908: [413, 427, 485, 522, 622, 687, 696, 865],
    7909: [90, 100, 160, 346, 407, 456, 470, 494, 550, 570, 649, 733, 777, 836, 865, 983],
    7910: [74, 131, 179, 208, 710, 722, 792, 813, 842],
    7911: [55, 375, 431, 535, 755, 994],
    7912: [23, 284, 371, 378, 498, 512, 574, 621, 846, 917],
    7913: [97, 148, 159, 163, 304, 322, 464, 532, 609, 689, 690, 706, 812],
    7914: [50, 94, 196, 268, 290, 329, 332, 347, 544, 732, 811, 899, 945, 950, 955, 991],
    7915: [359, 368, 380, 650],
    7916: [50, 304, 309, 592, 627, 639],
    7917: [130, 623],
}

T = 1000
n_units = len(data)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Inter-failure times

failure_records = []
inter_records = []

for aircraft, times in data.items():
    previous_time = 0

    for failure_no, t in enumerate(times, start=1):
        failure_records.append({
            "aircraft": aircraft,
            "failure_no": failure_no,
            "time": t
        })

        inter_records.append({
            "aircraft": aircraft,
            "failure_no": failure_no,
            "inter_failure_time": t - previous_time,
            "failure_time": t
        })

        previous_time = t

failures = pd.DataFrame(failure_records)
inter_failures = pd.DataFrame(inter_records)

summary = inter_failures.groupby("aircraft")["inter_failure_time"].agg(
    failures="count",
    mean="mean",
    median="median",
    std="std",
    min="min",
    max="max"
)

summary["cv"] = summary["std"] / summary["mean"]
summary = summary.round(3)

summary.to_csv("results/interfailure_statistics.csv")
inter_failures.to_csv("results/interfailure_times.csv", index=False)

print("\nInter-failure time summary:")
print(summary)

# -------------------- MCRF calculation --------------------

all_times = np.sort(failures["time"].to_numpy())
unique_times = np.array(sorted(set(all_times)))

mcrf_values = []

for t in unique_times:
    counts = []

    for aircraft, times in data.items():
        count = np.sum(np.array(times) <= t)
        counts.append(count)

    mcrf_values.append(np.mean(counts))

mcrf_values = np.array(mcrf_values)

mcrf_df = pd.DataFrame({
    "time": unique_times,
    "mcrf": mcrf_values
})

mcrf_df.to_csv("results/mcrf_results.csv", index=False)

# HPP and Power Law NHPP estimation 

r = len(all_times)

# HPP: lambda_hat = total failures / total exposure
lambda_hpp = r / (n_units * T)

# Power Law Process / NHPP:
# Mean function per aircraft: m(t) = alpha * t^beta
beta_nhpp = r / np.sum(np.log(T / all_times))
alpha_nhpp = r / (n_units * T ** beta_nhpp)

# Log-likelihoods
ll_hpp = r * np.log(lambda_hpp) - n_units * lambda_hpp * T

ll_nhpp = (
    np.sum(np.log(alpha_nhpp * beta_nhpp) + (beta_nhpp - 1) * np.log(all_times))
    - n_units * alpha_nhpp * T ** beta_nhpp
)

# AIC and BIC
aic_hpp = 2 * 1 - 2 * ll_hpp
aic_nhpp = 2 * 2 - 2 * ll_nhpp

bic_hpp = np.log(r) * 1 - 2 * ll_hpp
bic_nhpp = np.log(r) * 2 - 2 * ll_nhpp

model_parameters = pd.DataFrame({
    "model": ["HPP", "Power Law NHPP"],
    "parameters": [
        f"lambda = {lambda_hpp:.6f}",
        f"alpha = {alpha_nhpp:.6f}, beta = {beta_nhpp:.6f}"
    ],
    "log_likelihood": [ll_hpp, ll_nhpp],
    "AIC": [aic_hpp, aic_nhpp],
    "BIC": [bic_hpp, bic_nhpp]
})

model_parameters.to_csv("results/model_parameters.csv", index=False)

print("\nModel estimates:")
print(model_parameters)

# Interval counts and goodness-of-fit 

bins = np.arange(0, 1100, 100)
observed_counts, edges = np.histogram(all_times, bins=bins)

expected_hpp = np.repeat(r / 10, 10)
expected_nhpp = n_units * alpha_nhpp * (edges[1:] ** beta_nhpp - edges[:-1] ** beta_nhpp)

interval_table = pd.DataFrame({
    "interval": [f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(len(observed_counts))],
    "observed": observed_counts,
    "HPP_expected": expected_hpp,
    "NHPP_expected": expected_nhpp
})

interval_table.to_csv("results/interval_counts.csv", index=False)

chi_hpp = np.sum((observed_counts - expected_hpp) ** 2 / expected_hpp)
p_hpp = 1 - chi2.cdf(chi_hpp, df=10 - 1 - 1)

chi_nhpp = np.sum((observed_counts - expected_nhpp) ** 2 / expected_nhpp)
p_nhpp = 1 - chi2.cdf(chi_nhpp, df=10 - 1 - 2)

ks_hpp = kstest(all_times / T, "uniform")
ks_nhpp = kstest((all_times / T) ** beta_nhpp, "uniform")

gof_table = pd.DataFrame({
    "diagnostic": [
        "100-day grouped chi-square, HPP",
        "100-day grouped chi-square, NHPP",
        "Conditional event-time KS, HPP",
        "Conditional event-time KS, NHPP"
    ],
    "statistic": [
        chi_hpp,
        chi_nhpp,
        ks_hpp.statistic,
        ks_nhpp.statistic
    ],
    "p_value": [
        p_hpp,
        p_nhpp,
        ks_hpp.pvalue,
        ks_nhpp.pvalue
    ]
})

gof_table.to_csv("results/goodness_of_fit.csv", index=False)

print("\nInterval counts:")
print(interval_table.round(3))

print("\nGoodness-of-fit:")
print(gof_table.round(3))

# Plot settings 

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25
})

aircraft_ids = list(data.keys())

# Figure 1: Failure-time rug plot 

fig, ax = plt.subplots(figsize=(8, 4.8))

for i, aircraft in enumerate(aircraft_ids):
    ax.scatter(
        data[aircraft],
        np.repeat(i, len(data[aircraft])),
        marker="|",
        s=160,
        linewidths=2
    )

ax.set_yticks(range(len(aircraft_ids)))
ax.set_yticklabels([str(a) for a in aircraft_ids])
ax.invert_yaxis()
ax.set_xlim(0, T)
ax.set_xlabel("Failure time (days)")
ax.set_ylabel("Aircraft ID")
ax.set_title("Failure-time rug plot by aircraft")

fig.tight_layout()
plt.savefig("figures/failure_timeline.png", dpi=300)
plt.close()

# Figure 2: Inter-failure boxplot 

fig, ax = plt.subplots(figsize=(8, 4.8))

box_values = [
    inter_failures[inter_failures["aircraft"] == aircraft]["inter_failure_time"].to_numpy()
    for aircraft in aircraft_ids
]

ax.boxplot(
    box_values,
    labels=[str(a) for a in aircraft_ids],
    showmeans=True
)

ax.set_xlabel("Aircraft ID")
ax.set_ylabel("Inter-failure time (days)")
ax.set_title("Inter-failure time distributions")
plt.xticks(rotation=45)

fig.tight_layout()
plt.savefig("figures/inter_failure_boxplot.png", dpi=300)
plt.close()

# Figure 3: MCRF with fitted models 

fig, ax = plt.subplots(figsize=(8, 4.8))

t_grid = np.linspace(0, T, 300)

ax.step(unique_times, mcrf_values, where="post", label="Observed MCRF")
ax.plot(t_grid, lambda_hpp * t_grid, linestyle="--", label=f"HPP fit: M(t)={lambda_hpp:.4f}t")
ax.plot(t_grid, alpha_nhpp * t_grid ** beta_nhpp, linestyle=":", label=f"PLP NHPP: beta={beta_nhpp:.3f}")

ax.set_xlim(0, T)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Mean cumulative failures per aircraft")
ax.set_title("Mean cumulative repair function")
ax.legend()

fig.tight_layout()
plt.savefig("figures/mcrf_plot.png", dpi=300)
plt.close()

# Figure 4: 100-day interval counts 

fig, ax = plt.subplots(figsize=(8, 4.8))

centers = (edges[:-1] + edges[1:]) / 2

ax.bar(centers, observed_counts, width=70, alpha=0.75, label="Observed counts")
ax.plot(centers, expected_hpp, marker="o", linestyle="--", label="HPP expected")
ax.plot(centers, expected_nhpp, marker="s", linestyle=":", label="PLP expected")

ax.set_xlabel("Time interval (days)")
ax.set_ylabel("Number of failures")
ax.set_title("Failures per 100-day interval")
ax.set_xticks(centers)
ax.set_xticklabels([f"{int(edges[i])}-{int(edges[i + 1])}" for i in range(10)], rotation=45)
ax.legend()

fig.tight_layout()
plt.savefig("figures/interval_counts.png", dpi=300)
plt.close()

# Figure 5: Cumulative pooled failures 

fig, ax = plt.subplots(figsize=(8, 4.8))

pooled_cumulative = np.arange(1, r + 1)

ax.step(all_times, pooled_cumulative, where="post", label="Observed pooled count")
ax.plot(t_grid, n_units * lambda_hpp * t_grid, linestyle="--", label="HPP cumulative mean")
ax.plot(t_grid, n_units * alpha_nhpp * t_grid ** beta_nhpp, linestyle=":", label="PLP cumulative mean")

ax.set_xlim(0, T)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Cumulative failures across all aircraft")
ax.set_title("Observed cumulative count and fitted mean functions")
ax.legend()

fig.tight_layout()
plt.savefig("figures/cumulative_fit.png", dpi=300)
plt.close()

# Figure 6: Time-rescaling diagnostic 

fig, ax = plt.subplots(figsize=(7, 4.8))

empirical_quantiles = (np.arange(1, r + 1) - 0.5) / r

hpp_transformed = np.sort(all_times / T)
nhpp_transformed = np.sort((all_times / T) ** beta_nhpp)

ax.plot(empirical_quantiles, hpp_transformed, marker="o", linestyle="", markersize=3, label="HPP transform")
ax.plot(empirical_quantiles, nhpp_transformed, marker="s", linestyle="", markersize=3, label="PLP transform")
ax.plot([0, 1], [0, 1], linestyle="--", label="Uniform reference")

ax.set_xlabel("Theoretical uniform quantile")
ax.set_ylabel("Transformed event-time quantile")
ax.set_title("Time-rescaling diagnostic")
ax.legend()

fig.tight_layout()
plt.savefig("figures/time_rescaling.png", dpi=300)
plt.close()

print("\nDone.") 
print("CSV files saved in: results/")
print("Figures saved in: figures/")
