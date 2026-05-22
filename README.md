# Aircraft AC Reliability Analysis

Reliability analysis of repairable aircraft air-conditioning systems using recurrent failure data.

This project was completed for **ECE313 Probability with Engineering Applications** at the **Zhejiang University – University of Illinois Urbana-Champaign Institute (ZJU-UIUC Institute)**.

The dataset contains recurrent failure records from 11 aircraft observed over a 1000-day operational period. Since the air-conditioning units are repaired and returned to service after each failure, the systems are treated as **repairable systems** rather than single-lifetime components.

The project includes:

- **Inter-failure time analysis** to examine the time intervals between consecutive failures
- **Mean Cumulative Repair Function (MCRF)** analysis to study how failures accumulate over time
- **Homogeneous Poisson Process (HPP)** modeling assuming a constant failure intensity
- **Non-Homogeneous Poisson Process (NHPP)** modeling using the Power Law Process to evaluate possible aging effects
- **Goodness-of-fit evaluation** using chi-square and Kolmogorov–Smirnov tests
- **Fleet-level reliability interpretation** and discussion of aircraft heterogeneity

The results showed that the MCRF increased approximately linearly and the estimated NHPP parameter β was close to 1, indicating that the pooled fleet behavior was reasonably consistent with an HPP model.

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- SciPy
