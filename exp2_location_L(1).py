import numpy as np
import pandas as pd
from scipy.stats import chi2

import dgp_L

# Equivalent of source("dgp_L.R")
run_sgd = dgp_L.run_sgd
theta_star = dgp_L.theta_star
Gamma_fun = dgp_L.Gamma_fun
d = dgp_L.d


# =====================================================================
# EXPERIMENT 2
# Localization of the limiting covariance at u = 1
# =====================================================================

SMOKE = True

n_E2 = 10_000 if SMOKE else 100_000
R_E2 = 2_000 if SMOKE else 5_000

# Inside the "good" range of learning rates
gamma_E2 = n_E2 ** (-1.0 / 2.0)


# Correct candidate: end-of-run covariance coefficient
Gam1 = float(Gamma_fun(1.0))

# Candidate: covariance coefficient averaged over the whole run
u_grid = np.linspace(0.0, 1.0, 101)

Gambar = float(
    np.mean(
        [Gamma_fun(u) for u in u_grid]
    )
)

print(
    f"  Gamma(1) = {Gam1:.3f}   "
    f"Gamma_bar = {Gambar:.3f}"
)


clocs = [Gam1, Gambar]
cov_names = ["Gamma(1)", "Gamma_bar"]


def extract_error_matrix(output):
    """
    Return the d x R matrix of final errors.

    Supports:
      1. run_sgd() returning {"Errors": errors, ...};
      2. run_sgd() returning {"Theta": Theta, ...};
      3. run_sgd() returning the error matrix directly.
    """
    if isinstance(output, dict):

        if "Errors" in output:
            errors = np.asarray(
                output["Errors"],
                dtype=float,
            )

        elif "Theta" in output:
            theta_final = np.asarray(
                output["Theta"],
                dtype=float,
            )

            final_target = theta_star(1.0)[:, None]
            errors = theta_final - final_target

        else:
            raise KeyError(
                'run_sgd() returned a dictionary, but it has neither '
                '"Errors" nor "Theta".'
            )

    else:
        errors = np.asarray(output, dtype=float)

    # For d=1, ensure shape is (1, R), not (R,)
    if errors.ndim == 1:
        errors = errors[None, :]

    if errors.shape[0] != d:
        raise ValueError(
            f"Expected errors with {d} rows, "
            f"but received shape {errors.shape}."
        )

    return errors


def coverage(
    errors,
    gamma,
    c_loc,
    levels=(0.90, 0.95, 0.99),
):
    """
    Empirical coverage of the nominal ellipsoid

        ||error||^2 / (gamma * c_loc)
            <= chi-square quantile with d degrees of freedom.

    Parameters
    ----------
    errors : ndarray, shape (d, R)
        One final error vector in each column.
    gamma : float
        Learning rate.
    c_loc : float
        Candidate covariance coefficient.
    levels : sequence of float
        Nominal coverage levels.

    Returns
    -------
    ndarray
        Empirical coverages corresponding to the requested levels.
    """
    errors = np.asarray(errors, dtype=float)

    if errors.ndim == 1:
        errors = errors[None, :]

    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    if c_loc <= 0:
        raise ValueError("c_loc must be positive.")

    levels = np.asarray(levels, dtype=float)

    # Squared Euclidean norm of every error column
    squared_norms = np.sum(errors**2, axis=0)

    # Standardized quadratic statistic, one value per replication
    statistic = squared_norms / (gamma * c_loc)

    # Chi-square critical values
    critical_values = chi2.ppf(levels, df=d)

    # Compare every statistic to every critical value
    empirical_coverages = np.mean(
        statistic[None, :]
        <= critical_values[:, None],
        axis=1,
    )

    return empirical_coverages


# One row for every (rho, covariance candidate) combination
e2_records = []

for rho in [0.0, 5.0]:

    output = run_sgd(
        n=n_E2,
        gamma=gamma_E2,
        R=R_E2,
        rho=rho,
        seed=int(200 + rho),
    )

    errors = extract_error_matrix(output)

    for candidate_name, candidate_value in zip(
        cov_names,
        clocs,
    ):
        empirical_coverage = coverage(
            errors=errors,
            gamma=gamma_E2,
            c_loc=candidate_value,
        )

        e2_records.append(
            {
                "rho": rho,
                "candidate": candidate_name,
                "cov90": empirical_coverage[0],
                "cov95": empirical_coverage[1],
                "cov99": empirical_coverage[2],
            }
        )


e2_table = pd.DataFrame(e2_records)

# Equivalent to:
# write.csv(e2_table, "e2_coverage.csv", row.names=FALSE)
#
# Uncomment to save:
# e2_table.to_csv("e2_coverage.csv", index=False)

print()
print(
    e2_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)
