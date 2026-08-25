import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Python equivalent of:
# source("dgp_L.R")
#
# If run_sgd() and theta_star() are saved in dgp_L.py, use:
# from dgp_L import run_sgd, theta_star
#
# If they were already defined earlier in the notebook/script,
# no import is needed.


SMOKE = True

# TRUE = quick check; FALSE = full run
ns_E1 = (
    np.array([10_000, 100_000], dtype=int)
    if SMOKE
    else np.array([10_000, 100_000, 1_000_000], dtype=int)
)

R_E1 = 500 if SMOKE else 2000
ngam_E1 = 8 if SMOKE else 15


def extract_error_matrix(output):
    """
    Convert the output of run_sgd() into the d x R final-error matrix.

    Handles either:
      1. run_sgd() returning the error matrix directly; or
      2. run_sgd() returning a dictionary containing "Theta".
    """
    if isinstance(output, dict):
        if "Theta" not in output:
            raise KeyError(
                'run_sgd() returned a dictionary without a "Theta" entry.'
            )

        theta_final = np.asarray(output["Theta"], dtype=float)
        final_target = theta_star(1.0)[:, None]

        return theta_final - final_target

    error = np.asarray(output, dtype=float)

    # Ensure d=1 output is represented as a 1 x R matrix.
    if error.ndim == 1:
        error = error[None, :]

    return error


# Storage: one row per (n, gamma) combination
e1_records = []

for n_value in ns_E1:

    # Equivalent to:
    # exp(seq(log(n^-1), log(n^(-1/3)), length.out=ngam_E1))
    gammas = np.exp(
        np.linspace(
            np.log(n_value**(-1.0)),
            np.log(n_value**(-1.0 / 3.0)),
            num=ngam_E1,
        )
    )

    for gamma in gammas:

        # Equivalent to:
        # 100 + n + round(1e6 * g)
        seed = int(
            100
            + n_value
            + np.rint(1_000_000 * gamma)
        )

        output = run_sgd(
            n=int(n_value),
            gamma=float(gamma),
            R=R_E1,
            rho=0.0,
            seed=seed,
        )

        error = extract_error_matrix(output)

        # R equivalent:
        # mean(colSums(err^2))
        #
        # error**2                  has shape d x R
        # np.sum(..., axis=0)       sums over the d coordinates
        # np.mean(...)              averages over R replications
        mse = float(
            np.mean(
                np.sum(error**2, axis=0)
            )
        )

        e1_records.append(
            {
                "n": int(n_value),
                "gamma": float(gamma),
                "mse": mse,
            }
        )

    print(f" E1 finished n = {n_value}")


e1_results = pd.DataFrame(e1_records)

# Equivalent to:
# write.csv(e1_results, "e1_results.csv", row.names=FALSE)
#
# Uncomment to save:
# e1_results.to_csv("e1_results.csv", index=False)

print()
print(e1_results)
