import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dgp_L import run_sgd, d, sigma0


SMOKE = False

# Per-step target drift values
tau = 1e-5

R_E1 = 1000 if SMOKE else 2000
ngam_E1 = 15 if SMOKE else 31

ns_E1 = np.array(
    [10_000, 100_000, 1_000_000],
    dtype=int,
)

# Because tau is fixed, the theoretical gamma scale is the same for all n
gamma_center = (
    4.0 * tau**2 / (d * sigma0**2)
) ** (1.0 / 3.0)

gammas = np.geomspace(
    0.15 * gamma_center,
    6.0 * gamma_center,
    ngam_E1,
)

gammas = gammas[
    (gammas > 0.0) & (gammas < 1.0)
]


e1_records = []

# n is always the number of SGD iterations
for n_index, n in enumerate(ns_E1):

    for gamma_index, gamma in enumerate(gammas):

        seed = (
            1000
            + 100 * n_index
            + gamma_index
        )

        print(
            f"Running n={n:,}, "
            f"gamma {gamma_index + 1}/{len(gammas)}, "
            f"gamma={gamma:.4e}"
        )

        output = run_sgd(
            tau=float(tau),
            gamma=float(gamma),
            R=R_E1,
            rho=0.0,
            seed=seed,
            steps=int(n),
        )

        # Check immediately that run_sgd actually used this n
        if int(output["Steps"]) != int(n):
            raise RuntimeError(
                f"run_sgd used {output['Steps']:,} steps, "
                f"but n={n:,} was requested."
            )

        e1_records.append(
            {
                "n": int(n),
                "tau": float(tau),
                "gamma": float(gamma),
                "steps": int(output["Steps"]),
                "bias2": float(output["Bias squared"]),
                "variance": float(output["Total variance"]),
                "mse": float(output["MSE"]),
            }
        )

    # Check immediately after completing each n
    rows_for_n = sum(
        record["n"] == int(n)
        for record in e1_records
    )

    if rows_for_n != len(gammas):
        raise RuntimeError(
            f"For n={n:,}, expected {len(gammas)} stored gamma values, "
            f"but found {rows_for_n}."
        )

    # Save completed work after every n
    pd.DataFrame(e1_records).to_csv(
        "e1_results_partial.csv",
        index=False,
    )

    print(
        f"E1 finished: n={n:,}, "
        f"tau={tau:.1e}, "
        f"stored gamma values={rows_for_n}"
    )


e1_results = pd.DataFrame(e1_records)

print("\nNumber of gamma values for each n:")
print(
    e1_results
    .groupby("n")
    .size()
    .reindex(ns_E1, fill_value=0)
)

print("\nSimulation results:")
print(e1_results)

FIT_POINTS = 6


def fit_loglog_regression(data, y_column="mse"):
    """
    Fit

        log10(y) = intercept + slope * log10(gamma)

    and return the slope, intercept, and R-squared.
    """
    x = np.log10(data["gamma"].to_numpy(dtype=float))
    y = np.log10(data[y_column].to_numpy(dtype=float))

    slope, intercept = np.polyfit(x, y, deg=1)

    fitted_y = intercept + slope * x

    residual_sum_squares = np.sum((y - fitted_y) ** 2)
    total_sum_squares = np.sum((y - np.mean(y)) ** 2)

    if total_sum_squares > 0:
        r_squared = 1.0 - residual_sum_squares / total_sum_squares
    else:
        r_squared = np.nan

    return float(slope), float(intercept), float(r_squared)


diagnostic_records = []

plt.figure(figsize=(9, 6))

for n_value in ns_E1:

    subset = (
        e1_results[e1_results["n"] == n_value]
        .sort_values("gamma")
        .reset_index(drop=True)
    )

    if len(subset) < 2 * FIT_POINTS:
        raise ValueError(
            f"For n={n_value:,}, there are only {len(subset)} gamma values. "
            f"At least {2 * FIT_POINTS} are needed to fit "
            f"{FIT_POINTS} points on each side."
        )

    # Six smallest gamma values
    left_points = subset.iloc[:FIT_POINTS]

    # Six largest gamma values
    right_points = subset.iloc[-FIT_POINTS:]

    left_slope, left_intercept, left_r2 = fit_loglog_regression(
        left_points,
        y_column="mse",
    )

    right_slope, right_intercept, right_r2 = fit_loglog_regression(
        right_points,
        y_column="mse",
    )

    # Tested grid point with the smallest MSE
    best_index = subset["mse"].idxmin()
    best = subset.loc[best_index]

    diagnostic_records.append(
        {
            "n": int(n_value),
            "tau": float(best["tau"]),
            "left_slope": left_slope,
            "left_R2": left_r2,
            "right_slope": right_slope,
            "right_R2": right_r2,
            "best_gamma": float(best["gamma"]),
            "minimum_mse": float(best["mse"]),
            "bias2_at_minimum": float(best["bias2"]),
            "variance_at_minimum": float(best["variance"]),
        }
    )

    # Plot the complete simulated MSE curve
    line, = plt.loglog(
        subset["gamma"],
        subset["mse"],
        marker="o",
        linewidth=2,
        label=(
            f"n={n_value:,}: "
            f"left={left_slope:.2f}, "
            f"right={right_slope:.2f}"
        ),
    )

    curve_color = line.get_color()

    # Fitted line for the left side
    left_gamma_grid = np.geomspace(
        left_points["gamma"].min(),
        left_points["gamma"].max(),
        100,
    )

    left_mse_fit = (
        10.0 ** left_intercept
        * left_gamma_grid ** left_slope
    )

    plt.loglog(
        left_gamma_grid,
        left_mse_fit,
        linestyle="--",
        linewidth=2,
        color=curve_color,
    )

    # Fitted line for the right side
    right_gamma_grid = np.geomspace(
        right_points["gamma"].min(),
        right_points["gamma"].max(),
        100,
    )

    right_mse_fit = (
        10.0 ** right_intercept
        * right_gamma_grid ** right_slope
    )

    plt.loglog(
        right_gamma_grid,
        right_mse_fit,
        linestyle=":",
        linewidth=2,
        color=curve_color,
    )

    # Mark the tested minimum-MSE point
    plt.scatter(
        best["gamma"],
        best["mse"],
        marker="*",
        s=180,
        color=curve_color,
        zorder=5,
    )
    print(n_value)


plt.xlabel(r"$\gamma$")
plt.ylabel("MSE")
plt.title(
    "MSE versus Learning Rate\n"
    "Dashed: left fit; dotted: right fit; stars: minima"
)
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()


diagnostics = pd.DataFrame(diagnostic_records)

print("\nLog-log slopes and minimum-MSE points:\n")

print(
    diagnostics.to_string(
        index=False,
        formatters={
            "tau": lambda x: f"{x:.3e}",
            "left_slope": lambda x: f"{x:.3f}",
            "left_R2": lambda x: f"{x:.3f}",
            "right_slope": lambda x: f"{x:.3f}",
            "right_R2": lambda x: f"{x:.3f}",
            "best_gamma": lambda x: f"{x:.6e}",
            "minimum_mse": lambda x: f"{x:.6e}",
            "bias2_at_minimum": lambda x: f"{x:.6e}",
            "variance_at_minimum": lambda x: f"{x:.6e}",
        },
    )
)
