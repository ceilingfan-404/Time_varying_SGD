import numpy as np

d = 1
sigma0 = 1.0

# Unit direction in which the target moves
target_direction = np.full(d, 1.0 / np.sqrt(d))


def theta_star(t, tau):
    """
    Target at integer time t.

    The target moves Euclidean distance tau per iteration:
        ||theta_star_tau(t, tau)
          - theta_star_tau(t - 1, tau)|| = tau.
    """
    return float(t) * tau * target_direction


def run_sgd(
    tau,
    gamma,
    R,
    rho=0.0,
    seed=1,
    steps=None,
    forgetting_constant=20.0,
):
    """
    Run R independent SGD trajectories with per-step target drift tau.

    Parameters
    ----------
    tau : float
        Euclidean movement of the target per iteration.
    gamma : float
        Constant learning rate.
    R : int
        Number of Monte Carlo replications.
    rho : float
        Initial distance from theta_star_tau(0, tau).
    seed : int
        Random seed.
    steps : int or None
        Number of iterations. If None, choose enough iterations to
        make the initial-condition weight approximately exp(-20).
    forgetting_constant : float
        Number of effective memory lengths used when steps is None.
    """
    if tau < 0:
        raise ValueError("tau must be nonnegative.")

    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    if R <= 0:
        raise ValueError("R must be positive.")

    # SGD remembers roughly 1/gamma observations.
    # With steps = 20/gamma, the initial weight is roughly exp(-20).
    if steps is None:
        steps = int(np.ceil(forgetting_constant / gamma))

    rng = np.random.RandomState(seed)

    # Initial estimate:
    # theta_0 = rho * target_direction
    Theta = np.tile(
        (rho * target_direction)[:, None],
        (1, R),
    )

    for t in range(1, steps + 1):

        # Current target; no t/n normalization
        mu = theta_star(t, tau)

        # Independent N(0, sigma0^2) observation errors
        z = rng.normal(size=d * R).reshape(
            (d, R),
            order="F",
        )

        eps = sigma0 * z

        # Current observations for all R replications
        X = mu[:, None] + eps

        # SGD update
        Theta = Theta - gamma * (Theta - X)

    final_target = theta_star(steps, tau)[:, None]
    errors = Theta - final_target

    # Bias in each coordinate
    bias_vector = np.mean(errors, axis=1)

    # One scalar: squared Euclidean norm of the bias
    bias_squared = float(
        np.sum(bias_vector**2)
    )

    # Coordinatewise variances
    variance_by_coordinate = np.var(
        errors,
        axis=1,
        ddof=0,
    )

    # One scalar: sum of coordinate variances
    total_variance = float(
        np.sum(variance_by_coordinate)
    )

    # E ||error||^2
    mse = float(
        np.mean(
            np.sum(errors**2, axis=0)
        )
    )

    return {
        "Tau": tau,
        "Gamma": gamma,
        "Steps": steps,
        "Final target": final_target[:, 0],
        "Bias vector": bias_vector,
        "Lag vector": -bias_vector,
        "Bias squared": bias_squared,
        "Variance by coordinate": variance_by_coordinate,
        "Total variance": total_variance,
        "MSE": mse,
        "Theta": Theta,
        "Errors": errors,
    }


def run_sgd_tau(
    tau,
    gamma,
    R,
    rho=0.0,
    seed=1,
    steps=None,
    forgetting_constant=20.0,
    delta=1.345,
    noise="t5",
    df=5,
):
    """
    Stabilized Huber-loss SGD from the attached handout.

    Loss:
        q_{delta, lambda}(z)
            = h_delta(z) + (lambda / 2) * z**2

    Score:
        phi_{delta, lambda}(z)
            = psi_delta(z) + lambda * z

    where:
        delta = 1.345
        lambda = 0.1

    The default observation noise is:
        epsilon_t = sigma0 * Z_t,
        Z_t ~ Student-t with 5 degrees of freedom.

    Parameters
    ----------
    tau : float
        Euclidean target movement per iteration.
    gamma : float
        Constant learning rate.
    R : int
        Number of independent Monte Carlo replications.
    rho : float
        Initial distance from theta_star(0, tau).
    seed : int
        Random seed.
    steps : int or None
        Number of SGD iterations. If None, enough iterations are
        chosen to approximately forget the initial value.
    forgetting_constant : float
        Used only if steps is None.
    delta : float
        Huber clipping threshold. The handout uses 1.345.
    noise : str
        "t5" or "student_t" for Student-t noise.
        "normal" remains available for comparison.
    df : int
        Degrees of freedom for Student-t noise. The handout uses 5.
    """
    if tau < 0:
        raise ValueError("tau must be nonnegative.")

    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    if R <= 0:
        raise ValueError("R must be positive.")

    if delta <= 0:
        raise ValueError("delta must be positive.")

    if steps is None:
        steps = int(np.ceil(forgetting_constant / gamma))

    steps = int(steps)

    # Quadratic stabilization constant from the handout
    lam = 0.1

    rng = np.random.RandomState(seed)

    # Initial estimate:
    # theta_0 = rho * target_direction
    #
    # Shape: (d, R), with one replication in each column.
    Theta = np.tile(
        (rho * target_direction)[:, None],
        (1, R),
    )

    for t in range(1, steps + 1):

        # Current target
        mu = theta_star(t, tau)

        # ---------------------------------------------------------
        # Noise from the handout: raw sigma0 * t_5
        # ---------------------------------------------------------
        if noise in ("t5", "student_t"):

            z = rng.standard_t(
                df=df,
                size=d * R,
            ).reshape(
                (d, R),
                order="F",
            )

        elif noise == "normal":

            z = rng.normal(
                size=d * R,
            ).reshape(
                (d, R),
                order="F",
            )

        else:
            raise ValueError(
                'noise must be "t5", "student_t", or "normal".'
            )

        # No division by sqrt(5/3), and no sqrt(1 + u) factor:
        # this is raw sigma0 * t_5 noise.
        eps = sigma0 * z

        # Current observations for all R replications
        X = mu[:, None] + eps

        # Residual entering the loss
        residual = Theta - X

        # Pure Huber score:
        #
        # psi_delta(z) = z                  when |z| <= delta
        #              = delta * sign(z)   when |z| > delta
        huber_score = np.clip(
            residual,
            -delta,
            delta,
        )

        # Stabilized Huber score from the handout:
        #
        # phi_{delta,lambda}(z)
        #     = psi_delta(z) + lambda*z
        stabilized_huber_score = (
            huber_score
            + lam * residual
        )

        # Stabilized Huber SGD update
        Theta = (
            Theta
            - gamma * stabilized_huber_score
        )

    final_target = theta_star(
        steps,
        tau,
    )[:, None]

    errors = Theta - final_target

    # Bias in each coordinate
    bias_vector = np.mean(
        errors,
        axis=1,
    )

    # Squared Euclidean norm of the bias vector
    bias_squared = float(
        np.sum(bias_vector**2)
    )

    # Variance in each coordinate
    variance_by_coordinate = np.var(
        errors,
        axis=1,
        ddof=0,
    )

    # Sum of coordinate variances
    total_variance = float(
        np.sum(variance_by_coordinate)
    )

    # Monte Carlo estimate of E ||error||^2
    mse = float(
        np.mean(
            np.sum(errors**2, axis=0)
        )
    )

    return {
        "Tau": tau,
        "Gamma": gamma,
        "Steps": steps,
        "Delta": delta,
        "Noise": noise,
        "Final target": final_target[:, 0],
        "Bias vector": bias_vector,
        "Lag vector": -bias_vector,
        "Bias squared": bias_squared,
        "Variance by coordinate": variance_by_coordinate,
        "Total variance": total_variance,
        "MSE": mse,
        "Theta": Theta,
        "Errors": errors,
    }

from scipy.integrate import quad
from scipy.stats import t as student_t


def Gamma_fun_huber(
    u,
    delta=1.345,
    df=3,
):
    """
    Local small-gamma covariance coefficient for coordinatewise
    Huber SGD with scaled Student-t noise.

    Var(epsilon(u)) = (1 + u) * sigma0^2.
    """
    scale = (
        sigma0
        * np.sqrt(
            (1.0 + u) * (df - 2.0) / df
        )
    )

    cutoff = delta / scale

    # A(u) = P(|epsilon(u)| <= delta)
    A = (
        2.0
        * student_t.cdf(cutoff, df=df)
        - 1.0
    )

    # E[T^2 1{|T| <= cutoff}]
    truncated_second_moment = 2.0 * quad(
        lambda x: x**2 * student_t.pdf(x, df=df),
        0.0,
        cutoff,
    )[0]

    # B(u) = E[min(epsilon(u)^2, delta^2)]
    B = (
        scale**2 * truncated_second_moment
        + delta**2 * (1.0 - A)
    )

    return B / (2.0 * A)
