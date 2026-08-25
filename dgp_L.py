import numpy as np

# Equivalent of:
# rm(list = ls())
# set.seed(2026)

np.random.seed(2026)

d = 1               # dimension
sigma0 = 1.0        # base noise scale
L_theta = 2.0       # |theta*(1) - theta*(0)|, linear path


def theta_star(u):
    """
    Return the d-dimensional path value theta*(u).
    """
    return np.full(d, L_theta / np.sqrt(d)) * u


def Gamma_fun(u):
    """
    Scalar multiplier in Gamma(u) = Gamma_fun(u) * I_d.
    """
    return (1.0 + u) * sigma0**2 / 2.0


# ---------------------------------------------------------------------
# Returns a d x R NumPy array of final errors:
# theta_n - theta*(1)
# ---------------------------------------------------------------------
def run_sgd(n, gamma, R, rho=0.0, seed=1):
    rng = np.random.RandomState(seed)

    # Unit vector specifying the direction of the initial offset
    v0 = np.full(d, 1.0 / np.sqrt(d))

    # theta_0 = theta*(0) + rho*v0 = rho*v0
    # Each of the R columns corresponds to one replication.
    Theta = np.tile((rho * v0)[:, None], (1, R))

    for t in range(1, n + 1):
        u = t / n

        # Shape: (d,)
        mu = theta_star(u)

        # Generate d*R independent N(0,1) random variables.
        # order="F" mimics R's column-wise filling of a matrix.
        z = rng.normal(size=d * R).reshape((d, R), order="F")

        # Shape: (d, R)
        #eps = np.sqrt(1.0 + u) * sigma0 * z
        eps = sigma0 * z

        # mu[:, None] gives mu shape (d,1), so it is added to
        # every one of the R simulation columns.
        X = mu[:, None] + eps

        Theta = Theta - gamma * (Theta - X)

    # NumPy requires an explicit column dimension for broadcasting.
    return {"MSE": np.mean((Theta - theta_star(1.0)[:, None])**2),
            "Theta": Theta,
    }


print(
    "Setup complete. d =",
    d,
    " L_theta =",
    L_theta,
    " sigma0 =",
    sigma0,
)
print(run_sgd(n=1000, gamma=0.05, R=100, rho=1.0)["MSE"])
