import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# n is number of iterations, R is number of replications
n=5000
R=200
def central_limit(x):
    n = len(x)
    mean = np.mean(x)
    s = np.std(x, ddof=1)

    z = 1.96

    lower = mean - z * s / np.sqrt(n)
    upper = mean + z * s / np.sqrt(n)
    return [float(mean), (float(lower), float(upper))]
CIs = list()
taus = list(np.linspace(0.0005, 0.002, 5))
#theta_star = np.tile(np.array([[1], [1], [2]]), (1, R))
rng = np.random.default_rng(seed=1)
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
ax2[0].set_title(r"(a) First coordinate")
ax2[1].set_title(r"(b) Second coordinate")
ax2[2].set_title(r"(c) Third coordinate")
colors = ['#440154', '#3B528B', '#21918C', '#5EC962', '#FDE725']
color_labels = [r'$\tau=0.0005$', r'$\tau=0.000875$', r'$\tau=0.00125$', r'$\tau=0.001625$', r'$\tau=0.002$']
style_labels = ['Target Path', 'SGD Estimate']
MSEs = list()
for tau in taus:
    fullMSEs = list()
    theta = np.array([[0], [0], [0]])
    a_i = [0]
    b_i = [0]
    c_i = [0]
    a_star = [1 / np.sqrt(2)]
    b_star = [0]
    c_star = [0]
    gamma = tau**(2/3)
    for i in range(n):
        theta_star = np.tile(np.array([[np.cos((i+1)*tau)], [np.sin((i+1)*tau)], [(i+1)*tau]]), (1, R))/(np.sqrt(2))
        sensor = rng.uniform(low=-50.0, high=50.0, size = (3, R))
        eps = rng.normal(loc=0.0, scale=1.0, size = (1, R))*0.2
        real_d = (((sensor-theta_star)**2).sum(axis=0))**0.5
        pseudo_d = (((sensor-theta)**2).sum(axis=0))**0.5
        d = real_d+eps
        loss = (sensor - theta)*(-2*(pseudo_d-d)*((pseudo_d)**(-1.0)))
        a_star.append(theta_star[0][0])
        b_star.append(theta_star[1][0])
        c_star.append(theta_star[2][0])
        theta = theta - gamma * loss
        a_i.append(theta[0][0])
        b_i.append(theta[1][0])
        c_i.append(theta[2][0])
    fullMSEs.append(np.mean((theta - theta_star)**2, axis=0))
    fullMSEs=np.array(fullMSEs)
    MSEs.append(np.mean(np.mean((theta - theta_star)**2, axis=0)))
    CIs.append(central_limit(fullMSEs)[1])
    a_i = np.array(a_i)
    b_i = np.array(b_i)
    c_i = np.array(c_i)
    a_star = np.array(a_star)
    b_star = np.array(b_star)
    c_star = np.array(c_star)
    ax2[0].plot(a_i, color=colors[taus.index(tau)])
    ax2[0].plot(a_star, linestyle='--', color=colors[taus.index(tau)])
    ax2[1].plot(b_i, color=colors[taus.index(tau)])
    ax2[1].plot(b_star, linestyle='--', color=colors[taus.index(tau)])
    ax2[2].plot(c_i, color=colors[taus.index(tau)])
    ax2[2].plot(c_star, linestyle='--', color=colors[taus.index(tau)])
for ax in ax2:
    ax.set_xlabel(r"Iteration $i$")
ax2[0].set_ylabel("Coordinate value")
color_handles = [
    Line2D([0], [0], color=c, lw=2)
    for c in colors
]

style_handles = [
    Line2D([0], [0], color="black", lw=2, ls="-"),
    Line2D([0], [0], color="black", lw=2, ls="--")
]
all_handles = color_handles + style_handles
all_labels = color_labels + ["SGD Estimate", "Target Path"]

fig2.legend(
    all_handles,
    all_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=7,
    frameon=False
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('OSLcoordsplot.pdf', format='pdf', bbox_inches='tight')
plt.show()
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.scatter(taus, MSEs)
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.title("Log-Log Plot of Mean Squared Error vs Tau")
plt.xlabel("Tau")
plt.ylabel("Mean Squared Error")
plt.savefig('loglogOSLplot.pdf', format='pdf', bbox_inches='tight')
plt.show()
slope, intercept = np.polyfit(np.log10(taus), np.log10(MSEs), 1)

print(CIs)
print(f"Best fit slope (m): {slope:.4f}")
print(f"Y-intercept (b): {intercept:.4f}")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# --------------------------------------------------
# Suppose these are already computed in your simulation
# theta_path      : shape (n, 3)
# theta_star_path : shape (n, 3)
# residuals       : shape (n,)
# gamma           : scalar
# --------------------------------------------------

alpha = 0.05
m = 100                    # rolling window size for sigma^2 estimate
chi2_quant = chi2.ppf(1 - alpha, df=3)

n = theta.shape[0]

# Plug-in estimator of sigma^2_t
sigma2_hat = np.full(n, np.nan)
for t in range(m - 1, n):
    sigma2_hat[t] = np.mean(residuals[t - m + 1:t + 1] ** 2)

# Confidence-ball radius
radius = np.sqrt(gamma * sigma2_hat * chi2_quant)

# --------------------------------------------------
# Helper function to draw a sphere
# --------------------------------------------------
def plot_sphere(ax, center, r, color='cornflowerblue', alpha=0.12, ngrid=20):
    u = np.linspace(0, 2 * np.pi, ngrid)
    v = np.linspace(0, np.pi, ngrid)

    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)

# --------------------------------------------------
# 3D plot
# --------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Estimated trajectory
ax.plot(theta[:, 0], theta[:, 1], theta[:, 2],
        color='tab:blue', lw=2, label='SGD estimate')

# True trajectory
ax.plot(theta_star[:, 0], theta_star[:, 1], theta_star[:, 2],
        color='tab:red', lw=2, ls='--', label='True target')

# Draw confidence balls only every "step" points to avoid clutter
step = 200
for t in range(max(m - 1, 500), n, step):   # since you said t >= 500
    if np.isfinite(radius[t]):
        plot_sphere(ax, theta[t], radius[t], color='tab:blue', alpha=0.10)

# Mark a few centers if desired
sample_idx = range(max(m - 1, 500), n, step)
ax.scatter(theta[list(sample_idx), 0],
           theta[list(sample_idx), 1],
           theta[list(sample_idx), 2],
           color='tab:blue', s=12)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D trajectory with joint 95% confidence balls')
ax.legend()
plt.tight_layout()
plt.show()

