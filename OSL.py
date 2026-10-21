import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, chi2
from pathlib import Path
# n is number of iterations, R is number of replications
n=5000
R=500
m=100
alpha = 0.05
z_crit = norm.ppf(1 - alpha / 2)
chi2_crit = chi2.ppf(1 - alpha, df=3)
a_top = list()
b_top = list()
c_top = list()
a_bottom = list()
b_bottom = list()
c_bottom = list()
CIs = list()
taus = list(np.linspace(0.0005, 0.002, 5)) #3 different 3d graphs for each tau
theta_star = np.tile(np.array([[1], [1], [2]]), (1, R))
rng = np.random.default_rng(seed=1)
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4))
ax2[0].set_title(r"(a) First coordinate")
ax2[1].set_title(r"(b) Second coordinate")
ax2[2].set_title(r"(c) Third coordinate")
colors = ['#440154', '#3B528B', '#21918C', '#5EC962', '#FDE725']
#colors = ['#440154', '#21918C', "#066F0A"]
color_labels = [r'$\tau=0.0005$', r'$\tau=0.000875$', r'$\tau=0.00125$', r'$\tau=0.001625$', r'$\tau=0.002$']
#color_labels = [r'$\tau=0.0005$', r'$\tau=0.00125$', r'$\tau=0.002$']
style_labels = ['Target Path', 'SGD Estimate']
MSEs = list()
result_rows = list()
for tau in taus:
    residuals = list()
    fullMSEs = list()
    theta = np.array([[0], [0], [0]])
    a_i = [[0]*R]
    b_i = [[0]*R]
    c_i = [[0]*R]
    a_star = [1 / np.sqrt(2)]
    b_star = [0]
    c_star = [0]
    gamma = tau**(2/3) #change to 2/3 later
    for i in range(n):
        theta_star = np.tile(np.array([[np.cos((i+1)*tau)], [np.sin((i+1)*tau)], [(i+1)*tau]]), (1, R))/(np.sqrt(2))
        sensor = rng.uniform(low=-50.0, high=50.0, size=(3,R))
        eps = rng.normal(loc=0.0, scale=1.0, size = (1, R))*0.2
        real_d = (((sensor-theta_star)**2).sum(axis=0))**0.5
        pseudo_d = (((sensor-theta)**2).sum(axis=0))**0.5
        d = real_d+eps
        residuals.append(d-pseudo_d)
        loss = (sensor - theta)*(-2*(pseudo_d-d)*((pseudo_d)**(-1.0)))
        a_star.append(theta_star[0][0])
        b_star.append(theta_star[1][0])
        c_star.append(theta_star[2][0])
        theta = theta - gamma * loss
        a_i.append(theta[0])
        b_i.append(theta[1])
        c_i.append(theta[2])
    fullMSEs.append(np.mean((theta - theta_star)**2, axis=0))
    fullMSEs=np.array(fullMSEs)
    MSEs.append(np.mean(np.mean((theta - theta_star)**2, axis=0)))
    a_i = np.vstack(a_i)
    b_i = np.vstack(b_i)
    c_i = np.vstack(c_i)
    a_star = np.array(a_star)
    b_star = np.array(b_star)
    c_star = np.array(c_star)
    ax2[0].plot(a_i[:,0], color=colors[taus.index(tau)])
    ax2[0].plot(a_star, linestyle='--', color=colors[taus.index(tau)])
    ax2[1].plot(b_i[:,0], color=colors[taus.index(tau)])
    ax2[1].plot(b_star, linestyle='--', color=colors[taus.index(tau)])
    ax2[2].plot(c_i[:,0], color=colors[taus.index(tau)])
    ax2[2].plot(c_star, linestyle='--', color=colors[taus.index(tau)])
for tau in taus:
    residuals = list()
    fullMSEs = list()
    theta = np.array([[0], [0], [0]])
    a_i = [[0]*R]
    b_i = [[0]*R]
    c_i = [[0]*R]
    a_star = [1 / np.sqrt(2)]
    b_star = [0]
    c_star = [0]
    a_top = list()
    b_top = list()
    c_top = list()
    a_bottom = list()
    b_bottom = list()
    c_bottom = list()
    gamma = 10*tau**(2/3) #change to 2/3 later
    sum=0
    sigmas = list()
    for i in range(n):
        theta_star = np.tile(np.array([[np.cos((i+1)*tau)], [np.sin((i+1)*tau)], [(i+1)*tau]]), (1, R))/(np.sqrt(2))
        sensor = rng.uniform(low=-50.0, high=50.0, size=(3,R))
        eps = rng.normal(loc=0.0, scale=1.0, size = (1, R))*0.2
        real_d = (((sensor-theta_star)**2).sum(axis=0))**0.5
        pseudo_d = (((sensor-theta)**2).sum(axis=0))**0.5
        d = real_d+eps
        residuals.append(d-pseudo_d)
        loss = (sensor - theta)*(-2*(pseudo_d-d)*((pseudo_d)**(-1.0)))
        a_star.append(theta_star[0][0])
        b_star.append(theta_star[1][0])
        c_star.append(theta_star[2][0])
        theta = theta - gamma * loss
        a_i.append(theta[0])
        b_i.append(theta[1])
        c_i.append(theta[2])
        if i+1 >= 500:
            sum = 0
            for j in range(i-m+1, i+1):
                sum += residuals[j]**2
            sigma_hat = ((1/m)*(sum))**0.5
            sigmas.append(sigma_hat)
            a_top.append(a_i[i+1] + z_crit*sigma_hat[0,:]*(gamma**0.5))
            b_top.append(b_i[i+1] + z_crit*sigma_hat[0,:]*(gamma**0.5))
            c_top.append(c_i[i+1] + z_crit*sigma_hat[0,:]*(gamma**0.5))
            a_bottom.append(a_i[i+1] - z_crit*sigma_hat[0,:]*(gamma**0.5))
            b_bottom.append(b_i[i+1] - z_crit*sigma_hat[0,:]*(gamma**0.5))
            c_bottom.append(c_i[i+1] - z_crit*sigma_hat[0,:]*(gamma**0.5))
    sigmas = np.vstack(sigmas)
    fullMSEs=np.array(fullMSEs)
    a_i = np.vstack(a_i)
    b_i = np.vstack(b_i)
    c_i = np.vstack(c_i)
    a_star = np.array(a_star)
    b_star = np.array(b_star)
    c_star = np.array(c_star)
    a_top = np.vstack(a_top)
    b_top = np.vstack(b_top)
    c_top = np.vstack(c_top)
    a_bottom = np.vstack(a_bottom)
    b_bottom = np.vstack(b_bottom)
    c_bottom = np.vstack(c_bottom)
    condition_a = (a_bottom[-1] <= np.full(R,a_star[-1])) & (np.full(R,a_star[-1]) <= a_top[-1])
    condition_b = (b_bottom[-1] <= np.full(R,b_star[-1])) & (np.full(R,b_star[-1]) <= b_top[-1])
    condition_c = (c_bottom[-1] <= np.full(R,c_star[-1])) & (np.full(R,c_star[-1]) <= c_top[-1])
    print("Tau: " + str(tau))
    print("a: " + str(np.mean(condition_a)*100) + "%")
    print("b: " + str(np.mean(condition_b)*100)+ "%")
    print("c: " + str(np.mean(condition_c)*100)+ "%")
    ax2[0].fill_between(np.linspace(500, 5000, 4501), a_top[:,0], a_bottom[:,0], color=colors[taus.index(tau)], alpha=0.6)
    ax2[1].fill_between(np.linspace(500, 5000, 4501), b_top[:,0], b_bottom[:,0], color=colors[taus.index(tau)], alpha=0.6)
    ax2[2].fill_between(np.linspace(500, 5000, 4501), c_top[:,0], c_bottom[:,0], color=colors[taus.index(tau)], alpha=0.6)
    sigmas = np.array(sigmas)
    times = np.arange(500, n + 1, 500)
    rep = 0
    times = np.arange(500, n + 1, 500)
    rep = 0
    radii = sigmas[times - 500, rep] * np.sqrt(gamma * chi2_crit)
    distances = (a_i[-1]-a_star[-1])**2 + (b_i[-1]-b_star[-1])**2+ (c_i[-1]-c_star[-1])**2
    percentage = np.mean(distances <= gamma * sigmas[-1, :]**2 * chi2_crit)*100
    print(percentage)
    result_rows.append([tau,MSEs[taus.index(tau)],100 * np.mean(condition_a),100 * np.mean(condition_b),100 * np.mean(condition_c),percentage])
residuals = np.array(residuals)
for ax in ax2:
    ax.set_xlabel(r"Iteration $i$")
ax2[0].set_ylabel("Coordinate value")
color_handles = [Line2D([0], [0], color=c, lw=2) for c in colors]
style_handles = [Line2D([0], [0], color="black", lw=2, ls="-"),Line2D([0], [0], color="black", lw=2, ls="--")]
all_handles = color_handles + style_handles
all_labels = color_labels + ["SGD Estimate", "Target Path"]
fig2.legend(all_handles,all_labels,loc="lower center",bbox_to_anchor=(0.5, 0.01),ncol=7,frameon=False)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('OSLcoordsplot.pdf', format='pdf', bbox_inches='tight')
plt.show()
for tau in taus:
    residuals = list()
    fullMSEs = list()
    theta = np.array([[0], [0], [0]])
    a_i = [[0]*R]
    b_i = [[0]*R]
    c_i = [[0]*R]
    a_i_plot = [[0]*R]
    b_i_plot = [[0]*R]
    c_i_plot = [[0]*R]
    a_star = [1 / np.sqrt(2)]
    b_star = [0]
    c_star = [0]
    a_top = list()
    b_top = list()
    c_top = list()
    a_bottom = list()
    b_bottom = list()
    c_bottom = list()
    gamma = 10*tau**(2/3) #change to 2/3 later
    sum=0
    sigmas = list()
    for i in range(n):
        theta_star = np.tile(np.array([[np.cos((i+1)*tau)], [np.sin((i+1)*tau)], [(i+1)*tau]]), (1, R))/(np.sqrt(2))
        sensor = rng.uniform(low=-50.0, high=50.0, size=(3,R))
        eps = rng.normal(loc=0.0, scale=1.0, size = (1, R))*0.2
        real_d = (((sensor-theta_star)**2).sum(axis=0))**0.5
        pseudo_d = (((sensor-theta)**2).sum(axis=0))**0.5
        d = real_d+eps
        residuals.append(d-pseudo_d)
        loss = (sensor - theta)*(-2*(pseudo_d-d)*((pseudo_d)**(-1.0)))
        a_star.append(theta_star[0][0])
        b_star.append(theta_star[1][0])
        c_star.append(theta_star[2][0])
        theta = theta - gamma * loss
        a_i.append(theta[0])
        b_i.append(theta[1])
        c_i.append(theta[2])
        if i+1 >= 500:
            sum = 0
            for j in range(i-m+1, i+1):
                sum += residuals[j]**2
            sigma_hat = ((1/m)*(sum))**0.5
            sigmas.append(sigma_hat)
            a_top.append(a_i[i+1] + z_crit*sigma_hat[0,:]*(gamma**0.5))
            b_top.append(b_i[i+1] + z_crit*sigma_hat[0,:]*(gamma**0.5))
            c_top.append(c_i[i+1] + z_crit*sigma_hat[0,:]*(gamma**0.5))
            a_bottom.append(a_i[i+1] - z_crit*sigma_hat[0,:]*(gamma**0.5))
            b_bottom.append(b_i[i+1] - z_crit*sigma_hat[0,:]*(gamma**0.5))
            c_bottom.append(c_i[i+1] - z_crit*sigma_hat[0,:]*(gamma**0.5))
    gamma = tau**(2/3)
    theta = np.array([[0], [0], [0]])
    for i in range(n):
        theta_star = np.tile(np.array([[np.cos((i+1)*tau)], [np.sin((i+1)*tau)], [(i+1)*tau]]), (1, R))/(np.sqrt(2))
        sensor = rng.uniform(low=-50.0, high=50.0, size=(3,R))
        eps = rng.normal(loc=0.0, scale=1.0, size = (1, R))*0.2
        real_d = (((sensor-theta_star)**2).sum(axis=0))**0.5
        pseudo_d = (((sensor-theta)**2).sum(axis=0))**0.5
        d = real_d+eps
        residuals.append(d-pseudo_d)
        loss = (sensor - theta)*(-2*(pseudo_d-d)*((pseudo_d)**(-1.0)))
        theta = theta - gamma * loss
        a_i_plot.append(theta[0])
        b_i_plot.append(theta[1])
        c_i_plot.append(theta[2])
    a_i_plot = np.vstack(a_i_plot)
    b_i_plot = np.vstack(b_i_plot)
    c_i_plot = np.vstack(c_i_plot)
    sigmas = np.vstack(sigmas)
    fullMSEs=np.array(fullMSEs)
    a_i = np.vstack(a_i)
    b_i = np.vstack(b_i)
    c_i = np.vstack(c_i)
    a_star = np.array(a_star)
    b_star = np.array(b_star)
    c_star = np.array(c_star)
    a_top = np.vstack(a_top)
    b_top = np.vstack(b_top)
    c_top = np.vstack(c_top)
    a_bottom = np.vstack(a_bottom)
    b_bottom = np.vstack(b_bottom)
    c_bottom = np.vstack(c_bottom)
    gamma = 10*tau**(2/3)
    sigmas = np.array(sigmas)
    fig2 = plt.figure(figsize=(6, 5))
    ax = fig2.add_subplot(projection="3d")
    ax.plot3D(a_star, b_star, c_star, label="Target path")
    ax.plot3D(a_i_plot[:,0], b_i_plot[:,0], c_i_plot[:,0], label="SGD estimate", lw=0.5)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    radii = sigmas[times - 500, rep] * np.sqrt(gamma * chi2_crit)
    for t, r in zip(np.arange(500, a_i.shape[0], 500), radii):
        x = a_i[t][0] + r * np.outer(np.cos(u), np.sin(v))
        y = b_i[t][0] + r * np.outer(np.sin(u), np.sin(v))
        z = c_i[t][0] + r * np.outer(np.ones(np.size(u)), np.cos(v))
        surf = ax.plot_surface(x, y, z, alpha = 0.3, cmap='viridis', edgecolor='none')
    ax.set_xlabel("First coordinate (a)", labelpad=6, fontsize = 9)
    ax.set_ylabel("Second coordinate (b)", labelpad=6, fontsize = 9)
    ax.set_zlabel("Third coordinate (c)", labelpad=6, fontsize = 9)
    ax.set_title(rf"Source localization: $\tau={tau:g}$")
    ax.dist = 11
    ax.tick_params(labelsize=7, pad=0.01)
    x_range = np.diff(ax.get_xlim3d())[0]
    y_range = np.diff(ax.get_ylim3d())[0]
    z_range = np.diff(ax.get_zlim3d())[0]
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()
    z_min, z_max = ax.get_zlim3d()
    ax.set_xticks(np.arange(np.floor(x_min * 2) / 2, np.floor(x_max * 2) / 2 + 0.1, 0.5))
    ax.set_yticks(np.arange(np.ceil(y_min * 2) / 2, np.ceil(y_max * 2) / 2 + 0.1, 0.5))
    ax.set_box_aspect([x_range, y_range, z_range], zoom=0.80)
    fig2.subplots_adjust(left=0.15, right=0.85, bottom=0.20, top=0.85)
    plt.tight_layout()
    fig2.legend()
    plt.savefig(f"3dplot_tau_{tau:.7f}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.2)
results_file = Path("osl_results.csv")
np.savetxt(results_file,np.asarray(result_rows, dtype=float),delimiter=",",header="tau,MSE_per_coordinate,Coord1_pct,Coord2_pct,Coord3_pct,Ball_pct",comments="",fmt="%.10g")
print("Results saved to:", results_file.resolve())
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.scatter(taus, MSEs)
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.title("MSE vs Tau")
plt.xlabel("Tau")
plt.ylabel("Mean Squared Error")
plt.savefig('loglogOSLplot.pdf', format='pdf', bbox_inches='tight')
plt.show()
slope, intercept = np.polyfit(np.log10(taus), np.log10(MSEs), 1)

print(CIs)
print(f"Best fit slope (m): {slope:.4f}")
print(f"Y-intercept (b): {intercept:.4f}")
