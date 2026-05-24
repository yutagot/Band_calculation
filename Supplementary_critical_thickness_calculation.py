# This script is to plot the critical thicknress of the thin film as a function of the Sn composition.
# Basic equation is based on the People and Bean model

import numpy as np
import matplotlib.pyplot as plt

#%% plot color
nn_blue = (19/255, 173/255, 181/255)
violet = (212/255, 0/255, 85/255)

#%% Define the parameters
b = 0.4    # nm, the Burgers vector

# c11_ge = 1.2853    # Ge, elastic constant 
# c12_ge = 0.4826    # Ge, elastic constant 

# c11_sn = 0.69   # Sn, elastic constant
# c12_sn = 0.293   # Sn, elastic constant

c11_ge = 1.292    # Ge, elastic constant 
c12_ge = 0.479    # Ge, elastic constant 

c11_sn = 0.725   # Sn, elastic constant
c12_sn = 0.297   # Sn, elastic constant

# a_ge = 5.658   # Ge, lattice parameter (Å)
# a_sn = 6.489   # Sn, lattice parameter (Å)

a_ge = 5.6579   # Ge, lattice parameter (Å)
a_sn = 6.4892   # Sn, lattice parameter (Å)

#%% Calculate the critical thickness
# Calculate the poisson's ratio for Ge and Sn
nu_ge = c12_ge / (c11_ge + c12_ge)
nu_sn = c12_sn / (c11_sn + c12_sn)

# Sn composition range
x = np.linspace(0, 0.2, 100)

# Calculate the lattice parameter of the alloy using Vegard's law
a_alloy = a_ge * (1 - x) + a_sn * x
a_alloy_nm = a_alloy / 10  # Convert from Å to nm
# Calculate the poisson's ratio of the alloy using a linear interpolation
nu_alloy = nu_ge * (1 - x) + nu_sn * x

# Calculate the lattice mismatch
f = (a_alloy - a_ge) / a_ge

# Calculate the critical thickness using the People and Bean model
with np.errstate(divide='ignore', invalid='ignore'):
    K = (
        ((1 - nu_alloy) / (1 + nu_alloy))
        * (1 / (16 * np.pi * np.sqrt(2)))
        * (b**2 / a_alloy_nm)
        * np.where(f != 0, 1 / f**2, np.inf)
    )

h_c = np.full_like(x, 10.0, dtype=float)  # initial guess for critical thickness in nm

max_iter = 2000  # maximum number of iterations for convergence
tol = 1e-9

mask = np.isfinite(K) & (K > 0)
converged = np.zeros_like(x, dtype=bool)
iter_to_converge = np.full_like(x, -1, dtype=int)
last_delta = np.full_like(x, np.nan, dtype=float)

for it in range(1, max_iter + 1):
    h_old = h_c.copy()
    h_new = h_c.copy()

    active = mask & ~converged
    if np.any(active):
        with np.errstate(divide='ignore', invalid='ignore'):
            h_new[active] = K[active] * np.log(h_old[active] / b)

    h_new[~mask] = np.inf

    delta = np.full_like(h_c, np.nan, dtype=float)
    delta[active] = np.abs(h_new[active] - h_old[active])
    last_delta = delta

    newly_converged = active & np.isfinite(delta) & (delta < tol)
    iter_to_converge[newly_converged] = it
    converged = converged | newly_converged

    h_c = h_new

    if np.all(converged | ~mask):
        break

converged_valid = mask & converged & np.isfinite(h_c) & (h_c > 0)
not_converged = mask & ~converged

print(f"Converged points: {np.sum(converged_valid)}/{np.sum(mask)}")
if np.any(not_converged):
    print("Not converged x (first 10):", x[not_converged][:10])

print("Critical thickness h_c =", h_c, "nm")

#%% Plot (exclude inf at x=0 where f=0)
plot_mask = converged_valid

fig, ax = plt.subplots()
ax.plot(x[plot_mask], h_c[plot_mask], color=nn_blue)
ax.set_xlabel("Sn composition x")
ax.set_ylabel("Critical thickness $h_c$ (nm)")
ax.set_yscale("log")
ax.grid(True)
ax.set_xlim(0, 0.20)
ax.set_xticks(np.arange(0, 0.21, 0.05))
ax.set_ylim(1e0, 6e4)
ax.minorticks_on()
ax.tick_params(which="both", direction="in", top=True, right=True)
fig.tight_layout()
fig.savefig("critical_thickness_vs_x.pdf", dpi=300)
plt.show()
