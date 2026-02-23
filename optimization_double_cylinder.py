import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
from geometry_functions import *
from geometry_visuals import *
from rcwa_machinery import *
from plot_spectra import *
from loss_functions import *
from sweep_functions import *


# -------------------------------
# Constants
# -------------------------------

lambda0 = 1.5
hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)

lambdas = np.linspace(1.49, 1.505, 30)
# Initial Guess (now in u-space)
# -------------------------------

x0 = np.array([
    0.2,   # r1
    0.15,  # r2
    0.5,   # d
    1.0,   # a
    0.5    # hpattern
])


sigma0 = 0.1

# -------------------------------
# CMA bounds (for u variables)
# -------------------------------

opts = {
    "bounds": [
        [0.05, 0, 0.05, 0.8, 0.1],   # lower
        [0.45, 0.45, 0.5, 1.2, 0.8]     # upper
    ],
    "popsize": 16,
    "maxiter": 40,
    "verb_disp": 1
}

# -------------------------------
# Decode function
# -------------------------------

def decode(x):
    r1, r2, d, a, hpattern = x

    geometry_params = np.array([r1, r2, d])
    normal_params = np.array([a, a, hpattern, hs_dbr, hs_SiO2_dbr])

    return geometry_params, normal_params


def is_valid_geometry(r1, r2, d, a):

    # Radii must fit inside cell
    if r1 <= 0 or r2 <= 0:
        return False

    if r1 >= a/2 or r2 >= a/2:
        return False

    # No overlap
    if d < (r1 + r2):
        return False

    # Stay inside lattice
    if (d/2 + r1) > a/2:
        return False
    
    if (d/2 + r2)> a/2:
        return False

    return True


target_slope = 150

# -------------------------------
# Objective
# -------------------------------

def objective(x):

    r1, r2, d, a, hpattern = x

    # ----- Hard geometry validation -----
    if not is_valid_geometry(r1, r2, d, a):
        return 1e6  # large penalty

    geometry_params, normal_params = decode(x)

    return loss_three_region_polyfit_trans(
    get_epgrid_double_cylinder_d_new,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_center=1.0,
    w_trans=10.0,
    save_file="good_cubic_geometries.txt"
)

# -------------------------------
# Run CMA
# -------------------------------
best_loss_so_far = np.inf
best_x_so_far = None




es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

while not es.stop():
    solutions = es.ask()
    losses = [objective(x) for x in solutions]
    es.tell(solutions, losses)
    es.disp()

    # ---- CHECKPOINT ----
    if es.best.f < best_loss_so_far:
        best_loss_so_far = es.best.f
        best_x_so_far = es.best.x.copy()

        # Save as TXT
        np.savetxt("best_geometry_checkpoint.txt",
                   best_x_so_far.reshape(1, -1),
                   fmt="%.10f")

        print("Checkpoint saved. Loss =", best_loss_so_far)


result = es.result

best_x = result.xbest
best_loss = result.fbest

print("\nOptimization Finished")
print("Best u-parameters:", best_x)
print("Best Loss:", best_loss)

# -------------------------------
# Decode best solution
# -------------------------------

geometry_params, normal_params = decode(best_x)

print("Best geometry params:", geometry_params)
print("Best normal params:", normal_params)

# -------------------------------
# Plot result
# -------------------------------

plot_full_spectrum(
    get_epgrid_double_cylinder_d_new,
    geometry_params,
    lambdas,
    normal_params
)

