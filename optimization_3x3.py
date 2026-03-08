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

lambdas = np.concatenate([np.linspace(1.49, 1.496, 20), np.linspace(1.496, 1.505, 20)])

# -------------------------------
# Initial Guess (11 parameters)
# 9 grid values + a + h
# -------------------------------

x0 = np.array([
    0.74370793, 0.44277883, 0.9856485, 0.64488753, 0.01934644, 0.74827313,
    0.26574786, 0.10920034, 0.75270867,   # 9 grid cells
    0.96542485, 0.390            # height h
])

sigma0 = 0.2

# -------------------------------
# CMA bounds
# Grid entries: 0 → 1
# a: 0.8 → 1.2
# h: 0.1 → 0.8
# -------------------------------

lower_bounds = [0.0]*9 + [0.8, 0.1]
upper_bounds = [1.0]*9 + [1, 0.8]

opts = {
    "bounds": [lower_bounds, upper_bounds],
    "popsize": 12,
    "maxiter": 40,
    "verb_disp": 1
}

# -------------------------------
# Decode function
# -------------------------------

def decode(x):

    pattern = x[:9]
    a = x[9]
    hpattern = x[10]

    geometry_params = pattern
    normal_params = np.array([a, a, hpattern, hs_dbr, hs_SiO2_dbr])

    return geometry_params, normal_params


# -------------------------------
# Geometry wrapper for RCWA
# This wraps your get_epgrid_3x3
# -------------------------------

def geometry_wrapper(pattern, eps, L1, L2):
    return get_epgrid_3x3(pattern, eps, L1, L2)


# -------------------------------
# Objective
# -------------------------------

def objective(x):

    pattern = x[:9]
    a = x[9]
    hpattern = x[10]

    # Optional: encourage binary behavior
    binary_penalty = np.sum(pattern*(1-pattern))
    
    geometry_params, normal_params = decode(x)

    loss =  loss_center_symmetric_quadratic(
    geometry_wrapper,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    center_lambda=1.496,
    w_slope=50,
    w_sym=20,
    w_refl=10,
    save_file="good_quadratic_geometries_centered.txt"
)

    return loss + 5.0*binary_penalty


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

    if es.best.f < best_loss_so_far:
        best_loss_so_far = es.best.f
        best_x_so_far = es.best.x.copy()

        np.savetxt(
            "best_geometry_checkpoint_3x3.txt",
            best_x_so_far.reshape(1,-1),
            fmt="%.10f"
        )

        print("Checkpoint saved. Loss =", best_loss_so_far)

result = es.result

best_x = result.xbest
best_loss = result.fbest

print("\nOptimization Finished")
print("Best parameters:", best_x)
print("Best Loss:", best_loss)

# -------------------------------
# Decode best solution
# -------------------------------

geometry_params, normal_params = decode(best_x)

print("Best pattern:")
print(np.array(geometry_params).reshape(3,3))
print("Best normal params:", normal_params)

# -------------------------------
# Plot result
# -------------------------------

plot_full_spectrum(
    geometry_wrapper,
    geometry_params,
    lambdas,
    normal_params
)