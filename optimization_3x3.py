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



'''
This file performs inverse design using CMA-ES optimization

It:

Defines a 3×3 metasurface pattern
Runs RCWA simulations
Uses a loss function (physics target)
Optimizes parameters to find the best geometry

'''








# -------------------------------
# Initial Guess (11 parameters)
# 9 grid values + a + h
# -------------------------------

x0 = np.array([
    0.74370793, 0.44277883, 0.9856485, 0.64488753, 0.01934644, 0.74827313,
    0.26574786, 0.10920034, 0.75270867,   # 9 grid cells
    0.96542485, 0.390            # height h
])

sigma0 = 1

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
    return get_epgrid_3x3_new(pattern, eps, L1, L2)


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

    loss =  loss_three_region_quadfit6_trans_freq(
    get_epgrid_double_cylinder_d_new,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_curve=1.0,
    w_fit=0,
    save_file="good_quadratic_geometries7.txt"
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