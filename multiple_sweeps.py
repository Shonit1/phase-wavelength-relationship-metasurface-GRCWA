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
from sweep_functions import *
import re
import os


'''
This script is a post-processing + screening tool for optimized geometries

It:

Reads saved geometries from optimization
Recomputes their phase and reflectance
Plots results
Saves each geometry’s performance as an image

'''






# -----------------------------------
# USER SETTINGS
# -----------------------------------

input_file = "good_geometries.txt"
output_folder = "geometry_plots_45"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# -----------------------------------
# FUNCTION: PARSE TXT FILE
# -----------------------------------

def parse_geometry_file(filename):

    with open(filename, "r") as f:
        text = f.read()

    geom_blocks = re.findall(
        r"GeometryParams\s*=\s*\[([^\]]+)\].*?normalParams\s*=\s*\[([^\]]+)\]",
        text,
        re.S,
    )

    geometries = []

    for g, n in geom_blocks:

        g_vals = np.fromstring(g, sep=" ")
        n_vals = np.fromstring(n, sep=" ")

        geometries.append((g_vals, n_vals))

    return geometries


# -----------------------------------
# LOAD GEOMETRIES
# -----------------------------------

geometries = parse_geometry_file(input_file)

print("Total geometries found:", len(geometries))


# -----------------------------------
# WAVELENGTH GRID
# -----------------------------------


lambda0 = 1.5
omega0 = 1.26e15  # rad/s

wmax = 1.26e15
wmin = 1.25e15

lambdas = np.linspace(1.5, 1.52, 40)
omegas = np.linspace(wmin, wmax, 40)

geometry_func = get_epgrid_double_cylinder_d_new
# -----------------------------------
# LOOP THROUGH GEOMETRIES
# -----------------------------------


# -----------------------------------
# Convergence settings
# -----------------------------------

nG_list = [101,201,301,401,501,601,701,801,901,1001]
lambdas1 = [1.5, 1.52]

tol = 0.01   # 1% convergence tolerance


for i, (geometry_params, normal_params) in enumerate(geometries):

    print("Running geometry", i)

    # -----------------------------------
    # Check convergence
    # -----------------------------------
    
    phase_list = check_convergence_linear(
        geometry_params,
        normal_params,
        geometry_func,
        lambdas1,
        nG_list
    )

    phase_array = np.array(phase_list).flatten()

    # relative change between last two points
    if len(phase_array) >= 2:
        rel_change = abs(phase_array[-1] - phase_array[-2]) / (abs(phase_array[-1]) + 1e-9)
    else:
        rel_change = np.nan

    converged = rel_change < tol
    
    # -----------------------------------
    # Compute phase spectrum
    # -----------------------------------

    phis, Rs, Ts, sums = compute_phase_and_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS)

    if phis is None:
        continue

    phis = np.unwrap(phis)

    # -----------------------------------
    # PLOT
    # -----------------------------------

    fig, ax = plt.subplots(3, 1, figsize=(6,12))

    # -----------------------------------
    # Original Phase
    # -----------------------------------

    ax[0].plot(lambdas, phis, linewidth=2)
    ax[0].set_title("Phase")
    ax[0].set_xlabel("Wavelength (µm)")
    ax[0].set_ylabel("Phase (rad)")

    
    
    ax[1].plot(lambdas, Rs, linewidth=2)
    ax[1].set_title("Reflectance")
    ax[1].set_xlabel("Wavelength (µm)")
    ax[1].set_ylabel("Reflectance")
    # -----------------------------------
    # RCWA convergence
    # -----------------------------------
    
    ax[2].plot(nG_list, phase_array, marker='o', linewidth=2)
    ax[2].set_title("RCWA Convergence")
    ax[2].set_xlabel("Number of Fourier Harmonics (nG)")
    ax[2].set_ylabel("Δφ (rad)")
    


    
    

    
    plt.tight_layout()

    filename = os.path.join(output_folder, f"geometry_{i}.png")

    plt.savefig(filename, dpi=200)
    plt.close()

    print("Saved:", filename)



