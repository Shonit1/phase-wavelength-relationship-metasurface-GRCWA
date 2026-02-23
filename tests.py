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





lambda0 = 1.5


hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)



lambdas = np.linspace(1.5, 1.52, 40)



geometry_params, normal_params = [0.12040733, 0.01271771, 0.46051845 ], [1.14230889, 1.14230889, 0.61387762, 0.10782679, 0.25862069]

geometry_func = get_epgrid_double_cylinder_d_new



print("Geometry Parameters (r1, r2, shift):", geometry_params  )
print("Normal Parameters (L1[0], L1[1], hpattern, hs_dbr, hs_SiO2_dbr):", normal_params)


phis, Rs = compute_phase(
    geometry_func,
    geometry_params,
    lambdas,
    normal_params,
    DBR_PAIRS
)



# ============================================
# 1️⃣ Fit to sqrt(lambda)
# phi = A*sqrt(lambda) + B
# ============================================

lambda_c = 1.5

# Shift wavelength
shifted = lambdas - lambda_c

# Prevent negative values (important near the pole)
eps = 1e-12
shifted = np.maximum(shifted, eps)

sqrt_shifted = np.sqrt(shifted)

# Linear fit: phi = A * sqrt(λ - λc) + B
coeffs_sqrt = np.polyfit(sqrt_shifted, phis, 1)
A_sqrt, B_sqrt = coeffs_sqrt

phi_sqrt_fit = A_sqrt * sqrt_shifted + B_sqrt

rms_sqrt = np.sqrt(np.mean((phis - phi_sqrt_fit)**2))


# ============================================
# 2️⃣ Fit to linear lambda
# phi = A*lambda + B
# ============================================s

coeffs_lin = np.polyfit(lambdas, phis, 1)
A_lin, B_lin = coeffs_lin
phi_lin_fit = A_lin * lambdas + B_lin

rms_lin = np.sqrt(np.mean((phis - phi_lin_fit)**2))





# Cubic polynomial fit (degree 3)
coeffs_cubic = np.polyfit(lambdas, phis, 3)

A3, A2, A1, A0 = coeffs_cubic  # λ^3, λ^2, λ^1, constant

phi_cubic_fit = (
    A3 * lambdas**3 +
    A2 * lambdas**2 +
    A1 * lambdas +
    A0
)

# RMS error
rms_cubic = np.sqrt(np.mean((phis - phi_cubic_fit)**2))





# ============================================
# 1️⃣ Fit to 1/(lambda - lambda_c)
# phi = A / (lambda - lambda_c) + B
# ============================================

lambda_c = 1.5

# Exclude region too close to singularity
mask = np.abs(lambdas - lambda_c) > 0.0005

l_fit = lambdas[mask]
phi_fit_region = phis[mask]

shifted = l_fit - lambda_c
inv_shifted = 1.0 / shifted

coeffs_inv = np.polyfit(inv_shifted, phi_fit_region, 1)
A_inv, B_inv = coeffs_inv

# Evaluate model on full lambda range
phi_inv_fit = A_inv / (lambdas - lambda_c) + B_inv


# RMS error
rms_inv = np.sqrt(np.mean((phis - phi_inv_fit)**2))

# R² score (important!)
ss_res = np.sum((phis - phi_inv_fit)**2)
ss_tot = np.sum((phis - np.mean(phis))**2)

r2_inv = 1 - ss_res / ss_tot

print("A_inv =", A_inv)
print("B_inv =", B_inv)
print("RMS error =", rms_inv)
print("R² =", r2_inv)







lambda_c = 1.5

mask = lambdas > (lambda_c + 0.0005)

l_fit = lambdas[mask]
phi_fit_region = phis[mask]

shifted = l_fit - lambda_c
inv_shifted = 1.0 / shifted

# Design matrix
X = np.vstack([
    inv_shifted,
    l_fit,
    np.ones_like(l_fit)
]).T

coeffs, _, _, _ = np.linalg.lstsq(X, phi_fit_region, rcond=None)

A_fit, D_fit, B_fit = coeffs

# Reconstruct full curve
phi_model = (
    A_fit / (lambdas - lambda_c)
    + D_fit * lambdas
    + B_fit
)

# Metrics
rms = np.sqrt(np.mean((phis - phi_model)**2))

ss_res = np.sum((phis - phi_model)**2)
ss_tot = np.sum((phis - np.mean(phis))**2)
r2 = 1 - ss_res / ss_tot

print("A =", A_fit)
print("D =", D_fit)
print("B =", B_fit)
print("RMS =", rms)
print("R² =", r2)






# ============================================
# Print comparison
# ============================================

print("\n===== Fit Comparison =====")
print("SQRT Fit:")
print("A =", A_sqrt)
print("B =", B_sqrt)
print("RMS_sqrt =", rms_sqrt)

print("\nLINEAR Fit:")
print("A =", A_lin)
print("B =", B_lin)
print("RMS_linear =", rms_lin)

print("\nCUBIC Fit:")
print("A3 =", A3)
print("A2 =", A2)
print("A1 =", A1)
print("A0 =", A0)
print("RMS_cubic =", rms_cubic)

print("\nINV Fit:")
print("A =", A_inv)
print("B =", B_inv)
print("RMS_inv =", rms_inv)

print("\nINV + LINEAR Fit:")
print("A =", A_fit)
print("D =", D_fit)
print("B =", B_fit)
print("RMS =", rms)

# ============================================
# Plot phase comparison
# ============================================

plt.figure(figsize=(8,6))
plt.plot(lambdas, phis, label="RCWA Phase", linewidth=2)
#plt.plot(lambdas, phi_sqrt_fit, '--', label="A√(λ-λc) + B Fit", linewidth=2)
plt.plot(lambdas, phi_lin_fit, ':', label=f"Aλ + B Fit, A={A_lin:.3f}, RMS={rms_lin:.3e}", linewidth=2)
#plt.plot(lambdas, phi_inv_fit, '-.', label="A/(λ-λc) + B Fit", linewidth=2)
#plt.plot(lambdas, phi_cubic_fit, '-.', label="Cubic Fit", linewidth=2)
#plt.plot(lambdas, phi_model, '--', label="A/(λ-λc) + Dλ + B Fit", linewidth=2)

plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title("Phase Fit Comparison")
plt.legend()
plt.grid(True)
plt.show()


# ============================================
# Plot residuals comparison
# ============================================

plt.figure(figsize=(8,5))
plt.plot(lambdas, phis - phi_sqrt_fit, label="Residual √λ")
plt.plot(lambdas, phis - phi_lin_fit, label="Residual Linear")

plt.xlabel("Wavelength (µm)")
plt.ylabel("Residual (rad)")
plt.title("Residual Comparison")
plt.legend()
plt.grid(True)
plt.show()
