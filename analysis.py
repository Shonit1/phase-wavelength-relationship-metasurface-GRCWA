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
from rcwa_machinery_frequency_domain import *




'''
"""
This script analyzes a specific structure using RCWA.

It takes a fixed geometry (already optimized or chosen) and evaluates:

1. Reflection spectrum vs wavelength
2. Numerical convergence with respect to Fourier modes (nG)
3. Angular response (phase, reflectance, transmittance vs incident angle)

Main purpose:
→ Validate and characterize a given design

This is NOT an optimization script.
It is used after design to understand performance.
"""

'''



geometry_params = [0.18440598, 0, 0.39424837]
normal_params   = [0.92150683, 0.92150683, 0.75147275, 0.10782679, 0.25862069]

geometry_func = get_epgrid_double_cylinder_d_new


plot_full_spectrum(geometry_func, geometry_params,
                       lambdas, normal_params)


nG_list = [101,201,301,401,501,601,701,801,901,1001]


phase_list = check_convergence_linear(geometry_params,normal_params,geometry_func,lambdas, nG_list)

plt.figure(figsize=(10,6))
plt.plot(nG_list, phase_list, marker='o')
plt.show()



'''to run this script successfully add additional argument theta(angle) to rcwa_obj in rcwa_machinery.py'''

theta_list = np.linspace(0,70,320) *np.pi/180
phi_list = []
R_list = []
T_list = []
for theta in theta_list:
    phis = compute_phase_angle(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS,theta)
    phi_list.append(phis[0])
    R,T = compute_reflectance_angle(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS,theta)
    R_list.append(R)
    T_list.append(T)



plt.plot(theta_list*180/np.pi, phi_list)
plt.xlabel("Angle of Incidence (degrees)")
plt.ylabel("Phase (rad)")
plt.title("Phase vs. Angle of Incidence")
plt.show()

plt.plot(theta_list*180/np.pi, R_list, label="Reflectance (0,0)")
plt.plot(theta_list*180/np.pi, T_list, label="Transmittance (-1,0)")
plt.xlabel("Angle of Incidence (degrees)")
plt.ylabel("Reflectance (0,0)/Transmittance (-1,0)")
plt.title("Reflectance and Transmittance vs. Angle of Incidence")
plt.legend()
plt.show()

