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



lambda0 = 1.5



lambdas = np.linspace(1.5, 1.52, 40)


hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)



geometry_params = [0.12040733, 0.01271771, 0.46051845]  # r1, r2, shift
normal_params = [1.14230889, 1.14230889, 0.61387762, hs_dbr, hs_SiO2_dbr]

geometry_func = get_epgrid_double_cylinder_d_new


print("Geometry Parameters (r1, r2, shift):", geometry_params)
print("Normal Parameters (a, a, hpattern, hs_dbr, hs_SiO2_dbr):", normal_params)



r1, r2, d = geometry_params
a = normal_params[0]
eps = 3
L1=L2=a


plot_double_cylinder_xy_from_grid(geometry_params, eps, L1, L2)

plot_phase(geometry_func, geometry_params,
                       lambdas, normal_params)  

plot_full_spectrum(geometry_func, geometry_params,
                       lambdas, normal_params)








'''
r1, r2, shift = geometry_params
eps = 12.25
L1 = [normal_params[0], 0]
L2 = [0, normal_params[1]]



plot_double_cylinder(r1, r2, shift, eps)

hpattern = normal_params[2]


plot_dual_cylinder_structure(
    r1, r2, shift, L1[0],
    hpattern,
    DBR_PAIRS,
    hs_SiO2_dbr,
    hs_dbr
)

'''