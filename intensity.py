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



'''This script computes and visualizes the electric and magnetic field intensities for a specific geometry and wavelength. 
It uses the defined geometry parameters, normal parameters, and DBR pairs to calculate the fields and plot them in various ways,
 including cross-sectional views and full structure intensity maps.'''



'''lambda0 is the wavelength for which dbr layers are designed'''
lambda0 = 1.5
lam = 1.51
lambdas = np.linspace(1.5, 1.52, 40)


'''Thickness of DBR layers are defined here'''
hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)

geometry_params = [0, 0.144, 0] # r1, r2, shift
normal_params = [0.84353636, 0.84353636, 0.473, 0.10782679, 0.25862069]

geometry_func = get_epgrid_double_cylinder_d_new


'''The value at which the xy fields are calculated in the meta surface layer'''

z_value=0.23

obj = intensity_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS,nG)


plot_xy_EH_fields(obj, normal_params,
                     z_value,
                     layer_index=1,
                     stride=4,
                     cmap='inferno')

plot_single_z_intensity(obj, normal_params,
                           z_value,
                           layer_index=1,
                           cmap='inferno')






x, z, Ixz, layer_bounds = compute_full_structure_xz_intensity_magnetic(
    obj,
    normal_params,
    N_dbr_pairs=DBR_PAIRS,
    Nz_per_layer=80
)

plot_full_structure_xz_intensity(
    x, z, Ixz,
    layer_bounds,
    title=f"Full structure Geometry 1 Magnetic Field intensity | λ={lam:.6f} µm",
    fname="full_xz_field.png"
)