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
lam = 1.51
lambdas = np.linspace(1.5, 1.52, 40)

hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)

geometry_params = [0.28884285, 0.20469249, 0.51173306] # r1, r2, shift
normal_params = [1.09972403 ,1.09972403, 0.79574098 , hs_dbr, hs_SiO2_dbr]

geometry_func = get_epgrid_double_cylinder_d



obj = intensity_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS)




x, z, Ixz, layer_bounds = compute_full_structure_xz_intensity(
    obj,
    normal_params,
    N_dbr_pairs=DBR_PAIRS,
    Nz_per_layer=80
)

plot_full_structure_xz_intensity(
    x, z, Ixz,
    layer_bounds,
    title=f"Full structure | λ={lam:.6f} µm",
    fname="full_xz_field.png"
)