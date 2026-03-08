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


lambdas = np.linspace(1.49, 1.505, 40)



hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)



geometry_params = [0.99068852, 0.01841712, 0.58133218, 0.0402274, 0.59880306, 0.79753591, 0.41705112, 0.59276579, 0.76895974]
normal_params = [0.9941009, 0.9941009, 0.76684399, hs_dbr, hs_SiO2_dbr]

geometry_func = get_epgrid_3x3

a = normal_params[0]
hpattern = normal_params[2]

'''
plot_3x3_geometry(geometry_params, a ,threshold=0.5)

plot_3x3_pattern_xz(
    geometry_params,
    a,
    hpattern,
    DBR_PAIRS,
    hs_SiO2_dbr,
    hs_dbr,
    threshold=0.5
)

'''
'''
plot_phase(geometry_func, geometry_params,
               lambdas, normal_params)

'''



lambdas = [1.49,1.505]


nG_list = [3000,4000]

phase_list =[]

phase_list = check_convergence_linear(geometry_params,normal_params,geometry_func,lambdas, nG_list)





