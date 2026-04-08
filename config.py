import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa



'''This file contains all the constants, settings, and epsilon function that are used across multiple scripts.'''

'''Lattice vectors'''
L1 = [1,0]
L2 = [0,1]

'''Refractive indices'''

nsio2 = 1.44
nair = 1

'''Derived permittivities'''
esio2 = nsio2**2
eair = nair**2


'''Number of Fourier orders for RCWA'''
nG = 101

'''Angle of incidence (in radians)'''
theta = np.pi/180 * 0
phi = 0

'''Number of points in the Grid'''
Nx = 2000
Ny = 2000

'''Number of DBR pairs'''
DBR_PAIRS = 5





'''DBR designed for lambda0 = 1.5µm'''

lambda0 = 1.5

'''DBR layer thicknesses'''
hs_dbr = lambda0/(4*3.4778)
hs_SiO2_dbr = lambda0/(4*1.45)


'''Wavelength range for simulations'''



N = 40
lambdas = np.linspace(1.5, 1.52, N) 









def epsilon_lambda(wavelength, _cache={}):

    """
    Returns dielectric permittivity (ε = n²) at a given wavelength using
    cubic interpolation of refractive index data from a CSV file.
    The interpolation is cached to avoid reloading the file each time.
    """



    if "interp" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values

        _cache["interp"] = interp1d(
            wl, n, kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    n_val = _cache["interp"](wavelength)
    return n_val**2

