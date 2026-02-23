import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa

L1 = [1,0]
L2 = [0,1]



nsio2 = 1.44
nair = 1


esio2 = nsio2**2
eair = nair**2

nG = 101

theta = np.pi/180 * 0
phi = 0

Nx = 300
Ny = 300


DBR_PAIRS = 5



def epsilon_lambda(wavelength, _cache={}):
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

