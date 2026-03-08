import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
from rcwa_machinery import *



def rcwa_obj_f(geometry_func,geometry_params,f,normal_params,DBR_PAIRS):
    
    L10,L20,hpattern,hs_dbr,hsio2_dbr = normal_params

    L1 = [L10, 0]
    L2 = [0, L20]
    lam = 1/f
    eps_si = epsilon_lambda(lam)
    obj = grcwa.obj(nG,L1,L2,f,theta,phi,verbose=0)
    
    
    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(hpattern, Nx, Ny)

    for _ in range(DBR_PAIRS):
            obj.Add_LayerUniform(hsio2_dbr, esio2)
            obj.Add_LayerUniform(hs_dbr, eps_si)

    obj.Add_LayerUniform(0.1, esio2)

    obj.Init_Setup()

    ep = geometry_func(geometry_params, eps_si,L1,L2).flatten()
    obj.GridLayer_geteps(ep)        


    obj.MakeExcitationPlanewave(1, 0, 0, 0)

    return obj










def compute_phase_and_reflectance_f(geometry_func, geometry_params, omegas, normal_params, DBR_PAIRS):
    
    phis = []
    Rs = []
    Ts = []
    sums = []
    for f in omegas:
        
        obj = rcwa_obj_f(geometry_func,geometry_params,f,normal_params,DBR_PAIRS)
        r00 = relection_amplitude_computation(obj)
        R,T,sum = reflectance_transmittance(obj)

        phis.append(np.angle(r00))
        Rs.append(R)
        Ts.append(T)
        sums.append(sum)
    
    return np.unwrap(np.array(phis)), np.array(Rs), np.array(Ts), np.array(sums)






def compute_phase_f(geometry_func, geometry_params, omegas, normal_params, DBR_PAIRS):
    
    phis = []
    r_amp = []
    for f in omegas:
        
        obj = rcwa_obj_f(geometry_func,geometry_params,f,normal_params,DBR_PAIRS)
        r00 = relection_amplitude_computation(obj)
        phis.append(np.angle(r00))
        r_amp.append((np.abs(r00))**2)
    return np.unwrap(np.array(phis)),r_amp






def plot_full_spectrum_f(geometry_func, geometry_params,
                       omegas, normal_params):

    phis, Rs, Ts, sums = compute_phase_and_reflectance_f(
        geometry_func,
        geometry_params,
        omegas,
        normal_params,
        DBR_PAIRS
    )

    

    fig, ax = plt.subplots(3, 1, figsize=(6,9), sharex=True)

    # Phase
    ax[0].plot(omegas, phis, linewidth=2)
    ax[0].set_ylabel("Phase (rad)")
    ax[0].set_title("Phase")

    # Reflectance
    ax[1].plot(omegas, Rs, linewidth=2)
    ax[1].axhline(0.8, linestyle='--')
    ax[1].set_ylabel("Reflectance")
    ax[1].set_ylim(0,1.05)
    ax[1].set_title("Reflectance")

    # Transmission
    ax[2].plot(omegas, Ts, linewidth=2)
    ax[2].set_ylabel("Transmission")
    ax[2].set_xlabel("Wavelength (µm)")
    ax[2].set_ylim(0,1.05)
    ax[2].set_title("Transmission")

    for a in ax:
        a.grid(True)

    plt.tight_layout()
    plt.show()