import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *



def rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS):
    
    L10,L20,hpattern,hs_dbr,hsio2_dbr = normal_params

    L1 = [L10, 0]
    L2 = [0, L20]

    eps_si = epsilon_lambda(lam)
    obj = grcwa.obj(nG,L1,L2,1/lam,theta,phi,verbose=0)
    
    
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





def intensity_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS):
    
    L10,L20,hpattern,hs_dbr,hsio2_dbr = normal_params

    L1 = [L10, 0]
    L2 = [0, L20]

    eps_si = epsilon_lambda(lam)
    obj = grcwa.obj(nG,L1,L2,1/lam,theta,phi,verbose=0)
    
    
    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(hpattern, Nx, Ny)

    for _ in range(DBR_PAIRS):
            obj.Add_LayerGrid(hsio2_dbr, Nx, Ny)
            obj.Add_LayerGrid(hs_dbr, Nx, Ny)

    obj.Add_LayerUniform(0.1, esio2)

    obj.Init_Setup()

    ep = geometry_func(geometry_params, eps_si,L1,L2).flatten()
    epSio2_dbr  = np.full(Nx * Ny, esio2)
    epSi_dbr = np.full(Nx * Ny, eps_si)

    ep_all = np.concatenate([ep] + [epSio2_dbr,epSi_dbr]*5)
    obj.GridLayer_geteps(ep_all)

          


    obj.MakeExcitationPlanewave(1, 0, 0, 0)

    return obj












def relection_amplitude_computation(obj):
    


    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)


    nV = obj.nG
        
    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


    



def reflectance_transmittance(obj):


    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    R,T = obj.RT_Solve(normalize=1,byorder=1)
    Sum = np.sum(R) + np.sum(T)

    return R[k0],T[k0],Sum



def compute_full_structure_xz_intensity(obj, normal_params,
                                        N_dbr_pairs=5,
                                        Nz_per_layer=80,
                                        y_index=None):
    """
    Returns:
    x (Nx)
    z (Nz_total)
    I (Nz_total, Nx)
    layer_bounds (z positions)
    """

    L1, L2, hpattern, hs_dbr, hs_SiO2_dbr = normal_params

    # ----- Build layer thickness list -----
    layer_thicknesses = (
        [hpattern] +
        [hs_SiO2_dbr,hs_dbr] * N_dbr_pairs
    )

    I_all = []
    z_all = []
    z_offset_global = 0.0
    layer_bounds = []

    for layer_index, h in enumerate(layer_thicknesses):

        z_vals = np.linspace(0, h, Nz_per_layer)
        I_layer = np.zeros((Nz_per_layer, Nx))

        if y_index is None:
            y_index = Ny // 2

        for i, z in enumerate(z_vals):

            E, _ = obj.Solve_FieldOnGrid(
                which_layer=layer_index + 1,
                z_offset=z
            )

            Ex, Ey, Ez = E
            I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

            I_layer[i, :] = I[:, y_index]

        z_shifted = z_vals + z_offset_global

        I_all.append(I_layer)
        z_all.append(z_shifted)

        z_offset_global += h
        layer_bounds.append(z_offset_global)

    I_all = np.vstack(I_all)
    z_all = np.concatenate(z_all)

    x = np.linspace(0, L1, Nx)

    return x, z_all, I_all, layer_bounds




def compute_phase(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS):
    
    phis = []
    r_amp = []
    for lam in lambdas:
        f = 1 / lam
        obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS)
        r00 = relection_amplitude_computation(obj)
        phis.append(np.angle(r00))
        r_amp.append((np.abs(r00))**2)
    return np.unwrap(np.array(phis)),r_amp




def compute_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS):
    
    Rs = []
    for lam in lambdas:
        f = 1 / lam
        obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS)
        R,T,sum = reflectance_transmittance(obj)
        Rs.append(R)

    return np.array(Rs)



def compute_phase_and_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS):
    
    phis = []
    Rs = []
    Ts = []
    sums = []
    for lam in lambdas:
        f = 1 / lam
        obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS)
        r00 = relection_amplitude_computation(obj)
        R,T,sum = reflectance_transmittance(obj)

        phis.append(np.angle(r00))
        Rs.append(R)
        Ts.append(T)
        sums.append(sum)
    
    return np.unwrap(np.array(phis)), np.array(Rs), np.array(Ts), np.array(sums)







    
    