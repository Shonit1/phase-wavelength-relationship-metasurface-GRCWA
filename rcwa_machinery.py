import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *



'''
This file contains all the machinery for setting up and running RCWA simulations using the grcwa library. 
It includes functions to build RCWA objects, compute reflection coefficients and phases, and extract field intensities.

'''






def rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS):


    '''
    """
    Builds an RCWA object for the given geometry and wavelength.

    - Defines lattice vectors and material permittivities
    - Adds air layer, patterned layer, and DBR stack (Si/SiO2)
    - Assigns geometry permittivity to grid
    - Excites structure with a plane wave

    Returns:
        obj : Configured RCWA simulation object
    """
    '''
    



    
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





def intensity_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS,nG):


    '''
    """
    Builds an RCWA object with all layers represented on a grid.

    - Uses grid layers instead of uniform layers for DBR
    - Constructs full permittivity array for all layers
    - Suitable for field and intensity visualization

    Returns:
        obj : RCWA object with full grid representation
    """
    '''


    
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


    '''
    """
    Computes reflection amplitude (r00) for zeroth diffraction order.

    - Identifies k=0 mode
    - Extracts incident (ai) and reflected (bi) amplitudes
    - Returns reflection coefficient r = b/a

    Returns:
        complex : Reflection amplitude
    """
    '''
    


    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)


    nV = obj.nG
        
    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


    



def reflectance_transmittance(obj):

    """
    Computes reflectance and transmittance for zeroth order.

    - Uses RCWA solver to compute R and T
    - Extracts zeroth diffraction order values
    - Also returns total energy (R + T)

    Returns:
        R0 : Reflectance (zeroth order)
        T0 : Transmittance (zeroth order)
        Sum : Total energy (sanity check)
    """


    k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
    R,T = obj.RT_Solve(normalize=1,byorder=1)
    Sum = np.sum(R) + np.sum(T)
    
    return R[k0],T[k0],Sum
    
    


def compute_full_structure_xz_intensity(obj, normal_params,
                                        N_dbr_pairs=5,
                                        Nz_per_layer=80,
                                        y_index=None):
    """
    Computes electric field intensity (|E|^2) in x-z plane.

    - Sweeps through all layers and z positions
    - Solves field distribution at each point
    - Extracts intensity along a fixed y slice

    Returns:
        x : x-coordinates
        z : z-coordinates (full structure)
        I : Intensity map (z, x)
        layer_bounds : z positions of layer interfaces
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





def compute_full_structure_xz_intensity_magnetic(obj, normal_params,
                                        N_dbr_pairs=5,
                                        Nz_per_layer=80,
                                        y_index=None):
    """
    Computes magnetic field intensity (|H|^2) in x-z plane.

    Same as electric version but uses magnetic field components.

    Returns:
        x, z, I, layer_bounds
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

            _, H = obj.Solve_FieldOnGrid(
                which_layer=layer_index + 1,
                z_offset=z
            )

            Hx, Hy, Hz = H
            I = np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2

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





def compute_single_z_xy_intensity(obj, normal_params,
                                 z_value,
                                 layer_index=1):
    """
    Computes 2D field intensity at a given z-plane.

    - Solves fields at specified layer and z offset
    - Computes both electric and magnetic intensities

    Returns:
        x : x-coordinates
        y : y-coordinates
        I : Electric field intensity (|E|^2)
        I_H : Magnetic field intensity (|H|^2)
    """

    L1, L2, hpattern, hs_dbr, hs_SiO2_dbr = normal_params

    # ---- Solve field at given z in chosen layer ----
    E, H = obj.Solve_FieldOnGrid(
        which_layer=layer_index,
        z_offset=z_value
    )

    Ex, Ey, Ez = E
    Hx, Hy, Hz = H

    # ---- Compute intensity ----
    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2   # shape (Nx, Ny)
    I_H = np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2
    # ---- Axes ----
    x = np.linspace(0, L1, Nx)
    y = np.linspace(0, L2, Ny)

    return x, y, I, I_H











def compute_phase(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS):


    """
    Computes reflection phase and amplitude vs wavelength.

    - Builds RCWA object for each wavelength
    - Extracts reflection coefficient
    - Computes phase and unwraps it

    Returns:
        phis : Unwrapped phase
        r_amp : Reflection intensity (|r|^2)
    """


    
    phis = []
    r_amp = []
    for lam in lambdas:
        f = 1 / lam
        obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS)
        r00 = relection_amplitude_computation(obj)
        phis.append(np.angle(r00))
        r_amp.append((np.abs(r00))**2)
    return np.unwrap(np.array(phis)),r_amp





def compute_phase_angle(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS,theta):
    
    """
    Computes reflection phase for a fixed wavelength and angle.

    Returns:
        phis : Unwrapped phase value
    """




    lam = 1.51
    
    f = 1 / lam
    obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS,theta)
    r00 = relection_amplitude_computation(obj)
    phis = np.angle(r00)
        
    return np.unwrap(np.array([phis]))








def compute_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS):


    """
    Computes reflectance vs wavelength.

    Returns:
        Rs : Array of reflectance values
    """


    
    Rs = []
    for lam in lambdas:
        f = 1 / lam
        obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS)
        R,T,sum = reflectance_transmittance(obj)
        Rs.append(R)

    return np.array(Rs)



def compute_reflectance_angle(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS,theta):


    """
    Computes reflectance and transmittance for a fixed angle.

    Returns:
        R : Reflectance
        T : Transmittance
    """


    
    lam = 1.51
    
    f = 1 / lam
    obj = rcwa_obj(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS,theta)
    R,T,sum = reflectance_transmittance(obj)
    

    return R,T







def compute_phase_and_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS):


    """
    Computes phase, reflectance, transmittance, and energy conservation.

    Returns:
        phis : Phase
        Rs : Reflectance
        Ts : Transmittance
        sums : R + T (energy check)
    """


    
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






def compute_phase_nG(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS,nG):



    """
    Computes phase using a custom number of Fourier modes (nG).

    Returns:
        phis : Phase
        r_amp : Reflection intensity
    """

    
    phis = []
    r_amp = []
    
    f = 1 / lambdas
    obj = rcwa_obj_nG(geometry_func,geometry_params,lambdas,normal_params,DBR_PAIRS,nG)
    r00 = relection_amplitude_computation(obj)
    phis.append(np.angle(r00))
    r_amp.append((np.abs(r00))**2)


    return np.unwrap(np.array(phis)),r_amp
    
    




def rcwa_obj_nG(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS,nG):


    """
    Same as rcwa_obj but allows custom nG (number of Fourier harmonics).

    Returns:
        obj : RCWA object
    """


    
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
    






def compute_phase_new(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS,nG):


    """
    Computes phase vs wavelength using rcwa_obj_nG.

    Returns:
        phis : Phase
        r_amp : Reflection intensity
    """

    
    phis = []
    r_amp = []
    for lam in lambdas:
        
        obj = rcwa_obj_nG(geometry_func,geometry_params,lam,normal_params,DBR_PAIRS,nG)
        r00 = relection_amplitude_computation(obj)
        phis.append(np.angle(r00))
        r_amp.append((np.abs(r00))**2)

    return np.unwrap(np.array(phis)),r_amp