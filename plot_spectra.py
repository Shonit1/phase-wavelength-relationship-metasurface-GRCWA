import numpy as np
import matplotlib.pyplot as plt
from config import *
from geometry_functions import *
from rcwa_machinery import *


'''



"""
Provides plotting utilities to visualize phase, reflectance, transmission,
and electromagnetic field distributions of metasurface structures.
"""

'''





def plot_phase(geometry_func, geometry_params,
               lambdas, normal_params):
    
    '''
    Plots reflection phase vs wavelength for a given geometry.
    '''



    phis, Rs= compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    phis = np.unwrap(phis)
    
    plt.figure(figsize=(6,4))
    plt.plot(lambdas, phis, linewidth=2)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Phase (rad)")
    plt.title("Reflection Phase vs Wavelength")
    plt.grid(True)
    plt.tight_layout()
    plt.show()







def plot_reflectance(geometry_func, geometry_params,
                     lambdas, normal_params):


    '''
    Plots reflectance spectrum (R vs λ) and shows a reference line (e.g., R = 0.8).
    '''


    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    plt.figure(figsize=(6,4))
    plt.plot(lambdas, Rs, linewidth=2)
    plt.axhline(0.8, linestyle='--')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Reflectance")
    plt.title("Reflectance Spectrum")
    plt.ylim(0,1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()






def plot_transmission(geometry_func, geometry_params,
                      lambdas, normal_params):
    

    '''Plots transmission spectrum (T vs λ) of the structure.'''



    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    plt.figure(figsize=(6,4))
    plt.plot(lambdas, Ts, linewidth=2)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Transmission")
    plt.title("Transmission Spectrum")
    plt.ylim(0,1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.show()






def plot_full_spectrum(geometry_func, geometry_params,
                       lambdas, normal_params):
    


    '''Plots phase, reflectance, and transmission together for complete spectral analysis.'''




    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    

    fig, ax = plt.subplots(3, 1, figsize=(6,9), sharex=True)

    # Phase
    ax[0].plot(lambdas, phis, linewidth=2)
    ax[0].set_ylabel("Phase (rad)")
    ax[0].set_title("Phase")

    # Reflectance
    ax[1].plot(lambdas, Rs, linewidth=2)
    ax[1].axhline(0.8, linestyle='--')
    ax[1].set_ylabel("Reflectance (0,0)")
    ax[1].set_ylim(0,1.05)
    ax[1].set_title("Reflectance")

    # Transmission
    ax[2].plot(lambdas, Ts, linewidth=2)
    ax[2].set_ylabel("Transmission (-1,0)")
    ax[2].set_xlabel("Wavelength (µm)")
    ax[2].set_ylim(0,1.05)
    ax[2].set_title("Transmission")

    for a in ax:
        a.grid(True)

    plt.tight_layout()
    plt.show()





def plot_energy_balance(geometry_func, geometry_params,
                        lambdas, normal_params):
    
    '''Plots R + T vs wavelength to check energy conservation (should be ≈ 1).'''






    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    plt.figure(figsize=(6,4))
    plt.plot(lambdas, Rs + Ts, linewidth=2)
    plt.axhline(1.0, linestyle='--')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("R + T")
    plt.title("Energy Conservation Check")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def plot_full_structure_xz_intensity(x, z, I, layer_bounds,
                                     title, fname):
    

    '''Visualizes field intensity (|E|²) in the x–z cross-section of the full device.'''





    plt.figure(figsize=(6, 6))

    plt.pcolormesh(x, z, I, shading="auto", cmap="inferno")
    plt.colorbar(label="|H|²")

    for zb in layer_bounds:
        plt.axhline(zb, color="white", lw=0.6, alpha=0.6)

    plt.xlabel("x (µm)")
    plt.ylabel("z (µm)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()







def plot_single_z_intensity(obj, normal_params,
                           z_value,
                           layer_index=1,
                           cmap='inferno'):
    """
    Plots intensity I(x, y) at a given z and layer.
    """

    # ---- Get intensity ----
    x, y, I, I_H = compute_single_z_xy_intensity(
        obj,
        normal_params,
        z_value=z_value,
        layer_index=layer_index
    )

    # ---- Create meshgrid for plotting ----
    X, Y = np.meshgrid(x, y, indexing='ij')

    # ---- Plot ----
    plt.figure(figsize=(6, 5))

    im = plt.pcolormesh(X, Y, I_H, shading='auto', cmap=cmap)

    # ---- Labels ----
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Magnetic Field Intensity - Geometry 1 at z = {z_value:.3f} (Layer {layer_index})')

    # ---- Colorbar ----
    cbar = plt.colorbar(im)
    cbar.set_label('|H|²')

    # ---- Make it look nicer ----
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    plt.show()










def plot_xy_EH_fields(obj, normal_params,
                     z_value,
                     layer_index=1,
                     stride=4,
                     cmap='inferno'):
    """
    Plots:
    - Left: |E|^2 + (Ex, Ey) vectors
    - Right: |H|^2 + (Hx, Hy) vectors
    """

    L1, L2, hpattern, hs_dbr, hs_SiO2_dbr = normal_params

    # ---- Solve field ----
    E, H = obj.Solve_FieldOnGrid(
        which_layer=layer_index,
        z_offset=z_value
    )

    Ex, Ey, Ez = E
    Hx, Hy, Hz = H

    # ---- Intensities ----
    I_E = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    I_H = np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2

    # ---- Axes ----
    x = np.linspace(0, L1, Nx)
    y = np.linspace(0, L2, Ny)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ===================== E FIELD =====================
    ax = axes[0]

    im1 = ax.pcolormesh(X, Y, I_E, shading='auto', cmap=cmap)

    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        Ex.real[::stride, ::stride],
        Ey.real[::stride, ::stride],
        color='white',
        scale=20,
        width=0.003
    )

    ax.set_title(f'E-field at z = {z_value:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.set_label('|E|²')

    # ===================== H FIELD =====================
    ax = axes[1]

    im2 = ax.pcolormesh(X, Y, I_H, shading='auto', cmap=cmap)

    # ---- Normalize H vectors ----
    Hx_r = Hx.real
    Hy_r = Hy.real

    mag = np.sqrt(Hx_r**2 + Hy_r**2) + 1e-12  # avoid divide by zero

    Hx_norm = Hx_r / mag
    Hy_norm = Hy_r / mag

    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        Hx_norm[::stride, ::stride],
        Hy_norm[::stride, ::stride],
        color='white',
        scale=25,      # adjust if needed
        width=0.003
    )

    ax.set_title(f'H-field at z = {z_value:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    cbar2 = fig.colorbar(im2, ax=ax)
    cbar2.set_label('|H|²')

    plt.tight_layout()
    plt.show()








def plot_phase_refl(geometry_func, geometry_params,
               lambdas, normal_params):
    

    '''Plots phase and reflectance together for quick comparison.'''



    phis, Rs = compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    phis = np.unwrap(phis)

    fig, axs = plt.subplots(2, 1, figsize=(6,6), sharex=True)

    # Phase plot
    axs[0].plot(lambdas, phis, linewidth=2)
    axs[0].set_ylabel("Phase (rad)")
    axs[0].set_title("Reflection Phase vs Wavelength")
    axs[0].grid(True)

    # Reflectance plot
    axs[1].plot(lambdas, Rs, linewidth=2)
    axs[1].set_xlabel("Wavelength (µm)")
    axs[1].set_ylabel("Reflectance")
    axs[1].set_title("Reflectance vs Wavelength")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()