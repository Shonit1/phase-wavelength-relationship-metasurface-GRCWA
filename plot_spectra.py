import numpy as np
import matplotlib.pyplot as plt
from config import *
from geometry_functions import *
from rcwa_machinery import *


def plot_phase(geometry_func, geometry_params,
               lambdas, normal_params):

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
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
    ax[1].set_ylabel("Reflectance")
    ax[1].set_ylim(0,1.05)
    ax[1].set_title("Reflectance")

    # Transmission
    ax[2].plot(lambdas, Ts, linewidth=2)
    ax[2].set_ylabel("Transmission")
    ax[2].set_xlabel("Wavelength (µm)")
    ax[2].set_ylim(0,1.05)
    ax[2].set_title("Transmission")

    for a in ax:
        a.grid(True)

    plt.tight_layout()
    plt.show()





def plot_energy_balance(geometry_func, geometry_params,
                        lambdas, normal_params):

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

    plt.figure(figsize=(6, 6))

    plt.pcolormesh(x, z, I, shading="auto", cmap="inferno")
    plt.colorbar(label="|E|²")

    for zb in layer_bounds:
        plt.axhline(zb, color="white", lw=0.6, alpha=0.6)

    plt.xlabel("x (µm)")
    plt.ylabel("z (µm)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()