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








def coarse_sweep_1D(geometry_func,
                    param_name,
                    param_values,
                    base_params,
                    lambdas,
                    normal_params):

    """
    Sweeps a single parameter while keeping others fixed.
    Plots reflectance spectrum for each value.
    """

    plt.figure(figsize=(7,5))

    for val in param_values:
        params = base_params.copy()
        params[param_name] = val

        phis, Rs, Ts, sums = compute_phase_and_reflectance(
            geometry_func,
            params,
            lambdas,
            normal_params,
            DBR_PAIRS
        )

        plt.plot(lambdas, Rs, label=f"{param_name}={val:.3f}")

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Reflectance")
    plt.title(f"Coarse Sweep of {param_name}")
    plt.ylim(0,1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()







def sweep_cylinder_radius(geometry_func,lambdas, normal_params,geometry_params, eps):

    

    base_params = {
        "r": 0.2,
        "eps": eps
    }

    r_values = np.linspace(0.1, 0.35, 6)

    coarse_sweep_1D(
        geometry_func,
        "r",
        r_values,
        base_params,
        lambdas,
        normal_params
    )






def sweep_ellipse_rx(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "rx": 0.2,
        "ry": 0.15,
        "eps": eps
    }

    rx_values = np.linspace(0.1, 0.35, 6)

    coarse_sweep_1D(
        geometry_func,
        "rx",
        rx_values,
        base_params,
        lambdas,
        normal_params
    )





def sweep_ring_outer(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "r_inner": 0.15,
        "r_outer": 0.25,
        "eps": eps
    }

    r_values = np.linspace(0.2, 0.4, 6)

    coarse_sweep_1D(
        geometry_func,
        "r_outer",
        r_values,
        base_params,
        lambdas,
        normal_params
    )





def sweep_double_ring(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "r1": 0.1,
        "r2": 0.18,
        "r3": 0.25,
        "r4": 0.35,
        "eps": eps
    }

    r_values = np.linspace(0.08, 0.18, 5)

    coarse_sweep_1D(
        geometry_func,
        "r1",
        r_values,
        base_params,
        lambdas,
        normal_params
    )







def sweep_cross_width(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "width": 0.1,
        "eps": eps
    }

    width_values = np.linspace(0.05, 0.25, 6)

    coarse_sweep_1D(
        geometry_func,
        "width",
        width_values,
        base_params,
        lambdas,
        normal_params
    )





def sweep_square_frame(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "w_outer": 0.35,
        "w_inner": 0.2,
        "eps": eps
    }

    outer_values = np.linspace(0.25, 0.45, 6)

    coarse_sweep_1D(
        geometry_func,
        "w_outer",
        outer_values,
        base_params,
        lambdas,
        normal_params
    )






def sweep_diagonal_period(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "period": 0.3,
        "eps": eps
    }

    period_values = np.linspace(0.2, 0.6, 6)

    coarse_sweep_1D(
        geometry_func,
        "period",
        period_values,
        base_params,
        lambdas,
        normal_params
    )






def sweep_split_gap(geometry_func,lambdas, normal_params, geometry_params, eps):

    base_params = {
        "r": 0.25,
        "gap": 0.03,
        "eps": eps
    }

    gap_values = np.linspace(0.01, 0.08, 6)

    coarse_sweep_1D(
        geometry_func,
        "gap",
        gap_values,
        base_params,
        lambdas,
        normal_params
    )



def sweep_double_cylinder_shift(geometry_func,lambdas, normal_params , geometry_params, eps):

    

    base_params = {
        "r1": 0.22,
        "r2": 0.18,
        "shift": 0.05,
        "eps": eps
    }

    shift_values = np.linspace(0.01, 0.12, 6)

    coarse_sweep_1D(
        geometry_func,
        "shift",
        shift_values,
        base_params,
        lambdas,
        normal_params
    )






def check_convergence_linear(geometry_params,normal_params,geometry_func,lambdas, nG_list):

    results = []

    for nG in nG_list:

        global N_G
        N_G = nG

        phis, Rs, Ts, sums = compute_phase_and_reflectance(
            geometry_func,
            geometry_params,
            lambdas,
            normal_params,
            DBR_PAIRS
        )

        if phis is None:
            print(f"nG={nG} → Simulation failed")
            continue

        # unwrap phase
        phis = np.unwrap(phis)

        # linear fit
        A, B = np.polyfit(lambdas, phis, 1)
        phi_fit = A * lambdas + B

        # RMS deviation from linear
        rms = np.sqrt(np.mean((phis - phi_fit)**2))

        # average reflectance
        R_avg = np.mean(Rs)

        results.append((nG, A, rms, R_avg))

        print(
            f"nG={nG}, "
            f"slope={A:.3f}, "
            f"RMS={rms:.4f}, "
            f"Ravg={R_avg:.4f}"
        )

    return results






def check_convergence_cubic(
    geometry_params,
    normal_params,
    geometry_func,
    lambdas,
    nG_list
):

    results = []

    for nG in nG_list:

        global N_G
        N_G = nG

        phis, Rs, Ts, sums = compute_phase_and_reflectance(
            geometry_func,
            geometry_params,
            lambdas,
            normal_params,
            DBR_PAIRS
        )

        if phis is None:
            print(f"nG={nG} → Simulation failed")
            continue

        # unwrap phase
        phis = np.unwrap(phis)

        # cubic polynomial fit
        coeffs = np.polyfit(lambdas, phis, 3)
        A, B, C, D = coeffs  # cubic, quadratic, linear, constant

        phi_fit = np.polyval(coeffs, lambdas)

        # RMS deviation from cubic fit
        rms = np.sqrt(np.mean((phis - phi_fit)**2))

        # average reflectance
        R_avg = np.mean(Rs)

        results.append((nG, A, B, C, rms, R_avg))

        print(
            f"nG={nG}, "
            f"CubicCoeff={A:.6e}, "
            f"QuadCoeff={B:.6e}, "
            f"LinearCoeff={C:.6e}, "
            f"RMS={rms:.4f}, "
            f"Ravg={R_avg:.4f}"
        )

    return results
