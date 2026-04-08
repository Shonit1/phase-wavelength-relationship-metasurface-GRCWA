import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from scipy.signal import savgol_filter
from config import *
from geometry_functions import *
from rcwa_machinery import *




'''
In this file you have various loss functions that you can use for optimization. They are designed to target specific phase behaviors (linear,cubic, quadratic, square-root, inverse).
There are two types of loss functions for both quadratic and cubic and both can be useful.

'''




def loss_target_slope(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    target_slope,
    slope_control=0,   # 5% tolerance
    alpha=0,             # linearity weight
    beta=0,               # reflectance weight
    save_filename="good_geometries3.txt"
):
    

    '''This gives you the target slope. I recommend starting with 0 alpha,0 beta, quickly find the slope and then reduce
     sigma and increase alpha and beta by orders of magnitude to get good geometries. '''



    #phis,Rs,Ts,Sums = compute_phase_and_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS)
    phis,Rs = compute_phase(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS)

    if phis is None:
        return 1e6

    # unwrap
    phis = np.unwrap(phis)

    # light smoothing
    phis = savgol_filter(phis, 7, 3)

    # linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    # RMS deviation from linearity
    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    Ravg = np.mean(Rs)
    #Tavg = np.mean(Ts)
    #Savg = np.mean(Sums)
    # --- normalization constants ---
    rms_target = 1e-2   # tune this

    # --- normalized terms ---
    slope_error = ((np.abs(A) - target_slope) / target_slope)**2
    rms_norm = (rms / rms_target)**2
    refl_penalty = (1 - Ravg)**2
    raise_slope = - np.abs(A)  # encourage higher slope overall, not just close to target

    # --- total loss ---
    loss = slope_control*slope_error + alpha * rms_norm + beta * refl_penalty + raise_slope

    print(
        f"Slope={A:.3f}, "
        f"Ravg={Ravg:.3f}, "
        #f"Tavg={Tavg:.3f}, "
        #f"Savg={Savg:.3f}, "

        f"Target={target_slope:.3f}, "
        f"RMS={rms:.3e}, "
        
        f"LOSS={loss:.3f}"
    )

    # 🔥 Auto-save good geometries
    if (
        #220<np.abs(A)<250 and rms<9e-1 
        800<np.abs(A)
    ):
        with open(save_filename, "a") as f:
            f.write(
                f"Slope={A}, GeometryParams={geometry_params},normalParams = {normal_params}\n"
                f"RMS={rms}\n"
                f"Ravg={Ravg}\n"
            )


        print("🔥🔥 Linear geometry SAVED 🔥🔥")    

    return loss







def loss_three_region_quadfit6_trans_freq(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_curve=1.0,
    w_fit=0,
    save_file="good_quadratic_geometries7.txt"
):

    
    '''Simplified version using slope difference. This gets you a tilted quadratic.'''
    c = 3e8

    phis, Rs = compute_phase(
        geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS
    )

    if phis is None:
        return 1e6

    phis = np.unwrap(phis)

    # ---------- Convert wavelength → frequency ----------
    lambdas_m = lambdas * 1e-6
    omegas = 2*np.pi*c / lambdas_m

    # ---------- Normalize frequency axis ----------
    omega0 = np.mean(omegas)
    x = (omegas - omega0) / omega0

    # ---------- Define Regions ----------
    mask1 = (lambdas >= 1.50) & (lambdas < 1.51)
    mask3 = (lambdas >= 1.51) & (lambdas <= 1.52)

    if not (np.any(mask1) and np.any(mask3)):
        return 1e6

    # ---------- Quadratic fits in outer regions ----------
    M1, C1 = np.polyfit(x[mask1], phis[mask1],1 )
    M3, C3 = np.polyfit(x[mask3], phis[mask3], 1)


   
    slope = abs(abs(M1) - abs(M3))

    
    # ---------- Loss ----------
    loss = (
        - w_curve * slope
        
        
        
    )

    # ---------- Save good geometries ----------
    save_condition = (
        slope > 100 
        
    )

    if save_condition:

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")
            f.write(f"S1 = {M1:.6f}\n")
            f.write(f"S3 = {M3:.6f}\n\n")

            

        print(">>> Geometry saved")

    print(
        f"S1={M1:.3f}, "
        f"S3={M3:.3f}, "
        f"slope={slope:.3f}, "
        
        f"LOSS={loss:.3f}"
    )

    return loss





def loss_three_region_polyfit_trans_freq(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_center=1.0,
    save_file="good_cubic_geometries44.txt"
):


    '''This is somewhat good for cubic geometries. We move to frequency domain in this case.'''

    c = 3e8

    phis, Rs = compute_phase(
        geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS
    )

    if phis is None:
        return 1e6

    phis = np.unwrap(phis)

    # ---------- Convert wavelength → frequency ----------
    lambdas_m = lambdas * 1e-6
    omegas = 2*np.pi*c / lambdas_m

    # ---------- Normalize frequency axis ----------
    omega0 = np.mean(omegas)
    wshift = (omegas - omega0) / omega0   # normalized frequency

    # ---------- Define Regions ----------
    mask1 = (lambdas >= 1.5)  & (lambdas < 1.508)
    mask2 = (lambdas >= 1.508) & (lambdas < 1.512)
    mask3 = (lambdas >= 1.512)   & (lambdas <= 1.52)

    if not (np.any(mask1) and np.any(mask2) and np.any(mask3)):
        return 1e6

    # ---------- Linear fits using normalized frequency ----------
    M1, _ = np.polyfit(wshift[mask1], phis[mask1], 1)
    M2, _ = np.polyfit(wshift[mask2], phis[mask2], 1)
    M3, _ = np.polyfit(wshift[mask3], phis[mask3], 1)

    # ---------- Curvature terms ----------
    '''
    # ---------- HARD constraint ----------
    if S1 * S3 > 0:
        return 1e6
    '''
    # ---------- Curvature ----------
    A1 = abs(M1)
    A2 = abs(M2)
    A3 = abs(M3)

    TARGET = 150

    curv_target_penalty = (
        abs(A1 - TARGET)+
        abs(A3 - TARGET)
    )


    

    outer_strength = min(A1, A3)
    '''
    # center flatness
    S2_penalty = abs(S2) / (A1 + A3 + 1e-9)

    # encourage opposite curvature
    opposite_reward = - (S1 * S3) / (A1*A3 + 1e-9)

    # symmetry
    antisym_penalty = abs(S1 + S3) / (A1 + A3 + 1e-9)
    '''
    loss = (
    -w_outer * outer_strength
    +A2
    + 10.0 * curv_target_penalty)
    '''
    + w_center * S2_penalty
    - 2.0 * opposite_reward
    + 0.5 * antisym_penalty
    '''
       # <-- new term

    # ---------- Save geometry if conditions satisfied ----------
    if (
    (
        A1 > A2 and A3 > A2 
    )
    
):

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")

            f.write(f"S1 = {M1:.4f}\n")
            f.write(f"S2 = {M2:.4f}\n")
            f.write(f"S3 = {M3:.4f}\n\n")

            
            

            

        print(">>> Geometry saved")

    print(
        f"M1={M1:.3f}, "
        f"M2={M2:.3f}, "
        f"M3={M3:.3f}, "
        
        
        
        f"LOSS={loss:.3f}"
    )

    return loss






def loss_dual_slope_max(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_slope=1.0,
    w_refl=10.0,
    save_file="good_cubic_geometries.txt"
):


    '''Encourages cubic-like phase via multiple slope regions.'''


    phis, Rs = compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    # unwrap phase (important for cubic fitting)
    phis = np.unwrap(phis)

    # ----------- DEFINE TWO REGIONS -----------
    region1_mask = (lambdas >= 1.49) & (lambdas < 1.495)
    region2_mask = (lambdas >= 1.495) & (lambdas < 1.5)
    region3_mask = (lambdas >= 1.5) & (lambdas < 1.505)

    if not np.any(region1_mask) or not np.any(region3_mask):
        return 1e6

    # ----------- COMPUTE SLOPES -----------
    dphi = np.gradient(phis, lambdas)

    slope1 = np.mean(np.abs(dphi[region1_mask]))
    slope2 = np.mean(np.abs(dphi[region2_mask]))
    slope3 = np.mean(np.abs(dphi[region3_mask]))

    slope_metric = np.min([slope1, slope3])
    slope_flat_penalty = slope2

    # ----------- CUBIC FIT (FULL WINDOW) -----------
    coeffs = np.polyfit(lambdas, phis, 3)
    phi_fit = np.polyval(coeffs, lambdas)

    cubic_rms = np.sqrt(np.mean((phis - phi_fit)**2))

    # ----------- REFLECTANCE -----------
    
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg) ** 2)
    

    # ----------- FINAL LOSS -----------
    loss = (
        w_slope * (-slope_metric)  +
        w_refl  * refl_penalty
    )

    print(
        f"slope1={slope1:.2e}, "
        f"slope2={slope2:.2e}, "
        f"slope3={slope3:.2e}, "
        f"min_slope={slope_metric:.2e}, "
        f"cubic_rms={cubic_rms:.4f}, "
        f"Ravg_window={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    # ----------- CONDITIONAL SAVE -----------
    if (
        loss<0
        ):
        with open(save_file, "a") as f:
            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n")
            f.write(f"slope1 = {slope1:.4e}\n")
            f.write(f"slope2 = {slope2:.4e}\n")
            f.write(f"cubic_rms = {cubic_rms:.6f}\n")
            f.write(f"Ravg_window = {Ravg:.4f}\n\n")

        print(">>> Geometry saved (meets cubic + slope criteria)")

    return loss





def loss_three_region_polyfit(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_center=1.0,
    w_refl=10.0,
    save_file="good_cubic_geometries.txt"
):



    '''This gets you phase = lambda^3 relationship, but with more control on the center region'''


    phis, Rs = compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6
    
    # unwrap phase
    phis = np.unwrap(phis)

    # ---------- Define Regions ----------
    mask1 = (lambdas >= 1.49)  & (lambdas < 1.495)
    mask2 = (lambdas >= 1.495) & (lambdas < 1.5)
    mask3 = (lambdas >= 1.5)  & (lambdas <= 1.505)

    if not (np.any(mask1) and np.any(mask2) and np.any(mask3)):
        return 1e6
    
    # ---------- Linear fits in each region ----------
    A1, B1 = np.polyfit(lambdas[mask1], phis[mask1], 1)
    A2, B2 = np.polyfit(lambdas[mask2], phis[mask2], 1)
    A3, B3 = np.polyfit(lambdas[mask3], phis[mask3], 1)

    A1 = abs(A1)
    A2 = abs(A2)
    A3 = abs(A3)

    outer_min = min(A1, A3)

    # ---------- Reflectance penalty ----------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg) ** 2)

    # ---------- Final Loss ----------
    loss = (
        -w_outer * outer_min   # maximize min(A1, A3)
        + w_center * A2        # minimize A2
        + w_refl * refl_penalty
    )

    if (
        loss<0
        ):
        with open(save_file, "a") as f:
            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n")
            f.write(f"A1 = {A1:.4e}\n")
            f.write(f"A2 = {A2:.4e}\n")
            f.write(f"A3 = {A3:.4e}\n")
            f.write(f"loss = {loss:.4e}\n")
            f.write(f"Ravg = {Ravg:.4f}\n\n")

        print(">>> Geometry saved (meets cubic + slope criteria)")


    print(
        f"A1={A1:.3e}, "
        f"A2={A2:.3e}, "
        f"A3={A3:.3e}, "
        f"min_outer={outer_min:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss





def loss_three_region_sqrt(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    lambda_c=1.497,
    w_fit=1.0,
    w_pole=3.0,
    w_refl=10.0
):

    '''Enforces √(λ − λc) behavior + strong resonance (pole) + reflectance.'''           


    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    phi = np.unwrap(phis)

    left_mask   = (lambdas >= 1.475) & (lambdas < lambda_c)
    pole_mask   = (lambdas >= lambda_c) & (lambdas <= 1.503)
    right_mask  = (lambdas > 1.503) & (lambdas <= 1.545)
    refl_mask   = lambdas > 1.505

    if not np.any(right_mask) or not np.any(pole_mask):
        return 1e6

    # ---------------- FIT REGION ----------------
    shifted = lambdas[right_mask] - lambda_c
    if np.any(shifted <= 0):
        return 1e6

    sqrt_term = np.sqrt(shifted)

    X = np.vstack([-sqrt_term, np.ones_like(sqrt_term)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, phi[right_mask], rcond=None)

    A_fit, B_fit = coeffs
    phi_fit = -A_fit * sqrt_term + B_fit

    fit_error = np.mean((phi[right_mask] - phi_fit)**2)

    # ---------------- POLE REGION ----------------
    dphi = np.gradient(phi, lambdas)
    idx_pole = np.argmin(np.abs(lambdas - lambda_c))
    pole_strength = np.abs(dphi[idx_pole])

    # ---------------- REFLECTANCE ----------------
    if np.any(refl_mask):
        R_window = Rs[refl_mask]
        refl_penalty = np.mean(np.maximum(0, 0.8 - R_window)**2)
        Ravg_window = np.mean(R_window)
    else:
        refl_penalty = 0.0
        Ravg_window = 0.0

    # ---------------- AUTO SAVE CONDITION ----------------
    if fit_error < 1e-3 and pole_strength > 1e3:

    # Combine geometry + normal parameters
        all_params = np.concatenate([
        np.array(geometry_params).flatten(),
        np.array(normal_params).flatten()
    ])

        with open("good_sqrt_resonance_solutions.txt", "ab") as f:
            np.savetxt(f,
                    all_params.reshape(1, -1),
                    fmt="%.10f")

        print("🔥 GOOD SOLUTION SAVED")
        print("fit_error =", fit_error)
        print("pole_strength =", pole_strength)


    # ---------------- FINAL LOSS ----------------
    loss = (
        w_fit  * fit_error +
        w_pole * (-pole_strength) +
        w_refl * refl_penalty
    )

    print(
        f"fit={fit_error:.2e}, "
        f"|dφ/dλ|@λc={pole_strength:.2e}, "
        f"A_fit={A_fit:.3f}, "
        f"Ravg_window={Ravg_window:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss




def loss_global_inverse(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    lambda_c=1.497,
    w_fit=10.0,
    w_pole=5,
    w_refl=10.0
):


    '''This tries to get you the true 1/x relationship with a pole, not just a local slope'''


    phis, Rs = compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    

    # ---------------------------------------------------
    # Exclude only tiny region near singularity
    # ---------------------------------------------------
    exclusion_width = 5e-4
    mask = np.abs(lambdas - lambda_c) > exclusion_width

    if np.sum(mask) < 5:
        return 1e6

    l_fit = lambdas[mask]
    phi_fit_region = phis[mask]

    shifted = l_fit - lambda_c
    inv_term = 1.0 / shifted   # preserve sign

    # ---------------------------------------------------
    # Inverse fit: phi = A/(λ-λc) + B
    # ---------------------------------------------------
    X = np.vstack([inv_term, np.ones_like(inv_term)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, phi_fit_region, rcond=None)

    A_fit, B_fit = coeffs

    # Evaluate model on full spectrum
    phi_model = A_fit / (lambdas - lambda_c) + B_fit

    # Compute fit error on masked region only
    fit_error = np.mean((phi_fit_region - (A_fit / shifted + B_fit))**2)

    # ---------------------------------------------------
    # Pole strength (global derivative)
    # ---------------------------------------------------
    dphi = np.gradient(phis, lambdas)
    idx_pole = np.argmin(np.abs(lambdas - lambda_c))
    pole_strength = np.abs(dphi[idx_pole])

    # ---------------------------------------------------
    # Reflectance penalty (global)
    # ---------------------------------------------------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg)**2)

    # ---------------------------------------------------
    # Auto-save condition (true inverse-like)
    # ---------------------------------------------------
    ss_res = np.sum((phi_fit_region - (A_fit / shifted + B_fit))**2)
    ss_tot = np.sum((phi_fit_region - np.mean(phi_fit_region))**2)
    r2_inverse = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    if fit_error < 0.1 and pole_strength > 50 :

        all_params = np.concatenate([
            np.array(geometry_params).flatten(),
            np.array(normal_params).flatten(),
            np.array([
                fit_error,
                pole_strength,
                A_fit
            ])
        ])

        with open("good_inverse_resonance_solutions.txt", "ab") as f:
            np.savetxt(
                f,
                all_params.reshape(1, -1),
                fmt="%.10f"
            )

        print("🔥 SAVED")
        print("fit_error =", fit_error)
        print("pole_strength =", pole_strength)

    # ---------------------------------------------------
    # Final loss
    # ---------------------------------------------------
    loss = (
        w_fit  * fit_error
        - w_pole * pole_strength
        + w_refl * refl_penalty
    )

    print(
        f"fit={fit_error:.2e}, "
        f"|dφ/dλ|={pole_strength:.2e}, "
        f"A_fit={A_fit:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss






def loss_shifted_sqrt_target(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    lambda_c=1.497
):


    '''Fits phase to √(λ − λc), i.e., resonance-like shifted behavior.'''



    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    phi = np.unwrap(phis)
    Ravg = np.mean(Rs)

    shifted = lambdas - lambda_c

    if np.any(shifted <= 0):
        return 1e6

    sqrt_term = np.sqrt(shifted)

    # Linear fit: phi = -A*sqrt_term + B
    X = np.vstack([-sqrt_term, np.ones_like(sqrt_term)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, phi, rcond=None)

    A_fit = coeffs[0]
    B_fit = coeffs[1]

    phi_fit = -A_fit * sqrt_term + B_fit

    rms_fit = np.sqrt(np.mean((phi - phi_fit)**2))

    # Reflectance penalty
    refl_penalty = np.maximum(0.99 - Ravg, 0)**2

    loss = rms_fit + 10.0 * refl_penalty

    print(
        f"RMS_fit={rms_fit:.3e}, "
        f"A_fit={A_fit:.3f}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss





def loss_max_slope_only(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    alpha=0.0
):
    


    '''Maximizes slope of phase vs wavelength.'''

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    # unwrap
    phis = np.unwrap(phis)

    # optional light smoothing (keep minimal)
    phis = savgol_filter(phis, 7, 3)

    # linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    # RMS deviation from linear
    rms = np.sqrt(np.mean((phis - phi_fit)**2))

    # reflectance penalty
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)

    # LOSS: maximize |slope|
    loss = -np.abs(A) + alpha * rms + 0 * R_penalty

    print(
        f"Slope={A:.3f}, "
        f"RMS={rms:.3e}, "
        f"Ravg={np.mean(Rs):.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss







def loss_three_region_quadfit5_trans_freq(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_curve=1.0,
    w_fit=0,
    save_file="good_quadratic_geometries7.txt"
):

    
    '''This is based on curvature.'''
    c = 3e8

    phis, Rs = compute_phase(
        geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS
    )

    if phis is None:
        return 1e6

    phis = np.unwrap(phis)

    # ---------- Convert wavelength → frequency ----------
    lambdas_m = lambdas * 1e-6
    omegas = 2*np.pi*c / lambdas_m

    # ---------- Normalize frequency axis ----------
    omega0 = np.mean(omegas)
    x = (omegas - omega0) / omega0

    # ---------- Define Regions ----------
    mask1 = (lambdas >= 1.50) & (lambdas < 1.51)
    mask3 = (lambdas >= 1.51) & (lambdas <= 1.52)

    if not (np.any(mask1) and np.any(mask3)):
        return 1e6

    # ---------- Quadratic fits in outer regions ----------
    S1, M1, C1 = np.polyfit(x[mask1], phis[mask1], 2)
    S3, M3, C3 = np.polyfit(x[mask3], phis[mask3], 2)

    # ---------- Predicted phase ----------
    phi_fit1 = S1*x[mask1]**2 + M1*x[mask1] + C1
    phi_fit3 = S3*x[mask3]**2 + M3*x[mask3] + C3

    # ---------- Mean square errors ----------
    mse1 = np.mean((phi_fit1 - phis[mask1])**2)
    mse3 = np.mean((phi_fit3 - phis[mask3])**2)

    mse_total = mse1 + mse3

    # ---------- Curvature (quadratic strength) ----------
    curvature = abs(S1) + abs(S3)

    

    

    # ---------- Loss ----------
    loss = (
        - w_curve * curvature
        + w_fit * mse_total
        
        
    )

    # ---------- Save good geometries ----------
    save_condition = (
        (np.abs(S1) >np.abs(S3)) 
        and (mse_total < 0.1) and (curvature > 1000)
    )

    if save_condition:

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")

            f.write(f"S1 = {S1:.6f}\n")
            f.write(f"S3 = {S3:.6f}\n\n")

            f.write(f"MSE1 = {mse1:.6e}\n")
            f.write(f"MSE3 = {mse3:.6e}\n")
            f.write(f"MSE_total = {mse_total:.6e}\n\n")

        print(">>> Geometry saved")

    print(
        f"S1={S1:.3f}, "
        f"S3={S3:.3f}, "
        f"curvature={curvature:.3f}, "
        f"MSE={mse_total:.3e}, "
        f"LOSS={loss:.3f}"
    )

    return loss




