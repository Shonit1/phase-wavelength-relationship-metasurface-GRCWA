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









# Linear Relationship


def loss_linear(geometry_func, geometry_params,
                normal_params, lambdas,
                alpha=10.0, beta=50.0, gamma=200.0):

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func, geometry_params,
        lambdas, normal_params, DBR_PAIRS
    )

    # Linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    # RMS deviation from linear
    rms = np.sqrt(np.mean((phis - phi_fit)**2))

    # Reflectance penalty
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)

    # Broadband reflectance reward
    R_avg = np.mean(Rs)

    loss = alpha*rms + beta*R_penalty - gamma*R_avg

    print(f"A={A:.3f}, RMS={rms:.3e}, Ravg={R_avg:.3f}, LOSS={loss:.3f}")
    return loss










def loss_sqrt_lambda(geometry_func, geometry_params,
                     normal_params, lambdas,
                     alpha=10.0, beta=50.0, gamma=200.0):

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func, geometry_params,
        lambdas, normal_params, DBR_PAIRS
    )

    sqrt_lam = np.sqrt(lambdas)

    A, B = np.polyfit(sqrt_lam, phis, 1)
    phi_fit = A * sqrt_lam + B

    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)
    R_avg = np.mean(Rs)

    loss = alpha*rms + beta*R_penalty - gamma*R_avg

    print(f"A={A:.3f}, RMS={rms:.3e}, Ravg={R_avg:.3f}, LOSS={loss:.3f}")
    return loss





def loss_lambda_square(geometry_func, geometry_params,
                       normal_params, lambdas,
                       alpha=10.0, beta=50.0, gamma=200.0):

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func, geometry_params,
        lambdas, normal_params, DBR_PAIRS
    )

    lam_sq = lambdas**2

    A, B = np.polyfit(lam_sq, phis, 1)
    phi_fit = A * lam_sq + B

    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)
    R_avg = np.mean(Rs)

    loss = alpha*rms + beta*R_penalty - gamma*R_avg

    print(f"A={A:.3f}, RMS={rms:.3e}, Ravg={R_avg:.3f}, LOSS={loss:.3f}")
    return loss








def loss_inverse_lambda(geometry_func, geometry_params,
                        normal_params, lambdas,
                        alpha=10.0, beta=50.0, gamma=200.0):

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func, geometry_params,
        lambdas, normal_params, DBR_PAIRS
    )

    inv_lam = 1.0 / lambdas

    A, B = np.polyfit(inv_lam, phis, 1)
    phi_fit = A * inv_lam + B

    rms = np.sqrt(np.mean((phis - phi_fit)**2))
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)
    R_avg = np.mean(Rs)

    loss = alpha*rms + beta*R_penalty - gamma*R_avg

    print(f"A={A:.3f}, RMS={rms:.3e}, Ravg={R_avg:.3f}, LOSS={loss:.3f}")
    return loss







def loss_sqrt_resonance_forced(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas
):

    phis, Rs, Ts, sums = compute_phase_and_reflectance(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    # unwrap
    phis = np.unwrap(phis)

    # smooth to remove numerical spikes
    phis = savgol_filter(phis, 7, 3)

    lam_center = np.mean(lambdas)
    lam_scale = (np.max(lambdas) - np.min(lambdas)) / 2.0
    lam_norm = (lambdas - lam_center) / lam_scale

    phi_target = 50.0 * np.sqrt(lambdas)
    B_opt = np.mean(phis - phi_target)
    phi_target += B_opt

    # derivatives
    dphi = np.gradient(phis, lam_norm)
    d2phi = np.gradient(dphi, lam_norm)

    dphi_target = np.gradient(phi_target, lam_norm)
    d2phi_target = np.gradient(dphi_target, lam_norm)

    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)

    rms_phi = np.sqrt(np.mean((norm(phis) - norm(phi_target))**2))
    rms_curv = np.sqrt(np.mean((norm(d2phi) - norm(d2phi_target))**2))

    # normalized curvature sign penalty
    d2phi_norm = d2phi / (np.max(np.abs(d2phi)) + 1e-12)
    curv_sign_penalty = np.mean(np.maximum(0, d2phi_norm)**2)

    # reflectance constraint
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)

    loss = (
        3.0 * rms_phi +
        2.0 * rms_curv +
        3.0 * curv_sign_penalty +
        50.0 * R_penalty
    )

    print(
        f"RMS_phi={rms_phi:.3e}, "
        f"RMS_curv={rms_curv:.3e}, "
        f"CurvSignPenalty={curv_sign_penalty:.3e}, "
        f"Ravg={np.mean(Rs):.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss




def loss_sqrt_curvature_strong(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS
):
    """
    Enforces strong sqrt(lambda)-like curvature
    using second-derivative shape matching.
    """

    # -------------------------------------------------
    # 1) Compute phase + reflectance
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2) Normalize wavelength to [0,1]
    # -------------------------------------------------
    lam_norm = (lambdas - lambdas.min()) / (lambdas.max() - lambdas.min())

    # Avoid zero at left boundary (important for λ^(3/2))
    lam_norm = lam_norm + 1e-6

    # -------------------------------------------------
    # 3) Compute numerical derivatives
    # -------------------------------------------------
    d1 = np.gradient(phi, lam_norm)
    d2 = np.gradient(d1, lam_norm)

    # -------------------------------------------------
    # 4) Target sqrt(lambda) curvature
    # f''(λ) = -1/(4 λ^(3/2))
    # -------------------------------------------------
    target_d2 = -1.0 / (4.0 * lam_norm**(3/2))

    # -------------------------------------------------
    # 5) Normalize curvature shapes (remove magnitude cheating)
    # -------------------------------------------------
    d2_norm = d2 / (np.max(np.abs(d2)) + 1e-8)
    target_norm = target_d2 / (np.max(np.abs(target_d2)) + 1e-8)

    rms_curv_shape = np.sqrt(np.mean((d2_norm - target_norm)**2))

    # -------------------------------------------------
    # 6) Penalize weak curvature (avoid linear solutions)
    # -------------------------------------------------
    curvature_strength = np.mean(np.abs(d2))
    weak_curv_penalty = 1.0 / (curvature_strength + 1e-6)

    # -------------------------------------------------
    # 7) Penalize curvature sign flips (must stay negative)
    # -------------------------------------------------
    sign_penalty = np.mean(np.maximum(0, d2))  # positive curvature penalized

    # -------------------------------------------------
    # 8) Reflectance penalty
    # -------------------------------------------------
    Ravg = np.mean(Rs)
    reflect_penalty = np.maximum(0, 0.95 - Ravg)

    # -------------------------------------------------
    # 9) Final loss
    # -------------------------------------------------
    loss = (
        5.0 * rms_curv_shape +      # shape match
        0.5 * weak_curv_penalty +   # enforce strength
        2.0 * sign_penalty +        # enforce negative curvature
        10.0 * reflect_penalty      # enforce high R
    )

    print(f"CurvShape={rms_curv_shape:.3e}, "
          f"WeakPen={weak_curv_penalty:.3e}, "
          f"SignPen={sign_penalty:.3e}, "
          f"Ravg={Ravg:.3f}, "
          f"LOSS={loss:.3f}")

    return loss








def loss_fixed_sqrt_target(
        geometry_func,
        geometry_params,
        normal_params,
        lambdas,
        DBR_PAIRS,
        A_target=10.0
    ):

    # ---------------------------------------
    # 1. Compute phase + reflectance
    # ---------------------------------------

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

    # ---------------------------------------
    # 2. Build target sqrt curve
    #    phi_target = -A*sqrt(lambda) + B
    #    Solve optimal B analytically
    # ---------------------------------------

    sqrt_lambda = np.sqrt(lambdas)

    # Optimal B that minimizes RMS
    B_opt = np.mean(phi + A_target * sqrt_lambda)

    phi_target = -A_target * sqrt_lambda + B_opt

    # ---------------------------------------
    # 3. RMS error to target curve
    # ---------------------------------------

    rms_fit = np.sqrt(np.mean((phi - phi_target)**2))

    # ---------------------------------------
    # 4. Reflectance penalty
    # ---------------------------------------

    Ravg = np.mean(Rs)

    R_penalty = 50.0 * max(0, 0.98 - Ravg)**2

    # ---------------------------------------
    # 5. Curvature strength penalty
    #    (avoid linear collapse)
    # ---------------------------------------

    d1 = np.gradient(phi, lambdas)
    d2 = np.gradient(d1, lambdas)

    mean_abs_d2 = np.mean(np.abs(d2))

    # Enforce minimum curvature strength
    curvature_penalty = 5.0 * max(0, 0.5 - mean_abs_d2)**2

    # ---------------------------------------
    # 6. Total loss
    # ---------------------------------------

    loss = rms_fit + R_penalty + curvature_penalty

    print(f"RMS_fit={rms_fit:.3e}, "
          f"CurvStrength={mean_abs_d2:.3e}, "
          f"Ravg={Ravg:.3f}, "
          f"LOSS={loss:.3f}")

    return loss









def loss_shifted_sqrt_target(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    lambda_c=1.497
):

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



'''This gives you maximum slope relationship'''

def loss_max_slope_only(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    alpha=0.0
):

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




'''This gives you square root relationship'''


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








'''This sometimes gets you phase = lambda^3 relationship'''



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

'''This sometimes gets you phase = lambda^3 relationship, but with more control on the center region'''


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













def loss_dual_slope_strict(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_slope=1.0,
    w_refl=10.0,
    penalty_same_sign=200.0
):

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

    region1_mask = (lambdas >= 1.49) & (lambdas < 1.500)
    region2_mask = (lambdas >= 1.505) & (lambdas < 1.515)

    if not np.any(region1_mask) or not np.any(region2_mask):
        return 1e6

    dphi = np.gradient(phi, lambdas)

    slope1 = np.mean(dphi[region1_mask])
    slope2 = np.mean(dphi[region2_mask])

    # ---------------- SIGN CHECK ----------------
    if slope1 * slope2 > 0:
        # Same sign → strong penalty
        loss = penalty_same_sign + abs(slope1) + abs(slope2)
        print(
        f"slope1={slope1:.2e}, "
        f"slope2={slope2:.2e}, "
        f"same_sign_penalty={penalty_same_sign:.2f} ,"
        f"LOSS={loss:.3f}"
    )
        return loss

    # Opposite signs → reward strong slopes
    slope_metric = np.min([abs(slope1), abs(slope2)])

    # ---------------- REFLECTANCE ----------------
    refl_mask = lambdas > 1.505

    if np.any(refl_mask):
        R_window = Rs[refl_mask]
        refl_penalty = np.mean(np.maximum(0, 0.8 - R_window)**2)
        Ravg_window = np.mean(R_window)
    else:
        refl_penalty = 0.0
        Ravg_window = 0.0

    loss = (
        -w_slope * slope_metric +
        w_refl * refl_penalty
    )

    print(
        f"slope1={slope1:.2e}, "
        f"slope2={slope2:.2e}, "
        f"min|slope|={slope_metric:.2e}, "
        f"Ravg_window={Ravg_window:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss





def loss_max_slope_positive(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    alpha=0.0
):

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

    # light smoothing
    phis = savgol_filter(phis, 7, 3)

    # linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    # RMS deviation from linear
    rms = np.sqrt(np.mean((phis - phi_fit)**2))

    # reflectance penalty
    R_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)

    # ---------------- AUTO SAVE CONDITION ----------------
    if A > 3e2:
        all_params = np.concatenate([
            np.array(geometry_params).flatten(),
            np.array(normal_params).flatten()
        ])

        with open("positive_large_slope_solutions.txt", "ab") as f:
            np.savetxt(
                f,
                all_params.reshape(1, -1),
                fmt="%.10f"
            )

        print("🔥 SAVED: slope > 1e3 and positive")
        print(f"Slope = {A:.3f}")

    # LOSS: maximize |slope|
    loss = -np.abs(A) + alpha * rms

    print(
        f"Slope={A:.3f}, "
        f"RMS={rms:.3e}, "
        f"Ravg={np.mean(Rs):.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss





def loss_three_region_inverse(
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

    # ---------------- REGION MASKS ----------------
    left_mask   = (lambdas >= 1.475) & (lambdas < lambda_c)
    pole_mask   = (lambdas >= lambda_c) & (lambdas <= 1.499)
    right_mask  = (lambdas > 1.497+0.0005) & (lambdas <= 1.545)
    refl_mask   = lambdas > 1.505

    if not np.any(right_mask) or not np.any(pole_mask):
        return 1e6

    # ---------------- FIT REGION ----------------
    shifted = lambdas[right_mask] - lambda_c

    # prevent division by zero or negative region
    if np.any(shifted <= 0):
        return 1e6

    inv_term = 1.0 / shifted

    # Linear least squares fit:
    # phi ≈ A * (1/(λ-λc)) + B
    X = np.vstack([inv_term, np.ones_like(inv_term)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, phi[right_mask], rcond=None)

    A_fit, B_fit = coeffs
    phi_fit = A_fit * inv_term + B_fit

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
    if fit_error < 9e-3 :

        all_params = np.concatenate([
            np.array(geometry_params).flatten(),
            np.array(normal_params).flatten()
        ])

        with open("good_inverse_resonance_solutions.txt", "ab") as f:
            np.savetxt(
                f,
                all_params.reshape(1, -1),
                fmt="%.10f"
            )

        print("🔥 GOOD INVERSE-POLE SOLUTION SAVED")
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










def loss_inverse_pole_strict(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    lambda_c=1.497,
    w_fit=10.0,
    w_pole=1.0,
    w_refl=10.0
):

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

    # ---------------- REGION MASKS ----------------
    right_mask = (lambdas > lambda_c) & (lambdas <= 1.545)
    refl_mask  = lambdas > 1.505

    if not np.any(right_mask):
        return 1e6

    shifted = lambdas[right_mask] - lambda_c

    if np.any(shifted <= 0):
        return 1e6

    inv_term = 1.0 / shifted

    # ---------------- INVERSE FIT ----------------
    X = np.vstack([inv_term, np.ones_like(inv_term)]).T
    coeffs, _, _, _ = np.linalg.lstsq(X, phi[right_mask], rcond=None)

    A_fit, B_fit = coeffs
    phi_fit = A_fit * inv_term + B_fit

    fit_error = np.mean((phi[right_mask] - phi_fit)**2)

    # ---------------- R² CHECK ----------------
    phi_actual = phi[right_mask]
    ss_res = np.sum((phi_actual - phi_fit)**2)
    ss_tot = np.sum((phi_actual - np.mean(phi_actual))**2)

    if ss_tot == 0:
        return 1e6

    r2_inverse = 1 - ss_res / ss_tot

    # ---------------- POLE STRENGTH ----------------
    dphi = np.gradient(phi, lambdas)
    idx_c = np.argmin(np.abs(lambdas - lambda_c))
    pole_strength = np.abs(dphi[idx_c])

    # ---------------- GLOBAL CONSISTENCY ----------------
    dphi_window = np.gradient(phi[right_mask], lambdas[right_mask])
    A_local = -dphi_window * (shifted**2)

    consistency_std  = np.std(A_local)
    consistency_mean = np.mean(np.abs(A_local)) + 1e-6
    relative_consistency = consistency_std / consistency_mean

    # ---------------- REFLECTANCE ----------------
    if np.any(refl_mask):
        R_window = Rs[refl_mask]
        refl_penalty = np.mean(np.maximum(0, 0.8 - R_window)**2)
        Ravg_window = np.mean(R_window)
    else:
        refl_penalty = 0.0
        Ravg_window = 0.0

    # ---------------- SAVE CONDITION ----------------
    if (
        fit_error < 1e-3
        and pole_strength > 1e2
        and r2_inverse > 0.995
        and relative_consistency < 0.2
        and np.abs(A_fit) > 1e-3
        and Ravg_window > 0.9
    ):

        all_params = np.concatenate([
            np.array(geometry_params).flatten(),
            np.array(normal_params).flatten()
        ])

        with open("true_inverse_pole_solutions.txt", "ab") as f:
            np.savetxt(
                f,
                all_params.reshape(1, -1),
                fmt="%.10f"
            )

        print("🔥 TRUE 1/x GEOMETRY SAVED")
        print(f"A_fit={A_fit:.4e}")
        print(f"R²={r2_inverse:.5f}")
        print(f"Consistency={relative_consistency:.3f}")
        print(f"Slope={pole_strength:.2f}")

    # ---------------- FINAL LOSS ----------------
    loss = (
    w_fit * fit_error
    - w_pole * pole_strength
    + w_refl * refl_penalty
    + 50.0 * (1 - r2_inverse)
    + 20.0 * relative_consistency
)


    print(
        f"fit={fit_error:.2e}, "
        f"|dφ/dλ|={pole_strength:.2e}, "
        f"A_fit={A_fit:.3e}, "
        f"R²={r2_inverse:.4f}, "
        f"consistency={relative_consistency:.3f}, "
        f"Ravg={Ravg_window:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss




'''This tries to get you the true 1/x relationship with a pole, not just a local slope'''

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










'''failed'''



def loss_dual_inverse(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    lambda_c1=1.50,
    lambda_c2=1.53,
    w_fit=100.0,
    w_pole=5.0,
    w_refl=10.0
):

    phis, Rs = compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    # =====================================================
    # Define fitting windows
    # =====================================================

    # Left inverse window
    mask_left = (lambdas >= 1.49) & (lambdas <= 1.51)

    # Right inverse window
    mask_right = (lambdas >= 1.52) & (lambdas <= 1.54)

    # Exclude tiny region around singularities
    exclusion = 5e-4
    mask_left &= np.abs(lambdas - lambda_c1) > exclusion
    mask_right &= np.abs(lambdas - lambda_c2) > exclusion

    if np.sum(mask_left) < 5 or np.sum(mask_right) < 5:
        return 1e6

    # =====================================================
    # -------- LEFT FIT  A1/(λ-λc1) + B1 ------------------
    # =====================================================

    l_left = lambdas[mask_left]
    phi_left = phis[mask_left]

    shifted_left = l_left - lambda_c1
    inv_left = 1.0 / shifted_left

    X1 = np.vstack([inv_left, np.ones_like(inv_left)]).T
    coeffs1, _, _, _ = np.linalg.lstsq(X1, phi_left, rcond=None)
    A1, B1 = coeffs1

    phi_model_left = A1 / shifted_left + B1
    fit_error_left = np.mean((phi_left - phi_model_left) ** 2)

    # R² for left
    ss_res1 = np.sum((phi_left - phi_model_left) ** 2)
    ss_tot1 = np.sum((phi_left - np.mean(phi_left)) ** 2)
    r2_left = 1 - ss_res1 / ss_tot1 if ss_tot1 > 0 else 0


    # =====================================================
    # -------- RIGHT FIT  A2/(λ-λc2) + B2 -----------------
    # =====================================================

    l_right = lambdas[mask_right]
    phi_right = phis[mask_right]

    shifted_right = l_right - lambda_c2
    inv_right = 1.0 / shifted_right

    X2 = np.vstack([inv_right, np.ones_like(inv_right)]).T
    coeffs2, _, _, _ = np.linalg.lstsq(X2, phi_right, rcond=None)
    A2, B2 = coeffs2

    phi_model_right = A2 / shifted_right + B2
    fit_error_right = np.mean((phi_right - phi_model_right) ** 2)

    # R² for right
    ss_res2 = np.sum((phi_right - phi_model_right) ** 2)
    ss_tot2 = np.sum((phi_right - np.mean(phi_right)) ** 2)
    r2_right = 1 - ss_res2 / ss_tot2 if ss_tot2 > 0 else 0


    # =====================================================
    # Pole strengths (slope magnitude at both poles)
    # =====================================================

    dphi = np.gradient(phis, lambdas)

    idx1 = np.argmin(np.abs(lambdas - lambda_c1))
    idx2 = np.argmin(np.abs(lambdas - lambda_c2))

    pole_strength1 = np.abs(dphi[idx1])
    pole_strength2 = np.abs(dphi[idx2])


    # =====================================================
    # Reflectance penalty
    # =====================================================

    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg) ** 2)


    # =====================================================
    # AUTO SAVE CONDITION
    # =====================================================

    if (
        fit_error_left < 0.05
        and fit_error_right < 0.05
        and pole_strength1 > 5
        and pole_strength2 > 5
    ):

        all_params = np.concatenate([
            np.array(geometry_params).flatten(),
            np.array(normal_params).flatten(),
            np.array([
                fit_error_left,
                fit_error_right,
                pole_strength1,
                pole_strength2,
                A1,
                A2
            ])
        ])

        with open("good_dual_inverse_solutions.txt", "ab") as f:
            np.savetxt(f, all_params.reshape(1, -1), fmt="%.10f")

        print("🔥🔥 DUAL POLE SAVED 🔥🔥")
        print("Left fit =", fit_error_left)
        print("Right fit =", fit_error_right)


    # =====================================================
    # FINAL LOSS
    # =====================================================

    # ---------------------------------------------------
# Balanced enforcement
# ---------------------------------------------------

    fit_total = max(fit_error_left, fit_error_right)
    pole_total = min(pole_strength1, pole_strength2)

    loss = (
        w_fit * fit_total
        - w_pole * pole_total
        + w_refl * refl_penalty
    )


    print(
        f"Lfit={fit_error_left:.2e}, "
        f"Rfit={fit_error_right:.2e}, "
        f"Pole1={pole_strength1:.2e}, "
        f"Pole2={pole_strength2:.2e}, "
        f"LOSS={loss:.3f}"
    )

    return loss




def loss_target_slope(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    target_slope,
    slope_tolerance=0.05,   # 5% tolerance
    alpha=10.0,             # linearity weight
    beta=5.0,               # reflectance weight
    save_filename="good_geometries.txt"
):

    phis, Rs = compute_phase(
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

    # light smoothing
    phis = savgol_filter(phis, 7, 3)

    # linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    # RMS deviation from linearity
    rms = np.sqrt(np.mean((phis - phi_fit)**2))

    # reflectance penalty
    R_avg = np.mean(Rs)
    R_penalty = np.maximum(0, 0.8 - R_avg)**2

    # slope matching term
    slope_error = (np.abs(A) - target_slope)**2

    # total loss
    loss = slope_error + alpha * rms + beta * R_penalty

    print(
        f"Slope={A:.3f}, "
        f"Target={target_slope:.3f}, "
        f"RMS={rms:.3e}, "
        f"Ravg={R_avg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    # 🔥 Auto-save good geometries
    if (
        np.abs(A)>1e2 and rms<3e-2 and R_avg > 0.8
    ):
        with open(save_filename, "a") as f:
            f.write(
                f"Slope={A}, GeometryParams={geometry_params},normalParams = {normal_params}\n"
                f"RMS={rms}\n"
            )


        print("🔥🔥 Linear geometry SAVED 🔥🔥")    

    return loss







'''This lets you get geometries with random slope using the if condition'''

def loss_max_slope_only(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    alpha=100.0,
    save_filename="good_geometries.txt"
):

    phis, Rs= compute_phase(
        geometry_func,
        geometry_params,
        lambdas,
        normal_params,
        DBR_PAIRS
    )

    if phis is None:
        return 1e6

    
    

    # optional light smoothing (keep minimal)
    phis = savgol_filter(phis, 7, 3)

    # linear fit
    A, B = np.polyfit(lambdas, phis, 1)
    phi_fit = A * lambdas + B

    # RMS deviation from linear
    rms = np.sqrt(np.mean((phis - phi_fit)**2))

    # reflectance penalty
    R_avg = np.mean(Rs)
    R_penalty = np.maximum(0, 0.8 - R_avg)**2

    # LOSS: maximize |slope|
    loss = -np.abs(A) + alpha * rms + 0 * R_penalty


    if (
        20>np.abs(A)>10 and rms<3e-2 and R_avg > 0.8
    ):
        with open(save_filename, "a") as f:
            f.write(
                f"Slope={A}, GeometryParams={geometry_params},normalParams = {normal_params}\n"
                f"RMS={rms}\n"
                f"Ravg={R_avg}\n"
            )



    print(
        f"Slope={A:.3f}, "
        f"RMS={rms:.3e}, "
        f"Ravg={np.mean(Rs):.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss











def loss_three_region_polyfit_trans(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_center=1.0,
    w_trans=10.0,
    save_file="good_cubic_geometries.txt"
):

    phis, Rs,Ts,sum = compute_phase_and_reflectance(geometry_func, geometry_params, lambdas, normal_params, DBR_PAIRS)

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

    # ---------- Transmittance penalty ----------
    Tavg = np.mean(Ts)
    trans_penalty = np.mean(np.maximum(0, 0.8 - Tavg) ** 2)

    # ---------- Final Loss ----------
    loss = (
        -w_outer * outer_min   # maximize min(A1, A3)
        + w_center * A2        # minimize A2
        + w_trans * trans_penalty
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
            f.write(f"Tavg = {Tavg:.4f}\n\n")

        print(">>> Geometry saved (meets cubic + slope criteria)")


    print(
        f"A1={A1:.3e}, "
        f"A2={A2:.3e}, "
        f"A3={A3:.3e}, "
        f"min_outer={outer_min:.3e}, "
        f"Tavg={Tavg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss










def loss_three_region_quadratic(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_center=1.0,
    w_refl=10.0,
    save_file="good_quadratic_geometries2.txt"
):

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

    if not (np.any(mask1)  and np.any(mask3)):
        return 1e6
    
    # ---------- Linear fits in each region ----------
    A1, B1 = np.polyfit(lambdas[mask1], phis[mask1], 1)
    A2, B2 = np.polyfit(lambdas[mask2], phis[mask2], 1)
    A3, B3 = np.polyfit(lambdas[mask3], phis[mask3], 1)

    S1,S2,S3 = A1,A2,A3

    A1 = abs(A1)
    A2 = abs(A2)
    A3 = abs(A3)

    outer_min = min(A1, A3)



    # ---------- Quadratic Fit in Center Region ----------
    lambda_c = 1.4975   # define your center (or compute from data)
    mask_all = mask1 | mask3
    l_center = lambdas[mask_all]
    phi_center = phis[mask_all]

    shifted = l_center - lambda_c

    # Design matrix for: A(x-xc)^2 + B(x-xc) + C
    # ---------- Pure Quadratic (No Linear Term) ----------
    Xq = np.vstack([
        shifted**2,
        np.ones_like(shifted)
    ]).T

    coeffs_q, _, _, _ = np.linalg.lstsq(Xq, phi_center, rcond=None)

    Aq, Cq = coeffs_q

    phi_quad_fit = Aq*shifted**2 + Cq

    quad_error = np.mean((phi_center - phi_quad_fit)**2)






    # ---------- Reflectance penalty ----------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg) ** 2)

    # ---------- Sign Penalty ----------
    sign_penalty = 0.0
    w_sign = 5.0  # tune this
    if S1 * S3 > 0:   # same sign
        sign_penalty = (abs(S1) + abs(S3))  # magnitude-based penalty


    # ---------- Final Loss ----------
    loss = (
    -w_outer * outer_min
    #+ w_center * A2
    + w_refl * refl_penalty
    + w_sign * sign_penalty)

    if (
        loss<0 
        ):
        with open(save_file, "a") as f:
            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n")
            f.write(f"S1 = {S1:.4e}\n")
            #f.write(f"S2 = {S2:.4e}\n")
            f.write(f"S3 = {S3:.4e}\n")
            f.write(f"quadratic_error = {quad_error:.4e}\n")
            f.write(f"loss = {loss:.4e}\n")
            f.write(f"Ravg = {Ravg:.4f}\n\n")

        print(">>> Geometry saved (meets cubic + slope criteria)")


    print(
        f"S1={S1:.3e}, "
        #f"S2={S2:.3e}, "
        f"S3={S3:.3e}, "
        f"sign_penalty={sign_penalty:.3e}, "
        f"quadratic_error={quad_error:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss














'''This is somewhat good for cubic geometries.'''


def loss_three_region_polyfit_trans_freq(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_center=1.0,
    w_trans=10.0,
    save_file="good_cubic_geometries2.txt"
):

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
    mask1 = (lambdas >= 1.49)  & (lambdas < 1.495)
    mask2 = (lambdas >= 1.495) & (lambdas < 1.5)
    mask3 = (lambdas >= 1.5)   & (lambdas <= 1.505)

    if not (np.any(mask1) and np.any(mask2) and np.any(mask3)):
        return 1e6

    # ---------- Linear fits using normalized frequency ----------
    S1, _ = np.polyfit(wshift[mask1], phis[mask1], 1)
    S2, _ = np.polyfit(wshift[mask2], phis[mask2], 1)
    S3, _ = np.polyfit(wshift[mask3], phis[mask3], 1)

    A1 = abs(S1)
    A2 = abs(S2)
    A3 = abs(S3)

    outer_min = min(A1, A3)

    # ---------- Reflectance penalty ----------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg)**2)

    # ---------- Constraint penalties ----------
    sign_penalty = 0
    magnitude_penalty = 0
    center_penalty = 0

    # SAME sign condition
    if S1 * S3 < 0:
        sign_penalty += 10

    # OPPOSITE sign condition
    if S1 * S2 > 0:
        sign_penalty += 10

    # slope magnitude match
    magnitude_penalty = ((abs(S3) - abs(S1))/abs(S1))**2

    # S2 within ±20% of S1
    ratio = abs(S2)/abs(S1)

    if ratio < 0.8 or ratio > 1.2:
        center_penalty = (ratio - 1)**2

    # ---------- Quadratic Fit ----------
    quad_coeffs = np.polyfit(wshift, phis, 2)

    Aq = quad_coeffs[0]
    Bq = quad_coeffs[1]
    Cq = quad_coeffs[2]

    phi_quad = np.polyval(quad_coeffs, wshift)
    quad_error = np.mean((phis - phi_quad)**2)

    # ---------- Cubic Fit ----------
    cubic_coeffs = np.polyfit(wshift, phis, 3)

    Ac = cubic_coeffs[0]
    Bc = cubic_coeffs[1]
    Cc = cubic_coeffs[2]
    Dc = cubic_coeffs[3]

    phi_cubic = np.polyval(cubic_coeffs, wshift)
    cubic_error = np.mean((phis - phi_cubic)**2)

    # ---------- Final Loss ----------
    loss = (
        -w_outer * outer_min
        + w_center * A2
        + w_trans * refl_penalty
        + 50 * magnitude_penalty
        + 50 * center_penalty
        + sign_penalty
    )

    # ---------- Save geometry if conditions satisfied ----------
    if (
    (
        (S1 * S3 > 0) and
        (S1 * S2 < 0) and
        (
            (0.8 <= abs(S2)/abs(S1) <= 1.2) or
            (0.8 <= abs(S2)/abs(S3) <= 1.2)
        )
    )
    or (loss < 0)
):

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")

            f.write(f"S1 = {S1:.4f}\n")
            f.write(f"S2 = {S2:.4f}\n")
            f.write(f"S3 = {S3:.4f}\n\n")

            f.write("---- Quadratic Fit ----\n")
            f.write(f"Aq = {Aq:.4e}\n")
            f.write(f"Bq = {Bq:.4e}\n")
            f.write(f"Cq = {Cq:.4e}\n")
            f.write(f"quadratic_error = {quad_error:.4e}\n\n")

            f.write("---- Cubic Fit ----\n")
            f.write(f"Ac = {Ac:.4e}\n")
            f.write(f"Bc = {Bc:.4e}\n")
            f.write(f"Cc = {Cc:.4e}\n")
            f.write(f"Dc = {Dc:.4e}\n")
            f.write(f"cubic_error = {cubic_error:.4e}\n\n")

            f.write(f"Ravg = {Ravg:.4f}\n\n")

        print(">>> Geometry saved")

    print(
        f"S1={S1:.3f}, "
        f"S2={S2:.3f}, "
        f"S3={S3:.3f}, "
        f"quad_err={quad_error:.3e}, "
        f"cubic_err={cubic_error:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss





'''This is somewhat good for quadratic geometries'''



def loss_three_region_quadfit_trans_freq(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_outer=1.0,
    w_trans=10.0,
    save_file="good_quadratic_geometries3.txt"
):

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
    wshift = (omegas - omega0) / omega0

    # ---------- Define Regions ----------
    mask1 = (lambdas >= 1.49) & (lambdas < 1.4975)
    mask3 = (lambdas >= 1.4975) & (lambdas <= 1.505)

    if not (np.any(mask1) and np.any(mask3)):
        return 1e6

    # ---------- Linear fits ----------
    S1, _ = np.polyfit(wshift[mask1], phis[mask1], 1)
    S3, _ = np.polyfit(wshift[mask3], phis[mask3], 1)

    A1 = abs(S1)
    A3 = abs(S3)

    outer_min = min(A1, A3)

    # ---------- Reflectance penalty ----------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg)**2)

    # ---------- Constraint penalties ----------
    sign_penalty = 0

    # Opposite sign condition
    if S1 * S3 > 0:
        sign_penalty += 10

    # slope magnitude matching
    magnitude_penalty = (abs(A3 - A1) / max(A1, 1e-9))**2

    # optional balance term to help optimizer
    outer_balance = abs(A1 - A3)

    # ---------- Quadratic Fit ----------
    quad_coeffs = np.polyfit(wshift, phis, 2)

    Aq = quad_coeffs[0]
    Bq = quad_coeffs[1]
    Cq = quad_coeffs[2]

    phi_quad = np.polyval(quad_coeffs, wshift)
    quad_error = np.mean((phis - phi_quad)**2)

    # ---------- Cubic Fit ----------
    cubic_coeffs = np.polyfit(wshift, phis, 3)

    Ac = cubic_coeffs[0]
    Bc = cubic_coeffs[1]
    Cc = cubic_coeffs[2]
    Dc = cubic_coeffs[3]

    phi_cubic = np.polyval(cubic_coeffs, wshift)
    cubic_error = np.mean((phis - phi_cubic)**2)

    # ---------- Final Loss ----------
    loss = (
        -w_outer * outer_min
        + w_trans * refl_penalty
        + 50 * magnitude_penalty
        + 0.1 * outer_balance
        + sign_penalty
    )

    # ---------- Save geometry if condition satisfied ----------
    save_condition = (
        (S1 * S3 < 0) and
        (abs(A1 - A3) <= 10)
    )

    if save_condition:

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")

            f.write(f"S1 = {S1:.4f}\n")
            f.write(f"S3 = {S3:.4f}\n\n")

            f.write("---- Quadratic Fit ----\n")
            f.write(f"Aq = {Aq:.4e}\n")
            f.write(f"Bq = {Bq:.4e}\n")
            f.write(f"Cq = {Cq:.4e}\n")
            f.write(f"quadratic_error = {quad_error:.4e}\n\n")

            f.write("---- Cubic Fit ----\n")
            f.write(f"Ac = {Ac:.4e}\n")
            f.write(f"Bc = {Bc:.4e}\n")
            f.write(f"Cc = {Cc:.4e}\n")
            f.write(f"Dc = {Dc:.4e}\n")
            f.write(f"cubic_error = {cubic_error:.4e}\n\n")

            f.write(f"Ravg = {Ravg:.4f}\n\n")

        print(">>> Geometry saved (opposite slopes, similar magnitude)")

    print(
        f"S1={S1:.3f}, "
        f"S3={S3:.3f}, "
        f"|S1|-|S3|={abs(A1-A3):.3f}, "
        f"quad_err={quad_error:.3e}, "
        f"cubic_err={cubic_error:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss







def loss_three_region_quadfit2_trans_freq(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    w_trans=10.0,
    save_file="good_quadratic_geometries4.txt"
):

    import numpy as np

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
    wshift = (omegas - omega0) / omega0

    # ---------- Define Regions ----------
    mask1 = (lambdas >= 1.49) & (lambdas < 1.4975)
    mask3 = (lambdas >= 1.4975) & (lambdas <= 1.505)

    if not (np.any(mask1) and np.any(mask3)):
        return 1e6

    # ---------- Linear slope fits ----------
    S1, _ = np.polyfit(wshift[mask1], phis[mask1], 1)
    S3, _ = np.polyfit(wshift[mask3], phis[mask3], 1)

    A1 = abs(S1)
    A3 = abs(S3)

    outer_min = min(A1, A3)

    # ---------- Reflectance penalty ----------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Ravg)**2)

    # ---------- Opposite slope constraint ----------
    sign_penalty = 0
    if S1 * S3 > 0:
        sign_penalty += 10

    # ---------- Encourage symmetric slopes ----------
    magnitude_penalty = (abs(A3 - A1) / max(A1, 1e-9))**2

    # ---------- Polynomial Fits ----------
    quad_coeffs = np.polyfit(wshift, phis, 2)
    cubic_coeffs = np.polyfit(wshift, phis, 3)

    Aq = quad_coeffs[0]
    Ac = cubic_coeffs[0]

    # ---------- Cubic vs Quadratic ratio ----------
    eps = 1e-9
    cubic_ratio = abs(Ac) / (abs(Aq) + eps)

    # ---------- Final Loss ----------
    loss = (
        - 2.0 * outer_min            # maximize slope magnitude
        + 100 * cubic_ratio          # penalize cubic relative to quadratic
        + w_trans * refl_penalty
        + 20 * magnitude_penalty
        + sign_penalty
    )

    # ---------- Save geometry condition ----------
    save_condition = (
        (S1 * S3 < 0) and
        (abs(A1 - A3) <= 10) and
        (cubic_ratio < 0.1)          # cubic < 10% of quadratic
    )

    if save_condition:

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")

            f.write(f"S1 = {S1:.4f}\n")
            f.write(f"S3 = {S3:.4f}\n\n")

            f.write("---- Polynomial Coefficients ----\n")
            f.write(f"Aq (quadratic) = {Aq:.4e}\n")
            f.write(f"Ac (cubic)     = {Ac:.4e}\n")
            f.write(f"cubic_ratio    = {cubic_ratio:.4e}\n\n")

            f.write(f"Ravg = {Ravg:.4f}\n\n")

        print(">>> Geometry saved (quadratic-dominant phase)")

    print(
        f"S1={S1:.3f}, "
        f"S3={S3:.3f}, "
        f"Aq={Aq:.3e}, "
        f"Ac={Ac:.3e}, "
        f"ratio={cubic_ratio:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss







def loss_center_symmetric_quadratic(
    geometry_func,
    geometry_params,
    normal_params,
    lambdas,
    DBR_PAIRS,
    center_lambda=1.496,
    w_slope=50,
    w_sym=20,
    w_refl=10,
    save_file="good_quadratic_geometries_centered.txt"
):

    import numpy as np

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
    wshift = (omegas - omega0) / omega0

    # ---------- Split left/right around center ----------
    mask_left = lambdas < center_lambda
    mask_right = lambdas > center_lambda

    if not (np.any(mask_left) and np.any(mask_right)):
        return 1e6

    # ---------- Linear fits (slope on each side) ----------
    S_left, _ = np.polyfit(wshift[mask_left], phis[mask_left], 1)
    S_right, _ = np.polyfit(wshift[mask_right], phis[mask_right], 1)

    A_left = abs(S_left)
    A_right = abs(S_right)

    # ---------- Slope magnitude matching ----------
    slope_match_penalty = (A_left - A_right)**2

    # ---------- Opposite sign condition ----------
    sign_penalty = 0
    if S_left * S_right > 0:
        sign_penalty = 20

    # ---------- Symmetry penalty ----------
    # enforce φ(λ0+Δ) ≈ φ(λ0−Δ)

    sym_error = []

    for i in range(len(lambdas)):
        lam = lambdas[i]

        mirror = 2*center_lambda - lam

        idx = np.argmin(np.abs(lambdas - mirror))

        sym_error.append((phis[i] - phis[idx])**2)

    symmetry_penalty = np.mean(sym_error)

    # ---------- Reflectance penalty ----------
    Ravg = np.mean(Rs)
    refl_penalty = np.mean(np.maximum(0, 0.8 - Rs)**2)

    # ---------- Quadratic fit (diagnostic only) ----------
    quad_coeffs = np.polyfit(wshift, phis, 2)
    phi_quad = np.polyval(quad_coeffs, wshift)
    quad_error = np.mean((phis - phi_quad)**2)

    Aq, Bq, Cq = quad_coeffs

    # ---------- Cubic fit (diagnostic only) ----------
    cubic_coeffs = np.polyfit(wshift, phis, 3)
    phi_cubic = np.polyval(cubic_coeffs, wshift)
    cubic_error = np.mean((phis - phi_cubic)**2)

    Ac, Bc, Cc, Dc = cubic_coeffs

    # ---------- Final Loss ----------
    loss = (
        w_slope * slope_match_penalty
        + sign_penalty
        + w_sym * symmetry_penalty
        + w_refl * refl_penalty
    )

    # ---------- Save geometry if good ----------
    save_condition = (
        (S_left * S_right < 0)
        and (abs(A_left - A_right) < 0.1)
        and (symmetry_penalty < 0.01)
    )

    if save_condition:

        with open(save_file, "a") as f:

            f.write("=====================================\n")
            f.write(f"geometry_params = {geometry_params}\n")
            f.write(f"normal_params   = {normal_params}\n\n")

            f.write(f"S_left  = {S_left:.4f}\n")
            f.write(f"S_right = {S_right:.4f}\n")
            f.write(f"|S_left|-|S_right| = {abs(A_left-A_right):.4f}\n\n")

            f.write("---- Quadratic Fit ----\n")
            f.write(f"Aq = {Aq:.4e}\n")
            f.write(f"Bq = {Bq:.4e}\n")
            f.write(f"Cq = {Cq:.4e}\n")
            f.write(f"quadratic_error = {quad_error:.4e}\n\n")

            f.write("---- Cubic Fit ----\n")
            f.write(f"Ac = {Ac:.4e}\n")
            f.write(f"Bc = {Bc:.4e}\n")
            f.write(f"Cc = {Cc:.4e}\n")
            f.write(f"Dc = {Dc:.4e}\n")
            f.write(f"cubic_error = {cubic_error:.4e}\n\n")

            f.write(f"Ravg = {Ravg:.4f}\n\n")

        print(">>> Geometry saved (center symmetric quadratic)")

    print(
        f"S_left={S_left:.3f}, "
        f"S_right={S_right:.3f}, "
        f"|S_left|-|S_right|={abs(A_left-A_right):.3f}, "
        f"sym={symmetry_penalty:.3e}, "
        f"quad_err={quad_error:.3e}, "
        f"cubic_err={cubic_error:.3e}, "
        f"Ravg={Ravg:.3f}, "
        f"LOSS={loss:.3f}"
    )

    return loss