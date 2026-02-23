import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
import matplotlib.patches as mpatches
from geometry_functions import *

def _plot(ep, title):
    plt.figure(figsize=(5,5))
    plt.imshow(np.real(ep).T,
               origin="lower",
               extent=[0, L1[0], 0, L2[1]],
               cmap="gray")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Re(ε)")
    plt.tight_layout()
    plt.show()




###plot cylinder


def plot_cylinder(r, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    mask = (xc**2 + yc**2) < r**2

    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps

    _plot(ep, f"Cylinder (r={r})")



###2️⃣ Plot Ring




def plot_ring(r_inner, r_outer, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2
    r = np.sqrt(xc**2 + yc**2)

    ep = np.ones((Nx, Ny)) * eair
    ep[(r > r_inner) & (r < r_outer)] = eps

    _plot(ep, f"Ring ({r_inner},{r_outer})")



####3️⃣ Plot Double Ring




def plot_double_ring(r1, r2, r3, r4, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2
    r = np.sqrt(xc**2 + yc**2)

    ep = np.ones((Nx, Ny)) * eair
    ep[(r > r1) & (r < r2)] = eps
    ep[(r > r3) & (r < r4)] = eps

    _plot(ep, "Double Ring")

 
 
#### 4️⃣ Plot Split Cylinder





def plot_split_cylinder(r, gap, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    circle = (xc**2 + yc**2) < r**2
    gap_region = np.abs(xc) < gap

    ep = np.ones((Nx, Ny)) * eair
    ep[circle] = eps
    ep[gap_region] = eair

    _plot(ep, "Split Cylinder")




####5️⃣ Plot Ellipse






def plot_ellipse(rx, ry, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    mask = (xc/rx)**2 + (yc/ry)**2 < 1

    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps

    _plot(ep, "Ellipse")





####6️⃣ Plot Cross





def plot_cross(width, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = np.abs(X - L1[0]/2)
    yc = np.abs(Y - L2[1]/2)

    mask = (xc < width) | (yc < width)

    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps

    _plot(ep, "Cross")



###squareframe



def plot_square_frame(w_outer, w_inner, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = np.abs(X - L1[0]/2)
    yc = np.abs(Y - L2[1]/2)

    outer = (xc < w_outer) & (yc < w_outer)
    inner = (xc < w_inner) & (yc < w_inner)

    ep = np.ones((Nx, Ny)) * eair
    ep[outer] = eps
    ep[inner] = eair

    _plot(ep, "Square Frame")





####8️⃣ Plot Double Cylinder






def plot_double_cylinder(r1, r2, shift, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc1 = X - L1[0]/2 - shift
    yc1 = Y - L2[1]/2

    xc2 = X - L1[0]/2 + shift
    yc2 = Y - L2[1]/2

    mask1 = (xc1**2 + yc1**2) < r1**2
    mask2 = (xc2**2 + yc2**2) < r2**2

    ep = np.ones((Nx, Ny)) * eair
    ep[mask1 | mask2] = eps

    _plot(ep, "Double Cylinder")




def plot_3x3_binary(pattern,eps):
    pattern = np.array(pattern).reshape(3,3)

    ep = np.ones((Nx, Ny)) * eair

    dx = L1[0] / 3
    dy = L2[1] / 3

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    for i in range(3):
        for j in range(3):
            if pattern[i,j] == 1:
                mask = (X >= i*dx) & (X < (i+1)*dx) & \
                       (Y >= j*dy) & (Y < (j+1)*dy)

                ep[mask] = eps

    plt.figure(figsize=(5,5))
    plt.imshow(ep.T, origin="lower",
               extent=[0, L1[0], 0, L2[1]],
               cmap="gray")
    plt.title("3x3 Binary Pattern")
    plt.colorbar(label="ε")
    plt.show()







def plot_4x4_binary(pattern,eps):
    pattern = np.array(pattern).reshape(4,4)

    ep = np.ones((Nx, Ny)) * eair

    dx = L1[0] / 4
    dy = L2[1] / 4

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    for i in range(4):
        for j in range(4):
            if pattern[i,j] == 1:
                mask = (X >= i*dx) & (X < (i+1)*dx) & \
                       (Y >= j*dy) & (Y < (j+1)*dy)

                ep[mask] = eps

    plt.figure(figsize=(5,5))
    plt.imshow(ep.T, origin="lower",
               extent=[0, L1[0], 0, L2[1]],
               cmap="gray")
    plt.title("4x4 Binary Pattern")
    plt.colorbar(label="ε")
    plt.show()







def plot_diagonal(period, eps):
    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ep = np.ones((Nx, Ny)) * eair

    mask = ((X + Y) % period) < period/2
    ep[mask] = eps

    _plot(ep, f"Diagonal Stripe (period={period})")









def plot_dual_cylinder_structure(
    r1, r2, shift, a,
    hpattern,
    DBR_PAIRS,
    hsio2_dbr,
    hs_dbr
):
    fig, ax = plt.subplots(figsize=(8, 6))

    z = 0

    # ------------------ COLORS ------------------
    color_air   = 'lightblue'
    color_sio2  = 'lightgray'
    color_si    = 'brown'
    color_cyl1  = 'black'
    color_cyl2  = 'black'
    color_glass = 'lightgreen'

    # ------------------ TOP AIR ------------------
    h_air_top = 0.1
    ax.add_patch(plt.Rectangle((0, z), a, h_air_top,
                 color=color_air, alpha=0.6))
    z += h_air_top

    # ------------------ PATTERN BACKGROUND ------------------
    ax.add_patch(plt.Rectangle((0, z), a, hpattern,
                 color=color_air, alpha=0.6))

    # Cylinder centers
    x1 = a/2 - shift/2 - r1/2
    x2 = a/2 + shift/2 - r2/2

    # Cylinders
    ax.add_patch(plt.Rectangle((x1 - r1, z),
                 2*r1, hpattern,
                 color=color_cyl1))

    ax.add_patch(plt.Rectangle((x2 - r2, z),
                 2*r2, hpattern,
                 color=color_cyl2))

    z += hpattern

    # ------------------ DBR STACK ------------------
    for _ in range(DBR_PAIRS):

        # SiO2
        ax.add_patch(plt.Rectangle((0, z), a, hsio2_dbr,
                     color=color_sio2, alpha=0.6))
        z += hsio2_dbr

        # Si
        ax.add_patch(plt.Rectangle((0, z), a, hs_dbr,
                     color=color_si, alpha=0.6))
        z += hs_dbr

    # Bottom spacer
    h_bottom = 0.1
    ax.add_patch(plt.Rectangle((0, z), a, h_bottom,
                 color=color_glass, alpha=0.6))
    z += h_bottom

    # ------------------ LEGEND ------------------
    legend_patches = [
        mpatches.Patch(color=color_air, label="Air"),
        mpatches.Patch(color=color_sio2, label="SiO₂"),
        mpatches.Patch(color=color_si, label="Silicon"),
        mpatches.Patch(color=color_cyl1, label="Cylinder 1 (Si)"),
        mpatches.Patch(color=color_cyl2, label="Cylinder 2 (Si)")
    ]

    ax.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0
)

    # ------------------ FORMAT ------------------
    ax.set_xlim(0, a)
    ax.set_ylim(0, z)
    ax.set_xlabel("x (period a)")
    ax.set_ylabel("z")
    ax.set_title("Dual Cylinder Metasurface on DBR (x–z plane)")
    ax.set_aspect('equal')
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()










def plot_double_cylinder_xy_from_grid(geometry_params, eps, L1, L2):
    L1 = [L1,0]
    L2 = [0,L2]
    # Generate actual permittivity grid
    ep = get_epgrid_double_cylinder_d_new(
        geometry_params, eps, L1, L2
    )

    r1, r2, d = geometry_params
    a = L1[0]

    x = np.linspace(0, a, Nx)
    y = np.linspace(0, L2[1], Ny)

    plt.figure(figsize=(6,6))

    plt.pcolormesh(
        x, y,
        np.real(ep).T,
        shading="auto",
        cmap="coolwarm"
    )

    plt.colorbar(label="Permittivity")

    # Draw unit cell boundary
    plt.plot([0,a,a,0,0],
             [0,0,L2[1],L2[1],0],
             'k--', linewidth=1)

    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Double Cylinder Geometry (Simulation Grid)")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

