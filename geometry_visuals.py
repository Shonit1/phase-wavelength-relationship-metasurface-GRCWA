import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
import matplotlib.patches as mpatches
from geometry_functions import *
from matplotlib.colors import ListedColormap



'''
This file contains visualization utilities for metasurface geometries.
It takes different geometric designs (cylinders, rings, patterns, etc.) and converts them into 2D or cross-sectional plots of permittivity (ε) for quick inspection and debugging.

Works in x–y plane (top view) and x–z plane (cross-section)
Uses matplotlib for visualization
Helps verify geometry before running RCWA/simulations

'''





def _plot(ep, title):

    '''Generic function to display a permittivity grid as an image.'''

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







def plot_cylinder(r, eps):


    '''Plots a single circular cylinder.'''

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    mask = (xc**2 + yc**2) < r**2

    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps

    _plot(ep, f"Cylinder (r={r})")





def plot_dual_cylinder_structure(
    r1, r2, shift, a,
    hpattern,
    DBR_PAIRS,
    hsio2_dbr,
    hs_dbr
):
    
    '''
    Plots full device stack (x–z view) including:

    cylinders
    air region
    DBR layers (Si/SiO₂)
    substrate
    '''




    fig, ax = plt.subplots(figsize=(8, 6))

    z = 0

    # ------------------ COLORS ------------------
    color_air   = 'lightblue'
    color_sio2  = 'lightgray'
    color_si    = '#8B4513'
    color_cyl1  = '#8B4513'
    color_cyl2  = '#8B4513'
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
    x1 = a/2 
    x2 = a/2 

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
        mpatches.Patch(color=color_cyl1, label="Cylinder  (Si)"),
        mpatches.Patch(color=color_glass, label="Glass")
        #mpatches.Patch(color=color_cyl2, label="Cylinder 2 (Si)")
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
    ax.set_title("Cylinder Metasurface on DBR (x–z plane)")
    ax.set_aspect('equal')
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()










def plot_double_cylinder_xy_from_grid(geometry_params, eps, L1, L2):

    '''Generates geometry from permittivity grid and plots it in x–y view.'''

    L1 = [L1, 0]
    L2 = [0, L2]

    # Generate permittivity grid
    ep = get_epgrid_double_cylinder_d_new(
        geometry_params, eps, L1, L2
    )

    r1, r2, d = geometry_params
    a = L1[0]

    x = np.linspace(0, a, Nx)
    y = np.linspace(0, L2[1], Ny)

    # Convert to binary mask (1 = cylinder, 0 = background)
    ep_real = np.real(ep)
    mask = ep_real > 1.1   # threshold (adjust if needed)

    # Define custom colors
    cmap = ListedColormap([
        "#D3D3D3",  # light gray (background)
        "#8B4513"   # brown (cylinders)
    ])

    plt.figure(figsize=(6, 6))

    plt.pcolormesh(
        x, y,
        mask.T,
        shading="auto",
        cmap=cmap
    )

    # Draw unit cell boundary
    plt.plot([0, a, a, 0, 0],
             [0, 0, L2[1], L2[1], 0],
             'k--', linewidth=1)

    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Cylinder Geometry 3")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
















def plot_ring(r_inner, r_outer, eps):

    '''Plots a ring between inner and outer radii.'''

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2
    r = np.sqrt(xc**2 + yc**2)

    ep = np.ones((Nx, Ny)) * eair
    ep[(r > r_inner) & (r < r_outer)] = eps

    _plot(ep, f"Ring ({r_inner},{r_outer})")








def plot_double_ring(r1, r2, r3, r4, eps):

    '''Plots two concentric rings.'''

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

 
 






def plot_split_cylinder(r, gap, eps):

    '''Plots a cylinder with a central gap (split resonator).'''

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









def plot_ellipse(rx, ry, eps):

    '''Plots an elliptical shape.'''

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    mask = (xc/rx)**2 + (yc/ry)**2 < 1

    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps

    _plot(ep, "Ellipse")











def plot_cross(width, eps):

    '''Plots a cross-shaped structure.'''

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    xc = np.abs(X - L1[0]/2)
    yc = np.abs(Y - L2[1]/2)

    mask = (xc < width) | (yc < width)

    ep = np.ones((Nx, Ny)) * eair
    ep[mask] = eps

    _plot(ep, "Cross")







def plot_square_frame(w_outer, w_inner, eps):

    '''Plots a hollow square frame.'''

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
















def plot_3x3_binary(pattern,eps):

    '''Visualizes a 3×3 binary pattern (on/off cells).'''

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


    '''Visualizes a 4×4 binary pattern.'''

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


    '''Plots diagonal stripe patterns.'''

    x = np.linspace(0, L1[0], Nx, endpoint=False)
    y = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ep = np.ones((Nx, Ny)) * eair

    mask = ((X + Y) % period) < period/2
    ep[mask] = eps

    _plot(ep, f"Diagonal Stripe (period={period})")
















def plot_3x3_geometry(pattern, a=1.0, threshold=0.5):
    
    '''Plots 3×3 metasurface + DBR stack in cross-sectional (x–z) view.'''

    pattern = np.array(pattern).reshape(3,3)
    
    dx = dy = a/3
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    for i in range(3):
        for j in range(3):
            
            # Binary decision
            val = 1 if pattern[i,j] >= threshold else 0
            
            color = "black" if val == 1 else "white"
            
            rect = plt.Rectangle((i*dx, j*dy),
                                 dx, dy,
                                 facecolor=color,
                                 edgecolor="gray")
            ax.add_patch(rect)
    
    ax.set_xlim(0,a)
    ax.set_ylim(0,a)
    ax.set_aspect("equal")
    ax.set_title("3x3 Metasurface Geometry (Binary)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()





  

def plot_3x3_pattern_xz(
    pattern,
    a,
    hpattern,
    DBR_PAIRS,
    hsio2_dbr,
    hs_dbr,
    threshold=0.5
):
    
    '''Plots 3×3 metasurface + DBR stack in cross-sectional (x–z) view.'''


    fig, ax = plt.subplots(figsize=(8, 6))

    pattern = np.array(pattern).reshape(3,3)

    z = 0

    # ------------------ COLORS ------------------
    color_air   = 'lightblue'
    color_sio2  = 'lightgray'
    color_si    = 'brown'
    color_glass = 'lightgreen'

    # ------------------ TOP AIR ------------------
    h_air_top = 0.1
    ax.add_patch(plt.Rectangle((0, z), a, h_air_top,
                 color=color_air, alpha=0.6))
    z += h_air_top

    # ------------------ PATTERN REGION ------------------
    dx = a / 3

    # Background air first
    ax.add_patch(plt.Rectangle((0, z), a, hpattern,
                 color=color_air, alpha=0.6))

    # Add silicon blocks where pattern exists
    for i in range(3):
        for j in range(3):   # include full grid info
            if pattern[i, j] >= threshold:
                # In x–z slice we project along y
                # If ANY cell in column i is Si → fill column
                ax.add_patch(plt.Rectangle((i*dx, z),
                             dx, hpattern,
                             color=color_si))

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
        mpatches.Patch(color=color_glass, label="Glass")
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
    ax.set_title("3×3 Metasurface Pattern on DBR (x–z plane)")
    ax.set_aspect('equal')
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()