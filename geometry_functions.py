import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *
import time
import os
from datetime import datetime


'''This file contains functions that generate the permittivity grid (epgrid) for various geometric patterns. 
Each function takes in geometry parameters and returns a 2D array representing the spatial distribution of permittivity for that pattern. 
These epgrids are used as input for RCWA simulations to compute optical responses.'''




def get_epgrid_double_cylinder_d_new(geometry_params, eps, L1, L2):


    """
    Generates two cylinders of material (eps) placed symmetrically with a
    horizontal shift from the center.
    """

    r1, r2, d = geometry_params
    a = L1[0]

    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    # Symmetric centers
    x1 = a/2 - d/2
    x2 = a/2 + d/2
    y_center = L2[1]/2

    mask1 = (X - x1)**2 + (Y - y_center)**2 < r1**2
    mask2 = (X - x2)**2 + (Y - y_center)**2 < r2**2

    # Optional overlap debug
    # if np.any(mask1 & mask2):
    #     raise ValueError("Overlap detected")

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask1 | mask2] = eps

    return ep








def get_epgrid_3x3(pattern, eps, L1, L2):

    """
    Generates a 2D permittivity grid using a 3×3 pattern, where each cell
    is filled based on a weighted mix of material (eps) and air (eair).
    The pattern values (0–1) control the local material fraction.
    """

    a = L1[0]
    pattern = np.array(pattern).reshape(3,3)
    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/3
    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            f = np.clip(pattern[i,j],0,1)
            ep[(X>=i*dx)&(X<(i+1)*dx)&
               (Y>=j*dy)&(Y<(j+1)*dy)] = f*eps + (1-f)*eair
    return ep





def get_epgrid_4x4(pattern, eps, L1,L2):
    """
    Creates a 4×4 pixelated permittivity grid where each cell is filled
    using a weighted mix of material (eps) and air based on the pattern.
    """
    a = L1[0]
    pattern = np.array(pattern).reshape(4, 4)
    ep = np.ones((Nx, Ny), dtype=complex) * eair

    dx = dy = a / 4
    x = np.linspace(0, a, Nx, endpoint=False)
    y = np.linspace(0, a, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    for i in range(4):
        for j in range(4):
            f = np.clip(pattern[i, j], 0, 1)
            mask = (
                (X >= i * dx) & (X < (i + 1) * dx) &
                (Y >= j * dy) & (Y < (j + 1) * dy)
            )
            ep[mask] = f * eps + (1 - f) * eair

    return ep








def get_epgrids_cylinder(geometry_params, eps,L1,L2):

    """
    Generates a permittivity grid with a circular cylinder of material (eps)
    centered in the unit cell, surrounded by air.
    """

    r = geometry_params[0]
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')

    x_c = x - L1[0] / 2
    y_c = y - L2[1] / 2

    mask = (x_c**2 + y_c**2) < r**2

    epgrid = np.ones((Nx, Ny), dtype=complex) * 1.0
    epgrid[mask] = eps

    return epgrid







def get_epgrid_ring(geometry_params, eps,L1,L2):

    """
    Creates a ring-shaped structure with material (eps) between inner and
    outer radii, and air elsewhere.
    """

    r_inner, r_outer = geometry_params
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2
    r = np.sqrt(xc**2 + yc**2)

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[(r > r_inner) & (r < r_outer)] = eps

    return ep








def get_epgrid_double_ring(geometry_params, eps,L1,L2):

    """
    Generates two concentric rings (double resonance structure) using two
    radial regions filled with material (eps).
    """

    r1, r2, r3, r4 = geometry_params
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2
    r = np.sqrt(xc**2 + yc**2)

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[(r > r1) & (r < r2)] = eps
    ep[(r > r3) & (r < r4)] = eps

    return ep







def get_epgrid_ellipse(geometry_params, eps,L1,L2):

    """
    Creates an elliptical pillar of material (eps) centered in the grid,
    with air outside.
    """

    rx, ry = geometry_params

    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    mask = (xc/rx)**2 + (yc/ry)**2 < 1

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = eps

    return ep





#cross resonator

def get_epgrid_cross(geometry_params, eps,L1,L2):

    """
    Generates a cross-shaped structure where horizontal and vertical bars
    of width w are filled with material (eps).
    """

    w = geometry_params[0]
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = np.abs(X - L1[0]/2)
    yc = np.abs(Y - L2[1]/2)

    mask = (xc < w) | (yc < w)

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = eps

    return ep




#Hollow Square Frame


def get_epgrid_square_frame(geometry_params, eps,L1,L2):


    """
    Creates a hollow square frame with outer region filled with material
    (eps) and inner region as air.
    """

    w_outer, w_inner = geometry_params

    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = np.abs(X - L1[0]/2)
    yc = np.abs(Y - L2[1]/2)

    outer = (xc < w_outer) & (yc < w_outer)
    inner = (xc < w_inner) & (yc < w_inner)

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[outer] = eps
    ep[inner] = eair

    return ep







def get_epgrid_diagonal(geometry_params, eps,L1,L2):
     

     
    """
    Generates a diagonal stripe pattern where alternating regions are filled
    with material (eps) based on a given period.
    """
    period = geometry_params[0]

    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    mask = ((X + Y) % period) < period/2

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = eps

    return ep






def get_epgrid_split_cylinder(geometry_params, eps,L1,L2):


    """
    Creates a circular cylinder with a vertical air gap (split), forming a
    split-resonator-like structure.
    """

    r,gap = geometry_params
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2

    circle = (xc**2 + yc**2) < r**2
    gap_region = np.abs(xc) < gap

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[circle] = eps
    ep[gap_region] = eair

    return ep















def get_epgrid_radial_gradient(eps, power=2):


    """
    Generates a radially graded permittivity profile that smoothly varies
    from air to material (eps) based on distance from center.
    """

    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc = X - L1[0]/2
    yc = Y - L2[1]/2
    r = np.sqrt(xc**2 + yc**2)

    r_norm = r / np.max(r)
    f = r_norm**power

    ep = f*eps + (1-f)*eair
    return ep






def get_epgrid_single_cylinder(geometry_params, eps, L1, L2):

    """
    Creates a single centered circular cylinder of material (eps) in air. This is particularyly used for intensity calculations.
    """

    r = geometry_params[0]
    a = L1[0]

    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    # Single cylinder centered in cell
    x_center = a / 2
    y_center = L2[1] / 2

    mask = (X - x_center)**2 + (Y - y_center)**2 < r**2

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = eps

    return ep






def get_epgrid_elliptical_cylinder(geometry_params, eps, L1, L2):


    """
    Generates an elliptical cylinder centered in the unit cell with
    material (eps) inside and air outside.
    """

    rx = geometry_params[0]
    ry = geometry_params[1]
    a = L1[0]

    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    x_center = a / 2
    y_center = L2[1] / 2

    mask = ((X - x_center)**2)/(rx**2) + ((Y - y_center)**2)/(ry**2) < 1

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = eps

    return ep












def smooth_step(x, width):

    """
    Provides a smooth transition function (tanh-based) used to soften sharp
    boundaries in geometry.
    """

    return 0.5*(1 + np.tanh(x/width))


def get_epgrid_3x3_new(pattern, eps, L1, L2):


    """
    Creates a smoothed 3×3 pixelated permittivity grid where boundaries
    between cells are softened using a smooth step function.
    """

    smooth = 0.02
    a = L1[0]
    pattern = np.array(pattern).reshape(3,3)

    ep = np.ones((Nx,Ny), dtype=complex) * eair

    dx = dy = a/3

    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)

    X,Y = np.meshgrid(x,y,indexing="ij")

    # smoothing width in real units
    w = smooth * dx

    for i in range(3):
        for j in range(3):

            f = np.clip(pattern[i,j],0,1)

            x1 = i*dx
            x2 = (i+1)*dx
            y1 = j*dy
            y2 = (j+1)*dy

            mask_x = smooth_step(X-x1,w) * (1 - smooth_step(X-x2,w))
            mask_y = smooth_step(Y-y1,w) * (1 - smooth_step(Y-y2,w))

            mask = mask_x * mask_y

            ep += mask * (f*eps + (1-f)*eair - eair)

    return ep