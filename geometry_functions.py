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






def get_epgrid_3x3(pattern, eps, L1, L2):
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




#Ring



def get_epgrid_ring(geometry_params, eps,L1,L2):
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





#Double Ring (Two Resonance Coupling)


def get_epgrid_double_ring(geometry_params, eps,L1,L2):
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





#elliptical pillar

def get_epgrid_ellipse(geometry_params, eps,L1,L2):


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





#diagonal stripe pattern

def get_epgrid_diagonal(geometry_params, eps,L1,L2):
    period = geometry_params[0]

    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    mask = ((X + Y) % period) < period/2

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = eps

    return ep




#Split cylinder

def get_epgrid_split_cylinder(geometry_params, eps,L1,L2):

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





#double_cylinder

def get_epgrid_double_cylinder(geometry_params, eps,L1,L2):
    r1, r2, shift = geometry_params
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    xc1 = X - L1[0]/2 - shift
    yc1 = Y - L2[1]/2

    xc2 = X - L1[0]/2 + shift
    yc2 = Y - L2[1]/2

    mask1 = (xc1**2 + yc1**2) < r1**2
    mask2 = (xc2**2 + yc2**2) < r2**2

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask1 | mask2] = eps

    return ep




def get_epgrid_double_cylinder_d(geometry_params, eps, L1, L2):

    r1, r2, d = geometry_params
    a = L1[0]

    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    X, Y = np.meshgrid(x0, y0, indexing='ij')

    # Centers placed symmetrically
    x1 = a/2 - d/2
    x2 = a/2 + d/2
    y_center = L2[1]/2

    xc1 = X - x1
    yc1 = Y - y_center

    xc2 = X - x2
    yc2 = Y - y_center

    mask1 = (xc1**2 + yc1**2) < r1**2
    mask2 = (xc2**2 + yc2**2) < r2**2

    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask1 | mask2] = eps

    return ep




def get_epgrid_double_cylinder_d_new(geometry_params, eps, L1, L2):

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






#Graded Radial

def get_epgrid_radial_gradient(eps, power=2):
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


