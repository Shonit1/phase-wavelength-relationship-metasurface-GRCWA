# 🔷 Photonic Structure Inverse Design using RCWA

## 📌 Overview
This project implements an inverse design framework for photonic structures 
using Rigorous Coupled Wave Analysis (RCWA) and CMA-ES optimization. 

It allows:
- Simulation of optical response (phase, reflectance) of the reflected light.
- Parameter sweeps for design intuition
- Optimization of geometry
- Post-design validation and analysis

---

## 🚀 Features

- RCWA-based electromagnetic simulation
- Parameter sweep tools for geometry exploration
- CMA-ES optimization for inverse design
- Phase and reflectance computation
- Convergence testing (Fourier modes nG)
- Angular response analysis
- Phase curve fitting (linear, quadratic, cubic, resonant models)

---

## 📁 Project Structure

### 1. Core RCWA Functions can be found in rcwa_machinery.py
- Builds simulation objects
- Computes phase, reflectance, and fields

### 2. Parameter Sweep Module can be found in sweep_functions.py
- Sweeps geometry parameters
- Helps understand sensitivity

### 3. Optimization Script which can be found in 3 separate files for (single cylinder, double cylinder and patterned surface 3x3) in single_cylinder_optimization.py, optimization_double_cylinder.py and optimization_3x3.py.
- Uses CMA-ES to find optimal geometry
- Minimizes custom loss function

### 4. Validation / Analysis Script (analysis.py)
- Checks convergence (nG)
- plots spectra
- Studies angular response

### 5. Phase Fitting Script (phase_fitting.py)
- Fits phase vs wavelength
- Compares analytical models

### 6. Intensity plotting (intensity.py)

### 7. lumerical data plotting(lumerical.py and lumerical_quadratic.py)
- Imports lumerical data and plots the pulse and intensity.
- Various pulse related calculations are also included

---

## ⚙️ Installation

Core install required libraries:

```bash
pip install numpy matplotlib cma grcwa 





## Note

I have included various geometric_functions(geometry_functions.py) that I didnt get time to explore. You can just insert the various functions in the script and explore various phase vs wavelength relationships. Just need to insert the function and setup the optimization accordingly.

You will see geometry_params and normal_params which are essential for simulating the geometry. When you change the geometry function shape, you just need to modify geometry_params. For example for double cylinder geometry_params = [r1,r2,d]. Normal_params contain information about the lattice, height of metasurface and dbr. If you just change the geometry function, you just need to change the geometry_params, the height of metasurface,dbr and lattice constants wont change. 




