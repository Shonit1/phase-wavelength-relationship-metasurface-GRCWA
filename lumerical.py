import h5py
import numpy as np
import matplotlib.pyplot as plt


'''
imports lumerical data and
analyzes input and reflected pulses in time domain to compute group delay and other calculations for geometries that show linear relationship between phase and wavelength.

'''




# ==============================
# LOAD DATA
# ==============================

with h5py.File("input_pulse1.mat", "r") as f:
    t_in = np.array(f['t']).squeeze()
    E_in = np.array(f['E']).squeeze()

with h5py.File("output_pulse_15.mat", "r") as f:
    t_ref = np.array(f['t']).squeeze()
    E_ref = np.array(f['E']).squeeze()

# ==============================
# CONVERT TO FEMTOSECONDS
# ==============================

t_in *= 1e15
t_ref *= 1e15

# ==============================
# NORMALIZE
# ==============================

#E_in /= np.max(np.abs(E_in))
#E_ref /= np.max(np.abs(E_ref))

# ==============================
# LIMIT RANGE (0–6000 fs)
# ==============================

mask_in = (t_in >= 0) & (t_in <= 6000)
mask_ref = (t_ref >= 0) & (t_ref <= 6000)

t_in, E_in = t_in[mask_in], E_in[mask_in]
t_ref, E_ref = t_ref[mask_ref], E_ref[mask_ref]

# ==============================
# FIND PEAKS
# ==============================

idx_in = np.argmax(np.abs(E_in))
idx_ref = np.argmax(np.abs(E_ref))

t_peak_in = t_in[idx_in]
t_peak_ref = t_ref[idx_ref]

delay = t_peak_ref - t_peak_in

# ==============================
# INTENSITY
# ==============================

I_in = E_in**2
I_ref = E_ref**2

I_in /= np.max(I_in)
I_ref /= np.max(I_in)

# ==============================
# -------- FIGURE 1: E FIELD --------
# ==============================

plt.figure(figsize=(9,5))

plt.plot(t_in, E_in, linewidth=2, alpha=0.7, label="Input Pulse (E)")
plt.plot(t_ref, E_ref, linewidth=2, alpha=0.7, label="Reflected Pulse (E)")

plt.fill_between(t_in, E_in, alpha=0.15)
plt.fill_between(t_ref, E_ref, alpha=0.15)

plt.axvline(t_peak_in, linestyle='--', linewidth=2)
plt.axvline(t_peak_ref, linestyle='--', linewidth=2)

# Delay arrow
y_arrow = -0.7
plt.annotate('', xy=(t_peak_ref, y_arrow), xytext=(t_peak_in, y_arrow),
             arrowprops=dict(arrowstyle='<->', linewidth=2))

plt.text((t_peak_in + t_peak_ref)/2, y_arrow - 0.12,
         f"Δt = {delay:.2f} fs", ha='center')

plt.xlabel("Time (fs)")
plt.ylabel("Ex (Normalized)")
plt.title("Electric Field Comparison (Time Domain) Geometry 1")

plt.legend()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.8)
plt.grid(which='minor', linestyle=':', linewidth=0.5)

plt.ylim(-1.2, 1.2)

plt.tight_layout()
plt.show()

# ==============================
# -------- FIGURE 2: INTENSITY --------
# ==============================

plt.figure(figsize=(9,5))

plt.plot(t_in, I_in, linewidth=2, alpha=0.8, label="Input Intensity")
plt.plot(t_ref, I_ref, linewidth=2, alpha=0.8, label="Reflected Intensity")

plt.fill_between(t_in, I_in, alpha=0.15)
plt.fill_between(t_ref, I_ref, alpha=0.15)

# Peak lines (same positions)
plt.axvline(t_peak_in, linestyle='--', linewidth=2)
plt.axvline(t_peak_ref, linestyle='--', linewidth=2)

# Delay arrow
y_arrow = 0.2
plt.annotate('', xy=(t_peak_ref, y_arrow), xytext=(t_peak_in, y_arrow),
             arrowprops=dict(arrowstyle='<->', linewidth=2))

plt.text((t_peak_in + t_peak_ref)/2, y_arrow + 0.08,
         f"Δt = {delay:.2f} fs", ha='center')

plt.xlabel("Time (fs)")
plt.ylabel("Intensity (Normalized) |E|²")
plt.title("Intensity (Time Domain) Geometry 1")

plt.legend()
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.8)
plt.grid(which='minor', linestyle=':', linewidth=0.5)

plt.ylim(0, 1.2)

plt.tight_layout()
plt.show()

# ==============================
# PRINT DELAY
# ==============================

print(f"Group Delay = {delay:.2f} fs")