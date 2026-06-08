import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================================
# 1. GRID AND GEOMETRY CONFIGURATION
# =========================================================================
Nx = 300
Ny = 300
dx = 1

# =========================================================================
# 2. THERMAL PHYSICAL PARAMETERS
# =========================================================================
alpha = 0.25
dt = 0.08

Tm = 950
ambient_temp = 300
cooling_coeff = 0.07
rho = 1
cp = 1

# =========================================================================
# 3. MOVING HEAT SOURCE PROPERTIES
# =========================================================================
row_source = 150
col_source = 10
source_radius = 110
source_strength = 255

# Initial conditions
T = np.ones((Nx, Ny)) * ambient_temp

# =========================================================================
# 4. MATPLOTLIB REAL-TIME RENDER SETUP
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 7))
# Background initialization using the 'inferno' colormap for realistic heat tracking
im = ax.imshow(T, cmap='inferno', origin='lower', vmin=ambient_temp, vmax=1100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Temperature (K)", fontsize=11)

# Add a contour boundary line representing the active Melt Pool Isotherm (Tm = 950K)
contour_line = ax.contour(T, levels=[Tm], colors='cyan', linewidths=2, origin='lower')

ax.set_xlabel("X-Axis (Pixels)")
ax.set_ylabel("Y-Axis (Pixels)")

# =========================================================================
# 5. CORE EXPLICIT THERMAL SOLVER STEP FUNCTION
# =========================================================================
def update_thermal_field(frame):
    global T, col_source, contour_line
    
    # 1. Apply Finite Difference Internal Conduction Stencil (Vectorized)
    d2Tdx2 = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2
    d2Tdy2 = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx**2
    
    T_new = T.copy()
    T_new[1:-1, 1:-1] = (T[1:-1, 1:-1] + 
                         alpha * dt * (d2Tdx2 + d2Tdy2) - 
                         cooling_coeff * (T[1:-1, 1:-1] - ambient_temp) * dt)
    T = T_new

    # 2. Continuous Motion Controller for the Heat Source
    source_active = True
    if col_source < Ny + source_radius:
        col_source += 1
    else:
        source_active = False

    # 3. Map Symmetry-Corrected Energy Input to the Grid Coordinates
    if source_active:
        # Constrain loops strictly around the local operational radius
        r_start_i = max(0, row_source - source_radius)
        r_end_i = min(Nx, row_source + source_radius + 1)
        r_start_j = max(0, col_source - source_radius)
        r_end_j = min(Ny, col_source + source_radius + 1)
        
        for i in range(r_start_i, r_end_i):
            for j in range(r_start_j, r_end_j):
                r2 = ((i - row_source) ** 2 + (j - col_source) ** 2)
                # Apply localized exponential decay
                Q = source_strength * np.exp(-r2 / 2600)
                T[i, j] += (Q * dt) / (rho * cp)

    # 4. Enforce Fixed Ambient Boundary Conditions
    T[0, :] = ambient_temp
    T[-1, :] = ambient_temp
    T[:, 0] = ambient_temp
    T[:, -1] = ambient_temp

    # 5. Extract Dynamic Quantitative Telemetry
    max_t = np.max(T)
    melted_cells = np.sum(T > Tm)
    
    # 6. Refresh Graphics Frame Buffers
    im.set_data(T)
    
    # MODERN API FIX: Remove the contour line object cleanly from the axis before regenerating
    if contour_line is not None:
        contour_line.remove()
        
    contour_line = ax.contour(T, levels=[Tm], colors='cyan', linewidths=2, origin='lower')
    
    ax.set_title(f"Explicit Thermal Field Visualizer\nStep: {frame} | Max T: {max_t:.1f} K | Melted Pool Elements: {melted_cells}", fontsize=11)
    
    return [im]

# =========================================================================
# 6. RUN THE RENDERING ENGINE LOOP
# =========================================================================
print("Starting explicit thermal tracker animation pipeline...")
ani = FuncAnimation(fig, update_thermal_field, frames=400, interval=20, blit=False, repeat=False)
plt.show()
