import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# GRID SIZE
# =========================================================

Nx = 300
Ny = 300

# =========================================================
# THERMAL PARAMETERS
# =========================================================

alpha = 0.25
dt = 0.1
dx = 1

Tm = 950
ambient_temp = 300
cooling_coeff = 0.07
rho = 1
cp = 1
# =========================================================
# HEAT SOURCE PARAMETERS
# =========================================================

row_source = 150
col_source = 10

source_radius = 28
source_strength = 300

# =========================================================
# INITIAL TEMPERATURE FIELD
# =========================================================

T = np.ones((Nx, Ny)) * ambient_temp


# THERMAL GRADIENT ARRAYS
Gx = np.zeros((Nx, Ny))
Gy = np.zeros((Nx, Ny))
G = np.zeros((Nx, Ny))

# CELLULAR AUTOMATON STATE MATRIX
state = np.zeros((Nx, Ny), dtype=int)

# Inactive boundaries
state[0, :] = -1
state[-1, :] = -1
state[:, 0] = -1
state[:, -1] = -1


# MELT TRACKING MATRIX
melted = np.zeros((Nx, Ny), dtype=bool)

# SUBSTRATE GRAIN INITIALIZATION
num_nuclei = 2000

orientation = np.zeros(num_nuclei + 1)

# Place nuclei
for grain_id in range(1, num_nuclei + 1):

    row = np.random.randint(1, Nx - 1)
    col = np.random.randint(1, Ny - 1)

    state[row, col] = grain_id

    orientation[grain_id] = np.random.uniform(0, 180)

# PRE-GROW SUBSTRATE TO FILL DOMAIN
for fill_step in range(200):

    new_state = state.copy()

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if state[i, j] == 0:

                neighborhood = state[i - 1:i + 2, j - 1:j + 2]
                neighboring_grains = neighborhood[neighborhood > 0]

                if len(neighboring_grains) > 0:

                    chosen_grain = np.random.choice(neighboring_grains)
                    new_state[i, j] = chosen_grain

    state = new_state

# GROWTH PARAMETERS
k = 0.06
anisotropy_strength = 0.4

# VISUALIZATION SETUP
fig, ax = plt.subplots()
im = ax.imshow(state,cmap='jet',origin='lower')
plt.colorbar(im)

# MAIN SIMULATION LOOP
for step in range(350): 

    T_old = T.copy()

    if col_source < Ny - source_radius - 1:
        col_source += 1

    T_new = T.copy()

    for i in range(1, Nx - 1):

        for j in range(1, Ny - 1):

            d2Tdx2 = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx**2
            d2Tdy2 = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dx**2
            T_new[i, j] = (T[i, j] + alpha * dt * (d2Tdx2 + d2Tdy2) - cooling_coeff * (T[i, j] - ambient_temp) * dt)

    T = T_new

    for i in range(row_source - source_radius,row_source + source_radius + 1):
        for j in range(col_source - source_radius,col_source + source_radius + 1):

            if 0 <= i < Nx and 0 <= j < Ny:

                r2 = ((i - row_source) ** 2 + (j - col_source) ** 2)
                Q = source_strength * np.exp(-r2 / 1200)

                T[i,j] += (Q * dt) / (rho * cp)


    T[0,:] = ambient_temp
    T[-1,:] = ambient_temp
    T[:,0] = ambient_temp
    T[:,-1] = ambient_temp
    
    #print("MAX TEMP =", np.max(T))

    print(
    "MAX =", np.max(T),
    "MIN =", np.min(T),
    "CELLS ABOVE Tm =",
    np.sum(T > Tm)
)

    # MELTING LOGIC
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):

            if T[i, j] > Tm:
                state[i, j] = 0
                melted[i, j] = True

    
    # COMPUTE THERMAL GRADIENTS
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):

            Gx[i, j] = (T[i, j + 1] - T[i, j - 1]) / (2 * dx)
            Gy[i, j] = (T[i + 1, j] - T[i - 1, j]) / (2 * dx)
            G[i, j] = np.sqrt(Gx[i, j]**2 + Gy[i, j]**2)

    # COOLING RATE
    cooling_rate = (T_old - T) / dt

    # CA SOLIDIFICATION UPDATE
    new_state = state.copy()
    cells = []

    for i in range(1, Nx - 1):
                    for j in range(1, Ny - 1):

                        if state[i, j] == 0 and melted[i, j]:
                            cells.append((i, j))

    np.random.shuffle(cells)

    for i, j in cells:
        # Only previously melted liquid may solidify
        if (state[i, j] == 0 and melted[i, j] and T[i, j] < Tm):

                deltaT = Tm - T[i, j]

                captured = False
                #randomise neighbour checking to prevent directional growth as in photo of 1st trial
                neighbors = []

                for ni in range(i - 1, i + 2):
                    for nj in range(j - 1, j + 2):

                        if ni == i and nj == j:
                            continue

                        neighbors.append((ni, nj))

                np.random.shuffle(neighbors)

                for ni, nj in neighbors:

                    if captured:
                        break

                    neighbor_grain = state[ni, nj]

                    if neighbor_grain > 0:

                            # Direction toward neighbor
                            di = ni - i
                            dj = nj - j

                            direction_angle = np.degrees(np.arctan2(di, dj))

                            # Grain orientation
                            theta_grain = orientation[int(neighbor_grain)]

                            # Crystallographic misalignment
                            misalignment = abs((direction_angle - theta_grain + 180) % 360 - 180)

                            # Dendritic anisotropy
                            anisotropy_factor = (1+ anisotropy_strength * np.cos(np.radians(4 * misalignment)))

                            # Thermal gradient direction
                            gradient_angle = np.degrees(np.arctan2(Gy[i, j],Gx[i, j]))

                            # Thermal alignment
                            thermal_misalignment = abs((theta_grain - gradient_angle + 180) % 360 - 180)
                            thermal_factor = abs(np.cos(np.radians(thermal_misalignment)))

                            # Growth probability
                            P_growth = (k * deltaT * anisotropy_factor * thermal_factor)
                            P_growth = min(P_growth, 1)

                            # Probabilistic capture
                            if np.random.random() < P_growth:

                                new_state[i, j] = neighbor_grain
                                captured = True
                                break

    # Update microstructure
    state = new_state


    # VISUALIZATION
    im.set_data(state)

    ax.set_title(
        f"Coupled CA-Thermal Welding Simulation | Step {step}"
    )

    plt.pause(0.001)

plt.show()
