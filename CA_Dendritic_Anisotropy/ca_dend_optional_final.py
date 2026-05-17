import numpy as np
import matplotlib.pyplot as plt

# Create grid
Nx = 300
Ny = 300

state = np.zeros((Nx, Ny), dtype=int) # 0: liquid, >0: solid with grain ID, -1: non-interactive boundary
T = np.zeros((Nx,Ny))

# Initialize temperature field (linear gradient)
for i in range(Nx):

    T[i,:] = 800 + (1200-800)*(i/Nx)


# Visualize initial temperature field
plt.imshow(T, cmap='hot')
plt.colorbar()
plt.title("Temperature Field")
plt.show()

#Put multiple nuclei---------
num_nuclei = 20

orientation = np.zeros(num_nuclei + 1) #assign a random orientation to each grain
for grain_id in range(1, num_nuclei + 1):

    orientation[grain_id] = np.random.uniform(0,180) #now every grain has a random orientation between 0 and 180 degrees

for grain_id in range(1, num_nuclei+1):

    x = np.random.randint(0, Nx)
    y = np.random.randint(0, Ny)

    state[x,y] = grain_id
#-----------------------------
# Set boundaries to -1 (non-interactive), else the while loop will never end as the boundaries will always be liquid (0) and never solidify. Thus, infinite loop.
state[0,:] = -1
state[-1,:] = -1
state[:,0] = -1
state[:,-1] = -1
# Define melting temperature
Tm = 1000
step = 0
k = 0.01 #probability scaling factor
theta_pref = 90 # preferred orientation
max_steps = 500 # maximum number of steps to prevent infinite loop in case of issues

while np.any(state == 0) and step < max_steps:
    T = T - 1  # Cool down the system
    step += 1
    new_state = state.copy()

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            if state[i,j] == 0:

                if T[i,j] < Tm:

                    deltaT = Tm - T[i,j]

                    # Check every neighboring cell individually
                    captured = False

                    for ni in range(i-1, i+2):

                        if captured:
                            break

                        for nj in range(j-1, j+2):

                            if ni == i and nj == j:
                                continue

                            neighbor_grain = state[ni,nj]

                            if neighbor_grain > 0:

                                di = ni - i
                                dj = nj - j

                                direction_angle = np.degrees(np.arctan2(di,dj))
                                theta_grain = orientation[int(neighbor_grain)]
                                misalignment = abs((direction_angle - theta_grain + 180) % 360 - 180)
                                anisotropy_strength = 0.3

                                anisotropy_factor = 1 + anisotropy_strength * np.cos(np.radians(4 * misalignment))
                                P_growth = k * deltaT * anisotropy_factor

                                P_growth = min(P_growth,1)

                                if np.random.random() < P_growth:

                                    new_state[i,j] = neighbor_grain
                                    captured = True
                                    break
    state = new_state

    # Display animation
    plt.clf()
    plt.imshow(state, cmap='jet')
    plt.title(f"Step {step}")
    plt.pause(0.01)

plt.show()
