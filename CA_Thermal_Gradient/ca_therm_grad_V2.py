import numpy as np
import matplotlib.pyplot as plt

# Create grid
Nx = 100
Ny = 100

state = np.zeros((Nx, Ny))
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
# Run solidification - either in a fixed number of steps or until all liquid has solidified
#for step in range(100):
while np.any(state == 0):
    T = T - 2  # Cool down the system
    step += 1
    new_state = state.copy()

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            if state[i,j] == 0:

                neighborhood = state[i-1:i+2, j-1:j+2]

                neighboring_grains = neighborhood[neighborhood > 0]

                if T[i,j] < Tm and len(neighboring_grains) > 0:

                    chosen_grain = np.random.choice(neighboring_grains)

                    new_state[i,j] = chosen_grain

    state = new_state

    # Display animation
    plt.clf()
    plt.imshow(state, cmap='jet')
    plt.title(f"Step {step}")
    plt.pause(0.05)

plt.show()
