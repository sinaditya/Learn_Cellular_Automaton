import numpy as np
import matplotlib.pyplot as plt

# Create grid
Nx = 100
Ny = 100

state = np.zeros((Nx, Ny))

# Put one solid nucleus in center
#state[50, 50] = 1

#Put multiple nuclei---------
num_nuclei = 20

for grain_id in range(1, num_nuclei+1):

    x = np.random.randint(0, Nx)
    y = np.random.randint(0, Ny)

    state[x,y] = grain_id
#-----------------------------

# Run solidification
for step in range(50):

    new_state = state.copy()

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            # Only liquid cells
            if state[i, j] == 0:

                # Moore neighborhood
                neighborhood = state[i-1:i+2, j-1:j+2]

                # If any neighbor solidifies
                neighboring_grains = neighborhood[neighborhood > 0]

                #--------For random grains-------
                if len(neighboring_grains) > 0:

                    chosen_grain = np.random.choice(neighboring_grains)
                    new_state[i,j] = chosen_grain

                #-------For seed at center----------
                #if np.any(neighborhood == 1):

                   # new_state[i, j] = 1

    state = new_state

    # Display animation
    plt.clf()
    #--------------------
    #plt.imshow(state)

    #--------------------
    plt.imshow(state, cmap='jet')
    #--------------------
    plt.title(f"Step {step}")
    plt.pause(0.05)

plt.show()