import numpy as np
import pyvista as pv

# =========================================================
# GRID SETUP
# =========================================================
Nx, Ny, Nz = 80, 40, 20

# grain ID field
# 0 = liquid
# >0 = grain ID
state = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

# =========================================================
# RANDOM NUCLEI
# =========================================================
num_nuclei = 15

# different nuclei every run
np.random.seed(None)

for g in range(1, num_nuclei + 1):

    x = np.random.randint(1, Nx - 1)
    y = np.random.randint(1, Ny - 1)
    z = np.random.randint(1, Nz - 1)

    state[x, y, z] = g

# =========================================================
# TEMPERATURE FIELD
# =========================================================
T_melt = 933.0

# initial temperature field
T = np.full((Nx, Ny, Nz), 1000.0)

# directional thermal gradient
for x in range(Nx):

    T[x, :, :] = 1000 - 3*x

# cooling per timestep
cooling_rate = 5.0

# =========================================================
# 26-NEIGHBORHOOD
# =========================================================
neighbors = [

    (dx, dy, dz)

    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)

    if not (dx == 0 and dy == 0 and dz == 0)
]

# =========================================================
# THERMAL CA STEP
# =========================================================
def step(state, T):

    new_state = state.copy()

    # exclude outer boundaries
    for x in range(1, Nx - 1):
        for y in range(1, Ny - 1):
            for z in range(1, Nz - 1):

                # only liquid cells
                if state[x, y, z] == 0:

                    # thermal solidification condition
                    if T[x, y, z] < T_melt:

                        # check neighboring grains
                        for dx, dy, dz in neighbors:

                            ng = state[
                                x + dx,
                                y + dy,
                                z + dz
                            ]

                            # grain found
                            if ng > 0:

                                # inherit grain ID
                                new_state[x, y, z] = ng
                                break

    return new_state

# =========================================================
# TRUE GRAIN-BOUNDARY DETECTION
# =========================================================
def compute_boundaries(state):

    boundary = np.zeros_like(
        state,
        dtype=np.uint8
    )

    # compare against all neighbors
    for dx, dy, dz in neighbors:

        shifted = np.roll(
            state,
            shift=(dx, dy, dz),
            axis=(0, 1, 2)
        )

        # interface exists if grain IDs differ
        boundary |= (

            (shifted != state) &
            (state > 0)
        )

    # remove artificial periodic-wrap edges
    boundary[0,:,:] = 0
    boundary[-1,:,:] = 0

    boundary[:,0,:] = 0
    boundary[:,-1,:] = 0

    boundary[:,:,0] = 0
    boundary[:,:,-1] = 0

    return boundary.astype(np.uint8)

# =========================================================
# PYVISTA SETUP
# =========================================================
plotter = pv.Plotter()

# optional outer domain box
plotter.add_mesh(
    pv.Box(bounds=(0, Nx, 0, Ny, 0, Nz)),
    style="wireframe",
    color="black",
    line_width=1,
    opacity=0.15
)

# =========================================================
# MAIN GRAIN GRID
# =========================================================
grid = pv.ImageData()

grid.dimensions = np.array(state.shape) + 1
grid.spacing = (1, 1, 1)

grid.cell_data["grain"] = state.flatten(order="F")

# =========================================================
# INITIAL BOUNDARY FIELD
# =========================================================
initial_boundary = compute_boundaries(state)

# separate boundary-only grid
boundary_grid = pv.ImageData()

boundary_grid.dimensions = np.array(state.shape) + 1
boundary_grid.spacing = (1, 1, 1)

boundary_grid.cell_data["boundary"] = (
    initial_boundary.flatten(order="F")
)

# =========================================================
# INITIAL BOUNDARY MESH
# =========================================================
boundary_mesh = boundary_grid.threshold(0.5)

# -------------------------
# GRAIN VOLUME
# -------------------------
grain_mesh = grid.threshold(0.5)

grain_actor = plotter.add_mesh(
    grain_mesh,
    scalars="grain",
    cmap="tab20",
    opacity=0.7,
    show_edges=False
)

# -------------------------
# BOUNDARY NETWORK
# -------------------------
boundary_actor = plotter.add_mesh(
    boundary_mesh,
    color="black",
    opacity=0.3
)
# =========================================================
# SHOW WINDOW
# =========================================================
plotter.show(
    interactive_update=True,
    auto_close=False
)

# =========================================================
# LIVE SIMULATION LOOP
# =========================================================
n_steps = 37

for t in range(n_steps):

    # -----------------------------------------
    # COOL DOMAIN
    # -----------------------------------------
    T -= cooling_rate

    # -----------------------------------------
    # CA UPDATE
    # -----------------------------------------
    state = step(state, T)

    # -----------------------------------------
    # COMPUTE TRUE BOUNDARIES
    # -----------------------------------------
    boundary = compute_boundaries(state)

    # -----------------------------------------
    # UPDATE GRAIN FIELD
    # -----------------------------------------
    grid.cell_data["grain"][:] = (
        state.flatten(order="F")
    )

    # -----------------------------------------
    # UPDATE BOUNDARY FIELD
    # -----------------------------------------
    boundary_grid.cell_data["boundary"][:] = (
        boundary.flatten(order="F")
    )

    # -----------------------------------------
    # GENERATE BOUNDARY VOXEL MESH
    # -----------------------------------------
    boundary_mesh = boundary_grid.threshold(0.5)

    # update grain geometry
    grain_mesh = grid.threshold(0.5)
    grain_actor.mapper.dataset = grain_mesh

    # update boundary geometry
    boundary_actor.mapper.dataset = boundary_mesh

    # -----------------------------------------
    # RENDER FRAME
    # -----------------------------------------
    plotter.render()

    # -----------------------------------------
    # TERMINAL OUTPUT
    # -----------------------------------------
    print(
        f"Step {t:3d} | "
        f"Solid Cells = {np.sum(state > 0):6d} | "
        f"Tmin = {T.min():7.2f}"
    )

# =========================================================
# KEEP WINDOW OPEN
# =========================================================
plotter.show()