import numpy as np
import matplotlib.pyplot as plt

''' Everything will be in SI units (m, s, K, etc.) '''

T_melt = 933.0     # melting point (K)
k_max = 237.0 
k_min = 120.0      # thermal conductivity (W/mK)
rho = 2700.0       # density (kg/m3)
cp_min = 900.0     
cp_max = 1100.0    # specific heat (J/kgK)
eta = 0.3          # absorptivity(1/m)  
epsilon = 0.3      # emmisivity
sigma = 5.67e-8    # stephen-boltzmann constant
h = 50.0           # convection coefficient (W/m2K)
beta = 12000.0      # penetration decay coefficient (1/m)

def thermal_conductivity(T):
    transition = 0.5*(1 + np.tanh((T_melt - T)/50))
    return k_min + (k_max-k_min)*transition

def specific_heat(T):
    transition = 0.5*(1 + np.tanh((T_melt - T)/50))
    return cp_max + (cp_min-cp_max)*transition

Lx = 0.004      # 4 mm
Ly = 0.002      # 2 mm
Lz = 0.001      # 1 mm

#size of 1 voxel 
dx = 50e-6
dy = dx
dz = dx

# number of grid points
Nx = int(Lx/dx)
Ny = int(Ly/dy)
Nz = int(Lz/dz)

#stability condition
alpha_max = 237.0/(rho*900.0)
dt_stable = dx**2/(6*alpha_max)
dt = 0.5*dt_stable


#diagnosis 
print("Grid size:", Nx, Ny, Nz)
print("Stable dt:", dt_stable)
print("Using dt:", dt)

#running simulation for __ seconds
total_time = 0.02

#number of steps
Nt = int(total_time/dt)

#initial condition: T=300 K everywhere
T0 = 301.0
T_boundary = 301.0
T_old = np.ones((Nx,Ny,Nz))*T0
T_new = T_old.copy()


#laser parameters
P = 30.0          # power (W)
r = 0.0002         # beam radius (m)
v = 0.05           # velocity (m/s)

A = (2*eta*beta*P)/(np.pi*r**2) #gaussian coefficient , volumetric source strength(W/m3)

#coordinates' arrays
x = np.arange(Nx)*dx
y = np.arange(Ny)*dy
z = np.arange(Nz)*dz

#top surface depth coordinates
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

top = Nz-1
depth_decay = np.exp(-beta*(Lz - Z)) #exponential decay with depth


#data storage
temperature_history = []
time_history = []
gradient_mag_history = []
gradient_x_history = []
gradient_y_history = []
gradient_z_history = []
cooling_rate_history = []
melt_pool_history = []

previous_saved_temperature = None   
previous_saved_time = None



#solver
for n in range(Nt):

    #local properties   
    k_local = thermal_conductivity(T_old)
    cp_local = specific_heat(T_old)
    alpha_local = k_local/(rho*cp_local)

    Fo_local = alpha_local*dt/dx**2

    #calculatelaplacian
    laplacian = (T_old[2:,1:-1,1:-1] + T_old[:-2,1:-1,1:-1] + T_old[1:-1,2:,1:-1] + T_old[1:-1,:-2,1:-1] + T_old[1:-1,1:-1,2:]+ T_old[1:-1,1:-1,:-2] - 6*T_old[1:-1,1:-1,1:-1])

    #update temperature field
    T_new[1:-1,1:-1,1:-1] = (T_old[1:-1,1:-1,1:-1]+ Fo_local[1:-1,1:-1,1:-1] * laplacian)

    #time-dependent laser source
    time = n*dt
    x_center = 0.0005 +v*time
    y_center = Ly/2

    R2 = ((X - x_center)**2 + (Y - y_center)**2) #distance^2

    q = A*np.exp(-3*R2/r**2)*depth_decay #laser heat flux
    source_term = (q*dt)/(rho*cp_local) #convert to temperature increase
    T_new += source_term #add upto a depth from top surface

    # convection cooling
    cooling = (h*(T_new[:,:,top] - T0)*dt)/(rho*cp_local[:,:,top]*dz)
    T_new[:,:,top] -= cooling

    #radiation
    radiation = (epsilon*sigma*(T_new[:,:,top]**4 - T0**4)*dt)/(rho*cp_local[:,:,top]*dz)
    T_new[:,:,top] -= radiation

    #boundary conditions - ideal heat sinks as constant temperature
    T_new[0,:,:]  = T_boundary
    T_new[-1,:,:] = T_boundary
    T_new[:,0,:]  = T_boundary
    T_new[:,-1,:] = T_boundary
    T_new[:,:,0] = T0
    #T_new[:,:,-1] = T0

    #observation- middle z plane
    mid_z = Nz//2

    #plot every 20 steps
    if n%20 == 0:
        plt.clf()
        plt.imshow(T_new[:,Ny//2,:].T , origin='lower', cmap='hot', extent=[0,Lx*1000,0,Lz*1000], aspect='auto', vmin = 300, vmax = 940) # for top layer visualisation
        plt.colorbar(label='Temperature (K)')
        plt.title(f"Step {n}")
        plt.pause(0.01)

    
    if n%100 == 0:
        T_snapshot = T_new.copy() # SAVE TEMPERATURE FIELD
        temperature_history.append(T_snapshot.astype(np.float32)) # store as float32 to save memory
        time_history.append(time)

        dTdx = np.gradient(T_snapshot, dx, axis=0)  # THERMAL GRADIENT
        dTdy = np.gradient(T_snapshot, dy, axis=1)
        dTdz = np.gradient(T_snapshot, dz, axis=2)
        G = np.sqrt(dTdx**2 + dTdy**2 + dTdz**2)
        gradient_mag_history.append(G.astype(np.float32))
        gradient_x_history.append(dTdx.astype(np.float32))
        gradient_y_history.append(dTdy.astype(np.float32))
        gradient_z_history.append(dTdz.astype(np.float32))

        melt_pool = T_snapshot >= T_melt  # MELT POOL MASK
        melt_pool_history.append(melt_pool)

        if previous_saved_temperature is None:  # COOLING RATE
            cooling_rate = np.zeros_like(T_snapshot)

        else:
            dt_frame = time - previous_saved_time
            cooling_rate = (previous_saved_temperature - T_snapshot) / dt_frame

        cooling_rate_history.append(cooling_rate.astype(np.float32))

        # update previous frame
        previous_saved_temperature = T_snapshot.copy()
        previous_saved_time = time


        # DIAGNOSTICS
        total_energy = np.sum(rho*cp_local*(T_new - T0))*dx*dy*dz

        print("Energy =", total_energy)
        print("x_center =", x_center)
        print("Max T =", np.max(T_new))
        print("Max Fo =", np.max(Fo_local))
        print("Max G =", np.max(G))
        print("Min cooling =", np.min(cooling_rate))


    T_old, T_new = T_new, T_old #swap references for next iteration


temperature_history = np.array(temperature_history)
time_history = np.array(time_history)

gradient_mag_history = np.array(gradient_mag_history)
gradient_x_history = np.array(gradient_x_history)
gradient_y_history = np.array(gradient_y_history)
gradient_z_history = np.array(gradient_z_history)

cooling_rate_history = np.array(cooling_rate_history)
melt_pool_history = np.array(melt_pool_history)

# SAVE FILES
np.save("temperature_history.npy", temperature_history)
np.save("time_history.npy", time_history)
np.save("gradient_mag_history.npy", gradient_mag_history)
np.save("gradient_x_history.npy", gradient_x_history)
np.save("gradient_y_history.npy", gradient_y_history)
np.save("gradient_z_history.npy", gradient_z_history)
np.save("cooling_rate_history.npy", cooling_rate_history)
np.save("melt_pool_history.npy", melt_pool_history)
print("All thermal metrics exported.")

plt.show()
