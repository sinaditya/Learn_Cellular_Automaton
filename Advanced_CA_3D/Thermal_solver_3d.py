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

def thermal_conductivity(T):
    return np.where(T < T_melt , k_max , k_min)

def specific_heat(T):
    return np.where(T < T_melt , cp_min , cp_max)

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
T0 = 300.0

T_old = np.ones((Nx,Ny,Nz))*T0
T_new = T_old.copy()


#laser parameters
P = 35.0          # power (W)
r = 0.0005         # beam radius (m)
v = 0.05           # velocity (m/s)

A = (2*eta*P)/(np.pi*r**2) #gaussian coefficient

#coordinates' arrays
x = np.arange(Nx)*dx
y = np.arange(Ny)*dy
z = np.arange(Nz)*dz

#top surface coordinates
X, Y = np.meshgrid(x, y, indexing='ij')

top = Nz-1

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

    #boundary conditions - ideal heat sinks as constant temperature
    T_new[0,:,:]  = T_new[1,:,:]
    T_new[-1,:,:] = T_new[-2,:,:]
    T_new[:,0,:]  = T_new[:,1,:]
    T_new[:,-1,:] = T_new[:,-2,:]
    T_new[:,:,0] = T0
    #T_new[:,:,-1] = T0

    #time-dependent laser source
    time = n*dt
    x_center = 0.0005 +v*time
    y_center = Ly/2

    R2 = ((X - x_center)**2 + (Y - y_center)**2) #distance^2

    q = A*np.exp(-3*R2/r**2) #laser heat flux
    source_term = (q*dt)/(rho*cp_local[:,:,top]*dz) #convert to temperature increase
    T_new[:,:,top] += source_term #add to top surface

    # convection cooling
    cooling = (h*(T_new[:,:,top] - T0)*dt)/(rho*cp_local[:,:,top]*dz)
    T_new[:,:,top] -= cooling

    #radiation
    radiation = (epsilon*sigma*(T_new[:,:,top]**4 - T0**4)*dt)/(rho*cp_local[:,:,top]*dz)
    T_new[:,:,top] -= radiation

    #observation- middle z plane
    mid_z = Nz//2

    #plot every 20 steps
    if n%20 == 0:
        plt.clf()
        plt.imshow(T_new[:,Ny//2,:].T , origin='lower', cmap='hot', extent=[0,Lx*1000,0,Ly*1000], aspect='auto', vmin = 300, vmax = 1500) # for top layer visualisation
        plt.colorbar(label='Temperature (K)')
        plt.title(f"Step {n}")
        plt.pause(0.01)

        print(x_center)
        print("Max T =", np.max(T_new))

    T_old, T_new = T_new, T_old #swap references for next iteration

plt.show()