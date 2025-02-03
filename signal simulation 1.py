# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:55:22 2024

@author: av439
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:02:17 2024

@author: av439
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:30:21 2024

@author: av439
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define time parameters
T = 1

# Dimensions
road_length = 2.0  # in km
road_width = 0.0075  # in km

# Number of grid cells
nx = 100  # along the length of the road
ny = 10 # along the width of the road

# Spatial step sizes
dx = road_length / nx  # spatial step size in km along the length
dy = road_width / ny   # spatial step size in km along the width

# Define constants
W =0.0075  # Width of the road in km
r_i = {'A': 0.89, 'B': 0.78, 'C': 0.74, 'D': 0.50}
D_A, D_B, D_C, D_D = 0.0000006, 0.0000005, 0.0000004, 0.0000003  # Diffusion coefficients in km^2/h
a_A, a_B, a_C, a_D = 0.0018 * 0.0006, 0.0026 * 0.0014, 0.004 * 0.0016, 0.0065 * 0.0022
# Maximum AO values for each class
AO_max_A, AO_max_B, AO_max_C, AO_max_D = 0.80, 0.78, 0.68, 0.50
# Initialize densities with random values
np.random.seed(0)  # For reproducibility

epsilon = 1e-10
# Target average densities for the different regions
target_density_A = 1
target_density_B = 1
target_density_C = 0.5
target_density_D = 0.1

# Generate random density fields and normalize
def generate_random_density(target_density, shape):
    random_density = np.random.rand(*shape)  # Random values between 0 and 1
    normalized_density = random_density * target_density * (1 / np.mean(random_density)+ epsilon)
    capped_density = np.minimum(normalized_density, target_density)  # Cap values at target density
    return capped_density
# Initialize densities
rho_A = generate_random_density(target_density_A, (nx, ny))
rho_B = generate_random_density(target_density_B, (nx, ny))
rho_C = generate_random_density(target_density_C, (nx, ny))
rho_D = generate_random_density(target_density_D, (nx, ny))
#generate initial flow matrices
#assumption delta AO' is 10^-5 initially, derivative found as delta AO'/delta x and delta y respectively delta x is total length /no of grids similarly for y
# Function to generate random flow with specified average
def generate_random_Q(Q_target, shape):
    random_Q = np.random.rand(*shape)  # Random values between 0 and 1
    return random_Q * Q_target * (1 / np.mean(random_Q))  # Scale to target velocity

# Target average flow (in veh/kmhr)
Q_target_A_x, Q_target_A_y = 45, 2.5
Q_target_B_x, Q_target_B_y = 42.1, 1.4
Q_target_C_x, Q_target_C_y = 26.4, 2.5
Q_target_D_x, Q_target_D_y = 4.72,0.43

# Initialize flow
Q_x_A = generate_random_Q(Q_target_A_x, (nx, ny))
Q_y_A = generate_random_Q(Q_target_A_y, (nx, ny))
Q_x_B = generate_random_Q(Q_target_B_x, (nx, ny))
Q_y_B = generate_random_Q(Q_target_B_y, (nx, ny))
Q_x_C = generate_random_Q(Q_target_C_x, (nx, ny))
Q_y_C = generate_random_Q(Q_target_C_y, (nx, ny))
Q_x_D = generate_random_Q(Q_target_D_x, (nx, ny))
Q_y_D = generate_random_Q(Q_target_D_y, (nx, ny))

# Initialize AO matrix
AO = a_A * rho_A + a_B * rho_B + a_C * rho_C + a_D * rho_D

# Function to generate random velocity fields with specified average
def generate_random_velocity(u_target, shape):
    random_velocity = np.random.rand(*shape)  # Random values between 0 and 1
    return random_velocity * u_target * (1 / np.mean(random_velocity))  # Scale to target velocity

# Target average velocities (in km/h)
u_target_A_x, u_target_A_y = 45.0, 2.5
u_target_B_x, u_target_B_y = 42.1, 1.4
u_target_C_x, u_target_C_y = 52.9, 5.6
u_target_D_x, u_target_D_y = 47.2, 4.3

# Initialize velocity fields
u_x_A = generate_random_velocity(u_target_A_x, (nx, ny))
u_y_A = generate_random_velocity(u_target_A_y, (nx, ny))
u_x_B = generate_random_velocity(u_target_B_x, (nx, ny))
u_y_B = generate_random_velocity(u_target_B_y, (nx, ny))
u_x_C = generate_random_velocity(u_target_C_x, (nx, ny))
u_y_C = generate_random_velocity(u_target_C_y, (nx, ny))
u_x_D = generate_random_velocity(u_target_D_x, (nx, ny))
u_y_D = generate_random_velocity(u_target_D_y, (nx, ny))

# Define inflow values (vehicles per hour)
q_A_inflow = 50
q_B_inflow = 80
q_C_inflow = 70
q_D_inflow = 60

# Determine maximum velocities
u_x_max = max(np.max(u_x_A), np.max(u_x_B), np.max(u_x_C), np.max(u_x_D))
u_y_max = max(np.max(u_y_A), np.max(u_y_B), np.max(u_y_C), np.max(u_y_D))

# Determine maximum diffusion coefficient
D_max = max(D_A, D_B, D_C, D_D)

# Calculate the time step to satisfy CFL conditions
dt_advection_x = dx / u_x_max
dt_advection_y = dy / u_y_max
dt_diffusion_x = dx**2 / (2 * D_max)
dt_diffusion_y = dy**2 / (2 * D_max)

# Choose the smaller time step to ensure all conditions are satisfied
dt = min(dt_advection_x, dt_advection_y, dt_diffusion_x, dt_diffusion_y)

# Calculate number of time steps
time_steps = int(T/dt)
time = np.linspace(0, T, time_steps)

# Define stopping region and time-dependent stopping function
stop_start_interval = 35 / 3600  # Interval in hours (35 seconds)
#stop_length = 30/ 1000  # Length in km (20 meters)
#slowdown_length = 20 / 1000  # Length in km (20 meters)
stop_position = 1  # Position in km (1 kilometer)

stop_start_times = np.arange(0, T, 2 * stop_start_interval)
stop_end_times = np.arange(stop_start_interval, T, 2 * stop_start_interval)

def is_stopping(t):
    for start, end in zip(stop_start_times, stop_end_times):
        if start <= t < end: #if a certain time falls in between stop time interval this function will be true
            return True
    return False

def apply_stopping_behavior_u(u_x, u_y, t):
    if is_stopping(t): #if is stopping function is true
        stopping_region = (np.arange(nx) * dx == stop_position)
        u_x[stopping_region, :] = 0
        u_y[stopping_region, :] = 0
    return u_x, u_y
#np.arange(nx):Creates an array of integers from 0 to nx - 1.
def apply_stopping_behavior_f(flux_x, flux_y, t):
    if is_stopping(t): #if is stopping function is true
        stopping_region = (np.arange(nx) * dx == stop_position) 
        flux_x[stopping_region, :] = 0
        flux_y[stopping_region, :] = 0
    return flux_x, flux_y
epsilon = 1e-10
# Function to calculate fluxes using upwind scheme in FVM
def calculate_fluxes_upwind(Q_x, Q_y,AO, AO_minus,a, u_x, u_y, D, dx, dy):
    # Ensure u_x, u_y, and rho are always > 0
    u_x = np.abs(u_x)
    u_y = np.abs(u_y)

    # Initialize fluxes with Q values
    flux_x = Q_x.copy()  # Use the provided Q_x as the initial flux_x
    flux_y = Q_y.copy()  # Use the provided Q_y as the initial flux_y
    
    
    """# Replace denominators that are zero or close to zero with epsilon
    #dx_safe = np.where((np.isclose(dx,0)), epsilon, dx)
    #dy_safe = np.where((np.isclose(dy,0)), epsilon, dy)
    a_safe = np.where((np.isclose(a,0)), epsilon, a)
    
    # Replace denominators that are zero or close to zero with epsilon
    dx_safe = np.where(dx == 0, epsilon, dx)
    dy_safe = np.where(dy == 0, epsilon, dy)
    a_safe = np.where(a == 0, epsilon, a)"""
    #Compute fluxes in the x direction
    #flux_x[1:, :] = (1 / a) * (u_x[1:, :] * (AO[1:, :] - AO_minus[1:, :]) - D * ((AO[1:, :] - AO_minus[1:, :]) - (AO[:-1, :] - AO_minus[:-1, :]))) / (dx + epsilon)
    #starting from 1 because if we start from zero there is nothing before it
    # Compute fluxes in the y direction
    #flux_y[:, 1:] = (1 / a) * (u_y[:, 1:] * (AO[:, 1:] - AO_minus[:, 1:]) - D * ((AO[:, 1:] - AO_minus[:, 1:]) - (AO[:, :-1] - AO_minus[:, :-1]))) /  (dy + epsilon)

    #return flux_x, flux_y, flux_x + flux_y #finding flux through this method to find density as AO itself will get updated using density
    for i in range(1, Q_x.shape[0]):
        for j in range(Q_x.shape[1]):
            flux_x[i, j]=(1/a)*((u_x[i, j] * (AO[i, j] - AO_minus[i, j]))-D*((AO[i, j] - AO_minus[i, j]) - (AO[i - 1, j] - AO_minus[i - 1, j]) / (dx + epsilon)))
            flux_x[i,j]=max(0,flux_x[i,j])
    # Compute fluxes in the y direction
    for i in range(Q_y.shape[0]):
        for j in range(1, Q_y.shape[1]):
            flux_y[i,j]=(1 / a) * ((u_y[i, j] * (AO[i, j] - AO_minus[i, j]))-D*((AO[i, j] - AO_minus[i, j]) - (AO[i, j - 1] - AO_minus[i, j - 1])/(dy + epsilon)))
            flux_y[i,j]=max(0,flux_y[i,j])
    return flux_x, flux_y, flux_x + flux_y
"see before eqn 20 of the last report for the eqn" 
def update_density_fvm(rho,AO, AO_minus,a, u_x, u_y, D, q_inflow, dx, dy,Q_x, Q_y, dt):
    #inflow density at the boundary is calculated as the total number of vehicles entering per unit time, distributed uniformly across the width of the road.
    nx, ny = rho.shape
    rho_new = rho.copy()
    flux_x, flux_y, Q = calculate_fluxes_upwind(Q_x, Q_y,AO, AO_minus,a, u_x, u_y, D, dx, dy)
    # Distribute inflow uniformly to all rows at the left boundary (x=0)
    inflow = q_inflow/ (W*ny)

    # Update density using finite volume method
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # Incoming flux in x-direction
            if i == 1:
                flux_x_in = inflow #if i =1 to those cells a constant inflow is given
            else:
                flux_x_in = flux_x[i-1, j] # to other cells flux is same as previous cells

            # Outgoing flux in x-direction
            flux_x_out = flux_x[i, j] #out going flux is flux in that cell

            # Incoming flux in y-direction
            #Calculations similar to x directiom
            if j == 1:
                flux_y_in = 0  # No flux from below at the bottom boundary
            else:
                flux_y_in = flux_y[i, j-1]

            # Outgoing flux in y-direction
            if j == ny-2:
                flux_y_out = 0  # No flux to above at the top boundary
            else:
                flux_y_out = flux_y[i, j]

            # Update the density
            rho_new[i, j] = rho[i, j] - dt/dx * (flux_x_out - flux_x_in) - dt/dy * (flux_y_out - flux_y_in)
            # Ensure densities remain non-negative
            rho_new[i, j] = np.maximum(rho_new[i, j], 0.00000000001)
    return rho_new


# Initialize dictionaries to store data at target times
#data_at_targets = { 'density_A': {}, 'density_B': {}, 'density_C': {}, 'density_D': {}, 'velocity_A': {}, 'velocity_B': {}, 'velocity_C': {}, 'velocity_D': {}, 'flow_A': {}, 'flow_B': {}, 'flow_C': {}, 'flow_D': {}}




# Define a tolerance for floating-point comparisons
tolerance = 1e-5
plotted_times = set()

# Time-stepping loop
time_points = np.linspace(0, T,20 ) # To visualize at 10 points in time
time_index = 0


# Time-stepping loop
for t in time:
    # Recalculate AO
    AO = a_A * rho_A + a_B * rho_B + a_C * rho_C + a_D * rho_D



    # Update velocities
    u_x_A = u_target_A_x * (1 - np.exp(1 - np.exp(r_i['A'] *( AO_max_A / AO) - r_i['A'])))
    u_y_A = u_target_A_y * (1 - np.exp(1 - np.exp(r_i['A'] * (AO_max_A / AO) - r_i['A'])))
    u_x_B = u_target_B_x * (1 - np.exp(1 - np.exp(r_i['B'] * (AO_max_B / AO) - r_i['B'])))
    u_y_B = u_target_B_y * (1 - np.exp(1 - np.exp(r_i['B'] * (AO_max_B / AO) - r_i['B'])))
    u_x_C = u_target_C_x * (1 - np.exp(1 - np.exp(r_i['C'] * (AO_max_C / AO) - r_i['C'])))
    u_y_C = u_target_C_y * (1 - np.exp(1 - np.exp(r_i['C'] * (AO_max_C / AO) - r_i['C'])))
    u_x_D = u_target_D_x * (1 - np.exp(1 - np.exp(r_i['D'] * (AO_max_D / AO) - r_i['D'])))
    u_y_D = u_target_D_y * (1 - np.exp(1 - np.exp(r_i['D'] * (AO_max_D / AO) - r_i['D'])))

    # Apply stopping behavior
    u_x_A, u_y_A = apply_stopping_behavior_u(u_x_A, u_y_A, t)
    u_x_B, u_y_B = apply_stopping_behavior_u(u_x_B, u_y_B, t)
    u_x_C, u_y_C = apply_stopping_behavior_u(u_x_C, u_y_C, t)
    u_x_D, u_y_D = apply_stopping_behavior_u(u_x_D, u_y_D, t)

    
    
    # Calculate AO_minus for each category
    AO_minus_A = AO - a_A * rho_A
    AO_minus_B = AO - a_B * rho_B
    AO_minus_C = AO - a_C * rho_C
    AO_minus_D = AO - a_D * rho_D
    # Update densities using FVM with upwind scheme
    
    

    Q_x_A, Q_y_A, Q_A = calculate_fluxes_upwind( Q_x_A, Q_y_A, AO, AO_minus_A, a_A, u_x_A, u_y_A, D_A, dx, dy)
    Q_x_B, Q_y_B, Q_B = calculate_fluxes_upwind( Q_x_B, Q_y_B, AO, AO_minus_B, a_B, u_x_B, u_y_B, D_B, dx, dy)
    Q_x_C, Q_y_C, Q_C = calculate_fluxes_upwind(Q_x_C, Q_y_C, AO, AO_minus_C, a_C, u_x_C, u_y_C, D_C, dx, dy)
    Q_x_D, Q_y_D, Q_D = calculate_fluxes_upwind(Q_x_D, Q_y_D, AO, AO_minus_D, a_D, u_x_D, u_y_D, D_D, dx, dy)
    
    Q_x_A, Q_y_A = apply_stopping_behavior_f(Q_x_A, Q_y_A, t)
    Q_x_B, Q_y_B = apply_stopping_behavior_f(Q_x_B, Q_y_B, t)
    Q_x_C, Q_y_C = apply_stopping_behavior_f(Q_x_C, Q_y_C, t)
    Q_x_D, Q_y_D = apply_stopping_behavior_f(Q_x_D, Q_y_D, t)
    
  

    # Update densities using FVM with upwind scheme
    # Update densities using FVM with upwind scheme
    rho_A = update_density_fvm(rho_A, AO, AO_minus_A, a_A, u_x_A, u_y_A, D_A, q_A_inflow, dx, dy, Q_x_A, Q_y_A, dt)
    rho_B = update_density_fvm(rho_B, AO, AO_minus_B, a_B, u_x_B, u_y_B, D_B, q_B_inflow, dx, dy, Q_x_B, Q_y_B, dt)
    rho_C = update_density_fvm(rho_C, AO, AO_minus_C, a_C, u_x_C, u_y_C, D_C, q_C_inflow, dx, dy, Q_x_C, Q_y_C, dt)
    rho_D = update_density_fvm(rho_D, AO, AO_minus_D, a_D, u_x_D, u_y_D, D_D, q_D_inflow, dx, dy, Q_x_D, Q_y_D, dt)
    # Recalculate AO
    

    
   
   
    # Calculate velocity magnitudes
    v_A = np.sqrt(u_x_A**2 + u_y_A**2)
    v_B = np.sqrt(u_x_B**2 + u_y_B**2)
    v_C = np.sqrt(u_x_C**2 + u_y_C**2)
    v_D = np.sqrt(u_x_D**2 + u_y_D**2) 
    closest_time = min(time_points, key=lambda tp: abs(t - tp))#line is used to find the closest time point from time_points to the current simulation time t
  # Generate 3D contour plots only at specified time points
    if (abs(t - closest_time) < tolerance and closest_time not in plotted_times and (closest_time == 0 or closest_time == T or abs(t - closest_time) < tolerance)):
        plotted_times.add(closest_time)
        #A plot is generated only if t is close to one of the time_points (accounting for floating-point precision).
#Each time_point is plotted only once.
#Special cases (start at t=0 and end at t=T) are always handled explicitly and accurately.
 
        X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
  # Generate 3D contour plots only at specified time points
    #if any(np.abs(t - tp) < tolerance for tp in time_points):
       # X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)

        # Create a new figure for each time step
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle(f'Time: {closest_time:.2f} hours')  # Update the title with the current time

        # Density plots
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, rho_A.T, cmap='viridis')
        ax1.set_title('Density A(Veh/km^2)')
        ax1.set_xlabel('X-axis (km)')
        ax1.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        cbar.set_label(' Density (A)(Veh/km^2)', fontsize=12)

        ax2 = fig.add_subplot(3, 4, 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, rho_B.T, cmap='viridis')
        ax2.set_title('Density B(Veh/km^2)')
        ax2.set_xlabel('X-axis (km)')
        ax2.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        cbar.set_label(' Density (B)(Veh/km^2)', fontsize=12)

        ax3 = fig.add_subplot(3, 4, 3, projection='3d')
        surf3 = ax3.plot_surface(X, Y, rho_C.T, cmap='viridis')
        ax3.set_title('Density C(Veh/km^2)')
        ax3.set_xlabel('X-axis(km)')
        ax3.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
        cbar.set_label(' Density (C)(Veh/km^2)', fontsize=12)

        ax4 = fig.add_subplot(3, 4, 4, projection='3d')
        surf4 = ax4.plot_surface(X, Y, rho_D.T, cmap='viridis')
        ax4.set_title('Density D(Veh/km^2)')
        ax4.set_xlabel('X-axis(km)')
        ax4.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
        cbar.set_label(' Density (D)(Veh/km^2)', fontsize=12)

        # Velocity magnitude plots
        ax5 = fig.add_subplot(3, 4, 5, projection='3d')
        surf5 = ax5.plot_surface(X, Y, v_A.T, cmap='magma')
        ax5.set_title('Velocity Magnitude A (km/hr) ')
        ax5.set_xlabel('X-axis(km)')
        ax5.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10)
        cbar.set_label('Velocity Magnitude A (km/hr) ', fontsize=12)

        ax6 = fig.add_subplot(3, 4, 6, projection='3d')
        surf6 = ax6.plot_surface(X, Y, v_B.T, cmap='magma')
        ax6.set_title('Velocity Magnitude B (km/hr) ')
        ax6.set_xlabel('X-axis(km)')
        ax6.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf6, ax=ax6, shrink=0.5, aspect=10)
        cbar.set_label('Velocity Magnitude B (km/hr) ', fontsize=12)

        ax7 = fig.add_subplot(3, 4, 7, projection='3d')
        surf7 = ax7.plot_surface(X, Y, v_C.T, cmap='magma')
        ax7.set_title('Velocity Magnitude C (km/hr) ')
        ax7.set_xlabel('X-axis(km)')
        ax7.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf7, ax=ax7, shrink=0.5, aspect=10)
        cbar.set_label('Velocity Magnitude C (km/hr) ', fontsize=12)
        
        ax8 = fig.add_subplot(3, 4, 8, projection='3d')
        surf8 = ax8.plot_surface(X, Y, v_D.T, cmap='magma')
        ax8.set_title('Velocity Magnitude D (km/hr) ')
        ax8.set_xlabel('X-axis(km)')
        ax8.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf8, ax=ax8, shrink=0.5, aspect=10)
        cbar.set_label('Velocity Magnitude D (km/hr) ', fontsize=12)

        # Flow plots
        ax9 = fig.add_subplot(3, 4, 9, projection='3d')
        surf9 = ax9.plot_surface(X, Y, Q_A.T, cmap='plasma')
        ax9.set_title('Flux density_A (veh/kmhr) ')
        ax9.set_xlabel('X-axis(km)')
        ax9.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf9, ax=ax9, shrink=0.5, aspect=10)
        cbar.set_label('Flux density_A (veh/kmhr) ', fontsize=12)

        ax10 = fig.add_subplot(3, 4, 10, projection='3d')
        surf10 = ax10.plot_surface(X, Y, Q_B.T, cmap='plasma')
        ax10.set_title('Flux density_B (veh/kmhr) ')
        ax10.set_xlabel('X-axis(km)')
        ax10.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf10, ax=ax10, shrink=0.5, aspect=10)
        cbar.set_label('Flux density_A (veh/kmhr) ', fontsize=12)

        ax11 = fig.add_subplot(3, 4, 11, projection='3d')
        surf11 = ax11.plot_surface(X, Y, Q_C.T, cmap='plasma')
        ax11.set_title('Flux density_C (veh/kmhr) ')
        ax11.set_xlabel('X-axis(km)')
        ax11.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf11, ax=ax11, shrink=0.5, aspect=10)
        cbar.set_label('Flux density_C (veh/kmhr) ', fontsize=12)

        ax12 = fig.add_subplot(3, 4, 12, projection='3d')
        surf12 = ax12.plot_surface(X, Y, Q_D.T, cmap='plasma')
        ax12.set_title('Flux density_D (veh/kmhr) ')
        ax12.set_xlabel('X-axis(km)')
        ax12.set_ylabel('Y-axis(km)')
        cbar = fig.colorbar(surf12, ax=ax12, shrink=0.5, aspect=10)
        cbar.set_label('Flux density_D (veh/kmhr) ', fontsize=12)
        # Check if stopping behavior is applied
        stopping_applied_u_A = apply_stopping_behavior_u(u_x_A, u_y_A, t)
        stopping_applied_u_B = apply_stopping_behavior_u(u_x_B, u_y_B, t)
        stopping_applied_u_C = apply_stopping_behavior_u(u_x_C, u_y_C, t)
        stopping_applied_u_D = apply_stopping_behavior_u(u_x_D, u_y_D, t)

        stopping_applied_f_A = apply_stopping_behavior_f(Q_x_A, Q_y_A, t)
        stopping_applied_f_B = apply_stopping_behavior_f(Q_x_B, Q_y_B, t)
        stopping_applied_f_C = apply_stopping_behavior_f(Q_x_C, Q_y_C, t)
        stopping_applied_f_D = apply_stopping_behavior_f(Q_x_D, Q_y_D, t)

   

    plt.tight_layout()
    plt.show()

    time_index = (time_index + 1) % len(time_points)  # Increment time_index only when plotting