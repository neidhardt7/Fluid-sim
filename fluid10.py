"""
Allen-Cahn equation simulation of a liquid droplet on a solid surface.
This simulation models the behavior of a droplet on a surface using the phase-field approach,
which is particularly good at handling interface dynamics and wetting phenomena.

Key features:
- Droplet evolution with surface tension
- Contact angle measurement
- Wetting boundary conditions
- Interface tracking
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import simpledialog
from matplotlib.animation import PillowWriter

# ----------------------------
# Parameters and Domain Setup
# ----------------------------

# External force parameters
# Set these to non-zero values to enable external forces
# For gravity-like force: set force_y to negative value (e.g. -0.1)
# For electric field: set force_x and/or force_y to desired values
force_x = 0.0    # External force in x-direction
force_y = 0.1    # External force in y-direction (negative for gravity-like force)

# Domain: rectangular simulation box.
grid_points_x = 200      # Number of grid points in x-direction
grid_points_y = 200      # Number of grid points in y-direction
domain_length_x = 200.0  # Physical length of domain in x-direction
domain_length_y = 200.0  # Physical length of domain in y-direction
grid_spacing_x = domain_length_x / (grid_points_x - 1)  # Spatial step in x (assuming dx = dy)
grid_spacing_y = domain_length_y / (grid_points_y - 1)  # Spatial step in y
force_x*=-1.0
force_y*=-1.0

# Model parameters for free energy: f0 = (bulk_coeff/4)(phi^2-1)^2, plus gradient term.
# Bulk coefficient (controls phase separation and interface sharpness)
# - Higher values (> 2.0): Creates sharper interfaces, stronger phase separation
# - Moderate values (0.5-2.0): Balanced phase separation, stable interfaces
# - Lower values (< 0.5): Softer interfaces, weaker phase separation
# Default: 1.0 (balanced behavior)
bulk_coeff = 2.0    # Bulk free energy coefficient

# Gradient coefficient (controls interface width and surface tension)
# - Higher values (> 2.0): Wider interfaces, higher surface tension, more stable droplets
# - Moderate values (0.5-2.0): Standard interface width, balanced surface tension
# - Lower values (< 0.5): Narrower interfaces, lower surface tension, more prone to breakup
# Default: 1.0 (balanced behavior)
gradient_coeff = 2.0    # Gradient energy coefficient

# Mobility coefficient (controls the rate of phase evolution)
# - Higher values (> 2.0): Faster phase changes, may require smaller time steps
# - Moderate values (0.5-2.0): Standard evolution rate, good stability
# - Lower values (< 0.5): Slower phase changes, more stable but slower simulation
# Default: 1.0 (balanced behavior)
# Note: If increasing mobility, consider reducing time_step proportionally
mobility = 0.5    # Mobility coefficient

# Cohesive force parameter (controls liquid-liquid interactions)
# Higher values make the liquid phase more cohesive (stronger liquid-liquid interactions)
# Lower values make the liquid phase less cohesive (weaker liquid-liquid interactions)
# Suggested: 0.0 (no cohesion), 0.5 (moderate), 1.0 (strong)
cohesive_force = 5.0    # Cohesive force parameter   # Suggested: 0.0, 0.5, 1.0

# DENSITY AND SOLID PHASE INTERACTION PARAMETERS
# ============================================
# Liquid density (normalized)
# Higher values make the liquid phase more dense
# Lower values make the liquid phase less dense
# Suggested: 0.5 (light), 1.0 (default), 2.0 (heavy)
rho_liquid = 1.0    # Liquid density (normalized)   # Suggested: 0.5, 1.0, 2.0

# Gas density (normalized)
# Higher values make the gas phase more dense
# Lower values make the gas phase less dense
# Suggested: 0.01 (high contrast), 0.1 (default), 0.5 (low contrast)
rho_gas = 0.1      # Gas density (normalized)   # Suggested: 0.01, 0.1, 0.5

# Wall density (affects solid-liquid interaction)
# Higher values make the wall more attractive to liquid
# Lower values make the wall less attractive to liquid
# Suggested: 0.5 (hydrophobic), 1.0 (neutral), 2.0 (hydrophilic)
rho_wall = 1.0     # Wall density (normalized)   # Suggested: 0.5, 1.0, 2.0

# Prescribed contact angle at the solid walls (top and bottom)
prescribed_angle_deg = 60.0                  
prescribed_angle_rad = np.deg2rad(prescribed_angle_deg)
# Following eq. (9.64): h = sqrt(2*bulk_coeff*gradient_coeff)*cos(prescribed_angle_rad)
wetting_coeff = np.sqrt(2 * bulk_coeff * gradient_coeff) * np.cos(prescribed_angle_rad)
print("Prescribed contact angle (deg):", prescribed_angle_deg)
print("Using wetting coefficient h =", wetting_coeff)

# ----------------------------
# Initialize the Order Parameter Field
# ----------------------------

# We set up a droplet (phi ~ +1) adjacent to the bottom wall (solid)
droplet_radius = 30.0         # Initial radius of the droplet
droplet_center_x = domain_length_x / 2.0  # Droplet center x-coordinate (placed at mid-domain)
droplet_center_y = droplet_radius*1.1  # Droplet center y-coordinate (so that it touches y=0)

# Create mesh grid for spatial coordinates
x_coords = np.linspace(0, domain_length_x, grid_points_x)
y_coords = np.linspace(0, domain_length_y, grid_points_y)
x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

# Double well potential based density mapping
# This approach uses the full free energy functional to determine the density
# The free energy density is: f(φ) = (bulk_coeff/4)(φ^2-1)^2 + (gradient_coeff/2)|∇φ|^2
# The chemical potential is: μ = ∂f/∂φ = bulk_coeff*φ*(φ^2-1) - gradient_coeff*∇²φ
# The density is then determined by the equilibrium condition: μ = 0
# This gives a more physically accurate density profile that respects the phase field thermodynamics

def compute_density_from_potential(phi, laplacian_phi):
    """
    Compute density using the double well potential approach.
    This is based on the equilibrium condition of the chemical potential.
    
    Parameters:
    phi: order parameter field
    laplacian_phi: Laplacian of the order parameter field
    
    Returns:
    density field that satisfies the equilibrium condition
    """
    # Compute the bulk term of the chemical potential
    bulk_term = bulk_coeff * phi * (phi**2 - 1)
    
    # Compute the gradient term of the chemical potential
    gradient_term = -gradient_coeff * laplacian_phi
    
    # Total chemical potential
    chemical_potential = bulk_term + gradient_term
    
    # The equilibrium density is determined by the condition that the chemical potential
    # should be constant across the interface. We use this to compute the density:
    # ρ = ρ_gas + (ρ_liquid - ρ_gas) * (1 + tanh(μ/ε))/2
    # where ε is a small parameter that controls the interface width
    epsilon = 0.1  # Interface width parameter
    
    # Map the chemical potential to density using a smooth transition
    normalized_potential = chemical_potential / (bulk_coeff + gradient_coeff)  # Normalize to [-1,1]
    density = rho_gas + (rho_liquid - rho_gas) * (1 + np.tanh(normalized_potential/epsilon))/2
    
    return density

# Initialize the order parameter field
distance_from_center = np.sqrt((x_mesh - droplet_center_x)**2 + (y_mesh - droplet_center_y)**2)
order_parameter = np.tanh((droplet_radius - distance_from_center) / np.sqrt(2))

# Compute initial Laplacian for density calculation
initial_laplacian = np.zeros_like(order_parameter)
initial_laplacian[1:-1, :] = (order_parameter[2:, :] + order_parameter[:-2, :] +
                             np.roll(order_parameter[1:-1, :], -1, axis=1) +
                             np.roll(order_parameter[1:-1, :], 1, axis=1) -
                             4 * order_parameter[1:-1, :]) / grid_spacing_x**2

# Compute initial density using the double well potential approach
density = compute_density_from_potential(order_parameter, initial_laplacian)

# ----------------------------
# Time-stepping (Allen–Cahn Dynamics)
# ----------------------------

time_step = 0.01        # Time step (must satisfy stability criteria for explicit scheme)
max_iterations = 20000  # Maximum number of iterations
convergence_tolerance = 1e-6  # Convergence criterion

order_parameter_new = np.copy(order_parameter)
density_new = np.copy(density)

# For storing snapshots (frames) for the animation.
animation_frames = []
frame_interval = 100  # Save a frame every 'frame_interval' iterations

# Main simulation loop.
for iteration in range(max_iterations):
    # Compute Laplacian using a central difference (periodic in x using np.roll)
    laplacian = np.zeros_like(order_parameter)
    laplacian[1:-1, :] = (order_parameter[2:, :] + order_parameter[:-2, :] +
                          np.roll(order_parameter[1:-1, :], -1, axis=1) +
                          np.roll(order_parameter[1:-1, :], 1, axis=1) -
                          4 * order_parameter[1:-1, :]) / grid_spacing_x**2

    # Compute gradients for force coupling
    grad_x = np.zeros_like(order_parameter)
    grad_y = np.zeros_like(order_parameter)
    
    # Compute x-gradient (central difference with periodic BC)
    grad_x[1:-1, :] = (np.roll(order_parameter[1:-1, :], -1, axis=1) - 
                       np.roll(order_parameter[1:-1, :], 1, axis=1)) / (2 * grid_spacing_x)
    
    # Compute y-gradient (central difference)
    grad_y[1:-1, :] = (order_parameter[2:, :] - order_parameter[:-2, :]) / (2 * grid_spacing_y)

    # Compute local average of order parameter for cohesive forces
    local_avg = np.zeros_like(order_parameter)
    local_avg[1:-1, :] = (order_parameter[2:, :] + order_parameter[:-2, :] +
                         np.roll(order_parameter[1:-1, :], -1, axis=1) +
                         np.roll(order_parameter[1:-1, :], 1, axis=1)) / 4.0

    # Chemical potential (functional derivative of free energy)
    # Now coupling forces with phase field gradient for proper force application
    force_term = force_x * grad_x + force_y * grad_y
    chemical_potential = (bulk_coeff * order_parameter * (order_parameter**2 - 1) - 
                         gradient_coeff * laplacian +
                         cohesive_force * (local_avg - order_parameter) +
                         force_term)  # Coupled force term

    # Update order parameter in the interior using explicit Euler time stepping
    order_parameter_new[1:-1, :] = order_parameter[1:-1, :] - time_step * mobility * chemical_potential[1:-1, :]

    # Update density using the new order parameter and laplacian
    density_new = compute_density_from_potential(order_parameter_new, laplacian)

    # ----------------------------
    # Impose Boundary Conditions
    # ----------------------------
    # Using periodic boundaries in x have been handled by np.roll.
    # Implement wetting boundary condition at the solid walls.
    # Bottom wall (y = 0, outward normal (0, -1)): 
    order_parameter_new[0, :] = order_parameter_new[1, :] + grid_spacing_y * (wetting_coeff / gradient_coeff)
    # Top wall (y = Ly, outward normal (0, 1)): 
    order_parameter_new[-1, :] = order_parameter_new[-2, :] + grid_spacing_y * (wetting_coeff / gradient_coeff)
    
    # Update density at boundaries
    density_new[0, :] = rho_wall  # Bottom wall density
    density_new[-1, :] = rho_wall  # Top wall density
    
    # Compute the maximum change for convergence
    max_change = np.max(np.abs(order_parameter_new - order_parameter))
    order_parameter[:] = order_parameter_new  # Update for the next iteration
    density[:] = density_new  # Update density field
    
    # Store a snapshot for the animation every 'frame_interval' iterations
    if iteration % frame_interval == 0:
        animation_frames.append(np.copy(density))  # Store density instead of order parameter
        print("Iteration", iteration, "max change =", max_change)
    
    # Check for convergence (after a minimum number of iterations)
    if max_change < convergence_tolerance and iteration > 100:
        animation_frames.append(np.copy(density))
        print("Converged after", iteration, "iterations.")
        break
else:
    print("Did not converge after", max_iterations, "iterations.")

# ----------------------------
# Extract the Droplet Interface and Perform Circle Fitting
# ----------------------------

# For the interface we use phi = 0 (allowing a small threshold around zero).
interface_threshold = 0.1
interface_indices = np.where(np.abs(order_parameter) < interface_threshold)
interface_y = y_coords[interface_indices[0]]  # y-coordinates of interface points
interface_x = x_coords[interface_indices[1]]  # x-coordinates of interface points

if len(interface_x) < 10:
    print("Too few interface points were found for circle fitting!")
else:
    # Set up the least squares problem for circle fitting:
    # x^2 + y^2 + D*x + E*y + F = 0.
    circle_fit_matrix = np.column_stack((interface_x, interface_y, np.ones_like(interface_x)))
    circle_fit_vector = -(interface_x**2 + interface_y**2)
    solution, _, _, _ = np.linalg.lstsq(circle_fit_matrix, circle_fit_vector, rcond=None)
    D_fit, E_fit, F_fit = solution
    circle_center_x = -D_fit / 2.0  # Fitted circle center x-coordinate
    circle_center_y = -E_fit / 2.0  # Fitted circle center y-coordinate
    circle_radius = np.sqrt(circle_center_x**2 + circle_center_y**2 - F_fit)  # Fitted circle radius
    
    # For a droplet resting on a horizontal wall (at y=0), the contact angle satisfies:
    #    measured_angle = arccos(circle_center_y / circle_radius)
    measured_angle_rad = np.arccos(circle_center_y / circle_radius)
    measured_angle_deg = np.rad2deg(measured_angle_rad)
    
    #print("\nFitted circle center: (%.3f, %.3f), radius = %.3f" % (circle_center_x, circle_center_y, circle_radius))
    print("Measured contact angle (deg): %.3f" % (measured_angle_deg))
    #print("Prescribed contact angle (deg): %.3f" % (prescribed_angle_deg))

# ----------------------------
# Animation: Show All Frames of the Simulation
# ----------------------------

fig, ax = plt.subplots(figsize=(6, 5))
density_plot = ax.imshow(animation_frames[0], origin='lower', extent=[0, domain_length_x, 0, domain_length_y], cmap='bwr')
ax.set_title("Evolution of Order Parameter φ")
ax.set_xlabel("x")
ax.set_ylabel("y")
colorbar = fig.colorbar(density_plot, ax=ax)

def update_animation(frame):
    """Update function for animation"""
    density_plot.set_data(frame)
    return [density_plot]

animation = animation.FuncAnimation(fig, update_animation, frames=animation_frames,
                                  interval=100, blit=True)

# Create a root window but hide it
root = tk.Tk()
root.withdraw()

# Show the animation
plt.show()

# After the animation window is closed, ask for the filename
filename = simpledialog.askstring("Save Animation", 
                                "Enter filename for the GIF (without extension):",
                                parent=root)

if filename:
    # Add .gif extension if not present
    if not filename.endswith('.gif'):
        filename += '.gif'
        
    # Save the animation as GIF
    writer = PillowWriter(fps=10)
    animation.save(filename, writer=writer)
    print(f"Animation saved as {filename}")
    
    # Open file for writing and write header
    with open("contact_angles.tsv", "ab") as f:
        np.savetxt(f,np.column_stack(([filename], [measured_angle_deg])),delimiter='\t',fmt='%s')
        
# Clean up
root.destroy()
