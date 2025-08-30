import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# Parameters and Domain Setup
# ----------------------------

# Domain: rectangular simulation box.
Nx, Ny = 200, 200      # grid points in x and y
Lx, Ly = 200.0, 200.0  # physical domain size
dx = Lx / (Nx - 1)     # spatial step (assuming dx = dy)
dy = Ly / (Ny - 1)

# Model parameters for free energy: f0 = (A/4)(phi^2-1)^2, plus gradient term.
A = 1.0    # bulk free energy parameter
K = 1.0    # gradient energy coefficient
M = 1.0    # mobility used in Allen–Cahn time stepping

# Prescribed contact angle at the solid walls (top and bottom)
theta_deg = 60.0                  
theta_prescribed = np.deg2rad(theta_deg)
# Following eq. (9.64): h = sqrt(2*A*K)*cos(theta_prescribed)
h = np.sqrt(2 * A * K) * np.cos(theta_prescribed)
print("Prescribed contact angle (deg):", theta_deg)
print("Using h =", h)

# ----------------------------
# Initialize the Order Parameter Field
# ----------------------------

# We set up a droplet (phi ~ +1) adjacent to the bottom wall (solid)
R_drop = 30.0         # droplet radius
cx = Lx / 2.0         # droplet center in x (placed at mid-domain)
cy = R_drop           # droplet center in y (so that it touches y=0)

# Create mesh grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize phi with a hyperbolic tangent profile.
# (Note: tanh profile here smoothly connects phi ~ +1 inside the droplet to phi ~ -1 outside.)
phi = np.tanh((R_drop - np.sqrt((X - cx)**2 + (Y - cy)**2))/ np.sqrt(2))

# ----------------------------
# Time-stepping (Allen–Cahn Dynamics)
# ----------------------------

dt = 0.01        # time step (must satisfy stability criteria for explicit scheme)
max_iter = 10000
tol = 1e-6

phi_new = np.copy(phi)

# For storing snapshots (frames) for the animation.
frames = []
frame_interval = 100  # save a frame every 'frame_interval' iterations

# Main simulation loop.
for it in range(max_iter):
    # Compute Laplacian using a central difference (periodic in x using np.roll)
    lap = np.zeros_like(phi)
    lap[1:-1, :] = (phi[2:, :] + phi[:-2, :] +
                    np.roll(phi[1:-1, :], -1, axis=1) +
                    np.roll(phi[1:-1, :], 1, axis=1) -
                    4 * phi[1:-1, :]) / dx**2

    # Chemical potential (functional derivative of free energy)
    chem_pot = A * phi * (phi**2 - 1) - K * lap

    # Update phi in the interior using explicit Euler time stepping.
    phi_new[1:-1, :] = phi[1:-1, :] - dt * M * chem_pot[1:-1, :]

    # ----------------------------
    # Impose Boundary Conditions
    # ----------------------------
    # Using periodic boundaries in x have been handled by np.roll.
    # Implement wetting boundary condition at the solid walls.
    # Bottom wall (y = 0, outward normal (0, -1)): 
    #   ∂φ/∂y = -h/K  →  using a forward difference:
    phi_new[0, :] = phi_new[1, :] + dy * (h / K)
    # Top wall (y = Ly, outward normal (0, 1)): 
    #   ∂φ/∂y = h/K  →  using a backward difference:
    phi_new[-1, :] = phi_new[-2, :] + dy * (h / K)
    
    # Compute the maximum change for convergence.
    diff = np.max(np.abs(phi_new - phi))
    phi[:] = phi_new  # update for the next iteration
    
    # Store a snapshot for the animation every 'frame_interval' iterations.
    if it % frame_interval == 0:
        frames.append(np.copy(phi))
        print("Iteration", it, "max change =", diff)
    
    # Check for convergence (after a minimum number of iterations)
    if diff < tol and it > 100:
        frames.append(np.copy(phi))
        print("Converged after", it, "iterations.")
        break
else:
    print("Did not converge after", max_iter, "iterations.")

# ----------------------------
# Extract the Droplet Interface and Perform Circle Fitting
# ----------------------------

# For the interface we use phi = 0 (allowing a small threshold around zero).
threshold = 0.1
indices = np.where(np.abs(phi) < threshold)
y_int = y[indices[0]]  # y corresponding to interface
x_int = x[indices[1]]  # x corresponding to interface

if len(x_int) < 10:
    print("Too few interface points were found for circle fitting!")
else:
    # Set up the least squares problem for circle fitting:
    # x^2 + y^2 + D*x + E*y + F = 0.
    A_ls = np.column_stack((x_int, y_int, np.ones_like(x_int)))
    B_ls = -(x_int**2 + y_int**2)
    sol, _, _, _ = np.linalg.lstsq(A_ls, B_ls, rcond=None)
    D_fit, E_fit, F_fit = sol
    a_fit = -D_fit / 2.0  # circle center x
    b_fit = -E_fit / 2.0  # circle center y
    r_fit = np.sqrt(a_fit**2 + b_fit**2 - F_fit)  # circle radius
    
    # For a droplet resting on a horizontal wall (at y=0), the contact angle satisfies:
    #    theta_meas = arccos(b_fit / r_fit)
    theta_meas = np.arccos(b_fit / r_fit)
    theta_meas_deg = np.rad2deg(theta_meas)
    
    print("\nFitted circle center: (%.3f, %.3f), radius = %.3f" % (a_fit, b_fit, r_fit))
    print("Measured contact angle (deg): %.3f" % (theta_meas_deg))
    print("Prescribed contact angle (deg): %.3f" % (theta_deg))

# ----------------------------
# Animation: Show All Frames of the Simulation
# ----------------------------

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(frames[0], origin='lower', extent=[0, Lx, 0, Ly], cmap='bwr')
ax.set_title("Evolution of Order Parameter φ")
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = fig.colorbar(im, ax=ax)

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames,
                              interval=100, blit=True)

plt.show()

# Optionally, to save the animation as an MP4 video file, uncomment the following lines:
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
# ani.save("wetting_simulation.mp4", writer=writer)
