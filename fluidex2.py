import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters: Domain, LBM (D2Q9) and Shan–Chen constants
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Domain size (number of lattice nodes)
Nx, Ny = 200, 200           # grid size in x and y
Lx, Ly = 200.0, 200.0       # physical domain size
dx = Lx / (Nx - 1)          # spatial step
dy = Ly / (Ny - 1)          # spatial step
timesteps = 2000           # total number of LBM time steps
frame_interval = 100        # save a frame every this many time steps

# Lattice Boltzmann parameters (D2Q9)
c_s = 1.0/np.sqrt(3)        # speed of sound in lattice units
# Lattice weights and discrete velocities for D2Q9:
w = np.array([4/9,
              1/9, 1/9, 1/9, 1/9,
              1/36, 1/36, 1/36, 1/36])
ex = np.array([ 0,  1,  0, -1,  0,  1, -1, -1,  1])
ey = np.array([ 0,  0,  1,  0, -1,  1,  1, -1, -1])
num_dirs = 9
tau = 1.0                   # relaxation time

# Shan–Chen pseudopotential parameters:
G = -5.0                    # fluid–fluid interaction (negative for separation)
rho_g = 0.144               # gas (low) density
rho_l = 1.95                # liquid (high) density

# Prescribed contact angle at the solid walls (top and bottom)
theta_deg = 60.0
theta_prescribed = np.deg2rad(theta_deg)
# Following eq. (9.64): h = sqrt(2*A*K)*cos(theta_prescribed)
# For Shan-Chen model, we use rho_wall to achieve similar effect
rho_wall = 1.226            # chosen to achieve similar contact angle
psi_wall = 1 - np.exp(-rho_wall)

print("Prescribed contact angle (deg):", theta_deg)
print("Using wall density =", rho_wall)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialization: Fields, Solid Mask & Droplet
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Create fluid density array and velocity fields; start with gas everywhere.
rho = np.full((Nx, Ny), rho_g)
ux = np.zeros((Nx, Ny))
uy = np.zeros((Nx, Ny))

# Create a Boolean mask for solid nodes: top (y = Ny-1) and bottom (y = 0) are solid.
solid = np.zeros((Nx, Ny), dtype=bool)
solid[:, 0] = True       # bottom wall
solid[:, -1] = True      # top wall

# Impose the wall (solid) density
rho[:, 0] = rho_wall
rho[:, -1] = rho_wall

# Initialize a droplet of liquid next to the bottom wall.
R_drop = 30              # droplet radius (in lattice units)
cx = Nx // 2            # droplet center in x
cy = R_drop + 1         # droplet center in y (so it touches bottom)
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
droplet_mask = ((X - cx)**2 + (Y - cy)**2 < R_drop**2) & (~solid)
rho[droplet_mask] = rho_l

# Initialize the LBM distribution functions, f_i, at equilibrium.
def feq(i, rho, ux, uy):
    eu = ex[i]*ux + ey[i]*uy
    u_sq = ux**2 + uy**2
    return w[i] * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u_sq)

f = np.zeros((num_dirs, Nx, Ny))
for i in range(num_dirs):
    f[i, :, :] = feq(i, rho, ux, uy)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper Function: Shift Field with Non-Periodic Y
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def shift_field(field, sx, sy):
    """
    Shifts a 2D field by (sx, sy). x-shift is periodic and y-shift is non-periodic.
    Out-of-bound indices in y are filled with psi_wall (the solid wall value).
    """
    shifted = np.roll(field, shift=sx, axis=0)  # periodic in x
    shifted = np.roll(shifted, shift=sy, axis=1)  # tentative shift in y
    # Now override the wrapped portion in y with psi_wall.
    if sy > 0:
        shifted[:, :sy] = psi_wall
    elif sy < 0:
        shifted[:, sy:] = psi_wall
    return shifted

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main Time Evolution Loop: LBM with Shan–Chen Forcing
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

frames = []   # to save intermediate phi fields for animation

# Numerical stability parameters
max_rho = 10.0  # Maximum allowed density
min_rho = 0.01  # Minimum allowed density
max_vel = 0.5   # Maximum allowed velocity

for t in range(timesteps):
    # Compute the pseudopotential: psi = 1 - exp(-rho)
    rho = np.clip(rho, min_rho, max_rho)
    psi = 1 - np.exp(-rho)
    
    # Compute the interaction (Shan–Chen) force F = (F_x, F_y)
    F_x = np.zeros((Nx, Ny))
    F_y = np.zeros((Nx, Ny))
    for i in range(1, num_dirs):
        psi_shift = shift_field(psi, ex[i], ey[i])
        F_x += w[i] * psi_shift * ex[i]
        F_y += w[i] * psi_shift * ey[i]
    F_x = -G * psi * F_x
    F_y = -G * psi * F_y
    
    # Compute macroscopic density and momentum
    rho = np.sum(f, axis=0)
    mom_x = np.sum(f * ex[:, None, None], axis=0) + 0.5 * F_x
    mom_y = np.sum(f * ey[:, None, None], axis=0) + 0.5 * F_y
    
    # Calculate macroscopic velocity u = momentum/rho in fluid nodes
    non_solid = ~solid
    ux[non_solid] = mom_x[non_solid] / np.maximum(rho[non_solid], min_rho)
    uy[non_solid] = mom_y[non_solid] / np.maximum(rho[non_solid], min_rho)
    
    # Clip velocities to prevent numerical instability
    ux = np.clip(ux, -max_vel, max_vel)
    uy = np.clip(uy, -max_vel, max_vel)
    
    ux[solid] = 0.0
    uy[solid] = 0.0
    
    # Collision Step: use BGK with Guo's forcing scheme
    for i in range(num_dirs):
        eu = ex[i]*ux + ey[i]*uy
        u_sq = np.clip(ux**2 + uy**2, 0, max_vel**2)
        feq_i = w[i] * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u_sq)
        e_dot_F = ex[i]*F_x + ey[i]*F_y
        u_dot_F = ux * F_x + uy * F_y
        F_i = (1 - 0.5/tau) * w[i] * (3*e_dot_F + 9*eu*e_dot_F - 3*u_dot_F)
        f[i, :, :] = f[i, :, :] - (f[i, :, :] - feq_i)/tau + F_i

    # Streaming Step
    for i in range(num_dirs):
        f[i, :, :] = np.roll(f[i, :, :], shift=ex[i], axis=0)
        f[i, :, :] = np.roll(f[i, :, :], shift=ey[i], axis=1)
    
    # Bounce-Back for Solid Nodes
    bounce_pairs = {1:3, 2:4, 3:1, 4:2, 5:7, 6:8, 7:5, 8:6}
    for i in range(1, num_dirs):
        opp = bounce_pairs[i]
        f[i, solid] = f[opp, solid]
    
    # Reinforce the wall condition: solid nodes are fixed at rho_wall
    rho[solid] = rho_wall
    for i in range(num_dirs):
        f[i, solid] = feq(i, rho[solid], np.zeros_like(rho[solid]), np.zeros_like(rho[solid]))
    
    # Save frames at selected intervals for later animation
    if t % frame_interval == 0:
        # Define order parameter φ = 2*(ρ - ρ_g)/(ρₗ-ρ_g) - 1
        phi_field = 2*(rho - rho_g)/(rho_l - rho_g) - 1
        frames.append(phi_field.copy())
        print("Timestep", t)

#%%%%%%%%%%%%%%%%%%%%%
# Circle Fitting and Contact Angle Measurement
#%%%%%%%%%%%%%%%%%%%%%
# Extract interface points as those where |φ| is small
threshold = 0.1
indices = np.where((np.abs(2*(rho - rho_g)/(rho_l - rho_g) - 1) < threshold) & (~solid))
if len(indices[0]) < 10:
    print("Too few interface points found for circle fitting!")
else:
    # In our grid coordinates, fit the circle
    x_int = indices[0]  # x indices
    y_int = indices[1]  # y indices
    # Fit circle: x² + y² + D x + E y + F = 0
    A_ls = np.column_stack((x_int, y_int, np.ones_like(x_int)))
    B_ls = -(x_int**2 + y_int**2)
    sol, _, _, _ = np.linalg.lstsq(A_ls, B_ls, rcond=None)
    D_fit, E_fit, F_fit = sol
    a_fit = -D_fit / 2.0   # x-coordinate of circle center
    b_fit = -E_fit / 2.0   # y-coordinate of circle center
    r_fit = np.sqrt(a_fit**2 + b_fit**2 - F_fit)
    
    # For a droplet adhered to a horizontal wall at y=0, the measured contact angle is:
    theta_meas = np.arccos(b_fit / r_fit)
    theta_meas_deg = np.rad2deg(theta_meas)
    
    print("\nFitted circle center: ({:.3f}, {:.3f}), radius = {:.3f}".format(a_fit, b_fit, r_fit))
    print("Measured contact angle (deg): {:.3f}".format(theta_meas_deg))
    print("Prescribed contact angle (deg): {:.3f}".format(theta_deg))

#%%%%%%%%%%%%%%%%%%%%%
# Animation of the Simulation Frames
#%%%%%%%%%%%%%%%%%%%%%
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(frames[0], origin='lower', cmap='bwr', extent=[0, Lx, 0, Ly])
ax.set_title("Evolution of Order Parameter φ (LB Shan–Chen)")
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = fig.colorbar(im, ax=ax)

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames,
                              interval=100, blit=True)
plt.show()

# Optional: Save the animation as a video file by uncommenting the lines below
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
# ani.save("lb_shanchen_wetting.mp4", writer=writer)
