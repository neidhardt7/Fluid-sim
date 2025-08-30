import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
nx, ny = 300, 300  # Grid dimensions
tau = 0.6  # Relaxation time
initial_radius = 50  # Radius of the droplet
iterations = 250  # Number of simulation steps
omega = 1 / tau  # Relaxation parameter
# Parameters for pseudopotential
rho_0 = 1.0  # Reference density
G = -1.0     # Interaction strength parameter

# New parameters for density and forces
rho_liquid = 0.5  # Density of the droplet
rho_gas = 1.0  # Density of the surrounding medium
force_strength = 0.1  # Strength of intermolecular forces (surface tension)

# Define the discrete velocity set for D2Q9 model
velocities = [
    [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, -1], [1, -1], [-1, 1]
]
weights = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4

# Initialize grid
rho = np.ones((nx, ny)) * rho_gas  # Initialize as gas density
ux = np.zeros((nx, ny))  # x-component of velocity
uy = np.zeros((nx, ny))  # y-component of velocity

# Initialize distribution functions
f = np.zeros((nx, ny, len(velocities)))
for i in range(len(velocities)):
    f[:, :, i] = weights[i] * rho

# Place droplet in the center
for i in range(nx):
    for j in range(ny):
        if (i - nx // 2) ** 2 + (j - ny // 2) ** 2 < initial_radius ** 2:
            rho[i, j] = rho_liquid
for i in range(len(velocities)):
    f[:, :, i] = weights[i] * rho

# Equilibrium distribution function
def equilibrium(rho, ux, uy):
    f_eq = np.zeros((nx, ny, len(velocities)))
    for i, (cx, cy) in enumerate(velocities):
        cu = cx * ux + cy * uy
        f_eq[:, :, i] = weights[i] * rho * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * (ux ** 2 + uy ** 2))
    return f_eq

# Collision and streaming steps
def collision_and_streaming(f, rho, ux, uy, force_strength):
    # Compute equilibrium distribution
    f_eq = equilibrium(rho, ux, uy)
    # Add intermolecular force effects (placeholder force model)
    force_x = force_strength * (rho - rho_gas)
    force_y = force_strength * (rho - rho_gas)
    ux += force_x / rho
    uy += force_y / rho
    # Collision step
    f += omega * (f_eq - f)
    # Streaming step
    for i, (cx, cy) in enumerate(velocities):
        f_new = np.roll(f[:, :, i], cx, axis=0)
        f_new = np.roll(f_new, cy, axis=1)
        f[:, :, i] = f_new

# Compute macroscopic variables
def compute_macroscopic(f):
    rho = np.sum(f, axis=2)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    for i, (cx, cy) in enumerate(velocities):
        ux += f[:, :, i] * cx
        uy += f[:, :, i] * cy
    ux /= rho
    uy /= rho
    return rho, ux, uy

# Animation function
fig, ax = plt.subplots()

def update(frame):
    global rho, ux, uy, f
    collision_and_streaming(f, rho, ux, uy, force_strength)
    rho, ux, uy = compute_macroscopic(f)
    ax.clear()
    ax.imshow(rho.T, cmap="viridis", origin="lower")
    ax.set_title(f"Iteration {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

# Create animation
anim = FuncAnimation(fig, update, frames=iterations, repeat=False)
plt.show()
