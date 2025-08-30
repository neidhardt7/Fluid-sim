import numpy as np

# Parameters
N = 64
omega = 1.0
g_aa = -4.7
rho0 = 1.0
tau = 1 / omega
dt = 1.0
c_s_sq = 1 / 3  # Speed of sound squared (D2Q9)

# D2Q9 lattice parameters
weights = np.array([4/9] + [1/9]*4 + [1/36]*4)
lattice_vectors = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                             [1, 1], [-1, 1], [-1, -1], [1, -1]])

# Initialize fields
f = np.ones((N, N, 9))  # Particle distribution functions
rho = np.ones((N, N))   # Density
vel = np.ones((N, N, 2))  # Velocity

# Define the pseudopotential
def psi(density):
    return rho0 * (1.0 - np.exp(-density / rho0))

# Initialize fields
for x in range(N):
    for y in range(N):
        if (x - N/2.0)**2 + (y - N/2.0)**2 <= 15**2:
            rho[x, y] = 2.1
        else:
            rho[x, y] = 0.15
        f[x, y, :] = weights * rho[x, y]

# Main loop
for step in range(1000):
    # Compute macroscopic variables
    rho = f.sum(axis=2)
    # Add small epsilon to prevent division by zero
    vel[:, :, :] = np.dot(f, lattice_vectors) / (rho[:, :, None] + 1e-10)
    
    # Clip velocity to prevent instability
    vel = np.clip(vel, -1.0, 1.0)

    # Compute interaction force
    force = np.zeros((N, N, 2))
    for d, e in enumerate(lattice_vectors):
        psi_neighbors = np.roll(np.roll(psi(rho), e[0], axis=0), e[1], axis=1)
        force[:, :, 0] += weights[d] * e[0] * psi_neighbors
        force[:, :, 1] += weights[d] * e[1] * psi_neighbors
    force *= -g_aa * psi(rho)[:, :, None]

    # Collision step
    for d, e in enumerate(lattice_vectors):
        u_dot_e = vel[:, :, 0] * e[0] + vel[:, :, 1] * e[1]
        u_sq = vel[:, :, 0]**2 + vel[:, :, 1]**2
        equilibrium = weights[d] * rho * (1 + 3*u_dot_e + 4.5*u_dot_e**2 - 1.5*u_sq)
        f[:, :, d] += -(1 / tau) * (f[:, :, d] - equilibrium) + dt * weights[d] * (
            3 * (force[:, :, 0] * e[0] + force[:, :, 1] * e[1]))

    # Streaming step
    f_new = np.zeros_like(f)
    for d, e in enumerate(lattice_vectors):
        f_new[:, :, d] = np.roll(np.roll(f[:, :, d], e[0], axis=0), e[1], axis=1)
    f = f_new

# Reference density profile comparison
ref = np.array([0.185757, 0.185753, 0.185743, 0.185727, 0.185703, 0.185672, 0.185636, 0.185599, 0.185586, 0.185694,
                    0.186302, 0.188901, 0.19923, 0.238074, 0.365271, 0.660658, 1.06766, 1.39673, 1.56644, 1.63217,
                    1.65412, 1.66064, 1.66207, 1.66189, 1.66123, 1.66048, 1.65977, 1.65914, 1.65861, 1.6582, 1.6579,
                    1.65772, 1.65766, 1.65772, 1.6579, 1.6582, 1.65861, 1.65914, 1.65977, 1.66048, 1.66123, 1.66189,
                    1.66207, 1.66064, 1.65412, 1.63217, 1.56644, 1.39673, 1.06766, 0.660658, 0.365271, 0.238074,
                    0.19923, 0.188901, 0.186302, 0.185694, 0.185586, 0.185599, 0.185636, 0.185672, 0.185703, 0.185727,
                    0.185743, 0.185753])
#assert np.allclose(rho[N // 2, :], ref)
