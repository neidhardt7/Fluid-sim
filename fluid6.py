"""
Lattice Boltzmann Method (LBM) simulation of a liquid droplet on a surface in 3D.
This simulation models:
1. Surface tension (liquid-liquid cohesion)
2. Wetting behavior (liquid-surface adhesion)
3. Interface dynamics (liquid-gas interaction)
4. Contact angle formation
Using the D3Q19 lattice model with Shan-Chen pseudopotential for multiphase flow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import logging
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure

logger = logging.getLogger(__name__)

class LBMFluid3D:
    def __init__(self, nx=50, ny=50, nz=50, tau=1.5):
        # Physical grid parameters
        self.nx, self.ny, self.nz = nx, ny, nz
        self.tau = tau
        self.omega = 1.0 / self.tau  # Collision frequency
        
        # Physical parameters
        self.gravity = 9.8e-6  # Reduced gravitational acceleration
        self.surface_tension = -2.0  # Reduced surface tension coefficient
        self.velocity_scale = 0.8  # Reduced velocity scale for stability
        
        # D3Q19 lattice velocities: rest particle + 18 directions
        self.velocities = np.array([
            [0, 0, 0],  # rest
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],  # face centers
            [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],  # face diagonals
            [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],  # face diagonals
            [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]   # face diagonals
        ])
        
        # Lattice weights for D3Q19
        self.weights = np.array([
            1/3,  # rest
            1/18, 1/18, 1/18, 1/18, 1/18, 1/18,  # face centers
            1/36, 1/36, 1/36, 1/36,  # face diagonals
            1/36, 1/36, 1/36, 1/36,  # face diagonals
            1/36, 1/36, 1/36, 1/36   # face diagonals
        ])
        
        # Opposite directions for bounce-back
        self.opposites = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
        
        # Fluid physical parameters
        self.rho0 = 1.0  # Reference density for pseudopotential
        self.g_aa = self.surface_tension
        self.c_s_sq = 1/3  # Square of speed of sound in lattice units
        
        # Initialize macroscopic fields
        self.rho = np.ones((nx, ny, nz))  # Density field
        self.ux = np.zeros((nx, ny, nz))  # x-velocity field
        self.uy = np.zeros((nx, ny, nz))  # y-velocity field
        self.uz = np.zeros((nx, ny, nz))  # z-velocity field
        
        # Create solid walls (5 lattice units thick)
        wall_thickness = 5
        self.wall_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.wall_mask[:, :, :wall_thickness] = True  # Bottom wall
        self.wall_mask[:, :, -wall_thickness:] = True  # Top wall
        self.wall_mask[:, :wall_thickness, :] = True  # Front wall
        self.wall_mask[:, -wall_thickness:, :] = True  # Back wall
        self.wall_mask[:wall_thickness, :, :] = True  # Left wall
        self.wall_mask[-wall_thickness:, :, :] = True  # Right wall
        
        # Create solid surface (everything except the droplet)
        x, y, z = np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
        r = np.sqrt((x - nx/2)**2 + (y - ny/2)**2 + (z - nz/2)**2)  # Center the droplet
        
        # Set initial densities
        self.rho[self.wall_mask] = 3.0  # Solid walls
        self.rho[(r >= 15) & ~self.wall_mask] = 3.0  # Solid surface
        self.rho[(r < 15) & ~self.wall_mask] = 2.0  # Liquid phase (inside droplet)
        
        # Initialize distribution functions to local equilibrium
        self.f = np.zeros((nx, ny, nz, 19))
        for i in range(19):
            self.f[:, :, :, i] = self.weights[i] * self.rho
            
        # Storage for interface analysis
        self.sphere_params = None

    def psi(self, density):
        """Pseudopotential function for non-ideal gas behavior"""
        density = np.clip(density, 0.1, 2.0)  # Numerical stability
        return self.rho0 * (np.tanh(3.0 * density / self.rho0) + 1.0)

    def bounce_back_walls(self):
        """Apply bounce-back boundary condition at all solid walls"""
        wall_thickness = 5
        
        # Bottom wall
        for i in range(19):
            if self.velocities[i, 2] < 0:  # Particles moving downward
                self.f[:, :, :wall_thickness, i] = self.f[:, :, :wall_thickness, self.opposites[i]]
        
        # Top wall
        for i in range(19):
            if self.velocities[i, 2] > 0:  # Particles moving upward
                self.f[:, :, -wall_thickness:, i] = self.f[:, :, -wall_thickness:, self.opposites[i]]
        
        # Front wall
        for i in range(19):
            if self.velocities[i, 1] < 0:  # Particles moving forward
                self.f[:, :wall_thickness, :, i] = self.f[:, :wall_thickness, :, self.opposites[i]]
        
        # Back wall
        for i in range(19):
            if self.velocities[i, 1] > 0:  # Particles moving backward
                self.f[:, -wall_thickness:, :, i] = self.f[:, -wall_thickness:, :, self.opposites[i]]
        
        # Left wall
        for i in range(19):
            if self.velocities[i, 0] < 0:  # Particles moving left
                self.f[:wall_thickness, :, :, i] = self.f[:wall_thickness, :, :, self.opposites[i]]
        
        # Right wall
        for i in range(19):
            if self.velocities[i, 0] > 0:  # Particles moving right
                self.f[-wall_thickness:, :, :, i] = self.f[-wall_thickness:, :, :, self.opposites[i]]

    def calculate_forces(self):
        """Calculate all forces including gravity"""
        # Surface tension forces
        force = np.zeros((self.nx, self.ny, self.nz, 3))
        psi_rho = self.psi(self.rho)
        
        for i, (cx, cy, cz) in enumerate(self.velocities):
            psi_neighbors = np.roll(np.roll(np.roll(psi_rho, cx, axis=0), cy, axis=1), cz, axis=2)
            force[:, :, :, 0] += self.weights[i] * cx * psi_neighbors
            force[:, :, :, 1] += self.weights[i] * cy * psi_neighbors
            force[:, :, :, 2] += self.weights[i] * cz * psi_neighbors
        
        force *= -self.g_aa * psi_rho[:, :, :, np.newaxis]
        
        # Add gravitational force (only z-component)
        force[:, :, :, 2] += self.gravity * (self.rho - np.min(self.rho))
        
        return np.clip(force, -0.1, 0.1)

    def equilibrium(self):
        """Calculate equilibrium distribution functions"""
        f_eq = np.zeros_like(self.f)
        ux_safe = np.clip(self.ux, -0.1, 0.1)
        uy_safe = np.clip(self.uy, -0.1, 0.1)
        uz_safe = np.clip(self.uz, -0.1, 0.1)
        usqr = ux_safe**2 + uy_safe**2 + uz_safe**2
        
        for i, (cx, cy, cz) in enumerate(self.velocities):
            cu = cx*ux_safe + cy*uy_safe + cz*uz_safe
            f_eq[:, :, :, i] = self.weights[i] * self.rho * (
                1 + 3*cu + 4.5*cu**2 - 1.5*usqr
            )
        return f_eq

    def step(self):
        """Advance simulation by one timestep"""
        # Calculate macroscopic density and velocity
        self.rho = np.sum(self.f, axis=3)
        
        # Enforce phase separation through density thresholding
        transition_width = 0.05
        
        # First handle the solid phase (walls and surface)
        self.rho[self.wall_mask] = 3.0  # Set solid phase density
        self.rho[self.rho > 2.5] = 3.0  # Maintain solid surface
        
        # Then handle liquid phase
        liquid_mask = ~self.wall_mask & (self.rho <= 2.5)
        self.rho[liquid_mask] = 2.0 / (1.0 + np.exp(-(self.rho[liquid_mask] - 1.0) / transition_width)) + 0.1
        
        # Calculate velocities
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        self.uz = np.zeros_like(self.rho)
        for i, (cx, cy, cz) in enumerate(self.velocities):
            self.ux += cx * self.f[:, :, :, i]
            self.uy += cy * self.f[:, :, :, i]
            self.uz += cz * self.f[:, :, :, i]
        
        # Scale velocities for stability
        self.ux *= self.velocity_scale
        self.uy *= self.velocity_scale
        self.uz *= self.velocity_scale
        
        # Divide by density (avoid division by zero)
        self.ux = np.divide(self.ux, self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
        self.uy = np.divide(self.uy, self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
        self.uz = np.divide(self.uz, self.rho, out=np.zeros_like(self.uz), where=self.rho > 0.1)
        
        # Zero velocity in solid phase
        solid_mask = (self.rho >= 2.5) | self.wall_mask
        self.ux[solid_mask] = 0
        self.uy[solid_mask] = 0
        self.uz[solid_mask] = 0
        
        # Apply forces with reduced magnitude
        force = self.calculate_forces()
        force *= 0.5  # Reduce force magnitude
        self.ux += np.divide(force[:, :, :, 0], self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
        self.uy += np.divide(force[:, :, :, 1], self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
        self.uz += np.divide(force[:, :, :, 2], self.rho, out=np.zeros_like(self.uz), where=self.rho > 0.1)
        
        # Collision step: relax toward equilibrium
        f_eq = self.equilibrium()
        self.f += self.omega * (f_eq - self.f)
        
        # Add force contribution to distribution functions
        for i, (cx, cy, cz) in enumerate(self.velocities):
            force_term = 3 * self.weights[i] * (force[:, :, :, 0] * cx + 
                                              force[:, :, :, 1] * cy + 
                                              force[:, :, :, 2] * cz)
            self.f[:, :, :, i] += np.clip(force_term, -0.1, 0.1)
        
        # Ensure positive distributions
        self.f = np.maximum(self.f, 1e-10)
        
        # Streaming step: propagate to neighboring sites
        for i, (cx, cy, cz) in enumerate(self.velocities):
            # Only stream in non-solid regions
            temp_f = np.roll(np.roll(np.roll(self.f[:, :, :, i], cx, axis=0), cy, axis=1), cz, axis=2)
            self.f[~solid_mask, i] = temp_f[~solid_mask]
        
        # Apply bounce-back boundary condition at solid walls
        self.bounce_back_walls()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create 3D fluid simulation
    fluid = LBMFluid3D(nx=50, ny=50, nz=50, tau=1.5)
    
    # Set up 3D visualization
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')  # Density field
    ax2 = fig.add_subplot(132, projection='3d')  # Velocity field
    ax3 = fig.add_subplot(133, projection='3d')  # Interface detection
    
    # Create custom colormap for density field
    colors = [(0.2, 0.2, 0.2),  # Dark gray for solid walls
              (0.8, 0.8, 0.8),  # Light gray for gas
              (0.0, 0.2, 0.8),  # Dark blue for liquid
              (0.0, 0.4, 1.0)]  # Bright blue for high density liquid
    cmap_density = LinearSegmentedColormap.from_list('fluid_cmap', colors, N=256)
    
    def update(frame):
        # Take one step
        fluid.step()
        
        # Clear all axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Create 3D grid
        x, y, z = np.meshgrid(range(fluid.nx), range(fluid.ny), range(fluid.nz), indexing='ij')
        
        # 1. Density plot
        # Create isosurface for liquid phase
        try:
            # Normalize density for marching cubes
            rho_normalized = (fluid.rho - np.min(fluid.rho)) / (np.max(fluid.rho) - np.min(fluid.rho))
            level = 0.5  # Middle of the normalized range
            vertices, faces, _, _ = measure.marching_cubes(rho_normalized, level=level)
            ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           triangles=faces, cmap=cmap_density)
        except ValueError as e:
            logger.warning(f"Marching cubes failed: {e}")
            # Fallback to volume rendering
            ax1.voxels(fluid.rho > 2.0, facecolors='blue', edgecolor='k', alpha=0.3)
        
        ax1.set_title('Density Field')
        
        # 2. Velocity field plot (downsample for clarity)
        skip = 5
        x_sub = x[::skip, ::skip, ::skip]
        y_sub = y[::skip, ::skip, ::skip]
        z_sub = z[::skip, ::skip, ::skip]
        ux_sub = fluid.ux[::skip, ::skip, ::skip]
        uy_sub = fluid.uy[::skip, ::skip, ::skip]
        uz_sub = fluid.uz[::skip, ::skip, ::skip]
        speed = np.sqrt(ux_sub**2 + uy_sub**2 + uz_sub**2)
        ax2.quiver(x_sub, y_sub, z_sub, ux_sub, uy_sub, uz_sub, 
                  length=0.5, normalize=True, cmap='coolwarm')
        ax2.set_title('Velocity Field')
        
        # 3. Interface detection plot
        try:
            # Use the same normalized density for interface detection
            vertices, faces, _, _ = measure.marching_cubes(rho_normalized, level=level)
            ax3.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                           triangles=faces, cmap=cmap_density)
        except ValueError as e:
            logger.warning(f"Marching cubes failed: {e}")
            # Fallback to volume rendering
            ax3.voxels(fluid.rho > 2.0, facecolors='blue', edgecolor='k', alpha=0.3)
        
        ax3.set_title('Interface Detection')
        
        # Set equal aspect ratio for all axes
        for ax in [ax1, ax2, ax3]:
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(0, fluid.nx)
            ax.set_ylim(0, fluid.ny)
            ax.set_zlim(0, fluid.nz)
        
        # Add step counter
        fig.suptitle(f'Simulation Step: {frame}')
        
        # Adjust layout
        plt.tight_layout()
    
    # Run animation
    anim = FuncAnimation(fig, update, frames=100, interval=200)
    plt.show()

if __name__ == '__main__':
    main() 