"""
Lattice Boltzmann Method (LBM) simulation of a liquid droplet on a solid surface
with strong immiscibility between phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from datetime import datetime as dt

class LBMImmiscible:
    def __init__(self, nx=200, ny=300, tau=0.6):
        # Domain parameters
        self.nx, self.ny = nx, ny
        self.tau = tau
        self.omega = 1.0 / tau
        
        # Physical parameters
        self.gravity = 0.5
        self.surface_tension = 1.0
        self.wall_repulsion = 10.0  # Stronger repulsion for wood-like surface
        
        # Lattice parameters
        self.velocities = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1], 
                                  [1,1], [-1,-1], [1,-1], [-1,1]])
        self.weights = np.array([4/9] + [1/9]*4 + [1/36]*4)
        self.opposites = np.array([0, 2, 1, 4, 3, 7, 6, 5, 8])
        
        # Initialize fields
        self.rho = np.ones((nx, ny)) * 0.1  # Light ambient fluid
        self.ux = np.zeros((nx, ny))
        self.uy = np.zeros((nx, ny))
        
        # Create solid surface (wood-like)
        wall_thickness = 8  # Thicker wall for better immiscibility
        self.wall_mask = np.zeros((nx, ny), dtype=bool)
        self.wall_mask[:, :wall_thickness] = True  # Bottom wall
        self.rho[self.wall_mask] = 50.0  # Much higher density for solid wood
        
        # Create initial droplet with fixed mass
        x, y = np.meshgrid(range(nx), range(ny), indexing='ij')
        r = np.sqrt((x - nx/2)**2 + (y - 2*ny/20)**2)
        droplet_radius = 20
        droplet_mask = r < droplet_radius
        self.rho[droplet_mask] = 2.0
        self.uy[droplet_mask] = -1.0  # Initial downward velocity
        
        # Store initial mass for conservation
        self.initial_mass = np.sum(self.rho[~self.wall_mask])
        
        # Initialize distribution functions
        self.f = np.zeros((nx, ny, 9))
        for i in range(9):
            cx, cy = self.velocities[i]
            cu = cx * self.ux + cy * self.uy
            usqr = self.ux**2 + self.uy**2
            self.f[:, :, i] = self.weights[i] * self.rho * (
                1 + 3*cu + 4.5*cu**2 - 1.5*usqr
            )
        
        # Contact angle tracking
        self.contact_angles = []
        self.contact_times = []
        self.has_contacted = False
        self.contact_step = None
        self.current_angle = None

    def psi(self, density):
        """Pseudopotential function for phase separation"""
        return np.exp(-1.0 / density)

    def calculate_forces(self):
        """Calculate forces including strong wall repulsion"""
        force = np.zeros((self.nx, self.ny, 2))
        psi_rho = self.psi(self.rho)
        
        # Surface tension forces
        for i, (cx, cy) in enumerate(self.velocities):
            psi_neighbors = np.roll(np.roll(psi_rho, cx, axis=0), cy, axis=1)
            force[:, :, 0] += self.weights[i] * cx * psi_neighbors
            force[:, :, 1] += self.weights[i] * cy * psi_neighbors
        
        force *= -self.surface_tension * psi_rho[:, :, np.newaxis]
        
        # Gravity
        force[:, :, 1] -= self.gravity * (self.rho - 0.1)
        
        # Strong wall repulsion force (wood-like surface)
        wall_distance = np.zeros_like(self.rho)
        wall_distance[:, :] = np.arange(self.ny)[np.newaxis, :]
        repulsion = self.wall_repulsion * np.exp(-wall_distance / 3)  # Sharper decay
        force[:, :, 1] += repulsion * (self.rho > 1.0)
        
        # Zero force in wall regions
        force[self.wall_mask] = 0
        
        return force

    def zou_he_walls(self):
        """Non-absorbing wall boundary conditions"""
        wall_thickness = 8
        
        # Bottom wall boundary condition
        for i in range(9):
            if self.velocities[i, 1] < 0:  # Particles moving downward
                self.f[:, :wall_thickness, i] = self.f[:, :wall_thickness, self.opposites[i]]
        
        # Maintain wall density and zero velocity
        self.rho[self.wall_mask] = 20.0  # High density for wood
        self.ux[self.wall_mask] = 0
        self.uy[self.wall_mask] = -1.0

    def enforce_mass_conservation(self):
        """Ensure total mass remains constant"""
        current_mass = np.sum(self.rho[~self.wall_mask])
        mass_ratio = self.initial_mass / current_mass
        self.rho[~self.wall_mask] *= mass_ratio

    def find_interface_points(self):
        """Detect liquid-gas interface"""
        density = gaussian_filter(self.rho, sigma=1.0)
        grad_x = np.gradient(density, axis=0)
        grad_y = np.gradient(density, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        threshold = np.max(gradient_magnitude) * 0.3
        interface_points = gradient_magnitude > threshold
        
        points = np.where(interface_points)
        valid_points = []
        for y, x in zip(points[0], points[1]):
            # Look for points near the wall but not in it
            if 8 <= y <= 30 and abs(x - self.nx//2) < self.nx//3:
                valid_points.append((x, y))
                
        return valid_points

    def calculate_contact_angle(self):
        """Measure contact angle between surface and tangent line at contact point"""
        points = self.find_interface_points()
        if len(points) < 10:
            return None
            
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])
        
        try:
            # Find the contact point (lowest point)
            contact_idx = np.argmin(y_points)
            x_contact = x_points[contact_idx]
            y_contact = y_points[contact_idx]
            
            # Get nearby points to calculate tangent
            nearby_mask = np.abs(y_points - y_contact) < 5  # Points within 5 lattice units
            if np.sum(nearby_mask) < 3:
                return None
                
            # Fit a line to nearby points to get tangent
            coeffs = np.polyfit(x_points[nearby_mask], y_points[nearby_mask], 1)
            slope = coeffs[0]
            
            # Calculate angle between tangent and horizontal (wall)
            angle = np.arctan(slope) * 180 / np.pi
            
            # Convert to the angle between tangent and wall
            contact_angle = 180 - angle if angle > 0 else -angle
            
            return contact_angle if 0 <= contact_angle <= 180 else None
            
        except Exception as e:
            print(f"Contact angle calculation error: {str(e)}")
            return None

    def equilibrium(self):
        """Calculate equilibrium distribution"""
        f_eq = np.zeros_like(self.f)
        ux_safe = np.clip(self.ux, -0.1, 0.1)
        uy_safe = np.clip(self.uy, -0.1, 0.1)
        usqr = ux_safe**2 + uy_safe**2
        
        for i, (cx, cy) in enumerate(self.velocities):
            cu = cx*ux_safe + cy*uy_safe
            f_eq[:, :, i] = self.weights[i] * self.rho * (
                1 + 3*cu + 4.5*cu**2 - 1.5*usqr
            )
        return f_eq

    def step(self):
        """Advance simulation one step"""
        # Macroscopic calculations
        self.rho = np.sum(self.f, axis=2)
        self.rho = np.clip(self.rho, 0.1, 2.0)
        
        # Enforce mass conservation
        self.enforce_mass_conservation()
        
        # Calculate velocities
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        for i, (cx, cy) in enumerate(self.velocities):
            self.ux += cx * self.f[:, :, i]
            self.uy += cy * self.f[:, :, i]
        
        self.ux = np.divide(self.ux, self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
        self.uy = np.divide(self.uy, self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
        
        # Apply forces
        force = self.calculate_forces()
        self.ux += np.divide(force[:, :, 0], self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
        self.uy += np.divide(force[:, :, 1], self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
        
        # Collision step
        f_eq = self.equilibrium()
        self.f += self.omega * (f_eq - self.f)
        
        # Force contribution
        for i, (cx, cy) in enumerate(self.velocities):
            force_term = 3 * self.weights[i] * (force[:, :, 0] * cx + force[:, :, 1] * cy)
            self.f[:, :, i] += force_term
        
        self.f = np.maximum(self.f, 1e-10)
        
        # Streaming step
        for i, (cx, cy) in enumerate(self.velocities):
            self.f[:, :, i] = np.roll(np.roll(self.f[:, :, i], cx, axis=0), cy, axis=1)
        
        # Apply boundary conditions
        self.zou_he_walls()
        
        # Check for contact and record angle
        if not self.has_contacted:
            points = self.find_interface_points()
            if points:
                min_y = min(p[1] for p in points)
                if min_y <= 10:  # Just above wall thickness (8)
                    self.has_contacted = True
                    self.contact_step = len(self.contact_times)
                    print(f"Contact detected at step {self.contact_step}")
        
        # Calculate and store contact angle
        if self.has_contacted:
            angle = self.calculate_contact_angle()
            if angle is not None:
                self.contact_angles.append(angle)
                self.contact_times.append(len(self.contact_times))
                self.current_angle = angle
                print(f"Contact angle: {angle:.1f}°")

def main():
    # Create simulation
    fluid = LBMImmiscible(nx=200, ny=300)
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots
    ax1 = plt.subplot(131)  # Density field
    ax2 = plt.subplot(132)  # Velocity field
    ax3 = plt.subplot(133)  # Interface detection
    
    def update(frame):
        # Advance simulation
        fluid.step()
        
        # Clear plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Density plot
        ax1.imshow(fluid.rho.T, cmap='viridis', origin='lower', vmin=0.5, vmax=1.5)
        ax1.set_title('Density Field')
        
        # Velocity field
        skip = 5
        x, y = np.meshgrid(range(0, fluid.nx, skip), range(0, fluid.ny, skip))
        ux_sub = fluid.ux[::skip, ::skip]
        uy_sub = fluid.uy[::skip, ::skip]
        speed = np.sqrt(ux_sub**2 + uy_sub**2)
        ax2.quiver(x, y, ux_sub.T, uy_sub.T, speed.T, cmap='coolwarm', scale=2)
        ax2.set_title('Velocity Field')
        
        # Interface detection
        ax3.imshow(fluid.rho.T, cmap='gray', origin='lower', vmin=0.5, vmax=1.5)
        
        points = fluid.find_interface_points()
        if points:
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            ax3.scatter(x_points, y_points, color='red', s=20)
            
            # Display contact angle in top right corner
            if fluid.current_angle is not None:
                angle_text = f'Contact Angle: {fluid.current_angle:.1f}°'
                ax3.text(0.95, 0.95, angle_text, transform=ax3.transAxes,
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
        
        # Update main title
        title = f'Step: {frame}'
        if fluid.has_contacted:
            title += f' (Contact at {fluid.contact_step})'
        fig.suptitle(title)
        
        plt.tight_layout()
    
    # Run animation
    anim = FuncAnimation(fig, update, frames=100, repeat=False, interval=300)
   
    plt.show()
     # Save animation as GIF
    anim.save('lbm_droplet.gif', writer=PillowWriter(fps=4))

if __name__ == '__main__':
    main()
