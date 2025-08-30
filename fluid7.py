"""
Lattice Boltzmann Method (LBM) simulation of a liquid droplet on a solid surface.
This simulation models:
1. Surface tension (liquid-liquid cohesion)
2. Wetting behavior (liquid-surface adhesion)
3. Interface dynamics (liquid-solid interaction)
4. Contact angle formation
Using the D2Q9 lattice model with Shan-Chen pseudopotential for multiphase flow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import logging
from matplotlib.colors import LinearSegmentedColormap
import csv
from datetime import datetime as dt

logger = logging.getLogger(__name__)

class LBMFluid:
    def __init__(self, nx=200, ny=300, tau=0.6):
        # Physical grid parameters
        self.nx, self.ny = nx, ny
        self.tau = tau
        self.omega = 1.0 / tau
        
        # Physical parameters for impact dynamics
        self.gravity = 1.0
        self.surface_tension = -2.0
        self.wall_adhesion = -2.0
        
        # Impact parameters
        self.impact_velocity = 1.0
        self.weber_number = 15
        
        # D2Q9 lattice parameters
        self.velocities = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1], 
                                  [1,1], [-1,-1], [1,-1], [-1,1]])
        self.weights = np.array([4/9] + [1/9]*4 + [1/36]*4)
        self.opposites = np.array([0, 2, 1, 4, 3, 7, 6, 5, 8])
        
        # Fluid parameters
        self.rho0 = 5.0
        self.g_aa = self.surface_tension
        self.c_s_sq = 1/3
        
        # Initialize fields
        self.rho = np.ones((nx, ny)) * 0.1
        self.ux = np.zeros((nx, ny))
        self.uy = np.zeros((nx, ny))
        
        # Create solid surfaces (top and bottom walls)
        wall_thickness = 6
        self.wall_mask = np.zeros((nx, ny), dtype=bool)
        self.wall_mask[:, :wall_thickness] = True  # Bottom wall
        self.wall_mask[:, -wall_thickness:] = True  # Top wall
        self.rho[self.wall_mask] = 100.0  # Much higher solid wall density
        
        # Create initial droplet with initial velocity
        x, y = np.meshgrid(range(nx), range(ny), indexing='ij')
        r = np.sqrt((x - nx/2)**2 + (y - 2*ny/20)**2)
        droplet_radius = 20
        self.rho[r < droplet_radius] = 1.8
        self.uy[r < droplet_radius] = -self.impact_velocity
        
        # Initialize distribution functions
        self.f = np.zeros((nx, ny, 9))
        for i in range(9):
            cx, cy = self.velocities[i]
            cu = cx * self.ux + cy * self.uy
            usqr = self.ux**2 + self.uy**2
            self.f[:, :, i] = self.weights[i] * self.rho * (
                1 + 3*cu + 4.5*cu**2 - 1.5*usqr
            )
        
        # Storage for interface analysis
        self.circle_params = None
        self.contact_angles = []
        self.contact_times = []
        self.has_contacted = False
        self.contact_step = None
        self.angle_readings = 0
        self.required_readings = 50
        self.reading_interval = 5

    def psi(self, density):
        """Pseudopotential function for non-ideal gas behavior
        Creates surface tension through particle interaction
        Hyperbolic tangent form ensures bounded force"""
        density = np.clip(density, 0.1, 2.0)  # Numerical stability
        return self.rho0 * np.tanh(density / self.rho0)

    def zou_he_walls(self):
        """Non-absorbing wall boundary conditions"""
        wall_thickness = 6
        
        # Bottom wall boundary condition
        for i in range(9):
            if self.velocities[i, 1] < 0:  # Particles moving downward
                self.f[:, :wall_thickness, i] = self.f[:, :wall_thickness, self.opposites[i]]
        
        # Top wall boundary condition
        for i in range(9):
            if self.velocities[i, 1] > 0:  # Particles moving upward
                self.f[:, -wall_thickness:, i] = self.f[:, -wall_thickness:, self.opposites[i]]
        
        # Force zero velocity in wall regions
        self.ux[self.wall_mask] = 0
        self.uy[self.wall_mask] = 0
        
        # Maintain wall density
        self.rho[self.wall_mask] = 100.0

    def calculate_forces(self):
        """Calculate forces for impact dynamics"""
        force = np.zeros((self.nx, self.ny, 2))
        psi_rho = self.psi(self.rho)
        
        # Surface tension forces
        for i, (cx, cy) in enumerate(self.velocities):
            psi_neighbors = np.roll(np.roll(psi_rho, cx, axis=0), cy, axis=1)
            force[:, :, 0] += self.weights[i] * cx * psi_neighbors
            force[:, :, 1] += self.weights[i] * cy * psi_neighbors
        
        force *= -self.g_aa * psi_rho[:, :, np.newaxis]
        
        # Gravity and impact forces
        density_factor = (self.rho - 0.1) / 1.7
        force[:, :, 1] -= self.gravity * density_factor
        
        # Add impact-induced spreading force
        impact_zone = (self.rho > 1.0) & (~self.wall_mask)  # Exclude wall regions
        height = np.arange(self.ny)[np.newaxis, :] / self.ny
        spread_factor = np.exp(-height * 5)
        force[:, :, 0] *= (1 + spread_factor * impact_zone)
        
        # Add repulsion force near solid walls
        wall_repulsion = np.zeros_like(force)
        wall_distance = np.minimum(
            np.arange(self.ny)[np.newaxis, :],
            self.ny - 1 - np.arange(self.ny)[np.newaxis, :]
        )
        repulsion_strength = 0.2 * np.exp(-wall_distance / 5)
        wall_repulsion[:, :, 1] = repulsion_strength * (self.rho > 1.0)
        force += wall_repulsion
        
        # Zero force in wall regions
        force[self.wall_mask] = 0
        
        # Prevent fluid from appearing out of nowhere
        force[self.rho < 0.1] = 0  # No force in empty regions
        
        return force

    def find_interface_points(self):
        """Detect liquid-gas interface using density gradient"""
        try:
            density = gaussian_filter(self.rho, sigma=1.0)
            
            # Calculate density gradient magnitude
            grad_x = np.gradient(density, axis=0)
            grad_y = np.gradient(density, axis=1)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Interface is where gradient is large
            threshold = np.max(gradient_magnitude) * 0.3
            interface_points = gradient_magnitude > threshold
            
            # Focus on bottom half for contact angle
            y_center = self.ny // 2
            x_center = self.nx // 2
            
            points = np.where(interface_points)
            valid_points = []
            for y, x in zip(points[0], points[1]):
                if y > y_center and abs(x - x_center) < self.nx//4:
                    valid_points.append((x, y))
                    
            return valid_points
        except Exception as e:
            print(f"Error in find_interface_points: {e}")
            return []

    def calculate_contact_angle(self):
        """Measure contact angle between droplet and surface"""
        try:
            points = self.find_interface_points()
            if len(points) < 10:  # Need sufficient points for reliable fit
                return None
                
            x_points = np.array([p[0] for p in points])
            y_points = np.array([p[1] for p in points])
            
            # Sort points for consistent fitting
            sort_idx = np.argsort(x_points)
            x_points = x_points[sort_idx]
            y_points = y_points[sort_idx]
            
            def circle(x, x0, y0, r):
                """Circle equation for fitting droplet shape"""
                inside = r**2 - (x - x0)**2
                inside = np.maximum(inside, 0)  # Prevent imaginary values
                return y0 + np.sqrt(inside)
                
            # Estimate initial parameters from data
            x0_guess = np.mean(x_points)  # Center x
            y0_guess = np.min(y_points)   # Bottom y
            r_guess = np.std(x_points)    # Approximate radius
            
            # Physical constraints on circle parameters
            bounds = (
                [x0_guess - r_guess, y0_guess - r_guess, 0],
                [x0_guess + r_guess, y0_guess + r_guess, r_guess * 2]
            )
            
            # Fit circle to interface points
            popt, pcov = curve_fit(circle, x_points, y_points, 
                                 p0=[x0_guess, y0_guess, r_guess],
                                 bounds=bounds,
                                 maxfev=10000)
            
            x0, y0, r = popt
            
            # Calculate contact angle from circle geometry
            x_contact = np.min(x_points)  # Leftmost interface point
            dx = x_contact - x0
            
            if r**2 - dx**2 < 0:
                return None
                
            dy = np.sqrt(r**2 - dx**2)
            # Contact angle is between tangent line and surface
            angle = np.arctan2(dy, abs(dx)) * 180 / np.pi
            
            self.circle_params = (x0, y0, r)
            
            return angle if 0 <= angle <= 180 else None
            
        except Exception as e:
            print(f"Error in calculate_contact_angle: {e}")
            return None

    def equilibrium(self):
        """Calculate equilibrium distribution functions
        Based on local density and velocity
        Includes up to second order terms in velocity"""
        f_eq = np.zeros_like(self.f)
        # Limit velocities for stability
        ux_safe = np.clip(self.ux, -0.1, 0.1)
        uy_safe = np.clip(self.uy, -0.1, 0.1)
        usqr = ux_safe**2 + uy_safe**2
        
        for i, (cx, cy) in enumerate(self.velocities):
            cu = cx*ux_safe + cy*uy_safe  # Velocity projection
            # Second-order equilibrium distribution
            f_eq[:, :, i] = self.weights[i] * self.rho * (
                1 + 3*cu + 4.5*cu**2 - 1.5*usqr
            )
        return f_eq

    def is_settled(self):
        """Check if the droplet has settled on the surface"""
        # Calculate average velocity in the droplet region
        droplet_mask = self.rho > 1.0
        avg_velocity = np.sqrt(self.ux[droplet_mask]**2 + self.uy[droplet_mask]**2).mean()
        return avg_velocity < 0.1

    def step(self):
        """Enhanced stepping for impact dynamics"""
        try:
            # Original macroscopic calculations
            self.rho = np.sum(self.f, axis=2)
            self.rho = np.clip(self.rho, 0.1, 2.0)
            
            # Calculate velocities
            self.ux = np.zeros_like(self.rho)
            self.uy = np.zeros_like(self.rho)
            for i, (cx, cy) in enumerate(self.velocities):
                self.ux += cx * self.f[:, :, i]
                self.uy += cy * self.f[:, :, i]
            
            # Normalize velocities
            self.ux = np.divide(self.ux, self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
            self.uy = np.divide(self.uy, self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
            
            # Apply forces
            force = self.calculate_forces()
            self.ux += np.divide(force[:, :, 0], self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
            self.uy += np.divide(force[:, :, 1], self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
            
            # Collision step with enhanced stability
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
            
            # Check for contact and record angles
            if not self.has_contacted:
                points = self.find_interface_points()
                if points:
                    min_y = min(p[1] for p in points)
                    if min_y <= wall_thickness + 2:  # Within 2 lattice units of bottom wall
                        self.has_contacted = True
                        self.contact_step = len(self.contact_times)
                        print(f"Initial contact detected at step {self.contact_step}")
            
            # Record contact angles after contact
            if self.has_contacted and self.angle_readings < self.required_readings:
                if len(self.contact_times) % self.reading_interval == 0:
                    angle = self.calculate_contact_angle()
                    if angle is not None:
                        self.contact_angles.append(angle)
                        self.contact_times.append(len(self.contact_times))
                        self.angle_readings += 1
                        print(f"Contact angle reading {self.angle_readings}/{self.required_readings}: {angle:.1f}°")
                        # Save to CSV immediately
                        with open(filename, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([len(self.contact_times), angle])
            
        except Exception as e:
            print(f"Error in step: {e}")

def main():
    try:
        # Create simulation with larger domain
        fluid = LBMFluid(nx=200, ny=300)
        fig = plt.figure(figsize=(15, 10))
        
        # Create three subplots
        ax1 = plt.subplot(131)  # Density field
        ax2 = plt.subplot(132)  # Velocity field
        ax3 = plt.subplot(133)  # Interface detection
        
        # Create a file to save the contact angle data
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        filename = f'contact_angle_data_{timestamp}.csv'
        
        # Open file for writing and write header
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'contact_angle'])
        
        def update(frame):
            try:
                # Take only one step per frame
                fluid.step()
                
                # Clear all axes
                ax1.clear()
                ax2.clear()
                ax3.clear()
                
                # 1. Density plot
                density_plot = ax1.imshow(fluid.rho.T, cmap='viridis', 
                                        origin='lower', vmin=0.5, vmax=1.5)
                ax1.set_title('Density Field')
                
                # 2. Velocity field plot
                skip = 5
                x, y = np.meshgrid(range(0, fluid.nx, skip), range(0, fluid.ny, skip))
                ux_sub = fluid.ux[::skip, ::skip]
                uy_sub = fluid.uy[::skip, ::skip]
                speed = np.sqrt(ux_sub**2 + uy_sub**2)
                ax2.quiver(x, y, ux_sub.T, uy_sub.T, speed.T, 
                          cmap='coolwarm', scale=2)
                ax2.set_title('Velocity Field')
                
                # 3. Interface detection plot
                ax3.imshow(fluid.rho.T, cmap='gray', origin='lower', 
                          vmin=0.5, vmax=1.5)
                
                # Draw interface points
                points = fluid.find_interface_points()
                if points:
                    x_points = [p[0] for p in points]
                    y_points = [p[1] for p in points]
                    ax3.scatter(x_points, y_points, color='red', s=20, 
                               label='Interface')
                    
                    # Draw fitted circle if available
                    angle = fluid.calculate_contact_angle()
                    if angle is not None and fluid.has_contacted:
                        # Draw the fitted circle
                        if fluid.circle_params is not None:
                            x0, y0, r = fluid.circle_params
                            circle = plt.Circle((x0, y0), r, fill=False, 
                                              color='yellow', linestyle='--')
                            ax3.add_artist(circle)
                        
                        # Add contact angle information to title
                        title = f'Contact Angle: {angle:.1f}°'
                        if fluid.has_contacted:
                            time_since_contact = frame - fluid.contact_step
                            title += f'\nTime since contact: {time_since_contact} steps'
                            title += f'\nReadings: {fluid.angle_readings}/{fluid.required_readings}'
                        ax3.set_title(title)
                    
                ax3.set_title('Interface Detection')
                
                # Add step counter and contact information
                title = f'Simulation Step: {frame}'
                if fluid.has_contacted:
                    title += f' (Contact at step {fluid.contact_step})'
                    title += f' (Readings: {fluid.angle_readings}/{fluid.required_readings})'
                fig.suptitle(title)
                
                # Adjust layout
                plt.tight_layout()
                
            except Exception as e:
                print(f"Error in update: {e}")
        
        # Run animation
        anim = FuncAnimation(fig, update, frames=500, repeat=False, interval=200)
        plt.show()
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == '__main__':
    main() 
