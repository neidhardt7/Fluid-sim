"""
Lattice Boltzmann Method (LBM) simulation of a liquid droplet on a surface.
This simulation models:
1. Surface tension (liquid-liquid cohesion)
2. Wetting behavior (liquid-surface adhesion)
3. Interface dynamics (liquid-gas interaction)
4. Contact angle formation
Using the D2Q9 lattice model with Shan-Chen pseudopotential for multiphase flow.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)

class LBMFluid:
    def __init__(self, nx=100, ny=100, tau=1.0):
        # Physical grid parameters
        # nx, ny: domain size in lattice units
        # tau: relaxation time, related to fluid viscosity by: viscosity = (tau - 0.5)/3
        self.nx, self.ny = nx, ny
        self.tau = tau
        self.omega = 1.0 / tau  # Collision frequency
        
        # Physical parameters
        self.gravity = 9.8e-5  # Gravitational acceleration (reduced for stability)
        self.surface_tension = -1.0  # Surface tension coefficient
        self.velocity_scale = 2.0  # Scale factor for velocities
        
        # D2Q9 lattice velocities: rest particle + 8 directions
        # Arranged as: rest, horizontal/vertical, diagonal directions
        self.velocities = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1], 
                                  [1,1], [-1,-1], [1,-1], [-1,1]])
        # Lattice weights chosen to ensure isotropy up to 4th order
        self.weights = np.array([4/9] + [1/9]*4 + [1/36]*4)
        
        # Opposite directions for bounce-back
        self.opposites = np.array([0, 2, 1, 4, 3, 7, 6, 5, 8])
        
        # Fluid physical parameters
        self.rho0 = 1.0  # Reference density for pseudopotential
        self.g_aa = self.surface_tension
        self.c_s_sq = 1/3  # Square of speed of sound in lattice units
        
        # Initialize macroscopic fields
        self.rho = np.ones((nx, ny))  # Density field
        self.ux = np.zeros((nx, ny))  # x-velocity field
        self.uy = np.zeros((nx, ny))  # y-velocity field
        
        # Create initial droplet configuration
        # Density contrast between liquid (1.5) and gas (0.5) phases
        # Higher density in droplet creates surface tension through pseudopotential
        x, y = np.meshgrid(range(nx), range(ny), indexing='ij')
        r = np.sqrt((x - nx/2)**2 + (y - 3*ny/4)**2)  # Start higher for gravity
        self.rho[r < 25] = 1.5  # Liquid phase (inside droplet)
        self.rho[r >= 25] = 0.5  # Gas phase (surrounding)
        
        # Initialize distribution functions to local equilibrium
        self.f = np.zeros((nx, ny, 9))
        for i in range(9):
            self.f[:, :, i] = self.weights[i] * self.rho
            
        # Storage for interface analysis
        self.circle_params = None

    def psi(self, density):
        """Pseudopotential function for non-ideal gas behavior
        Creates surface tension through particle interaction
        Hyperbolic tangent form ensures bounded force"""
        density = np.clip(density, 0.1, 2.0)  # Numerical stability
        return self.rho0 * np.tanh(density / self.rho0)

    def zou_he_bottom_wall(self):
        """Apply Zou-He boundary condition at bottom wall"""
        j = 0  # Bottom wall
        
        # Known distributions (from streaming)
        f1 = self.f[:, j, 1]
        f3 = self.f[:, j, 3]
        f4 = self.f[:, j, 4]
        f7 = self.f[:, j, 7]
        f8 = self.f[:, j, 8]
        
        # Calculate density
        rho_w = (f1 + f3 + f4 + 2*(f7 + f8))/(1 - self.uy[:, j])
        
        # Calculate unknown distributions
        self.f[:, j, 2] = f4 + (2/3)*rho_w*self.uy[:, j]
        self.f[:, j, 5] = f7 - (1/6)*rho_w*self.uy[:, j]
        self.f[:, j, 6] = f8 - (1/6)*rho_w*self.uy[:, j]

    def calculate_forces(self):
        """Calculate all forces including gravity"""
        # Surface tension forces
        force = np.zeros((self.nx, self.ny, 2))
        psi_rho = self.psi(self.rho)
        
        for i, (cx, cy) in enumerate(self.velocities):
            psi_neighbors = np.roll(np.roll(psi_rho, cx, axis=0), cy, axis=1)
            force[:, :, 0] += self.weights[i] * cx * psi_neighbors
            force[:, :, 1] += self.weights[i] * cy * psi_neighbors
        
        force *= -self.g_aa * psi_rho[:, :, np.newaxis]
        
        # Add gravitational force (only y-component)
        # Scale by density difference to make buoyancy effects visible
        force[:, :, 1] += self.gravity * (self.rho - np.min(self.rho))
        
        return np.clip(force, -0.1, 0.1)

    def find_interface_points(self):
        """Detect liquid-gas interface using density gradient
        Interface is where density changes rapidly
        Uses Gaussian smoothing to reduce noise"""
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

    def calculate_contact_angle(self):
        """Measure contact angle between droplet and surface
        Uses circle fitting to approximate droplet shape
        Contact angle is measured at intersection with surface"""
        points = self.find_interface_points()
        if len(points) < 10:  # Need sufficient points for reliable fit
            return None
            
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])
        
        # Sort points for consistent fitting
        sort_idx = np.argsort(x_points)
        x_points = x_points[sort_idx]
        y_points = y_points[sort_idx]
        
        try:
            def circle(x, x0, y0, r):
                """Circle equation for fitting droplet shape
                (x-x0)² + (y-y0)² = r²"""
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
            
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Circle fitting failed: {e}")
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

    def step(self):
        """Advance simulation by one timestep
        Consists of:
        1. Calculate macroscopic variables
        2. Apply forces
        3. Collide (relax toward equilibrium)
        4. Stream (propagate to neighboring sites)"""
        # Calculate macroscopic density and velocity
        self.rho = np.sum(self.f, axis=2)
        self.rho = np.clip(self.rho, 0.1, 2.0)  # Ensure positive density
        
        # Calculate velocities
        self.ux = np.zeros_like(self.rho)
        self.uy = np.zeros_like(self.rho)
        for i, (cx, cy) in enumerate(self.velocities):
            self.ux += cx * self.f[:, :, i]
            self.uy += cy * self.f[:, :, i]
        
        # Scale velocities for stability
        self.ux *= self.velocity_scale
        self.uy *= self.velocity_scale
        
        # Divide by density (avoid division by zero)
        self.ux = np.divide(self.ux, self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
        self.uy = np.divide(self.uy, self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
        
        # Apply forces
        force = self.calculate_forces()
        self.ux += np.divide(force[:, :, 0], self.rho, out=np.zeros_like(self.ux), where=self.rho > 0.1)
        self.uy += np.divide(force[:, :, 1], self.rho, out=np.zeros_like(self.uy), where=self.rho > 0.1)
        
        # Collision step: relax toward equilibrium
        f_eq = self.equilibrium()
        self.f += self.omega * (f_eq - self.f)
        
        # Add force contribution to distribution functions
        for i, (cx, cy) in enumerate(self.velocities):
            force_term = 3 * self.weights[i] * (force[:, :, 0] * cx + force[:, :, 1] * cy)
            self.f[:, :, i] += np.clip(force_term, -0.1, 0.1)
        
        # Ensure positive distributions
        self.f = np.maximum(self.f, 1e-10)
        
        # Streaming step: propagate to neighboring sites
        for i, (cx, cy) in enumerate(self.velocities):
            self.f[:, :, i] = np.roll(np.roll(self.f[:, :, i], cx, axis=0), cy, axis=1)
        
        # Apply Zou-He boundary condition at bottom wall
        self.zou_he_bottom_wall()

def main():
    fluid = LBMFluid()
    fig = plt.figure(figsize=(15, 5))
    
    # Create three subplots
    ax1 = plt.subplot(131)  # Density field
    ax2 = plt.subplot(132)  # Velocity field
    ax3 = plt.subplot(133)  # Interface detection
    
    # Data storage for contact angles
    angles = []
    times = []  # Store timesteps for each measurement
    has_contacted = False  # Flag to detect initial contact
    contact_time = None  # Store when contact first occurs
    quiver = None
    
    # Create a file to save the data
    with open('contact_angle_data.csv', 'w') as f:
        f.write('timestep,contact_angle\n')
    
    def update(frame):
        nonlocal has_contacted, contact_time
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
        
        # 2. Velocity field plot (downsample for clarity)
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
            
            # Check for initial contact with surface
            min_y = min(y_points)
            if min_y <= 2 and not has_contacted:  # Within 2 lattice units of bottom
                has_contacted = True
                contact_time = frame
                print(f"Initial contact detected at timestep {frame}")
            
            # Draw fitted circle if available
            angle = fluid.calculate_contact_angle()
            if angle is not None:
                angles.append(angle)
                times.append(frame)
                
                # Calculate moving average of angle
                avg_angle = np.mean(angles[-10:]) if len(angles) > 10 else angle
                
                # Save data to file
                with open('contact_angle_data.csv', 'a') as f:
                    f.write(f'{frame},{angle}\n')
                
                # Draw the fitted circle
                if fluid.circle_params is not None:
                    x0, y0, r = fluid.circle_params
                    circle = plt.Circle((x0, y0), r, fill=False, 
                                      color='yellow', linestyle='--')
                    ax3.add_artist(circle)
                
                # Add contact time information to title if available
                title = f'Contact Angle: {avg_angle:.1f}°'
                if has_contacted:
                    time_since_contact = frame - contact_time
                    title += f'\nTime since contact: {time_since_contact} steps'
                ax3.set_title(title)
            
        ax3.set_title('Interface Detection')
        
        # Add step counter and contact information
        title = f'Simulation Step: {frame}'
        if has_contacted:
            title += f' (Contact at step {contact_time})'
        fig.suptitle(title)
        
        # Adjust layout
        plt.tight_layout()
        
    # Run animation with much longer interval
    anim = FuncAnimation(fig, update, frames=500, repeat=False, interval=200)
    plt.show()
    
    # After simulation ends, plot the contact angle evolution
    if times and angles:
        plt.figure(figsize=(10, 6))
        plt.plot(times, angles, 'b-', label='Instantaneous')
        
        # Add moving average
        window = 10
        if len(angles) > window:
            moving_avg = np.convolve(angles, np.ones(window)/window, mode='valid')
            plt.plot(times[window-1:], moving_avg, 'r-', 
                    label=f'{window}-point moving average')
        
        plt.xlabel('Simulation Step')
        plt.ylabel('Contact Angle (degrees)')
        plt.title('Contact Angle Evolution')
        if contact_time is not None:
            plt.axvline(x=contact_time, color='g', linestyle='--', 
                       label='Initial Contact')
        plt.legend()
        plt.grid(True)
        plt.savefig('contact_angle_evolution.png')
        plt.show()

if __name__ == '__main__':
    main()
