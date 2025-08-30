import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import laplace

class FloryHugginsSimulation:
    """
    Real-time simulation of Flory-Huggins phase separation using the Cahn-Hilliard equation.
    """
    
    def __init__(self, N1=1, N2=100, grid_size=100, dx=1.0, dt=0.1, chi=1.0, M=1.0, kappa=0.1):
        """
        Initialize the simulation parameters.
        
        Parameters:
        -----------
        N1, N2 : float
            Degrees of polymerization for components 1 and 2
        grid_size : int
            Size of the simulation grid (grid_size x grid_size)
        dx : float
            Grid spacing
        dt : float
            Time step
        chi : float
            Flory-Huggins interaction parameter
        M : float
            Mobility coefficient
        kappa : float
            Gradient energy coefficient
        """
        self.N1 = N1
        self.N2 = N2
        self.grid_size = grid_size
        self.dx = dx
        self.dt = dt
        self.chi = chi
        self.M = M
        self.kappa = kappa
        
        # Initialize the concentration field with small random fluctuations
        self.phi = 0.5 + 0.01 * np.random.randn(grid_size, grid_size)
        self.phi = np.clip(self.phi, 0.01, 0.99)  # Ensure phi stays in valid range
        
        # Store history of concentration variance
        self.variance_history = []
        self.time_history = []
        self.current_time = 0
        
        # Create figure with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main simulation plot
        self.im = self.ax1.imshow(self.phi, cmap='RdBu', vmin=0, vmax=1)
        cbar = plt.colorbar(self.im, ax=self.ax1, label='Volume fraction (φ)')
        self.ax1.set_title('Phase Separation Simulation\nRed = Polymer, Blue = Solvent')
        
        # Add text annotation for polymer phase
        self.ax1.text(0.02, 0.02, 'Red = Polymer Phase', 
                     transform=self.ax1.transAxes,
                     color='red', fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Variance plot
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Concentration Variance')
        self.ax2.set_title('Phase Separation Progress')
        self.ax2.grid(True)
        self.variance_line, = self.ax2.plot([], [], 'b-', label='Variance')
        self.threshold_line = self.ax2.axhline(y=0.1, color='r', linestyle='--', 
                                             label='Phase Separation Threshold')
        self.ax2.legend()
        
        # Set initial plot limits
        self.ax2.set_xlim(0, 1000)
        self.ax2.set_ylim(0, 0.25)
        
        # Add text for phase separation status
        self.status_text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                        verticalalignment='top', bbox=dict(boxstyle='round',
                                        facecolor='white', alpha=0.8))
    
    def calculate_variance(self):
        """Calculate the variance of the concentration field."""
        return np.var(self.phi)
    
    def is_phase_separated(self, variance):
        """Determine if phase separation has occurred based on variance threshold."""
        return variance > 0.1  # Threshold can be adjusted
    
    def chemical_potential(self, phi):
        """
        Calculate the chemical potential for the Cahn-Hilliard equation.
        """
        phi1 = 1 - phi
        # Flory-Huggins chemical potential
        mu = (np.log(phi) / self.N2 - np.log(phi1) / self.N1 + 
              self.chi * (1 - 2 * phi))
        return mu
    
    def update(self, frame):
        """
        Update the concentration field for one time step.
        """
        # Calculate chemical potential and update field
        mu = self.chemical_potential(self.phi)
        laplace_mu = laplace(mu, mode='wrap') / (self.dx**2)
        self.phi += self.dt * self.M * laplace_mu
        
        # Add noise to simulate thermal fluctuations
        noise = 0.001 * np.random.randn(self.grid_size, self.grid_size)
        self.phi += noise
        
        # Ensure phi stays in valid range
        self.phi = np.clip(self.phi, 0.01, 0.99)
        
        # Update time and calculate variance
        self.current_time += self.dt
        variance = self.calculate_variance()
        self.variance_history.append(variance)
        self.time_history.append(self.current_time)
        
        # Update plots
        self.im.set_array(self.phi)
        self.variance_line.set_data(self.time_history, self.variance_history)
        
        # Update status text
        status = "Phase Separated" if self.is_phase_separated(variance) else "Mixing"
        color = 'red' if self.is_phase_separated(variance) else 'blue'
        self.status_text.set_text(f'Status: {status}\nVariance: {variance:.4f}')
        self.status_text.set_color(color)
        
        # Update variance plot limits if needed
        if len(self.time_history) > 0:
            self.ax2.set_xlim(0, max(self.time_history))
            self.ax2.set_ylim(0, max(0.25, max(self.variance_history) * 1.1))
        
        return [self.im, self.variance_line, self.status_text]
    
    def run_simulation(self, frames=200, interval=50):
        """
        Run the simulation and create an animation.
        
        Parameters:
        -----------
        frames : int
            Number of frames to simulate
        interval : int
            Interval between frames in milliseconds
        """
        ani = FuncAnimation(self.fig, self.update, frames=frames,
                          interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
        return ani

def run_example_simulation():
    """
    Run an example simulation with default parameters.
    """
    # Create simulation with parameters that will show phase separation
    sim = FloryHugginsSimulation(
        N1=5,          # Solvent
        N2=1,        # Polymer
        grid_size=100, # 100x100 grid
        chi=1.5,       # Above critical chi for phase separation
        dt=0.1,        # Time step
        M=0.5,         # Mobility
        kappa=0.1      # Gradient energy coefficient
    )
    
    # Run the simulation for 1000 frames
    ani = sim.run_simulation(frames=1000, interval=50)
    return ani

if __name__ == "__main__":
    # Run the example simulation
    run_example_simulation() 