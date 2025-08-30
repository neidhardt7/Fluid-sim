import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class FloryHugginsModel:
    """
    Implementation of the Flory-Huggins theory for liquid-liquid phase separation
    in binary polymer solutions.
    """
    
    def __init__(self, N1=1, N2=100):
        """
        Initialize the Flory-Huggins model.
        
        Parameters:
        -----------
        N1 : float
            Degree of polymerization of component 1 (solvent)
        N2 : float
            Degree of polymerization of component 2 (polymer)
        """
        self.N1 = N1
        self.N2 = N2
    
    def free_energy_mixing(self, phi, chi):
        """
        Calculate the Flory-Huggins free energy of mixing per lattice site.
        
        Parameters:
        -----------
        phi : float
            Volume fraction of component 2 (polymer)
        chi : float
            Flory-Huggins interaction parameter
            
        Returns:
        --------
        float
            Free energy of mixing per lattice site
        """
        phi1 = 1 - phi  # Volume fraction of component 1 (solvent)
        return (phi1 * np.log(phi1) / self.N1 + 
                phi * np.log(phi) / self.N2 + 
                chi * phi1 * phi)
    
    def chemical_potential_difference(self, phi, chi):
        """
        Calculate the difference in chemical potential between components.
        
        Parameters:
        -----------
        phi : float
            Volume fraction of component 2 (polymer)
        chi : float
            Flory-Huggins interaction parameter
            
        Returns:
        --------
        float
            Chemical potential difference
        """
        phi1 = 1 - phi
        return (np.log(phi) / self.N2 - np.log(phi1) / self.N1 + 
                chi * (1 - 2 * phi))
    
    def find_binodal(self, chi):
        """
        Find the binodal points (phase separation concentrations) for a given chi.
        
        Parameters:
        -----------
        chi : float
            Flory-Huggins interaction parameter
            
        Returns:
        --------
        tuple
            (phi_alpha, phi_beta) - Volume fractions of the two phases
        """
        def objective(phi):
            # For phase equilibrium, chemical potentials must be equal
            return self.chemical_potential_difference(phi, chi)
        
        # Initial guesses for the two phases
        phi_alpha = fsolve(objective, 0.1)[0]
        phi_beta = fsolve(objective, 0.9)[0]
        
        return phi_alpha, phi_beta
    
    def plot_phase_diagram(self, chi_range=(0, 2), n_points=100):
        """
        Plot the phase diagram showing binodal and spinodal curves.
        
        Parameters:
        -----------
        chi_range : tuple
            Range of chi values to plot
        n_points : int
            Number of points to calculate
        """
        chi_values = np.linspace(chi_range[0], chi_range[1], n_points)
        binodal_alpha = []
        binodal_beta = []
        
        for chi in chi_values:
            try:
                phi_alpha, phi_beta = self.find_binodal(chi)
                binodal_alpha.append(phi_alpha)
                binodal_beta.append(phi_beta)
            except:
                continue
        
        plt.figure(figsize=(10, 6))
        plt.plot(binodal_alpha, chi_values, 'b-', label='Binodal (α phase)')
        plt.plot(binodal_beta, chi_values, 'b-', label='Binodal (β phase)')
        
        # Calculate and plot spinodal
        phi_values = np.linspace(0.01, 0.99, 100)
        spinodal_chi = []
        for phi in phi_values:
            chi_spinodal = 0.5 * (1/(self.N1 * (1-phi)) + 1/(self.N2 * phi))
            spinodal_chi.append(chi_spinodal)
        
        plt.plot(phi_values, spinodal_chi, 'r--', label='Spinodal')
        
        plt.xlabel('Volume fraction of polymer (φ)')
        plt.ylabel('Flory-Huggins parameter (χ)')
        plt.title('Phase Diagram (Flory-Huggins Theory)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a model with N1=1 (solvent) and N2=100 (polymer)
    model = FloryHugginsModel(N1=1, N2=100)
    
    # Plot the phase diagram
    model.plot_phase_diagram(chi_range=(0, 2))
    
    # Example of calculating free energy for specific conditions
    phi = 0.5
    chi = 0.5
    free_energy = model.free_energy_mixing(phi, chi)
    print(f"Free energy of mixing at φ={phi}, χ={chi}: {free_energy:.4f}") 