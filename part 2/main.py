import sys

import numpy as np
from  matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# to do:
#-tweak absorbing layers
#-4th order scheme
# calculate probability current and fft
# calculate and validate transmission 
# calculate I-V
#extra experiments


# everything in SI units



class constantsSI:
    def __init__(self):
        self.mass_electron=9.109*10**(-31) #effective mass of the electon
        self.hbar=1.054*10**(-34)   
        self.eVtoJ=1.60217663 * 10**(-19) #one electron volt in joules/ charge of an electron
        
class zone:
    """
    A class to represent a zone in the system.
    with parameters - zone width in meters
                    - potential energy in joules
                    - effective mass in kg  

    """
    def __init__(self,zone_width,potential_energy=0,effective_mass=9.109*10**(-31)):
        self.zone_width =zone_width
        self.potential_energy_J = potential_energy
        self.effective_mass = effective_mass

        self.wavevector = None 

    def calculate_wavevector(self,energy_current_J):  
        #calculating the wave vector in the zone using the energy of the current and the potential energy of the zone
        hbar = constantsSI().hbar
        wave_vector = np.sqrt(2*self.effective_mass*(energy_current_J-self.potential_energy_J)+0j)/hbar
        return wave_vector 


class zones:
    """
    This class will be used to store and manipulate the different zones of the system.
    warning the zones will be read left to right so append from left to right and not the other way around, otherwise the interface matrix will be wrong.
    """
    def __init__(self):
        self.zonesarray = []
        self.amount_zones = 0

    def add_zone(self,zones):
        self.zonesarray.append(zones)
        self.amount_zones = len(self.zonesarray)

    def propagate_matrix(self,zone):
        matrix= np.array([[np.exp(1j*zone.wavevector*zone.zone_width),                                          0],
                        [0                                        ,np.exp(-1j*zone.wavevector*zone.zone_width)]]
                        )
        
        return matrix

    def interface_matrix(self,zoneleft,zoneright):
        kleft = zoneleft.wavevector
        kright = zoneright.wavevector
        diagonalterm = 1 /2 * (1 + kright/kleft)
        offdiagonalterm = 1 /2 * (1 - kright/kleft)
        matrix = np.array([[diagonalterm, offdiagonalterm],
                        [offdiagonalterm, diagonalterm]]
                        )
        return matrix

    def validate_transmission(self,energy_current_J):
        Matrices = np.eye(2)  # Start with the identity matrix
        for i in range(self.amount_zones-1):
            self.zonesarray[i].wavevector = self.zonesarray[i].calculate_wavevector(energy_current_J)
            self.zonesarray[i+1].wavevector = self.zonesarray[i+1].calculate_wavevector(energy_current_J)
            Matrices = Matrices @ self.interface_matrix(self.zonesarray[i], self.zonesarray[i+1]) @ self.propagate_matrix(self.zonesarray[i+1])    

        T = 1 / np.abs(Matrices[0, 0])**2
        return T 
    
    def plot_zones(self):
        # this function will plot the potential energy of the zones as a function of position
        x = []
        V = []
        current_position = 0
        for zone in self.zonesarray:
            x.append(current_position)
            V.append(zone.potential_energy_J)
            current_position += zone.zone_width
            x.append(current_position)
            V.append(zone.potential_energy_J)

        plt.plot(x, V)
        plt.xlabel("Position (m)")
        plt.ylabel("Potential Energy (J)")
        plt.title("Potential Energy Profile")
        plt.show()

    def plot_transmissionspectrum(self,energies):
        """
        energies: a numpy array of energies in joules
        """
        T_values = []
        for E in energies:
            T = self.validate_transmission(E)  # scalar
            T_values.append(T)
        
        plt.plot(energies/c.eVtoJ, T_values)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission Coefficient")
        plt.title("Transmission Coefficient vs Energy")

        plt.show()


class Simulation:
    def __init__(self,zones,Nt,Nx,simulation_length,dc_voltage_eV):
        self.c = constantsSI()
        self.hbar = self.c.hbar
        self.effective_mass = 0.023 * self.c.mass_electron #value given from course       
        self.dc_voltage_eV = dc_voltage_eV
        self.Nt = Nt
        self.Nx = Nx
        self.simulation_length = simulation_length
        self.zones = zones

        self.initialize_discretization(zones)

        self.psi_real = np.zeros(self.num_points)
        self.psi_imag = np.zeros(self.num_points)
        self.current_density = np.zeros(self.num_points)


        self.V = self.build_potential()
        self.Vdamp = self.damping_layer()
        self.frames = []
    
    def initialize_discretization(self,zones):
        # wave characteristics and spatial discretization
        self.Energy_current_eV = self.dc_voltage_eV #eV
        Energy_current_J = Energy_current_eV * self.c.eVtoJ #J
        self.wave_vector = np.sqrt(2*self.effective_mass*Energy_current_J)/self.hbar

        wave_length = 2*np.pi/self.wave_vector
        self.delta_x = wave_length/20
        self.num_points = int(self.simulation_length/self.delta_x)
        self.max_potential = max(z.potential_energy_J for z in zones)

        stability_condition= 2/(((2*self.hbar/self.effective_mass)*(1/(self.delta_x**2)))+(np.max(self.max_potential)/self.hbar)) 
        self.CFL = 1.0
        self.delta_t=self.CFL * stability_condition
        self.total_time = self.Nt * self.delta_t

    def initialize_wavefunction(self):
        # this function will initialize the wave function as a gaussian wave packet
        x = np.linspace(0, self.simulation_length, self.num_points)
        self.psi_real = np.exp(-(x - self.simulation_length/4)**2 / (2*(self.simulation_length/20)**2)) * np.cos(self.wave_vector * x)
        self.psi_imag = np.exp(-(x - self.simulation_length/4)**2 / (2*(self.simulation_length/20)**2)) * np.sin(self.wave_vector * x)
        self.probability_density = np.sqrt(self.psi_real**2 + self.psi_imag**2)/np.linalg.norm(np.sqrt(self.psi_real**2 + self.psi_imag**2),1)

    def build_potential(self):
        potential = np.zeros(self.num_points)
        current_position = 0
        for zone in self.zones:
            start_index = int(current_position / self.delta_x)
            end_index = int((current_position + zone.zone_width) / self.delta_x)
            potential[start_index:end_index] = zone.potential_energy_J
            current_position += zone.zone_width
        return potential
    
    
    def damping_layer(self):
        Vdamp = np.zeros(self.num_points)  
        Nlayer = int(0.1 * self.num_points)
        m = 4 
        sigma = 1.3267185763347041e-16
        
        for i in range(Nlayer):
            # Linker rand
            Vdamp[i] =  sigma * (i / Nlayer)**m
            # Rechter rand  
            Vdamp[-i-1] =  sigma * (i / Nlayer)**m
        
        return Vdamp

    def laplacian_second_order(self,field):
        # this function will calculate the second order laplacian of the wave function phi using central differences
        laplacian = np.zeros_like(field)
        laplacian[1:-1] = (field[2:] + field[:-2] - 2*field[1:-1]) / self.delta_x**2
        return laplacian
        

    def update_equations(self):#  without damping
        self.psi_real -= ((self.hbar*self.delta_t/(2*self.effective_mass))*self.laplacian_second_order(self.psi_imag) 
                        - self.delta_t/self.hbar*(1+self.Vdamp)*self.V*self.psi_imag)
        self.psi_imag += ((self.hbar*self.delta_t/(2*self.effective_mass))*self.laplacian_second_order(self.psi_real) 
                          - self.delta_t/self.hbar*self.V*self.psi_real)
        self.current_density[:-1] = (self.hbar/(2*self.effective_mass*self.delta_x)) * (self.psi_real[:-1]*self.psi_imag[1:] - self.psi_real[1:]*self.psi_imag[:-1])
        self.probability_density= np.sqrt(self.psi_real**2 + self.psi_imag**2)
        self.frames.append(self.current_density.copy())


    def run(self, snapshot_interval=None):
        """Run simulation, storing only every Nth frame for animation."""
        if snapshot_interval is None:
            # Store ~200 frames max for smooth animation
            snapshot_interval = max(1, self.Nt // 200)
        
        for step in range(self.Nt):
            self.update_equations()
            if step % snapshot_interval == 0:
                self.frames.append(self.probability_density.copy())
        
        print(f"Stored {len(self.frames)} frames (every {snapshot_interval}th step)")

    def animate(self, interval_ms=30, show_potential=True):
        """
        Optimized animation with reduced frames and proper blitting.
        
        Parameters:
            interval_ms: delay between frames in milliseconds (lower = faster)
            show_potential: overlay the potential barrier
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.linspace(0, self.simulation_length, self.num_points)
        
        # Pre-compute potential for overlay
        if show_potential:
            V_scaled = self.V * (np.max(self.frames[0]) / np.max(np.abs(self.V) + 1e-30)) * 0.3
            ax.plot(x * 1e9, V_scaled, 'b--', lw=1, alpha=0.6, label='Potential')
        
        # Initialize line with first frame
        line, = ax.plot(x * 1e9, self.frames[0], lw=2, color='firebrick', label='|ψ|')
        
        # Dynamic y-limits based on all frames (pre-computed)
        global_max = np.max(self.frames) * 1.1
        ax.set_ylim(0, global_max)
        ax.set_xlim(0, self.simulation_length * 1e9)
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Wave Packet Propagation')
        ax.legend(loc='upper right')
        
        # Time display
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Pre-compute time labels
        dt_per_frame = self.total_time / len(self.frames)
        time_labels = [f'{i * dt_per_frame * 1e15:.1f} fs' for i in range(len(self.frames))]
        
        def init():
            line.set_ydata(self.frames[0])
            time_text.set_text(time_labels[0])
            return line, time_text
        
        def update(frame_idx):
            line.set_ydata(self.frames[frame_idx])
            time_text.set_text(time_labels[frame_idx])
            return line, time_text
        
        # Use blit=True for speed, but only if y-limits are fixed
        ani = FuncAnimation(fig, update, frames=len(self.frames),
                            init_func=init, blit=True, interval=interval_ms,
                            repeat=False)
        
        plt.tight_layout()
        plt.show()
        return ani

if __name__ == "__main__":

    c = constantsSI()
    hbar = c.hbar
    effective_mass = 0.023 * c.mass_electron #value given from course


    # wave characteristics and spatial discretization
    Energy_current_eV = 1.6 #eV
    Energy_current_J = Energy_current_eV * c.eVtoJ #J
    wave_vector = np.sqrt(2*effective_mass*Energy_current_J)/hbar

    wave_length = 2*np.pi/wave_vector
    Nt = 1000
    Nx = 1000

    # barrier creation
    barrier_height_eV = 1.0 #eV
    barrier_height_J = barrier_height_eV * c.eVtoJ

    wave_vector_barrier = np.sqrt(2*effective_mass*(Energy_current_J-barrier_height_J))/hbar
    print("wave vector in the barrier", wave_vector_barrier)
    print("wave vector in the zones", wave_vector)

    #creating nodes
    zone1_width = 20*1e-9 #m
    barrier1_width = 5*1e-9 #m
    zone2_width = 15*1e-9 #m
    barrier2_width = 5*1e-9 #m
    zone3_width = 20*1e-9 #m

    system = zones()

    system.add_zone( zone(zone1_width,0,effective_mass))
    system.add_zone( zone(barrier1_width,barrier_height_J,effective_mass)) 
    system.add_zone( zone(zone2_width,0,effective_mass))
    system.add_zone( zone(barrier2_width,barrier_height_J,effective_mass))
    system.add_zone( zone(zone3_width,0,effective_mass)   )
    energies = np.linspace(0.01, 2.0, 500) * c.eVtoJ


    system.plot_zones()
    system.plot_transmissionspectrum(energies)


    zonesarray = system.zonesarray

    Nt = 1000
    Nx = 1000
    simulation_length = sum(zone.zone_width for zone in zonesarray)

    sim= Simulation(zonesarray, Nt,Nx,simulation_length,dc_voltage_eV=Energy_current_eV)
    sim.initialize_wavefunction()
    sim.run()
    sim.animate()



    #geometry 
