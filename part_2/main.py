from matplotlib import animation
import numpy as np
from  matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.__config__ import show

# to do:
#-tweak absorbing layers
#-4th order scheme
# calculate probability current and fft
# calculate and validate transmission 
# calculate I-V
#extra experiments


# everything in SI units

# helper functions

def gaussian_envelope(x, center, width):
    return np.exp(-(x - center)**2 / (2 * width**2))

def plane_wave(x, k):

    return np.exp(1j * k * x)

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

    def calculate_wavevector(self,energy_wavefunction_J):  
        #calculating the wave vector in the zone using the energy of the current and the potential energy of the zone
        hbar = constantsSI().hbar
        wave_vector = np.sqrt(2*self.effective_mass*(energy_wavefunction_J-self.potential_energy_J)+0j)/hbar
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
        c = constantsSI()
        T_values = []
        for E in energies:
            T = self.validate_transmission(E)  # scalar
            T_values.append(T)
        
        plt.plot(energies/c.eVtoJ, T_values)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission Coefficient")
        plt.title("Transmission Coefficient vs Energy")

        plt.show()

class observer:
    def __init__(self,position):
        self.position = position
        self.probability_density = []
        self.current_density = []
    
    def plot_observables(self, delta_t):
        time = np.arange(len(self.probability_density)) * delta_t
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Probability Density', color=color)
        ax1.plot(time, self.probability_density, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Current Density (A/m)', color=color)  # we already handled the x-label with ax1
        ax2.plot(time, self.current_density, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Observables at Position {self.position:.2e} m')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

class Observers:
    def __init__(self, observer_object_list):
        self.observers = observer_object_list

    def plot_all_observables(self, delta_t):
        for obs in self.observers:
            obs.plot_observables(delta_t)




class Simulation:
    def __init__(self,zones,Nt,Nx,simulation_length,dc_voltage_eV=0,energy_current_eV = None):
        self.c = constantsSI()
        self.hbar = self.c.hbar
        self.effective_mass = 0.023 * self.c.mass_electron #value given from course       
        self.dc_voltage_eV = dc_voltage_eV # voltage applied to the system in eV
        self.energy_current_eV = energy_current_eV # energy of the current in eV
        self.numpoints_time = Nt
        self.num_points_space = Nx
        self.simulation_length = simulation_length
        self.zones = zones

        self.initialize_discretization(zones)

        self.psi_real = np.zeros(self.num_points_space)
        self.psi_imag = np.zeros(self.num_points_space)
        self.current_density = np.zeros(self.num_points_space)


        self.V = self.build_potential()
        #self.Vdamp = None
        self.Vdamp = self.initialise_damping_layer()
        self.frames = []

        self.observers = []



    def initialize_discretization(self,zones):
        # wave characteristics and spatial discretization
        Energy_current_J = self.energy_current_eV * self.c.eVtoJ #J
        self.wave_vector = np.sqrt(2*self.effective_mass*Energy_current_J)/self.hbar

        self.delta_x = self.simulation_length/self.num_points_space
        self.max_potential = max(z.potential_energy_J for z in zones)

        self.stability_condition= 2/(((2*self.hbar/self.effective_mass)*(1/(self.delta_x**2)))+(np.max(self.max_potential)/self.hbar)) 
        self.CFL = 1.0
        self.delta_t=self.CFL * self.stability_condition
        self.total_time = self.numpoints_time * self.delta_t
        print(f"Delta x: {self.delta_x:.2e} m, Delta t: {self.delta_t:.2e} s, Total time: {self.total_time:.2e} s")

    def stability_check(self):
        # this function will check the stability condition for the simulation and print a warning if it is not met
        if self.delta_t > self.stability_condition:
            print("Warning: Stability condition not met. Consider reducing delta_t or increasing delta_x.")
        else:
            print("Stability condition met.")

    def initialize_wavefunction(self):
        # this function will initialize the wave function as a gaussian wave packet
        x = np.linspace(0, self.simulation_length, self.num_points_space)

        self.psi_initial = np.exp(-(x - self.simulation_length/3)**2 / (2*(self.simulation_length/20)**2)) * np.exp(1j * self.wave_vector * x)
        self.psi_real = self.psi_initial.real
        self.psi_imag = self.psi_initial.imag
        norm = np.sqrt(np.sum(self.psi_real**2 + self.psi_imag**2) * self.delta_x)
        self.psi_real /= norm
        self.psi_imag /= norm
        self.probability_density = self.psi_real**2 + self.psi_imag**2  

    def initialize_wavefunction_quantum_well(self,n=1):
        # this function will initialize the wave function as a gaussian wave packet
        x = np.linspace(0, self.simulation_length, self.num_points_space)
        # Standing wave eigenstate: ψ_n(x) = sqrt(2/L) * sin(nπx/L)
        self.psi_real = np.sqrt(2/self.simulation_length) * np.sin(n * np.pi * x / self.simulation_length)
        self.psi_imag = np.sqrt(2/self.simulation_length) * np.cos(n * np.pi * x / self.simulation_length)
        
        norm = np.sqrt(np.sum(self.psi_real**2 + self.psi_imag**2) * self.delta_x)
        self.psi_real /= norm
        self.psi_imag /= norm
        
        self.probability_density = self.psi_real**2 + self.psi_imag**2
        print(f"Ground state normalization check: {np.sum(self.probability_density)*self.delta_x:.6f}")

    def build_potential(self):
        potential = np.zeros(self.num_points_space)
        current_position = 0
        for zone in self.zones:
            start_index = int(current_position / self.delta_x)
            end_index = int((current_position + zone.zone_width) / self.delta_x)
            potential[start_index:end_index] = zone.potential_energy_J
            current_position += zone.zone_width
        return potential
    
    
    def initialise_damping_layer(self):
        Vdamp = np.zeros(self.num_points_space)
        Nlayer = int(0.2 * self.num_points_space) # 25% of the edge is damping
        sigma = 1.3e-18 # Adjust this if it reflects
        m = 3
        for i in range(Nlayer):
            # Cubic ramp: smooth start (0) at the device, max at the wall
            # Left side
            Vdamp[i] = sigma * ((Nlayer - i) / Nlayer)**m
            # Right side
            Vdamp[-i-1] = sigma * ((Nlayer - i) / Nlayer)**m
        return Vdamp
        
    def plot_damping_potential(self,Vdamp):
        x = np.linspace(0, self.simulation_length, self.num_points_space)
        plt.plot(x,Vdamp)
        plt.title("the damping potential")
        plt.xlabel("index point in domain")
        plt.ylabel("absorbing potential")
        plt.show()

    def laplacian_second_order(self,field):
        # this function will calculate the second order laplacian of the wave function phi using central differences
        laplacian = np.zeros_like(field)
        laplacian[1:-1] = (field[2:] + field[:-2] - 2*field[1:-1]) / self.delta_x**2
        return laplacian
        
    def _update_real_part(self):
        # Pre-calculate alpha for speed
        alpha = self.delta_t / (2 * self.hbar)
        
        # 1. Update REAL part
        kinetic_real = (self.hbar * self.delta_t / (2 * self.effective_mass)) * self.laplacian_second_order(self.psi_imag)
        potential_real = (self.delta_t / self.hbar) * self.V * self.psi_imag
        
        # The Damping logic: (1 - alpha*Gamma) / (1 + alpha*Gamma)
        # Using sigma = 1.3e-19 (positive) for this logic
        self.psi_real = (self.psi_real * (1 - alpha * self.Vdamp) - (kinetic_real - potential_real)) / (1 + alpha * self.Vdamp)

    def _update_imaginary_part(self):
        alpha = self.delta_t / (2 * self.hbar)
        kinetic_imag = (self.hbar * self.delta_t / (2 * self.effective_mass)) * self.laplacian_second_order(self.psi_real)
        potential_imag = (self.delta_t / self.hbar) * self.V * self.psi_real
        
        self.psi_imag = (self.psi_imag * (1 - alpha * self.Vdamp) + (kinetic_imag - potential_imag)) / (1 + alpha * self.Vdamp)


    def _update_current_density(self):
        self.current_density[:-1] = (self.hbar/(2*self.effective_mass*self.delta_x)) * (self.psi_real[:-1]*self.psi_imag[1:] - self.psi_real[1:]*self.psi_imag[:-1])
    
    def _update_normalized_probability_density(self):
        self.probability_density= (self.psi_real**2 + self.psi_imag**2)/ (np.sum(self.psi_real**2 + self.psi_imag**2) * self.delta_x)
    def _update_probability_density(self):
        self.probability_density= (self.psi_real**2 + self.psi_imag**2)
        
    def _set_boundary_conditions_quantum_well(self):
        # this function will set the boundary conditions for the wave function, we will use absorbing boundary conditions to prevent reflections at the boundaries
        self.psi_real[0] = 0
        self.psi_real[-1] = 0
        self.psi_imag[0] = 0
        self.psi_imag[-1] = 0


    def add_observers(self,observers_container):
        self.observers=observers_container.observers
    
    def _update_observers(self):
        for obs in self.observers:
            index = int(obs.position / self.delta_x)
            obs.probability_density.append(self.probability_density[index])
            obs.current_density.append(self.current_density[index])
        
    
    def update_equations(self):
        self._update_real_part()
        self._update_imaginary_part()
        self._update_current_density()
        self._update_probability_density()
        #self._set_boundary_conditions()
        self._update_observers()


    def run(self, snapshot_interval=None):
        """Run simulation, storing only every Nth frame for animation."""
        if snapshot_interval is None:
            # Store ~200 frames max for smooth animation
            snapshot_interval = max(1, 30* self.num_points_space // self.numpoints_time)
        
        for step in range(self.numpoints_time):
            self.update_equations()
            if step % snapshot_interval == 0:
                self.frames.append(self.probability_density.copy())
        
        print(f"Stored {len(self.frames)} frames (every {snapshot_interval}th step)")

    
    def analytical_quantum_well(self):
        currentenergy_J = self.energy_current_eV * self.c.eVtoJ
        self.wave_vector = np.sqrt(2 * self.effective_mass * currentenergy_J) / self.hbar
        # this function will calculate the analytical solution for the quantum well
        x = np.linspace(0, self.simulation_length, self.num_points_space)
        psi_analytical = np.sqrt(2/self.simulation_length) * np.cos(self.wave_vector * x)* np.exp(-1j*self.hbar*self.wave_vector**2/(2*self.effective_mass)*self.total_time)
        return psi_analytical
    
    def analytical_current_density(self):
        # this function will calculate the analytical current density for the quantum well
        x = np.linspace(0, self.simulation_length, self.num_points_space)
        psi_analytical = self.analytical_quantum_well()
        current_density_analytical = (self.hbar/(2*self.effective_mass*self.delta_x)) * (psi_analytical[:-1].real*psi_analytical[1:].imag - psi_analytical[1:].real*psi_analytical[:-1].imag)
        return current_density_analytical
    
    def animate_analytical(self, interval_ms=30):
        """
        Animates the analytical solution for the quantum well, showing how the probability density evolves over time.

        parameters:
            interval_ms: delay between time steps in milliseconds (lower = faster)

        returns:
            ani: the FuncAnimation object for further manipulation or saving
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.linspace(0, self.simulation_length, self.num_points_space)
        psi_analytical = self.analytical_quantum_well()     # shape (100,)
        y_initial = np.abs(psi_analytical)**2
        
        line, = ax.plot(x * 1e9, np.abs(psi_analytical)**2, lw=2, color='green', label='Analytical |ψ|')
        ax.set_ylim(0, np.max(np.abs(psi_analytical)) * 1.1)
        ax.set_xlim(0, self.simulation_length * 1e9)
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Analytical Solution for Quantum Well')
        ax.legend(loc='upper right')
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        dt_per_frame = self.total_time / len(self.frames)
        time_labels = [f'{i * dt_per_frame * 1e15:.1f} fs' for i in range(len(self.frames))]
        
        def init():
            line.set_ydata(y_initial)
            time_text.set_text(time_labels[0])
            return line, time_text
        
        def update(frame_idx):
            # Update the analytical solution based on time
            t = frame_idx * dt_per_frame
            psi_analytical_t = np.sqrt(2/self.simulation_length) * np.cos(self.wave_vector * x) * np.exp(-1j*self.hbar*self.wave_vector**2/(2*self.effective_mass)*t)
            line.set_ydata(np.abs(psi_analytical_t)**2)
            time_text.set_text(time_labels[frame_idx])
            return line, time_text
        
        ani = FuncAnimation(fig, update, frames=len(self.frames),
                            init_func=init, blit=True, interval=interval_ms,
                            repeat=False)
        
        plt.tight_layout()
        plt.show()
        return ani
    
    def animate_with_features(self, observers=None, interval_ms=20, speed_multiplier=20):
        """
        Sim: The Simulation object
        observers: A list of observer objects [obs1, obs2, ...]
        speed_multiplier: Skips frames to make the playback faster
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        x_nm = np.linspace(0, self.simulation_length, self.num_points_space) * 1e9
        
        damping_mask = np.abs(self.Vdamp) > 0
        if np.any(damping_mask):
           
            ax.fill_between(x_nm, 0, np.max(self.frames)*1.1, where=damping_mask, 
                            color='gray', alpha=0.2, label='Absorbing Layer (CAP)')

        # 2. DRAW THE OBSERVERS
        if observers:
            colors = ['blue', 'green', 'orange', 'purple']
            for i, obs in enumerate(observers.observers):
                pos_nm = obs.position * 1e9
                ax.axvline(pos_nm, color=colors[i % len(colors)], linestyle='--', 
                            alpha=0.8, label=f'Observer @ {pos_nm:.1f}nm')

        if np.max(np.abs(self.V)) > 0:
            V_scaled = self.V * (np.max(self.frames) / np.max(np.abs(self.V))) * 0.5
            ax.plot(x_nm, V_scaled, 'k-', lw=1, alpha=0.4, label='Potential Barrier')

        line, = ax.plot(x_nm, self.frames[0], lw=2, color='firebrick', label=r'$|\psi|^2$')
        
        ax.set_ylim(0, np.max(self.frames) * 1.1)
        ax.set_xlim(0, self.simulation_length * 1e9)
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Quantum Transport with Absorbing Boundaries')
        ax.legend(loc='upper right', fontsize='small')

        plot_frames = self.frames[::speed_multiplier]

        def update(i):
            line.set_ydata(plot_frames[i])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(plot_frames), 
                                    interval=interval_ms, blit=True)
        plt.show()
        return ani

    def animate(self, interval_ms=30, show_potential=True):
        """
        Optimized animation with reduced frames and proper blitting.
        
        Parameters:
            interval_ms: delay between frames in milliseconds (lower = faster)
            show_potential: overlay the potential barrier
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.linspace(0, self.simulation_length, self.num_points_space)
        
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

    def save_fast_animation(self, filename, fps=60, speed_multiplier=10,show_potential=True):
        """
        Saves a high-speed animation of the wave propagation.
        speed_multiplier: Skips frames to make the video appear faster.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.linspace(0, self.simulation_length, self.num_points_space)

        if show_potential:
            V_scaled = self.V * (np.max(self.frames[0]) / np.max(np.abs(self.V) + 1e-30)) * 0.3
            ax.plot(x * 1e9, V_scaled, 'b--', lw=1, alpha=0.6, label='Potential')
        
        # Setup axis
        ax.set_xlim(0, self.simulation_length * 1e9)
        # Scale y-axis to the maximum probability density
        ax.set_ylim(0, np.max(self.frames) * 1.1)
        
        line, = ax.plot(x * 1e9, self.frames[0], lw=2, color='blue')
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Fast Forward Wave Propagation ({speed_multiplier}x speed)')

        # We only take every Nth frame from the already sampled frames to speed it up
        plot_frames = self.frames[::speed_multiplier]

        def update(i):
            line.set_ydata(plot_frames[i])
            return line,

        ani = animation.FuncAnimation(fig, update, frames=len(plot_frames), blit=True)
        
        print(f"Saving animation to {filename}...")
        
        # Using bitrate to ensure quality and fps to control speed
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
        plt.close()
        print("Done!")
    
        
def fourier(v,dt):
            V=np.fft.fft(v,n=100*len(v))
            f=np.fft.fftfreq(len(v)*100,d=dt)
            V=np.fft.fftshift(V)*dt
            f=np.fft.fftshift(f)
            return f,V



def simulation_RDT():
    c = constantsSI()
    hbar = c.hbar
    effective_mass = 0.023 * c.mass_electron #value given from course


    # wave characteristics and spatial discretization
    Energy_current_eV = 10 #eV
    Energy_current_J = Energy_current_eV * c.eVtoJ #J
    wave_vector = np.sqrt(2*effective_mass*Energy_current_J)/hbar

    # barrier creation
    barrier_height_eV = 0.6 #eV
    barrier_height_J = barrier_height_eV * c.eVtoJ

    wave_vector_barrier = np.sqrt(2*effective_mass*(Energy_current_J-barrier_height_J))/hbar
    print("wave vector in the barrier", wave_vector_barrier)
    print("wave vector in the zones", wave_vector)

    #creating nodes
    zone1_width = 100*1e-9 #m
    barrier1_width = 5*1e-9 #m
    zone2_width = 15*1e-9 #m
    barrier2_width = 5*1e-9 #m
    zone3_width = 100*1e-9 #m

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

    Nt = 30000
    Nx = 1000
    simulation_length = sum(zone.zone_width for zone in zonesarray)

    obs1 = observer(position=3*zone1_width/4)
    obs2 = observer(position=zone1_width + barrier1_width + zone2_width/2+ barrier2_width+1*zone3_width/4)
    observers = Observers([obs1, obs2])

    sim= Simulation(zonesarray, Nt,Nx,simulation_length,energy_current_eV=Energy_current_eV)

    sim.add_observers(observers)
    sim.initialize_wavefunction()
    sim.plot_damping_potential(sim.Vdamp)
    sim.run()
    
    observers.plot_all_observables(sim.delta_t)
    sim.animate_with_features(sim, interval_ms=20, speed_multiplier=20)
    #sim.save_fast_animation(filename = "current_simulation.gif", fps=60, speed_multiplier=100,show_potential=True)

def simulation_quantum_well():
    c = constantsSI()
    hbar = c.hbar
    effective_mass = 0.023 * c.mass_electron #value given from course


    # wave characteristics and spatial discretization
   
    Energy_current_J = hbar**2/(2*effective_mass) #J
    wave_vector = np.sqrt(2*effective_mass*Energy_current_J)/hbar


    # barrier creation
    barrier_height_eV = 0 #eV
    barrier_height_J = barrier_height_eV * c.eVtoJ

    wave_vector_barrier = np.sqrt(2*effective_mass*(Energy_current_J-barrier_height_J))/hbar
    print("wave vector in the barrier", wave_vector_barrier)
    print("wave vector in the zones", wave_vector)

    #creating nodes
    zone1_width = 60*1e-9 #m

    system = zones()

    system.add_zone( zone(zone1_width,0,effective_mass))

    # wave characteristics and spatial discretization
   
    energy_current_J = hbar**2/(2*effective_mass)*(np.pi/zone1_width)**2 #J
    energy_current_eV = energy_current_J / c.eVtoJ
    wave_vector = np.sqrt(2*effective_mass*Energy_current_J)/hbar

    energies = np.linspace(0.01, 2.0, 500) * c.eVtoJ


    system.plot_zones()
    system.plot_transmissionspectrum(energies)


    zonesarray = system.zonesarray

    Nt = 40000
    Nx = 1000
    simulation_length = sum(zone.zone_width for zone in zonesarray)
    obs1 = observer(position=zone1_width/2)
    obs2 = observer(position=zone1_width/2+bar)

    sim= Simulation(zonesarray, Nt,Nx,simulation_length,energy_current_eV=energy_current_eV)
    sim.initialize_wavefunction_quantum_well()
    sim.run()
    sim.animate(interval_ms=60, show_potential=True)
    sim.animate_analytical(interval_ms=60)
    sim.save_fast_animation(filename = "quantum_well.gif", fps=60, speed_multiplier=100)

    

    #geometry 
 
if __name__ == "__main__":
    simulation_RDT()
    #simulation_quantum_well()