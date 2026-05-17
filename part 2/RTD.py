from os import name
from re import M

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
#from sympy import plot
#to do :
#-tweak absorbing layers
#-4th order sceme
#calculate probabilaty current and fft
#calculate and validate transmission
#calculate I-V
#extra experiments


#everything in SI units
class constants:
    def __init__(self):

        self.effective_mass=0.023*9.109*10**(-31) #effective mass of the electon
        self.hbar=1.054*10**(-34)   
        self.eVtoJ=1.60217663 * 10**(-19) #one electron volt in joules/ charge of an electron

c=constants()

def fourier(v,dt):
            V=np.fft.fft(v,n=100*len(v))
            f=np.fft.fftfreq(len(v)*100,d=dt)
            V=np.fft.fftshift(V)*dt
            f=np.fft.fftshift(f)
            return f,V





def propagate_matrix(zone):
    matrix= np.array([[np.exp(-1j*zone.wavevector*zone.zone_width),                                          0],
                      [0                                        ,np.exp(1j*zone.wavevector*zone.zone_width)]]
                    )
    
    return matrix

def interface_matrix(zoneleft,zoneright):
    kleft = zoneleft.wavevector
    kright = zoneright.wavevector
    diagonalterm = 1 /2 * (1 + kright/kleft)
    offdiagonalterm = 1 /2 * (1 - kright/kleft)
    matrix = np.array([[diagonalterm, offdiagonalterm],
                      [offdiagonalterm, diagonalterm]]
                      )
    return matrix

def validate_transmission(zones):
    Matrices = np.eye(2)  # Start with the identity matrix

    for i in range(len(zones)-1):
        if zones[i].wavevector != zones[i+1].wavevector:
            print("warning: wave vector changes at interface between zone", i, "and", i+1)

        Matrices = Matrices @ interface_matrix(zones[i], zones[i+1])

        # alleen propagatie toevoegen als het niet de laatste zone is
        if i + 1 < len(zones) - 1:
            Matrices = Matrices @ propagate_matrix(zones[i+1])

    T = 1 / np.abs(Matrices[0, 0])**2
    return T

def init_absorbing_layer(n,Nlayer,Nspace):
    S = -1.3*10**(-19)  #the sigma chosen negative beceuase otherwise instabilaty still needs tuning
    deg = 3   #degree of polynomial
    Vdamp=np.heaviside(-n,1)*S*((-n)/Nlayer)**deg+np.heaviside(n-Nspace,1)*S*((n-Nspace)/Nlayer)**deg  #damping potential
    

    return Vdamp

def initial_wave_packet(x_coordinates, L, bL, aL, k0):

        #initial wave packet shape:
        gaussian_center_value_=bL/2  #center of wave packet (->keep values at edges zero)
        gaussian_width=(L-bL-aL)/2.5  #deviation from center (->keep values at edges zero)
        phi_initial=np.exp(-(x_coordinates-gaussian_center_value_)**2/(4*gaussian_width**2)) #wave shape
        print(phi_initial[Nodes_absorbing_Layers+int(bL/delta_x)],"moet nul blijven")
        print(phi_initial[Nodes_absorbing_Layers],"moet nul blijven")

        #normelising wave shape
        norm = np.sqrt(np.trapezoid(np.abs(phi_initial)**2, x_coordinates))

        phi_normalized=phi_initial/norm

       
        phi_initial_real=phi_normalized*np.cos(k0*x_coordinates)
        phi_initial_imag=phi_normalized*np.sin(k0*x_coordinates)

        return phi_normalized, phi_initial_real, phi_initial_imag, 1/norm,gaussian_width



def init_animation():
        line.set_data([], [])
        # Clear the title on init
        title.set_text('')
        return line, title

def update(frame):
    global x_coordinates
    y = frames_data[frame]
    line.set_data(x_coordinates, y)
    
    # 2. Update the title text
    # You can use f-strings to format it nicely
    #title.set_text(f'Time: {frame*delta_t}')
    
    # 3. CRITICAL: Return the title in the tuple for blitting
    return line, title

def plot_damping_potential(n,Vdamp):
     n = n
    # plt.plot(n*delta_x,Vdamp)
    # plt.title("the damping potential")
    # plt.xlabel("index point in domain")
    # plt.ylabel("absorbing potential")
    # plt.show()

def init_potential_barrier(x, a, w, E):
    return E * np.heaviside(-np.abs(x - a) + w / 2, 1)

class zone:
    def __init__(self,zone_start,zone_end,k0):
        self.zone_start=zone_start
        self.zone_end=zone_end
        self.wavevector=k0
        self.zone_width =zone_end-zone_start






if __name__=="__main__":
    
    
    hbar = c.hbar
    m = c.effective_mass
    #domain
    device_Length=25*10**(-9) #meters (length of device)
    bL=300*10**(-9)      #length domain before device
    aL=100*10**(-9)      #length domain after device
    domain_length=device_Length+aL+bL    #length of domain (without absorbing layers)
    Energy_current_eV = 0.4 #eV
    Energy_current_J = Energy_current_eV * c.eVtoJ #J
    k0 = np.sqrt(2*m*Energy_current_J)/hbar
    v=hbar*k0/m

    #parameters
    
    average_wave_length=(2*np.pi)/k0  #[m]  
    delta_x=average_wave_length/18        #discretisation step (m) (based on average wave length ->prob better to use max wave length + less small needed when going to 4th order)
    Nodes_device=int(domain_length/delta_x)     #amount of space points
    print("nodes device",Nodes_device)
    Nodes_absorbing_Layers=1500     #amount of points in absorbing layer
    layer_distance=Nodes_absorbing_Layers*delta_x  #distance corresponding to layer
    print(f"layer distance: {layer_distance*1e9} nanometers")

    T=2*(domain_length+2*layer_distance)/v   #s  (time based on velocity and dimension of space)
    print(f"total length of domain: {domain_length*1e9} nanometers")
    print(f"total simulation time: {T*1e15} femtoseconds")

    nodes_domain=np.arange(-Nodes_absorbing_Layers,Nodes_device+Nodes_absorbing_Layers+1)  #the points in domain including absorbing layer
    x_coordinates=nodes_domain*delta_x  # space values(m)


    barrier_height_eV = 0.6 #height of barrier in eV
    barrier_height_J = barrier_height_eV*c.eVtoJ  # Convert eV to Joules
    barrier_width=5*10**(-9)   #width of both bariers
    a1=barrier_width/2+bL #position midle of first barrier
    a2=bL+device_Length-barrier_width/2 #position midle of seccond barrier

    thicknesbeforedevice = bL
    thicknessbarrier1 = barrier_width
    thicknessbetweenbarriers= device_Length - 2*barrier_width
    thicknessbarrier2 = barrier_width
    thicknessafterdevice = aL
    thicknesses = [thicknesbeforedevice, thicknessbarrier1, thicknessbetweenbarriers, thicknessbarrier2, thicknessafterdevice]
    wavevectors = [k0, np.sqrt(2*m*(Energy_current_J-barrier_height_J)+0j)/hbar, k0, np.sqrt(2*m*(Energy_current_J-barrier_height_J)+0j)/hbar, k0]  #wave vectors in each zone (based on energy and potential)

    zones= []
    for i in range(len(thicknesses)):
        print(f"zone {i} thickness: {thicknesses[i]*1e9} nanometers")
        zone_i = zone(sum(thicknesses[:i]), sum(thicknesses[:i+1]),wavevectors[i])  #zone before device
        zones.append(zone_i)

    print(zones[0].zone_start*1e9, zones[0].zone_end*1e9)

    print("transmission value:",validate_transmission(zones)) # validatie transmission using transfer matrix method



    Vdamp=init_absorbing_layer(nodes_domain, Nodes_absorbing_Layers,Nspace=Nodes_device)  #damping potential
    plot_damping_potential(nodes_domain,Vdamp)

    #potential:
    Vdc_J=0*c.eVtoJ   #DC voltage aplied ->determines hill
    v_volatage=(-Vdc_J/(device_Length)*(x_coordinates-bL))*(np.heaviside(x_coordinates-bL,1))-(-Vdc_J/(device_Length)*(x_coordinates-bL)+Vdc_J)*np.heaviside((x_coordinates-bL)-(device_Length),1)  #potential due to voltage (hill)
    #plt.plot(x_coordinates,v_volatage,label="potential introduces by voltage")
    #plt.plot(x_coordinates,-np.ones_like(x_coordinates)*Vdc_J,label="voltage at end of device")
    #plt.legend()
    #plt.show()

    # potential barriers
    barrier_height_eV =0.6 #height of barrier in eV
    barrier_height_J=barrier_height_eV*c.eVtoJ  # Convert eV to Joules
    w12=5*10**(-9)   #width of both bariers
    a1=w12/2+bL #position midle of first barrier
    a2=bL+device_Length-w12/2 #position midle of seccond barrier
    print(f"position midle of seccond barrier: {a2}")

    vbarier1=barrier_height_J*np.heaviside(-np.abs(x_coordinates-a1)+w12/2,1) #barier 1
    vbarier2=barrier_height_J*np.heaviside(-np.abs(x_coordinates-a2)+w12/2,1)  #barier 2

    #potential of device
    vdevice=v_volatage+vbarier1+vbarier2+np.ones_like(x_coordinates)*Vdc_J   #total potential across barrier
    # plt.plot(x_coordinates,vdevice)
    # plt.title("vdevice")
    # plt.xlabel("position (m)")
    # plt.ylabel("potential energy (J)")
    # plt.show()

    V=vdevice

    #test barrier
    E=hbar**2*k0**2/(2*m) #averagle energy(J) (energy used for constructing test potential barrier)
    a=0.6*domain_length #position (midle of the) barier
    w=domain_length/20 #width barier

    #V=E*np.heaviside(-np.abs(x-a)+w/2,1) #the potential just to test

   

    # Load analytical data exported from main.py
    loaded_data = np.load('my_data.npz')

    # analytical transmission
    e = loaded_data['data_a']
    Trans = loaded_data['data_b']
    
    
    def solver(V):
        phi_initial, phi_initial_real, phi_initial_imag, C,sigma =initial_wave_packet(x_coordinates, domain_length, bL, aL, k0)

        delta_E = hbar**2 * k0 / (m * sigma) / c.eVtoJ
        print(f"energiebreedte golfpakket: {delta_E} eV")



        #check momentum content for debugginh
        phi=phi_initial_imag*1j+phi_initial_real
        f,PHI=fourier(phi,delta_x)
        k=2*np.pi*f
        E=k**2*c.hbar**2/(2*c.effective_mass)
        # plt.title("momentum content of initial wave")
        # plt.plot(k,np.abs(PHI),label="fft")
        klim=k0+3/(sigma*np.sqrt(2))
        Elim=klim**2*c.hbar**2/(2*c.effective_mass)
        # plt.axvline(x=klim)
        # plt.axvline(x=k0)
        # plt.legend()
        # plt.show()




        #plotting initial wave and potential
        # plt.plot(x_coordinates,V*(10**19)*C) #scaling for plotting reasons
        # plt.plot(x_coordinates,phi_initial)
        #plt.show()

        #time step
        print("max V:",np.max(vdevice))
        delta_t=2/((((2*hbar/m)*(1/(delta_x**2)))+(np.max(vdevice)/hbar))) #stabilaty condition
        time_steps=int(T/(delta_t))      #amount of time steps in time domain
        print("time_steps:",time_steps)
        t=np.arange(time_steps)*delta_t
        




        #initialise objects
        phi_real=np.zeros((time_steps,len(phi_initial_real)))
        phi_imag=np.zeros((time_steps,len(phi_initial_imag)))
        phi_real[0]=phi_initial_real
        phi_imag[0]=phi_initial_imag

        #integrate shroedinger equation (seccond order accurate laplacian)
        for i in range(time_steps-1):
            phi_real[i+1,1:-1]=(phi_real[i,1:-1]*(1+(delta_t/(2*hbar))*Vdamp[1:-1])-(hbar*delta_t/(2*m*delta_x**2))*(phi_imag[i,2:]+phi_imag[i,:-2]-2*phi_imag[i,1:-1])+V[1:-1]*delta_t/hbar*phi_imag[i,1:-1])/(1-(delta_t/(2*hbar))*Vdamp[1:-1])
            phi_imag[i+1,1:-1]=(phi_imag[i,1:-1]*(1+(delta_t/(2*hbar))*Vdamp[1:-1])+(hbar*delta_t/(2*m*delta_x**2))*(phi_real[i+1,2:]+phi_real[i+1,:-2]-2*phi_real[i+1,1:-1])-V[1:-1]*delta_t/hbar*phi_real[i+1,1:-1])/(1-(delta_t/(2*hbar))*Vdamp[1:-1])

        P=phi_real**2+phi_imag**2
        probability = np.trapezoid(
            phi_real[150,:]**2 + phi_imag[150,:]**2,
            x_coordinates
        )

        print(probability)

        print("point of observation,",bL+device_Length,domain_length+layer_distance)

        obs_idx = Nodes_absorbing_Layers+int((bL+device_Length+aL/2)/delta_x)

        #looking at the probabilaty at the edge of the domain
        phi_real1=phi_real[:,obs_idx]
        phi_real2=phi_real[:,obs_idx+1]
        phi_imag1=phi_imag[:,obs_idx]
        phi_imag2=phi_imag[:,obs_idx+1]

        psi1 = phi_real1+ 1j * phi_imag1
        psi2 = phi_real2+ 1j * phi_imag2

        f, PSI1 = fourier(psi1, delta_t)
        f, PSI2 = fourier(psi2, delta_t)

        
        # f,Phi_real1=fourier(phi_real1,delta_t)
        # f,Phi_real2=fourier(phi_real2,delta_t)
        # f,Phi_imag1=fourier(phi_imag1,delta_t)
        # f,Phi_imag2=fourier(phi_imag2,delta_t)
        P_energy_domain = np.abs(PSI1)**2
        E= 2*np.pi*f*hbar
        J = hbar * np.real(-1.j *np.conj(PSI1) * (PSI2 - PSI1) / delta_x) / m
        return E,J,t,P,C, P_energy_domain, sigma


    E1,J_free,t,P_free,C_free, P_free_E,sigma=solver(0*x_coordinates)
    E2,J_bar,t,P_bar,C_bar, P_bar_E,sigma=solver(vdevice)
    

    # animating simulation:
    frames_data = P_bar[::30]
    delta_t=2/(((2*hbar/m)*(1/(delta_x**2)))+(np.max(vdevice)/hbar)) #stabilaty condition
    dt_simulation = delta_t * 30*1e15  # tijdstap per frame (aangezien je elke 30e frame neemt)

    # Bereken het observatiepunt
    obs_index = Nodes_absorbing_Layers + int((bL + device_Length + aL/2)/delta_x)
    obs_position = x_coordinates[obs_index]

    fig, ax = plt.subplots()

    ax.set_xlim(-layer_distance, domain_length + layer_distance)
    ax.set_ylim(0, np.max(frames_data) * 1.1)

    # Wave packet lijn
    line, = ax.plot([], [], lw=2, color='firebrick', label='Wave packet')

    # Observatiepunt verticale lijn
    obs_line = ax.axvline(x=obs_position, color='green', linestyle='--', 
                        linewidth=2, alpha=0.8, label='Observation point')

    # titel
    title = ax.set_title('')

    # ===== potential/barriers toevoegen =====

    # schaal de potentiaal zodat hij zichtbaar is
    V_scaled = vdevice * (
        np.max(frames_data[0]) /
        (np.max(np.abs(vdevice)) + 1e-30)
    ) * 0.3

    # plot van de barriers/potentiaal
    potential_line, = ax.plot(
        x_coordinates,
        V_scaled,
        'b--',
        lw=1.5,
        alpha=0.7,
        label='Potential'
    )

    # Tijd tekst
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()

    # =======================================

    def init_animation():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        # update wave packet
        line.set_data(x_coordinates, frames_data[frame])
        
        # update tijd
        current_time = frame * dt_simulation
        time_text.set_text(f'Tijd: {current_time:.4f} s')
        
        # update titel (optioneel)
        title.set_text(f'Golfpakket propagatie - Frame {frame+1}/{np.shape(frames_data)[0]}')
        
        return line, time_text, title

    ani = FuncAnimation(
        fig,
        update,
        frames=np.shape(frames_data)[0],
        init_func=init_animation,
        blit=False,
        interval=0.5
    )
    plt.show()




    #processing and validating
    plt.title("wave function right after the seccond barrier")
    plt.xlabel("time (s)")
    plt.ylabel("wave function sqrt(1/m)")
    plt.plot(t,P_free[:,Nodes_absorbing_Layers+int((bL+device_Length+aL/2)/delta_x)],label="no barrier")
    plt.plot(t,P_bar[:,Nodes_absorbing_Layers+int((bL+device_Length+aL/2)/delta_x)],label="with barrier")
    plt.legend()
    plt.show()

    lim=10**(-18)
    plt.title("wave functions in energy domain")
    plt.xlabel("Energy (eV)")
    plt.ylabel("wave function sqrt(1/m)")
    plt.plot(E1[(E1>0) & (E1<lim)]/c.eVtoJ,np.abs(J_free[(E1>0) & (E1<lim)]),label="no barrier")
    # Voeg verticale lijn toe bij 0.9 eV
    plt.axvline(x=Energy_current_eV, color='red', linestyle='--', linewidth=2, 
            label=f'Energy = {Energy_current_eV} eV')
    plt.plot(E2[(E2>0) & (E2<lim)]/c.eVtoJ,np.abs(J_bar[(E2>0) & (E2<lim)]),label="with barrier")
    plt.legend()
    plt.show()

    E_center = k0**2 * c.hbar**2 / (2 * c.effective_mass)/ c.eVtoJ
    E_width = 3 * (1/(2*sigma)) * c.hbar**2 * k0 / c.effective_mass / c.eVtoJ # Geschatte energie spreiding

    plt.title("transmission")
    plt.xlabel("Energy(eV)")
    plt.ylabel("Transmission")
    plt.plot(E1[(E1>-lim) & (E1<lim)]/c.eVtoJ, J_bar[(E1>-lim) & (E1<lim)]/J_free[(E1>-lim) & (E1<lim)], label="numerical")
    plt.plot(e, Trans, label="analytical")
    plt.ylim(0, 1)
    plt.xlim(E_center-E_width, E_center+E_width)  # Beperk x-as van 0 tot 2 eV
    plt.legend()
    plt.show()




    


    
    """
    # 1. Your existing data
    x_values = t
    y_values = x_coordinates   # 1D array (Length 10)
    z_data = np.transpose(P)   # 2D array (Shape 10x20)

    # 2. Plotting
    plt.figure(figsize=(8, 4))
    # Use pcolormesh(X, Y, Z)
    plt.pcolormesh(x_values, y_values, z_data, shading='auto', cmap='viridis')

    plt.colorbar(label='Intensity')
    plt.xlabel('time')
    plt.ylabel('space')
    plt.show()

    """