from os import name
from re import M

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import plot
#to do :
#-tweak absorbing layers
#-4th order sceme
#calculate probabilaty current and fft
#calculate and validate transmission
#calculate I-V
#extra experiments


#everything in SI units

def propagate_matrix(zone):
    matrix= np.array([[np.exp(1j*zone.wavevector*zone.zone_width),                                          0],
                      [0                                        ,np.exp(-1j*zone.wavevector*zone.zone_width)]
                     ]
                      )
    
    return matrix

def interface_matrix(zoneleft,zoneright):
    kleft = zoneleft.wavevector
    kright = zoneright.wavevector
    diagonalterm = 1 /2 * (1 + kright/kleft)
    offdiagonalterm = 1 /2 * (1 - kright/kleft)
    matrix = np.array([[diagonalterm, offdiagonalterm],
                      [offdiagonalterm, diagonalterm]
                     ]
                      )
    return matrix

def validate_transmission(zones):
    Matrices = np.eye(2)  # Start with the identity matrix
    for i in range(len(zones)-1):
        if zones[i].wavevector != zones[i+1].wavevector:
            print("warning: wave vector changes at interface between zone",i,"and",i+1)
        Matrices = Matrices @ interface_matrix(zones[i], zones[i+1]) @ propagate_matrix(zones[i+1])    

    T = 1 / np.abs(Matrices[0, 0])**2
    return T    

def init_absorbing_layer(n,Nlayer):
    S=-0.07*1.3267185763347041e-16  #the sigma chosen negative beceuase otherwise instabilaty still needs tuning
    deg=7   #degree of polynomial
    Vdamp=np.heaviside(-n,1)*S*((-n)/Nlayer)**deg+np.heaviside(n-N,1)*S*((n-N)/Nlayer)**deg  #damping potential
    return Vdamp

def initial_wave_packet(x_coordinates, L, bL, aL, k0):

        #initial wave packet shape:
        x0=5.5*10**(-9)  #center of wave packet (->keep values at edges zero)
        sigma=(L-bL-aL)/30  #deviation from center (->keep values at edges zero)
        phi_initial=np.exp(-(x_coordinates-x0)**2/(4*sigma**2)) #wave shape

        #normelising wave shape
        norm=np.trapezoid(phi_initial)
        C=1/norm
        phi_initial=C*phi_initial

        #splitting initial wave packet in real and imaginary parts
        phi_initial_real=phi_initial*np.cos(k0*x_coordinates)
        phi_initial_imag=phi_initial*np.sin(k0*x_coordinates)

        return phi_initial, phi_initial_real, phi_initial_imag, C
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
    title.set_text(f'Time: {frame*delta_t}')
    
    # 3. CRITICAL: Return the title in the tuple for blitting
    return line, title

def plot_damping_potential(n,Vdamp):
    plt.plot(n,Vdamp)
    plt.title("the damping potential")
    plt.xlabel("index point in domain")
    plt.ylabel("absorbing potential")
    plt.show()

def init_potential_barrier(x, a, w, E):
    return E * np.heaviside(-np.abs(x - a) + w / 2, 1)

class zone:
    def __init__(self,zone_start,zone_end,k0):
        self.zone_start=zone_start
        self.zone_end=zone_end
        self.wavevector=k0
        self.zone_width =zone_end-zone_start


class constants:
    def __init__(self):
        self.effective_mass=0.023*9.109*10**(-31) #effective mass of the electon
        self.hbar=1.054*10**(-34)   
        self.eVtoJ=1.60217663 * 10**(-19) #one electron volt in joules/ charge of an electron



if __name__=="__main__":
    
    c=constants()
    hbar = c.hbar
    m = c.effective_mass
    #domain
    device_L=25*10**(-9) #meters (length of device)
    bL=15*10**(-9)      #length domain before device
    aL=40*10**(-9)      #length domain after device
    L=device_L+aL+bL    #length of domain (without absorbing layers)
   
    v=10**7    #m/s  (arbitrary group velocity)
    T=L/v   #s  (time based on velocity and dimension of space)
    print(f"total length of domain: {L*1e9} nanometers")
    print(f"total simulation time: {T*1e15} femtoseconds")

    #parameters
    k0=m*v/hbar  #wave vector average for wave packet (based on electron momentum)
    la=(2*np.pi)/k0    #average wavelenth (1/m?)     
    delta=la/50        #discretisation step (m) (based on average wave length ->prob better to use max wave length + less small needed when going to 4th order)
    N=int(L/delta)     #amount of space points
    print(N)
    Nlayer=3*75+20     #amount of points in absorbing layer
    layer_distance=Nlayer*delta  #distance corresponding to layer
    print(f"layer distance: {layer_distance*1e9} nanometers")
    n=np.arange(-Nlayer,N+Nlayer+1)  #the points in domain including absorbing layer
    x_coordinates=n*delta  # space values(m)


    barrier_height_eV =0.6 #height of barrier in eV
    barrier_height_J=barrier_height_eV*c.eVtoJ  # Convert eV to Joules
    barrier_width=5*10**(-9)   #width of both bariers
    a1=barrier_width/2+bL #position midle of first barrier
    a2=bL+device_L-barrier_width/2 #position midle of seccond barrier

    thicknesbeforedevice = bL
    thicknessbarrier1 = barrier_width
    thicknessbetweenbarriers= device_L - 2*barrier_width
    thicknessbarrier2 = barrier_width
    thicknessafterdevice = aL
    thicknesses = [thicknesbeforedevice, thicknessbarrier1, thicknessbetweenbarriers, thicknessbarrier2, thicknessafterdevice]

    zones= []
    for i in range(len(thicknesses)):
        print(f"zone {i} thickness: {thicknesses[i]*1e9} nanometers")
        zone_i = zone(sum(thicknesses[:i]), sum(thicknesses[:i+1]), k0)  #zone before device
        zones.append(zone_i)

    print(zones[0].zone_start*1e9, zones[0].zone_end*1e9)

    print(validate_transmission(zones)) # validatie transmission using transfer matrix method
    Vdamp=init_absorbing_layer(n, Nlayer)  #damping potential

    plot_damping_potential(n,Vdamp)

    #potential:
    Vdc_J=0*c.eVtoJ   #DC voltage aplied ->determines hill
    v_volatage=(-Vdc_J/(L-aL-bL)*(x_coordinates-bL))*(np.heaviside(x_coordinates-bL,1))-(-Vdc_J/(L-aL-bL)*(x_coordinates-bL)+Vdc_J)*np.heaviside((x_coordinates-bL)-(L-aL-bL),1)  #potential due to voltage (hill)
    plt.plot(x_coordinates,v_volatage,label="potential introduces by voltage")
    plt.plot(x_coordinates,-np.ones_like(x_coordinates)*Vdc_J,label="voltage at end of device")
    plt.legend()
    plt.show()

    # potential barriers
    barrier_height_eV =0.6 #height of barrier in eV
    barrier_height_J=barrier_height_eV*c.eVtoJ  # Convert eV to Joules
    w12=5*10**(-9)   #width of both bariers
    a1=w12/2+bL #position midle of first barrier
    a2=bL+device_L-w12/2 #position midle of seccond barrier
    print(f"position midle of seccond barrier: {a2}")

    vbarier1=barrier_height_J*np.heaviside(-np.abs(x_coordinates-a1)+w12/2,1) #barier 1
    vbarier2=barrier_height_J*np.heaviside(-np.abs(x_coordinates-a2)+w12/2,1)  #barier 2

    #potential of device
    vdevice=v_volatage+vbarier1+vbarier2+np.ones_like(x_coordinates)*Vdc_J   #total potential across barrier
    plt.plot(x_coordinates,vdevice)
    plt.show()

    V=vdevice

    #test barrier
    E=hbar**2*k0**2/(2*m) #averagle energy(J) (energy used for constructing test potential barrier)
    a=0.6*L #position (midle of the) barier
    w=L/20 #width barier
    #V=E*np.heaviside(-np.abs(x-a)+w/2,1) #the potential just to test

    #free particle
    #V=x*0 #for free particle simulation

    
    phi_initial, phi_initial_real, phi_initial_imag, C =initial_wave_packet(x_coordinates, L, bL, aL, k0)

    #plotting initial wave and potential
    plt.plot(x_coordinates,V*(10**19)*C) #scaling for plotting reasons
    plt.plot(x_coordinates,phi_initial)
    plt.show()

    #time step
    print("max V:",np.max(V))
    delta_t=2/(((2*hbar/m)*(1/(delta**2)))+(np.max(V)/hbar)) #stabilaty condition
    time_steps=int(T/(delta_t))      #amount of time steps in time domain
    print(time_steps)
    t=np.arange(time_steps)*delta_t



    #initialise objects
    phi_real=np.zeros((time_steps,len(phi_initial_real)))
    phi_imag=np.zeros((time_steps,len(phi_initial_imag)))
    phi_real[0]=phi_initial_real
    phi_imag[0]=phi_initial_imag

    #integrate shroedinger equation (seccond order accurate laplacian)
    for i in range(time_steps-1):
        phi_real[i+1,1:-1]=(phi_real[i,1:-1]*(1+(delta_t/(2*hbar))*Vdamp[1:-1])-(hbar*delta_t/(2*m*delta**2))*(phi_imag[i,2:]+phi_imag[i,:-2]-2*phi_imag[i,1:-1])+V[1:-1]*delta_t/hbar*phi_imag[i,1:-1])/(1-(delta_t/(2*hbar))*Vdamp[1:-1])
        phi_imag[i+1,1:-1]=(phi_imag[i,1:-1]*(1+(delta_t/(2*hbar))*Vdamp[1:-1])+(hbar*delta_t/(2*m*delta**2))*(phi_real[i+1,2:]+phi_real[i+1,:-2]-2*phi_real[i+1,1:-1])-V[1:-1]*delta_t/hbar*phi_real[i+1,1:-1])/(1-(delta_t/(2*hbar))*Vdamp[1:-1])

    P=np.sqrt(phi_real**2+phi_imag**2)
    #looking at the probabilaty at the edge of the domain

    #checking reflection
    print("percentage reflection:",np.max((P[-1,x_coordinates>0])/np.max(P[0,x_coordinates>0]))*100,"%")


    #animating simulation:
    frames_data=P
    fig, ax = plt.subplots()
    ax.set_xlim(-layer_distance, L+layer_distance)
    ax.set_ylim(0, np.max(frames_data))
    line, = ax.plot([], [], lw=2, color='firebrick')
    # 1. Create a title object (empty for now)
    title = ax.set_title('')



    ani = FuncAnimation(fig, update, frames=np.shape(frames_data)[0], 
                        init_func=init_animation, blit=False, interval=0.5)
    plt.show()

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
