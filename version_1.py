import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
#to do :
#-add absorbing layers
#-4th order sceme
#add the required potetial
#calculate probabilaty current and fft
#calculate and validate transmission
#calculate I-V
#extra experiments





#everything in SI units
#physical info
m=0.023*9.109*10**(-31)
h=1.054*10**(-34)



#domain
L=25*10**(-9) #meters (length of device)
v=10**7    #m/s  (arbitrary group velocity)
T=10*L/v   #s  (time based on velocity and dimension of space)

#parameters
k0=m*v/h  #wave vector average for wave packet (based on electron momentum)
E=h**2*k0**2/(2*m) #averagle energy(J) (energy used for constructing potential barrier)
la=(2*np.pi)/k0    #average wavelenth (1/m?)     
print(la)
delta=la/50        #discretisation step (m) (based on average wave length ->prob better to use max wave length + less small needed when going to 4th order)
print(delta)
N=int(L/delta)     #amount of space points
print(N)
x=np.arange(0,N+1)*delta   # space values(m)


#potential:
a=0.6*L #position (midle of the) barier
w=L/20 #width barier
V=E*np.heaviside(-np.abs(x-a)+w/2,1) #the potential (still need to add absorbing layer)
#V=x*0 #for free particle simulation


#initial wave packet:
x0=L/3  #center of wave packet (->keep values ad edges zero)
sigma=L/20  #deviation from center (->keep values ad edges zero)
phi_initial=np.exp(-(x-x0)**2/(4*sigma**2)) #wave shape

#normelising wave shape
norm=np.trapezoid(phi_initial)
C=1/norm
phi_initial=C*phi_initial

#splitting wave packet in real and imaginary parts
phi_initial_real=phi_initial*np.cos(k0*x)
phi_initial_imag=phi_initial*np.sin(k0*x)

#plotting initial wave and potential
plt.plot(x,V*(10**19)*C) #scaling for plotting reasons
plt.plot(x,phi_initial)
plt.show()

#time step
print("max V:",np.max(V))
delta_t=2/(((2*h/m)*(1/(delta**2)))+(np.max(V)/h)) #stabilaty condition
time_steps=int(T/(5*delta_t))      #amount of time steps in time domain
print("time")
print(T)
print(time_steps)





#initialise objects
phi_real=np.zeros((time_steps,len(phi_initial_real)))
phi_imag=np.zeros((time_steps,len(phi_initial_imag)))
phi_real[0]=phi_initial_real
phi_imag[0]=phi_initial_imag

#integrate shroedinger equation (seccond order accurate laplacian)
for i in range(time_steps-1):
    phi_real[i+1,1:-1]=phi_real[i,1:-1]-(h*delta_t/(2*m*delta**2))*(phi_imag[i,2:]+phi_imag[i,:-2]-2*phi_imag[i,1:-1])+V[1:-1]*delta_t/h*phi_imag[i,1:-1]
    phi_imag[i+1,1:-1]=phi_imag[i,1:-1]+(h*delta_t/(2*m*delta**2))*(phi_real[i+1,2:]+phi_real[i+1,:-2]-2*phi_real[i+1,1:-1])-V[1:-1]*delta_t/h*phi_real[i+1,1:-1]

#animating simulation:

frames_data=np.sqrt(phi_real**2+phi_imag**2)

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, np.max(frames_data))

line, = ax.plot([], [], lw=2, color='firebrick')

# 1. Create a title object (empty for now)
title = ax.set_title('')

def init():
    line.set_data([], [])
    # Clear the title on init
    title.set_text('')
    return line, title

def update(frame):
    y = frames_data[frame]
    line.set_data(x, y)
    
    # 2. Update the title text
    # You can use f-strings to format it nicely
    title.set_text(f'Time Step: {frame}')
    
    # 3. CRITICAL: Return the title in the tuple for blitting
    return line, title

ani = FuncAnimation(fig, update, frames=np.shape(frames_data)[0], 
                    init_func=init, blit=False, interval=10)

plt.show()