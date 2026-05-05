import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
#to do :
#-tweak absorbing layers
#-4th order sceme
#calculate probabilaty current and fft
#calculate and validate transmission
#calculate I-V
#extra experiments





#everything in SI units
#physical info
m=0.023*9.109*10**(-31) #effective mass of the electon
h=1.054*10**(-34)   #reduced planck constant
eV=1.60217663 * 10**(-19) #one electron volt in joules/ charge of an electron


#domain
device_L=25*10**(-9) #meters (length of device)
bL=15*10**(-9)      #length domain before device
aL=40*10**(-9)      #length domain after device
L=device_L+aL+bL    #length of domain (without absorbing layers)
print(L)
v=10**7    #m/s  (arbitrary group velocity)
T=L/v   #s  (time based on velocity and dimension of space)
print(T)

#parameters
k0=m*v/h  #wave vector average for wave packet (based on electron momentum)
la=(2*np.pi)/k0    #average wavelenth (1/m?)     
delta=la/50        #discretisation step (m) (based on average wave length ->prob better to use max wave length + less small needed when going to 4th order)
N=int(L/delta)     #amount of space points
print(N)
Nlayer=3*75+20     #amount of points in absorbing layer
layer_distance=Nlayer*delta  #distance corresponding to layer
print("layer distance:",layer_distance)
n=np.arange(-Nlayer,N+Nlayer+1)  #the points in domain including absorbing layer
x=n*delta  # space values(m)     

#absorbing potential layer

S=-0.07*1.3267185763347041e-16  #the sigma chosen negative beceuase otherwise instabilaty still needs tuning
deg=7   #degree of polynomial
Vdamp=np.heaviside(-n,1)*S*((-n)/Nlayer)**deg+np.heaviside(n-N,1)*S*((n-N)/Nlayer)**deg  #damping potential
#Vdamp=n*0         #line for turning off damping
plt.plot(n,Vdamp)
plt.title("the damping potential")
plt.xlabel("index point in domain")
plt.ylabel("absorbing potential")
plt.show()


#potential:
Vdc=0*eV   #DC voltage aplied ->determines hill
v_volatage=(-Vdc/(L-aL-bL)*(x-bL))*(np.heaviside(x-bL,1))-(-Vdc/(L-aL-bL)*(x-bL)+Vdc)*np.heaviside((x-bL)-(L-aL-bL),1)  #potential due to voltage (hill)
plt.plot(x,v_volatage,label="potential introduces by voltage")
plt.plot(x,-np.ones_like(x)*Vdc,label="voltage at end of device")
plt.legend()
plt.show()

# potential barriers
barrier_height=0.6*eV 
w12=5*10**(-9)   #width of both bariers
a1=w12/2+bL #position midle of first barrier
a2=bL+device_L-w12/2 #position midle of seccond barrier
print("a2:",a2)

vbarier1=barrier_height*np.heaviside(-np.abs(x-a1)+w12/2,1) #barier 1
vbarier2=barrier_height*np.heaviside(-np.abs(x-a2)+w12/2,1)  #barier 2

#potential of device
vdevice=v_volatage+vbarier1+vbarier2+np.ones_like(x)*Vdc   #total potential across barrier
plt.plot(x,vdevice)
plt.show()

V=vdevice

#test barrier
E=h**2*k0**2/(2*m) #averagle energy(J) (energy used for constructing test potential barrier)
a=0.6*L #position (midle of the) barier
w=L/20 #width barier
#V=E*np.heaviside(-np.abs(x-a)+w/2,1) #the potential just to test

#free particle
#V=x*0 #for free particle simulation




#initial wave packet shape:
x0=5.5*10**(-9)  #center of wave packet (->keep values at edges zero)
sigma=(L-bL-aL)/30  #deviation from center (->keep values at edges zero)
phi_initial=np.exp(-(x-x0)**2/(4*sigma**2)) #wave shape

#normelising wave shape
norm=np.trapezoid(phi_initial)
C=1/norm
phi_initial=C*phi_initial

#splitting initial wave packet in real and imaginary parts
phi_initial_real=phi_initial*np.cos(k0*x)
phi_initial_imag=phi_initial*np.sin(k0*x)

#plotting initial wave and potential
plt.plot(x,V*(10**19)*C) #scaling for plotting reasons
plt.plot(x,phi_initial)
plt.show()

#time step
print("max V:",np.max(V))
delta_t=2/(((2*h/m)*(1/(delta**2)))+(np.max(V)/h)) #stabilaty condition
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
    phi_real[i+1,1:-1]=(phi_real[i,1:-1]*(1+(delta_t/(2*h))*Vdamp[1:-1])-(h*delta_t/(2*m*delta**2))*(phi_imag[i,2:]+phi_imag[i,:-2]-2*phi_imag[i,1:-1])+V[1:-1]*delta_t/h*phi_imag[i,1:-1])/(1-(delta_t/(2*h))*Vdamp[1:-1])
    phi_imag[i+1,1:-1]=(phi_imag[i,1:-1]*(1+(delta_t/(2*h))*Vdamp[1:-1])+(h*delta_t/(2*m*delta**2))*(phi_real[i+1,2:]+phi_real[i+1,:-2]-2*phi_real[i+1,1:-1])-V[1:-1]*delta_t/h*phi_real[i+1,1:-1])/(1-(delta_t/(2*h))*Vdamp[1:-1])

P=np.sqrt(phi_real**2+phi_imag**2)
#looking at the probabilaty at the edge of the domain

#checking reflection
print("percentage refelction:",np.max((P[-1,x>0])/np.max(P[0,x>0]))*100,"%")


#animating simulation:
frames_data=P
fig, ax = plt.subplots()
ax.set_xlim(-layer_distance, L+layer_distance)
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
    title.set_text(f'Time: {frame*delta_t}')
    
    # 3. CRITICAL: Return the title in the tuple for blitting
    return line, title

ani = FuncAnimation(fig, update, frames=np.shape(frames_data)[0], 
                    init_func=init, blit=False, interval=0.1)
plt.show()






# 1. Your existing data
x_values = t
y_values = x   # 1D array (Length 10)
z_data = np.transpose(P)   # 2D array (Shape 10x20)

# 2. Plotting
plt.figure(figsize=(8, 4))
# Use pcolormesh(X, Y, Z)
plt.pcolormesh(x_values, y_values, z_data, shading='auto', cmap='viridis')

plt.colorbar(label='Intensity')
plt.xlabel('time')
plt.ylabel('space')
plt.show()


