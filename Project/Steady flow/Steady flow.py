#First-order interpolation boundary condition at the outlet
#Flow though cylinder
#Vortex streets at uin > 1/6, 100*400, Viscosity = 1/18
import numpy as np
import math as m
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('error', category=RuntimeWarning)

sim_f = False
sim_velocity = True
sim_density = True

#Total Time
t = 30000
#Time per frame
tpf = 10
#Average density
pavg = 1
#Inlet speed_x
pin = 1.025
#Outlet density
pout = 0.975

ex = np.array([0, 1, 0, -1, 0 , 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
len_x, len_y = 40, 32
#Time step: dt
dt = 1
#Fluid Kinematic Viscosity: v
v = 1/18
#Relaxation time: T
T = 1
#Lattice Speed: c = m.sqrt((6.0*v*dt)/(2.0*T-1))
c = 1

def initialize():
    global p, u_hat, f, f_eq,bndryF
    u_hat = np.zeros((len_x, len_y, 2))
    f = np.ones((len_x, len_y, 9))
    f[:,:] *= w
    f_eq = np.zeros((len_x, len_y, 9))
    
    #Initial Conditions
    p = np.sum(f,2)
    

def streaming():
    global p, u_hat, f, f_eq,bndryF

    #Streaming
    f[:,:,1] = np.roll(f[:,:,1],1,axis = 0)
    f[:,:,2] = np.roll(f[:,:,2],1,axis = 1)
    f[:,:,3] = np.roll(f[:,:,3],-1,axis = 0)
    f[:,:,4] = np.roll(f[:,:,4],-1,axis = 1)
    f[:,:,5] = np.roll(f[:,:,5],1,axis = 0)
    f[:,:,5] = np.roll(f[:,:,5],1,axis = 1)
    
    f[:,:,6] = np.roll(f[:,:,6],-1,axis = 0)
    f[:,:,6] = np.roll(f[:,:,6],1,axis = 1)
                       
    f[:,:,7] = np.roll(f[:,:,7],-1,axis = 0)
    f[:,:,7] = np.roll(f[:,:,7],-1,axis = 1)
    
    f[:,:,8] = np.roll(f[:,:,8],1,axis = 0)
    f[:,:,8] = np.roll(f[:,:,8],-1,axis = 1)

    #Top Wall(Bounce Back)
    f[:,-1,[8,4,7]] = f[:,-1,[6,2,5]]

    #Bottom Wall(Bounce Back)
    f[:,0,[6,2,5]] = f[:,0,[8,4,7]]

    #Zou-He boundary condition***
    uin = 1.0 -((1./pin)*(f[0,:,0]+f[0,:,2]+f[0,:,4]+2.0*(f[0,:,3]+f[0,:,6]+f[0,:,7])))
    f[0,:,1] = f[0,:,3] + (2./3)*pin*uin#*******
    f[0,:,5] = f[0,:,7] - (1.0/2)*(f[0,:,2] - f[0,:,4]) + (1.0/6)*pin*uin
    f[0,:,8] = f[0,:,6] + (1.0/2)*(f[0,:,2] - f[0,:,4]) + (1.0/6)*pin*uin
    
    u = ((1./pout)*(f[-1,:,0]+f[-1,:,2]+f[-1,:,4]+2.0*(f[-1,:,1]+f[-1,:,5]+f[-1,:,8]))) - 1.0
    f[-1,:,3] = f[-1,:,1] - (2./3)*pout*u#*******
    f[-1,:,7] = f[-1,:,5] + (1.0/2)*(f[-1,:,2] - f[-1,:,4]) - (1.0/6)*pout*u
    f[-1,:,6] = f[-1,:,8] - (1.0/2)*(f[-1,:,2] - f[-1,:,4]) - (1.0/6)*pout*u

    p = np.sum(f,2)
    u_hat[:,:,0] = np.sum(f * ex,2)*c/p
    u_hat[:,:,1] = np.sum(f * ey,2)*c/p
    
    f_eq = np.zeros((len_x, len_y, 9))
    for i, cx, cy, W in zip(range(9), ex, ey, w):
        #Calculate f_eq
        f_eq[:,:,i] = p*W*((1.0+((3.0/c)*(cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))+4.5*((cx*u_hat[:,:,0]+cy*u_hat[:,:,1])/c)**2 -1.5*(u_hat[:,:,0]**2 + u_hat[:,:,1]**2)/(c**2)))

def collision():
    global p, u_hat, f, f_eq, bndryF
    #Boundary Condition(The cylinder) (On-grid bounce-back)
    #Collision
    f += -(f-f_eq)/T            

def main():
    global p, u_hat, f, f_eq, bndryF
    initialize()
    
    velocity_output = np.zeros((len_x,len_y, 2 ,t // tpf))
    
    for count in range(t):
        print(count)
        streaming()
        collision()
        if count % tpf == 0:
            velocity_output[:,:,:,count // tpf] = u_hat
                
    if sim_velocity == True:
        np.save('SF_Re150',velocity_output)
    
    
if __name__ == "__main__":
    try:
        main()
    except RuntimeWarning:
        print("Overflow encountered!")
