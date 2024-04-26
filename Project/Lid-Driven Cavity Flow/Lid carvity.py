import numpy as np
import math as m
import matplotlib.pyplot as plt

sim_velocity = True
sim_f = False
sim_density = False

#Total Time
t = 50000
#Time per frame
tpf = 100

ex = np.array([0, 1, 0, -1, 0 , 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
len_x, len_y = 256, 256
#Time step: dt
dt = 1
#Fluid Kinematic Viscosity: v 
v = 1/18
#Relaxation time: T
T = 3*v*dt+0.5
#Lattice Speed: c = m.sqrt((6.0*v*dt)/(2.0*T-1))
c = 1
#Velocity of the lid
u0 = 0.08681

def initialize():
    global p, u_hat, f, f_eq
    u_hat = np.zeros((len_x, len_y, 2))
    f = np.ones((len_x, len_y, 9))
    f[:,:] *= w
    f_eq = np.zeros((len_x, len_y, 9))
    

def streaming():
    global p, u_hat, f, f_eq

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

    #Boundary Condition(The wall) (On-grid bounce-back)
    #Left Wall
    f[0,:,[5,1,8]] = f[0,:,[6,3,7]]
    #Right Wall
    f[-1,0:-1,[6,3,7]] = f[-1,0:-1,[8,1,5]]
    #Bottom Wall
    f[0:-1,0,[6,2,5]] = f[0:-1,0,[8,4,7]]

    #Zou-He boundary condition(Top Wall)
    p0 = (f[:,-1,0] + f[:,-1,1] + f[:,-1,3] + 2*(f[:,-1,5]+f[:,-1,2]+f[:,-1,6]))
    f[:,-1,4] = f[:,-1,2]
    f[:,-1,8] = f[:,-1,6] + (1.0/2)*(f[:,-1,3]-f[:,-1,1]) + (1./2)*(p0*u0)
    f[:,-1,7] = f[:,-1,5] - (1.0/2)*(f[:,-1,3]-f[:,-1,1]) - (1./2)*(p0*u0)

    
    p = np.sum(f,2)
    u_hat[:,:,0] = np.sum(f * ex,2)*c/p
    u_hat[:,:,1] = np.sum(f * ey,2)*c/p

    u_hat[:,-1,0] = u0
    u_hat[:,-1,1] = 0
    
    f_eq = np.zeros((len_x, len_y, 9))
    for i, cx, cy, W in zip(range(9), ex, ey, w):
        #Calculate f_eq
        f_eq[:,:,i] = p*W*((1.0+((3.0/c)*(cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))+4.5*((cx*u_hat[:,:,0]+cy*u_hat[:,:,1])/c)**2 -1.5*(u_hat[:,:,0]**2 + u_hat[:,:,1]**2)/(c**2)))

def collision():
    global p, u_hat, f, f_eq
    #Collision
    f += -(f-f_eq)/T
    
def main():
    global p, u_hat, f, f_eq
    initialize()

    velocity_output = np.zeros((len_y,len_x, 2 ,t // tpf))
    
    for count in range(t):
        print(count)
        streaming()
        collision()
        if count % tpf == 0:
            velocity_output[:,:,:,count // tpf] = u_hat
            
                
    np.save('Lid-driven Carvity flow Re400 256 256',velocity_output)
    
if __name__ == "__main__":
    main()
