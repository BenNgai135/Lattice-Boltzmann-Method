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
t = 10000
#Time per frame
tpf = 100
#Average density
pavg = 1
#Inlet speed_x
uin =   1/6
#Outlet density
#pout = 0.00556

ex = np.array([0, 1, 0, -1, 0 , 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
len_x, len_y = 400, 100
#Time step: dt
dt = 1
#Fluid Kinematic Viscosity: v
v = 1/18
#Relaxation time: T
T = 3*v*dt+0.5
#Lattice Speed: c = m.sqrt((6.0*v*dt)/(2.0*T-1))
c = 1


def BC(x,y):
    a = (x - len_x/10)**2 + (y - len_y/2)**2 < (len_y/10)**2
    return a



def initialize():
    global p, u_hat, f, f_eq, boundary,bndryF
    u_hat = np.zeros((len_x, len_y, 2))
    f = np.ones((len_x, len_y, 9))
    f[:,:] *= w
    f_eq = np.zeros((len_x, len_y, 9))
    
    #Initial Conditions
    X,Y = np.meshgrid(range(len_x), range(len_y),indexing='ij')
    p = np.sum(f,2)
    
    X,Y = np.meshgrid(range(len_x), range(len_y),indexing='ij')
    boundary = BC(X,Y)
    

def streaming():
    global p, u_hat, f, f_eq,boundary,bndryF

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
    p0 = (1./(1-uin))*((f[0,:,0]+f[0,:,2]+f[0,:,4]+2*(f[0,:,3]+f[0,:,6]+f[0,:,7])))
    f[0,:,1] = f[0,:,3] + (2./3)*p0*uin#*******
    f[0,:,5] = f[0,:,7] - (1.0/2)*(f[0,:,2] - f[0,:,4]) + (1.0/6)*p0*uin
    f[0,:,8] = f[0,:,6] + (1.0/2)*(f[0,:,2] - f[0,:,4]) + (1.0/6)*p0*uin
    
    #u = ((1./pout)*(f[-1,:,0]+f[-1,:,2]+f[-1,:,4]+2.0*(f[-1,:,1]+f[-1,:,5]+f[-1,:,8])))-1.0   
    #Open boundary
    f[-1,:,3] = f[-2,:,3]
    f[-1,:,7] = f[-2,:,7]
    f[-1,:,6] = f[-2,:,6]

    p = np.sum(f,2)
    u_hat[:,:,0] = np.sum(f * ex,2)*c/p
    u_hat[:,:,1] = np.sum(f * ey,2)*c/p
    
    f_eq = np.zeros((len_x, len_y, 9))
    for i, cx, cy, W in zip(range(9), ex, ey, w):
        #Calculate f_eq
        f_eq[:,:,i] = p*W*((1.0+((3.0/c)*(cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))+4.5*((cx*u_hat[:,:,0]+cy*u_hat[:,:,1])/c)**2 -1.5*(u_hat[:,:,0]**2 + u_hat[:,:,1]**2)/(c**2)))

def collision():
    global p, u_hat, f, f_eq, boundary, bndryF
    #Boundary Condition(The cylinder) (On-grid bounce-back)
    bndryF = f[boundary,:]
    bndryF = bndryF[:,[0,3,4,1,2,7,8,5,6]]

    #Collision
    f += -(f-f_eq)/T
    
    #Boundary Condition(The cylinder) (On-grid bounce-back)
    f[boundary,:] = bndryF
                    
                    

def main():
    global p, u_hat, f, f_eq, boundary, bndryF
    initialize()
    
    velocity_output = np.zeros((len_x,len_y, 2 ,t // tpf))
    
    for count in range(t):
        print(count)
        streaming()
        collision()
        if count % tpf == 0:
            u_hat[boundary,:] = np.nan
            velocity_output[:,:,:,count // tpf] = u_hat
                
    np.save('FTC_Re150',velocity_output)
    
    
if __name__ == "__main__":
    try:
        main()
    except RuntimeWarning:
        print("Overflow encountered!")
