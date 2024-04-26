#Rayleigh Benard convection
import numpy as np
import math as m
import matplotlib.pyplot as plt

#Total Time
time = 30000
#Time per frame
tpf = 10
#Average density
pavg = 1

#lattice size
len_x, len_y = 200, 50

#D2Q9 model for g
ex_d2q9 = np.array([0, 1, 0, -1, 0 , 1, -1, -1, 1])
ey_d2q9 = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
w_d2q9 = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
#D2Q4 model for T
ex_d2q4 = np.array([1, 0, -1, 0])
ey_d2q4 = np.array([0, 1, 0, -1])
w_d2q4 = np.array([1/4,1/4,1/4,1/4])

beta = 1 #Coefficient of thermal expansion
temp0 = 0.5 #average temperature
dt = 1 #Time step
gr = 0.001 #Gravitational accelaeration
T = 1/1.95 #Relaxation time
Ra = 10000000 #Rayleigh number
Pr = 0.71 #Prandtl number


c = 1 #Lattice Speed
Tc = 0 #temperature cold
Th = 1 #temperature hot
v = np.sqrt((Pr/Ra)*gr*beta*dt*len_y**3) #Kinematic viscosity
T = 3*v+1/2 #Relaxation time for fluid temerature disturbution function
k = beta*gr*dt*len_y**3/Ra/v #Thermal Diffusivity
T_prime = 2*k+0.5 #Relaxation time for temperature disturbution function

def initialize():
    global p, u_hat, g, g_eq, t, t_eq, temp, forcing_term
    u_hat = np.zeros((len_x, len_y, 2))
    g = np.ones((len_x, len_y, 9))
    g[:,:] *= 1/9
    g_eq = g
    p = np.zeros((len_x, len_y))

    t = np.zeros((len_x, len_y, 4))
    t[:,-1,:] = Tc
    t[:,0,:] = Th
    t[len_x//2,-1,:] = 0.9
    t[:,:] *= w_d2q4
    t_eq = np.zeros((len_x, len_y, 4))
    temp = np.zeros((len_x, len_y))
    forcing_term = np.zeros((len_x, len_y, 9))
    

def streaming():
    global p, u_hat, g, g_eq, t, t_eq, temp
    #Calculate marcoscopic properties
    p = np.zeros((len_x, len_y))
    p = np.sum(g,2)
    temp = np.sum(t,2)
    u_hat[:,:,0] = np.sum(g[:,:] * ex_d2q9,2)/p
    u_hat[:,:,1] = np.sum(g[:,:] * ey_d2q9,2)/p


    for i, cx, cy in zip(range(4), ex_d2q4, ey_d2q4):
        t_eq[:,:,i] = (temp/4)*(1. + 2.*(cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))
    
    for i, cx, cy, W in zip(range(9), ex_d2q9, ey_d2q9, w_d2q9):
        if(i == 0):
            g_eq[:,:,i] = p*W*(1 + (-1.5*(u_hat[:,:,0]**2 + u_hat[:,:,1]**2)))
        elif(i >= 1 and i <= 4):
            g_eq[:,:,i] = p*W*(1 + ((3.0)*(cx*u_hat[:,:,0]+cy*u_hat[:,:,1])+4.5*((cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))**2 - 1.5*(u_hat[:,:,0]**2 + u_hat[:,:,1]**2)/(c**2)))
        elif(i >= 5 and i <= 8):
            g_eq[:,:,i] = p*W*(1 + (((3.0)*(cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))+4.5*((cx*u_hat[:,:,0]+cy*u_hat[:,:,1]))**2 - 1.5*(u_hat[:,:,0]**2 + u_hat[:,:,1]**2)/(c**2)))

    G = -1/2 * gr * p * (temp - np.mean(temp))
    #forcing_term[:,:,0] = np.zeros((len_x, len_y))
    #forcing_term[:,:,1] = np.zeros((len_x, len_y))
    forcing_term[:,:,2] = -np.ones((len_x, len_y)) * G
    #forcing_term[:,:,3] = np.zeros((len_x, len_y))
    forcing_term[:,:,4] = np.ones((len_x, len_y)) * G
    #forcing_term[:,:,5] = np.ones((len_x, len_y)) * G
    #forcing_term[:,:,6] = np.ones((len_x, len_y)) * G
    #forcing_term[:,:,7] = -np.ones((len_x, len_y)) * G
    #forcing_term[:,:,8] = -np.ones((len_x, len_y)) * G
    #forcing_term[:,:] *= 3 * w_d2q9
    g += -(g-g_eq)/T + forcing_term
    t += -(t-t_eq)/T_prime

    #Fluid Streaming
    g[:,:,1] = np.roll(g[:,:,1],1,axis = 0)
    g[:,:,2] = np.roll(g[:,:,2],1,axis = 1)
    g[:,:,3] = np.roll(g[:,:,3],-1,axis = 0)
    g[:,:,4] = np.roll(g[:,:,4],-1,axis = 1)
    g[:,:,5] = np.roll(g[:,:,5],1,axis = 0)
    g[:,:,5] = np.roll(g[:,:,5],1,axis = 1)
    
    g[:,:,6] = np.roll(g[:,:,6],-1,axis = 0)
    g[:,:,6] = np.roll(g[:,:,6],1,axis = 1)
                       
    g[:,:,7] = np.roll(g[:,:,7],-1,axis = 0)
    g[:,:,7] = np.roll(g[:,:,7],-1,axis = 1)
    
    g[:,:,8] = np.roll(g[:,:,8],1,axis = 0)
    g[:,:,8] = np.roll(g[:,:,8],-1,axis = 1)

    #Temperture Streaming
    t[:,:,0] = np.roll(t[:,:,0],1,axis = 0)
    t[:,:,1] = np.roll(t[:,:,1],1,axis = 1)
    t[:,:,2] = np.roll(t[:,:,2],-1,axis = 0)
    t[:,:,3] = np.roll(t[:,:,3],-1,axis = 1)

    #Top Wall(Bounce Back)
    g[:,-1,[8,4,7]] = g[:,-1,[6,2,5]]

    #Bottom Wall(Bounce Back)
    g[:,0,[6,2,5]] = g[:,0,[8,4,7]]

    #Zou-He boundary condition(Top Wall)
    t[:,-1,3] = Tc - t[:,-1,0] - t[:,-1,1] - t[:,-1,2]
    #Zou-He boundary condition(Bottom Wall)
    t[:,0,1] =  Th - t[:,0,0] - t[:,0,2] - t[:,0,3]

def main():
    global p, u_hat, g, g_eq, t, t_eq, temp
    initialize()
    temp_output = np.zeros((len_x,len_y ,time // tpf))
    velocity_output = np.zeros((len_x,len_y, 2 ,time // tpf))
    for count in range(time):
        print(count)
        streaming()
        if count % tpf == 0:
            temp_output[:,:,count // tpf] = temp
            velocity_output[:,:,:,count // tpf] = u_hat
    np.save('output_temp_Ra_1e7_tau1195',temp_output)
    np.save('output_velocity_Ra_1e7_tau1195',velocity_output)
    
    
if __name__ == "__main__":
    try:
        main()
    except RuntimeWarning:
        print("Overflow encountered!")

