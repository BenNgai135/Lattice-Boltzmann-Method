import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
#Change All the vorticity to speed if you want to see the speed 

data = np.load("velocity_Re360_400_100.npy")[:,:,:,:]

speed = np.sqrt(data[:,:,0,:]**2 + data[:,:,1,:]**2)
vorticity = (np.roll(data[:,:,1,:],1,axis = 0) - np.roll(data[:,:,1,:],-1,axis = 0)) - ((np.roll(data[:,:,0,:],1,axis = 1) - np.roll(data[:,:,0,:],-1,axis = 1)))


fig, ax = plt.subplots()
img = ax.imshow(np.transpose(vorticity[:,-1:0:-1,0]), cmap='viridis')

time_step = 1.0  

time_text = ax.text(0.72, 1.05, '', transform=ax.transAxes,color='black',bbox=dict(facecolor='white', alpha=1, edgecolor='none'))

plt.colorbar(img) 

def update(frame):
    print(np.mean(vorticity[:,:,frame]))
    new_matrix = np.transpose(vorticity[:,-1:0:-1,frame])
    img.set_array(new_matrix)
    # Update the time text
    time_text.set_text('Frame = %.1f' % (frame * time_step))
    return img, time_text,

animation = FuncAnimation(fig, update, frames=np.shape(vorticity)[2], interval=10, blit=True)
plt.show()
