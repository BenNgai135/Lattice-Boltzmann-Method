import numpy as np
import matplotlib.pyplot as plt

ex = np.array([0, 1, 0, -1, 0 , 1, -1, -1, 1])
ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
frame = -1
data = np.load('velocity_Re360_400_100.npy')[:,:,:,frame]


# Define the vector field function
def vector_field(x, y):
    u = data[x,y,0]  # x-component of the vector field
    v = data[x,y,1]  # y-component of the vector field
    return u, v

# Create a grid of points
x = np.arange(0, np.shape(data)[0],1)
y = np.arange(0, np.shape(data)[1],1)
X, Y = np.meshgrid(x, y)
# Evaluate the vector field at each point on the grid
U, V = vector_field(X, Y)

fig, ax = plt.subplots()
img = ax.streamplot(X, Y, U, V, color='blue', linewidth=0.5, density=5, arrowstyle='->', arrowsize=1.5)

# Plot the vector field with arrows
plt.show()
