'''
Python code to simulate and animate heat equtaion
'''
import numpy as np
from scipy.ndimage import laplace

import pylab as plt
from matplotlib.animation import FuncAnimation

numberOfGridPoints = 50
dt = 0.1
alpha = 1
numberOfIterations = 100000
u = np.zeros((numberOfGridPoints, numberOfGridPoints))
#initial condition (heat peak in center)
u[numberOfGridPoints//2, numberOfGridPoints//2] = 100
#boundary condition (Dirichlet boundary condition)
b = 1.0

def update(u, alpha, dt, b):
    return u + dt*alpha*laplace(u, mode='constant', cval=b)


#Plotting only
fig, ax = plt.subplots()

def animate(i):
    global u
    u = update(u, alpha, dt, b)
    ax.imshow(u)
    ax.set_axis_off()

anim = FuncAnimation(fig, animate, frames=numberOfIterations, interval=10)

plt.show()
