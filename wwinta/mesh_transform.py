import os
import numpy as np
import matplotlib.pyplot as plt

""" computational domain """
I = 21
K = 21

Lx = 200

x = np.linspace(0, Lx, I)
z = np.linspace(0, 1, K)
xx, zz =np.meshgrid(x,z)

""" wave parameter """
phi = np.random.rand() * np.pi
def eta(x,t,phi):
    g = 9.81
    a = 3.2
    ka = 0.2

    k = ka/a
    wl = 2*np.pi / k    
    T = np.sqrt(2*np.pi*wl/g)
    omg = 2*np.pi/T
    c = np.sqrt(g/k)
    return a*np.cos(omg*t - k*x + phi)

""" physical domain """
t = 0
H = 100 # height of the domain

xx_ = np.zeros(xx.shape)
zz_ = np.zeros(zz.shape)

for k in range(K):
    for i in range(I):
        xx_[k,i] = xx[k,i]
        zz_[k,i] = zz[k,i] * (H - eta(xx[k,i],t,phi)) + eta(xx[k,i],t,phi)

""" plot """
### computational domain
fig, axs = plt.subplots(figsize=(6.0, 6.0))
plt.scatter(xx, zz, 1, marker='o', color='k')
saveDir = '/scratch/projects/wwinta/photo/mesh_transform'
saveName = 'cartesian.png'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()

### physical domain
fig, axs = plt.subplots(figsize=(6.0, 6.0))
plt.scatter(xx_, zz_, 1, marker='o', color='k')
saveDir = '/scratch/projects/wwinta/photo/mesh_transform'
saveName = 'znormalized.png'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
plt.show()








