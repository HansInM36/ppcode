import os
import numpy as np
import matplotlib.pyplot as plt

def plot2d(xx,yy,v,v_min,v_max,dv,ind,saveDir,save=False):
    fig, axs = plt.subplots(figsize=(6,6), constrained_layout=False)
    cbreso = 100
    levels = np.linspace(v_min, v_max, cbreso + 1)
    CS = axs.contourf(xx, yy, v, cbreso, levels=levels, cmap='jet', vmin=v_min, vmax=v_max)

    cbartickList = np.linspace(v_min, v_max, int((v_max-v_min)/dv)+1)
    cbar = plt.colorbar(CS, ax=axs, orientation='vertical', ticks=cbartickList, fraction=.1)
    plt.title('')
    if save:
        saveName = "%.4d" % ind + '.png'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        plt.savefig(saveDir + '/' + saveName, bbox_inches='tight')
    plt.show()
    plt.close('all')
    return 
   

#""" Example: 2D steady convection-diffusion problem """
#u, w = 0.8, 0.8
#nu = 0.1
#
#I, K = 100, 100
#dx, dz = 1, 1
#
#N = 60
#dt = 0.5
#
#c_x, c_z = u*dt/dx, w*dt/dz
#
#x, z = np.arange(I+1)*dx, np.arange(K+1)*dz
#xx, zz =np.meshgrid(x,z)
#
#F = np.zeros((N+1,K+1,I+1))
#
### initial condition (1)
##F[0,:,:] = np.random.rand(K+1,I+1) * 10
#
## initial condition (2)
#Fs = np.zeros(I)
#Fn = np.zeros(I)
#Fl = np.zeros(K)
#Fr = np.zeros(K)
#
#for i in range(I):
#    Fs[i] = 10
##    Fn[i] = i
#for j in range(K):
#    Fl[j] = 10
##    Fr[j] = J - j    
#
#F[0,0,:-1] = Fs
#F[0,-1,1:] = Fn
#F[0,1:,0] = Fl
#F[0,:-1,-1]   
#
##plot2d(xx,yy,F[0,:,:],0,10,2)
#
#for n in range(1,N+1):    
#    for i in range(0,I+1):
#        for k in range(0,K+1):
#            s_, n_, l_, r_ = k-1, k+1, i-1, i+1
#            if i == 0:
#                l_ = I-1
#            if i == I:
#                r_ = 1
#            if k == 0:
#                s_ = J-1
#            if k == J:
#                n_ = 1
#            F[n,k,i] = F[n-1,k,i] * (1 - c_x - c_z - 2*nu*dt/dx/dx - 2*nu*dt/dz/dz) + \
#                       F[n-1,k,l_] * (c_x + nu*dt/dx/dx) + \
#                       F[n-1,k,r_] * (nu*dt/dx/dx) + \
#                       F[n-1,s_,i] * (c_z + nu*dt/dz/dz) + \
#                       F[n-1,n_,i] * (nu*dt/dz/dz)
#
#    
#plot2d(xx,zz,F[-1,:,:],0,10,2)


""" Example: 2D steady convection problem in transformed coordinates """
u, w = 0.4, 0.4
I, K = 100, 100
dx, dz = 1, 1
dx_, dz_ = 1, 0.01

N = 60
dt = 1.0

c_x, c_z = u*dt/dx, w*dt/dz
c_x_x, c_x_z, c_z_z = u*dt/dx_, u*dt/dz_, w*dt/dz_

x, z = np.arange(I+1)*dx, np.arange(K+1)*dz
xx, zz =np.meshgrid(x,z)

x_, z_ = np.arange(I+1)*dx_, np.arange(K+1)*dz_
xx_, zz_ =np.meshgrid(x_,z_)



H = 100

# wavy surface
wl = 50
wk = 2*np.pi/wl

def eta(x):
    return 10 * np.cos(wk*x) 

# mapping relation
def x(x_,z_):
    return x_
def z(x_,z_):
    return (H - eta(x(x_,z_)))*z_ + eta(x(x_,z_))


## initial condition (2)
#Fs = np.zeros(I)
#Fn = np.zeros(I)
#Fl = np.zeros(K)
#Fr = np.zeros(K)
#
#for i in range(I):
#    Fs[i] = 10
##    Fn[i] = i
#for j in range(K):
#    Fl[j] = 0
##    Fr[j] = J - j    
#
#F[0,0,:-1] = Fs
#F[0,-1,1:] = Fn
#F[0,1:,0] = Fl
#F[0,:-1,-1]   

## initial condition (3)
#F[0,20,:] = 10



### solve in cartesian coordinates
# initial condition
F = np.zeros((N+1,K+1,I+1))
for k in range(K+1):
    for i in range(I+1):
        if np.power(dx*i-30,2) + np.power(dz*k-30,2) <= 100:
            F[0,k,i] = 8
            
for n in range(1,N+1):    
    for k in range(1,K+1):
        for i in range(1,I+1):
            F[n,k,i] = F[n-1,k,i] * (1 - c_x - c_z) + \
                       F[n-1,k,i-1] * c_x + \
                       F[n-1,k-1,i] * c_z
    for k in range(1,K+1):
        F[n,k,0] = F[n,k,1]
    for i in range(0,I+1):
        F[n,0,i] = F[n,1,i]

rst_p = np.copy(F) # results computed in cartesian coordinates
for n in range(N+1):        
    plot2d(xx,zz,rst_pp[n,:,:],-2,10,1,n,'/scratch/palmdata/pp/wwinta/animation/pc_ps',save=True)

### solve in z-normalized coordinates
# initial condition
F = np.zeros((N+1,K+1,I+1))
for k in range(K+1):
    for i in range(I+1):
        if np.power(x(dx_*i,dz_*k)-30,2) + np.power(z(dx_*i,dz_*k)-30,2) <= 100:
            F[0,k,i] = 8
            
for n in range(1,N+1):    
    for i in range(1,I+1):
        for k in range(1,K+1):
            s_, n_, l_, r_ = k-1, k+1, i-1, i+1
                
            dzdx_ = (z(xx_[k,i],zz_[k,i]) - z(xx_[k,l_],zz_[k,l_]))/dx_
            J = H - eta(xx_[k,i])
            
            F[n,k,i] = F[n-1,k,i]*(1 - c_x_x + dzdx_/J*c_x_z - 1/J*c_z_z) + \
                       F[n-1,k,l_]*(c_x_x) + \
                       F[n-1,s_,i]*(- dzdx_/J*c_x_z + 1/J*c_z_z)
    for k in range(1,K+1):
        F[n,k,0] = F[n,k,1]
    for i in range(0,I+1):
        F[n,0,i] = F[n,1,i]   


rst_c = np.copy(F) # results computed in z-normalized coordinates
for n in range(N+1):        
    plot2d(x(xx_,zz_),z(xx_,zz_),rst_c[n,:,:],-2,10,1,n,'/scratch/palmdata/pp/wwinta/animation/cc_ps',save=True)


   

