### This script read HOS Post_processing output file VP_card_fitted.dat ###
import os
import sys
import numpy as np
from netCDF4 import Dataset

""" INPUT START """
## wave properties
wl = 100
ka = 0.1

## mesh
I = 256
J = 64
f_out = 10 # frequency of output in one wave period
Lx = 10 # length in wave length
Ly = 6 # length in wave length
Tn = 10 # time in wave period
""" INPUT END """


k = 2*np.pi/wl
A = ka/k
omg = np.power(9.81*k,0.5)
T = 2*np.pi/omg
f = 1/T

tSeq = np.linspace(0,T*Tn,f_out*Tn); tNum = tSeq.size
xSeq = np.linspace(0,wl*Lx,I);       xNum = xSeq.size
ySeq = np.linspace(0,wl*Ly,J);       yNum = ySeq.size

etaArray = np.zeros((tNum, yNum, xNum))
for tInd in range(tNum):
    for xInd in range(xNum):
        etaArray[tInd,:,xInd] = A * np.sin(omg*tSeq[tInd] - k*xSeq[xInd])

phiArray = np.zeros((tNum, yNum, xNum))
for tInd in range(tNum):
    for xInd in range(xNum):
        phiArray[tInd,:,xInd] = omg/k*A * np.cos(omg*tSeq[tInd] - k*xSeq[xInd])



### open a new file for netcdf data
saveDir = '/scratch/ppcode/wwinta'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
saveName = 'waveData_regular.nc'
ncfile = Dataset(saveDir + '/' + saveName, mode='w', format='NETCDF4_CLASSIC')

# create dimensions
time_dim = ncfile.createDimension('time', None)
y_dim = ncfile.createDimension('y', J)
x_dim = ncfile.createDimension('x', I)

# create variables
time = ncfile.createVariable('time', np.float32, ('time'))
y = ncfile.createVariable('y', np.float32, ('y'))
x = ncfile.createVariable('x', np.float32, ('x'))

eta = ncfile.createVariable('eta', np.float32, ('time', 'y', 'x'))
eta.units = 'm'
eta.long_name = 'surface_elevation'

phi = ncfile.createVariable('phi', np.float32, ('time', 'y', 'x'))
phi.units = 'm^2/s'
phi.long_name = 'velocity potential'

# title of this file
ncfile.title='2D data of eta and phi for case regular wave'

# store data
time[:] = tSeq
y[:] = ySeq
x[:] = xSeq

eta[:,:,:] = etaArray
phi[:,:,:] = phiArray

# close file
ncfile.close()
