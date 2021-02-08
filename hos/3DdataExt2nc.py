### This script read HOS Post_processing output file VP_card_fitted.dat ###
import os
import sys
import numpy as np
from netCDF4 import Dataset


prjDir = "/scratch/HOSdata/JOBS"
jobName = "wwinta_2"
suffix = ""


file = prjDir + '/' + jobName + '/' + "Results" + suffix + '/' + "3d.dat"
data_org = [i.strip().split() for i in open(file, encoding='latin1').readlines()]

datalen = len(data_org)

I = int(data_org[36][5][:-1]) # number of grid points along x-axis
J = int(data_org[36][7]) # number of grid points along y-axis

tlist = []

tInd = 0
r = 36 + (I*J+1) * tInd
while r < datalen:
    tlist.append(float(data_org[r][3][:-1]))
    tInd = tInd + 1
    r = 36 + (I*J+1) * tInd

tSeq = np.array(tlist)
tNum = tSeq.size

xSeq = np.zeros(I)
ySeq = np.zeros(J)

for i in range(I):
    xSeq[i] = data_org[37+i][0]
for j in range(J):
    ySeq[j] = data_org[37+j*I][1]


tmp = []
for t in range(tNum):
    tmp.append(np.array([[float(item) for item in data_org[37+t*(I*J+1)+r][-2:]] for r in range(I*J)]))
del data_org

etaList = []
phiList = []
for t in range(tNum):
    tmp0 = []
    tmp1 = []
    for j in range(J):
        tmp0.append(tmp[t][j*I:(j+1)*I,0])
        tmp1.append(tmp[t][j*I:(j+1)*I,1])
    etaList.append(np.array(tmp0))
    phiList.append(np.array(tmp1))

etaArray = np.array(etaList)
phiArray = np.array(phiList)


### open a new file for netcdf data
saveDir = '/scratch/ppcode/wwinta'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
saveName = "waveData" + suffix + '.nc'
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
ncfile.title='HOS 2D data of eta and phi for case ' + jobName

# store data
time[:] = tSeq
y[:] = ySeq
x[:] = xSeq

eta[:,:,:] = etaArray
phi[:,:,:] = phiArray

# close file
ncfile.close()
