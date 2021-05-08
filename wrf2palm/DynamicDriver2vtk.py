import os
import sys
import numpy as np
from netCDF4 import Dataset
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

readDir = '/scratch/palmdata/JOBS/EERASP3_1/INPUT'
readName = "EERASP3_1_dynamic"

data = Dataset(readDir + '/' + readName, "r", format="NETCDF4")

dimList = list(data.dimensions)
varList = list(data.variables)


### initial u (3d)
zSeq = data.variables['z'][:]
ySeq = data.variables['y'][:]
xuSeq = data.variables['xu'][:]

init_u = data.variables['init_atmosphere_u'][:]

# pointArray
I, J, K = xuSeq.size, ySeq.size, zSeq.size
yy, zz, xx = np.meshgrid(ySeq,zSeq,xuSeq) # must be in this order
pointArray = np.vstack((xx.ravel(),yy.ravel(),zz.ravel()))
pointArray = np.transpose(pointArray)

## pointArray (only for small domain)
#I, J, K = xuSeq.size, ySeq.size, zSeq.size
#pointArray = np.zeros((K*J*I,3))
#for k in range(K):    
#    pointArray[k*J*I:(k+1)*J*I,2] = zSeq[k]
#    for j in range(J):
#        pointArray[k*J*I+j*I:K*J*I+(j+1)*I,1] = ySeq[j]
#        for i in range(I):
#            pointArray[k*J*I+j*I+i,0] = xuSeq[i]

# uArray
uArray = init_u.reshape(init_u.size,1)

writeDir = "/scratch/projects/EERA-SP3/VTK/EERASP3_1/"
writeName = "init_u.vts"
sgData = vtk.vtkStructuredGrid()
sgData.SetDimensions([I,J,K])

points = vtk.vtkPoints()
points.SetData(numpy_to_vtk(pointArray))        
sgData.SetPoints(points)
sgData.GetPoints().Modified()
# vtk_to_numpy(sgData.GetPoints().GetData())

uArray_vtk = numpy_to_vtk(uArray, deep=True)
uArray_vtk.SetName("init_u")

sgData.GetPointData().SetScalars(uArray_vtk)
# vtk_to_numpy(sgData.GetPointData().GetVectors("U"))

sgData.GetPointData().Modified()

sg = vtk.vtkXMLStructuredGridWriter()
sg.SetFileName(writeDir + '/' + writeName)
sg.SetInputData(sgData)
sg.Write()
###



