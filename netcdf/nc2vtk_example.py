import os
import sys
import numpy as np
from netCDF4 import Dataset
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

def palm_3d(dir, jobName, run_no_list, var):
    print("+++ Preparing 3d data for run " + jobName + "...")
    run_num = len(run_no_list)
    # read the output data of all run_no_list
    nc_file_list = []
    tSeq_list = []
    uSeq_list = []
    for i in range(run_num):
        input_file = dir + "/OUTPUT/" + jobName + "_3d" + run_no_list[i] + ".nc"
        nc_file_list.append(Dataset(input_file, "r", format="NETCDF4"))
        tSeq_list.append(np.array(nc_file_list[i].variables['time'][:], dtype=type(nc_file_list[i].variables['time'])))
        uSeq_list.append(np.array(nc_file_list[i].variables[var][:], dtype=type(nc_file_list[i].variables[var])))
    
    # print(list(nc_file_list[0].dimensions)) #list all dimensions
    # print(list(nc_file_list[0].variables)) #list all the variables
    # print(list(nc_file_list[0].variables['u2'].dimensions)) #list dimensions of a specified variable
    
    # extract the values of all dimensions of the var
    zName = list(nc_file_list[0].variables[var].dimensions)[1] # the height name string
    zSeq = np.array(nc_file_list[0].variables[zName][:], dtype=type(nc_file_list[0].variables[zName])) # array of height levels
    zSeq = zSeq.astype(float)
    yName = list(nc_file_list[0].variables[var].dimensions)[2] # the height name string
    ySeq = np.array(nc_file_list[0].variables[yName][:], dtype=type(nc_file_list[0].variables[yName])) # array of height levels
    ySeq = ySeq.astype(float)
    xName = list(nc_file_list[0].variables[var].dimensions)[3] # the height name string
    xSeq = np.array(nc_file_list[0].variables[xName][:], dtype=type(nc_file_list[0].variables[xName])) # array of height levels
    xSeq = xSeq.astype(float)
    
    # concatenate arraies of all run_no_list along the first dimension (axis=0), i.e. time
    tSeq = np.concatenate([tSeq_list[i] for i in range(run_num)], axis=0); tSeq = tSeq.astype(float)
    uSeq = np.concatenate([uSeq_list[i] for i in range(run_num)], axis=0); uSeq = uSeq.astype(float)
    return tSeq, zSeq, ySeq, xSeq

jobName  = 'WRFPALM_20150701'
dir = "/scratch/palmdata/JOBS/" + jobName 
tSeq, zSeq, ySeq, xSeq = palm_3d(dir, jobName, ['.000'], 'u')


tInd = 0
I, J, K = xSeq.size, ySeq.size, zSeq.size


pointArray = np.zeros((K*J*I,3))

for k in range(K):    
    pointArray[k*J*I:(k+1)*J*I,2] = zSeq[k]
    for j in range(J):
        pointArray[k*J*I+j*I:K*J*I+(j+1)*I,1] = ySeq[j]
        for i in range(I):
            pointArray[k*J*I+j*I+i,0] = xSeq[i]


UArray = np.vstack((uSeq[tInd].ravel(),vSeq[tInd].ravel(),wSeq[tInd].ravel()))
UArray = np.transpose(UArray)


filename = "test.vts"
sgData = vtk.vtkStructuredGrid()
sgData.SetDimensions([I,J,K])

points = vtk.vtkPoints()
points.SetData(numpy_to_vtk(pointArray))        
sgData.SetPoints(points)
sgData.GetPoints().Modified()
# vtk_to_numpy(sgData.GetPoints().GetData())

UArray_vtk = numpy_to_vtk(UArray, deep=True)
UArray_vtk.SetName("U")

sgData.GetPointData().SetVectors(UArray_vtk)
# vtk_to_numpy(sgData.GetPointData().GetVectors("U"))

sgData.GetPointData().Modified()

sg = vtk.vtkXMLStructuredGridWriter()
sg.SetFileName(filename)
sg.SetInputData(sgData)
sg.Write()


