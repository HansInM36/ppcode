import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from numpy import *
import pickle


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'pcr_NBL_sourceFixed'
jobDir = prjDir + '/' + prjName + '/' + jobName
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

sliceGroup = 'slices'

sliceList = ['Ny0', 'Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7']

# slice = 'Ny0'
# time = '7200'
scalarList = ['T']
vectorList = ['U']

timestrList = os.listdir(jobDir + '/postProcessing/' + sliceGroup + '/' + '.')
timestrList.sort(key=float)

# data is too huge, so we only take the first 2400 time steps
timestrList = timestrList[:2400]

timeList = [np.float(i) for i in timestrList]
timeArray = np.array(timeList)



# ### another way to read file
# file = open(jobDir + '/postProcessing/' + sliceGroup + '/' + time + '/' + varName + '_' + slice + '.vtk', 'r')
# data_org = [i.strip().split() for i in file.readlines()]


for slice in sliceList:

    print('extracting data of slice ' + slice + ': ...... ')

    # ---------- get data for the slicetion ---------- #
    sliceData = {}
    # sliceData has data structure as following
    #
    #              --- time
    #              --- pNo
    # sliceData ------ point
    #              --- scalars
    #              --- vectors
    #              --- scalar0 ------ 3darray --- time --- point --- scalarValue
    #              --- scalar1
    #              --- vector0 ------ 4darrray --- time --- point --- vectorValue --- x --- y --- z
    #              --- vector1

    # ---------- get time array ---------- #
    sliceData['time'] = timeArray

    # ---------- get points' coordinates array ---------- #
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(jobDir + '/postProcessing/' + sliceGroup + '/' + timestrList[0] + '/' + vectorList[0] + '_' + slice + '.vtk')
    # reader.ReadAllVectorsOn()
    # reader.ReadAllScalarsOn()
    # reader.ReadAllTensorsOn()
    reader.Update()
    polyData = reader.GetOutput()
    points = polyData.GetPoints()
    pointArray = vtk_to_numpy(points.GetData())

    pNum = pointArray.shape[0]
    pNoArray = np.array([[i] for i in range(pNum)])


    # pointArray = np.concatenate((pNoArray,pointArray), axis=1)
    # ### sort the array accoding to columns, the most important sorting should be at last
    # pointArray = pointArray[pointArray[:,1].argsort()]
    # pointArray = pointArray[pointArray[:,2].argsort(kind='stable')] # stable means it won't change the original order of those with same values
    # pointArray = pointArray[pointArray[:,3].argsort(kind='stable')]

    sliceData['pNo'] = pNoArray
    del pNoArray
    sliceData['point'] = pointArray
    del pointArray

    sliceData['scalars'] = scalarList
    # ---------- get scalar array ---------- #
    for scalar in scalarList:
        sliceData[scalar] = []
        for time in timestrList:
            print('Processing scalar: ' + scalar + ', ' + time)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(jobDir + '/postProcessing/' + sliceGroup + '/' + time + '/' + scalar + '_' + slice + '.vtk')
            # reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
            # reader.ReadAllTensorsOn()
            reader.Update()
            polyData = reader.GetOutput()
            pointData = polyData.GetPointData()
            sliceData[scalar].append(vtk_to_numpy(pointData.GetScalars(scalar)))
        sliceData[scalar] = np.array(sliceData[scalar])

    sliceData['vectors'] = vectorList
    # ---------- get vector array ---------- #
    for vector in vectorList:
        sliceData[vector] = []
        for time in timestrList:
            print('Processing vector: ' + vector + ', ' + time)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(jobDir + '/postProcessing/' + sliceGroup + '/' + time + '/' + vector + '_' + slice + '.vtk')
            reader.ReadAllVectorsOn()
            # reader.ReadAllScalarsOn()
            # reader.ReadAllTensorsOn()
            reader.Update()
            polyData = reader.GetOutput()
            pointData = polyData.GetPointData()
            sliceData[vector].append(vtk_to_numpy(pointData.GetVectors(vector)))
        sliceData[vector] = np.array(sliceData[vector])

    ''' save sliceData into a binary file with pickle '''
    f = open(ppDir + '/data/' + slice, 'wb')
    pickle.dump(sliceData, f)
    f.close()


# def Dplus(M,cNo):
#     rNum = M.shape[0]
#
#     M = M[M[:,cNo].argsort()]
#
#     M_ = []
#
#     tmp = M[0,cNo]
#     r0 = 0
#     for r in range(rNum):
#         if r == rNum-1:
#             M_.append(M[r0:r])
#         if M[r,cNo] == tmp:
#             continue
#         else:
#             tmp = M[r,cNo]
#             M_.append(M[r0:r-1])
#             r0 = r
