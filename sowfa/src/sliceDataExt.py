import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from numpy import *
import pickle


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
jobName = 'pcr_NBL_U10'
ppDir = '/scratch/sowfadata/pp/' + jobName

sliceGroup = 'slices_1'

sliceList = ['Ny0', 'Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7']

# slice = 'Ny0'
# time = '7200'
scalarList = ['T']
vectorList = ['U']

timestrList = os.listdir(prjDir + '/' + jobName + '/postProcessing/' + sliceGroup + '/' + '.')
timestrList.sort(key=float)

timeList = [np.float(i) for i in timestrList]
timeArray = np.array(timeList)

# ### another way to read file
# file = open(prjDir + '/' + jobName + '/postProcessing/' + sliceGroup + '/' + time + '/' + varName + '_' + slice + '.vtk', 'r')
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
    reader.SetFileName(prjDir + '/' + jobName + '/postProcessing/' + sliceGroup + '/' + timestrList[0] + '/' + vectorList[0] + '_' + slice + '.vtk')
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
    sliceData['point'] = pointArray

    sliceData['scalars'] = scalarList
    # ---------- get scalar array ---------- #
    for scalar in scalarList:
        scalarArray = []
        for time in timestrList:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(prjDir + '/' + jobName + '/postProcessing/' + sliceGroup + '/' + time + '/' + scalar + '_' + slice + '.vtk')
            # reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
            # reader.ReadAllTensorsOn()
            reader.Update()
            polyData = reader.GetOutput()
            pointData = polyData.GetPointData()
            scalarArray.append(vtk_to_numpy(pointData.GetScalars(scalar)))
        scalarArray = np.array(scalarArray)
        sliceData[scalar] = scalarArray

    sliceData['vectors'] = vectorList
    # ---------- get vector array ---------- #
    for vector in vectorList:
        vectorArray = []
        for time in timestrList:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(prjDir + '/' + jobName + '/postProcessing/' + sliceGroup + '/' + time + '/' + vector + '_' + slice + '.vtk')
            reader.ReadAllVectorsOn()
            # reader.ReadAllScalarsOn()
            # reader.ReadAllTensorsOn()
            reader.Update()
            polyData = reader.GetOutput()
            pointData = polyData.GetPointData()
            vectorArray.append(vtk_to_numpy(pointData.GetVectors(vector)))
        vectorArray = np.array(vectorArray)
        sliceData[vector] = vectorArray


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
