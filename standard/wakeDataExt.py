import os
import csv
import numpy as np
from numpy import *
import pickle

# the directory where the wake data locate
projDir = '/home/rao/myproject/CHJournal/'
caseName = 'yaw30-turbIn-RANS'
secList = ['PlaneZ']

for sec in secList:
    print(sec, ' ......')
    secList = [sec]

    caseDir = caseName + '/postProcessing/surfaces/'

    timeList = os.listdir(projDir + caseDir + '.')
    timeList.sort()
    startIndex_1 = timeList.index('180')
    stopIndex_1 = timeList.index('300')

    timeList = timeList[startIndex_1:stopIndex_1]


    # initialize a dict storing wake data in each section of each time step
    wakeDataDict = dict(zip(timeList,timeList))

    for time in timeList:
        secDataDict = dict(zip(secList,secList)) # initialize a dict storing wake data in each section of a certain time step
        for sec in secList:
            file_mesh = open(projDir + caseDir + time + '/U' + '/' + sec + '.000.mesh.csv', 'r')
            file_U = open(projDir + caseDir + time + '/U' + '/' + sec + '.000.U.csv', 'r')
            data_mesh = csv.reader(file_mesh)
            data_U = csv.reader(file_U)
            del file_mesh
            del file_U
            rows_mesh = [row for row in data_mesh] # all information of the file are stored in a list containing lists the number
    # of which is the same as the number of lines in the file
            rows_U = [row for row in data_U]
            del data_mesh
            del data_U

            pNum = int(rows_mesh[8][0]) #total number of points in the transverse section

            secData = mat(zeros((pNum,7))) #initialize a matrix for store section data

            for i in range(0, pNum):
                secData[i,0] = int(i + 1)
                # secData[i,1] = float(rows_mesh[9+pNum*0+i][0])
                secData[i,1] = float(rows_mesh[9+pNum*1+i][0]) # ALM Solver
                # secData[i,2] = float(rows_mesh[9+pNum*1+i][0])
                secData[i,2] = float(rows_mesh[9+pNum*0+i][0]) # ALM Sovler
                secData[i,3] = float(rows_mesh[9+pNum*2+i][0])

                # secData[i,4] = float(rows_U[4+pNum*0+i][0])
                secData[i,4] = float(rows_U[4+pNum*1+i][0]) # ALM Solver
                # secData[i,5] = float(rows_U[4+pNum*1+i][0])
                secData[i,5] = float(rows_U[4+pNum*0+i][0]) # ALM Solver
                secData[i,6] = float(rows_U[4+pNum*2+i][0])

            secDataDict[sec] = secData
        wakeDataDict[time] = secDataDict

    ''' save wakeDataDict into a file with pickle '''
    # f = open(projDir + 'postProcessing_all/data.org/' + caseName + '_' + secList[0] + '_wakeData_part1', 'wb')
    # f = open(projDir + 'postProcessing_all/data.org/' + caseName + '_' + secList[0] + '_wakeData_part2', 'wb')
    f = open(projDir + 'postProcessing_all/data.org/' + caseName + '_' + sec + '_wakeData', 'wb')
    pickle.dump(wakeDataDict, f)
    f.close()
