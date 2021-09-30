import os
import sys
from sys import path
import pickle  # for reading the original wake data
import numpy as np

# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'examples'
jobName = 'NBL'
jobDir = prjDir + '/' + prjName + '/' + jobName
scalarList = []
vectorList = ['U']

vectorNum = len(vectorList)
scalarNum = len(scalarList)

list = os.listdir(jobDir + '/postProcessing/' + '.')
prbgList = [i for i in list if i[0:4]=='prbg']
prbgList.sort()

prbgList = ['prbg0']

for prbg in prbgList:
    print('Processing ' + prbg + ' ......')

    prbgData = {}

    # read info of this prbg, i.e (O, alpha, xList, yList, zList)
    readDir = jobDir + '/data/'
    readName = prbg + '_info'
    fr = open(readDir + readName, 'rb')
    prbgData['info'] = pickle.load(fr)
    fr.close()


    startTimeList = os.listdir(jobDir + '/postProcessing/' + prbg + '/.')
    startTimeList.sort(key=float)


    '''use the file in the first startTime to get basic information for probes'''

    file = open(jobDir + '/postProcessing/' + prbg + '/' + startTimeList[0] + '/' + 'U')
    data_org = [i.strip().replace('(','').replace(')','').split() for i in file.readlines()]
    file.close()

    rNum = len(data_org)

    # find the total number of probes
    prbNum = 0
    for row in data_org:
        if row[1] == 'Probe':
            prbNum += 1
        else:
            break
    prbNum -= 1

    # get probes' coordinates
    coors = []
    for r in range(prbNum):
        tmp = data_org[r][-3:]
        coors.append([float(i) for i in tmp])
    coors = np.array(coors)
    prbgData['coors'] = coors


    time = []
    vectorDataList = []; vector_org_list = []
    scalarDataList = []; scalar_org_list = []


    if vectorNum != 0:
        for i in range(vectorNum):
            tmp = [[] for prb in range(prbNum)]
            vectorDataList.append(tmp)

    if scalarNum != 0:
        for i in range(scalarNum):
            tmp = [[] for prb in range(prbNum)]
            scalarDataList.append(tmp)


    for startTime in startTimeList:

        if vectorNum != 0:
            for i in range(vectorNum):
                file = open(jobDir + '/postProcessing/' + prbg + '/' + startTime + '/' + vectorList[i])
                tmp = [i.strip().replace('(','').replace(')','').split() for i in file.readlines()]
                vector_org_list.append(tmp)
                file.close()

        if scalarNum != 0:
            for i in range(scalarNum):
                file = open(jobDir + '/postProcessing/' + prbg + '/' + startTime + '/' + scalarList[i])
                tmp = [i.strip().replace('(','').replace(')','').split() for i in file.readlines()]
                scalar_org_list.append(tmp)
                file.close()

        for r in range(prbNum+2,rNum):
            t_ = float(data_org[r][0])
            if startTime == startTimeList[0]:
                time.append(t_)
            else:
                if t_ > time[-1]:
                    time.append(t_)
                else: break
            for p in range(prbNum):
                if vectorNum != 0:
                    for i in range(vectorNum):
                        v_ = vector_org_list[i][r][1+p*3:1+p*3+3]
                        v_ = [float(i) for i in v_]
                        vectorDataList[i][p].append(v_)
                if scalarNum != 0:
                    for i in range(scalarNum):
                        s_ = scalar_org_list[i][r][1+p]
                        s_ = [float(s_)]
                        scalarDataList[i][p].append(s_)

    prbgData['time'] = np.array(time)
    
    if vectorNum != 0:
        for i in range(vectorNum):
            prbgData[vectorList[i]] = np.array(vectorDataList[i])
    if scalarNum != 0:
        for i in range(scalarNum):
            prbgData[scalarList[i]] = np.array(scalarDataList[i])


    ''' save probeData into a binary file with pickle '''
    f = open(jobDir + '/data/' + prbg, 'wb')
    pickle.dump(prbgData, f)
    f.close()
