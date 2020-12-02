import os
import sys
from sys import path
import pickle  # for reading the original wake data
import numpy as np

# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
prjName = 'deepwind'
jobName = 'gs20'
jobDir = prjDir + '/' + prjName + '/' + jobName
ppDir = '/scratch/sowfadata/pp/' + prjName + '/' + jobName

prbgList = os.listdir(jobDir + '/postProcessing/' + '.')
prbgList = [i for i in prbgList if i[0:5]=='probe']


data = {}

data['prbgNames'] = prbgList

for prbg in prbgList:

    startTimeList = os.listdir(jobDir + '/postProcessing/' + prbg + '/.')
    startTimeList.sort(key=float)


    prbgData = {}

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
    U = [[] for prb in range(prbNum)]

    for startTime in startTimeList:

        file = open(jobDir + '/postProcessing/' + prbg + '/' + startTime + '/' + 'U')
        data_org = [i.strip().replace('(','').replace(')','').split() for i in file.readlines()]
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
                U_ = data_org[r][1+p*3:1+p*3+3]
                U_ = [float(i) for i in U_]
                U[p].append(U_)

    prbgData['time'] = np.array(time)
    prbgData['U'] = np.array(U)

    data[prbg] = prbgData


''' save probeData into a binary file with pickle '''
f = open(ppDir + '/data/' + 'probeData', 'wb')
pickle.dump(data, f)
f.close()
