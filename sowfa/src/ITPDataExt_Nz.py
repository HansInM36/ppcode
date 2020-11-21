import sys
sys.path.append('/scratch/ppcode/sowfa/src')
import imp
import numpy as np
import pickle
import sliceDataClass as sdc
import matplotlib.pyplot as plt


# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
jobName = 'pcr_NBL_U10'
ppDir = '/scratch/sowfadata/pp/' + jobName

sliceList = ['Nz0', 'Nz1', 'Nz2', 'Nz3', 'Nz4', 'Nz5', 'Nz6', 'Nz7']
sliceNum = len(sliceList)

readDir = ppDir + '/data/'
readName = sliceList[0]
fr = open(readDir + readName, 'rb')
data_org = pickle.load(fr)
fr.close()
slc = sdc.Slice(data_org, 2)
tSeq = slc.data['time']
tNum = tSeq.size


for slice in sliceList:

    ITPData = {}

    readDir = ppDir + '/data/'
    readName = slice
    fr = open(readDir + readName, 'rb')
    data_org = pickle.load(fr)
    fr.close()

    slc = sdc.Slice(data_org, 2)

    ITPuList = []
    ITPvList = []
    ITPwList = []
    for tInd in range(tNum):
        print('processing ' + slice + ' %' + str(round(tInd/tNum*100,2)) + ' ......')
        utmp = slc.meshITP_Nz((0,2000,100), (0,2000,100), slc.data['U'][tInd][:,0], method_='cubic')
        vtmp = slc.meshITP_Nz((0,2000,100), (0,2000,100), slc.data['U'][tInd][:,1], method_='cubic')
        wtmp = slc.meshITP_Nz((0,2000,100), (0,2000,100), slc.data['U'][tInd][:,2], method_='cubic')
        ITPuList.append(utmp[2])
        ITPvList.append(vtmp[2])
        ITPwList.append(wtmp[2])

        if tInd == 0:
            ITPData['x'] = utmp[0]
            ITPData['y'] = utmp[1]

    ITPData['u'] = np.array(ITPuList)
    ITPData['v'] = np.array(ITPvList)
    ITPData['w'] = np.array(ITPwList)

    ITPData['time'] = tSeq

    ''' save sliceData into a binary file with pickle '''
    f = open(ppDir + '/data/' + slice + '_ITP', 'wb')
    pickle.dump(ITPData, f)
    f.close()
