import os
import numpy as np

# the directory where the wake data locate
prjDir = '/scratch/sowfadata/JOBS'
jobName = 'pcr_NBL'
ppDir = '/scratch/sowfadata/pp/' + jobName + '/data'

sourceHistoryData = {}

# more than one time folder will be created if we run the case by several times
startTimeList = os.listdir(prjDir + '/' + jobName + '/postProcessing/SourceHistory/.')
startTimeList.sort(key=float)

varList = ['SourceUXHistory', 'SourceUYHistory', 'SourceUZHistory', 'SourceTHistory']

# collect time sequences in all time folder and concatenate them into one single array
tmp = []
last_t_end = 0
t_start_list = {} # records the index of start time of every startTime folder (to avoid duplicated time steps)
for startTime in startTimeList:
    file = open(prjDir + '/' + jobName + '/postProcessing/SourceHistory/' + startTime + '/' + varList[0], 'r')
    data_org = [i.strip().split() for i in file.readlines()]
    file.close()
    tmpp = [np.float(row[0]) for row in data_org[1:]]
    if last_t_end <= 0:
        tmp = tmp + tmpp
        last_t_end = tmp[-1]
        ind_ = 0
    else:
        if tmpp[0] > last_t_end:
            tmp = tmp + tmpp
            last_t_end = tmp[-1]
            ind_ = 0
        else:
            ind_ = np.where(np.array(tmpp) - last_t_end > 0)[0][0]
            tmp = tmp + tmpp[ind_:]
            last_t_end = tmp[-1]
    t_start_list[startTime] = ind_

timeArray = np.array(tmp).astype('float')
sourceHistoryData['time'] = timeArray


for var in varList:
    tmp = []
    for startTime in startTimeList:
        file = open(prjDir + '/' + jobName + '/postProcessing/SourceHistory/' + startTime + '/' + var, 'r')
        data_org = [i.strip().split() for i in file.readlines()]
        file.close()
        tmpp = [row[2:] for row in data_org[1:]]
        tmp = tmp + tmpp[t_start_list[startTime]:]
    varArray = np.array(tmp).astype('float')
    sourceHistoryData[var] = varArray

# save horizontally averaged data into a binary file with pickle
f = open(ppDir + '/' + 'sourceHistoryData', 'wb')
pickle.dump(sourceHistoryData, f)
f.close()
