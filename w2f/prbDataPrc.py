#!/usr/bin/python3.8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# directory of the case
jobNameList = ['StokesFirst', 'StokesSecond', 'StokesFifth']

prjDir = '/scratch/w2fdata/JOB'
ppDir = '/scratch/w2fdata/pp'

prbgList = ['prb0', 'prb1', 'prb2', 'prb3', 'prb4', 'prb5', 'prb6', 'prb7']
prbNumList = [29,29,29,29,29,29,29,29]
prbgNum = len(prbList)

var = 'U'
varD = 1 # u:1, v:2, w:3

# prb's No
prbNo = 16

data = {}

for jobName in jobNameList:

    prbData_list = {}

    for i in range(prbgNum):
        # prb组名
        prbg = prbgList[i]
        prbNum = prbNumList[i]

        timestrList = os.listdir(prjDir + '/' + jobName + '/postProcessing/' + prbg + '/')

        timeList = []
        uList = []
        for startTime in timestrList:

            # 读取浪高仪数据，形成二维list，第一维为行，第二维为列
            # 第0行为表头，第1~3行分别为x，y，z的坐标，第4行开始为波高
            data_org = [i.strip().split() for i in open(prjDir + '/' + jobName + '/postProcessing/' + prbg + '/' + startTime + '/' + 'U').readlines()]
            rNum = len(data_org)
            for r in range(rNum):
                data_org[r] = [str.replace("(", " ") for str in data_org[r]]
                data_org[r] = [str.replace(")", " ") for str in data_org[r]]
                if r > prbNum + 1:
                    data_org[r] = [float(str) for str in data_org[r]]
            data_org = data_org[prbNum+2:]

            # 时间序列
            timeList += [float(i[0]) for i in data_org]
            # u 序列
            uList += [i[3*prbNo+varD] for i in data_org]

        prbData_list[prbg] = [timeList, uList]

    data[jobName] = prbData_list


jobName = 'StokesSecond'

varplotList = []

for i in range(prbgNum):
    prbg = prbgList[i]
    tplot = data[jobName][prbg][0]
    varSeq = data[jobName][prbg][1]
    varplotList.append(varSeq)


# 绘制浪高时历图
tplotNum = prbgNum
fig, ax = plt.subplots(figsize=(12,6))
colors = plt.cm.jet(np.linspace(0,1,tplotNum))

for i in range(tplotNum):
    plt.plot(tplot, varplotList[i], label=prbgList[i], linewidth=1.0, color=colors[i])
plt.ylabel('u (m/s)')
plt.xlabel('t (s)')
# xaxis_min = 60
# xaxis_max = 70
# xaxis_d = 2
# yaxis_min = 0
# yaxis_max = 800.0
# yaxis_d = 100.0
# plt.ylim(yaxis_min - 0.25*yaxis_d,yaxis_max)
# plt.xlim(xaxis_min - 0.25*xaxis_d,xaxis_max)
# plt.xticks(list(np.linspace(xaxis_min, xaxis_max, int((xaxis_max-xaxis_min)/xaxis_d)+1)))
# plt.yticks(list(np.linspace(yaxis_min, yaxis_max, int((yaxis_max-yaxis_min)/yaxis_d)+1)))
plt.legend(bbox_to_anchor=(1.05,0.5), loc=6, borderaxespad=0) # (1.05,0.5) is the relative position of legend to the origin, loc is the reference point of the legend
plt.grid()
plt.title('')
fig.tight_layout() # adjust the layout
saveName = 'u_ts' + '_' + jobName + '.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
