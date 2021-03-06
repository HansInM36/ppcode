#!/usr/bin/python3.8
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# directory of the case
jobName = 'StokesFirst'
prjDir = '/scratch/w2fdata/JOB/'
ppDir = '/scratch/w2fdata/pp/' + jobName

# 浪高仪组名字
wgName = 'wg0'

timestrList = os.listdir(prjDir + '/' + jobName + '/' + wgName + '/')
timestrList.sort(key=float)

# 布置的浪高仪总数
wgNum = 29

# 选择第几个浪高仪
wgN = 16

timelist = []
selist = []
for startTime in timestrList:
    # 读取浪高仪数据，形成二维list，第一维为行，第二维为列
    # 第0行为表头，第1~3行分别为x，y，z的坐标，第4行开始为波高
    data_org = [i.strip().split() for i in open(prjDir + '/' + jobName + '/' + wgName + '/' + startTime + '/' + 'surfaceElevation.dat').readlines()]

    # 测量时间点总数
    tNum = len(data_org) - 4

    # 时间序列
    timelist += [float(i[0]) for i in data_org[4:]]

    # 浪高序列
    selist += [float(i[wgN]) for i in data_org[4:]]

# 理论浪高
phi = np.pi*5/6 # 相位，需要手动调整
seidlist = [0.01*np.cos(np.pi*i + phi) for i in timelist]

# 绘制浪高时历图
plt.figure(figsize = (12,6))
plt.plot(timelist, selist, 'b-', label='simulation')
plt.plot(timelist, seidlist, 'r--', label='theory')
plt.title('surfaceElevation-time')
# plt.xticks(list(np.arange(0,26,1)),list(np.arange(0,26,1)))
# plt.yticks([-0.02,-0.01,0,0.01,0.02],[-0.02,-0.01,0,0.01,0.02])
plt.xlabel('time(s)')
plt.ylabel('SE(m)')
plt.grid()
plt.legend()
plt.savefig(ppDir + '/photo/se-t.png')
plt.show()
