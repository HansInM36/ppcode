#!/usr/bin/python3.8
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# directory of the case
prjDir = '/scratch/w2fdata/JOB'
ppDir = '/scratch/w2fdata/pp'

# directory of the case
jobNameList = ['StokesFirst', 'StokesSecond', 'StokesFifth']

# 浪高仪组名字
wgName = 'wg0'
wgNo = 16

time_plot_list = []
se_plot_list = []

for jobName in jobNameList:

    timestrList = os.listdir(prjDir + '/' + jobName + '/' + wgName + '/')
    timestrList.sort(key=float)

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
        selist += [float(i[wgNo]) for i in data_org[4:]]

    time_plot_list.append(timelist)
    se_plot_list.append(selist)


# 绘制浪高时历图
plt.figure(figsize = (12,6))
plt.plot(time_plot_list[0], se_plot_list[0], 'g-', label='StokesFirst')
plt.plot(time_plot_list[1], se_plot_list[1], 'b-', label='StokesSecond')
plt.plot(time_plot_list[2], se_plot_list[2], 'r-', label='StokesFifth')
plt.title('surfaceElevation-time')
plt.xlim(80,100)
# plt.ylim()
plt.xlabel('time(s)')
plt.ylabel('SE(m)')
plt.grid()
plt.legend()
saveName = 'se_ts.png'
plt.savefig(ppDir + '/' + saveName)
plt.show()
