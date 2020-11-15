### This script read HOS Post_processing output file VP_card_fitted.dat ###

import numpy as np
import pickle

prjdir = "/scratch/HOSdata/JOBS/"
prjname = "regular"
jobname = "ka03"
suffix = ""

file = prjdir + prjname + '/' + jobname + '/' + "Results" + suffix + '/' + "3d.dat"
data_org = [i.strip().split() for i in open(file, encoding='latin1').readlines()]

datalen = len(data_org)

I = int(data_org[36][5][:-1]) # number of grids along x-axis
J = int(data_org[36][7]) # number of grids along y-axis

tlist = []

tind = 0
r = 36 + (I*J+1) * tind
while r < datalen:
    tlist.append(float(data_org[r][3][:-1]))
    tind = tind + 1
    r = 36 + (I*J+1) * tind

tseq = np.array(tlist)
tNum = tseq.size

# t_start = float(data_org[36][3][:-1]) # start time
# t_end = float(data_org[-I*J-1][3][:-1]) # end time
# t_2 = float(data_org[36+I*J+1][3][:-1]) # the second time
# tNum = int((t_end - t_start) / (t_2 - t_start)) + 1
#
# tseq = np.linspace(t_start, t_end, tNum)

xseq = np.zeros(I)
yseq = np.zeros(J)

for i in range(I):
    xseq[i] = data_org[37+i][0]
for j in range(J):
    yseq[j] = data_org[37+j*I][1]


tmp = []
for t in range(tNum):
    tmp.append(np.array([[float(item) for item in data_org[37+t*(I*J+1)+r][-2:]] for r in range(I*J)]))
del data_org

data = {} # save data_org as a dict
data['time'] = tseq
data['x'] = xseq
data['y'] = yseq

eta = []
phi = []
for t in range(tNum):
    tmp0 = []
    tmp1 = []
    for j in range(J):
        tmp0.append(tmp[t][j*I:(j+1)*I,0])
        tmp1.append(tmp[t][j*I:(j+1)*I,1])
    eta.append(np.array(tmp0))
    phi.append(np.array(tmp1))

data['eta'] = np.array(eta)
data['phi'] = np.array(phi)

saveDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname + '/data/'
saveName = "2Ddata_Dict" + suffix
fw = open(saveDir + saveName, "wb")
pickle.dump(data, fw, 2)
fw.close()
