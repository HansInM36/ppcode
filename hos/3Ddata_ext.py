### This script read HOS Post_processing output file VP_card_fitted.dat ###

import numpy as np
import pickle

prjdir = "/scratch/HOSdata/JOBS/"
prjname = "regular"
jobname = "ka03"
suffix = ""

file = prjdir + prjname + '/' + jobname + '/' + "Postprocessing/Results" + '/' + "VP_card_fitted.dat"
data_org = [i.strip().split() for i in open(file, encoding='latin1').readlines()]

datalen = len(data_org)

I = int(data_org[2][5]) # number of grids along x-axis
J = int(data_org[2][7]) # number of grids along y-axis
K = int(data_org[2][9]) # number of grids along z-axis

tlist = []

tind = 0
r = 2 + (I*J*K+1) * tind
while r < datalen:
    tlist.append(float(data_org[r][3][:-1]))
    tind = tind + 1
    r = 2 + (I*J*K+1) * tind

tseq = np.array(tlist)
tNum = tseq.size

# t_start = float(data_org[2][3][:-1]) # start time
# t_end = float(data_org[-I*J*K-1][3][:-1]) # end time
# t_2 = float(data_org[2+I*J*K+1][3][:-1]) # the second time
# tNum = int((t_end - t_start) / (t_2 - t_start)) + 1
#
# tseq = np.linspace(t_start, t_end, tNum)

xseq = np.zeros(I)
yseq = np.zeros(J)
zseq = np.zeros(K)

for i in range(I):
    xseq[i] = data_org[3+i][0]
for j in range(J):
    yseq[j] = data_org[3+j*I][1]
for k in range(K):
    zseq[k] = data_org[3+k*I*J][2]

tmp = []
for t in range(tNum):
    tmp.append(np.array([[float(item) for item in data_org[3+t*(I*J*K+1)+r][-5:]] for r in range(I*J*K)]))

data = {} # save data_org as a dict
data['time'] = tseq
data['x'] = xseq
data['y'] = yseq
data['z'] = zseq

eta = []
u = []
v = []
w = []
p = []
for t in range(tNum):
    tmp0 = []
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    for k in range(K):
        tmp00 = []
        tmp11 = []
        tmp22 = []
        tmp33 = []
        tmp44 = []
        for j in range(J):
            tmp00.append(tmp[t][k*J*I + j*I:k*J*I + (j+1)*I,-5])
            tmp11.append(tmp[t][k*J*I + j*I:k*J*I + (j+1)*I,-4])
            tmp22.append(tmp[t][k*J*I + j*I:k*J*I + (j+1)*I,-3])
            tmp33.append(tmp[t][k*J*I + j*I:k*J*I + (j+1)*I,-2])
            tmp44.append(tmp[t][k*J*I + j*I:k*J*I + (j+1)*I,-1])
        tmp0.append(np.array(tmp00))
        tmp1.append(np.array(tmp11))
        tmp2.append(np.array(tmp22))
        tmp3.append(np.array(tmp33))
        tmp4.append(np.array(tmp44))
    eta.append(np.array(tmp0))
    u.append(np.array(tmp1))
    v.append(np.array(tmp2))
    w.append(np.array(tmp3))
    p.append(np.array(tmp4))

data['eta'] = np.array(eta)
data['u'] = np.array(u)
data['v'] = np.array(v)
data['w'] = np.array(w)
data['p'] = np.array(p)


saveDir = '/scratch/HOSdata/pp/' + prjname + '/' + jobname + '/data/'
saveName = "VP_card_fitted_Dict" + suffix
fw = open(saveDir + saveName, "wb")
pickle.dump(data, fw, 2)
fw.close()
