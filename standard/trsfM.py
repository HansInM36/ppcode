import numpy
from numpy import *

alpha = -15.52
alpha = 2*pi/360 * alpha

Mtrs = array([[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])
Ocoor = (431.6, 343.9, 0)


''' 各截面处十字形probes '''
sec = -3
# 打印 sec D位置的y向probes坐标 （坐标系为风机坐标系）
y = array([[148, 164, 180, 196, 212, 228, 244, 260, 276, 284, 292, 300, 308, 316, 324, 332, 340, 346, 352, 358, 364, 370, 376, 382, 388, 394, 400,\
406, 412, 418, 424, 430, 436, 442, 448, 454, 460, 468, 476, 484, 492, 500, 508, 516, 524, 540, 556, 572, 588, 604, 620, 636, 652]]).T
y = y - 400
x = array(zeros((y.shape[0],1))) + sec*126
z = array(zeros((y.shape[0],1))) + 90.1
coor = hstack((x,y,z))
coortrs = dot(coor, Mtrs)
coortrs[:,0] = coortrs[:,0] + Ocoor[0]
coortrs[:,1] = coortrs[:,1] + Ocoor[1]
coortrs[:,2] = coortrs[:,2] + Ocoor[2]
for row in coortrs:
    print('                  ','(',round(row[0],1),'',round(row[1],1),'',round(row[2],1),')')

# 打印 sec D位置的z向probes坐标 （坐标系为风机坐标系）
z = array([[6.1, 12.1, 18.1, 24.1, 30.1, 36.1, 42.1, 48.1, 52.1, 60.1, 66.1, 72.1, 78.1, 84.1, 90.1, 96.1, 102.1, 108.1, 114.1, 120.1, 126.1, 132.1,\
138.1, 144.1, 150.1, 158.1, 166.1, 174.1, 182.1, 190.1, 198.1, 206.1, 214.1, 230.1, 246.1, 262.1, 278.1, 294.1, 310.1, 326.1, 342.1]]).T
x = array(zeros((z.shape[0],1))) + sec*126
y = array(zeros((z.shape[0],1)))
coor = hstack((x,y,z))
coortrs = dot(coor, Mtrs)
coortrs[:,0] = coortrs[:,0] + Ocoor[0]
coortrs[:,1] = coortrs[:,1] + Ocoor[1]
coortrs[:,2] = coortrs[:,2] + Ocoor[2]
for row in coortrs:
    print('                  ','(',round(row[0],1),'',round(row[1],1),'',round(row[2],1),')')


''' 轮毂中心线probes '''
x = linspace(-378, 1512, 31)
x = array([x]).T + 0.1
y = array(zeros((x.shape[0],1)))
z = array(zeros((x.shape[0],1))) + 90.1
coor = hstack((x,y,z))
coortrs = dot(coor, Mtrs)
coortrs[:,0] = coortrs[:,0] + Ocoor[0]
coortrs[:,1] = coortrs[:,1] + Ocoor[1]
coortrs[:,2] = coortrs[:,2] + Ocoor[2]
for row in coortrs:
    print('                  ','(',round(row[0],1),'',round(row[1],1),'',round(row[2],1),')')
