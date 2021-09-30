import numpy
from numpy import *


def trs(M,O,alpha):
    """ 实现坐标转换 """
    '''
    输入：n行6列矩阵
    '''
    # alpha = 15.52
    alpha = 2*pi/360 * alpha

    Mtrs = array([[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]])
    # O = (431.6, 343.9, 0)

    M_ = zeros(shape(M))
    M_[:,0] = M[:,0].reshape(M_[:,0].shape) - O[0]
    M_[:,1] = M[:,1].reshape(M_[:,1].shape) - O[1]
    M_[:,2] = M[:,2].reshape(M_[:,2].shape) - O[2]
    M_[:,0:3] = dot(M_[:,0:3], Mtrs)
    M_[:,3:6] = dot(M[:,3:6], Mtrs)
    return M_
