import numpy as np
import math


def calc_parabola_vertex(x, y):
    '''
    # function
    calculate the vertex y value of a parabola defined by three points, this function can be used to estimate the maximum of surface elevation.
    # input arguments
    x: a tuple/list/1D-array of three elements, e.g. (x0,x1,x2)
    y: a tuple/list/1D-array of three elements, e.g. (y0,y1,y2)
    '''

    denom = (x[0]-x[1]) * (x[0]-x[2]) * (x[1]-x[2]);
    A     = (x[2] * (y[1]-y[0]) + x[1] * (y[0]-y[2]) + x[0] * (y[2]-y[1])) / denom;
    B     = (x[2]*x[2] * (y[0]-y[1]) + x[1]*x[1] * (y[2]-y[0]) + x[0]*x[0] * (y[1]-y[2])) / denom;
    C     = (x[1] * x[2] * (x[1]-x[2]) * y[0]+x[2] * x[0] * (x[2]-x[0]) * y[1]+x[0] * x[1] * (x[0]-x[1]) * y[2]) / denom;

    h = - B / 2 / A
    k = (4*A*C - np.power(B,2)) / 4 / A

    return (h, k)

def calc_zero_crossing(x ,y):
    '''
    # function
    calculate the position of zero-crossing point.
    # input arguments
    x: a tuple/list/1D-array of two elements, e.g. (x0,x1)
    y: a tuple/list/1D-array of two elements, e.g. (y0,y1)
    '''
    return (x[1]*y[0] - x[0]*y[1]) / (y[0] - y[1])
