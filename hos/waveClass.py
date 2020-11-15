import sys
sys.path.append('/scratch/ppcode/HOS')
import mathtool
from mathtool import *

def count_wave(x, y):
    '''
    # function
    count how many waves in a surface elevation sequences (spatial or temporal), output all zero-up-crossing positions, maximum and minimum of surface elevations of each wave
    # input
    x: sequence of temporal or spatial positions
    y: sequence of surface elevation
    # please garantee that there are enough sample points in each wave
    '''

    N = len(y)

    waveNum = 0
    zcp = []

    for i in range(N):
        if i > 0:
            if y[i-1] * y[i] <= 0 and y[i] > 0:
                p_zero = calc_zero_crossing([x[i-1], x[i]], [y[i-1], y[i]])
                zcp.append(p_zero)
                waveNum = waveNum + 1

    waveNum = waveNum - 1

    SEmin = []
    p_SEmin = []
    SEmax = []
    p_SEmax = []

    for i in range(N):
        if i > 1:
            d0 = y[i-1] - y[i-2]
            d1 = y[i] - y[i-1]
            if d0 * d1 < 0:
                if d1 < 0: # means this is a maximum
                    h, k = calc_parabola_vertex([x[i-2], x[i-1], x[i]], [y[i-2], y[i-1], y[i]])
                    SEmax.append(k)
                    p_SEmax.append(h)
                else: # means this is a minimum
                    h, k = calc_parabola_vertex([x[i-2], x[i-1], x[i]], [y[i-2], y[i-1], y[i]])
                    SEmin.append(k)
                    p_SEmin.append(h)
            elif d0 * d1 == 0: # this function suppose that only one of d0, d1 is zero, because if both are aero, that means the sample points for the wave are not enough
                y[i-1] = y[i-1] + 1e-6 # avoid that this extremum is counted twice
                if d0 == 0:
                    if d1 < 0:
                        h, k = calc_parabola_vertex([x[i-2], x[i-1], x[i]], [y[i-2], y[i-1], y[i]])
                        SEmax.append(k)
                        p_SEmax.append(h)
                    else:
                        h, k = calc_parabola_vertex([x[i-2], x[i-1], x[i]], [y[i-2], y[i-1], y[i]])
                        SEmin.append(k)
                        p_SEmin.append(h)
                if d1 == 0:
                    if d0 > 0:
                        h, k = calc_parabola_vertex([x[i-2], x[i-1], x[i]], [y[i-2], y[i-1], y[i]])
                        SEmax.append(k)
                        p_SEmax.append(h)
                    else:
                        h, k = calc_parabola_vertex([x[i-2], x[i-1], x[i]], [y[i-2], y[i-1], y[i]])
                        SEmin.append(k)
                        p_SEmin.append(h)

    waveL = [zcp[i+1] - zcp[i] for i in range(waveNum)]

    waveMax = [0 for i in range(waveNum)]
    waveMin = [0 for i in range(waveNum)]

    for i in range(len(p_SEmax)):
        if p_SEmax[i] > zcp[0] and p_SEmax[i] < zcp[-1]:
            tmp = np.array(zcp) - p_SEmax[i]
            tmp_ind = np.where(tmp < 0)
            ind = max(tmp_ind[0])
            waveMax[ind] = SEmax[i]
    for i in range(len(p_SEmin)):
        if p_SEmin[i] > zcp[0] and p_SEmin[i] < zcp[-1]:
            tmp = np.array(zcp) - p_SEmin[i]
            tmp_ind = np.where(tmp < 0)
            ind = max(tmp_ind[0])
            waveMin[ind] = SEmin[i]

    waveH = [waveMax[i] - waveMin[i] for i in range(waveNum)]

    return waveL, waveH, waveMax, waveMin

def k_to_omega(k, d):
    ''' dispersion relation '''
    return np.sqrt(9.81 * k * np.tanh(k * d))


def omega_to_k(omega, d, k0=np.pi, tol=1e-6, maxiter=1000):
    ''' dispersion relation '''
    g = 9.81
    k = k0
    f = lambda k: np.power(omega,2) - g * k * np.tanh(k*d)
    df = lambda k: - (g * np.tanh(k*d) + g*k*d * (np.power(np.cosh(k*d),2) - np.power(np.sinh(k*d),2)) / np.power(np.cosh(k*d),2))
    for i in range(maxiter):
        knew = k - f(k)/df(k)
        if abs(knew - k) < tol: break
        k = knew
    return knew, i
