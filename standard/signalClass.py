import numpy as np
from numpy import *

def rms(x):
    N = x.shape[0]
    return np.power(sum(np.power(x,2))/N, 0.5)



def nrm_seq(y,x):
    """ 对序列y(x)进行归一化，即曲线面积为1 """
    '''
    121
    '''
    sum = abs(0.5*y[0]*(x[1]-x[0])) + abs(0.5*y[-1]*(x[-1]-x[-2]))
    for i in range(1,x.shape[0]-1):
        sum += abs(0.5*y[i]*(x[i+1]-x[i-1]))
    return y/sum


def drv_seq(y,x):
    """ 对序列y(x)求导 """
    '''
    中心差分法
    '''
    '''
    y = np.array([10,7,1,2,29,8,4])
    x = np.array([1,2,3,4,5,6,7])
    drv_seq(y,x)
    '''
    N = y.shape[0]
    s = np.zeros(y.shape)
    s[0] = (y[1]-y[0]) / (x[1]-x[0])
    s[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
    for i in range(1,N-1):
        s[i] = (y[i+1]-y[i-1]) / (x[i+1]-x[i-1])
    return s

def corr_seqs(x,y):
    """ 求x和y的互相关函数 """
    '''
    x, y are signal sequences in 1D array with same size
    '''
    N = len(x)
    m = array([i for i in range(0,N)])
    Rxy = array(zeros(N))
    for i in range(0,N):
        Rxy[i] = sum([x[j] * y[j+i] for j in range(0,N-i)]) / N
    return vstack((m,Rxy))



class SignalSeq:
    ''' members '''
    X = 0
    N = 0
    E_X = 0
    sigma_X = 0
    var_X = 0

    ''' constructor '''
    def __init__(self, x):
        '''
        x: signal sequence, 1D array
        fs: sample frequency
        '''
        self.X = x
        self.N = len(x)
        self.E_X = mean(x)
        self.var_X = var(x)

    ''' method '''
    def AR(self, X=None):
        '''
        X is signal sequence in 1D array
        '''
        if X is None: # because we can't define the function like this def autocor(self, X=self.X):
            X = self.X
        N = len(X)
        m = array([i for i in range(0,N)])
        RX = array(zeros(N))
        for i in range(0,N):
            RX[i] = sum([X[j] * X[j+i] for j in range(0,N-i)]) / N
        return vstack((m,RX))


    def PSE_t_AM(self, fs, X=None): # use autocorrelation function method for temporal power spectrum estimation
        """ Power Spectra Estimation """
        '''
        X is signal sequence in 1D array
        fs is sample frequency
        '''
        if X is None:
            X = self.X
        N = len(X)
        f = linspace(0,fs,N) # the frequency axis
        f = f[range(int(N/2))]
        ps = fft.fft(self.AR(X)[1,:]) # use FFT algorithm to do the DTFT
        ps = abs(ps)
        ps = ps[range(int(N/2))]
        return vstack((f,ps))

    def PSE_s_AM(self, wl, X=None): # use autocorrelation function method for spatial power spectrum estimation
        '''
        X is signal sequence in 1D array
        wl is a tuple containing the smallest and the largest wave length (i.e. the smallest mesh length and the 1D domain length)
        '''
        if X is None:
            X = self.X
        N = len(X)
        L = max(wl)
        delta = min(wl)
        l = linspace(delta, L, N)
        k = array([2*pi/i for i in l[range(int(N/2))]])
        k.sort()
        ps = fft.fft(self.AR(X)[1,:])
        ps = abs(ps)
        ps = ps[range(int(N/2))]
        return vstack((k,ps))

    def SE_s(self, delta, X=None):
        if X is None:
            X = self.X
        N = len(X)
        kmax = 2*pi/delta/2
        k = linspace(0, kmax, round(N/2))
        ps = fft.fft(X)
        ps = abs(ps)
        ps = ps[range(round(N/2))]
        return vstack((k,ps))


class SignalMat:
    ''' members '''
    Mat = 0
    Size = 0
    E_X = 0
    sigma_X = 0
    var_X = 0

    ''' constructor '''
    def __init__(self, mat):
        '''
        x: signal sequence, 1D array
        fs: sample frequency
        '''
        self.Mat = mat
        self.Size = shape(mat)

    ''' method '''
    def AR(self, Mat=None):
        '''
        Mat is signal matrix in 2D array
        '''
        if Mat is None: # because we can't define the function like this def autocor(self, X=self.X):
            Mat = self.Mat
        Size = shape(Mat)
        Rmn = zeros((Size[0],Size[1]))
        for i in range(0,Size[0]):
            for j in range(0,Size[1]):
                for p in range(0,Size[0]-i):
                    for q in range(0,Size[1]-j):
                        Rmn[i,j] = Rmn[i,j] + Mat[i,j]*Mat[i+p,j+q]
        Rmn = Rmn / (Size[0] * Size[1])
        return Rmn

    def PSE_2D_AM(self, wl2D, Mat=None): # use autocorrelation function method for spatial power spectrum estimation
        '''
        Mat is signal matrix in 2D array
        wl2D is a 2*2 array containing the smallest and the largest wave length (i.e. the smallest mesh length and the 1D domain length) in x and y dimension
        '''
        if Mat is None:
            Mat = self.Mat
        Size = shape(Mat)
        Lx = max(wl2D[0,:])
        deltax = min(wl2D[0,:])
        Ly = max(wl2D[1,:])
        deltay = min(wl2D[1,:])
        lx = linspace(deltax, Lx, Size[0])
        ly = linspace(deltay, Ly, Size[1])
        kx = array([2*pi/i for i in lx[range(int(Size[0]/2))]])
        ky = array([2*pi/j for j in ly[range(int(Size[1]/2))]])
        kx.sort()
        ky.sort()
        ps = fft.fft2(self.AR(Mat))
        ps = abs(ps)
        return ps
