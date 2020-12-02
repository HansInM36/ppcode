import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

n=64
L=30
dx=L/n

x=np.arange(-L/2,L/2,dx,dtype="complex_")

#Create a Function
f=np.cos(x) * np.exp(-np.power(x,2)/25)
#Analytically obtain the Derivative

df= -(np.sin(x) * np.exp(-np.power(x,2)/25 + (2/25)*x*f))  #Derivative


##Approximate derivative by FDM
#Create a numpy array
#Discretizes the function into finite regions
dfFD=np.zeros(len(df), dtype='complex_')
#Iterate across array
for kappa in range(len(df)-1):
    dfFD[kappa]=(f[kappa+1]-f[kappa])/dx

dfFD[-1]=dfFD[-2]

##Approximate derivative by FFT
fhat = np.fft.fft(f)
kappa = (2*np.pi/L)*np.arange(-n/2,n/2)
#Re-order fft frequencies
kappa = np.fft.fftshift(kappa)
#Obtain real part of the function for plotting
dfhat = kappa*fhat*(1j)
#Inverse Fourier Transform
dfFFT = np.real(np.fft.ifft(dfhat))


##Plot results
plt.plot(x, df.real, color='k', LineWidth=2, label='True Derivative')
plt.plot(x, dfFD.real, '--', color='b', LineWidth=1.5, label='Finite Difference')
plt.plot(x, dfFFT.real, '--', color='c', LineWidth=1.5, label='Spectral Derivative')

y = f

N = y.size
L = dx * (N-1)
Y = np.fft.fft(y)
omg = 2*np.pi/L * np.arange(-n/2, n/2)
omg = np.fft.fftshift(omg)
dyFFT = (1j) * omg * Y
dy = np.fft.ifft(dyFFT)

plt.plot(x, dy, ':', color='g', LineWidth=1.5, label='self')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()
