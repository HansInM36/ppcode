import numpy as np


def z0(Hs,cp,us):
    return 3.35*Hs*np.power(us/cp,3.4)

Hs, cp, us = 3.92*2, 19.1, 0.1

z0(Hs,cp,us)
