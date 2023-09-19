import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
import scipy
import matplotlib.pyplot as plt
import sys
import os
cwd = os.getcwd()

def get_K_(s, x, n):
    N = len(x)-1
    K_s = 0
    if n < N-1:
        xn0 = x[n]
        xn = x[n+1]
        xn1 = x[n+2]

        a0 = 0
        a = 0
        a1 = 0

        if (s==xn0):
            a = (s-xn) * np.log(xn-s+0j)
            a1 = (s-xn1) * np.log(xn1-s+0j)
        elif (s==xn):
            a0 = (s-xn0) * np.log(xn0-s+0j)
            a1 = (s-xn1) * np.log(xn1-s+0j)
        elif (s==xn1):
            a0 = (s-xn0) * np.log(xn0-s+0j)
            a = (s-xn) * np.log(xn-s+0j)
        else:
            a0 = (s-xn0) * np.log(xn0-s+0j)
            a = (s-xn) * np.log(xn-s+0j)
            a1 = (s-xn1) * np.log(xn1-s+0j)
        
        K_s = K_s + a0/(xn0-xn) + a1/(xn-xn1) - ((xn0-xn1)*a)/((xn0-xn)*(xn-xn1)) 

        
    else:
        xN0 = x[N-1]
        xN = x[N]

        a0 = 0
        a = 0
        if (s==xN0):
            a = (s-xN) * np.log(xN-s+0j)
        elif (s==xN):
            a0 = (s-xN0) * np.log(xN0-s+0j)
        else:
            a = (s-xN) * np.log(xN-s+0j)
            a0 = (s-xN0) * np.log(xN0-s+0j)
        
        K_s = K_s + a0/(xN0-xN) + (xN/s)*np.log(xN+0j) + (xN0-xN-s)*a/(s*(xN0-xN)) + 1
        
    return K_s

def K_(x, s_range):
    '''
    if s_range == None:
        s_range = x[1:]
    '''
    M = len(s_range)
    N = len(x)-1
    K = np.zeros((M,N), dtype=complex)
    
    for m in range(M):
        s = s_range[m]
        t = 4-s
        for n in range(N):
            K[m,n] = get_K_(s, x, n) + get_K_(t, x, n)
    
    return K

def get_K_re_(s, x, n):
    N = len(x)-1
    K_s = 0
    if n < N-1:
        xn0 = x[n]
        xn = x[n+1]
        xn1 = x[n+2]

        a0 = 0
        a = 0
        a1 = 0

        if (s==xn0):
            a = (s-xn) * np.log(np.abs(xn-s))
            a1 = (s-xn1) * np.log(np.abs(xn1-s))
        elif (s==xn):
            a0 = (s-xn0) * np.log(np.abs(xn0-s))
            a1 = (s-xn1) * np.log(np.abs(xn1-s))
        elif (s==xn1):
            a0 = (s-xn0) * np.log(np.abs(xn0-s))
            a = (s-xn) * np.log(np.abs(xn-s))
        else:
            a0 = (s-xn0) * np.log(np.abs(xn0-s))
            a = (s-xn) * np.log(np.abs(xn-s))
            a1 = (s-xn1) * np.log(np.abs(xn1-s))
        
        K_s = K_s + a0/(xn0-xn) + a1/(xn-xn1) - ((xn0-xn1)*a)/((xn0-xn)*(xn-xn1)) 

        
    else:
        xN0 = x[N-1]
        xN = x[N]

        a0 = 0
        a = 0
        if (s==xN0):
            a = (s-xN) * np.log(np.abs(xN-s))
        elif (s==xN):
            a0 = (s-xN0) * np.log(np.abs(xN0-s))
        else:
            a = (s-xN) * np.log(np.abs(xN-s))
            a0 = (s-xN0) * np.log(np.abs(xN0-s))
        
        K_s = K_s + a0/(xN0-xN) + (xN/s)*np.log(xN) + (xN0-xN-s)*a/(s*(xN0-xN)) + 1
        
    return K_s


def K_Re_(x, s_range):
    '''
    if s_range == None:
        s_range = x[1:]
    '''
    M = len(s_range)
    N = len(x)-1
    K_re = np.zeros((M,N), dtype=float)
    
    for m in range(M):
        s = s_range[m]
        t = 4-s
        for n in range(N):
            K_re[m,n] = get_K_re_(s, x, n) + get_K_re_(t, x, n)
    
    return K_re
        

def Pole_(s_range, m_spec):
    m2 = m_spec**2
    J = 1/(2*m_spec * np.sqrt(4-m2))
    M = len(s_range)
    n_m = len(m_spec)

    poles = np.zeros((M,n_m), dtype = float)
    for i in range(M):
        s = s_range[i]
        t = 4-s
        poles[i,:] = -J * ( 1/(s-m2) + 1/(t-m2) )
    
    return poles


def Con_(var, K_re, poles, n_m):
    #c = np.zeros(n_con, dtype=float)
    c = np.zeros(K_re.shape[0] + 1, dtype=float)
    
    S_inf = var[-1]

    V = np.matmul(poles, var[0:n_m]) + np.matmul(K_re, var[n_m: -1])
    c[0:-1] = (S_inf + V)**2 + (np.pi * var[n_m: -1])**2 ### Assuming s_range == xgrid[1:]
    c[-1] = S_inf**2

    return c