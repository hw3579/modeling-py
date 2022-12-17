# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:31:04 2022

@author: 25358
"""

#import numpy, integrate and pyplot
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


from scipy.integrate import odeint
from scipy.integrate import solve_ivp

h=0.0001

# each of the time point
x=np.arange(0,50,h) 

# Define the impulse force
def impulse(t):  
     time=0  
     if time<1:
        time+=h
        return 1
     else:
        time+=h
        return 0


def Euler(Z,x): #function for Euler method ###################
    Z = np.zeros((len(x), 4))
    Z3=np.zeros(len(x))
    for i in range(len(x)-1):
        Z[i+1]=h*dSdt(x[i], Z[i])+Z[i]
        Z3[i]=Z[i][3]
    
    return Z3



def dSdt(x, S):
    x1,v1,x2,v2=S
    return np.array([v1,
            -x1-v1+v2-0.5,
            +v2,
            +v1-v2+0.5*impulse(x)])

S_0 = np.array([0,0,0,0])




Z = np.zeros((len(x), 4))
yEuler= Euler(Z,x)





F = lambda W,t: [W[1],W[3] - W[1] - W[0],W[3],impulse(x)/2 + W[1] - W[3]]##############
sol = odeint(F, S_0,x)


plt.plot(x,sol.T[0])
plt.plot(x,yEuler,'--',label='Euler')
plt.show()

   