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

h=0.01

# each of the time point
x=np.arange(0,50,h) 

#Define the Eulerian equation function with respect to the differential equation
def Euler(f,y0,x):
    
    #Define the step size
    h=x[1]-x[0]

    #define array of zeros for y1,y2,y3,y4
    y=np.zeros(len(x))
   
   #Define the value of the input signal
    y[0]=y0
    
    #Use loops for building Euler formula simulations
    for i in range(len(x)-1):
        
        #Using Euler's formula for every Y value
        y[i+1]=y[i]+h*f(y[i],x[i])   
      
    #Returns the value y1,y2,y3,y4        
    return y





# Define the impulse force
def impulse(t):     
    if t<1:
        return 1
    else:
        return 0


t = np.linspace(0,50,100)
def dSdt(t, S):
    x1,v1,x2,v2=S
    return [v1,
            -x1-v1+v2-0.5,
            +v2,
            +v1-v2+0.5*impulse(t)]
x1_0 = 0
v1_0 = 0
x2_0 = 0
v2_0 = 0

S_0 = (x1_0,v1_0,x1_0,v1_0)


yEuler= Euler(dSdt,0,x)
sol = odeint(dSdt, y0=S_0, t=t, tfirst=True)


plt.plot(t,sol.T[0])
plt.plot(x,yEuler,'--',label='Euler')
   