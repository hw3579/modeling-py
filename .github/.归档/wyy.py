from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

#Euler method
x1E_list = [] #use list[] to expand the scope of index
x2E_list = []
t_list = []

x1E = 0  #define system variables for using Euler method. Start from 0.
x2E = 0
z1E = 0 
z2E = 0 
dt = 0.001 #choose h=Δt=0.001
 #define time range 

#Using if statement to define system input:f(t)
def Force(t):
    if 0<t <=0.002:
        return 1
    else:
        return 0
    
#Using for statement to finish superposition process
for i in range(int(60/dt)):
    t = i*dt  
    x1E += dt*(z1E) #For ODE,dx1/dt=z1
    x2E += dt*(z2E) #dx2/dt=z2
    z1E += dt*(-z1E-1/2-x1E+z2E) #dz1/dt Simplify the given Eq(6),(7)
    z2E += dt*(Force(t)/2+z1E-z2E) #dz2/dt 
    x1E_list.append(x1E) #Append data in case of codes insufficient capacity 
    x2E_list.append(x2E)
    t_list.append(t)

#plt.plot(t_list,x1E_list,color ='g')
plt.plot(t_list,x2E_list,color ='r')

plt.title('Euler method simulation result')
plt.xlabel('t[s]')
plt.ylabel('output position x2(t)')

plt.show()

Error_x2E = abs(1/2-abs(x2E_list[-1])) #1/2 is FVT result,use it to calculate errors
Error_x2EP = Error_x2E/abs(x2E_list[-1])*100 #calculate percentage error, the result is 0.36%
print("the Euler method error percentage is",Error_x2EP)
#Heun method
x1H_list = [] #use list[] to expand the scope of index
x2H_list = []
t_list = []
dt = 0.01 #choose h=Δt=0.01
x1H = 0 #define system variables for using Heun method. Start from 0.
x2H = 0
z1H = 0 
z2H = 0 
k1_x1 = 0  #define k1^x1 (Heun method)
k2_z1=  0  #define k1^z1 (Heun method)
k1_x2 = 0  #define k1^x2 (Heun method)
k2_z2 = 0  #define k1^z2 (Heun method)
#Using if statement to define system input:f(t)
def Force(t):
    if  0<t<=1:
        return 1
    else:
        return 0

for i in range(int(60/dt)):
    t = i*dt
    k1_x1 = z1H #Using Heun, k1^x1=f1(t,x1,z1)
    k2_z1 += dt*(z1H) #k2^z1=f2(t,z1,z2)
    z1H += dt*(-z1H-k2_z1+z2H)
    x1H += dt*(1/2)*(k1_x1+z1H)
    
    k1_x2 = z2H
    k2_z2 += dt*(z2H)
    z2H += dt*(Force((t+dt))/2+z1H-z2H)
    x2H += dt*(1/2)*(k1_x2+z2H)
    
    x1H_list.append(x1H)
    x2H_list.append(x2H)
    t_list.append(t)  
  
#plt.plot(t_list,x1H_list,color ='b')
plt.plot(t_list,x2H_list,color ='orange')

plt.xlabel('t[s]')
plt.ylabel('output position x2(t)')
plt.title('Heun method simulation result')

plt.show()

Error_x2H = abs(1/2-abs(x2H_list[-1])) #1/2 is FVT result,use it to calculate errors
Error_x2HP = Error_x2E/abs(x2H_list[-1])*100 #calculate percentage error, the result is 0.36%
print("the Heun method error percentage is",Error_x2HP)