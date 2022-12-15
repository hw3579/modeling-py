import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import sympy

A=np.array([[0,1,0,0],[-1,-1,0,1],[0,0,0,1],[0,1,0,-1]])
#print(A)
B=np.array([0,0,0,0.5])
#print(B)
C=np.array([0,0,0,0]) #initial conditions






t = np.linspace(0,0.1,10)

#create 1000 uniformply distributed points in the interval [0,10T]
force = lambda t:t<=0.1

#f=lambda X1,X2,t: [np.multiply(A,[X1[0],X1[1],X2[0],X2[1]])+np.multiply(B,force(t))]

def f(t,X1,X2):
 dx1=X1[1]
 dx11=-X1[0]-X1[1]+    X2[1]
 dx2=X2[1]
 dx22=+X1[1]+    -X2[1]-0.5*force(t)
 return [dx1,dx11,dx2,dx22]



t = np.linspace(0, 10, 100)  



XOdeint = integrate.odeint(f,[0,0,0,0],t)
#[XOdeint,X2Odeint] = integrate.odeint(f, C, t,args=(force(t),))
X2Odeint = integrate.odeint(f, C, t)

plt.plot(x, XOdeint[:, 0])  # y数组（矩阵）的第一列，（因为维度相同，plt.plot(x, y)效果相同）
plt.grid()
plt.show() 