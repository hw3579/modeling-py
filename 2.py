import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import sympy

A=np.array([[0,1,0,0],[-1,-1,0,1],[0,0,0,1],[0,1,0,-1]])
print(A)
B=np.array([0,0,0,0.5])
print(B)
C=np.array([0,0,0,0]) #initial conditions






t = np.linspace(0,0.1,10)

#create 1000 uniformply distributed points in the interval [0,10T]
force = lambda t:t<=0.1

#f1=lambda X1,X2,t: [X1[1],-X1[0]-X1[1]+    X2[1]-0*force(t)]

'''
def f1(t,X1,X2):
 Z=np.multiply(A,[X1[0],X1[1],X2[0],X2[1]])

 return Z

def f1(X1,X2):
 dx1=X1[1]
 dx2=-X1[0]-X1[1]+    X2[1]
 return [dx1,dx2]

def f2(t,X21,X1):
 dx1=X21
 dx2=+X1[1]+    -X21-0.5*force(t)
 return [dx1,dx2]

'''

t = np.linspace(0, 10, 100)  
#f2=lambda X1,X2,t: [X2[1],      +X1[1]+    -X2[1]-0.5*force(t)]



f1=lambda X1,X21: [X1[1],-X1[0]-X1[1]+    X21]
f2=lambda X1,X21,t: [X21,      +X1[1]+    -X21-0.5*force(t)]

#X2 = np.linspace(0, 10, 100)  
XOdeint = integrate.odeint(f1,[0,0],t)
#X2Odeint = integrate.odeint(f2,[0,0], t,args=(XOdeint,))

plt.plot(t, XOdeint[:, 0])  # y数组（矩阵）的第一列，（因为维度相同，plt.plot(x, y)效果相同）
plt.grid()
plt.show() 