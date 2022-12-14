import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
    #sim data
m=2 #mass of the oscillator
b=0.2 #damper of the oscillator
k=1 #stiffness of the oscillator
omega = (k/m)**0.5 #angular frequency of the oscillator
T=2#*np.pi/omega #period of the oscillator
#create 1000 uniformply distributed points in the interval [0,10T]
t = np.linspace(0,10*T,1000)
#lambda returning the force
force = lambda t:t<=0.1
#lambda with the RHS of the equation, it evaluates the force and returns a list
f = lambda X,t: [X[1],1/m*(force(t)-k*X[0]-b*X[1])]
#solve equation using odeint, now the values of X for t=0 are given by a list
XOdeint = integrate.odeint(f,[0,0],t)


def Euler(f,y0,x):
    h=x[1]-x[0]
    y=np.zeros(len(x))
    y[0]=0
    for i in range(len(x)-1):
        y[i+1]=y[i]+h*f(y[i],x[i])
    return y

X2Odeint= Euler(f,0,t)



ti = np.linspace(0,10*T,1000)
#plot displacements from both solutions (first column of each solution)
plt.plot(t,X2Odeint[:,0])
plt.legend(['Odeint'])
plt.title('Displacement')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.show()