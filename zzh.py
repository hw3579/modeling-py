import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt 

'''
def f(t,u):
    z1,z2,z3,z4=u
    dz1=z2
    dz2=-z1-z2+z4
    dz3=z4
    dz4=-z2-z4+0.5
    dzdt=[dz1,dz2,dz3,dz4]
    return dzdt
'''

A=np.array([[0,1,0,0],[-1,-1,0,1],[0,0,0,1],[0,1,0,-1]])
#print(A)
B=np.array([0,0,0,0.5])
#print(B)
C=np.array([0,0,0,0]) #initial conditions

def f(t,u):
    z1,z2,z3,z4=u
    du=A*u+B
    print(du)
    return du



t = np.linspace(0,10,100) 
#solve equation using odeint, now the values of X for t=0 are given by a list
XOdeint = integrate.odeint(f,C,t,tfirst=True)



#plot velocities from both solutions (first column of each solution)
plt.plot(t,XOdeint[:,3])
plt.legend(['Odeint'])
plt.title('Velocitie')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$\dot{x}$')
plt.show()

#Save data
#np.savetxt('t.txt',t)
#np.savetxt('x.txt',XOdeint[:,0])