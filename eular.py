import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


def Euler(f1,f2,f3,f4,x): #function, step length
    h=x[1]-x[0]
    y=np.zeros(len(x))
    z1=np.zeros(len(x))
    z2=np.zeros(len(x))
    z3=np.zeros(len(x))
    z4=np.zeros(len(x))
    z4[0]=float('0.5') #the act force
    y[0]=0
    for i in range(len(x)-1):
        z1[i+1]=z1[i]+h*f1(z2[i],x[i])
        z2[i+1]=z2[i]+h*f2(z1[i],z2[i],z4[i],x[i])
        z3[i+1]=z3[i]+h*f3(z4[i],x[i])
        z4[i+1]=z4[i]+h*f4(z2[i],z4[i],x[i])
    return [z1,z2,z3,z4]


def heun(f1,f2,f3,f4,x):
    h=x[1]-x[0]
    y=np.zeros(len(x))
    z1=np.zeros(len(x))
    z2=np.zeros(len(x))
    z3=np.zeros(len(x))
    z4=np.zeros(len(x))
    z4[0]=float('0.5') #the act force
    y[0]=0
    for i in range(len(x)-1):
        z1[i+1]=z1[i]+h*(0.5*f1(z2[i],x[i])+0.5*f1(z2[i]+h,x[i]+h*(f1(z2[i],x[i]))))
        z2[i+1]=z2[i]+h*(0.5*f2(z1[i],z2[i],z4[i],x[i])+0.5*f2(z1[i]+h,z2[i]+h,z4[i]+h,x[i]+h*(f2(z1[i],z2[i],z4[i],x[i]))))
        z3[i+1]=z3[i]+h*(0.5*f3(z4[i],x[i])+0.5*f3(z4[i]+h,x[i]+h*(f3(z4[i],x[i]))))
        z4[i+1]=z4[i]+h*(0.5*f4(z2[i],z4[i],x[i])+0.5*f4(z2[i]+h,z4[i]+h,x[i]+h*(f4(z2[i],z4[i],x[i]))))
    return [z1,z2,z3,z4]



h=0.005 #step length


x=np.arange(0,20,h)

f1=lambda z2,x: z2
f2=lambda z1,z2,z4,x: -z1- z2+   z4
f3=lambda z4,x:            z4
f4=lambda z2,z4,x:-z2-z4

[z1,z2,z3,z4]= Euler(f1,f2,f3,f4,x)
[w1,w2,w3,w4]= heun(f1,f2,f3,f4,x)
yExact=x

#plt.plot(x,yExact)
plt.plot(x,z3,'--')
plt.plot(x,w3)
plt.fill_between(x, z3, w3, facecolor="gray")
#plt.plot(x,z1)
plt.legend(['Euler'],['Heun'])
plt.title('Displacement')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.show()
