import numpy as np
import matplotlib.pyplot as plt



def eluer(fun,tspan,y0,dt):
    M = tspan/dt
    M = int(M)
    y = np.zeros([4,M+1])
    t=np.linspace(0,tspan,num=M+1)
    y[:,0]=y0.T
    i=1
    while i<=M:
        x=fun(t[i-1],y[:,i-1])
        x=x.reshape((4,))
        y[:,i]=y[:,i-1]+dt*x
        i=i+1
    t=t[0:i-1]
    y=y[:,0:i-1]
    return t,y

def odefcn(t,y):
    u=np.zeros([4,1])
    u[0]=y[1]
    u[1]=y[3]-y[1]-y[0]-0.5
    u[2]=y[3]
    u[3]=-y[3]+y[1]
    return u


tmax=50
dt=tmax/1000
y0=np.zeros([4,1])

t,y=eluer(odefcn,tmax,y0,dt)
y1=y[0,:]
v1=y[1,:]
y2=y[2,:]
v2=y[3,:]
u1=abs(y1[len(y1)-1])
e1=0.5-u1
e2=(e1/0.5)*100
print(e1,e2)

plt.plot(t,y2)
plt.legend('x2(t)')
plt.title('euler')
plt.show()