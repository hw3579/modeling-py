import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt




h=float(input("step size:"))   #define the step size
#h=0.01 
print("h=")
print(h)


def Euler(f1,f2,f3,f4,x): #function for Euler method
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


def heun(f1,f2,f3,f4,x): #function for Heun method
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



def ode(t,u): #odeint - the exact solution
    z1,z2,z3,z4=u
    dz1=z2
    dz2=-z1-z2+z4
    dz3=z4
    dz4=-z2-z4+0.5
    dzdt=[dz1,dz2,dz3,dz4]
    return dzdt



x=np.arange(0,20,h) # each of the time point

f1=lambda z2,x: z2
f2=lambda z1,z2,z4,x: -z1- z2+   z4
f3=lambda z4,x:            z4
f4=lambda z2,z4,x:-z2-z4

[z1,z2,z3,z4]= Euler(f1,f2,f3,f4,x)  # z is the Euler method solution
[w1,w2,w3,w4]= heun(f1,f2,f3,f4,x)   # w is the Heun method solution
Exact = integrate.odeint(ode,[0,0,0,0],x,tfirst=True) # exact solution


exact=np.zeros(len(x))     # Extract the third variables - X2(t)
for i in range(len(x)-1):
 exact[i]=Exact[i][3]
 i+=1


Error_euler=np.zeros(len(x))   #error array initialize
Error_Heun=np.zeros(len(x))

def error(y,s):  # caclulate the error 
   result=abs((y-s)/s)*100
   return result

for i in range(len(x)-1):  #save the error and print the result
 Error_euler[i]=error(z3[i],exact[i])
 Error_Heun[i]=error(w3[i],exact[i])
 print(x[i],"\t",exact[i],"\t",z3[i],"\t ",error(z3[i],exact[i]),"\t",w3[i],error(w3[i],exact[i]),"\t",end="\t")
 print()





# plot the simulation
plt.plot(x,z3,'--',label='Euler')
plt.plot(x,w3,label='Heun')
plt.plot(x,exact,'r-',label='exact')
plt.legend()
plt.title('Displacement')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig('./1.jpg')
plt.show()


#plot the error
plt.plot(x,Error_euler,label='Euler')
plt.plot(x,Error_Heun,label='Heun')
plt.legend()
plt.title('Error in this step size')
plt.xlabel('$t$')
plt.ylabel('$Error%$')
plt.grid()
plt.savefig('./2.jpg')
plt.show()

