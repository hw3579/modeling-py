import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt




#h=float(input("step size:"))   #define the step size
h=0.1 #for test
print("h=")
print(h)

x=np.arange(0,45,h) # each of the time point


def impulse(t):
    if t<1:
        return 1
    else:
        return 0

def Euler(Z,x): #function for Euler method
    Z = np.zeros((len(x), 4))
    Z3=np.zeros(len(x))
    for i in range(len(x)-1):
        Z[i+1]=h*state_matrix(x[i], Z[i])+Z[i]
        Z3[i]=Z[i][2]
    
    return Z3

'''
    for i in range(len(x)-1):
        z1[i+1]=z1[i]+h*f1(z2[i],x[i])   
        z2[i+1]=z2[i]+h*f2(z1[i],z2[i],z4[i],x[i])
        z3[i+1]=z3[i]+h*f3(z4[i],x[i])
        z4[i+1]=z4[i]+h*f4(z2[i],z4[i],x[i])
#        test=x[i]
#       test1=impulse(x[i])
    return [z1,z2,z3,z4]
'''

def heun(Z,x): #function for Heun method
    Z = np.zeros((len(x), 4))
    Z3=np.zeros(len(x))
    for i in range(len(x)-1):
        Z[i+1]=Z[i]+h*(0.5*state_matrix(x[i], Z[i])+0.5*state_matrix(x[i]+h, Z[i]+h*state_matrix(x[i], Z[i])))
        Z3[i]=Z[i][2]
    
    return Z3

'''
    h=x[1]-x[0]
    y=np.zeros(len(x))
    z1=np.zeros(len(x))
    z2=np.zeros(len(x))
    z3=np.zeros(len(x))
    z4=np.zeros(len(x))
    z4[0]=z4[1]=0.5 #the act force
    y[0]=0
    for i in range(len(x)-1):
        z1[i+1]=z1[i]+h*(0.5*f1(z2[i],x[i])+0.5*f1(z2[i]+h,x[i]+h*(f1(z2[i],x[i]))))
        z2[i+1]=z2[i]+h*(0.5*f2(z1[i],z2[i],z4[i],x[i])+0.5*f2(z1[i]+h,z2[i]+h,z4[i]+h,x[i]+h*(f2(z1[i],z2[i],z4[i],x[i]))))
        z3[i+1]=z3[i]+h*(0.5*f3(z4[i],x[i])+0.5*f3(z4[i]+h,x[i]+h*(f3(z4[i],x[i]))))
        z4[i+1]=z4[i]+h*(0.5*f4(z2[i],z4[i],x[i])+0.5*f4(z2[i]+h,z4[i]+h,x[i]+h*(f4(z2[i],z4[i],x[i]))))
    return [z1,z2,z3,z4]
'''


def state_matrix(x,Z):
 dz1=Z[1]
 dz2=-Z[0]-Z[1]+Z[3]
 dz3=Z[3]
 dz4=Z[1]-Z[3]+0.5*impulse(x)

 return np.array([dz1,dz2,dz3,dz4])


'''
f1=lambda z2,x: z2
f2=lambda z1,z2,z4,x: -z1- z2+   z4 -0.5  #f2=lambda z1,z2,z4,x: -z1- 0.5*z2**2+   z4
f3=lambda z4,x:            z4
f4=lambda z2,z4,x:z2-z4


def ode(t,u): #odeint - the exact solution
    z1,z2,z3,z4=u
    dz1=z2
    dz2=-z1-z2+z4-0.5
    dz3=z4
    dz4=z2-z4


    dzdt=[dz1,dz2,dz3,dz4]
    return dzdt

'''

F = lambda Z,t: [Z[1],Z[3] - Z[1] - Z[0],Z[3],impulse(t)/2 + Z[1] - Z[3]]
Eu_result= Euler(state_matrix,x)  # z is the Euler method solution
He_result= heun(state_matrix,x)   # w is the Heun method solution
Exact = integrate.odeint(F,[0,0,0,0],x) # exact solution


exact=np.zeros(len(x))     # Extract the third variables - X2(t)
for i in range(len(x)-1):
 exact[i]=Exact[i][2]
 i+=1


Error_euler=np.zeros(len(x))   #error array initialize
Error_Heun=np.zeros(len(x))

def error(y,s):  # caclulate the error 
   result=abs((y-s)/s*100)
   return result

for i in range(len(x)-1):  #save the error and print the result
 Error_euler[i]=error(Eu_result[i],exact[i])
 Error_Heun[i]=error(He_result[i],exact[i])
 print(x[i],"\t",exact[i],"\t",Eu_result[i],"\t ",error(Eu_result[i],exact[i]),"\t",He_result[i],error(He_result[i],exact[i]),"\t",end="\t")
 print()

# plot the simulation
plt.plot(x,Eu_result,'--',label='Euler')
plt.plot(x,He_result,label='Heun')
plt.plot(x,exact,'r',label='exact')
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
plt.ylabel('$Error\%$')
plt.grid()
plt.savefig('./2.jpg')
plt.show()

