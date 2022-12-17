'''
# Course: Modeling of Engineering
# Author: Jiaqi Yao
# Date: Dec 16, 2022
'''

# Import modules and library functions
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

'''
Step 1: input the step size h
'''
#h=float(input("step size:"))   # input step size manually
h=0.01                          # choose one method to set the step size
print("h=")                     # print the step size
print(h)


'''
Step 2: Defind basic variables and functions
'''

x=np.arange(0,45,h) # Selecting time sampling points, from 0 to 45 in each h


def impulse(t):     # Define the impulse force
    if t<1:
        return 1
    else:
        return 0


def state_matrix(x,Z): # Define the state space matrix
 dz1=Z[1]
 dz2=-Z[0]-Z[1]+Z[3]    # There is a -0.5 be ignored because the constant disappears after differentiation.
 dz3=Z[3]
 dz4=Z[1]-Z[3]+0.5*impulse(x)

 return np.array([dz1,dz2,dz3,dz4])




def Euler(Z,x):     # function for Euler method
    Z = np.zeros((len(x), 4))   # Defining two-dimensional arrays Z: 
    Z3=np.zeros(len(x))         # Defining one-dimensional arrays Z3 for extract the X2(t)
    for i in range(len(x)-1):
        Z[i+1]=h*state_matrix(x[i], Z[i])+Z[i] # f(x)=f(x,y)+h*f'(x,y)
        Z3[i]=Z[i][2]      # Extract the X2(t)  
    Z3[-1]=Z3[-2] # The last one is zero and  make it equal to the second last one, in order to make figure more beautiful
    return Z3   # Return a one-dimensional arrays



# This is the old version of function by using one-dimensional arrays
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

def heun(Z,x):      #  function for Heun method
    Z = np.zeros((len(x), 4))
    Z3=np.zeros(len(x))
    for i in range(len(x)-1):
        Z[i+1]=Z[i]+h*(0.5*state_matrix(x[i], Z[i])+0.5*state_matrix(x[i]+h, Z[i]+h*state_matrix(x[i], Z[i]))) 
        #k1=f(x,y)
        #k2=f(x+h,y+k1*h)
        # f(x+1)=f(x,y)+h*(0.5*k1+0.5*k2)
        Z3[i]=Z[i][2]
    Z3[-1]=Z3[-2] 
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


# Exact solution by using odeint
F = lambda Z,t: [Z[1],Z[3] - Z[1] - Z[0],Z[3],impulse(t)/2 + Z[1] - Z[3]] 
# Defined the function
# There is a -0.5 be ignored because the constant disappears after differentiation.


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



'''
Step 3: Solve the function 
'''

Eu_result= Euler(state_matrix,x)  # z is the Euler method solution
He_result= heun(state_matrix,x)   # w is the Heun method solution
Exact = integrate.odeint(F,[0,0,0,0],x) #solve the function

exact=np.zeros(len(x))     # Extract the third variables - X2(t)
for i in range(len(x)-1):
 exact[i]=Exact[i][2]
 i+=1

'''
Step4: error analysis
'''

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





#1/2 is FVT result,use it to calculate errors
#calculate percentage error to choose appropriate end time
Error_FVT_Euler = abs(1/2-abs(Eu_result[-1]))
Error_FVT_Euler_percentage=Error_FVT_Euler/0.5*100
print("the final Euler method error is",Error_FVT_Euler)
print("the final Euler method error percentage is",Error_FVT_Euler_percentage)


Error_FVT_Heun = abs(1/2-abs(He_result[-1]))
Error_FVT_Heun_percentage=Error_FVT_Heun/0.5*100
print("the final Heun method error is",Error_FVT_Heun)
print("the final Heun method error percentage is",Error_FVT_Heun_percentage)


'''
Step5 : Plot the figure
'''


# plot the simulation
plt.plot(x,Eu_result,'--',label='Euler')
plt.plot(x,He_result,label='Heun')
exact[-1]=exact[-2]
plt.plot(x,exact,'r',label='exact')
plt.legend()
plt.title('Displacement')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$x_2(t)$')
plt.savefig('./1.jpg')
plt.savefig('1.eps',dpi=600,format='eps')
plt.show()


#plot the error
plt.plot(x,Error_euler,label='Euler')
plt.plot(x,Error_Heun,label='Heun')
plt.ylim((0, 5))
plt.yticks(np.arange(0, 5, 1))
plt.legend()
plt.title('Error in this step size')
plt.xlabel('$t$')
plt.ylabel('$Error\%$')
plt.grid()
plt.savefig('./2.jpg')
plt.savefig('2.eps',dpi=600,format='eps')
plt.show()

