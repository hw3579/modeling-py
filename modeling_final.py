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
 dz1=Z[1]              # Z1_dot=Z2
 dz2=-Z[0]-Z[1]+Z[3]   # Z2_dot=-Z1-Z2+Z4-0.5
 # There is a -0.5 be ignored because the constant disappears after differentiation and  ignoring the linearisation point
 dz3=Z[3]              # Z3_dot=Z4
 dz4=Z[1]-Z[3]+0.5*impulse(x) #Z4_dot=Z2-Z4+0.5*f(t)

 return np.array([dz1,dz2,dz3,dz4])  # return the expression of the function


def Euler(Z,x):     # function for Euler method
    Z = np.zeros((len(x), 4))   # Defining two-dimensional arrays Z
    Z3=np.zeros(len(x))         # Defining one-dimensional arrays Z3 for extracting the X2(t)
    for i in range(len(x)-1):
        Z[i+1]=h*state_matrix(x[i], Z[i])+Z[i] # f(x)=f(x,y)+h*f'(x,y)
        Z3[i+1]=Z[i+1][2]      # Extract the X2(t) 
        print("Caculating Euler method ! Progress: %f %%" % (i*100/(len(x)-1),)) # Print the progress
    return Z3   # Return a one-dimensional arrays


def heun(Z,x):      #  function for Heun method
    Z = np.zeros((len(x), 4))   # Defining two-dimensional arrays Z
    Z3=np.zeros(len(x))         # Defining one-dimensional arrays Z3 for extracting the X2(t)
    for i in range(len(x)-1):
        Z[i+1]=Z[i]+h*(0.5*state_matrix(x[i], Z[i])+0.5*state_matrix(x[i]+h, Z[i]+h*state_matrix(x[i], Z[i]))) 
        #k1=f(x,y)
        #k2=f(x+h,y+k1*h)
        # f(x+1)=f(x,y)+h*(0.5*k1+0.5*k2)
        Z3[i+1]=Z[i+1][2]    # extract
        print("Caculating Heun method ! Progress: %f %%" % (i*100/(len(x)-1))) # Print the progress
    return Z3  


# Exact solution by using odeint
F = lambda Z,t: [Z[1],Z[3] - Z[1] - Z[0],Z[3],impulse(t)/2 + Z[1] - Z[3]] 
# Defined the function for odeint
# There is a -0.5 be ignored because ignoring the linearisation point and the constant disappears after differentiation.


'''
Step 3: Solve the function 
'''

#solve the function
Eu_result= Euler(state_matrix,x)  
He_result= heun(state_matrix,x)   
Exact = integrate.odeint(F,[0,0,0,0],x) 

exact=np.zeros(len(x))     # Extract the third variables - X2(t)
for i in range(len(x)):
 exact[i]=Exact[i][2]
 i+=1
'''
Step4: error analysis
'''

Error_euler=np.zeros(len(x))   #error array initialize
Error_Heun=np.zeros(len(x))

def error(y,s):  # define the error function
   result=abs((y-s)/s*100)
   return result

for i in range(len(x)):  
 Error_euler[i]=error(Eu_result[i],exact[i]) #caculate the each step's error
 Error_Heun[i]=error(He_result[i],exact[i])
 # FOR test below
 #print(x[i],"\t",exact[i],"\t",Eu_result[i],"\t ",error(Eu_result[i],exact[i]),"\t",He_result[i],error(He_result[i],exact[i]),"\t",end="\t")
 #print()





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
plt.plot(x,exact,'r',label='exact')
plt.legend()
plt.title('Displacement')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$x_2(t)$')
#plt.savefig('./1.jpg') #For save imagine
#plt.savefig('1.eps',dpi=600,format='eps')
plt.show()


#plot the error
plt.plot(x,Error_euler,label='Euler')
plt.plot(x,Error_Heun,label='Heun')
plt.ylim((0, 5))
plt.yticks(np.arange(0, 5, 0.5))
plt.legend()
plt.title('Error in this step size')
plt.xlabel('$t$')
plt.ylabel('$Error\%$')
plt.grid()
#plt.savefig('./2.jpg') #For save imagine
#plt.savefig('2.eps',dpi=600,format='eps')
plt.show()

