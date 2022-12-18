import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
t = np.arange(0, 45, 0.0001)
'''we choose 55s as the end time, although we can see the system has become stable since 35s
add 10s to 45s just to ensure it is stable
we can see that,0.0001s as a step  to ensure the error percent lower than 1%'''

'''this is the unit impluse force input'''
def impulseForce(t):
    if t < 1:
        return 1
    else:
        return 0
    
def dX(t, X):
    '''The function is the equivalent of the derivative matrix"
    It requires two inputs
    - t, one single value of time
    - X, is the matrix of x1, x1_dot, x2, x2_dot'''
   
    x1_dot = X[1]
    x1_double_dot = X[3] - X[1] - X[0]
    x2_dot = X[3]
    x2_double_dot = impulseForce(t)/2 + X[1] - X[3]

    return np.array([x1_dot, x1_double_dot, x2_dot, x2_double_dot])
    
#define odeint
F = lambda X,t: [X[1],X[3] - X[1] - X[0],X[3],impulseForce(t)/2 + X[1] - X[3]]
Xodeint=integrate.odeint(F,[0,0,0,0],t)   


def eulerMethod(dX,t):
    '''This returns the approximated values of x2 for every time instants
    Requires 2 inputs:
    - dX, this is your derivative function
    - t, an array of time instances 
    '''
    #Create a matrix of zeros
    #The number of rows is equal to the number of time instances
    #The number of columns is 4
    X = np.zeros((len(t), 4))

    h = t[1] - t[0]

    #iteration
    for i in range(len(t)-1):
        X[i+1] = h * dX(t[i], X[i]) + X[i]

    #This returns the 3rd column of the matrix X which is x2 (mass 2's displacement)
    return X[:,2]

def RungeKuttaMethod(dX,t):
    '''This returns the approximated values of x2 for every time instants
    Requires 2 inputs:
    - dX, this is your derivative function
    - t, an array of time instances 
    '''
    #Create a matrix of zeros
    #The number of rows is equal to the number of time instances
    #The number of columns is 4
    X = np.zeros((len(t), 4))

    #Variables for runge kutta method
    a_1 = 1/2
    a_2 = 1/2
    p_1 = 1
    q_11 = 1

    h = t[1] - t[0]

    #iterate 
    for i in range(len(t)-1):
        k_1 = dX(t[i], X[i])
        k_2 = dX(t[i] + p_1 * h, X[i] + q_11 * k_1 * h)
        X[i+1] = X[i] + (a_1 * k_1 + a_2 * k_2) * h

    #This returns the 3rd column of the matrix X which is x2 (mass 2's displacement)
    return X[:,2]
#Make list of each version
list_of_x2_euler = eulerMethod(dX, t)
list_of_x2_runge = RungeKuttaMethod(dX, t)
list_of_x2_odeint = Xodeint[:,2]

#Plot them and display on the same graph
plt.plot(t, list_of_x2_euler)
plt.plot(t, list_of_x2_runge, '-.')
plt.plot(t,Xodeint[:,2],':')
plt.title("Output response using Euler and Runge Kutta Method")
plt.xlabel("t, seconds")
plt.ylabel("output position, x2(t)")
plt.grid()
plt.show()


#equation to find the final value using odeint and Euler method and Runge kutta method
final_Value_odeint=list_of_x2_odeint[len(list_of_x2_odeint)-1]
final_Value_Euler = list_of_x2_euler[len(list_of_x2_euler)-1]
final_Value_Runge = list_of_x2_runge[len(list_of_x2_runge)-1]

#print the final value of Euler, Runge Kutta and odeint
print("The final value using the Euler method is", final_Value_Euler)
print("The final value using the Runge Kutta method is", final_Value_Runge)
print("The final value using the odeint method is",final_Value_odeint)

#equation to find error using Euler method and Runge kutta method,we use odeint as final value
Euler_error = final_Value_Euler - final_Value_odeint
Runge_error = final_Value_odeint -final_Value_Runge 
print("The error using the Euler method is", Euler_error)
print("The percentage error using the Euler method is ", (Euler_error/final_Value_odeint)*100,"%") 
print("The error using the Runge Kutta method is", Runge_error)
print("The percentage error using the Runge Kutaa method is", (Runge_error/final_Value_odeint)*100,"%") 


