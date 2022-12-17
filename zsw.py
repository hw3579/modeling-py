from matplotlib import pyplot as plt
import numpy
from scipy import integrate
from sympy import *
x1=symbols('x1')
x2=symbols('x2')
x3=symbols('x3')
x4=symbols('x4')

X=numpy.array([x1,x2,x3,x4])


h=0.01
x=numpy.arange(0,45,h)



def impulse(t):     # Define the impulse force
    if t<1:
        return 1
    else:
        return 0


#X=numpy.array([x1,x2,x3,x4])

A=numpy.array([[0,1,0,0],
              [-1,-1,0,1],
              [0,0,0,1],
              [0,1,0,-1]])

C=numpy.array([0,-0.5,0,0])

def matrix():
    dotA=numpy.multiply(A,X)+numpy.multiply(numpy.identity(4),C)
    return dotA


#solution=integrate.odeint((sol,[0,0,0,0],x))

def array_to_sympy(i):
    eq=matrix()[i]
    eq1_list=eq.tolist()
    eq1_str=" ".join('%s' %id for id in eq1_list)
    eq1_str2=eq1_str.replace('0','')
    eq1_str3=eq1_str2.replace('.','0.')
    eq1_str4=eq1_str3.replace('x','+x')
    eq1_sympy=sympify(eq1_str4)
    return eq1_sympy


f1=array_to_sympy(0)
f2=array_to_sympy(1)
f3=array_to_sympy(2)
f4=array_to_sympy(3)

f_list=[f1,f2,f3,f4]



def Euler(f_list):
  y=numpy.zeros((len(x),5))
  for i in range(len(x-1)):
    y[i][4]=x[i]
  y[0][3]=0.5
  for j in range(len(x)-1):
   for i in range(3):
    y[j+1][i]=f_list[i].subs([(x1,y[j][i]),(x2,y[j][i]),(x3,y[j][i]),(x4,y[j][i])])
  return y[:2]

result=Euler(f_list)

plt.show()




'''

import matplotlib.pyplot as plt
plt.plot(x, sol[:, 2], 'b', label='theta(t)')
plt.plot(x, sol[:, 3], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()



class Equation:

    def __init__(self, vc, vd):
        self.vc = vc
        self.vd = vd
        return

     

def painter(xp, yp):
    fig, ax = plt.subplots()
    fig.set_figwidth(18)
    fig.set_figheight(18)
    ax.plot(xp, yp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    ax.set_title("ODE")
    plt.show()

eq = Equation([1, 1, -1, 1], [1, 1, 0, 0])

#xp, yp = eq.equation_solver_rungekutta(0, 10, [1, 1], 0.01)
#painter(xp, yp)

for i in range(len(xp)-1):
 print(xp[i])
'''