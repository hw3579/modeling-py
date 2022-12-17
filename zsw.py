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

a=matrix()

def Euler():
  y=numpy.zeros((len(x),4))
  for i in range(len(x)-1):
    eq=matrix()
    eq[1].subs=(x1,x[i][1])





  return 0


y=numpy.zeros((len(x),4))
for i in range(len(x)-1):
    eq=matrix()[0]
    eq1_list=eq.tolist()
    #eq1_str2="".join(eq)
    eq1_str3=" ".join('%s' %id for id in eq1_list)
    eq1_sympy=sympify(eq1_str2)

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