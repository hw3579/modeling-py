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
B=numpy.array([0,0,0,0.5])

C=numpy.array([0,-0.5,0,0])

def matrix():
    dotA=numpy.multiply(A,X)+numpy.multiply(numpy.identity(4),B)#+numpy.multiply(numpy.identity(4),C)
    return dotA


#solution=integrate.odeint((sol,[0,0,0,0],x))

def array_to_sympy(i):
    eq=matrix()[i]
    eq1_list=eq.tolist()
    eq1_str=" ".join('%s' %id for id in eq1_list)
    eq1_str2=eq1_str.replace('0','')
    eq1_str3=eq1_str2.replace('.','0.')
    eq1_str4=eq1_str3.replace('x','+x')
    eq1_str5=eq1_str4.replace('- +','-')
    eq1_str5=eq1_str5.replace(' 0.5','+')
    eq1_sympy=sympify(eq1_str5)
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
    k1=f_list[0].subs(x2,y[j][1])
    k2=f_list[1].subs([(x1,y[j][0]),(x2,y[j][1]),(x4,y[j][3])])
    k3=f_list[2].subs([(x4,y[j][3])])
    k4=f_list[3].subs([(x2,y[j][1]),(x4,y[j][3])])

    y[j+1][0]=h*k1+y[j][0]
    y[j+1][1]=h*k2+y[j][1]
    y[j+1][2]=h*k3+y[j][2]
    y[j+1][3]=h*k4+y[j][3]
  return y
xxxx=Euler(f_list)

result=numpy.zeros((len(x)))
for i in range(len(x)-1):
 result[i]=xxxx[i][2]

plt.plot(x,result)
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