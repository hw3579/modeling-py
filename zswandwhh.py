from matplotlib import pyplot as plt
import numpy
from scipy import integrate
from sympy import *

class data:

 h=0.5

 x1=symbols('x1')
 x2=symbols('x2')
 x3=symbols('x3')
 x4=symbols('x4')

 X=numpy.array([x1,x2,x3,x4])
 x=numpy.arange(0,45,h)

 A=numpy.array([[0,1,0,0],
              [-1,-1,0,1],
              [0,0,0,1],
              [0,1,0,-1]])
 B=numpy.array([0,0,0,0.5])
 C=numpy.array([0,-0.5,0,0])

class transfer:

    def array_to_sympy(self,i):
     eq=Function.matrix(self)[i]
     eq1_list=eq.tolist()
     eq1_str=" ".join('%s' %id for id in eq1_list)
     eq1_str2=eq1_str.replace('0','')
     eq1_str3=eq1_str2.replace('.','0.')
     eq1_str4=eq1_str3.replace('x','+x')
     eq1_str5=eq1_str4.replace('- +','-')
     eq1_str5=eq1_str5.replace(' 0.5','+')
     eq1_sympy=sympify(eq1_str5)
     return eq1_sympy


class Function:
    

    def __init__(self):
        return
    
    def matrix(self):
     dotA=numpy.multiply(data.A,data.X)+numpy.multiply(numpy.identity(4),data.B)
     return dotA

    def Euler(self,f_list):
      y=numpy.zeros((len(data.x),5))
      for i in range(len(data.x-1)):
       y[i][4]=data.x[i]
      y[0][3]=0.5
      for j in range(len(data.x)-1):
       k1=f_list[0].subs(data.x2,y[j][1])
       k2=f_list[1].subs([(data.x1,y[j][0]),(data.x2,y[j][1]),(data.x4,y[j][3])])
       k3=f_list[2].subs([(data.x4,y[j][3])])
       k4=f_list[3].subs([(data.x2,y[j][1]),(data.x4,y[j][3])])

       y[j+1][0]=data.h*k1+y[j][0]
       y[j+1][1]=data.h*k2+y[j][1]
       y[j+1][2]=data.h*k3+y[j][2]
       y[j+1][3]=data.h*k4+y[j][3]

       print("Euler is in caculate! percentage: %s" % (j/(len(data.x)-1)*100))
      return y


    def Heun(self,f_list):
      y=numpy.zeros((len(data.x),5))
      for i in range(len(data.x-1)):
        y[i][4]=data.x[i]
      y[0][3]=0.5
      for j in range(len(data.x)-1):
       k1=f_list[0].subs(data.x2,y[j][1])
       k2=f_list[1].subs([(data.x1,y[j][0]),(data.x2,y[j][1]),(data.x4,y[j][3])])
       k3=f_list[2].subs([(data.x4,y[j][3])])
       k4=f_list[3].subs([(data.x2,y[j][1]),(data.x4,y[j][3])])

       k11=f_list[0].subs(data.x2,y[j][1]+k1*data.h)
       k22=f_list[1].subs([(data.x1,(y[j][0]+k2*data.h)),(data.x2,(y[j][1]+k2*data.h)),(data.x4,(y[j][3])+k2*data.h)])
       k33=f_list[2].subs([(data.x4,(y[j][3])+k3*data.h)])
       k44=f_list[3].subs([(data.x2,(y[j][1]+k4*data.h)),(data.x4,(y[j][3]+k4*data.h))])

       y[j+1][0]=y[j][0]+data.h*(0.5*k1+0.5*k11)
       y[j+1][1]=y[j][1]+data.h*(0.5*k2+0.5*k22)
       y[j+1][2]=y[j][2]+data.h*(0.5*k3+0.5*k33)
       y[j+1][3]=y[j][3]+data.h*(0.5*k4+0.5*k44)

       print("Heun is in caculate! percentage: %s" % (j/(len(data.x)-1)*100))
      return y



if __name__ == '__main__':
     f=Function()
     tr=transfer()
     f1=tr.array_to_sympy(0)
     f2=tr.array_to_sympy(1)
     f3=tr.array_to_sympy(2)
     f4=tr.array_to_sympy(3)
     f_list=[f1,f2,f3,f4]
     euler_result=f.Euler(f_list)
     Heun_result=f.Heun(f_list)

     final_euler=numpy.zeros((len(data.x)))
     final_Heun=numpy.zeros((len(data.x)))
     for i in range(len(data.x)-1):
      final_euler[i]=euler_result[i][2]
      final_Heun[i]=Heun_result[i][2]


     plt.plot(data.x,final_euler)
     plt.plot(data.x,final_Heun)
     plt.show()

