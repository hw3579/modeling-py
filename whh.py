import numpy as np
import matplotlib.pyplot as plt
h=0.01
x=np.arange(0,45,h)
def d(X):
    X_=np.zeros([4,1])
    X_[0]=X[1]
    X_[1]=X[3]-X[1]-X[0]
    X_[2]=X[3]
    X_[3]=-X[3]+X[1]
    return X_
def euler(x):
    Y=np.zeros((len(x),4))
    Y[0][3]=0.5
    for i in range (0,len(x)-1):
        k=d(Y[i])
        Y[i+1]=Y[i]+h*k.reshape(4,)
    return Y[:,2]
plt.plot(x,euler(x))
plt.show()


error_Euler=(euler(x)[-1]-0.5)/0.5*100
print("The Euler error percentage is: ", error_Euler)