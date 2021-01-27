import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

x,y=make_regression(n_samples=100,
n_features=1, noise=3)
y=y.reshape(100,1)

X=np.hstack((x,np.ones((np.shape(x)[0],1))))

beta=np.random.randn(2,1)
def model(X,beta):
   return X.dot(beta)

y_pred=model(X,beta)
plt.scatter(x,y)
plt.plot(x,y_pred, c='r')
#plt.show()

def cost_function(X,y,beta):
  n=len(y)
  u=(model(X,beta)-y).T
  return (1/n*u.dot(u.T))[0,0]

print("mse:", cost_function(X,y,beta))


def grad(X,y,beta):
     n=len(y)
     return (2/n)*X.T.dot(model(X,beta)-y)





def gradient_descent(X,y,beta,learning_rate=0.001, n_iteration=1000):
   for i in range(n_iteration):
       beta=beta-learning_rate*grad(X,y,beta)
   return beta


beta_estimer=gradient_descent(X,y,beta,learning_rate=0.01,n_iteration=2000)

y_pred=model(X,beta_estimer)
plt.scatter(x,y)
plt.plot(x,y_pred, c='b')

plt.show()