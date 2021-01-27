#creation du dataset 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

np.random.seed(0)
nb_exemples=100
X=np.linspace(0,10,nb_exemples).reshape(nb_exemples,1)
y=X**2+X*np.random.randn(nb_exemples,1)



model=SVR(C=100)
model.fit(X,y)
sc=model.score(X,y)
print("Coefficient de détermination:",sc)
pred=model.predict(X)
plt.scatter(X,y)
plt.plot(X,pred, c='r')

plt.show()
