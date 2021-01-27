import numpy as np
def reglin(x,y):
	mx=np.ones((x.shape[0],2),np.float32)
	mx[:,0]=x.transpose()
	p=np.matmul(np.matmul(np.linalg.inv(np.matmul(mx.transpose(),mx)),mx.transpose()),y.reshape(x.shape[0],1))
	return p[0,0] ,p[1,0]

x=np.array([1,2,3,4],np.float32)
y=np.array([3,5,7,10],np.float32)


beta0,beta1=reglin(x,y)
print(beta0,beta1)
