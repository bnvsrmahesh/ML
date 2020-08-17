#!/usr/bin/env python
# coding: utf-8

# In[159]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from numpy.linalg import inv


# In[160]:


x = np.linspace(0,1,num=101)
x1 = np.sin(2 * np.pi * x)
x2 = np.exp(x1)
y = np.zeros(101)
for i in range(101):
    y[i] = x2[i] + np.random.normal(0, np.sqrt(0.2))
    
    
plt.plot(x,x2,'b')
plt.xlabel('Sample Points')
plt.ylabel('Target Value (Without Noise)')
plt.title('Target Function')
plt.show()


# In[161]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.80,test_size=0.20)


# In[162]:


degree = 1
data_size = 10

index = np.random.choice(x_train.size,data_size,replace = False)
x_rand = x_train[index]
y_rand = y_train[index]


# In[163]:


degree = 6


# In[164]:


def coeff(degree, data_size):
    X = np.zeros((data_size,(degree+1)))
    for i in range(data_size):
        for j in range(degree+1):
            X[i][j] = x_rand[i] ** j
    
    Y = np.zeros((data_size,1))
    for i in range(data_size):
        Y[i] = y_rand[i]
    
    inverse = inv((X.T).dot(X))
    
    w = inverse.dot((X.T).dot(Y))
    
    return w


w = np.zeros(((degree+1),1))
w = coeff(degree,data_size)    
print('w =')
print(w)


# In[165]:


def coeff_t(degree):
    X = np.zeros((21,(degree+1)))
    for i in range(21):
        for j in range(degree+1):
            X[i][j] = x_test[i] ** j
    
    Y = np.zeros((21,1))
    for i in range(21):
        Y[i] = y_test[i]
    
    inverse = inv((X.T).dot(X))
    
    w = inverse.dot((X.T).dot(Y))
    
    return w


# In[166]:


func = np.zeros(101)
for i in range(101):
    for j in range(degree+1):
        func[i] = func[i] + w[j] * (x[i] ** j)


# In[167]:


def y_target(xin):
    X = np.zeros((xin.shape[0],(degree+1)))
    for i in range(xin.shape[0]):
        for j in range(degree+1):
            X[i][j] = xin[i] ** j
    
    Y = X.dot(w)
    
    return Y

y_tar = y_target(x_rand)


# In[168]:


def plot():
    a = x
    b = y
    c = func
    d = y_tar
    
    #plt.plot(a,b,'r')
    plt.plot(a,c,'b')
    #plt.plot(x_rand,y_tar,marker='o',color='b')
    plt.scatter(x_rand,y_rand,color='r')
    
    axes = plt.gca()
    axes.set_xlim([0.0,1.0])
    axes.set_ylim([-5.0,5.0])
    
    
    plt.xlabel('Sample points')
    plt.ylabel('Target Value')
    plt.title('Regression Output for Degree ..., Data Size ...')
    plt.show()

    
plot()  


# In[169]:


def rms_train():
    
    error = 0
    
    y_tar = y_target(x_rand)
    
    for i in range(data_size):
        error = error + ((y_tar[i] - y_rand[i]) ** 2 )
    
    rms = np.sqrt((error/data_size))
    
    return rms

    
    
rms = np.zeros(10)


for degree in range(10):
    w = np.zeros(((degree+1),1))
    w = coeff(degree,data_size)
    y_tar = y_target(x_rand)
    rms[degree] = rms_train()[0]
    
m = np.linspace(0,9,num=10)

plt.plot(m,rms)
plt.plot(m,rms,'o')
axes = plt.gca()
axes.set_ylim([0.0,1.0])

plt.xlabel('Order of Polynomial Regression')
plt.ylabel('RMS error')
plt.title('Root Mean Square Error (RMS) for Training Data of size')
#plt.savefig('rms_train_80.png')



plt.show()


# In[170]:


def rms_test():
    
    error = 0
    
    for i in range(21):
        error = error + ((y_tar[i] - y_test[i]) ** 2 )
    
    rms = np.sqrt((error/21))
    
    return rms

    
    
rms_t = np.zeros(10)

for degree in range(10):
    w = np.zeros(((degree+1),1))
    w = coeff(degree,data_size)
    y_tar = y_target(x_test)
    rms_t[degree] = rms_test()[0]
    
m = np.linspace(0,9,num=10)


plt.plot(m,rms_t)
plt.plot(m,rms_t,'o')

plt.xlabel('Order of Polynomial Regression')
plt.ylabel('RMS error')
plt.title('Corresponding Root Mean Square Error (RMS) for Test Data ')
#plt.savefig('rms_test_80.png')


plt.show()



# In[171]:


degree = 6
data_size = 80
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.80,test_size=0.20)

index = np.random.choice(x_train.size,data_size,replace = False)
x_rand = x_train[index]
y_rand = y_train[index]

w = np.zeros(((degree+1),1))
w = coeff(degree,data_size)  
y_tar = y_target(x_rand)

plt.plot(y_rand,y_tar,'o')
axes = plt.gca()
axes.set_xlim([0.0,3.0])
axes.set_ylim([0.0,3.0])

plt.xlabel('Target Output tn')
plt.ylabel('Model Output y(xn,w)')
plt.title('Model Output vs Target Output for \n Degree 6 Regression for Training data')
plt.show()


# In[172]:


y_tar1 = y_target(x_test)

plt.plot(y_test,y_tar1,'o')
axes = plt.gca()
axes.set_xlim([0.0,3.0])
axes.set_ylim([0.0,3.0])

plt.xlabel('Target Output tn')
plt.ylabel('Model Output y(xn,w)')
plt.title('Model Output vs Target Output for \n Degree 6 Regression for Test data')
plt.show()


# In[ ]:




