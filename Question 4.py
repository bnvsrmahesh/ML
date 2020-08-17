#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import math
import array
import scipy.linalg as la
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random


# In[121]:


n = 100
k = 10

col = np.random.rand(n)

a = np.zeros((n,n))

for i in range(n):
    const = 0.05
    a[i] = ((const * col) + np.random.normal(0, 1, n))
    
max = np.amax(a)
min = np.amin(a)
a = a - min
a = a / (max - min)

norm1 = LA.norm(a)
print(norm1)


# In[122]:


def svd(a,n):
    aat = a.dot(a.T)
    ata = (a.T).dot(a)
    eigvals, eigvecs = la.eig(aat)
    u = np.zeros((n,n))
    for j in range(n):
        u[j] = eigvecs[j]
    eigvals, eigvecs = la.eig(ata)
    vt = np.zeros((n,n))
    for k in range(n):
        vt[k] = eigvecs[k]
    s = np.zeros((n,n))
    eigvals = eigvals.real
    s = np.diag(eigvals)


# In[123]:


def frac_captured(a, n, k, fb):
    aat = a.dot(a.T)
    ata = (a.T).dot(a)
    
    eigvals1, eigvecs1 = la.eig(aat)
    eigvals1 = eigvals1.real
    easc = np.argsort(eigvals1)
    edes1 = easc[::-1]
    ereq = edes1[:k]
    
    u = np.zeros((k,n))
    u = eigvecs1[ereq]
    
    s = np.zeros((k,k)) 
    s = np.diag(np.sqrt(eigvals1[ereq]))


    eigvals2, eigvecs2 = la.eig(ata)
    eigvals2 = eigvals2.real
    evals2 = np.argsort(eigvals2)
    edes2 = evals2[::-1]
    
    vt = np.zeros((k,n))
    vt = eigvecs2[ereq]
        
    req = (u.T).dot(s.dot(vt))
    
    norm = (LA.norm(req))
    
    frac =  norm / fb
    return frac*100

k = 10
print(frac_captured(a,n,k,norm1))


# In[124]:


def frac_captured_rand(a, n, k, fb):
    aat = a.dot(a.T)
    ata = (a.T).dot(a)
    
    eigvals1, eigvecs1 = la.eig(aat)
    eigvals1 = eigvals1.real
    index = np.random.choice(eigvals1.shape[0], k, replace=False)  

    u = np.zeros((k,n))
    u = eigvecs1[index]
    
    s = np.zeros((k,k)) 
    s = np.diag(np.sqrt(eigvals1[index]))
    
    eigvals2, eigvecs2 = la.eig(ata)
    eigvals2 = eigvals2.real
    
    vt = np.zeros((k,n))
    vt = eigvecs2[index]
        
    req = (u.T).dot(s.dot(vt))
    
    norm = (LA.norm(req))
    
    frac =  norm / fb
    return frac*100

print(frac_captured_rand(a,n,k,norm1))


# In[125]:


x = np.linspace(0,n,num=n+1)
y = np.zeros(n+1)

for i in range(n+1):
    y[i] = frac_captured(a,n,i,norm1)

plt.plot(x,y)
plt.xlabel('No of singular vectors')
plt.ylabel('Percentage of data captured')
plt.title('No. of singular vectors required \n to capture a % of data for \n Highly Correlated Entries \n')
plt.show()


# In[126]:


b = np.random.rand(n,n)
norm2 = LA.norm(b)
print(norm2)


# In[127]:


k = 10
print(frac_captured(b,n,k,norm2))


# In[128]:


print(frac_captured_rand(b,n,k,norm2))


# In[129]:


x2 = np.linspace(0,n,num=n+1)
y2 = np.zeros(n+1)

for i in range(n+1):
    y2[i] = frac_captured(b,n,i,norm2)

plt.plot(x2,y2)
plt.xlabel('No of singular vectors')
plt.ylabel('Percent Captured')
plt.title('No. of singular vectors required \n to capture a % of data for \n Independent Entries')
plt.show()

