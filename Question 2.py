#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[37]:


#Reading the data
Data = pd.read_csv('Dataset_5_Team_30.csv')
X = Data.values
X.shape


# In[38]:


#Part A estimating sample variance
X_var = np.var(X)
X_std = np.sqrt(X_var)
X_mean = np.mean(X)


# In[39]:


#Part B
def posterior(theta,mu0,n,sig,Sn,var):
#theta = mean term
#mu0 = prior mean
#n = number of samples
#sig = sigma/sigma0 square
#Sn = sample mean
#var = sample variance
    thetan = (n/(n+sig))*Sn + mu0/((n/sig)+1)
    sigman = X_var/(n+sig)
    A = -1*((theta-thetan)**2)*(1/(2*sigman))
    B = np.exp(A)
    C = B/(np.sqrt(2*np.pi*sigman))
    return (C)
    


# In[40]:


theta = np.linspace(-4.75,4.75,num = 500)
mu0   = -1
n     = 10
Sn = 0#sum of n samples
for i in range(n):
    Sn += X[i]
sig   = [.1,1,10,100]
y1     = posterior(theta,mu0,n,sig[0],Sn/n,X_var)
y2     = posterior(theta,mu0,n,sig[1],Sn/n,X_var)
y3     = posterior(theta,mu0,n,sig[2],Sn/n,X_var)
y4     = posterior(theta,mu0,n,sig[3],Sn/n,X_var)
fig = plt.figure
fig1,a  = plt.subplots(2,2,figsize = (30,15))
a[0][0].plot(theta,y1)
a[0][0].set_xlabel('Theta',fontsize='20')
a[0][0].set_ylabel('Posterior Density',fontsize='20')
a[0][0].set_xlim(-1.5,4.75)
a[0][0].set_xticks(np.linspace(-1.5,4.75,num =7))
a[0][0].set_title('n=10,sigm/sim0=0.1 plot',fontsize='20')


a[1][0].plot(theta,y2)
a[1][0].set_xlabel('Theta',fontsize='20')
a[1][0].set_ylabel('Posterior Density',fontsize='20')
a[1][0].set_xlim(-1.5,4.75)
a[1][0].set_xticks(np.linspace(-1.5,4.75,num =7))
a[1][0].set_title('n=10,sigm/sim0=1 plot',fontsize='20')

a[0][1].plot(theta,y3)
a[0][1].set_xlabel('Theta',fontsize='20')
a[0][1].set_ylabel('Posterior Density',fontsize='20')
a[0][1].set_xlim(-1.5,4.75)
a[0][1].set_xticks(np.linspace(-1.5,4.75,num =7))
a[0][1].set_title('n=10,sigm/sim0=10 plot',fontsize='20')

a[1][1].plot(theta,y1)
a[1][1].set_xlabel('Theta',fontsize='20')
a[1][1].set_ylabel('Posterior Density',fontsize='20')
a[1][1].set_xlim(-1.5,3.75)
a[1][1].set_xticks(np.linspace(-1.5,4.75,num =7))
a[1][1].set_title('n=10,sigm/sim0=100 plot',fontsize='20')
plt.savefig('n=10 plot')
plt.show()


# In[41]:


theta = np.linspace(-4.75,4.75,num = 500)
mu0   = -1
n     = 100
Sn = 0#sum of n samples
for i in range(n):
    Sn += X[i]
sig   = [.1,1,10,100]
y1     = posterior(theta,mu0,n,sig[0],Sn/n,X_var)
y2     = posterior(theta,mu0,n,sig[1],Sn/n,X_var)
y3     = posterior(theta,mu0,n,sig[2],Sn/n,X_var)
y4     = posterior(theta,mu0,n,sig[3],Sn/n,X_var)
fig = plt.figure
fig2,b  = plt.subplots(2,2,figsize = (30,15))
b[0][0].plot(theta,y1)
b[0][0].set_xlabel('Theta',fontsize='20')
b[0][0].set_ylabel('Posterior Density',fontsize='20')
b[0][0].set_xlim(-0.1,2.5)
b[0][0].set_xticks(np.linspace(-0.10,2.5,num =7))
b[0][0].set_title('n=100,sigm/sim0=0.1 plot',fontsize='20')


b[1][0].plot(theta,y2)
b[1][0].set_xlabel('Theta',fontsize='20')
b[1][0].set_ylabel('Posterior Density',fontsize='20')
b[1][0].set_xlim(-0.1,2.5)
b[1][0].set_xticks(np.linspace(-0.1,2.5,num =7))
b[1][0].set_title('n=100,sigm/sim0=1 plot',fontsize='20')

b[0][1].plot(theta,y3)
b[0][1].set_xlabel('Theta',fontsize='20')
b[0][1].set_ylabel('Posterior Density',fontsize='20')
b[0][1].set_xlim(-0.1,2.5)
b[0][1].set_xticks(np.linspace(-0.1,2.5,num =7))
b[0][1].set_title('n=100,sigm/sim0=10 plot',fontsize='20')

b[1][1].plot(theta,y1)
b[1][1].set_xlabel('Theta',fontsize='20')
b[1][1].set_ylabel('Posterior Density',fontsize='20')
b[1][1].set_xlim(-0.1,2.5)
b[1][1].set_xticks(np.linspace(-0.1,2.5,num =7))
b[1][1].set_title('n=100,sigm/sim0=100 plot',fontsize='20')
plt.savefig('n=100 plot')
plt.show()


# In[42]:


theta = np.linspace(-4.75,4.75,num = 500)
mu0   = -1
n     = 1000
Sn = 0#sum of n samples
for i in range(n):
    Sn += X[i]
sig   = [.1,1,10,100]
y1     = posterior(theta,mu0,n,sig[0],Sn/n,X_var)
y2     = posterior(theta,mu0,n,sig[1],Sn/n,X_var)
y3     = posterior(theta,mu0,n,sig[2],Sn/n,X_var)
y4     = posterior(theta,mu0,n,sig[3],Sn/n,X_var)
fig = plt.figure
fig3,c  = plt.subplots(2,2,figsize = (30,15))
c[0][0].plot(theta,y1)
c[0][0].set_xlabel('Theta',fontsize='20')
c[0][0].set_ylabel('Posterior Density',fontsize='20')
c[0][0].set_xlim(.75,1.5)
c[0][0].set_xticks(np.linspace(0.75,1.5,num =7))
c[0][0].set_title('n=1000,sigm/sim0=0.1 plot',fontsize='20')


c[1][0].plot(theta,y2)
c[1][0].set_xlabel('Theta',fontsize='20')
c[1][0].set_ylabel('Posterior Density',fontsize='20')
c[1][0].set_xlim(.75,1.5)
c[1][0].set_xticks(np.linspace(0.75,1.5,num =7))
c[1][0].set_title('n=1000,sigm/sim0=1 plot',fontsize='20')

c[0][1].plot(theta,y3)
c[0][1].set_xlabel('Theta',fontsize='20')
c[0][1].set_ylabel('Posterior Density',fontsize='20')
c[0][1].set_xlim(.75,1.5)
c[0][1].set_xticks(np.linspace(0.75,1.5,num =7))
c[0][1].set_title('n=1000,sigm/sim0=10 plot',fontsize='20')

c[1][1].plot(theta,y1)
c[1][1].set_xlabel('Theta',fontsize='20')
c[1][1].set_ylabel('Posterior Density',fontsize='20')
c[1][1].set_xlim(.75,1.5)
c[1][1].set_xticks(np.linspace(0.75,1.5,num =7))
c[1][1].set_title('n=1000,sigm/sim0=100 plot',fontsize='20')
plt.savefig('n=1000 plot')
plt.show()


# In[ ]:




