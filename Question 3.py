#!/usr/bin/env python
# coding: utf-8

# In[245]:


import numpy as np
import matplotlib.pyplot as plt 


# In[246]:


#input parameters
mu1  = np.array([0,0,0]).T
mu2  = np.array([1,5,-3]).T
mu3  = np.array([0,0,0]).T
cov1 = np.array([[3,0,0],[0,5,0],[0,0,2]])
cov2 = np.array([[1,0,0],[0,4,1],[0,1,6]])
cov3 = np.array([[10,0,0],[0,10,0],[0,0,10]])


# In[247]:


#Part A generating trivariate_normal train points for  classes
I1 = np.random.multivariate_normal(mu1,cov1,20)
I2 = np.random.multivariate_normal(mu2,cov2,20)
I3 = np.random.multivariate_normal(mu3,cov3,20)
I1_y = np.ones((I1.shape[0],1))#class lable
I2_y = np.ones((I2.shape[0],1))*2
I3_y = np.ones((I3.shape[0],1))*3
#Estimated means and covariance
est_mu1 = np.mean(I1,axis = 0)
est_mu2 = np.mean(I2,axis = 0)
est_mu3 = np.mean(I3, axis = 0)
est_cov1 = np.cov(I1.T)
est_cov2 = np.cov(I2.T)
est_cov3 = np.cov(I3.T)


# In[248]:


#Combining all train points
X_train = np.concatenate((I1,I2),axis = 0)
X_train = np.concatenate((X_train,I3),axis = 0)
Y_train = np.concatenate((I1_y,I2_y),axis = 0)
Y_train = np.concatenate((Y_train,I3_y),axis = 0)


# In[249]:


#Calculation of  assumed covariance from all 3 classes 
cov_assumed = np.cov(X_train.T)


# In[250]:


def shrinking(alpha,cov_assumed,ni,nt,est_cov):
    A = (1-alpha)*ni*est_cov
    B = nt*alpha*cov_assumed
    NUM = A+B
    DEN = (1-alpha)*ni+alpha*nt
    cov_shrink = NUM/DEN
    return(cov_shrink)
    


# In[251]:


def trivariatenormal(X,mean,cov):
    A = X-mean
    V = np.linalg.inv(cov)
    DEN = np.pi*2*(np.sqrt(np.pi*2*np.linalg.det(cov)))
    tri_out = np.zeros((A.shape[0],1))
    for i in range(A.shape[0]):
        p = np.matmul(np.matmul(A[i,0:3].T,V),A[i,0:3])
        p = p*(-0.5)
        NUM = np.exp(p)
        tri_out[i] += NUM/DEN
    return(tri_out)


# In[252]:


def bayesclassifier(X,mean0,mean1,mean2,cov0,cov1,cov2,L,P1,P2,P3):#Passing all Data atonce instead sample wise
    fX1 = trivariatenormal(X,mean=mean0,cov=cov0)#class conditional density for each class
    fX2 = trivariatenormal(X,mean=mean1,cov=cov1)
    fX3 = trivariatenormal(X,mean=mean2,cov=cov2)
    q1  = P1*fX1
    q2  = P2*fX2
    q3  = P3*fX3
    R1  = L[0,0]*q1+L[0,1]*q2+L[0,2]*q3
    R2  = L[1,0]*q1+L[1,1]*q2+L[1,2]*q3
    R3  = L[2,0]*q1+L[2,1]*q2+L[2,2]*q3
    Class =np.zeros((R1.shape[0],1))
    Density =np.zeros((R1.shape[0],1))
    for i in range (R1.shape[0]):       
        if (R1[i] <= R2[i]) & (R1[i] <= R3[i]):
            Class[i] = 1
            Density[i] = fX1[i]
        elif (R2[i] <= R1[i]) & (R2[i] <= R3[i]):
            Class[i] = 2
            Density[i] = fX2[i]
        else:
            Class[i] = 3 
            Density[i] = fX3[i]
    return (Class,Density)#Returns the class which its classified and Density Value


# In[253]:


def prediction(X,mean0,mean1,mean2,cov0,cov1,cov2,L,Y,P1,P2,P3):
    Class_pred,Density = bayesclassifier(X, mean0, mean1, mean2, cov0, cov1, cov2, L,P1,P2,P3) 
    true = 0
    false = 0
    for i in range (Class_pred.shape[0]):
        if Class_pred[i] == Y[i]:
            true += 1
        else:
            false += 1
    accuracy = (true /( true+false ))*100
    return (accuracy)


# In[254]:


#Part B Alpha vs Accuracy
L = np.array([[0,1,1],[1,0,1],[1,1,0]])#L[0,0]=0 that is 0 loss when properly assigned
alpha = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
P1 = I1.shape[0]/X_train.shape[0]
P2 = I2.shape[0]/X_train.shape[0]
P3 = I3.shape[0]/X_train.shape[0]
accuracy = []
for i in range (len(alpha)):
    cov1_shrink = shrinking(alpha[i],cov_assumed,I1.shape[0],X_train.shape[0],est_cov1)
    cov2_shrink = shrinking(alpha[i],cov_assumed,I2.shape[0],X_train.shape[0],est_cov2)
    cov3_shrink = shrinking(alpha[i],cov_assumed,I3.shape[0],X_train.shape[0],est_cov3)
    accuracy.append(prediction(X_train,est_mu1,est_mu2,est_mu3,cov1_shrink,cov2_shrink,cov3_shrink,L,Y_train,P1,P2,P3))
fig = plt.figure
plt.plot(alpha,accuracy)
plt.xlabel('alpha')
plt.ylabel('%train_accuracy')
plt.title('%Train_accuracy vs Alpha')
plt.savefig('Train_Accuracy vs Alpha')
plt.show()



# In[255]:


#Test Data generation
test_1 = np.random.multivariate_normal(mu1,cov1,50)
test_2 = np.random.multivariate_normal(mu2,cov3,50)
test_3 = np.random.multivariate_normal(mu3,cov3,50)
X_test = np.concatenate((test_1,test_2),axis = 0)
X_test = np.concatenate((X_test,test_3),axis = 0)
y1_test = np.ones((test_1.shape[0],1))
y2_test = np.ones((test_2.shape[0],1))
y3_test = np.ones((test_3.shape[0],1))
Y_test  = np.concatenate((y1_test,y2_test),axis = 0)
Y_test  = np.concatenate((Y_test,y3_test),axis=0)
alpha = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
accuracy2 = []
#For testing only input data changes other things like mean and covariance,P1,P2,P3 of train data only must be used
for i in range (len(alpha)):
    cov1_shrink = shrinking(alpha[i],cov_assumed,I1.shape[0],X_train.shape[0],est_cov1)#covariance of train data
    cov2_shrink = shrinking(alpha[i],cov_assumed,I2.shape[0],X_train.shape[0],est_cov2)
    cov3_shrink = shrinking(alpha[i],cov_assumed,I3.shape[0],X_train.shape[0],est_cov3)
    accuracy2.append(prediction(X_test,est_mu1,est_mu2,est_mu3,cov1_shrink,cov2_shrink,cov3_shrink,L,Y_test,P1,P2,P3))#means and covariance of train data being used 
fig = plt.figure
plt.plot(alpha,accuracy2)
plt.xlabel('alpha')
plt.ylabel('%test_accuracy')
plt.title('%Test_accuracy vs Alpha')
plt.savefig('Test_accuracy vs Alpha')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




