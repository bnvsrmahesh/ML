# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:48:55 2020

@author: bnvsrmahesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import time


def bivariatenormal(X,mean,cov):#ML estimate for mean,variance of Bivariate normal is the mean,covaraiance 
    #This calculates the Bivariatenoraml value for all samples at once and returns the out put array instead of calculating for each sample
    A = X-mean
    V = np.linalg.inv(cov)
    DEN = np.pi*2*(np.sqrt(np.linalg.det(cov)))
    Biv_out = np.zeros((A.shape[0],1))
    for i in range(A.shape[0]):
     k = np.dot(np.dot(A[i,0:2].T,V),A[i,0:2])
     k = k*(-0.5)
     NUM = np.exp(k)
     Biv_out[i] += NUM/DEN
    return(Biv_out)


def bayesclassifier(X,mean0,mean1,mean2,cov0,cov1,cov2,L):#Passing all Data atonce instead sample wise
    fX0 = bivariatenormal(X,mean=mean0,cov=cov0)#class conditional density for each class
    fX1 = bivariatenormal(X,mean=mean1,cov=cov1)
    fX2 = bivariatenormal(X,mean=mean2,cov=cov2)
    q0  = P0*fX0#P0 global variable no need to give as input
    q1  = P1*fX1
    q2  = P2*fX2
    R0  = L[0,0]*q0+L[0,1]*q1+L[0,2]*q2
    R1  = L[1,0]*q0+L[1,1]*q1+L[1,2]*q2
    R2  = L[2,0]*q0+L[2,1]*q1+L[2,2]*q2
    Class =np.zeros((R0.shape[0],1))
    Density =np.zeros((R0.shape[0],1))
    for i in range (R0.shape[0]):       
     if (R0[i] <= R1[i]) & (R0[i] <= R2[i]):
      Class[i] = 0
      Density[i] = fX0[i]
     elif (R1[i] <= R0[i]) & (R1[i] <= R2[i]):
      Class[i] = 1
      Density[i] = fX1[i]
     else:
      Class[i] = 2 
      Density[i] = fX2[i]
    return (Class,Density)#Returns the class which its classified and Density Value

def prediction(X,mean0,mean1,mean2,cov0,cov1,cov2,L):
    Class_pred,Density = bayesclassifier(X[:,0:2], mean0, mean1, mean2, cov0, cov1, cov2, L) 
    true = 0
    false = 0
    for i in range (Class_pred.shape[0]):
        if Class_pred[i] == X[i,2]:
            true += 1
        else:
            false += 1
    accuracy = (true /( true+false ))*100
    return (Class_pred,accuracy)


start_time =time.time()
Data = pd.read_csv("Dataset_1_Team_30.csv")#read a csv file csv=comma seprated values
Data_train,Data_test = train_test_split(Data,test_size=0.2)

D_test  = Data_test.values
D_train = Data_train.values

X0 = Data_train[Data_train['Class_label']==0]#Data belonging to class 0
X1 = Data_train[Data_train['Class_label']==1]#Data belonging to class 1
X2 = Data_train[Data_train['Class_label']==2]#Data belonging to class 2

#Converting from data frame to array
X0 = X0.values
X1 = X1.values
X2 = X2.values
X_train = D_train[:,0:3]#Dropping of the class values

#Prior Probabilities 
P0 = X0.shape[0]/(X0.shape[0]+X1.shape[0]+X2.shape[0])
P1 = X1.shape[0]/(X0.shape[0]+X1.shape[0]+X2.shape[0])
P2 = X2.shape[0]/(X0.shape[0]+X1.shape[0]+X2.shape[0])

mean0 = np.mean(X0[:,0:2],axis=0)#row wise mean calculation for X0[0],X0[]1
mean1 = np.mean(X1[:,0:2],axis=0)
mean2 = np.mean(X2[:,0:2],axis=0)


#Naive bayes is a diagonal matrix with its diagonals elements as variance
#Covariance matrix for Naive Bayes with identitiy Covariance matrix
Cov01 = np.array([[1,0],[0,1]])
Cov11= np.array([[1,0],[0,1]])
Cov21 = np.array([[1,0],[0,1]])

#Covariance matrix for Naive Bayes with same covariance matrix
Cov02 = np.cov(X_train[:,0:2])
Cov02 = np.array([[Cov02[0,0],0],[0,Cov02[1,1]]])
Cov12 = Cov02
Cov22 = Cov02

#Covariance matrix for Naive Bayes with different covaricance matrix
Cov03 = np.cov(X0[:,0:2].T) #Transpose is used because cov fucntion is giving row wise covirance but we want column
Cov13 = np.cov(X1[:,0:2].T)
Cov23 = np.cov(X2[:,0:2].T)
Cov03 = np.array([[Cov03[0,0],0],[0,Cov03[1,1]]])
Cov13 = np.array([[Cov13[0,0],0],[0,Cov13[1,1]]])
Cov23 = np.array([[Cov23[0,0],0],[0,Cov23[1,1]]])
 
#Covariance matrixor Bayes with same covirance
Cov04 = np.cov(X_train[:,0:2].T)
Cov14 = Cov04
Cov24 = Cov04

#Covariance Matrix Bayes with different coviance
Cov05 = np.cov(X0[:,0:2].T)
Cov15 = np.cov(X1[:,0:2].T)
Cov25 = np.cov(X2[:,0:2].T)

#loss Matrix
L = np.array([[0,2,1],[2,0,3],[1,3,0]])


#Bayes Classifer assuming x1,x2 to be bivariate normal
#Model 1 naive with Identity Matrix
pred_train_navieIbayes,pred_train_acc_navieIbayes = prediction(X_train, mean0, mean1, mean2, Cov01, Cov11, Cov21,L)
pred_test_navieIbayes,pred_test_acc_navieIbayes = prediction(D_test, mean0, mean1, mean2, Cov01, Cov11, Cov21,L)
#for test data we are using mean of the train data its important to note that we are developing classifier on train data and using it on test data
#Also covariance for train data is the test data covariance because we are using the classifier trained by test Data
print("pred_train_acc_navieIbayes =",pred_train_acc_navieIbayes)
print("pred_test_acc_navieIbayes =", pred_test_acc_navieIbayes)
print('\n')

#Model 2 Naive Bayes classifier with covariance same for all classes
pred_train_naviesamebayes,pred_train_acc_naviesamebayes = prediction(X_train, mean0, mean1, mean2, Cov02, Cov12, Cov22, L)
pred_test_naviesamebayes,pred_test_acc_naviesamebayes = prediction(D_test, mean0, mean1, mean2, Cov02, Cov12, Cov22, L)

print("pred_train_acc_naviesamebayes =",pred_train_acc_naviesamebayes)
print("pred_test_acc_naviesamebayes =", pred_test_acc_naviesamebayes)
print('\n')


# Model 3: Naive Bayes classifier with covariance dierent for all classes.
pred_train_naviediffbayes,pred_train_acc_naviediffbayes = prediction(X_train, mean0, mean1, mean2, Cov03, Cov13, Cov23, L)
pred_test_naviediffbayes,pred_test_acc_naviediffbayes = prediction(D_test, mean0, mean1, mean2, Cov03, Cov13, Cov23, L)


print("pred_train_acc_naviediffbayes =",pred_train_acc_naviediffbayes)
print("pred_test_acc_naviediffbayes =", pred_test_acc_naviediffbayes)
print('\n')

#Model 4: Bayes classifier with covariance same for all classes.
pred_train_samebayes,pred_train_acc_samebayes = prediction(X_train, mean0, mean1, mean2, Cov04, Cov14, Cov24, L)
pred_test_samebayes,pred_test_acc_samebayes = prediction(D_test, mean0, mean1, mean2, Cov04, Cov14, Cov24, L,)


print("pred_train_acc_samebayes =",pred_train_acc_samebayes)
print("pred_test_acc_samebayes =", pred_test_acc_samebayes)
print('\n')

#Model 5: Bayes classifier with covariance different for all classes
pred_train_diffbayes,pred_train_acc_diffbayes = prediction(X_train, mean0, mean1, mean2, Cov05, Cov15, Cov25, L)
pred_test_diffbayes,pred_test_acc_diffbayes = prediction(D_test, mean0, mean1, mean2, Cov05, Cov15, Cov25, L)


print("pred_train_acc_diffbayes =",pred_train_acc_diffbayes)
print("pred_test_acc_diffbayes =", pred_test_acc_diffbayes)
print('\n')

#Best model is said by test accuracy since all having same test accuracy using avg of test and train
acc1 = (pred_train_acc_naviesamebayes + pred_test_acc_naviesamebayes )/2
acc2 = (pred_train_acc_naviesamebayes + pred_test_acc_naviesamebayes )/2
acc3 = (pred_train_acc_naviediffbayes + pred_test_acc_naviediffbayes )/2
acc4 = (pred_train_acc_samebayes + pred_test_acc_samebayes )/2
acc5 = (pred_train_acc_diffbayes + pred_test_acc_diffbayes )/2

ACC=[acc1,acc2,acc3,acc4,acc5]
print (ACC.index(max(ACC)))


#Here for Naive bayes with diff covariance and bayes with diff covariance are having almost same accuracy
#Considering the Bayes with different covairance as better one because its more genralized and here data set is such that the covariance of two variables is almost zero
#That is the reason why they are almost equal or else the one with bayes of diff covariance gives the max accuracy

#Construction of confucion matrxi for bayes with diff covariance matrix
Con_diffbayes = np.zeros((3,3))#there are 3 classes
for i in range(pred_test_diffbayes.shape[0]):   
   if pred_test_diffbayes[i] == D_test[i,2]:#true positive case
      h = int(np.asscalar(pred_test_diffbayes[i]))
      Con_diffbayes[h,h] += 1#along diagonal
   else:
      k = int(np.asscalar(pred_test_diffbayes[i]))
      l = int(np.asscalar(D_test[i,2]))
      Con_diffbayes[k,l] += 1#row is predicted,column is true case


print(Con_diffbayes)
print(pred_test_diffbayes.shape[0])#Number of samples in test data
print (Con_diffbayes.sum())#Sum of all the elements in array must be same as aboveS

#eigen vectors for covariance matrix of best model
eig_values_diffbayes05,eig_vectors_diffbayes05 = np.linalg.eig(Cov05)
eig_values_diffbayes15,eig_vectors_diffbayes15 = np.linalg.eig(Cov15)
eig_values_diffbayes25,eig_vectors_diffbayes25 = np.linalg.eig(Cov25)
print (eig_vectors_diffbayes05)
print (eig_vectors_diffbayes15)
print (eig_vectors_diffbayes25)


#PLOTS
#accuracy doesnt make sense in surface plot because these are the points we created
y1 = np.linspace(-250,300,num = 600)#test x1 points created
y2 = np.linspace(-650,700,num = 600)#test x2 points created
y1,y2 = np.meshgrid(y1,y2)#creating a grid of test points so that when plotted on graph we can get continuous points
Y1 = y1.flatten()
Y2 = y2.flatten()
plot_arr = np.zeros((Y1.shape[0],2))#for zeros two brackets to be there
plot_db = np.zeros((plot_arr.shape[0],1))
plot_arr[:,0] += Y1#no need to use ravel because Y1 doesnt have a proper shape its mmore like values
plot_arr[:,1] += Y2
plot_out = np.zeros((plot_arr.shape[0],1))
plot_out,plot_db = bayesclassifier(plot_arr,mean0,mean1,mean2,Cov05,Cov15,Cov25,L)#in put must be the mean of train data that is what we mean by being trained by this data,covaraiance of model 5 taken since its best performed 
#Vectorizing Function calls
#In above call we are calculating all the class values atonce only one function call if we use for loop and call for each sample there will be more function calls which is not good
plot_out = plot_out.reshape(y1.shape[0],y1.shape[1])
plot_db = plot_db.reshape(y1.shape[0],y1.shape[1])
fig =plt.figure(3)
#Decision Boundary and Surface plot
plt.figure(1)#for figure 1
plt.contour(y1,y2,plot_db)
plt.contourf(y1,y2,plot_out)
plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.scatter(X0[:,0],X0[:,1],color="red",s=5,label="class0",marker='^')#points in test data that belong to class 0
plt.scatter(X1[:,0],X1[:,1],color="blue",s=5,label="class1",marker='*')
plt.scatter(X2[:,0],X2[:,1],color="violet",s=5,label="class2",marker='o')
plt.legend()
plt.title("Decision Boundary and Surface")
plt.show()

#Best model contour curves and eigen vectors
plt.figure(2)
origin0 = [mean0[0]], [mean0[1]] 
plt.quiver( *origin0,eig_vectors_diffbayes05[0]*-5, eig_vectors_diffbayes05[1]*-5, color=['b','b'],scale=30)
origin1 = [mean1[0]], [mean1[1]] # origin point
plt.quiver( *origin1,eig_vectors_diffbayes15[0]*-5, eig_vectors_diffbayes15[1]*-5, color=['r','r'],scale=25)
origin2 = [mean2[0]], [mean2[1]] # origin point
plt.quiver( *origin2,eig_vectors_diffbayes25[0]*-5, eig_vectors_diffbayes25[1]*-5, color=['g','g'],scale=25)
plt.contour(y1,y2,plot_db,cmap=cm.jet)
plt.scatter(X0[:,0],X0[:,1],color="red",s=5,label="Class0",marker='^')#points in test data that belong to class 0
plt.scatter(X1[:,0],X1[:,1],color="blue",s=5,label="Class1",marker='*')
plt.scatter(X2[:,0],X2[:,1],color="violet",s=5,label="Class2",marker='o')
plt.xlabel("Input X1")
plt.ylabel("Input X2")
plt.title("BEST MODEL Contour curves and Eigen Vectors")
plt.show()

# =============================
fig = plt.figure(3)
ax = fig.gca(projection='3d')
X, Y, Z = y1,y2,plot_db
ax.plot_surface(X, Y, Z, cmap = 'viridis')
cset = ax.contour(X, Y, Z, zdir='z', offset=-5e-5)
# cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
# cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('Input X1')
# ax.set_xlim(-40, 40)
ax.set_ylabel('Input X2')

ax.set_zlabel('Probability Density')
ax.set_zlim(-5e-5, 5e-5)
ax.set_title('Gaussian PDF and Contours for best model')

plt.show()
end_time = time.time()
print("Time tanken",end_time-start_time)







