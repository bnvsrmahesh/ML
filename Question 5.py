# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:04:51 2020

@author: bnvsrmahesh
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time

#m =number of rows will be the number of dimensions or features
#n =number of columns will be the samples 
def PCA(eig_vectors,p,IM,m,n):
    #p=percentage of PCA
    #IM=inpiut image
    #m,n size of image
    z=(p*m)//100#z number of principal compnents
    eig_m = eig_vectors[:,0:z]
    eig_md = eig_vectors[:,z:m]
    IMF=np.zeros((m,n))
    Cmean=np.mean(IM,axis=1).reshape(m,1)
    Cmeant=Cmean.T
    #mean summation term will be same for all
    const_sum = np.dot(Cmeant,eig_md)#This has matrix of all Xbar.uj where j is from m+1 to d
    const_sumvect = np.dot(const_sum,eig_md.T)#This is the constant vector term i.e sumof m+1 to d basis vectors
      #cal principal components for each column
    Bt=IM.T#Xi transpose term
    const_m = np.dot(Bt,eig_m)#PCA dimension values
    trans_x = np.dot(const_m,eig_m.T)#upto m basis vectors sum
    IMF += trans_x.T+const_sumvect.T
    return(IMF)


def PCArand(eig_vectors,p,IM,m,n):#PCA by choosing random components
    z=(p*m)//100 
    IMF=np.zeros((m,n))
    D=list(range(0,m))
    G=[]
    eig_m = np.zeros((m,z))
    eig_md = np.zeros((m,m-z))
    for i in range(z):      
      fd = random.randint(0,len(D)-2)#0,len(D)-2 are inclusive
      D.remove(D[fd])
      G.append(D[fd])
    for i in range(eig_m.shape[1]):
        eig_m[:,i] += eig_vectors[:,G[i]]
    for i in range(eig_md.shape[1]):
        eig_md[:,i] += eig_vectors[:,D[i]]
    Cmean=np.mean(IM,axis=1).reshape(m,1)
    Cmeant=Cmean.T
    const_sum = np.dot(Cmeant,eig_md)#This has matrix of all Xbar.uj where j is from m+1 to d
    const_sumvect = np.dot(const_sum,eig_md.T)#This is the constant vector term i.e sumof m+1 to d basis vectors
      #cal principal components for each column
    #Calculating all columns principal components
    Bt=IM.T#A transpose
    const_m = np.dot(Bt,eig_m)#PCA dimension values
    trans_x = np.dot(const_m,eig_m.T)#upto m basis vectors sum
    IMF = trans_x.T+const_sumvect.T        
    return(IMF)

    
def scaling(I):
    I=np.array(I,dtype=int)
    return(I)

#reading and converting the image into gray scale
start_time =time.time()
I0=cv2.imread('30.jpg',cv2.IMREAD_COLOR)#By default cv2 reads image as BGR instead of RGB
I1=cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)#Converting BGR to RGB
I=cv2.cvtColor(I0,cv2.COLOR_BGR2GRAY)#Converting to Gray
IM=np.array(I)
Column_mean=np.mean(IM,axis=1)
m,n=IM.shape
Cmean=Column_mean.reshape(m,1)
Cmeant=Cmean.T
Cov=np.zeros((m,m))

#Calculation of Covariance Matrix(C matrix)
for i in range(n):
    B=IM[:,i].reshape(m,1)
    Bt=IM[:,i].reshape(1,m)
    C=B-Cmean
    Ct=Bt-Cmeant
    Cov+=np.matmul(C,Ct)
Covf=Cov*(1/n)
eig_values,eig_vectors=np.linalg.eig(Covf)#eigen vectors and eigen values

#PCA on The Image
Image_modified10 = PCA(eig_vectors,10,IM,m,n)
Image_modified25 = PCA(eig_vectors,25,IM,m,n)
Image_modified50 = PCA(eig_vectors,50,IM,m,n)
Image_modifiedrand = PCArand(eig_vectors,10,IM,m,n)
Image_modified75 =PCA(eig_vectors,75,IM,m,n)
Image_modified85 =PCA(eig_vectors,85,IM,m,n)
Image_modified95 =PCA(eig_vectors,95,IM,m,n)
Image_modified100 =PCA(eig_vectors,100,IM,m,n)
Image_modified20 = PCA(eig_vectors,20,IM,m,n)
Image_modified65 =PCA(eig_vectors,65,IM,m,n)
Image_modified40 =PCA(eig_vectors,40,IM,m,n)

#Frobenius norm error,Dont do scaling before calculating error because of that approximation to int we are getting large error
error10 =np.linalg.norm(I-Image_modified10)
error25 =np.linalg.norm(I-Image_modified25)
error50 =np.linalg.norm(I-Image_modified50)
error75 =np.linalg.norm(I-Image_modified75)
error85=np.linalg.norm(I-Image_modified85)
error95=np.linalg.norm(I-Image_modified95)
error100=np.linalg.norm(I-Image_modified100)
error20 = np.linalg.norm(I-Image_modified20)
error65 =np.linalg.norm(I-Image_modified65)
error40 =np.linalg.norm(I-Image_modified40)


#Converting all the pixel values into integers for plotting image
Image_modified10 = scaling(Image_modified10)
Image_modified25 = scaling(Image_modified25)
Image_modified50 = scaling(Image_modified50)
Image_modifiedrand = scaling(Image_modifiedrand)
#Dont do INT conversion for calculating error
# Image_modified75 = scaling(Image_modified75)
# Image_modified85 = scaling(Image_modified85)
# Image_modified95 = scaling(Image_modified95)
# Image_modified100 = scaling(Image_modified100)




errorrand =np.linalg.norm(I-Image_modifiedrand)


fig=plt.figure(8)
#Color Image plot
plt.figure(1)
plt.imshow(I1)
plt.axis("off")
plt.title("INPUT COLOR IMAGE")
plt.show()
#Color to Gray Image Plot
plt.figure(2)
plt.imshow(I,cmap=plt.cm.gray)
plt.axis("off")
plt.title('Grayscale Image')
plt.show()
#10% components image
plt.figure(3)
fig3,b=plt.subplots(1,2)
b[0].imshow(Image_modified10,cmap=plt.cm.gray)
b[0].axis('off')
b[0].set_title('Image by 10% Components',fontsize='8')
b[1].imshow(I-Image_modified10,cmap=plt.cm.gray)
b[1].axis('off')
b[1].set_title('Error Image by 10% Components',fontsize='8')
plt.show()
#25% Components image
plt.figure(4)
fig4,c =plt.subplots(1,2)
c[0].imshow(Image_modified25,cmap=plt.cm.gray)
c[0].axis('off')
c[0].set_title('Image by 25% Components',fontsize='8')
c[1].imshow(I-Image_modified25,cmap=plt.cm.gray)
c[1].axis('off')
c[1].set_title('Error Image by 25% Components',fontsize='8')
plt.show()
#50% Components image
plt.figure(5)
fig5,d =plt.subplots(1,2)
d[0].imshow(Image_modified50,cmap=plt.cm.gray)
d[0].axis('off')
d[0].set_title('Image by 50% Components',fontsize='8')
d[1].imshow(I-Image_modified50,cmap=plt.cm.gray)
d[1].axis('off')
d[1].set_title('Error Image by 50% Components',fontsize='8')
plt.show()
#rand Image with its error image
plt.figure(6)
plt.imshow(Image_modifiedrand,cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image by Random 10% Components',fontsize='8')
plt.show() 
#Error image with rand10%
plt.figure(7)
fig2,a =plt.subplots(1,2)
a[0].imshow(Image_modifiedrand,cmap=plt.cm.gray)
a[0].axis('off')
a[0].set_title('Rand10% Image',fontsize='8')
a[1].imshow(I-Image_modifiedrand,cmap=plt.cm.gray)#error image is the image by subtracting reconstructed with original
a[1].axis('off')
a[1].set_title('Error Image rand10%',fontsize='9')
plt.show()
#Error vs N plot
error_list=[error10,error20,error25,error40,error50,error65,error75,error85,error95,error100]#obtained by running the error terms above seprately
p = [10,20,25,40,50,65,75,85,95,100]
N_list =[]
for i in range(len(p)):
 ff =(m*p[i])//100
 N_list.append(ff)

plt.figure(8)
plt.plot(N_list,error_list)
plt.xlabel('Number of Components')
plt.ylabel('Frobenius Error')
plt.title("Error vs N_Components")
plt.show()

end_time =time.time()
print('runtime is',end_time-start_time)
print("Error with top 10% less compared to random 10%")










