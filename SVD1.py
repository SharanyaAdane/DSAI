# -*- coding: utf-8 -*-
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize']=[16,8]

A=imread('lion1.jpeg')

A_R = A[:,:,0]
A_G = A[:,:,1]
A_B = A[:,:,2]
X=np.mean(A,-1)

#X = np.array([[1,0,0],[3,4,0],[5,6,9]])

img=plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

U,S,VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

j = 0
for r in (5,20,100):
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j +=1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()

    
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()

