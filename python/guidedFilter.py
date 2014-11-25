# Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import cv2
from js.data.rgbd.rgbdframe import RgbdFrame


rgbd = RgbdFrame(540)
rgbd.load('./table_0')


eps = 0.03**2

g = np.copy(rgbd.d.astype(np.float) )*1e-3
d = np.copy(rgbd.d.astype(np.float) )*1e-3

#qBilat = cv2.adaptiveBilateralFilter(d,(3,3),0.03)

print (g==0).sum()
haveData = g>=1e-2
Ns = cv2.integral(haveData.astype(np.float))

dSum = cv2.integral(d.astype(np.float))
#dSumSq = dSum**2
#dMean = np.nan_to_num(dSum/Ns)
#dSig = dSqSum - dSumSq/Ns

gSum,gSqSum = cv2.integral2(g.astype(np.float))
#gSumSq = gSum**2
#gMean = np.nan_to_num(gSum/Ns)
#gSig = np.nan_to_num(gSqSum - gSumSq/Ns)


def integralGet(I,i,j,w):
  return I[min(i+w,I.shape[0]-1),min(j+w,I.shape[1]-1)] \
       - I[min(i+w,I.shape[0]-1),max(j-w,0)] \
       - I[max(i-w,0),min(j+w,I.shape[1]-1)] \
       + I[max(i-w,0),max(j-w,0)]

prod = d*g
prod[np.logical_not(haveData)] = 0.
prodInt = cv2.integral(prod.astype(np.float))

a = np.zeros_like(g)
b = np.zeros_like(g)
for i in range(g.shape[0]):
  for j in range(g.shape[1]):
    if haveData[i,j]:
#    print integralGet(Ns,i,j,w), integralGet(prodInt,i,j,w), integralGet(gMean,i,j,w), integralGet(dMean,i,j,w), integralGet(gSig,i,j,w)
      n = integralGet(Ns,i,j,w) # number of datapoints 
      muG = integralGet(gSum,i,j,w)/n 
      muD = integralGet(dSum,i,j,w)/n
      sigG = (integralGet(gSqSum,i,j,w) - n*muG**2)/(n-1.)
#      print n,muG, muD, sigG
      a[i,j] = (integralGet(prodInt,i,j,w)/n - muG*muD) /( sigG + eps)
      b[i,j] = muD - muG*a[i,j]
#      print integralGet(prodInt,i,j,w)/n, muG, muD, muG*muD, sigG + eps
  if i%30 == 0:
    print i
#for i in range(g.shape[0]):
#  for j in range(g.shape[1]):
#    b[i,j] = integralGet(dMean,i,j,w) - integralGet(gMean,i,j,w)*a[i,j]
#  if i%30 == 0:
#    print i

a = np.nan_to_num(a)
b = np.nan_to_num(b)

aInt = cv2.integral(a) 
bInt = cv2.integral(b) 

plt.figure()
plt.subplot(221)
plt.imshow(a)
plt.colorbar()
plt.subplot(222)
plt.imshow(b)
plt.colorbar()
plt.subplot(223)
plt.imshow(a+b)
plt.colorbar()
plt.subplot(224)
plt.imshow(bInt)
plt.colorbar()

print 'reconstructing'
q = np.zeros_like(g)
for i in range(g.shape[0]):
  for j in range(g.shape[1]):
    if haveData[i,j]:
      n = integralGet(Ns,i,j,w) # number of datapoints 
      muA = integralGet(aInt,i,j,w)/n
      muB = integralGet(bInt,i,j,w)/n
      q[i,j] = muA*g[i,j] + muB
  if i%30 == 0:
    print i
#    print n,muA,muB,q[i,j]

#q[q>10.*np.max(d)] = np.nan
d[np.logical_not(haveData)] = np.nan
q[np.logical_not(haveData)] = np.nan

#qMax = np.percentile(q.ravel()[np.logical_not(np.isnan(q.ravel()))], 99.99)
#qMin = np.percentile(q.ravel()[np.logical_not(np.isnan(q.ravel()))], 10)
#print qMax
#q[q>qMax] = np.nan

#q *= np.max(d[haveData])/qMax

fig = plt.figure()
plt.subplot(221)
plt.imshow(d)
plt.colorbar()
plt.title('input')
plt.subplot(222)
plt.imshow(q)
plt.colorbar()
plt.title('output')
plt.subplot(223)
plt.imshow(np.abs(d-q))
plt.colorbar()
plt.title('absolute difference')
#plt.hist(q.ravel()[np.logical_not(np.isnan(q.ravel()))],bins=100)
fig.show()
plt.show()  

rgbq = RgbdFrame(540)
rgbq.load('./table_0')                                                          
rgbq.d = np.ceil(q*1000.).astype(np.uint16)

nd = rgbd.getNormals(algo='sobel',reCompute=True)
nq = rgbq.getNormals(algo='sobel',reCompute=True)

import ipdb
ipdb.set_trace()

fig = plt.figure()
plt.subplot(221)
plt.imshow(nd)
plt.colorbar()
plt.subplot(222)
plt.imshow(nq)
plt.colorbar()
plt.subplot(223)
plt.imshow(np.abs(nd-nq))
plt.colorbar()
#plt.hist(q.ravel()[np.logical_not(np.isnan(q.ravel()))],bins=100)
fig.show()


rgbq.showPc()
rgbd.showPc()

figm = mlab.figure()
rgbd.showNormals(None,as2D=False,reCompute=True)
figm = mlab.figure()
rgbq.showNormals(None,as2D=False,reCompute=True)

mlab.show(stop=True)

plt.show() 
