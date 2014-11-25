# Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import cv2
from js.data.rgbd.rgbdframe import RgbdFrame

import pycuda.compiler as comp                                                  
import pycuda.driver as drv                                                     
import pycuda.autoinit 
from pycuda import gpuarray

import os,time

class IntegralImg(object):
  def __init__(self):
    with open(os.path.dirname(__file__)+'./integral.cu', 'r') as content_file:
      kernel = content_file.read() 
    mod = comp.SourceModule(kernel)                                           
    self.integralGpu = mod.get_function('integral_kernel')   
    self.transposeGpu = mod.get_function('transpose_kernel')   
    self.addCarryGpu = mod.get_function('addCarryOver_kernel')   
  def compute(self,I):
    d = np.zeros((256*3,256*3),dtype=np.float32,order='C')
    d[0:I.shape[0],0:I.shape[1]] = I
    dSum = np.zeros_like(d).astype(np.float32)
    dSumCary = np.zeros((d.shape[0],d.shape[1]/(2*128)),dtype=np.float32,order='C')
    dSumT = np.zeros_like(d).astype(np.float32)                                      
  
    t  = time.time()
    d_gpu = gpuarray.to_gpu(d)
    dSum_gpu = gpuarray.to_gpu(dSum)
    dSumT_gpu = gpuarray.to_gpu(dSumT)
    dSumCary_gpu = gpuarray.to_gpu(dSumCary)
    
    self.integralGpu(d_gpu,dSum_gpu,dSumCary_gpu,np.int32(d.shape[1]),np.int32(d.shape[0]),
        grid=(d.shape[1]/256,d.shape[0],1),block=(128,1,1)) 
    self.addCarryGpu(dSum_gpu,dSumCary_gpu,np.int32(d.shape[1]),np.int32(d.shape[0]),
        grid=(d.shape[1]/256,d.shape[0],1),block=(256,1,1)) 
    self.transposeGpu(dSum_gpu,dSumT_gpu,np.int32(d.shape[1]),np.int32(d.shape[0]),
        grid=(d.shape[1]/16,d.shape[0]/16,1) ,block=(16,16,1)) 
    self.integralGpu(dSumT_gpu,dSum_gpu,dSumCary_gpu,np.int32(d.shape[1]),np.int32(d.shape[0]),
        grid=(d.shape[1]/256,d.shape[0],1),block=(128,1,1)) 
    self.addCarryGpu(dSum_gpu,dSumCary_gpu,np.int32(d.shape[1]),np.int32(d.shape[0]),
        grid=(d.shape[1]/256,d.shape[0],1),block=(256,1,1)) 
    self.transposeGpu(dSum_gpu,dSumT_gpu,np.int32(d.shape[1]),np.int32(d.shape[0]),
        grid=(d.shape[1]/16,d.shape[0]/16,1) ,block=(16,16,1)) 
    
    dSum = dSumT_gpu.get()
    print time.time()-t
    return dSum[:I.shape[0],:I.shape[1]]

if __name__ == "__main__":
  rgbd = RgbdFrame(540)
  rgbd.load('./table_0')
  
  d = np.copy(rgbd.d.astype(np.float))*1e-3
  
  
  integral = IntegralImg()
  dSum = integral.compute(d)
  print dSum
  print "done"

  t = time.time()
  dSumcv2 = cv2.integral(d.astype(np.float))
  print time.time() -t
  
  plt.figure()
  plt.subplot(221)
  plt.imshow(d,interpolation="nearest")
  plt.colorbar()
  plt.subplot(222)
  plt.imshow(dSum,interpolation="nearest")
  plt.colorbar()
  plt.subplot(223)
  plt.imshow(dSumcv2,interpolation="nearest")
  plt.colorbar()
  plt.subplot(224)
  plt.imshow(np.abs(dSumcv2[1::,1::]-dSum),interpolation="nearest")
  plt.colorbar()
  plt.show()
