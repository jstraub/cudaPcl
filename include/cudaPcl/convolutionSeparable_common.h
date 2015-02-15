/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#ifndef CONVOLUTIONSEPARABLE_COMMON_H                                           
#define CONVOLUTIONSEPARABLE_COMMON_H

//#define KERNEL_RADIUS_S 1

#define KERNEL_RADIUS 1
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernel(float *h_Kernel);

extern "C" void convolutionRowsGPU( float *d_Dst, float *d_Src, int
    imageW, int imageH);

extern "C" void convolutionColumnsGPU( float *d_Dst, float *d_Src, int
    imageW, int imageH);

#endif
