/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#ifndef CUDA_SPHERE_HELPERS_H
#define CUDA_SPHERE_HELPERS_H

#include <helper_cuda.h>
#include <stdint.h>
#include <stdio.h>

// ------------------------------------------------------------------------
// copied from ../dpMM/cuda/cuda_global.h

#define PI  3.141592653589793f
#define MIN_DOT -0.95
#define MAX_DOT 0.95

template<typename T>
{
  T dot = min(1.0,max(-1.0,q[0]*p[0] + q[1]*p[1] + q[2]*p[2]));
  // 2nd order taylor expansions for the limit cases obtained via mathematica
  T invSinc = 0.0;
  if(static_cast<T>(MIN_DOT) < dot && dot < static_cast<T>(MAX_DOT))
    invSinc = acos(dot)/sqrt(1.-dot*dot);
  else if(dot <= static_cast<T>(MIN_DOT))
    invSinc = PI/(sqrt(2.)*sqrt(dot+1.)) -1. + PI*sqrt(dot+1.)/(4.*sqrt(2.))
      -(dot+1.)/3. + 3.*PI*(dot+1.)*sqrt(dot+1.)/(32.*sqrt(2.)) 
      - 2./15.*(dot+1.)*(dot+1.);
  else if(dot >= static_cast<T>(MAX_DOT))
    invSinc = 1. - (dot-1)/3. + 2./5.*(dot-1.)*(dot-1.);
  x[0] = (q[0]-p[0]*dot)*invSinc;
  x[1] = (q[1]-p[1]*dot)*invSinc;
  x[2] = (q[2]-p[2]*dot)*invSinc;
}

/* just base function - empty because we are specializing if you look down 
template<typename T>
__device__ inline T atomicAdd_(T* address, T val)
{};

/* atomic add for double */
template<>
__device__ inline double atomicAdd_<double>(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
};

template<>
__device__ inline float atomicAdd_<float>(float* address, float val)
{
  return atomicAdd(address,val);
};

#endif
