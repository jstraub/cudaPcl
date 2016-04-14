/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <cudaPcl/realSenseSmoothNormalsGpu.hpp>

int main (int argc, char** argv)
{
    
  double f_d = 540.;
  double eps = 0.05*0.05;
  int32_t B = 10;
  bool compress = false;
  findCudaDevice(argc,(const char**)argv);
  cudaPcl::RealSenseSmoothNormalsGpu g(f_d, eps, B, compress);
  g.run ();
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
