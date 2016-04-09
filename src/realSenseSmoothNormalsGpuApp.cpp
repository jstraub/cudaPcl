/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <cudaPcl/realSenseSmoothNormalsGpu.hpp>

int main (int argc, char** argv)
{
  cudaPcl::RealSenseSmoothNormalsGpu g(640,480,60);
  g.run ();
  return (0);
}
