/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <cudaPcl/smoothNormalsGpu.hpp>
#include <rgbdGrabber/realSenseGrabber.hpp>

namespace cudaPcl {

/*
 */
class RealSenseSmoothNormalsGpu : public rgbdGrabber::RealSenseGrabber,
  SmoothNormalsGpu 
{
  public:
  RealSenseSmoothNormalsGpu(double f_d, double eps, uint32_t B, bool
      compress=false)
    : rgbdGrabber::RealSenseGrabber(640,480,60), SmoothNormalsGpu(f_d,
        eps, B, compress)
  { };

  virtual ~RealSenseSmoothNormalsGpu() { };

  virtual void rgbd_cb(const uint8_t* rgb, const uint16_t * depth)
  {
    depth_cb(depth, w_, h_);
  };

  protected:
};

}

