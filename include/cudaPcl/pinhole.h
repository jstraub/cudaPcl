/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#pragma once
#include <stdint.h>
#include <Eigen/Dense>

namespace cudaPcl{
class Pinhole {
 public:
  Pinhole(const Eigen::Matrix3f& R_C_W, const Eigen::Vector3f& t_C_W,
      float f, uint32_t w, uint32_t h);
  Eigen::Vector2f ProjectToFocalPlane(const Eigen::Vector3f& p_W, 
      Eigen::Vector3f* p_C) const;
  Eigen::Vector2i ProjectToImagePlane(const Eigen::Vector3f& p_W,
    Eigen::Vector3f* p_C) const;
  bool IsInImage(const Eigen::Vector3f& p_W, Eigen::Vector3f* p_C,
      Eigen::Vector2i* pI) const;
  Eigen::Vector3f UnprojectToCameraCosy(uint32_t u, uint32_t v, float d) const;
  Eigen::Vector3f UnprojectToWorld(uint32_t u, uint32_t v, float d) const;
  uint32_t GetW() const { return w_;}
  uint32_t GetH() const { return h_;}
  uint32_t GetSize() const { return h_*w_;}
 protected:
  Eigen::Matrix3f R_C_W_;
  Eigen::Vector3f t_C_W_;
  float f_;
  uint32_t w_;
  uint32_t h_;
};
}
