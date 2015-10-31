/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include "cudaPcl/pinhole.h"

namespace cudaPcl {
Pinhole::Pinhole(const Eigen::Matrix3f& R_C_W, const Eigen::Vector3f&
    t_C_W, float f, uint32_t w, uint32_t h)
  : R_C_W_(R_C_W), t_C_W_(t_C_W), f_(f), w_(w), h_(h) {
}

Eigen::Vector2f Pinhole::ProjectToFocalPlane(const Eigen::Vector3f& p_W,
    Eigen::Vector3f* p_C) const {
  Eigen::Vector3f p_C_ = R_C_W_ * p_W + t_C_W_;
  Eigen::Vector2f pF = p_C_.topRows<2>();
  pF /= p_C_(2);
  if (p_C) *p_C = p_C_;
  return pF;
}

Eigen::Vector2i Pinhole::ProjectToImagePlane(const Eigen::Vector3f& p_W,
    Eigen::Vector3f* p_C) const {
  Eigen::Vector2f pF;
  if (p_C)
    pF = ProjectToFocalPlane(p_W, p_C);
  else
    pF = ProjectToFocalPlane(p_W, NULL);
  Eigen::Vector2i pI;
  pI(0) = floor(pF(0)*f_ + w_/2);
  pI(1) = floor(pF(1)*f_ + h_/2);
  return pI;
}

bool Pinhole::IsInImage(const Eigen::Vector3f& p_W, 
    Eigen::Vector3f* p_C, Eigen::Vector2i* pI) const {
  Eigen::Vector2i pI_ = ProjectToImagePlane(p_W, p_C);
  if (pI) *pI = pI_;
  if (0 <= pI_(0) && pI_(0) < w_ && 0 <= pI_(1) && pI_(1) < h_)
    return true;
  else 
    return false;
}

Eigen::Vector3f Pinhole::UnprojectToCameraCosy(uint32_t u, uint32_t v,
    float d) const {
  Eigen::Vector3f p_C;
  p_C(0) = (u-(w_-1)*0.5) * d / f_;
  p_C(1) = (v-(h_-1)*0.5) * d / f_;
  p_C(2) = d;
  return p_C;
}

Eigen::Vector3f Pinhole::UnprojectToWorld(uint32_t u, uint32_t v,
    float d) const {
  return R_C_W_.transpose() * (UnprojectToCameraCosy(u,v,d) - t_C_W_);
}
}
