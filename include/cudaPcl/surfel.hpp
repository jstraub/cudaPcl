/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include <jsCore/gpuMatrix.hpp>

void surfelRenderGPU(float* s, int32_t N, float f, int32_t w, int32_t h, float *d);
void surfelRenderGPU(double* s, int32_t N, double f, int32_t w, int32_t h, double *d);

namespace cudaPcl {


typedef Eigen::Map<Eigen::Matrix<float,
        Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>,
        0,Eigen::OuterStride<7> > SurfelMap;

struct Surfel {
  float x;
  float y;
  float z;
  float nx;
  float ny;
  float nz;
  float rSq;
  Surfel(float x, float y, float z, float nx, float ny, float nz, float
      rSq) : x(x), y(y), z(z), nx(nx), ny(ny), nz(nz), rSq(rSq) {}
  ~Surfel() {};
  void makeValid();
};

class SurfelStore {
 public:
  SurfelStore() {} ;
  ~SurfelStore() {};

  void AddSurfel(const Surfel& surfel) { ss_.push_back(surfel); }

  const Surfel& GetSurfel(uint32_t i) { return ss_[i]; }
  SurfelMap GetXYZs() { return SurfelMap(&(ss_[0].x),ss_.size(),3); }
  SurfelMap GetNs() { return SurfelMap(&(ss_[0].nx),ss_.size(),3); }
  SurfelMap GetRSqs() { return SurfelMap(&(ss_[0].rSq),ss_.size(),1); }

  cv::Mat Render(const Eigen::Matrix3d& wRc, const Eigen::Vector3d&
      wtc, float f, uint32_t w, uint32_t h);

 private:
  std::vector<Surfel> ss_;
};

void Surfel::makeValid() { 
  float len = sqrt(nx*nx+ny*ny+nz*nz); 
  nx/=len; 
  ny/=len; 
  nz/=len;
}

cv::Mat SurfelStore::Render(const Eigen::Matrix3d& wRc, const Eigen::Vector3d& wtc,
    float f, uint32_t w, uint32_t h) {
  SurfelMap ss (&(ss_[0].x),ss_.size(),7);
  jsc::GpuMatrix<float> ss_d(ss.rows(), ss.cols());
  jsc::GpuMatrix<float> d_d(h,w);
  ss_d.set(ss.data(), ss.rows(), ss.cols(), ss.outerStride(), ss.cols());
  surfelRenderGPU(ss_d.data(), ss_.size(), f, w, h, d_d.data());
  cv::Mat d(h,w,CV_32FC1); 
  uint8_t* data = d.data;
  d_d.get(reinterpret_cast<float*>(data),h,w);
  return d;
}


}
