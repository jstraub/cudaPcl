/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <vector>
#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <anytReconstruct/surfel.hpp>

using namespace anyt;

cv::Mat colorizeDepth(const cv::Mat& dMap, float min, float max)
{
  cv::Mat d8Bit = cv::Mat::zeros(dMap.rows,dMap.cols,CV_8UC1);
  cv::Mat dColor;
  dMap.convertTo(d8Bit,CV_8UC1, 255./(max-min));
  cv::applyColorMap(d8Bit,dColor,cv::COLORMAP_JET);
  return dColor;
}

int main(int argc, char** argv) {
  
  Surfel s0(0,0,1, 1,1,1,0.2);
  Surfel s1(0.5,0.5,1, 4,4,4,0.5);
  s0.makeValid();
  s1.makeValid();

  std::vector<Surfel> ss_;
  ss_.push_back(s0);
  ss_.push_back(s1);

  Eigen::Map<Eigen::Matrix<float,7,1>> s0raw(&(ss_[0].x));
  std::cout << s0raw.transpose() << std::endl;

  Eigen::Map<Eigen::Matrix<float,7,1>> s1raw(&(ss_[1].x));
  std::cout << s1raw.transpose() << std::endl;

  Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::OuterStride<7>> xyz_(&(ss_[0].x), 2,3);
  std::cout << xyz_ << std::endl;

  Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::OuterStride<7>> n(&(ss_[0].nx), 2,3);
  std::cout << n << std::endl;

  Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, 0, Eigen::OuterStride<7>> rSq(&(ss_[0].rSq), 2,1);
  std::cout << rSq << std::endl;

  SurfelStore ss;
  ss.AddSurfel(s0);
  ss.AddSurfel(s1);
  std::cout << ss.GetXYZs() << std::endl;
  std::cout << ss.GetNs() << std::endl;
  std::cout << ss.GetRSqs() << std::endl;

  std::cout << "GPU tests" << std::endl;

  Eigen::Matrix3d wRc = Eigen::Matrix3d::Identity();
  Eigen::Vector3d wtc = Eigen::Vector3d::Zero();
  cv::Mat d = ss.Render(wRc, wtc, 540, 640, 480);
  cv::Mat dCol = colorizeDepth(d, 0.3,4.0);
  cv::imshow("d", dCol);
  cv::waitKey(0);

};
