/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <sstream>
#include <sys/time.h>
//#include <random> // can only use with C++11
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include "cudaPcl/pinhole.h"

#include <boost/program_options.hpp>
#include <boost/random.hpp>

namespace po = boost::program_options;
using std::cout;
using std::endl;

#include <jsCore/timer.hpp>

float ToDeg(float rad) {
  return rad*180./M_PI;
}
float ToRad(float deg) {
  return deg/180.*M_PI;
}
double ToDeg(double rad) {
  return rad*180./M_PI;
}
double ToRad(double deg) {
  return deg/180.*M_PI;
}

void SampleTransformation(float angle, float translation, 
    Eigen::Matrix3f& R, Eigen::Vector3f& t) {
  // Using boost here because C11 and CUDA seem to have troubles.
  timeval tNow; 
  gettimeofday(&tNow, NULL);
  boost::mt19937 gen(tNow.tv_usec);
  boost::normal_distribution<> N(0,1);
  // Sample axis of rotation:
  Eigen::Vector3f axis(N(gen), N(gen), N(gen));
  axis /= axis.norm();
  // Construct rotation:
  Eigen::AngleAxisf aa(ToRad(angle), axis);
  Eigen::Quaternionf q(aa);
  R = q.matrix();
  // Sample translation on sphere with radius translation:
  t = Eigen::Vector3f(N(gen), N(gen), N(gen));
  t *= translation / t.norm();

  std::cout << "sampled random transformation:\n" 
    << R << std::endl << t.transpose() << std::endl;
}

void RenderPointCloud(const pcl::PointCloud<pcl::PointXYZRGBNormal>& pcIn,
    const cudaPcl::Pinhole& c, pcl::PointCloud<pcl::PointXYZRGBNormal>&
    pcOut) {

  // Transform both points by T as well as surface normals by R
  // manually since the standard transformPointCloud does not seem to
  // touch the Surface normals.
  Eigen::MatrixXf d = 1e10*Eigen::MatrixXf::Ones(c.GetH(), c.GetW());
  uint32_t hits = 0;
  for (uint32_t i=0; i<pcIn.size(); ++i) {
    Eigen::Map<const Eigen::Vector3f> pA_W(&(pcIn.at(i).x));
    Eigen::Vector3f p_W = pA_W;
    Eigen::Vector3f p_C;
    Eigen::Vector2i pI;
    if (c.IsInImage(p_W, &p_C, &pI)) { 
      d(pI(1),pI(0)) = std::min(d(pI(1),pI(0)), p_C(2));
      ++hits;
    }
  }
  std::cout << " # hits: " << hits 
    << " for total number of pixels in output camera: " << c.GetSize()
    << " percentage: " << (100.*hits/float(c.GetSize())) << std::endl;

  for (uint32_t i=0; i<c.GetW(); ++i)
    for (uint32_t j=0; j<c.GetH(); ++j) 
      if (d(j,i) < 100.) {
        pcl::PointXYZRGBNormal pB;
        Eigen::Map<Eigen::Vector3f> p_C(&(pB.x));
        p_C = c.UnprojectToCameraCosy(i,j,d(j,i));
        pcOut.push_back(pB);
      }
  std::cout << " output pointcloud size is: " << pcOut.size() 
    << " percentage of input cloud: "
    << (100.*pcOut.size()/float(pcIn.size())) << std::endl;
}

uint32_t VisiblePointsOfPcInCam(const
    pcl::PointCloud<pcl::PointXYZRGBNormal>& pc, 
    const Eigen::Matrix3f& R_PC_W, const Eigen::Vector3f& t_PC_W, const
    cudaPcl::Pinhole& c) {
  uint32_t hits =0;
  for (uint32_t i=0; i<pc.size(); ++i) {
    Eigen::Vector3f p_PC = Eigen::Map<const Eigen::Vector3f> (&(pc.at(i).x));
    Eigen::Vector3f p_W = R_PC_W.transpose() * (p_PC - t_PC_W);
    if (c.IsInImage(p_W, NULL, NULL)) {
      ++hits;
    }
  }
  return hits;
}


int main (int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Render a point cloud from a different pose.");
  desc.add_options()
    ("help,h", "produce help message")
    ("input,i", po::value<string>(),"path to input point cloud")
    ("output,o", po::value<string>(),"path to output transformed point cloud")
    ("angle,a", po::value<double>(),"magnitude of rotation (deg)")
    ("translation,t", po::value<double>(),"magnitude of translation (m)")
    ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  double angle = 10.; // In degree
  double translation = 1.0;
  string inputPath = "./file.ply";
  string outputPath = "./out";
  if(vm.count("input")) inputPath = vm["input"].as<string>();
  if(vm.count("output")) outputPath = vm["output"].as<string>();
  if(vm.count("angle")) angle = vm["angle"].as<double>();
  if(vm.count("translation")) translation = vm["translation"].as<double>();

  std::stringstream ssOutPathA;
  std::stringstream ssOutPathB;
  std::stringstream ssTransformationFile;
  ssOutPathA << outputPath << "_A_angle_" << angle << "_translation_" <<
    translation << ".ply";
  ssOutPathB << outputPath << "_B_angle_" << angle << "_translation_" <<
    translation << ".ply";
  ssTransformationFile << outputPath << "_angle_" << angle <<
    "_translation_" << translation << "_TrueTransformation" << ".csv";

  std::string outputPathA = ssOutPathA.str();
  std::string outputPathB = ssOutPathB.str();
  std::string transformationOutputPath = ssTransformationFile.str();

  // Load point cloud.
  pcl::PointCloud<pcl::PointXYZRGBNormal> pcIn, pcOutA, pcOutB;
  pcl::PLYReader reader;
  if (reader.read(inputPath, pcIn)) 
    std::cout << "error reading " << inputPath << std::endl;
  else
    std::cout << "loaded pc from " << inputPath << ": " << pcIn.width << "x"
      << pcIn.height << std::endl;

  std::cout<< " input pointcloud from "<<inputPath<<std::endl;
  std::cout<< "  angular magnitude "<< angle <<std::endl;
  std::cout<< "  translational magnitude "<< translation <<std::endl;
  std::cout<< " output to "<<outputPathA<<std::endl << " and to " << outputPathB << std::endl;
  std::cout<< " sampled transformation to " << transformationOutputPath << std::endl;

  uint32_t w = 320;
  uint32_t h = 280;
  float f = 540.;
  Eigen::Matrix3f R_A_W, R_B_W;
  Eigen::Vector3f t_A_W, t_B_W;
  // sample pc A
  SampleTransformation(angle, translation, R_A_W, t_A_W);
  cudaPcl::Pinhole camA(R_A_W, t_A_W, f, w, h);
  RenderPointCloud(pcIn, camA, pcOutA);
  // sample pc B
  SampleTransformation(angle, translation, R_B_W, t_B_W);
  cudaPcl::Pinhole camB(R_B_W, t_B_W, f, w, h);
  RenderPointCloud(pcIn, camB, pcOutB);

  uint32_t hitsAinB = VisiblePointsOfPcInCam(pcOutA, R_A_W, t_A_W, camB);
  uint32_t hitsBinA = VisiblePointsOfPcInCam(pcOutB, R_B_W, t_B_W, camA);

  std::cout << "hits of pc A in cam B " << hitsAinB 
    << " that is " << 100.*hitsAinB/float(pcOutA.size()) << "%"
    << std::endl;
  std::cout << "hits of pc B in cam A " << hitsAinB 
    << " that is " << 100.*hitsBinA/float(pcOutB.size()) << "%"
    << std::endl;
  
  pcl::PLYWriter writer;
  writer.write(outputPathA, pcOutA, false, false);
  writer.write(outputPathB, pcOutB, false, false);

  Eigen::Matrix3f R_B_A = R_B_W * R_A_W.transpose();
  Eigen::Vector3f t_B_A = - R_B_W * R_A_W.transpose() * t_A_W + t_B_W;
  Eigen::Quaternionf q(R_B_A);
  Eigen::Vector3f t = t_B_A;

  std::cout << "magnitude of rotation: " << ToDeg(acos(q.w())*2.) 
    << " magnitude of translation: " << t_B_A.norm() << std::endl;

  std::ofstream out(transformationOutputPath.c_str());
  out << "q_w q_x q_y q_z t_x t_y t_z" << std::endl;
  out << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " 
    << t(0) << " " << t(1) << " " << t(2);
  out.close();
}

