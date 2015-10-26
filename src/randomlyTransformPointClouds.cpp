/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <sstream>
//#include <random> // can only use with C++11
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

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

int main (int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Apply random transformation to input point cloud.\n The affine transformation is sampled as: (1) sample random rotation axis (uniformly on the sphere) and use specified rotation magnitude to obtain rotation and (2) sample random rotation uniformly on the sphere with radius spcified in the translation argument.\nAllowed options");
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
  string outputPath = "./out.ply";
  if(vm.count("input")) inputPath = vm["input"].as<string>();
  if(vm.count("output")) outputPath = vm["output"].as<string>();
  if(vm.count("angle")) angle = vm["angle"].as<double>();
  if(vm.count("translation")) translation = vm["translation"].as<double>();

  std::stringstream ssOutPath;
  ssOutPath << outputPath << "_angle_" << angle << "_translation_" <<
    translation << ".ply";
  outputPath = ssOutPath.str();

  std::stringstream ssTransformationFile;
  ssTransformationFile << outputPath << "_TrueTransformation" <<
    "_angle_" << angle << "_translation_" << translation << ".csv";
  std::string transformationOutputPath = ssTransformationFile.str();

  // Load point cloud.
  pcl::PointCloud<pcl::PointXYZRGBNormal> pcIn, pcOut;
  pcl::PLYReader reader;
  if (reader.read(inputPath, pcIn)) 
    std::cout << "error reading " << inputPath << std::endl;
  else
    std::cout << "loaded pc from " << inputPath << ": " << pcIn.width << "x"
      << pcIn.height << std::endl;

  std::cout<< " input pointcloud from "<<inputPath<<std::endl;
  std::cout<< "  angular magnitude "<< angle <<std::endl;
  std::cout<< "  translational magnitude "<< translation <<std::endl;
  std::cout<< " output to "<<outputPath<<std::endl;
  std::cout<< " sampled transformation to " << transformationOutputPath << std::endl;

  // Using boost here because C11 and CUDA seem to have troubles.
  boost::mt19937 gen;
  boost::normal_distribution<> N(0,1);
  // Sample axis of rotation:
  Eigen::Vector3f axis(N(gen), N(gen), N(gen));
  axis /= axis.norm();
  // Construct rotation:
  Eigen::AngleAxisf aa(ToRad(angle), axis);
  Eigen::Quaternionf q(aa);
  // Sample translation on sphere with radius translation:
  Eigen::Vector3f t(N(gen), N(gen), N(gen));
  t *= translation / t.norm();

  Eigen::Affine3f T = Eigen::Affine3f::Identity();
  T.translation() = t;
  T.rotate(q);

  std::cout << "sampled random transformation:\n" 
    << T.matrix() << std::endl;

  // Transform both points by T as well as surface normals by R
  // manually since the standard transformPointCloud does not seem to
  // touch the Surface normals.
  pcOut = pcIn;
  for (uint32_t i=0; i<pcOut.size(); ++i) {
    Eigen::Map<Eigen::Vector3f> p_map(&(pcOut.at(i).x));
    Eigen::Vector4f p(pcOut.at(i).x, pcOut.at(i).x, pcOut.at(i).z,1.);
    p_map = (T*p).topRows<3>();
    Eigen::Map<Eigen::Vector3f> n(pcOut.at(i).normal);
    n = T.rotation()*n;
  }
  
  pcl::PLYWriter writer;
  writer.write(outputPath, pcOut, false, false);

  std::ofstream out(transformationOutputPath.c_str());
  out << "q_w q_x q_y q_z t_x t_y t_z" << std::endl;
  out << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " 
    << t(0) << " " << t(1) << " " << t(2);
  out.close();
}

