/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>

#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>

namespace po = boost::program_options;
using std::cout;
using std::endl;

#include <jsCore/timer.hpp>

void ComputePcBoundaries(const pcl::PointCloud<pcl::PointXYZ>&
    pc, Eigen::Vector3f& min, Eigen::Vector3f& max) {
  const Eigen::Map<const Eigen::MatrixXf,1,Eigen::OuterStride<> > x = pc.getMatrixXfMap(1, 4, 0); // this works for PointXYZRGBNormal
  const Eigen::Map<const Eigen::MatrixXf,1,Eigen::OuterStride<> > y = pc.getMatrixXfMap(1, 4, 1); // this works for PointXYZRGBNormal
  const Eigen::Map<const Eigen::MatrixXf,1,Eigen::OuterStride<> > z = pc.getMatrixXfMap(1, 4, 2); // this works for PointXYZRGBNormal
  min << x.minCoeff(), y.minCoeff(), z.minCoeff();
  max << x.maxCoeff(), y.maxCoeff(), z.maxCoeff();
}

int main (int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("input,i", po::value<string>(),"path to input")
    ("output,o", po::value<string>(),"path to output")
    ("scale,s", po::value<float>(),"scale for normal extraction search radius")
    ("noiseStd,n", po::value<float>(),"std of isotropic gaussian noise to be added")
    ("outlierRatio,r", po::value<float>(),"ratio of outliers to be achieved for the output pc")
    ("bbFactor,f", po::value<float>(),"bounding box scaling factor for outlier sampling")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  float bbFactor = 2.;
  float noiseStd = 0.;
  float outlier = 0.;
  float scale = 0.1;
  string inputPath = "./file.ply";
  string outputPath = "./out.ply";
  if(vm.count("input")) inputPath = vm["input"].as<string>();
  if(vm.count("output")) outputPath = vm["output"].as<string>();
  if(vm.count("scale")) scale = vm["scale"].as<float>();
  if(vm.count("noiseStd")) noiseStd = vm["noiseStd"].as<float>();
  if(vm.count("outlierRatio")) outlier = vm["outlierRatio"].as<float>();
  if(vm.count("bbFactor")) bbFactor = vm["bbFactor"].as<float>();

  cout<< " extracting from "<<inputPath<<endl;
  cout<< " scale for normal estimation radius: "<<scale<<endl;
  cout<< " noise stdandard deviation: "<<noiseStd<<endl;
  cout<< " final outlier ration : "<<outlier<<endl;
  cout<< " output to "<<outputPath<<endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PLYReader reader;
  if (reader.read(inputPath, *cloud)) 
    std::cout << "error reading " << inputPath << std::endl;
  else
    std::cout << "loaded pc from " << inputPath << ": " << cloud->width << "x"
      << cloud->height << std::endl;

  int w = cloud->width;
  int h = cloud->height;
  cout<<w << "x"<<h<<endl;

  boost::random::mt19937 gen;   
  if (noiseStd > 0.) {
    std::cout << " adding noise with std " << noiseStd << std::endl;
    boost::random::normal_distribution<float> normal(0.0,noiseStd);
    for(uint32_t i=0; i< cloud->size(); ++i) {
      cloud->at(i).x += normal(gen);
      cloud->at(i).y += normal(gen);
      cloud->at(i).z += normal(gen);
    }
  }
  if (outlier > 0.) {
    uint32_t nOutl = outlier/(1.-outlier)*cloud->size();
    std::cout << " adding " << nOutl << " outliers to achieve " 
      << outlier << " ratio" << std::endl;
    Eigen::Vector3f min, max;
    ComputePcBoundaries(*cloud, min, max);
    std::cout << "bounding box corners " << min.transpose() 
      << " -> " << max.transpose() << std::endl;
    std::cout << "inflating bounding box by a factor of " << bbFactor << std::endl;
    Eigen::Vector3f delta = bbFactor*(max - min);
    min -= 0.5*delta;
    max += 0.5*delta;
    std::cout << "bounding box corners " << min.transpose() 
      << " -> " << max.transpose() << std::endl;
    boost::random::uniform_01<float> unif;
    for(uint32_t i=0; i< nOutl; ++i) {
      cloud->push_back(pcl::PointXYZ(unif(gen)*(max(0)-min(0))+min(0), 
            unif(gen)*(max(1)-min(1))+min(1), 
            unif(gen)*(max(2)-min(2))+min(2)));       
    }
  }

  jsc::Timer t0;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
//  ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT); // 31ms
  //ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE); // 23ms
  //ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX); // 47ms
//  ne.setMaxDepthChangeFactor(0.02f);
//  ne.setNormalSmoothingSize(10.0f);
  ne.setInputCloud(cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  ne.setRadiusSearch(scale);

  ne.compute(*normals);
  t0.toc();
  cout<<t0<<endl;

  cout<<normals->width<<" "<<normals->height<<endl;

  pcl::PointCloud<pcl::PointXYZRGBNormal> ptNormals(normals->width,
      normals->height);
  for(uint32_t i=0; i< ptNormals.size(); ++i) {
    ptNormals.at(i).x = cloud->points[i].x;
    ptNormals.at(i).y = cloud->points[i].y;
    ptNormals.at(i).z = cloud->points[i].z;
    if (normals->points[i].normal_x == normals->points[i].normal_x
        && normals->points[i].normal_y == normals->points[i].normal_y
        && normals->points[i].normal_z == normals->points[i].normal_z) {
      ptNormals.at(i).normal_x = normals->points[i].normal_x;
      ptNormals.at(i).normal_y = normals->points[i].normal_y;
      ptNormals.at(i).normal_z = normals->points[i].normal_z;
    } else {
      ptNormals.at(i).normal_x = 0;
      ptNormals.at(i).normal_y = 0;
      ptNormals.at(i).normal_z = 1;
    }
  }
  
  pcl::PLYWriter writer;
  writer.write(outputPath, ptNormals, false, false);

//  return 0;

  // visualize normals
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  viewer.setBackgroundColor (0.3, 0.3, 0.3);
//  viewer.addPointCloud(cloud,"cloud");
//  viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud, normals,10,0.05,"normals");
  viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud, normals,1,0.001,"normals");
 
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return 0;
}
