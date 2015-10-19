/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <fstream>
#include <string>

// Utilities and system includes
//#include <helper_functions.h>
#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <cudaPcl/depthGuidedFilter.hpp>
#include <cudaPcl/normalExtractSimpleGpu.hpp>

#include <Eigen/Dense>
#include <pcl/io/ply_io.h>

namespace po = boost::program_options;
using namespace Eigen;
using std::cout;
using std::endl;


int main (int argc, char** argv)
{

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("input,i", po::value<string>(), "path to input depth image (16bit .png)")
    ("output,o", po::value<string>(), "path to output surface normals (csv)")
    ("f_d,f", po::value<double>(), "focal length of depth camera")
    ("eps,e", po::value<double>(), "sqrt of the epsilon parameter of the guided filter")
    ("B,b", po::value<int>(), "guided filter windows size (size will be (2B+1)x(2B+1))")
    ("compress,c", "compress the computed normals")
    ("display,d", "display the computed normals")
//    ("out,o", po::value<std::string>(), "output path where surfae normal images are saved to")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  double f_d = 540.;
  double eps = 0.2*0.2;
  int32_t B = 10;
  bool compress = false;
  bool display = false;
  string inputPath = "";
  string outputPath = "";
  if(vm.count("input")) inputPath = vm["input"].as<string>();
  if(vm.count("output")) outputPath = vm["output"].as<string>();
  if(vm.count("f_d")) f_d = vm["f_d"].as<double>();
  if(vm.count("eps")) eps = vm["eps"].as<double>();
  if(vm.count("B")) B = vm["B"].as<int>();
  if(vm.count("compress")) compress = true;
  if(vm.count("display")) display = true;
//  std::string outPath = ""
//  if(vm.count("out")) outPath = vm["out"].as<std::string>();

  if(inputPath.compare("") == 0)
  {
    cout<<"provide an input to a depth image file"<<endl;
    exit(1);
  }
  
  findCudaDevice(argc,(const char**)argv);

  cv::Mat depth = cv::imread(inputPath,CV_LOAD_IMAGE_ANYDEPTH);
  uint32_t w = depth.cols;
  uint32_t h = depth.rows;

  cudaPcl::DepthGuidedFilterGpu<float>* depthFilter = 
    new cudaPcl::DepthGuidedFilterGpu<float>(w,h,eps,B);
  cudaPcl::NormalExtractSimpleGpu<float>* normalExtract = 
    new cudaPcl::NormalExtractSimpleGpu<float>(f_d,w,h,compress);

  depthFilter->filter(depth);
  normalExtract->computeGpu(depthFilter->getDepthDevicePtr(),w,h);
  cv::Mat normalsImg = normalExtract->normalsImg();

  if(outputPath.compare("") != 0)
  {
    cout<<"output writen to "<<outputPath<<endl;
    std::ofstream out(outputPath.data(), std::ofstream::out | std::ofstream::binary);
    out<<h<<" "<<w<<" "<<3<<endl;
    char* data = reinterpret_cast<char*>(normalsImg.data);
    out.write(data, w*h*3*sizeof(float));
//    for (uint32_t i=0; i<h; ++i)
//      for (uint32_t j=0; j<w; ++j)
//        out<< normalsImg.at<cv::Vec3f>(i,j)[0] << " "
//          << normalsImg.at<cv::Vec3f>(i,j)[1] << " "
//          << normalsImg.at<cv::Vec3f>(i,j)[2] <<endl;
    out.close();
  }

  if(display)
  {
    cv::imshow("d",depth);
    cv::imshow("n",normalsImg);
    cv::waitKey(0);
  }

  cout<<cudaDeviceReset()<<endl;
  return (0);
}


