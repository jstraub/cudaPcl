/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */


#include <iostream>
#include <string>

// Utilities and system includes
//#include <helper_functions.h>
#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <Eigen/Dense>
#include <cudaPcl/openniSmoothDepth.hpp>
#include <cudaPcl/openniSmoothDepthGpu.hpp>
#include <cudaPcl/openniSmoothNormalsGpu.hpp>

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
    ("f_d,f", po::value<double>(), "focal length of depth camera")
    ("eps,e", po::value<double>(), "sqrt of the epsilon parameter of the guided filter")
    ("B,b", po::value<int>(), "guided filter windows size (size will be (2B+1)x(2B+1))")
    ("compress,c", "compress the computed normals")
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
  if(vm.count("f_d")) f_d = vm["f_d"].as<double>();
  if(vm.count("eps")) eps = vm["eps"].as<double>();
  if(vm.count("B")) B = vm["B"].as<int>();
  if(vm.count("compress")) compress = true;
//  std::string outPath = ""
//  if(vm.count("out")) outPath = vm["out"].as<std::string>();

  findCudaDevice(argc,(const char**)argv);
  cudaPcl::OpenniSmoothNormalsGpu v(f_d, eps, B, compress);

  if(true)
  { // load a specific image and process
    cv::Mat depth = cv::imread("../../python/table_0_d.png",CV_LOAD_IMAGE_ANYDEPTH);
    for(uint32_t t=0; t<1; ++t)
    {
      v.depth_cb((uint16_t*)depth.data,depth.cols,depth.rows);
      v.visualizeD();
      v.visualizeNormals();
      cv::waitKey(0);
    }
  }

  // run the grabber
  v.run ();
  
  cout<<cudaDeviceReset()<<endl;
  return (0);
}


