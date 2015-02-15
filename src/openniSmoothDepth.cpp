/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <string>

// Utilities and system includes
#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include <cudaPcl/openniSmoothDepthGpu.hpp>

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
    ("eps,e", po::value<double>(), "sqrt of the epsilon parameter of the guided filter")
    ("B,b", po::value<int>(), "guided filter windows size (size will be (2B+1)x(2B+1))")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  double eps = .5*.5;
  int32_t B = 10;
  if(vm.count("B")) B = vm["B"].as<int>();
  if(vm.count("eps")) eps = vm["eps"].as<double>();

  findCudaDevice(argc,(const char**)argv);
  OpenniSmoothDepthGpu v(eps,B);
  v.run ();
  cout<<cudaDeviceReset()<<endl;
  return (0);
}
