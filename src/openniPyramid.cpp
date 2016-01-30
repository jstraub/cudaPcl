/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <string>

// Utilities and system includes
#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include <cudaPcl/openniPyramid.hpp>

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
    ("nLvls,l", po::value<int>(), "number of pyramid levels")
    ("method,m", po::value<int>(), "method for downsampling")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  int32_t method = 0;
  int32_t nLvls = 4;
  if(vm.count("nLvls")) nLvls = vm["nLvls"].as<int>();
  if(vm.count("method")) method = vm["method"].as<int>();

//  findCudaDevice(argc,(const char**)argv);
  cudaPcl::OpenniPyramid v(nLvls, method);
  v.run ();
//  cout<<cudaDeviceReset()<<endl;
  return 0;
};
