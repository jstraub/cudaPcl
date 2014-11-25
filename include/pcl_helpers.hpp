/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <string>

using namespace Eigen;
using std::string;

bool updateCosy(const boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer
    ,const Matrix3f& R, string prefix="cosy", float scale=1.0f);

void addCosy(const boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
    const Matrix3f& R, string prefix="cosy", float scale=1.0f, int viewport=0);

