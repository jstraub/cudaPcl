/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include <cudaPcl/pcl_helpers.hpp>

bool updateCosy(const boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer
    ,const Matrix3f& R, string prefix, float scale)
{
  Matrix3f sR = R*scale;
  viewer->updateSphere (pcl::PointXYZ(sR(0,0),sR(1,0),sR(2,0)), 
      0.1, 1.0, 0.0, 0.0, prefix+"sphere1");
//  viewer->updateSphere (pcl::PointXYZ(-sR(0,0),-sR(1,0),-sR(2,0)), 
//      0.1, 1.0, 0.4, 0.4, prefix+"sphere2");
  viewer->updateSphere (pcl::PointXYZ(sR(0,1),sR(1,1),sR(2,1)), 
      0.1, 0.0, 1.0, 0.0, prefix+"sphere3");
//  viewer->updateSphere (pcl::PointXYZ(-sR(0,1),-sR(1,1),-sR(2,1)), 
//      0.1, 0.4, 1.0, 0.4, prefix+"sphere4");
  return viewer->updateSphere (pcl::PointXYZ(sR(0,2),sR(1,2),sR(2,2)), 
      0.1, 0.0, 0.0, 1.0, prefix+"sphere5");
//  return viewer->updateSphere (pcl::PointXYZ(-sR(0,2),-sR(1,2),-sR(2,2)), 
//      0.1, 0.4, 0.4, 1.0, prefix+"sphere6");
}

void addCosy(const boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
    const Matrix3f& R, string prefix, float scale, int viewport)
{
  Matrix3f sR = R*scale;
  viewer->addSphere (pcl::PointXYZ(sR(0,0),sR(1,0),sR(2,0)), 
      0.1, 1.0, 0.0, 0.0, prefix+"sphere1",viewport);
//  viewer->addSphere (pcl::PointXYZ(-sR(0,0),-sR(1,0),-sR(2,0)), 
//      0.1, 1.0, 0.4, 0.4, prefix+"sphere2",viewport);
  viewer->addSphere (pcl::PointXYZ(sR(0,1),sR(1,1),sR(2,1)), 
      0.1, 0.0, 1.0, 0.0, prefix+"sphere3",viewport);
//  viewer->addSphere (pcl::PointXYZ(-sR(0,1),-sR(1,1),-sR(2,1)), 
//      0.1, 0.4, 1.0, 0.4, prefix+"sphere4",viewport);
  viewer->addSphere (pcl::PointXYZ(sR(0,2),sR(1,2),sR(2,2)), 
      0.1, 0.0, 0.0, 1.0, prefix+"sphere5",viewport);
//  viewer->addSphere (pcl::PointXYZ(-sR(0,2),-sR(1,2),-sR(2,2)), 
//      0.1, 0.4, 0.4, 1.0, prefix+"sphere6",viewport);
}

