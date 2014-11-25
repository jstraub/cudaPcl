/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/common/time.h>

typedef pcl::PointXYZ MyPoint;
//typedef pcl::PointXYZRGBA MyPoint;

class SimpleOpenNIProcessor
{
public:
 
  pcl::visualization::CloudViewer viewer;
  bool grabPC;

  SimpleOpenNIProcessor(): viewer ("PCL OpenNI Viewer")
  {
    grabPC = false;
  }

  void cloud_cb_ (const pcl::PointCloud<MyPoint>::ConstPtr &cloud)
  {
    static unsigned count = 0;
    static double last = pcl::getTime ();
    if (++count == 30)
    {
      double now = pcl::getTime ();
      std::cout << "distance of center pixel :" << cloud->points [(cloud->width >> 1) * (cloud->height + 1)].z << " mm. Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
      count = 0;
      last = now;
    }

    if (!viewer.wasStopped())
      viewer.showCloud (cloud);

    if (grabPC){
      pcl::io::savePCDFileASCII ("test_pcd.pcd", pcl::PointCloud<MyPoint>(*cloud));
      std::cerr << "Saved " << cloud->points.size () << " data points to test_pcd.pcd." << std::endl;
      grabPC = false;
    }
  }
  
  void run ()
  {
    // create a new grabber for OpenNI devices
    pcl::Grabber* interface = new pcl::OpenNIGrabber();

    // make callback function from member function
    boost::function<void (const pcl::PointCloud<MyPoint>::ConstPtr&)> f =
      boost::bind (&SimpleOpenNIProcessor::cloud_cb_, this, _1);

    // connect callback function for desired signal. In this case its a point cloud with color values
    boost::signals2::connection c = interface->registerCallback (f);

    // start receiving point clouds
    interface->start ();

    // wait until user quits program with Ctrl-C, but no busy-waiting -> sleep (1);
    while (!viewer.wasStopped())
      boost::this_thread::sleep (boost::posix_time::seconds (1));

    // stop the grabber
    interface->stop ();
  }

  void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                        void* viewer_void)
  {
    std::cerr<<"keyboard event: "<<event.getKeySym ()<<endl;
    if (event.getKeySym () == "s" && event.keyDown ())
    {
      std::cout << "s was pressed => grab PointCloud" << std::endl;
      grabPC = true;
    }
  }

};

int main ()
{
  SimpleOpenNIProcessor v;
  v.run ();
  return (0);
}

