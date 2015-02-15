/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

/*
 * Class for obtaining frames via freenect2 via two callbacks: depth_cb as well as
 * rgb_cb
 */
class Freenect2Grabber
{
  libfreenect2::Freenect2 freenect2_;
  libfreenect2::Freenect2Device *dev_;
  bool shutdown_;

  public:
    Freenect2Grabber() : freenect2_(), shutdown_(false)
    {
      // create a new grabber for Freenect2 devices
      dev_ = freenect2_.openDefaultDevice();
      if(dev_ == 0)
      {
        std::cout << "no device connected or failure opening the default one!"
          << std::endl;
      }

//      signal(SIGINT,sigint_handler);
      shutdown_ = false;
    };
    virtual ~Freenect2Grabber()
    {
      delete dev_;
    };

    void sigint_handler(int s) { shutdown_ = true; }
   
    virtual void depth_cb(float * depth, uint32_t w, uint32_t h)
    {
        cv::imshow("depth", cv::Mat(h, w, CV_32FC1, depth) / 4500.0f);
    };  

    virtual void ir_cb(float * ir, uint32_t w, uint32_t h)
    {
        cv::imshow("ir", cv::Mat(h, w, CV_32FC1, ir) / 20000.0f);
    };  

    virtual void rgb_cb(uint8_t* rgb, uint32_t w, uint32_t h)
    {
        cv::imshow("rgb", cv::Mat(h, w, CV_8UC3, rgb));
    };  

    virtual void run ()
    {
      this->run_impl();
//      this->run_cleanup_impl();
    }

protected:

    void run_impl ()
    {

      libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color 
          | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
      libfreenect2::FrameMap frames;

      dev_->setColorFrameListener(&listener);
      dev_->setIrAndDepthFrameListener(&listener);
      dev_->start();

      std::cout << "device serial: " << dev_->getSerialNumber() << std::endl;
      std::cout << "device firmware: " << dev_->getFirmwareVersion() << std::endl;

      while(!shutdown_)
      {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

        depth_cb((float*)depth->data,depth->width,depth->height);
        ir_cb((float*)ir->data,ir->width,ir->height);
        rgb_cb(rgb->data,rgb->width,rgb->height);


        int key = cv::waitKey(1);
        // shutdown on escape
        shutdown_ = shutdown_ || (key > 0 && ((key & 0xFF) == 27)); 

        listener.release(frames);
      }
      dev_->stop();
      dev_->close();
    }

    cv::Mat rgb_;
    uint32_t w_,h_;
  private:
};
