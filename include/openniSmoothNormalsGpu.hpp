/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <openniSmoothDepthGpu.hpp>
#include <normalExtractSimpleGpu.hpp>

#include <opencvHelper.hpp>

/*
 * OpenniSmoothNormalsGpu smoothes the depth frame using a guided filter and
 * computes surface normals from it also on the GPU.
 *
 * Needs the focal length of the depth camera f_d and the parameters for the
 * guided filter eps as well as the filter size B.
 */
class OpenniSmoothNormalsGpu : public OpenniSmoothDepthGpu
{
  public:
  OpenniSmoothNormalsGpu(double f_d, double eps, uint32_t B, bool compress=false) 
    : OpenniSmoothDepthGpu(eps,B), f_d_(f_d), normalExtract(NULL) ,compress_(compress)
  { };

  virtual ~OpenniSmoothNormalsGpu() 
  { 
    if(normalExtract) delete normalExtract;
  };

  virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
  {
    if(!this->depthFilter)
    {
      this->depthFilter = new DepthGuidedFilterGpu<float>(w,h,eps_,B_);
      normalExtract = new NormalExtractSimpleGpu<float>(f_d_,w,h,compress_);
    }
    cv::Mat dMap = cv::Mat(h,w,CV_16U,const_cast<uint16_t*>(depth));

//    Timer t;
    this->depthFilter->filter(dMap);
//    t.toctic("smoothing");
    normalExtract->computeGpu(this->depthFilter->getDepthDevicePtr(),w,h);
//    t.toctic("normals");
    normals_cb(normalExtract->d_normalsImg(), normalExtract->d_haveData(),w,h);
//    t.toctic("normals callback");
  };  

  /* callback with smoothed normals 
   *
   * Note that the pointers are to GPU memory as indicated by the "d_" prefix.
   */
  virtual void normals_cb(float* d_normalsImg, uint8_t* d_haveData, uint32_t w, uint32_t h)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr nDispPtr = normalExtract->normalsPc();

    boost::mutex::scoped_lock updateLock(updateModelMutex);
    if(compress_){int32_t nComp =0; normalsComp_ = normalExtract->normalsComp(nComp); }
    nDisp_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr( new pcl::PointCloud<pcl::PointXYZRGB>(*nDispPtr));
    normalsImg_ = normalExtract->normalsImg();


    if(true)
    {
      static int frameN = 0;
      if(frameN==0) system("mkdir ./normals/");
      char path[100];
      // Save the image data in binary format
      sprintf(path,"./normals/%05d.bin",frameN ++);
      if(compress_)
        imwriteBinary(std::string(path), normalsComp_);
      else
        imwriteBinary(std::string(path), normalsImg_);
    }

    this->update_ = true;
  };

  virtual void visualizeD();
  virtual void visualizePc();

  protected:
  double f_d_;
  NormalExtractSimpleGpu<float> * normalExtract;
  bool compress_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr nDisp_;
  cv::Mat normalsImg_;
  cv::Mat normalsComp_;
};
// ------------------------ impl -----------------------------------------
void OpenniSmoothNormalsGpu::visualizeD()
{
  if (this->depthFilter)
  {
    cv::Mat dSmooth = this->depthFilter->getOutput();
    this->dColor_ = colorizeDepth(dSmooth,0.3,4.0);
    cv::imshow("d",dColor_);
  }
};

void OpenniSmoothNormalsGpu::visualizePc()
{
  //copy again
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr nDisp(
      new pcl::PointCloud<pcl::PointXYZRGB>(*nDisp_));
//  cv::Mat nI(nDisp->height,nDisp->width,CV_32FC3); 
//  for(uint32_t i=0; i<nDisp->width; ++i)
//    for(uint32_t j=0; j<nDisp->height; ++j)
//    {
//      // nI is BGR but I want R=x G=y and B=z
//      nI.at<cv::Vec3f>(j,i)[0] = (1.0f+nDisp->points[i+j*nDisp->width].z)*0.5f; // to match pc
//      nI.at<cv::Vec3f>(j,i)[1] = (1.0f+nDisp->points[i+j*nDisp->width].y)*0.5f; 
//      nI.at<cv::Vec3f>(j,i)[2] = (1.0f+nDisp->points[i+j*nDisp->width].x)*0.5f; 
//      nDisp->points[i+j*nDisp->width].rgb=0;
//    }
  cv::Mat nI (normalsImg_.rows,normalsImg_.cols, CV_8UC3);
  cv::Mat nIRGB(normalsImg_.rows,normalsImg_.cols,CV_8UC3);                              
  normalsImg_.convertTo(nI, CV_8UC3, 127.5,127.5);
  cv::cvtColor(nI,nIRGB,CV_RGB2BGR);
  cv::imshow("normals",nIRGB);             

  if (compress_)  cv::imshow("dcomp",normalsComp_);

  this->pc_ = nDisp;
//  this->pc_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(nDisp);
  
  if(!this->viewer_->updatePointCloud(pc_, "pc"))
    this->viewer_->addPointCloud(pc_, "pc");
}

