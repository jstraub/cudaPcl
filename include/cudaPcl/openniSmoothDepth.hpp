/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <iostream>
#include <stdint.h>
#include <math.h>

#include <jsCore/timer.hpp>
#include <cudaPcl/openniGrabber.hpp>
#include <cudaPcl/openniVisualizer.hpp>

using std::cout;
using std::endl;

using std::min;
using std::max;

namespace cudaPcl {
/*
 * OpenniSmoothDepth smoothes the depth frame using a guided filter making on
 * the CPU in about 30ms.
 */
class OpenniSmoothDepth : public OpenniGrabber
{
  public:
  OpenniSmoothDepth() : OpenniGrabber()
  {};

  virtual ~OpenniSmoothDepth() 
  {};

  virtual void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
  {
//    cout << "depth"<<endl;
    cv::Mat dMap = cv::Mat(h,w,CV_16U,const_cast<uint16_t*>(depth));
    double dMin, dMax;
    cv::minMaxLoc(dMap,&dMin,&dMax);
    cout<<"min "<<dMin<<" max "<<dMax<<endl;
    cout<<dMap.at<float>(0,0)<<endl;
    jsc::Timer t;
    cv::Mat dSmooth = smoothDepthCpu(dMap,0.08*0.08,10);
    t.toctic("smoothing");
    cv::Mat dColor = OpenniVisualizer::colorizeDepth(dMap);
    cv::Mat dSmoothColor = OpenniVisualizer::colorizeDepth(dSmooth);
    cv::imshow("d",dColor);
    cv::imshow("dSmooth",dSmoothColor);
    cv::waitKey(1);
  };  

  virtual cv::Mat smoothDepthCpu(const cv::Mat& depth, double eps, uint32_t w)
  {
    cv::Mat aInt;
    cv::Mat bInt;
    cv::Mat dFlt; 
    cv::Mat haveData;
    cv::Mat Ns;
    cv::Mat dSum; 
    cv::Mat dSqSum;
    cv::Mat a(depth.rows, depth.cols, CV_64F);
    cv::Mat b(depth.rows, depth.cols, CV_64F);
    cv::Mat dSmooth(depth.rows, depth.cols, CV_64F);
    jsc::Timer t;
    depth.convertTo(dFlt,CV_64F,1e-3);

    cv::Mat haveDataU8 = depth > 10;
    haveDataU8.convertTo(haveData,CV_64F,1./255.);

    t.toctic("inits");
    cv::integral(haveData,Ns,CV_64F);

    cv::integral(dFlt,dSum,dSqSum,CV_64F);
    t.toctic("integrals");
    
    // not needed since I alread y have dSqSum
//    cv::Mat prod = dFlt.mul(dFlt);
//    cv::Mat prodInt;
//    cv::integral(prod,prodInt,CV_64F);
    
    for(int32_t i=0; i<dFlt.rows; ++i)
      for(int32_t j=0; j<dFlt.cols; ++j)
        if(haveDataU8.at<uint8_t>(i,j) == 255)
      {
//        cout<<"have "<<i<<" "<<j<<endl; 
        double n = integralGet<double>(Ns,i,j,w); 
        double muG = integralGet<double>(dSum,i,j,w)/n;
        double muD = muG;
        double sqSum = integralGet<double>(dSqSum,i,j,w);
        double sigG = (sqSum - n*muG*muG)/(n-1.);
//;        cout<<"have "<<i<<" "<<j<<": "<<n<<" "<<muG<<" "<<sigG<<endl; 
//        a.at<double>(i,j) = (integralGet<double>(prodInt,i,j,w)/n - muD*muG)
        a.at<double>(i,j) = (sqSum/n - muD*muG) / (sigG + eps);
        b.at<double>(i,j) = muD - muG*a.at<double>(i,j);
      }

    t.toctic("a,b");
    cv::integral(a,aInt,CV_64F);
    cv::integral(b,bInt,CV_64F);
    t.toctic("a,b integrals");
    for(int32_t i=0; i<dFlt.rows; ++i)
      for(int32_t j=0; j<dFlt.cols; ++j)
        if(haveDataU8.at<uint8_t>(i,j) == 255)
      {
        double n = integralGet<double>(Ns,i,j,w); 
        double muA = integralGet<double>(aInt,i,j,w)/n;
        double muB = integralGet<double>(bInt,i,j,w)/n;
        dSmooth.at<double>(i,j) = muA*dFlt.at<double>(i,j) + muB;
      }
    t.toctic("outputComp");
    cout<<t.dtFromInit()<<endl;
    double dMin, dMax;
    cv::minMaxLoc(dSmooth,&dMin,&dMax);
    cout<<"min "<<dMin<<" max "<<dMax<<endl;

    return dSmooth;
  }

  protected:

  template<typename T>
  T integralGet(const cv::Mat& A, int32_t i, int32_t j, int32_t w)
  {
    return A.at<T>(min(i+w,A.rows-1),min(j+w,A.cols-1)) 
      - A.at<T>(min(i+w,A.rows-1),max(j-w,0)) 
      - A.at<T>(max(i-w,0),min(j+w,A.cols-1)) 
      + A.at<T>(max(i-w,0),max(j-w,0));
  };
};

}
