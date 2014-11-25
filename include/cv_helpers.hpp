/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#ifndef CV_HELPERS_HPP_INCLUDED
#define CV_HELPERS_HPP_INCLUDED

#include<opencv2/core/core.hpp>

//void normalize(cv::Mat& I);
//void showNans(cv::Mat& I);
//void showZeros(cv::Mat& I);

inline void normalizeImg(cv::Mat& I)
{
  float maxI = -99999.0f;
  float minI = 99999.0f;
  for(int i=0; i<I.cols; ++i)
    for(int j=0; j<I.rows; ++j)
    {
      if(maxI < I.at<float>(j,i))
      {
        maxI = I.at<float>(j,i) ;
      }else if(minI > I.at<float>(j,i))
      {
        minI = I.at<float>(j,i);
      }
    }
  for(int i=0; i<I.cols; ++i)
    for(int j=0; j<I.rows; ++j)
    {
      I.at<float>(j,i) = (I.at<float>(j,i)-minI)/(maxI-minI);
    }
}
inline void showNans(cv::Mat& I)
{
  for(int i=0; i<I.cols; ++i)
    for(int j=0; j<I.rows; ++j)
      if( I.at<float>(j,i)!=I.at<float>(j,i))
      {
        I.at<float>(j,i) = 1.0;
      }else{
        I.at<float>(j,i) = 0.0;
      }
}

inline void showZeros(cv::Mat& I)
{
  for(int i=0; i<I.cols; ++i)
    for(int j=0; j<I.rows; ++j)
      if( I.at<float>(j,i)==0.0f)
      {
        I.at<float>(j,i) = 1.0;
      }else{
        I.at<float>(j,i) = 0.0;
      }
}

#endif
