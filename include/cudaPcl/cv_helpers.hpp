/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#ifndef CV_HELPERS_HPP_INCLUDED
#define CV_HELPERS_HPP_INCLUDED

#include <fstream>
#include <opencv2/core/core.hpp>

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

inline void imwriteBinary(std::string path, cv::Mat img)
{
  std::fstream os(path.data(),std::ios::out|std::ios::trunc|std::ios::binary);
  os << (int)img.rows << " " << (int)img.cols << " " << (int)img.type() << " ";
  os.write((char*)img.data,img.step.p[0]*img.rows);
  os.close();
}

inline cv::Mat imreadBinary(std::string path)
{
  std::fstream is(path.data(),std::ios::in|std::ios::binary);
  if(!is.is_open()) 
  {
    cerr<<"could not open "<<path<<" for reading"<<endl;
    return cv::Mat();
  }
  int rows,cols,type;
  is >> rows; is.ignore(1);
  is >> cols; is.ignore(1);
  is >> type; is.ignore(1);
  cv::Mat img;
  img.create(rows,cols,type);
  is.read((char*)img.data,img.step.p[0]*img.rows);
  is.close();
  return img;
}

#endif
