#pragma once

#include <fstream>

inline void imwriteBinary(std::string path, cv::Mat img)
{
  std::ofstream os(,std::ios::out|std::ios::trunc|std::ios::binary);
  os << (int)img.rows << " " << (int)img.cols << " " << (int)img.type() << " ";
  os.write((char*)img.data,img.step.p[0]*img.rows);
  os.close();
}

inline cv::Mat imreadBinary(std::string path)
{
  std::ifstream is(path,std::ios::in|std::ios::binary);
  if(!is.is_open())
    return false;
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
