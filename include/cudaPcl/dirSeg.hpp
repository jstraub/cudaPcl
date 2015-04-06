/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <jsCore/timerLog.hpp>
#include <cudaPcl/normalExtractSimpleGpu.hpp>
#include <cudaPcl/depthGuidedFilter.hpp>
namespace cudaPcl{

void projectDirections(cv::Mat& I, const MatrixXf&
    dirs, double f_d, const Matrix<uint8_t,Dynamic,Dynamic>& colors)
{
  double scale = 0.1;
  VectorXf p0(3); p0 << 0.35,0.25,1;
  double u0 = p0(0)/p0(2)*f_d + 320.;
  double v0 = p0(1)/p0(2)*f_d + 240.;
  for(uint32_t k=0; k < dirs.cols(); ++k)
  {
    VectorXf p1 = p0 + dirs.col(k)*scale;
    double u1 = p1(0)/p1(2)*f_d + 320.;
    double v1 = p1(1)/p1(2)*f_d + 240.;
    cv::line(I, cv::Point(u0,v0), cv::Point(u1,v1),
        CV_RGB(colors(k,0),colors(k,1),colors(k,2)), 2, CV_AA);

    double arrowLen = 10.;
    double angle = atan2(v1-v0,u1-u0);

    double ru1 = u1 - arrowLen*cos(angle + M_PI*0.25);
    double rv1 = v1 - arrowLen*sin(angle + M_PI*0.25);
    cv::line(I, cv::Point(u1,v1), cv::Point(ru1,rv1),
        CV_RGB(colors(k,0),colors(k,1),colors(k,2)), 2, CV_AA);
    ru1 = u1 - arrowLen*cos(angle - M_PI*0.25);
    rv1 = v1 - arrowLen*sin(angle - M_PI*0.25);
    cv::line(I, cv::Point(u1,v1), cv::Point(ru1,rv1),
        CV_RGB(colors(k,0),colors(k,1),colors(k,2)), 2, CV_AA);
  }
  cv::circle(I, cv::Point(u0,v0), 2, CV_RGB(0,0,0), 2, CV_AA);
};

class DirSeg
{
  public:
    DirSeg(const cudaPcl::CfgSmoothNormals& cfgNormals, string pathOut);
    virtual ~DirSeg();

    /*
     * compute dirSeg from depth image stored on the CPU
     */
    virtual void compute(const cv::Mat& depth)
    {compute((uint16_t*)depth.data, depth.cols, depth.rows);};
    virtual void compute(const uint16_t* depth, uint32_t w, uint32_t h);
      /*
       * compute dirSeg from surface normals stored on the CPU
       */
    virtual void compute(const pcl::PointCloud<pcl::Normal>& normals);
    /*
     * compute dirSeg from point cloud
     */
    virtual void compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr & pc);

    virtual MatrixXf centroids() = 0;
    virtual const VectorXu& labels();
    cv::Mat labelsImg();
    cv::Mat normalsImg();
    cv::Mat smoothNormalsImg();
    cv::Mat smoothDepthImg();
    cv::Mat smoothDepth(){ return this->depthFilter_->getOutput();};
    cv::Mat normalsImgRaw(){ return normalExtract_->normalsImg();};
    cv::Mat overlaySeg(cv::Mat img);

protected:
    const static uint32_t K_MAX = 10;

    bool haveLabels_;
    jsc::TimerLog tLog_;

    cudaPcl::CfgSmoothNormals cfgNormals_;
    uint32_t w_, h_;

    uint32_t K_;
    VectorXu z_;

    boost::shared_ptr<jsc::ClDataGpuf> cld_; // clustered data
    cudaPcl::DepthGuidedFilterGpu<float> *depthFilter_;
    cudaPcl::NormalExtractSimpleGpu<float> *normalExtract_;

    Matrix<uint8_t,Dynamic,Dynamic> dirCols_;
    void fillJET();
    float JET_r_[256];
    float JET_g_[256];
    float JET_b_[256];

    /*
     * runs the actual compute; assumes that normalExtract_ contains
     * the normals on GPU already.
     */
    virtual void compute_() = 0;
    /* get lables in input format */
    virtual void getLabels_() = 0;
};

// ------------------- impl --------------------------------------
DirSeg::DirSeg(const cudaPcl::CfgSmoothNormals& cfgNormals, string
    pathOut)
  : haveLabels_(false),
  tLog_(pathOut+string("./timer.log"),3,10,"TimerLog"),
  cfgNormals_(cfgNormals),
  depthFilter_(NULL), normalExtract_(NULL)
{
  fillJET();
  dirCols_ = Matrix<uint8_t,Dynamic,Dynamic>(K_MAX,3);
  for(uint32_t k=0; k<K_MAX; ++k)
  {
    dirCols_(k,0) = static_cast<uint8_t>(floor(JET_r_[k*255/K_MAX]*255));
    dirCols_(k,1) = static_cast<uint8_t>(floor(JET_g_[k*255/K_MAX]*255));
    dirCols_(k,2) = static_cast<uint8_t>(floor(JET_b_[k*255/K_MAX]*255));
  }
};

DirSeg::~DirSeg()
{
  delete depthFilter_;
  delete normalExtract_;
};

void DirSeg::compute(const uint16_t* depth, uint32_t w, uint32_t h)
{
  w_ = w; h_ = h;
  tLog_.tic(-1); // reset all timers
  if(!depthFilter_)
  {
    depthFilter_ = new cudaPcl::DepthGuidedFilterGpu<float>(w_,h_,
        cfgNormals_.eps,cfgNormals_.B);
    normalExtract_ = new
      cudaPcl::NormalExtractSimpleGpu<float>(cfgNormals_.f_d,
          w_,h_,cfgNormals_.compress);
  }
  cout<<" -- guided filter for depth image "<<w_<<" "<<h_<<endl;
  cv::Mat dMap(h_,w_,CV_16U,const_cast<uint16_t*>(depth));
  cout<<dMap.rows<<" "<<dMap.cols<<endl;
  depthFilter_->filter(dMap);
  tLog_.toctic(0,1);
  cout<<" -- extract surface normals on GPU"<<endl;
  normalExtract_->computeGpu(depthFilter_->getDepthDevicePtr(),w_,h_);
  compute_();
};

void DirSeg::compute(const pcl::PointCloud<pcl::Normal>& normals)
{
  // pcl::Normal is a float[4] array per point. the 4th entry is the
  // curvature
  w_ = normals.width; h_ = normals.height;
  tLog_.tic(-1); // reset all timers
  tLog_.toctic(0,1);
  if(!normalExtract_)
  {
    normalExtract_ = new
      cudaPcl::NormalExtractSimpleGpu<float>(cfgNormals_.f_d,
          w_,h_,cfgNormals_.compress);
  }
  normalExtract_->setNormalsCpu(normals);
  compute_();
}

void DirSeg::compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr & pc)
{
  // pcl::Normal is a float[4] array per point. the 4th entry is the
  // curvature
  w_ = pc->width; h_ = pc->height;
  tLog_.tic(-1); // reset all timers
  tLog_.toctic(0,1);
  if(!normalExtract_)
  {
    normalExtract_ = new
      cudaPcl::NormalExtractSimpleGpu<float>(cfgNormals_.f_d,
          w_,h_,cfgNormals_.compress);
  }
  normalExtract_->compute(pc);
  compute_();
};

const VectorXu& DirSeg::labels()
{
  if(!haveLabels_)
  {
    if(z_.rows() != w_*h_) z_.resize(w_*h_);
    getLabels_();
    haveLabels_ = true;
  }
  return z_;
};

cv::Mat DirSeg::labelsImg()
{
  labels();
  cv::Mat zIrgb(h_,w_,CV_8UC3);
  for(uint32_t i=0; i<w_; i+=1)
    for(uint32_t j=0; j<h_; j+=1)
      if(z_(w_*j +i) < K_)
      {
        uint32_t idz = z_(w_*j +i);
        zIrgb.at<cv::Vec3b>(j,i)[0] = dirCols_(idz,0);
        zIrgb.at<cv::Vec3b>(j,i)[1] = dirCols_(idz,1);
        zIrgb.at<cv::Vec3b>(j,i)[2] = dirCols_(idz,2);
      }else{
        zIrgb.at<cv::Vec3b>(j,i)[0] = 255;
        zIrgb.at<cv::Vec3b>(j,i)[1] = 255;
        zIrgb.at<cv::Vec3b>(j,i)[2] = 255;
      }
  return zIrgb;
};

cv::Mat DirSeg::normalsImg()
{
  cv::Mat nI(h_,w_,CV_8UC3);
  cv::Mat nIRGB(h_,w_,CV_8UC3);
  this->normalsImgRaw().convertTo(nI,CV_8UC3,127.5f,127.5f);
  cv::cvtColor(nI,nIRGB,CV_RGB2BGR);
  return nIRGB;
}

cv::Mat DirSeg::smoothDepthImg()
{
  cv::Mat dI(h_,w_,CV_8UC1);
  this->smoothDepth().convertTo(dI,CV_8UC1,255./4.,-1.9);
  return dI;
}

cv::Mat DirSeg::overlaySeg(cv::Mat img)
{
  cv::Mat rgb;
  if(img.channels() == 1)
  {
    std::vector<cv::Mat> grays(3);
    grays.at(0) = img;
    grays.at(1) = img;
    grays.at(2) = img;
    cv::merge(grays, rgb);
  }else{
    rgb = img;
  }
  cv::Mat zI = labelsImg();
  cv::Mat Iout;
  cv::addWeighted(rgb , 0.7, zI, 0.3, 0.0, Iout);
  projectDirections(Iout,centroids(),cfgNormals_.f_d,dirCols_);
  return Iout;
};

void DirSeg::fillJET()
{
  float JET_r[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00588235294117645,0.02156862745098032,0.03725490196078418,0.05294117647058827,0.06862745098039214,0.084313725490196,0.1000000000000001,0.115686274509804,0.1313725490196078,0.1470588235294117,0.1627450980392156,0.1784313725490196,0.1941176470588235,0.2098039215686274,0.2254901960784315,0.2411764705882353,0.2568627450980392,0.2725490196078431,0.2882352941176469,0.303921568627451,0.3196078431372549,0.3352941176470587,0.3509803921568628,0.3666666666666667,0.3823529411764706,0.3980392156862744,0.4137254901960783,0.4294117647058824,0.4450980392156862,0.4607843137254901,0.4764705882352942,0.4921568627450981,0.5078431372549019,0.5235294117647058,0.5392156862745097,0.5549019607843135,0.5705882352941174,0.5862745098039217,0.6019607843137256,0.6176470588235294,0.6333333333333333,0.6490196078431372,0.664705882352941,0.6803921568627449,0.6960784313725492,0.7117647058823531,0.7274509803921569,0.7431372549019608,0.7588235294117647,0.7745098039215685,0.7901960784313724,0.8058823529411763,0.8215686274509801,0.8372549019607844,0.8529411764705883,0.8686274509803922,0.884313725490196,0.8999999999999999,0.9156862745098038,0.9313725490196076,0.947058823529412,0.9627450980392158,0.9784313725490197,0.9941176470588236,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9862745098039216,0.9705882352941178,0.9549019607843139,0.93921568627451,0.9235294117647062,0.9078431372549018,0.892156862745098,0.8764705882352941,0.8607843137254902,0.8450980392156864,0.8294117647058825,0.8137254901960786,0.7980392156862743,0.7823529411764705,0.7666666666666666,0.7509803921568627,0.7352941176470589,0.719607843137255,0.7039215686274511,0.6882352941176473,0.6725490196078434,0.6568627450980391,0.6411764705882352,0.6254901960784314,0.6098039215686275,0.5941176470588236,0.5784313725490198,0.5627450980392159,0.5470588235294116,0.5313725490196077,0.5156862745098039,0.5};
  float JET_g[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001960784313725483,0.01764705882352935,0.03333333333333333,0.0490196078431373,0.06470588235294117,0.08039215686274503,0.09607843137254901,0.111764705882353,0.1274509803921569,0.1431372549019607,0.1588235294117647,0.1745098039215687,0.1901960784313725,0.2058823529411764,0.2215686274509804,0.2372549019607844,0.2529411764705882,0.2686274509803921,0.2843137254901961,0.3,0.3156862745098039,0.3313725490196078,0.3470588235294118,0.3627450980392157,0.3784313725490196,0.3941176470588235,0.4098039215686274,0.4254901960784314,0.4411764705882353,0.4568627450980391,0.4725490196078431,0.4882352941176471,0.503921568627451,0.5196078431372548,0.5352941176470587,0.5509803921568628,0.5666666666666667,0.5823529411764705,0.5980392156862746,0.6137254901960785,0.6294117647058823,0.6450980392156862,0.6607843137254901,0.6764705882352942,0.692156862745098,0.7078431372549019,0.723529411764706,0.7392156862745098,0.7549019607843137,0.7705882352941176,0.7862745098039214,0.8019607843137255,0.8176470588235294,0.8333333333333333,0.8490196078431373,0.8647058823529412,0.8803921568627451,0.8960784313725489,0.9117647058823528,0.9274509803921569,0.9431372549019608,0.9588235294117646,0.9745098039215687,0.9901960784313726,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9901960784313726,0.9745098039215687,0.9588235294117649,0.943137254901961,0.9274509803921571,0.9117647058823528,0.8960784313725489,0.8803921568627451,0.8647058823529412,0.8490196078431373,0.8333333333333335,0.8176470588235296,0.8019607843137253,0.7862745098039214,0.7705882352941176,0.7549019607843137,0.7392156862745098,0.723529411764706,0.7078431372549021,0.6921568627450982,0.6764705882352944,0.6607843137254901,0.6450980392156862,0.6294117647058823,0.6137254901960785,0.5980392156862746,0.5823529411764707,0.5666666666666669,0.5509803921568626,0.5352941176470587,0.5196078431372548,0.503921568627451,0.4882352941176471,0.4725490196078432,0.4568627450980394,0.4411764705882355,0.4254901960784316,0.4098039215686273,0.3941176470588235,0.3784313725490196,0.3627450980392157,0.3470588235294119,0.331372549019608,0.3156862745098041,0.2999999999999998,0.284313725490196,0.2686274509803921,0.2529411764705882,0.2372549019607844,0.2215686274509805,0.2058823529411766,0.1901960784313728,0.1745098039215689,0.1588235294117646,0.1431372549019607,0.1274509803921569,0.111764705882353,0.09607843137254912,0.08039215686274526,0.06470588235294139,0.04901960784313708,0.03333333333333321,0.01764705882352935,0.001960784313725483,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  float JET_b[]={0.5,0.5156862745098039,0.5313725490196078,0.5470588235294118,0.5627450980392157,0.5784313725490196,0.5941176470588235,0.6098039215686275,0.6254901960784314,0.6411764705882352,0.6568627450980392,0.6725490196078432,0.6882352941176471,0.7039215686274509,0.7196078431372549,0.7352941176470589,0.7509803921568627,0.7666666666666666,0.7823529411764706,0.7980392156862746,0.8137254901960784,0.8294117647058823,0.8450980392156863,0.8607843137254902,0.8764705882352941,0.892156862745098,0.907843137254902,0.9235294117647059,0.9392156862745098,0.9549019607843137,0.9705882352941176,0.9862745098039216,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9941176470588236,0.9784313725490197,0.9627450980392158,0.9470588235294117,0.9313725490196079,0.915686274509804,0.8999999999999999,0.884313725490196,0.8686274509803922,0.8529411764705883,0.8372549019607844,0.8215686274509804,0.8058823529411765,0.7901960784313726,0.7745098039215685,0.7588235294117647,0.7431372549019608,0.7274509803921569,0.7117647058823531,0.696078431372549,0.6803921568627451,0.6647058823529413,0.6490196078431372,0.6333333333333333,0.6176470588235294,0.6019607843137256,0.5862745098039217,0.5705882352941176,0.5549019607843138,0.5392156862745099,0.5235294117647058,0.5078431372549019,0.4921568627450981,0.4764705882352942,0.4607843137254903,0.4450980392156865,0.4294117647058826,0.4137254901960783,0.3980392156862744,0.3823529411764706,0.3666666666666667,0.3509803921568628,0.335294117647059,0.3196078431372551,0.3039215686274508,0.2882352941176469,0.2725490196078431,0.2568627450980392,0.2411764705882353,0.2254901960784315,0.2098039215686276,0.1941176470588237,0.1784313725490199,0.1627450980392156,0.1470588235294117,0.1313725490196078,0.115686274509804,0.1000000000000001,0.08431372549019622,0.06862745098039236,0.05294117647058805,0.03725490196078418,0.02156862745098032,0.00588235294117645,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  for(uint32_t i=0; i<256; ++i)
  {
    JET_r_[i] = JET_r[i];
    JET_g_[i] = JET_g[i];
    JET_b_[i] = JET_b[i];
  }

}


}
