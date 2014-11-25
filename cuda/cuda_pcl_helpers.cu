/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */


#include <stdint.h>
#include <assert.h>
#include <helper_cuda.h> 
#include <cuda_pc_helpers.h>


/*
 * compute the xyz images using the inverse focal length invF
 */
template<typename T>
__global__ void depth2xyz(uint16_t* d, T* x, T* y, 
    T* z, T invF, int w, int h, T *xyz)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    T dd = T(d[id])*0.001; // convert to mm
    // have a buffer of nan pixels around the border to prohibit 
    // the filters to do bad stuff at the corners
    if ((0.0<dd)&&(dd<4.0)&&( BOARDER_SIZE<idx && idx<w-BOARDER_SIZE 
          && BOARDER_SIZE<idy && idy<h-BOARDER_SIZE)){
      // in combination with the normal computation this gives the right normals
      x[id] = dd*(T(idx)-(w-1.)*0.5)*invF;
      y[id] = dd*(T(idy)-(h-1.)*0.5)*invF;
      z[id] = dd;
    }else{
      x[id] = 0.0/0.0;
      y[id] = 0.0/0.0;
      z[id] = 0.0/0.0;
    }
    if (xyz != NULL){
      xyz[id*4] = x[id];
      xyz[id*4+1] = y[id];
      xyz[id*4+2] = z[id];
    }
  }
}

void depth2xyzGPU(uint16_t* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz=NULL)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  depth2xyz<float><<<blocks, threads>>>(d,x,y,z,invF,w,h,xyz);
  getLastCudaError("depth2xyzGPU() execution failed\n");
}
void depth2xyzGPU(uint16_t* d, double* x, double* y, double* z,
    double invF, int w, int h, double *xyz=NULL)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  depth2xyz<double><<<blocks, threads>>>(d,x,y,z,invF,w,h,xyz);
  getLastCudaError("depth2xyzGPU() execution failed\n");
}


template<typename T>
__global__ void depth2float(uint16_t* d, T* d_float, uint8_t* haveData, int w, int h, int outStep)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;
  const int idOut = idx+outStep*idy;

  if(idx<w && idy<h)
  {
    T dd = T(d[id])*0.001; // convert to mm
    if ((dd>0.0f)){
      d_float[idOut] = dd;
      haveData[idOut] = 1;
    }else{
      d_float[idOut] = 0.0; //TODO broke the nan trick! 0.0/0.0;
      haveData[idOut] = 0;
    }
  }
}

void depth2floatGPU(uint16_t* d, double* d_float, uint8_t* haveData,int w, int h, int outStep)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  if (outStep <0 ) outStep = w;
  depth2float<double><<<blocks, threads>>>(d,d_float,haveData,w,h,outStep);
  getLastCudaError("depth2floatGPU() execution failed\n");
}

void depth2floatGPU(unsigned short* d, float* d_float,uint8_t* haveData, int w, int h, int outStep)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  if (outStep <0 ) outStep = w;
  depth2float<float><<<blocks, threads>>>(d,d_float,haveData,w,h,outStep);
  getLastCudaError("depth2floatGPU() execution failed\n");
}

//#define SQRT2 1.4142135623730951
__global__ void depthFilter(float* d,
     int w, int h)
{
  const float thresh = 0.2; // 5cm
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

  __shared__ float duSq[8];
  if(tid==0) duSq[0] = 2.;
  if(tid==1) duSq[1] = 1.;
  if(tid==2) duSq[2] = 2.;
  if(tid==3) duSq[3] = 1.;
  if(tid==4) duSq[4] = 2.;
  if(tid==5) duSq[5] = 1.;
  if(tid==6) duSq[6] = 2.;
  if(tid==7) duSq[7] = 1.;

  __syncthreads(); // make sure that ys have been cached

  // filtering according to noise model from file:///home/jstraub/Downloads/Nguyen2012-ModelingKinectSensorNoise.pdf
  if(1<idx && idx<w-1 && 1<idy && idy<h-1)
  {
    float dd = d[id];
    if ((dd>0.0f))
    {
      float invSigSqL = 1.0f/0.5822699462742343; //for theta=30deg //0.8f + 0.035f*theta/(PI*0.5f -theta);
      float invSigSqZ = 1.0f/(0.0012f + 0.0019f*(dd-0.4f)*(dd-0.4f));
      invSigSqZ = invSigSqZ*invSigSqZ;
      float ds[8];
      ds[0] = d[idx-1+w*(idy-1)];
      ds[1] = d[idx  +w*(idy-1)];
      ds[2] = d[idx+1+w*(idy-1)];
      ds[3] = d[idx+1+w*idy];
      ds[4] = d[idx+1+w*(idy+1)];
      ds[5] = d[idx  +w*(idy+1)];
      ds[6] = d[idx-1+w*(idy+1)];
      ds[7] = d[idx-1+w*idy];
      float wSum = 0.0f;
      float dwSum = 0.0f;
#pragma unroll
      for(int32_t i=0; i<8; ++i)
      {
        float dz = fabs(ds[i]-dd);
        float wi = dz < thresh ? expf(-0.5f*(duSq[i]*invSigSqL + dz*dz*invSigSqZ)) : 0.0f;
        wSum += wi;
        dwSum += wi*ds[i];
      }
      d[id] = dwSum/wSum;
    }
  }
}

void depthFilterGPU(float* d, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  depthFilter<<<blocks, threads>>>(d,w,h);
  getLastCudaError("depthFilterGPU() execution failed\n");
}


template<typename T>
__global__ void depth2xyzFloat(T* d, T* x, T* y, 
    T* z, T invF, int w, int h, T *xyz)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    T dd = d[id]; // convert to mm
    // have a buffer of nan pixels around the border to prohibit 
    // the filters to do bad stuff at the corners
    if ((dd>0.0f)&&( BOARDER_SIZE<idx && idx<w-BOARDER_SIZE 
          && BOARDER_SIZE<idy && idy<h-BOARDER_SIZE)){
      // in combination with the normal computation this gives the right normals
      x[id] = dd*T(idx-w/2)*invF;
      y[id] = dd*T(idy-h/2)*invF;
      z[id] = dd;
    }else{
      x[id] = 0.0f/0.0f;
      y[id] = 0.0f/0.0f;
      z[id] = 0.0f/0.0f;
    }
    if (xyz != NULL){
      xyz[id*4] = x[id];
      xyz[id*4+1] = y[id];
      xyz[id*4+2] = z[id];
    }
  }
}

void depth2xyzFloatGPU(float* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz=NULL)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  depth2xyzFloat<float><<<blocks, threads>>>(d,x,y,z,invF,w,h,xyz);
  getLastCudaError("depth2xyzGPU() execution failed\n");
}

void depth2xyzFloatGPU(double* d, double* x, double* y, double* z,
    double invF, int w, int h, double *xyz=NULL)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  depth2xyzFloat<double><<<blocks, threads>>>(d,x,y,z,invF,w,h,xyz);
  getLastCudaError("depth2xyzGPU() execution failed\n");
}

inline __device__ float signf(float a)
{
  if (a<0.0f)
    return -1.0f;
  else
    return 1.0f;
//  else
//    return 0.0f;
}

inline __device__ float absf(float a)
{
  return a<0.0f?-a:a;
}

/*
 * derivatives2normals takes pointers to all derivatives and fills in d_n
 * d_n is a w*h*3 array for all three normal components (x,y,z)
 */
template<typename T>
__global__ void derivatives2normalsPcl(T* d_x, T* d_y, T* d_z, 
    T* d_xu, T* d_yu, T* d_zu, 
    T* d_xv, T* d_yv, T* d_zv, 
    T* d_n, int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // in combination with the depth to xyz computation this gives the right normals
    T xu=d_xu[id];
    T yu=d_yu[id];
    T zu=d_zu[id];
    T xv=d_xv[id];
    T yv=d_yv[id];
    T zv=d_zv[id];
    T invLenu = 1.0f/sqrtf(xu*xu + yu*yu + zu*zu);
    xu *= invLenu;
    yu *= invLenu;
    zu *= invLenu;
    T invLenv = 1.0f/sqrtf(xv*xv + yv*yv + zv*zv);
    xv *= invLenv;
    yv *= invLenv;
    zv *= invLenv;

    T nx = yu*zv - yv*zu;
    T ny = xv*zu - xu*zv;
    T nz = xu*yv - xv*yu;
    T lenn = sqrtf(nx*nx + ny*ny + nz*nz);
    T sgn = signf(d_x[id]*nx + d_y[id]*ny + d_z[id]*nz)/lenn;
    // normals are pointing away from where the kinect sensor is
    // ie. if pointed at the ceiling the normals will be (0,0,1)
    // the coordinate system is aligned with the image coordinates:
    // z points outward to the front 
    // x to the right (when standing upright on the foot and looking from behind)
    // y down (when standing upright on the foot and looking from behind)


//    if (absf(ny)<0.01f || absf(nx)<0.01f)
//{
//  nx=0.0f/0.0f;
//  ny=0.0f/0.0f;
//  nz=0.0f/0.0f;
//} 

    // the 4th component is always 1.0f - due to PCL conventions!
    d_n[id*X_STEP+X_OFFSET] = nx*sgn;
    d_n[id*X_STEP+X_OFFSET+1] = ny*sgn;
    d_n[id*X_STEP+X_OFFSET+2] = nz*sgn;
    d_n[id*X_STEP+X_OFFSET+3] = 1.0f;
    // f!=f only true for nans
    //d_nGood[id] = ((nx!=nx) | (ny!=ny) | (nz!=nz))?0:1;
  }
}

void derivatives2normalsPclGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  derivatives2normalsPcl<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);
  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}
void derivatives2normalsPclGPU(double* d_x, double* d_y, double* d_z, 
    double* d_xu, double* d_yu, double* d_zu, 
    double* d_xv, double* d_yv, double* d_zv, 
    double* d_n, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  derivatives2normalsPcl<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);
  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}


template<typename T>
__global__ void xyzImg2PointCloudXYZRGB(T* xyzImg, float* pclXYZRGB, int32_t w,
    int32_t h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    pclXYZRGB[id*X_STEP+X_OFFSET]   = (float) xyzImg[id*3];
    pclXYZRGB[id*X_STEP+X_OFFSET+1] = (float) xyzImg[id*3+1];
    pclXYZRGB[id*X_STEP+X_OFFSET+2] = (float) xyzImg[id*3+2];
    pclXYZRGB[id*X_STEP+X_OFFSET+3] = 1.0f;
  }
}

void xyzImg2PointCloudXYZRGB(double* d_xyzImg, float* d_pclXYZRGB, int32_t w,
    int32_t h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  xyzImg2PointCloudXYZRGB<double><<<blocks, threads>>>(d_xyzImg,d_pclXYZRGB,w,h);
  getLastCudaError("xyzImg2PointCloudXYZRGB() execution failed\n");
}

void xyzImg2PointCloudXYZRGB(float* d_xyzImg, float* d_pclXYZRGB, int32_t w,
    int32_t h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  xyzImg2PointCloudXYZRGB<float><<<blocks, threads>>>(d_xyzImg,d_pclXYZRGB,w,h);
  getLastCudaError("xyzImg2PointCloudXYZRGB() execution failed\n");
}


template<typename T>
__global__ void derivatives2normals(T* d_x, T* d_y, T* d_z, 
    T* d_xu, T* d_yu, T* d_zu, 
    T* d_xv, T* d_yv, T* d_zv, 
    T* d_n, uint8_t* d_haveData, int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // in combination with the depth to xyz computation this gives the right normals
    T xu=d_xu[id];
    T yu=d_yu[id];
    T zu=d_zu[id];
    T xv=d_xv[id];
    T yv=d_yv[id];
    T zv=d_zv[id];
    T* d_ni = d_n+id*3;
    if (d_haveData && (xu!=xu || yu!=yu || zu!=zu || xv!=xv || yv!=yv || zv!=zv))
    {
      d_haveData[id] = 0; 
      d_ni[0] = 0.0/0.0;
      d_ni[1] = 0.0/0.0;
      d_ni[2] = 0.0/0.0;
    }else{
      T invLenu = 1.0f/sqrtf(xu*xu + yu*yu + zu*zu);
      xu *= invLenu;
      yu *= invLenu;
      zu *= invLenu;
      T invLenv = 1.0f/sqrtf(xv*xv + yv*yv + zv*zv);
      xv *= invLenv;
      yv *= invLenv;
      zv *= invLenv;

      T nx = yu*zv - yv*zu;
      T ny = xv*zu - xu*zv;
      T nz = xu*yv - xv*yu;
      T lenn = sqrtf(nx*nx + ny*ny + nz*nz);
      T sgn = signf(d_x[id]*nx + d_y[id]*ny + d_z[id]*nz)/lenn;
    // normals are pointing away from where the kinect sensor is
    // ie. if pointed at the ceiling the normals will be (0,0,1)
    // the coordinate system is aligned with the image coordinates:
    // z points outward to the front 
    // x to the right (when standing upright on the foot and looking from behind)
    // y down (when standing upright on the foot and looking from behind)
//    if (absf(ny)<0.01f || absf(nx)<0.01f)
//{
//  nx=0.0f/0.0f;
//  ny=0.0f/0.0f;
//  nz=0.0f/0.0f;
//} 

    // the 4th component is always 1.0f - due to PCL conventions!
      d_ni[0] = nx*sgn;
      d_ni[1] = ny*sgn;
      d_ni[2] = nz*sgn;
      if(d_haveData)
        d_haveData[id] = (sgn!=sgn || (nx!=nx) || (ny!=ny) || (nz!=nz))?0:1;
    // f!=f only true for nans
    }
  }
}


void derivatives2normalsGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, uint8_t* d_haveData, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  derivatives2normals<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,d_haveData,w,h);
  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}
void derivatives2normalsGPU(double* d_x, double* d_y, double* d_z, 
    double* d_xu, double* d_yu, double* d_zu, 
    double* d_xv, double* d_yv, double* d_zv, 
    double* d_n, uint8_t* d_haveData, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);
  derivatives2normals<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,d_haveData,w,h);
  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}

/*
 * derivatives2normals takes pointers to all derivatives and fills in d_n
 * d_n is a w*h*3 array for all three normal components (x,y,z)
 */
__global__ void derivatives2normalsCleaner(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // in combination with the depth to xyz computation this gives the right normals
    float xu=d_xu[id];
    float yu=d_yu[id];
    float zu=d_zu[id];
    float xv=d_xv[id];
    float yv=d_yv[id];
    float zv=d_zv[id];
    float invLenu = 1.0f/sqrtf(xu*xu + yu*yu + zu*zu);
    xu *= invLenu;
    yu *= invLenu;
    zu *= invLenu;
    float invLenv = 1.0f/sqrtf(xv*xv + yv*yv + zv*zv);
    xv *= invLenv;
    yv *= invLenv;
    zv *= invLenv;

    float nx = 0.;
    float ny = 0.;
    float nz = 0.;
    float sgn = 1.;
    if (invLenu < 1./0.04 || invLenv < 1./0.04 )
    {
      nx=0.0f/0.0f;
      ny=0.0f/0.0f;
      nz=0.0f/0.0f;
    } else {
      nx = yu*zv - yv*zu;
      ny = xv*zu - xu*zv;
      nz = xu*yv - xv*yu;
      float lenn = sqrtf(nx*nx + ny*ny + nz*nz);
      sgn = signf(d_x[id]*nx + d_y[id]*ny + d_z[id]*nz)/lenn;
      // normals are pointing away from where the kinect sensor is
      // ie. if pointed at the ceiling the normals will be (0,0,1)
      // the coordinate system is aligned with the image coordinates:
      // z points outward to the front 
      // x to the right (when standing upright on the foot and looking from behind)
      // y down (when standing upright on the foot and looking from behind)
    }


    // the 4th component is always 1.0f - due to PCL conventions!
    d_n[id*X_STEP+X_OFFSET] = nx*sgn;
    d_n[id*X_STEP+X_OFFSET+1] = ny*sgn;
    d_n[id*X_STEP+X_OFFSET+2] = nz*sgn;
    d_n[id*X_STEP+X_OFFSET+3] = 1.0f;
    // f!=f only true for nans
    //d_nGood[id] = ((nx!=nx) | (ny!=ny) | (nz!=nz))?0:1;
  }
}

void derivatives2normalsCleanerGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  derivatives2normalsCleaner<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);

  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}

__device__ inline float square(float a )
{ return a*a;}


__global__ void weightsFromCov(float* z, float* weights,
    float theta, float invF, int w, int h)
{
// according to ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6375037
// weights are the inverse of the determinant of the covariance of the noise ellipse
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    float z_i = z[id];
    //float ang = theta/180.0f*M_PI;
    float sigma_z = 0.0012f + 0.019f * square(z_i-0.4f);
    //float sigma_l = (0.8f + 0.035f*ang/(M_PI*0.5f-ang))*z_i*invF;
    weights[id] = 1.0f/sigma_z;
    //weights[id] = 1.0f/(square(sigma_z)+2.0f*square(sigma_l));
  }
}

void weightsFromCovGPU(float* z, float* weights, float theta, float invF, int
    w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

    weightsFromCov<<<blocks, threads>>>(z,weights,theta,invF,w,h);
    getLastCudaError("depth2xyzGPU() execution failed\n");
}


__global__ void weightsFromArea(float* z, float* weights, int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // weight proportial to area that the pixel i observes at distance z_i
    // the area = z_i^2/f^2 but f is constant so we dont need divide by it.
    weights[id] = square(z[id]); 
  }
}

void weightsFromAreaGPU(float* z, float* weights, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  weightsFromArea<<<blocks, threads>>>(z,weights,w,h);
  getLastCudaError("weightsFromAreaGPU() execution failed\n");
}
  
// rotate point cloud
template<typename T, int32_t STEP>
__global__ void rotatePc_kernel(T* pc, T* d_R, int N)
{
  const int id = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ T R[9];
  if(threadIdx.x<9) R[threadIdx.x] = d_R[threadIdx.x];
  __syncthreads();

  if(id<N)
  {
    T* pc_i = pc+id*STEP;
    T pc_[3];
    pc_[0] = pc_i[0];
    pc_[1] = pc_i[1];
    pc_[2] = pc_i[2];
    T pp[3];
    pp[0] = R[0]*pc_[0] + R[3]*pc_[1] + R[6]*pc_[2];
    pp[1] = R[1]*pc_[0] + R[4]*pc_[1] + R[7]*pc_[2];
    pp[2] = R[2]*pc_[0] + R[5]*pc_[1] + R[8]*pc_[2];
//    pc_i[0] = pc_[0];
//    pc_i[1] = pc_[1];
//    pc_i[2] = pc_[2];
    pc_i[0] = pp[0];
    pc_i[1] = pp[1];
    pc_i[2] = pp[2];
  }
}

void rotatePcGPU(float* d_pc, float* d_R, int32_t N, int32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    rotatePc_kernel<float,3><<<blocks, threads>>>(d_pc,d_R,N);
  else if(step == 8) // pcl
    rotatePc_kernel<float,8><<<blocks, threads>>>(d_pc,d_R,N);
  else
    assert(false);
  getLastCudaError("rotatePc_kernel() execution failed\n");
}


void rotatePcGPU(double* d_pc, double* d_R, int32_t N, int32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    rotatePc_kernel<double,3><<<blocks, threads>>>(d_pc,d_R,N);
  else if(step == 8) // pcl
    rotatePc_kernel<double,8><<<blocks, threads>>>(d_pc,d_R,N);
  else
    assert(false);
  checkCudaErrors(cudaDeviceSynchronize());
}


template<typename T, int32_t STEP>
__global__ void copyShuffle_kernel(T* in, T* out, uint32_t* ind, int N)
{
  const int id = threadIdx.x + blockIdx.x*blockDim.x;

  if(id<N)
  {
    uint32_t ind_i = ind[id];
    T* in_i = in+ind_i*STEP;
    T* out_i = out+id*STEP;
//    T in_[3];
//    in_[0] = in_i[0];
//    in_[1] = in_i[1];
//    in_[2] = in_i[2];
    out_i[0] = in_i[0];
    out_i[1] = in_i[1];
    out_i[2] = in_i[2];
  }
}

void copyShuffleGPU(float* in, float* out, uint32_t* ind, int32_t N, int32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    copyShuffle_kernel<float,3><<<blocks, threads>>>(in,out,ind,N);
  else if(step == 8) // pcl
    copyShuffle_kernel<float,8><<<blocks, threads>>>(in,out,ind,N);
  else
    assert(false);
  checkCudaErrors(cudaDeviceSynchronize());
}

void copyShuffleGPU(double* in, double* out, uint32_t* ind, int32_t N, int32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    copyShuffle_kernel<double,3><<<blocks, threads>>>(in,out,ind,N);
  else if(step == 8) // pcl
    copyShuffle_kernel<double,8><<<blocks, threads>>>(in,out,ind,N);
  else
    assert(false);
  getLastCudaError("copyShuffle_kernel() execution failed\n");
}
  
template<typename T, int32_t STEP>
__global__ void copyShuffleInv_kernel(T* in, T* out, uint32_t* ind, int N)
{
  const int id = threadIdx.x + blockIdx.x*blockDim.x;

  if(id<N)
  {
    uint32_t ind_i = ind[id];
    T* in_i = in+id*STEP;
    T* out_i = out+ind_i*STEP;
//    T in_[3];
//    in_[0] = in_i[0];
//    in_[1] = in_i[1];
//    in_[2] = in_i[2];
    out_i[0] = in_i[0];
    out_i[1] = in_i[1];
    out_i[2] = in_i[2];
  }
}

void copyShuffleInvGPU(float* in, float* out, uint32_t* ind, int32_t N, int32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    copyShuffleInv_kernel<float,3><<<blocks, threads>>>(in,out,ind,N);
  else if(step == 8) // pcl
    copyShuffleInv_kernel<float,8><<<blocks, threads>>>(in,out,ind,N);
  else
    assert(false);
  checkCudaErrors(cudaDeviceSynchronize());
}

void copyShuffleInvGPU(double* in, double* out, uint32_t* ind, int32_t N, int32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    copyShuffleInv_kernel<double,3><<<blocks, threads>>>(in,out,ind,N);
  else if(step == 8) // pcl
    copyShuffleInv_kernel<double,8><<<blocks, threads>>>(in,out,ind,N);
  else
    assert(false);
  checkCudaErrors(cudaDeviceSynchronize());
}

template<typename T, uint32_t STEP>
__global__ void haveData_kernel(T* d_x, uint8_t* d_haveData, int32_t N)
{

  const int id = threadIdx.x + blockIdx.x*blockDim.x;
  if(id < N)
  {
    if (d_x[id*STEP] != d_x[id*STEP])
    {
      d_haveData[id] = 0; 
    }else{
      d_haveData[id] = 1; 
    }
  }
}
void haveDataGpu(float* d_x, uint8_t* d_haveData, int32_t N, uint32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    haveData_kernel<float,3><<<blocks, threads>>>(d_x,d_haveData,N);
  else if(step == 8) // pcl
    haveData_kernel<float,8><<<blocks, threads>>>(d_x,d_haveData,N);
  else
    assert(false);
  checkCudaErrors(cudaDeviceSynchronize());
}
void haveDataGpu(double* d_x, uint8_t* d_haveData, int32_t N, uint32_t step)
{
  dim3 threads(256,1,1);
  dim3 blocks(N/256 + (N%256>0?1:0),1,1);

  if(step == 3) // image layout
    haveData_kernel<double,3><<<blocks, threads>>>(d_x,d_haveData,N);
  else if(step == 8) // pcl
    haveData_kernel<double,8><<<blocks, threads>>>(d_x,d_haveData,N);
  else
    assert(false);
  checkCudaErrors(cudaDeviceSynchronize());
}
