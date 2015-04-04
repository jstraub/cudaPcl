This **gpu-enhanced** library implements **edge-preserving smoothing of depth
frames** from standard rgb-d sensors as well as **surface normal extraction** at a
**frame rate of 100Hz**.

![Depth before and after smoothing](doc/openniSmoothDepthGpuComparison.png)

Raw depth image vs. [guided filter](http://research.microsoft.com/en-us/um/people/jiansun/papers/guidedfilter_eccv10.pdf)
smoothed image on the right. Note that similar to a [bilateral filter](https://www.cs.duke.edu/~tomasi/papers/tomasi/tomasiIccv98.pdf)
the guided filter performs edge preserving filtering as can be observed at the
sharp discontinuities of the windows.

![smoothed surface normals](doc/openniSmoothNormalsGpuRGB.png)

Extracting surface normals using gradients of the depth image and a
cross-product operation yields smoothed surface normals.

### Dependencies

This code depends on the following other libraries and was tested under Ubuntu
14.04. 
- pcl 1.7 (and vtk 5.8)
- Opencv 2 (2.3.1)
- Eigen3 (3.0.5) 
- cuda 5.5 or 6.5 
- Boost (1.52)

The GPU kernels were tested on a Nvidia Quadro K2000M with compute
capability 3.0.

### Library
*libcudaPcl.so* collects all the cuda code into one shared library. The rest
of the code is in the form of header files.

### Executables
- *openniSmoothNormals*: grab RGB-D frames from an openni device, smooth the
  depth image using a fast GPU enhanced guided filter and extract surface normals.
```
  Allowed options:
    -h [ --help ]         produce help message
    -f [ --f_d ] arg      focal length of depth camera
    -e [ --eps ] arg      sqrt of the epsilon parameter of the guided filter
    -b [ --B ] arg        guided filter windows size (size will be (2B+1)x(2B+1))
    -c [ --compress ]     compress the computed normals
```
- *openniSmoothDepth*: grab RGB-D frames from an openni device, smooth the
  depth image using a fast GPU enhanced guided filter and display it.
```
  Allowed options:
    -h [ --help ]         produce help message
    -e [ --eps ] arg      sqrt of the epsilon parameter of the guided filter
    -b [ --B ] arg        guided filter windows size (size will be (2B+1)x(2B+1))
```
- *pclNormals*: extracts normals from a general (not necessarily aligned)
  point-cloud and outputs the point-cloud with normals.
```
  Allowed options:
    -h [ --help ]         produce help message
    -i [ --input ] arg    path to input
    -o [ --output ] arg   path to output
    -s [ --scale ] arg    scale for normal extraction search radius
```
- *pclBenchmark*: display RGBXYZ point-cloud (press 'r' to reset the viewpoint)
- *pclGrabber*: grab XYZ point-cloud; press 's' to save point-cloud to "./test_pcd.pcd"

