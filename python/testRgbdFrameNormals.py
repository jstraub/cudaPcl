import numpy as np
import mayavi.mlab as mlab
from js.data.rgbd.rgbdframe import RgbdFrame

rgbd = RgbdFrame(540.)
rgbd.load("./table_0_d.png")
rgbd.showRgbd()
rgbd.showRgbdSmooth()
plt.show()
rgbd.showNormals()
mlab.show(stop=True)
