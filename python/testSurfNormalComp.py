import numpy as np
import mayavi.mlab as mlab
from js.geometry.rotations import *

x = np.array([
        [0,0,2],
        [-0.1,0,2],
        [.1,0,2],
        [0,-.1,2],
        [0,.1,2]])

w,h = 640,480

def norm(v):
  return np.sqrt((v**2).sum())
def normed(v):
  return v/norm(v)
def project(Rc,tc,f,x):
    xc = (Rc.T.dot(x.T) - Rc.T.dot(tc)).T
    uv = np.zeros((x.shape[0],2))
    for i in range(x.shape[0]):
        uv[i,:] = np.array([f*xc[i,0]/xc[i,2]+(w-1.0)/2.,f*xc[i,1]/xc[i,2]+(h-1.0)/2.])
    return uv, xc
def computeNormal(Rc,tc,f,x):
  uv,xc = project(Rc,tc,f,x)
  zs = xc[:,2]
  dzdu = zs[2]-zs[1]
  dzdv = zs[4]-zs[3]
  dxdu = xc[2,0]-xc[1,0]
  dxdv = xc[4,0]-xc[3,0]
  dydu = xc[2,1]-xc[1,1]
  dydv = xc[4,1]-xc[3,1]

  ddu = np.array([dxdu,dydu,dzdu])
  ddv = np.array([dxdv,dydv,dzdv])
  n2 = np.cross(ddv,ddu)

  du = uv[0,0] - (w-1.0)/2. 
  dv = uv[0,1] - (h-1.0)/2. 
  z = zs[0]
  n = np.array([
    -(du*dzdu+dv*dzdv+z),
    -dzdv*f,
    -dzdu*f
    ])
  print uv
  print z,dzdu,dzdv,du,dv
  print norm(n)
  print normed(n)
  print normed(n2)
#  return normed(n2)
  return normed(n)
def plotR(Rc,tc):
  mlab.quiver3d(np.zeros(1)*tc[0],np.zeros(1)*tc[1],np.zeros(1)*tc[2],\
      np.ones(1)*Rc[0,0],np.ones(1)*Rc[1,0],np.ones(1)*Rc[2,0],\
      color=(1,0,0))
  mlab.quiver3d(np.zeros(1)*tc[0],np.zeros(1)*tc[1],np.zeros(1)*tc[2],\
      np.ones(1)*Rc[0,1],np.ones(1)*Rc[1,1],np.ones(1)*Rc[2,1],\
      color=(0,1,0))
  mlab.quiver3d(np.zeros(1)*tc[0],np.zeros(1)*tc[1],np.zeros(1)*tc[2],\
      np.ones(1)*Rc[0,2],np.ones(1)*Rc[1,2],np.ones(1)*Rc[2,2],\
      color=(0,0,1))

Rc = np.eye(3)
tc = np.zeros((3,1))
f = 470.

angs = np.linspace(0.,30.,3)/180.*np.pi
ns = np.zeros((angs.size,3))
for i,ang in enumerate(angs):
  qc = Quaternion()
  qc.fromAxisAngle(ang,normed(np.array([0,1,0])))
  ns[i,:] = computeNormal(qc.toRot().R,tc,f,x)
#print ns

fig = mlab.figure()
mlab.points3d(x[:,0],x[:,1],x[:,2],scale_factor=0.1)
mlab.quiver3d(np.ones(angs.size)*x[0,0], \
    np.ones(angs.size)*x[0,1], \
    np.ones(angs.size)*x[0,2],ns[:,0],ns[:,1],ns[:,2])
for i,ang in enumerate(angs):
  qc = Quaternion()
  qc.fromAxisAngle(ang,normed(np.array([0,1,0])))
  Rc = qc.toRot().R
  plotR(Rc,tc)
fig2 = mlab.figure()
for i,ang in enumerate(angs):
  qc = Quaternion()
  qc.fromAxisAngle(ang,normed(np.array([0,1,0])))
  Rc = qc.toRot().R
  xc = (Rc.T.dot(x.T) - Rc.T.dot(tc)).T
  mlab.points3d(xc[:,0],xc[:,1],xc[:,2],scale_factor=0.1)
  mlab.quiver3d(np.ones(1)*xc[0,0], \
      np.ones(1)*xc[0,1], \
      np.ones(1)*xc[0,2],np.ones(1)*ns[i,0],np.ones(1)*ns[i,1], \
      np.ones(1)*ns[i,2])
  plotR(np.eye(3),np.zeros(3))

mlab.show(stop=True)
