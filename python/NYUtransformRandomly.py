import os, re
import subprocess as subp

nyuPath = "/data/vision/fisher/data1/nyu_depth_v2/extracted/"
outPath = "/data/vision/scratch/fisher/jstraub/nyuRndTransformed/"

names = []
for root, dirs, files in os.walk(nyuPath):
  for file in files:
    name, ending = os.path.splitext(file)
    if ending == '.ply' and re.search("Smooth$",name) is None and \
      re.search("angle",name) is None:
      names.append(name)

print "found ", len(names), " NYU files"
raw_input()

for name in names:
  print " ---- " + name
  for theta in [10, 45, 90, 180]:
    for translation in [0, 1., 5., 10.]:
      args = ["../build/bin/rndTransformPc",
          "-i " + nyuPath + name + ".ply",
          "-o " + outPath + re.sub("_d", "", name),
          "-a {}".format(theta),  
          "-t {}".format(translation),
          ]
      print " ".join(args)
#      raw_input()
      err = subp.call(" ".join(args), shell=True);
      if err:
        print "error when executing " + " ".join(args)
  print " ---- " + name + " DONE -------"
