import os, re
import subprocess as subp

nyuPath = "/data/vision/fisher/data1/nyu_depth_v2/extracted/"

names = []
for root, dirs, files in os.walk(nyuPath):
  for file in files:
    name, ending = os.path.splitext(file)
    if ending == '.png' and not re.search("_d$",name) is None:
      names.append(name)

print "found ", len(names), " NYU files"
raw_input()

for name in names:
  print " ---- " + name
  args = ["../build/bin/openniSmoothNormals_file",
      "-i " + nyuPath + name + ".png",
      "-o " + nyuPath + re.sub("_d", "", name)]
  err = subp.call(" ".join(args), shell=True);
  if err:
    print "error when executing " + " ".join(args)
  print " ---- " + name + " DONE -------"
