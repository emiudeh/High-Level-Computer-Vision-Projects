import math

# 
# compute chi2 distance between x and y
#
def dist_chi2(x,y):
  # your code here
  dist = 0
  img_dim = len(x.shape)

  if img_dim == 1: 
    for i in range(x.shape[0]):
      dist += ((x[i] - y[i])**2 / (x[i] + y[i]))

  elif img_dim == 2: 
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        dist += ((x[i,j] - y[i,j])**2 / (x[i,j] + y[i,j]))

  elif img_dim == 3: 
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          dist += ((x[i,j,k] - y[i,j,k])**2 / (x[i,j,k] + y[i,j,k]))
      
  return dist
# 
# compute l2 distance between x and y
#
def dist_l2(x,y):
  # your code here
  dist = 0
  img_dim = len(x.shape)

  if img_dim == 1: 
    for i in range(x.shape[0]):
      dist += (x[i] - y[i])**2 

  elif img_dim == 2: 
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        dist += (x[i,j] - y[i,j])**2

  elif img_dim == 3: 
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          dist += (x[i,j,k] - y[i,j,k])**2

  return math.sqrt(dist)

# 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
  # your code here
  dist = 0
  img_dim = len(x.shape)

  if img_dim == 1: 
    for i in range(x.shape[0]):
      if x[i] == y[i]: 
        dist += 1

  elif img_dim == 2: 
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        if x[i,j] == y[i,j]: 
          dist += 1
      

  elif img_dim == 3: 
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for k in range(x.shape[2]):
          if x[i,j,k] == y[i,j,k]: 
            dist += 1

  return 1-dist

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert 'unknown distance: %s'%dist_name
  




