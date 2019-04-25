# 
# compute chi2 distance between x and y
#

import numpy as np

def dist_chi2(x,y):
  x = np.array(x)
  y = np.array(y)
  s = (x - y) ** 2 / (x + y + 1e-10)
  return 0.5 * sum(s)

# 
# compute l2 distance between x and y
#
def dist_l2(x,y):

  x = np.array(x)
  y = np.array(y)

  return np.sqrt(sum( (x - y) ** 2 ) )


# 
# compute intersection distance between x and y
# return 1 - intersection, so that smaller values also correspond to more similart histograms
#
def dist_intersect(x,y):
  x = np.array(x)
  y = np.array(y)
  dist = 0
  for i in range(len(x)):
    if x[i] == y[i]:
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





