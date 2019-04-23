import sys
sys.path.insert(0, '../filter-Q1')
import gauss_module

import numpy as np
from numpy import histogram as hist

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins=10):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    bins = np.linspace(np.min(img_gray), np.max(img_gray), num=num_bins+1)
    hists = np.empty(bins.shape[0]-1)
    for i in range(1, bins.shape[0]):
        hists[i-1] = np.count_nonzero((img_gray>=bins[i-1]) & (img_gray<bins[i]))
    hists /= np.sum(hists)

    # your code here
    return hists, bins



#  compute joint histogram for each color channel in the image, histogram should be normalized so that sum of all values equals 1
#  assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
def rgb_hist(img_color, num_bins):
    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'

    # define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))

    img_color = np.around((img_color/255) * (num_bins - 1))
    
    # execute the loop for each pixel in the image 
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G,B)
            # your code here
            hists[img_color[i, j, 1], img_color[i, j, 2], img_color[i, j, 3]] += 1

    # normalize the histogram such that its integral (sum) is equal 1
    # your code here
    hists /= np.sum(hists)

    hists = hists.reshape(hists.size)
    return hists



#  compute joint histogram for r/g values
#  note that r/g values should be in the range [0, 1];
#  histogram should be normalized so that sum of all values equals 1
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
def rg_hist(img_color, num_bins):

    assert len(img_color.shape) == 3, 'image dimension mismatch'
    assert img_color.dtype == 'float', 'incorrect image type'
  
    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # your code here
    img_color = np.around(img_color * (num_bins - 1))

    # execute the loop for each pixel in the image 
    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            # increment a histogram bin which corresponds to the value of pixel i,j; h(R,G)
            hists[img_color[i, j, 1], img_color[i, j, 2]] += 1

    # normalize the histogram such that its integral (sum) is equal 1
    hists /= np.sum(hists)

    hists = hists.reshape(hists.size)
    return hists



#  compute joint histogram of Gaussian partial derivatives of the image in x and y direction
#  for sigma = 7.0, the range of derivatives is approximately [-30, 30]
#  histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input grayvalue image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  note: you can use the function gaussderiv.m from the filter exercise.
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    # compute the first derivatives
    imgDx, imgDy = gauss_module.gaussderiv(img_gray, 7)

    # quantize derivatives to "num_bins" number of values
    img_gray = np.around(((img_gray+30)/60) * (num_bins - 1))

    # define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # execute the loop for each pixel in the image 
    for i in range(imgDx.shape[0]):
        for j in range(imgDx.shape[1]):
            hists[imgDx[i, j], imgDy[i, j]] += 1
    
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img1_gray, num_bins_gray, dist_name):
  if dist_name == 'grayvalue':
    return normalized_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rgb':
    return rgb_hist(img1_gray, num_bins_gray)
  elif dist_name == 'rg':
    return rg_hist(img1_gray, num_bins_gray)
  elif dist_name == 'dxdy':
    return dxdy_hist(img1_gray, num_bins_gray)
  else:
    assert 'unknown distance: %s'%dist_name
  
