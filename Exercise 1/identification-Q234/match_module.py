import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

#
# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image
#

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

  hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
  model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
  query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

  D = np.zeros((len(model_images), len(query_images)))

  #-------------------------------------------------------------
  #print(D.shape)
  #print(model_hists)
  #print(query_hists)

  for qe in (range( len(query_hists) ) ):

      for mo in ( range( len(model_hists) ) ):
          D[mo][qe] = dist_module.get_dist_by_name(query_hists[qe], model_hists[mo], dist_type)


  #index of best match to each querry image
  best_match = np.argmin(D,axis=0)

  return best_match, D

def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

  image_hist = []

  # compute hisgoram for each image and add it at the bottom of image_hist

  if hist_isgray:
      for name in image_list:
          img = Image.open('./' + name)
          arr_img = np.array(img,dtype=float)
          img_gray = rgb2gray(arr_img)
          hist, _ = histogram_module.get_hist_by_name(img_gray, num_bins, hist_type).tolist()
          image_hist.append(hist.tolist())
  else:
      for name in image_list:
          img = Image.open('./' + name)
          arr = np.array(img, dtype=float)
          hist = histogram_module.get_hist_by_name(arr, num_bins, hist_type).tolist()
          image_hist.append(hist)

  return image_hist

#
# for each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# note: use the previously implemented function 'find_best_match'
# note: use subplot command to show all the images in the same Python figure, one row per query image
#

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

  #plt.figure()

  num_nearest = 5  # show the top-5 neighbors

  _, dist = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
  nearest = dist.argsort(axis=0)[:num_nearest, :]
  print(nearest)

  for i in range(len(query_images)):
      fig, ax = plt.subplots(1, num_nearest + 1,figsize=(10,5))
      ax[0].imshow( Image.open('./' + query_images[i]) )

      for j in range(num_nearest ):
          ax[j + 1].imshow( Image.open('./' + model_images[ nearest[j, i] ]) )

      plt.show()




