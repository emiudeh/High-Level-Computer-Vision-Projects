# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math

def gauss(sigma):
    x = np.arange(-3*sigma, 3*sigma+1)
    Gx = (1/(math.sqrt(2*math.pi)*sigma)) * np.exp(-1*((x*x)/(2*sigma**2)))
    return Gx, x


def gaussderiv(img, sigma):
    imgDx = np.empty_like(img)
    imgDy = np.empty_like(img)
    # Generate 1D convolution kernel
    Gx, x = gaussdx(sigma)

    # k_dim is placeholder for the kernel "dimension" (size)
    k_dim = Gx.shape[0]
    pad_size = k_dim//2

    # create a padded image matrix
    img_paddded = np.pad(img, pad_width=pad_size, mode='constant') 
    
    # Horizontal convolution
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # conv = img_paddded[row+pad_size, col:col+(2*pad_size)+1] * Gx
            conv = img_paddded[row+pad_size, col:col+k_dim] * Gx
            imgDx[row, col] = np.sum(conv)

    # Reassign padded image matrix using updated original image
    img_paddded = np.pad(img, pad_width=pad_size, mode='constant')

    # Vertical convolution
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            conv = img_paddded[row:row+k_dim, col+pad_size] * Gx
            imgDy[row, col] = np.sum(conv)
    return imgDx, imgDy


def gaussdx(sigma):
    x = np.arange(-3*sigma, 3*sigma+1)
    D = ((-1*x)/(math.sqrt(2*math.pi)*sigma**3)) * np.exp(-1*((x*x)/(2*sigma**2)))
    return D, x

def gaussianfilter(img, sigma):
    outimage = np.empty_like(img)

    # Generate 1D convolution kernel
    Gx, x = gauss(sigma)

    # k_dim is placeholder for the kernel "dimension" (size)
    k_dim = Gx.shape[0]
    pad_size = k_dim//2
    # create a padded image matrix
    img_paddded = np.pad(img, pad_width=pad_size, mode='constant') 
    
    # Horizontal convolution
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # conv = img_paddded[row+pad_size, col:col+(2*pad_size)+1] * Gx
            conv = img_paddded[row+pad_size, col:col+k_dim] * Gx
            outimage[row, col] = np.sum(conv)

    # Reassign padded image matrix using updated outimage
    img_paddded = np.pad(outimage, pad_width=pad_size, mode='constant')

    # Vertical convolution
    for col in range(img.shape[1]):
        for row in range(img.shape[0]):
            conv = img_paddded[row:row+k_dim, col+pad_size] * Gx
            outimage[row, col] = np.sum(conv)

    return outimage
