
import sys
import logging
import argparse

import numpy as np
from scipy import ndimage
from scipy.misc import imread, imsave

import maxflow
from maxflow import fastmin


def localssd(im1, im2, K):
    """
    The local sum of squared differences between windows of two images.
    
    The size of each window is (2K+1)x(2K+1).
    """
    H = np.ones((2*K + 1, 2*K + 1))
    diff2 = (im1 - im2)**2
    if im1.ndim == 2:
        return ndimage.convolve(diff2, H, mode='constant')
    if diff2.ndim == 3 and 3<=diff2.shape[-1]<=4:
        res = np.empty_like(diff2)
        for channel in range(diff2.shape[-1]):
            res[...,channel] = ndimage.convolve(diff2[...,channel], H, mode='constant')
        return res.sum(2)
    else:
        raise ValueError("invalid number of dimensions for input images")


def ssd_volume(im1, im2, disps, K):
    """
    Compute the visual similarity between local windows of the image 1
    and the image 2 displaced horizontally.
    
    The two images are supposed to be rectified so that the epipolar
    lines correspond to the scan-lines. The image 2 is horizontally
    moved between w[0] and w[1] pixels and, for each value, the local
    similarity between the first image and the displaced second image
    is computed calling to localssd.
    
    The result matrix is defined so that the value D[l, i, j] is the
    SSD of the local windows between the window centered at im1[i,j]
    and the window centered at im2[i, j + l + w[0]]. In the event that
    the value im2[i, j + l + w[0]] was not defined, D[l, i, j] will be
    very large.
    """
    D = np.zeros((im1.shape[0], im1.shape[1], len(disps)))
    mask = np.zeros(D.shape, dtype=np.bool8)
    
    for idx, d in enumerate(disps):
        g = np.zeros(im2.shape)
        if d < 0:
            g[:, -d:] = im2[:,:d]
            mask[:, -d:, idx] = True
        elif d > 0:
            g[:,:-d] = im2[:,d:]
            mask[:, :-d, idx] = True
        else:
            g = im2
            mask[:, :, idx] = True
        aux = localssd(im1, g, K)
        D[..., idx] = aux
    
    D[mask==False] = np.inf #1000*np.abs(D.max())
    colmask = mask.all(-1).any(0)
    D = D[:, colmask]
    return D


def solve(img, disp_min, disp_max, smoothness):
    """img must be an array of np.float_"""
    D = ssd_volume(img, img, np.arange(disp_min, disp_max), 5)
    num_labels = D.shape[-1]
    X,Y = np.mgrid[:num_labels, :num_labels]
    V = smoothness*np.float_(np.abs(X-Y))
    sol = fastmin.aexpansion_grid(D, V, 3)
    return sol


def example():
    img = imread("stereogram.png")/255.0
    return solve(img, 100, 160, 2.0)


class MyParser(argparse.ArgumentParser): 
   def error(self, message):
      sys.stderr.write('error: %s\n' % message)
      self.print_help()
      sys.exit(2)


def main():
    
    parser = MyParser(
        description="Solve a stereogram using alpha-expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example of use:\n\tpython stereogram_solver.py stereogram.png 50 200 solution.png")
    parser.add_argument('input_image', type=str,
        help="stereogram image file name")
    parser.add_argument('min', type=int,
        help="the minimum disparity")
    parser.add_argument('max', type=int,
        help="the maximum disparity")
    parser.add_argument('-smooth', type=float, default=2.0,
        help="the strengh of the smoothness")
    parser.add_argument('output_image', type=str, default="output.png",
        help="output image file name")
    
    args = parser.parse_args()
    
    img = imread(args.input_image)
    if img.dtype == np.uint8:
        img = img / 255.0
    
    logging.basicConfig(level=logging.INFO)
    
    res = solve(img, args.min, args.max, args.smooth)
    imsave(args.output_image, res)


if __name__ == '__main__':
    main()
