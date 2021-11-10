import copy
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)
from scipy.optimize import minimize

import imageio
from ncempy.io.dm import dmReader 
from ncempy.io.ser import serReader
from ncempy.io.emdVelox import emdVeloxReader

from skimage.segmentation import watershed

import cv2

from matplotlib import pyplot as plt


#%%
"""General Crystallographic computations"""
def metric_tensor(a, b, c, alpha, beta, gamma):
    """Calculate the metric tensor for a lattice.
    
    Parameters
    ----------
    a, b, c : ints or floats
        Basis vector magnitudes
        
    alpha, beta, gamma : floats
        Basis vector angles in degrees 
        
    Returns
    -------
    g : 3x3 ndarray
        The metric tensor
        
    """
        
    [alpha, beta, gamma] = np.radians([alpha, beta, gamma])
    g=np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])
    g[abs(g) <= 1e-10] = 0
    return g

def bond_length(at1, at2, g):
    """Calculate distance between two lattice points.
    
    Parameters
    ----------
    at1, at2 : array_like of shape (1,3) or (3,)
        Lattice point vectors in fractional coordinates
        
    g : 3x3 ndarray
        The metric tensor
        
    Returns
    -------
    d : float
        Distance between the points in real units
        
    """
    
    at1 = np.array(at1)
    at2 = np.array(at2)
    at12 = at2 - at1
    d = np.sqrt(at12 @ g @ at12.T).item()
    return d

def bond_angle(at1 ,at2, at3, g):
    """Calculate the angle from a central point to two other points
    
    Parameters
    ----------
    at1, at3 : array_like of shape (1,3) or (3,)
        Two points forming rays from the vertex. Positions in fractional 
        coordinates.
        
    at2 : array_like of shape (1,3) or (3,)
        Vertex. Position in fractional coordinates.
        
    g : 3x3 ndarray
        The metric tensor
        
    Returns
    -------
    theta : float
        Angle in degrees
        
    """
    
    at1 = np.array(at1)
    at2 = np.array(at2)
    at3 = np.array(at3)
    
    p_q = np.array([at1 - at2, at3 - at2])
    
    [[pp, pq], [qp, qq]] = np.array(p_q @ g @ p_q.T)
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))
    return theta

def IntPlSpc(hkl, g):
    """Calculate the spacing of a set of lattice planes
    
    Parameters
    ----------
   hkl : array_like of ints of shape (1,3) or (3,)
        Miller indices of the lattice plane
    
    g : 3x3 ndarray
        The metric tensor
        
    Returns
    -------
    d_hkl : float
        Inter-planar spacing
        
    """
    hkl = np.array(hkl)
    d_hkl = (hkl @ np.linalg.inv(g) @ hkl.T)**-0.5
    return d_hkl

def IntPlAng(hkl_1 ,hkl_2 , g):
    """Calculate the spacing of a set of lattice planes
    
    Parameters
    ----------
    hkl_1 ,hkl_2 : array_like of ints of shape (1,3) or (3,)
        Miller indices of the lattice planes
    
    g : 3x3 ndarray
        The metric tensor
        
    Returns
    -------
    theta : float
        Inter-planar angle
        
    """
    
    p_q = np.array([hkl_1, hkl_2])
    [[pp, pq], [qp, qq]] = np.array(p_q @ np.linalg.inv(g) @ p_q.T)
    
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))
    return theta

def TwoTheta(hkl, g, wavelength):
    """Calculate two theta
    
    Parameters
    ----------
    hkl : array_like of ints of shape (1,3) or (3,)
         Miller indices of the lattice plane
    
    g : 3x3 ndarray
        The metric tensor
        
    wavelength : wavelength of the incident radiation in meters
        
    Returns
    -------
    two_theta : float
        Bragg diffraciton angle
        
    """
    
    hkl = np.array(hkl)
    d_hkl = IntPlSpc(hkl, g) * 1e-10
    print(d_hkl)
    two_theta=np.degrees(2 * np.arcsin(wavelength / (2 * d_hkl)))
    return two_theta
    
def elec_wavelength(V=200e3):
    """Electron wavelength as a function of accelerating voltage
    
    Parameters
    ----------
    V : int or float
         Accelerating voltage
         
    Returns
    -------
    wavelength : float
        Electron wavelength in meters
        
    """
    m_e = 9.109e-31 #electron mass (kg)
    e = 1.602e-19 #elementary charge (C)
    c = 2.997e8 #speed of light (m/s)
    h = 6.626e-34 #Plank's constant (Nms)

    wavelength = h/(2*m_e*e*V*(1+e*V/(2*m_e*c**2)))**.5 
    return wavelength
#%%
"""General image functions"""
def import_image(path, display_image=True):
    """Import image from path and plot
    
    Parameters
    ----------
    path : str
         The location of the image file. Supports .dm3/.dm4, .emd, .ser file 
         types in addition to open source formats such as .tif and .png (all 
         formats supported by imageio.imread()).
    display_image : bool
        Whether to plot image after importing.
        Default: True 
         
    Returns
    -------
    image : ndarray
        The imported image
        
    """
    
    if path[-3:] in ['dm4', 'dm3']:
        dm_file = dmReader(path)
        image = dm_file['data']
        
    elif path[-3:] == 'emd':
        emd_file = emdVeloxReader(path)
        image = emd_file['data']
        
    elif path[-3:] == 'ser':
        ser_file = serReader(path)
        image = ser_file['data']
    
    else:
        image = imageio.imread(path).astype('float64')
    
    image = image_norm(image)

    if display_image == True:
        fig, axs = plt.subplots()
        axs.imshow(image, cmap='gray')
        axs.set_xticks([])
        axs.set_yticks([])
        
    return image

def image_norm(image):
    """Norm an image to 0-1
    
    Parameters
    ----------
    image : ndarray
         Input image as an ndarray
         
    Returns
    -------
    image_normed : ndarray with the same shape as 'image'
        The normed image
        
    """
    
    [min_, max_] = [np.min(image), np.max(image)]
    image_normed = (image - min_)/(max_ - min_)
    return image_normed
  
    
def img_equ_ellip(image):
    """Calculate the equivalent ellipse
    
    Parameters
    ----------
    image : ndarray
         Input image as an ndarray
         
    Returns
    -------
    eigvals : squared magnitudes of the major and minor semi-axes, in that 
        order
    eigvects : unit vectors of the major and minor semi-axes, in that order
    x0, y0 : coordinates of the ellipse center
    
    """
    
    M = cv2.moments(image)
    [x0, y0] = [M['m10']/M['m00'], M['m01']/M['m00']]
    [u20, u11, u02] = [M['mu20']/M['m00'], M['mu11']/M['m00'], M['mu02']/M['m00']]
    cov = np.array([[u20, u11],
                    [u11, u02]])
    eigvals, eigvects = np.linalg.eig(cov)
    
    if eigvects[0,0]<0:
        eigvects[:,0] *= -1
    if eigvects[0,1]<0:
        eigvects[:,1] *= -1
        
    ind_sort = np.argsort(eigvals)
    eigvals = np.take_along_axis(eigvals, ind_sort, 0)
    eigvects = np.take_along_axis(eigvects, np.array([ind_sort,ind_sort]), 1)
    return eigvals, eigvects, x0, y0


def img_ellip_param(image):
    """Find parameters of the equivalent ellipse
    
    Calls img_equ_ellip and transforms result to a more intuitive form
    
    Parameters
    ----------
    image : ndarray
         Input image as an ndarray
         
    Returns
    -------
    x0, y0 : coordinates of the ellipse center
    eccen : eccentricity of the ellipse (standard mathmatical definition)
    theta : rotation angle of the major semi-axis relative to horizontal 
        in degrees (positive is counterclockwise)
    sig_1 : magnitude of the major semi-axis
    sig_2 : magnitude of the major semi-axis
    
    """
    
    eigvals, eigvects, x0, y0 = img_equ_ellip(image)
    major = np.argmax(eigvals)
    minor = np.argmin(eigvals)
    sig_1 = np.sqrt(eigvals[major])
    sig_2 = np.sqrt(eigvals[minor])
    eccen = np.sqrt(1-eigvals[minor]/eigvals[major])
    theta = np.degrees(-np.arcsin(np.cross(np.array([1,0]),
                                           eigvects[:, major])))
    
    return x0, y0, eccen, theta, sig_1, sig_2


def gaussian_2d(x, y, x0, y0, sig_1, sig_2, ang, A=1, I_o=0):
    """Sample a 2D, ellpitical Gaussian function.
    
    Samples a specified 2D Gaussian function at an array of points.
    
    Parameters
    ----------
    x, y : ndarrays, must have the same shape
        They x and y coordinates of each sampling point. If given arrays 
        generated by numpy.mgrid or numpy.meshgrid, will return an image
        of the Gaussian.
    x0, y0 : center of the Gaussian
    sig_1 : sigma of the major axis
    sig_ratio : ratio of sigma major to sigma minor
    ang : rotation angle of the major axis from horizontal
    A : Peak amplitude of the Gaussian
    I_o : Constant background value
         
    Returns
    -------
    I : the value of the function at the specified points. Will have the same
        shape as x, y inputs
    
    """
    
    ang = np.radians(-ang) #negative due to inverted y axis in python
    I = I_o + A*np.exp(-1/2*(
           ((np.cos(ang) * (x - x0) + np.sin(ang) * (y - y0)) / sig_1)**2
           +((-np.sin(ang) * (x - x0) + np.cos(ang) * (y - y0)) / sig_2)**2))
    
    return I


def gaussian2d_ss(p0, x, y, z, masks=None):
    """Sum of squares for a Gaussian function.
    
    Takes a parameter vector, coordinates, and corresponding data values;  
    returns the sum of squares of the residuals.
    
    Parameters
    ----------
    p0 : array_like with shape (n,7)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig_1, sig_2, ang, A, I_o]
    x, y : 1D array_like, must have the same shape
        The flattened arrays of x and y coordinates of image pixels
    z : 1D array_like, must have the same shape as x and y
        The flattened array of image values at the x, y coordinates
    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the  
        mask for the corresponding peak and 0 elsewhere.
         
    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals
    
    """
    
    p0 = p0.reshape((-1, 7))
    #Initialize model by getting background values:
    if p0.shape[0] > 1:
        if type(masks) == type(None):
            raise Exception('Must specify "masks" arg for fitting '
                            +'simultaneous peaks')
        model = np.sum(masks * p0[:,-1].reshape((-1,1)), axis=0)
        
    else:
        model = p0[0, -1]
    
    #Add Gaussians:
    x0, y0, sig_1, sig_2, ang, A, _ = np.split(p0, 7, axis=1)
    model += np.sum(A*np.exp(-1/2*(((np.cos(ang) * (x - x0)
                                     + np.sin(ang) * (y - y0)) / sig_1)**2
                                   +((-np.sin(ang) * (x - x0) 
                                      + np.cos(ang) * (y - y0)) / sig_2)**2)), 
                    axis=0)
    
    #Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()
    
    return r_sum_sqrd


def fit_gaussian2D(data, p0, masks=None, method='BFGS'):
    """Fit a 2D Gaussain function to data.
    
    Fits a 2D, elliptical Gaussian to an image. Zero values are ignored.
    
    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak
    p0 : array_like with shape (7,)
        Initial guess for the Gaussian function parameter vector:
            [x0, y0, sig_1, sig_2, ang, A, I_o]
    
    method : str, the minimization solver name
        Supported solvers are: 'L-BFGS-B', 'Powell', 'trust-constr'.
        Default: 'L-BFGS-B'
         
    Returns
    -------
    params : 1D array
        The best fit parameter vector found by least squares
    
    """
       
    if method not in ['BFGS', 'Powell', 'trust-constr']:
       raise Exception('only methods "BFGS"'
                        + ' are supported')
    p0 = p0.flatten()
    num_gauss = p0.shape[0]//7
    img_shape = data.shape
    
    p0_ = p0.reshape((num_gauss, 7))
    
    p0_[:, 4] *= -1
    
    y, x = np.indices(img_shape)
    z=data.ravel()
    
    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x, unmasked_data)
    y = np.take(y, unmasked_data)
    
    x0y0 = p0_[:,:2]    
    if type(masks) == type(None):
        image = np.zeros(img_shape)
        image[y,x] = z
        masks_labeled, _ = ndimage.label(image)
    
    elif (type(masks) == imageio.core.util.Array 
          or type(masks) == np.ndarray):
        masks_labeled, _ = ndimage.label(masks)
        
    masks_to_peaks = ndimage.map_coordinates(masks_labeled, 
                                             np.flipud(x0y0.T), 
                                             order=0).astype(int)
    masks_labeled = np.take(masks_labeled, unmasked_data).ravel()
    
    masks_labeled = np.array([np.where(masks_labeled == mask_num, 1, 0)
                              for i, mask_num in enumerate(masks_to_peaks)])
    
    # bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (1, np.inf), (1, np.inf), 
    #           (-4*np.pi, 4*np.pi), (0,2), (0,1)] * num_gauss 
    
    p0_ = p0_.flatten()
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                                lineno=182)
        params = minimize(gaussian2d_ss, p0_, args=(x, y, z, masks_labeled), 
                          bounds=None, method=method).x
    
    params = params.reshape((-1, 7))
    
    params[:, 4] *= -1
    params[:, 4] = ((params[:, 4] + np.pi/2) % np.pi) - np.pi/2
    
    return params

def pcf_radial(dr, coords, total_area=None):
    """Calculate a radial pair correlation function from 2D or 3D data.
    
    Parameters
    ----------
    dr : int or float
        The step size for binning distances
    x_coords, y_coords, z_coords : array_like with shape (n, d)
        The x, y, z coordinates of each point. n is the number of points, 
        d is the number of dimensions (i.e. 2 or 3)
    
    total_area : area or volume of region containing the points. Estimate 
        made if not given.
         
    Returns
    -------
    pcf : 1D array
        Histogram of point-point distances with bin size dr
    
    """
    
    if total_area == None:
        diag_vect = np.max(coords, axis=0) - np.min(coords, axis=0)
        total_area = diag_vect @ np.ones(diag_vect.shape)
        
    n = coords.shape[0]
    rho = n / total_area
    
    vects = np.array([coords - i for i in coords])
    
    dist = np.hypot(vects[:,:,0], vects[:,:,1])
    bins = (dist/dr).astype(int)
    
    r = np.arange(0, np.max(dist), dr)
    A_sh = np.array([np.pi * (r_**2 - (r_ - dr)**2) for r_ in r])
    
    hist = np.bincount(bins.flatten())
    hist[0] = 0
        
    pcf = hist / (n * rho * A_sh)
    
    return pcf


def detect_peaks(image, min_dist=4, thresh=0):
    """Segment an image using the Watershed algorithm.
    
    Parameters
    ----------
    image : 2D array_like
        The image to be analyzed.
    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
    thresh : int or float
        The minimum image value that should be considered a peak. Used to 
        remove low intensity background noise peaks.
         
    Returns
    -------
    peaks : 2D array_like with shape: image.shape
        Array with 1 indicating peak pixels and 0 elsewhere.
        
    """
    
    kern_rad = int(np.ceil(min_dist))
    size = 2*kern_rad + 1
    neighborhood = np.array([1 if np.hypot(i-kern_rad,j-kern_rad) <= min_dist 
                             else 0 
                             for j in range(size) for i in range(size)]
                            ).reshape((size,size))
    peaks = (maximum_filter(image,footprint=neighborhood)==image
             ) * (image > thresh)
    return peaks.astype(int)


def watershed_segment(image, sigma=None, buffer=0, local_thresh_factor = 0.95,
                      watershed_line=True, min_dist=4):
    """Segment an image using the Watershed algorithm.
    
    Parameters
    ----------
    image : 2D array_like
        The image to be segmented
    sigma : int or float
        The Laplacian of Gaussian sigma value to use for peak sharpening
    buffer : int
        The border within which peaks are ignored
    local_thresh_factor : float
        Removes background from each segmented region by thresholding. 
        Threshold value determined by finding the maximum value of edge pixels
        in the segmented region and multipling this value by the 
        local_thresh_factor value. The filtered image is used for this 
        calculation. Default 0.95.
    watershed_line : bool
        Seperate segmented regions by one pixel. Default True.
    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
         
    Returns
    -------
    masks_ref : 2D array with same shape as image
    num_masks : int
        The number of masks
    slices : List of image slices which contain each region
    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer
        
    """
    
    img_der = copy.deepcopy(image)
    [h, w] = image.shape
    
    if type(sigma) in (int, float, tuple):
        print('log...')
        img_der = image_norm(-gaussian_laplace(img_der, sigma))
        # img_der = image_norm(-gaussian_laplace(img_der, sigma))
        
    local_max, _ = ndimage.label(detect_peaks(image, min_dist=min_dist))
    
    masks = watershed(-img_der,local_max, watershed_line=watershed_line)
    slices = ndimage.find_objects(masks)
    num_masks = int(np.max(masks))
    
    masks_ref = np.zeros(image.shape)
    
    """Refine masks with local_thresh_factor"""
    for i in range(num_masks):
        mask_sl = np.where(masks[slices[i][0],slices[i][1]]== i+1, 1, 0)
        img_der_sl = img_der[slices[i][0],slices[i][1]]
        edge = mask_sl - ndimage.morphology.binary_erosion(mask_sl)
        thresh = np.max(edge*img_der_sl) * local_thresh_factor
        mask_sl = np.where(mask_sl*img_der_sl > thresh, 1, 0)
        masks_ref[slices[i][0],slices[i][1]] += mask_sl * (i+1)
        
    masks = masks_ref
    
    _, peak_xy = np.unique(local_max, return_index=True)
    peak_xy = np.fliplr(np.array(np.unravel_index(peak_xy, 
                                                  local_max.shape)).T[1:, :])
    
    peaks = pd.DataFrame.from_dict(
            {'x' : list(peak_xy[:, 0]),
             'y' : list(peak_xy[:, 1]),
             'label' : [i+1 for i in range(num_masks)]})
    
    peaks = peaks[((peaks.x >= buffer) &
                   (peaks.x <= w - buffer) &
                   (peaks.y >= buffer) &
                   (peaks.y <= h - buffer))]
    peaks = peaks.reset_index(drop=True)
    
    return masks, num_masks, slices, peaks

