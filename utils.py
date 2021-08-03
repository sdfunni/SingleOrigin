import copy
import numpy as np
import pandas as pd
from scipy import ndimage, fftpack

from scipy.optimize import minimize
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)
from scipy.interpolate import SmoothBivariateSpline

from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import psutil
from joblib import Parallel, delayed
import time
from tqdm import tqdm
import cv2

#%%
'''General Crystallographic computations'''
def metric_tensor(a, b, c, alpha, beta, gamma):
    '''Given lattice parameters (with angles in degrees) calculate the metric
        tensor'''
        
    [alpha, beta, gamma] = np.radians([alpha, beta, gamma])
    g=np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
                [b*a*np.cos(gamma), b**2, b*c*np.cos(alpha)],
                [c*a*np.cos(beta), c*b*np.cos(alpha), c**2]])
    g[abs(g) <= 1e-10] = 0
    return g

def bond_length(at1, at2, g):
    '''Given two atom positions (as list or Numpy array) and metric tensor,
    calculate the bond length'''
    
    at1 = np.array(at1)
    at2 = np.array(at2)
    at12 = at2 - at1
    d = np.sqrt(at12 @ g @ at12.T)
    return d

def dot_prod(vect1, vect2, g):
    '''Given two atom positions (as list or Numpy array) and metric tensor,
    calculate the bond length'''
    
    vect1 = np.array(vect1)
    vect2 = np.array(vect2)
    dot = vect1 @ g @ vect2.T    
    
    return dot

def bond_angle(at1 ,at2, at3, g):
    '''Given three atomic positions (as list or Numpy array), with at2 being 
    the vertex, and the metric tensor ("g"), find the bond angle if the second
    position is the angle vertex'''
    
    at1 = np.array(at1)
    at2 = np.array(at2)
    at3 = np.array(at3)
    
    p_q = np.array([at1 - at2, at3 - at2])
    
    [[pp, pq], [qp, qq]] = np.array(p_q @ g @ p_q.T)
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))
    return theta

def IntPlSpc(h, k, l, g):
    '''Given hkl plane indices and metric tensor, calculate interplanar
        spacing'''
    hkl = np.array([h, k, l])
    d_hkl = (hkl @ np.linalg.inv(g) @ hkl.T)**-0.5
    return d_hkl

def IntPlAng(hkl_1 ,hkl_2 , g):
    '''Given the hkl indices for two planes and metric tensor, calculate
        the angle between the planes'''
    p_q = np.array([hkl_1, hkl_2])
    [[pp, pq], [qp, qq]] = np.array(p_q @ np.linalg.inv(g) @ p_q.T)
    
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))
    return theta

def TwoTheta(h, k, l, g, wavelength):
    '''Given hlk plane indices, metric tensor and radiation wavelength,
        calculate the two theta scattering angle'''
    d_hkl = IntPlSpc(h ,k , l, g)
    two_theta=np.degrees(2 * np.arcsin(wavelength / (2 * d_hkl)))
    return two_theta
    
#%%
'''General image functions'''

def image_norm(image):
    '''Normalize intensity of a 1 channel image to the range 0-1.'''
    [min_, max_] = [np.min(image), np.max(image)]
    image = (image - min_)/(max_ - min_)
    return image

def image_moment(image, i, j):
    y, x = np.indices(image.shape)
    nonzero_data = np.nonzero(image.flatten())
    img_ = np.take(image, nonzero_data)
    x_ = np.take(x, nonzero_data)
    y_ = np.take(y, nonzero_data)
    M = np.sum(x_**i * y_**j * img_)
    return M
    
def img_equ_ellip(image):
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
    '''Returns equivalent ellipse eccentricity and rotation angle'''
    eigvals, eigvects, x0, y0 = img_equ_ellip(image)
    major = np.argmax(eigvals)
    minor = np.argmin(eigvals)
    sig_maj = np.sqrt(eigvals[major])
    sig_min = np.sqrt(eigvals[minor])
    eccen = np.sqrt(1-eigvals[minor]/eigvals[major])
    theta = np.degrees(-np.arcsin(np.cross(np.array([1,0]),
                                           eigvects[:, major])))
    
    return x0, y0, eccen, theta, sig_maj, sig_min

def gaussian_2d(x, y, x0, y0, sig_maj, sig_ratio, ang, A=1, I_o=0):
    """Returns a sampled eliptcal gaussian function with the given parameters 
       for generating data."""
    sig_maj = float(sig_maj)
    sig_min = float(sig_maj/sig_ratio)
    ang = np.radians(-ang) #negative due to origin being at top left of image
    I = I_o + A*np.exp(-1/2*(
           ((np.cos(ang) * (x - x0) + np.sin(ang) * (y - y0)) / sig_maj)**2
           +((-np.sin(ang) * (x - x0) + np.cos(ang) * (y - y0)) / sig_min)**2))
    return I

def gaussian2d_ss(p0, x, y, z):
    """Returns an eliptical gaussian function for fitting with 
       optimize.minimize. Parameters include major axis, minor axis and 
       counterclockwise rotation angle. Zero rotation angle is major axis 
       aligned with the x-axis."""
    [x0, y0, sig_maj, sig_ratio, ang, A, I_o] = p0
    sig_maj = float(sig_maj)
    sig_min = float(sig_maj/sig_ratio)
    R = (I_o + A*np.exp(-1/2*(
           ((np.cos(ang) * (x - x0) + np.sin(ang) * (y - y0)) / sig_maj)**2
           +((-np.sin(ang) * (x - x0) + np.cos(ang) * (y - y0)) / sig_min)**2))
        - z)
    r_sum_sqrd = (R @ R.T).flatten()
    return r_sum_sqrd

def fit_gaussian2D(data, p0):
    """Returns (x0, y0, sig_maj, sig_rat, ang, peak)
       the Gaussian parameters of a 2D distribution found by a fit"""
    p0[4] *= -1
    y, x = np.indices(data.shape)
    z=data.ravel()
    
    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x, unmasked_data)
    y = np.take(y, unmasked_data)
    
    bounds = [(None, None), (None, None), (1, np.inf), (1, None),
              (-np.pi/2, np.pi/2), (0,2), (0,1)]
    params = minimize(gaussian2d_ss, p0, args=(x, y, z), bounds=bounds).x
    params[4] *= -1
    p0[4] *= -1
    return params


def pcf_radial(dr, x_coords, y_coords, z_coords=None, total_area=None):
    '''PCF for specified sublattice'''
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    if type(z_coords) == type(None):
        coords = np.vstack((x_coords, y_coords)).T
    else:
        z_coords = z_coords.flatten()
        coords = np.vstack((x_coords, y_coords, z_coords)).T
    
    N = coords.shape[0]
    rho = N / total_area
    
    vects = np.array([coords - i for i in coords])
    
    dist = np.hypot(vects[:,:,0], vects[:,:,1])
    bins = (dist/dr).astype(int)
    
    r = np.arange(0, np.max(dist), dr)
    A_sh = np.array([np.pi * (r_**2 - (r_ - dr)**2) for r_ in r])
    
    hist = np.bincount(bins.flatten())
    hist[0] = 0
        
    pcf = hist / (N * rho * A_sh)
    
    return pcf
    
def filter_thresh_CoM(image, buffer, Gauss_filter=None,  LaplacianGauss=None,
                     Abs_Thresh=0.5):
    '''Apply filtering, thresholding. Find atomic column or diffraction spot
        locations by center-of-mass method.
        -image: atomic resolution image or diffraction pattern
        -buffer: image edge region to ignore in peak finding
        -Gauss_filter: apply Gaussian smoothing. If desired, float or two-tuple
        of sigma values. Default: False.
        -LaplacianGauss: apply Laplacian of Gaussian filtering. Helps 
        seperate closely spaced columns. If desired, float or two-tuple
        of sigma values. Default: False.
        -Adaptive_Thresh: apply adaptive thresholding. Helps if background
        intensity varies across the image. If desired, pass dictionary 
        with keys: blockSize & C. See cv2 module documentation for 
        details.
        -Abs_Thresh: Absolute image thresholding value. Applied after
        any specified filtering. Adaptive thresholding takes priority.
        Default is 0.5.'''
    
    img_der = copy.deepcopy(image)
    [h, w] = image.shape
    
    if type(LaplacianGauss) is (int or float or tuple):
        for i in range(2):
            img_der = image_norm(-gaussian_laplace(img_der, LaplacianGauss))

    if type(Gauss_filter) is (int or float or tuple):
        img_der = image_norm(gaussian_filter(img_der, sigma=Gauss_filter))
    
    img_th = np.where(img_der > Abs_Thresh, 1, 0)
        
    labels, nlabels = ndimage.label(img_th)
    slices = ndimage.find_objects(labels)
    coords = np.array(ndimage.center_of_mass(
        img_th*img_der, labels, np.arange(nlabels)+1))
    peaks = pd.DataFrame.from_dict(
            {'x' : list(coords[:, 1]),
             'y' : list(coords[:, 0]),
             'label' : [i+1 for i in range(nlabels)]})
    
    peaks = peaks[((peaks.x >= buffer) &
                   (peaks.x <= w - buffer) &
                   (peaks.y >= buffer) &
                   (peaks.y <= h - buffer))]
    peaks = peaks.reset_index(drop=True)
    
    return peaks, img_der, labels, slices

def watershed_segment(image, sigma=None, buffer=0, 
                      edge_max_thresholding = 0.95):
    
    img_der = copy.deepcopy(image)
    [h, w] = image.shape
    
    if type(sigma) is (int or float or tuple):
        img_der = image_norm(-gaussian_laplace(img_der, sigma))
        
    neighborhood = generate_binary_structure(2,2)
    local_max, _ = ndimage.label(maximum_filter(img_der,
                                footprint=neighborhood)==img_der)
    masks = watershed(-img_der,local_max, watershed_line=True)
    slices = ndimage.find_objects(masks)
    num_masks = int(np.max(masks))
    
    masks_ref = np.zeros(image.shape)
    
    '''Refine masks with edge_max_thresholding'''
    for i in range(num_masks):
        mask_sl = np.where(masks[slices[i][0],slices[i][1]]== i+1, 1, 0)
        img_der_sl = img_der[slices[i][0],slices[i][1]]
        edge = mask_sl - ndimage.morphology.binary_erosion(mask_sl)
        thresh = np.max(edge*img_der_sl) * edge_max_thresholding
        mask_sl = np.where(mask_sl*img_der_sl > thresh, 1, 0)
        masks_ref[slices[i][0],slices[i][1]] += mask_sl * (i+1)
    
    mask_com = np.fliplr(ndimage.center_of_mass(img_der,
                                                masks_ref,
                                                    [i+1 for i in 
                                                 range(num_masks)]))
    
    peaks = pd.DataFrame.from_dict(
            {'x' : list(mask_com[:, 0]),
             'y' : list(mask_com[:, 1]),
             'label' : [i+1 for i in range(num_masks)]})
    
    peaks = peaks[((peaks.x >= buffer) &
                   (peaks.x <= w - buffer) &
                   (peaks.y >= buffer) &
                   (peaks.y <= h - buffer))]
    peaks = peaks.reset_index(drop=True)
    
    return masks_ref, num_masks, slices, peaks

def gauss_position_refine(peaks, image, img_der=None, slices=None, labels=None,
                          fit_filtered_data=False, Gauss_smooth=3,
                          edge_max_thresholding = 0.95):
    '''Refine positions of atom columns or diffraction spots by Gaussian 
        fitting to intensity peaks.
        - "image": the data on which to perform the fitting
        - "labels": Found object labels from thresholding operation
        - "Gauss_smooth": the Gaussian filter sigma value for smoothing
            prior to thresholding. Smoothed version of the image is only used
            to determine the thresholded region within each window. All fitting
            is performed on the values in "image" unless 
            fit_filtered_data=False.
        -
            '''
    
    '''Check args for conditions, issue warnings or exceptions'''
    if fit_filtered_data == True:
        if Gauss_smooth > 0:
            img_der = image_norm(gaussian_filter(image, sigma=Gauss_smooth))

        else: 
            if type(img_der) == type(None):
                print('Warning: Assuming "image" is filtered.',
                      'Fitting to arg "image"')
                img_der = image
            
    # if (type(img_der) == type(None)) and (thresh_smoothing == 0):
    #     img_der = image
    # elif Gauss_smooth > 0:
    #     print('smoothing image')
    #     img_der = image_norm(gaussian_filter(image, sigma=Gauss_smooth))

    elif img_der.shape != image.shape:
        raise Exception("'img_der' and 'image' must have the same shape")
        
    '''Apply Watershed segmentation to generate masks'''
    masks, num_masks, slices, mask_com = watershed_segment(img_der)
    
    '''Find corresponding mask for each reference lattice position, 
        refine masks to remove asymmetric background,
        apply to image and create stack of masked atom column image slices'''
    def xy(row):
        xy = np.array([row['x_ref'], row['y_ref']]).T 
        return xy
    
    inds = [np.argmin(np.linalg.norm(mask_com.loc[:,'x':'y'] - xy(row), 
                                     axis=1)) for _, row in peaks.iterrows()]

    mskd_sl_stack = []
    masks_ref = np.zeros(image.shape)
    
    for i in inds:
        mask_sl = np.where(masks[slices[i][0],slices[i][1]]== i+1, 1, 0)
        img_sl = image[slices[i][0],slices[i][1]]
        masks_ref[slices[i][0],slices[i][1]] += mask_sl * (i+1)
        mskd_sl_stack.append(img_sl * mask_sl)
        
    '''save (x, y )start of each image slice and slice CoM coordinates'''
    sl_start = np.array([[slices[i][1].start, slices[i][0].start] 
                        for i in inds])
    
    '''Save all masks used for atom columns'''
    masks_ref = masks_ref.astype(int)
    
    '''Pack args into list of lists to pass to fit_column'''
    args_packed= [[mskd_sl_stack[i], sl_start[i]]
                 for i in range(len(mskd_sl_stack))]
    
    '''Define column fitting function for image slices'''
    def fit_column(args):
        [img_sl, xy_start] = args
        '''Estimate initial guess for 2D Gaussian parameters'''
        eigvals, eigvects, x0, y0 = img_equ_ellip(img_sl)
        max_ind = np.argmax(eigvals)
        min_ind = 1 if max_ind==0 else 0
        sig_max = np.sqrt(np.abs(eigvals[max_ind]))
        sig_min = np.sqrt(np.abs(eigvals[min_ind]))
        sig_r = sig_max / sig_min
        theta = -np.arcsin(np.cross(np.array([1,0]), eigvects[:, max_ind]))
        Z0 = (np.average(img_sl[img_sl != 0])
              - np.std(img_sl[img_sl != 0]))
        A0 = np.max(img_sl) - Z0
        
        '''Fit 2D Guassian'''
        p0 = np.array([x0, y0, sig_max, sig_r, theta, A0, Z0])
        
        params = fit_gaussian2D(img_sl, p0)
        
        return [params[0] + xy_start[0],
                params[1] + xy_start[1],
                params[2],
                params[2]/params[3],
                params[3],
                np.degrees(params[4]),
                params[5],
                params[6]]
    
    '''Run fitting routine'''
    '''Large data set: use parallel processing''' 
    t0 = time.time()
    if len(args_packed) >= 100:
        n_jobs = psutil.cpu_count(logical=False)
        results = np.array(Parallel(n_jobs=n_jobs)(delayed(fit_column)(arg) 
                                       for arg in tqdm(args_packed)),
                           dtype=np.float64)
    
    '''Small data set: use serial processing'''
    if len(args_packed) < 100:
        results = np.array([fit_column(args) for args in tqdm(args_packed)],
                           dtype=np.float64)
    
    print('Done. Fitting time:', time.time() - t0)
    
    '''Process results and return'''
    
    col_labels = ['x_fit', 'y_fit', 'sig_maj', 'sig_min', 'sig_rat',
                  'theta', 'peak_int', 'bkgd_int']
    
    for i, lab in enumerate(col_labels):
        if lab in list(peaks.columns): del peaks[lab]
        loc = peaks.shape[1]
        peaks.insert(loc, lab, results[:, i])
    
    return peaks, masks_ref


def slices_from_coordinates(x_coords, y_coords, window_size=7,
                            filter_label=None):
    '''Get slices from an image that cannot be properly thresholded using
        either the ACI or watershed methods. Slices are odd-shaped, square
        kernals around a central pixel given by "x_coords, y_coords."
        -window_size: single int or dictionary of ints of the form:
            {"column_type" : window_size}
        -filter_label: if dict passed for window_size, the dataframe column
            label in which to find the specified column types.
    '''
        
    slices = []
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    for ind in range(x_coords.shape[0]):
        if type(window_size) == dict:
            r = int(window_size[filter_label[ind]]/2)
            
        else: r = int(window_size/2)
        
        sl_rows = slice(int(y_coords[ind]) - r, int(y_coords[ind]) + (r+1))
        sl_cols = slice(int(x_coords[ind]) - r, int(x_coords[ind]) + (r+1))
        slices.append((sl_rows, sl_cols))
    return slices
