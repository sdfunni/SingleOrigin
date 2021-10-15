import copy
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib.pyplot as plt

from scipy.optimize import minimize, LinearConstraint
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)
from scipy.interpolate import SmoothBivariateSpline

from autograd import jacobian

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
    d_hkl = IntPlSpc(h ,k , l, g) / 1e10
    print(d_hkl)
    two_theta=np.degrees(2 * np.arcsin(wavelength / (2 * d_hkl)))
    return two_theta
    
def elec_wavelength(V=200e3):
    '''Electron wavelength for a given accelerating voltage
        V: accelerating voltage in volts'''
    m_e = 9.109e-31 #electron mass (kg)
    e = 1.602e-19 #elementary charge (C)
    c = 2.997e8 #speed of light (m/s)
    h = 6.626e-34 #Plank's constant (Nms)

    lamb = h/(2*m_e*e*V*(1+e*V/(2*m_e*c**2)))**.5 
    return lamb
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
    
    p0 = p0.flatten()
    num_gauss = p0.shape[0]//7
    # print(num_gauss)
    
    R = copy.deepcopy(z)
    
    if p0.shape[0] % 7 ==  1:
        I0 = p0[num_gauss*7]
        R -= I0
        p0 = p0[:-1].reshape((num_gauss, 7))
    else: p0 = p0.reshape((num_gauss, 7))
    
    for p0_ in p0:
        [x0, y0, sig_maj, sig_ratio, ang, A, I_o] = p0_
        sig_maj = float(sig_maj)
        sig_min = float(sig_maj/sig_ratio)
        R -= (I_o + A*np.exp(-1/2*(((np.cos(ang) * (x - x0)
                                     + np.sin(ang) * (y - y0)) / sig_maj)**2
                                   +((-np.sin(ang) * (x - x0) 
                                     + np.cos(ang) * (y - y0)) / sig_min)**2)))
    
    r_sum_sqrd = (R @ R.T).flatten()
    # print(r_sum_sqrd)
    return r_sum_sqrd

def fit_gaussian2D(data, p0, method='L-BFGS-B'):
    """Returns (x0, y0, sig_maj, sig_rat, ang, peak)
       the Gaussian parameters of a 2D distribution found by a fit"""
       
    if method not in ['L-BFGS-B', 'Powell', 'trust-constr']:
       raise Exception('only methods "L-BFGS-B", "Powell" and "trust-constr"'
                        + ' are supported')
    print(method)
    p0 = p0.flatten()
    num_gauss = p0.shape[0]//7
    I0 = p0[-1]
    
    if p0.shape[0] % 7 == 1:
        p0_ = p0[:-1].reshape((num_gauss, 7))
    else: p0_ = p0.reshape((num_gauss, 7))
    
    p0_[:, 4] *= -1
    
    y, x = np.indices(data.shape)
    z=data.ravel()
    
    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x, unmasked_data)
    y = np.take(y, unmasked_data)
    
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (1, np.inf), (1, np.inf), 
              (-np.pi/2, np.pi/2), (0,2), (0,0)] * num_gauss 
    
    p0_ = p0_.flatten()
    
    if p0.shape[0] % 7 == 1:
        bounds.append((0,1))
        p0_ = np.append(p0_.flatten(), I0)
    else: bounds[-1] = (0,1)
        
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, 
                                lineno=182)
        params = minimize(gaussian2d_ss, p0_, args=(x, y, z), 
                          bounds=bounds, method = method).x
        
    # if (method == 'L-BFGS-B') | (method == 'Powell'):
    #     params = minimize(gaussian2d_ss, p0_, args=(x, y, z), 
    #                       bounds=bounds, method = method).x
        
    # elif method == 'trust-constr':
    #     n = len(bounds)
    #     linear_constraint = LinearConstraint(
    #         np.identity(n),
    #         [bnd[0] for bnd in bounds],
    #         [bnd[1] for bnd in bounds])
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings('ignore', category=UserWarning, 
    #                                 lineno=182)
    #         params = minimize(gaussian2d_ss, p0_, args=(x, y, z), 
    #                               constraints=linear_constraint, 
    #                               # jac='cs',
    #                               # bounds = bounds,
    #                               method = method).x
        
    # else:
    #     raise Exception('only methods "L-BFGS-B", "Powell" and "trust-constr"'
    #                     + ' are supported')
    
    I0 = params[-1]
    if params.shape[0] % 7 == 1:
        params = params[:-1].reshape((num_gauss, 7))
    else: params = params.reshape((-1, 7))
    
    params[:, 4] *= -1
    params[:, -1] = I0
    
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
            {'x_ref' : list(coords[:, 1]),
             'y_ref' : list(coords[:, 0]),
             'label' : [i+1 for i in range(nlabels)]})
    
    peaks = peaks[((peaks.x_ref >= buffer) &
                   (peaks.x_ref <= w - buffer) &
                   (peaks.y_ref >= buffer) &
                   (peaks.y_ref <= h - buffer))]
    peaks = peaks.reset_index(drop=True)
    
    return peaks, img_der, labels, slices

def watershed_segment(image, sigma=None, buffer=0, 
                      edge_max_thresholding = 0.95,
                      watershed_line=True):
    
    img_der = copy.deepcopy(image)
    [h, w] = image.shape
    
    if type(sigma) is (int or float or tuple):
        img_der = image_norm(-gaussian_laplace(img_der, sigma))
        
    # neighborhood = generate_binary_structure(2,2)
    neighborhood = np.ones((9,9))
    local_max, _ = ndimage.label(maximum_filter(img_der,
                                footprint=neighborhood)==img_der)
    masks = watershed(-img_der,local_max, watershed_line=watershed_line)
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

def gauss_position_refine(peaks, image, slices=None, labels=None,
                          sigma=3, edge_max_thresholding = 0.95):
    '''Refine positions of atom columns or diffraction spots by Gaussian 
        fitting to intensity peaks.
        - "peaks": DataFrame containing the approximate peak positions
        - "image": the data on which to perform the fitting
        - "labels": Found object labels from thresholding operation
        - "sigma": the Gaussian filter sigma value for smoothing
            prior to thresholding. Smoothed version of the image is only used
            to determine the thresholded region within each window. All fitting
            is performed on the values in "image" unless 
            fit_filtered_data=False.
        -
            '''
    
    # '''Check args for conditions, issue warnings or exceptions'''
    # if fit_filtered_data == True:
    #     if sigma > 0:
    #         img_der = image_norm(gaussian_filter(image, sigma=sigma))

    #     else: 
    #         if type(img_der) == type(None):
    #             print('Warning: Assuming "image" is filtered.',
    #                   'Fitting to arg "image"')
    #             img_der = image

    # elif img_der.shape != image.shape:
    #     raise Exception("'img_der' and 'image' must have the same shape")
    
    img_gauss = image_norm(gaussian_filter(self.image, sigma))

    img_LoG = image_norm(-gaussian_laplace(self.image, sigma))
    
    '''Apply Watershed segmentation to generate masks'''
    
    masks, num_masks, slices, _ = watershed_segment(
        img_LoG, 
        edge_max_thresholding = edge_max_thresholding,
        watershed_line=True)

    '''Find corresponding mask for each reference lattice position, 
        refine masks to remove asymmetric background,
        apply to image and create stack of masked atom column image slices'''
    
    '''Find closest image peak for each referece lattice point and match 
        the peaks to the closest mask'''
    
    xy_ref = peaks.loc[:, 'x_ref':'y_ref'].to_numpy()
    
    labels_at_coords = ndimage.map_coordinates(masks, np.flipud(xy_ref.T), 
                                               order=0).astype(int)
    
    masks_used = np.unique(labels_at_coords)
    
    '''Save all masks which correspond to at least one reference lattice
        point'''
    masks_ref = np.where(np.isin(masks, masks_used), masks, 0)
    
    '''Find sets of reference columns for each mask'''
    mask_ref_matches = [[mask_num, np.argwhere(labels_at_coords==mask_num)]
                        for mask_num in masks_used if mask_num != 0]
    
    max_inds = np.max([match[1].shape[0] for match in mask_ref_matches])
    
    mskd_sl_stack = []
    for i in range(num_masks):
        mask_sl = np.where(masks[slices[i][0],slices[i][1]] == i+1, 1, 0)
        img_sl = image[slices[i][0],slices[i][1]]
        mskd_sl_stack.append(img_sl * mask_sl)
    
    sl_start = np.array([[slices[i][1].start, slices[i][0].start] 
                         for i in range(num_masks)])
        
    '''Pack args into list of lists to pass to fit_column'''
    args_packed = [[mskd_sl_stack[mask_num -1], 
                    sl_start[mask_num - 1],
                    xy_ref[inds, :].reshape((-1, 2)), 
                    peaks.index.to_numpy()[inds],
                    mask_num]
                   for [mask_num, inds]  in mask_ref_matches]
    
    '''Define column fitting function for image slices'''

    def fit_column(args):
        [img_sl, xy_start, xy_ref, inds, mask_num] = args
        num = xy_ref.shape[0]
        
        if num == 1:
            '''Estimate initial guess for 2D Gaussian parameters'''
            eigvals, eigvects, x0, y0 = img_equ_ellip(img_sl)
            max_ind = np.argmax(eigvals)
            min_ind = 1 if max_ind==0 else 0
            sig_max = np.sqrt(np.abs(eigvals[max_ind]))
            sig_min = np.sqrt(np.abs(eigvals[min_ind]))
            if sig_min == 0: sig_min = 1 
            sig_r = sig_max / sig_min
            theta = -np.arcsin(np.cross(np.array([1,0]), eigvects[:, max_ind]))
            I0 = (np.average(img_sl[img_sl != 0])
                  - np.std(img_sl[img_sl != 0]))
            # Z0 = 0
            A0 = np.max(img_sl) - I0
            
            '''Fit 2D Guassian'''
            p0 = np.array([[x0, y0, sig_max, sig_r, theta, A0, I0]])
            
            params = fit_gaussian2D(img_sl, p0)
            
            params = np.array([params[:,0] + xy_start[0],
                                params[:,1] + xy_start[1],
                                params[:,2],
                                params[:,2]/params[:,3],
                                params[:,3],
                                np.degrees(params[:,4]),
                                params[:,5],
                                params[:,6]]).T
            # print('***after')
            return params
        
        if num > 1:
            '''Estimate initial guess for 2D Gaussian parameters'''
            eigvals, _, _, _ = img_equ_ellip(img_sl)
            max_ind = np.argmax(eigvals)
            min_ind = 1 if max_ind==0 else 0
            
            x0 = xy_ref[:, 0] - xy_start[0]
            y0 = xy_ref[:, 1] - xy_start[1]
            sig_max = [np.sqrt(np.abs(eigvals[min_ind]))] * num
            sig_r = [1] * num
            theta = [0] * num
            Z0 = [np.average(img_sl[img_sl != 0])
                  - np.std(img_sl[img_sl != 0])] 
            I0 = [0] * num
            
            A0 = [ndimage.map_coordinates(img_sl, 
                                          np.array([y0, x0]), order=0)[0] 
                  - Z0[0] for i in range(num)]
            
            if min(A0) < 0.1:
                Z0 += min(A0) -0.1
                A0 = [ndimage.map_coordinates(img_sl, 
                                              np.array([y0, x0]), order=0)[0] 
                      - Z0[0] for i in range(num)]
            
            '''Fit 2D Guassian'''
            p0 = np.array([x0, y0, sig_max, sig_r, theta, A0, I0]).T
            p0 = np.append(p0.flatten(), Z0)
            
            params = fit_gaussian2D(img_sl, p0, method='trust-constr')
            
            
            params = np.array([params[:,0] + xy_start[0],
                                params[:,1] + xy_start[1],
                                params[:,2],
                                params[:,2]/params[:,3],
                                params[:,3],
                                np.degrees(params[:,4]),
                                params[:,5],
                                params[:,6]]).T
            
            return params
        
    '''Run fitting routine'''
     
    
    t0 = time.time()
    if len(args_packed) >= 1e2:
        '''Large data set: use parallel processing'''
        print('Using parallel processing')
        n_jobs = psutil.cpu_count(logical=False)
        
        results = Parallel(n_jobs=n_jobs)(delayed(fit_column)(arg) 
                                       for arg in tqdm(args_packed))
        
        results = np.concatenate([np.concatenate((result, args_packed[i][3]), 
                                                 axis=1)
                                  for i, result in enumerate(results)])
    
    else:
        '''Small data set: use serial processing'''
        results = np.concatenate([np.concatenate((fit_column(arg), arg[3]), 
                                                 axis=1)
                                  for arg in tqdm(args_packed)])
        
    warnings.resetwarnings()
    print('Done. Fitting time:', time.time() - t0)
    
    '''Process results and return'''
    
    col_labels = ['x_fit', 'y_fit', 'sig_maj', 'sig_min', 'sig_rat',
                  'theta', 'peak_int', 'bkgd_int']
    if not col_labels[0] in peaks.columns:
        empty = pd.DataFrame(index=peaks.index.tolist(), columns = col_labels)
        peaks = peaks.join(empty)
    
    results = pd.DataFrame(data=results[:, :-1], 
                           index=results[:,-1].astype(int), 
                           columns=col_labels).sort_index()
    
    peaks.update(results)
    
    return peaks, masks_ref, args_packed