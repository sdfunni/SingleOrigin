"""SingleOrigin is a module for atomic column position finding intended for
    high resolution scanning transmission electron microscope images.
    Copyright (C) 2022  Stephen D. Funni

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses"""


import os
import copy
import warnings
import numpy as np
# from numpy.linalg import norm
import pandas as pd
from scipy import ndimage
from scipy.ndimage import (gaussian_filter,
                           gaussian_laplace,
                           maximum_filter)
from scipy.signal import convolve2d
from scipy.optimize import minimize

from PyQt5.QtWidgets import QFileDialog as qfd
import imageio
from ncempy.io.dm import dmReader
from ncempy.io.ser import serReader
from ncempy.io.emdVelox import emdVeloxReader
from ncempy.io.emd import emdReader

from skimage.segmentation import watershed
from skimage.measure import (moments, moments_central)

from tifffile import imwrite

from matplotlib import pyplot as plt

# %%
"""Error message string(s)"""
no_mask_error = ("Float division by zero during moment estimation. No image "
                 + "intensity present to calculate moment. If raised during "
                 + "Guassian fitting of atom columns, this means that no "
                 + "pixel region was found for fitting of at least one atom "
                 + "column. This situation may result from: \n 1) Too high a "
                 + "'local_thresh_factor' value resulting in no pixels "
                 + "remaining for some atom columns. Check 'self.fit_masks' "
                 + "to see if some atom columns do not have mask regions. \n "
                 + "2) Due to mask regions splitting an atom column as a "
                 + "result of too small or too large a value used for "
                 + "'grouping_filter' or 'diff_filter'. Check "
                 + "'self.group_masks' and 'self.fit_masks' to see if this "
                 + "may be occuring. Too small a 'fitting_filter' value may "
                 + "cause noise peaks to be detected, splitting an atom "
                 + "column. Too large a value for either filter may cause low "
                 + "intensity peaks to be ignored by the watershed algorithm. "
                 + "\n Masks may also be checked with the 'show_masks()' "
                 + "method.'")
# %%
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
    g = np.array([[a**2, a*b*np.cos(gamma), a*c*np.cos(beta)],
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


def bond_angle(site1, site2, site3, g):
    """Calculate the angle between two bond two interatomic bonds using the
        dot product. The vertex is at the second atomic position.

    Parameters
    ----------
    site1, site2, site3 : array_like of shape (1,3n or (n,)
        Three atomic positions in fractional coordiantes. "site2" is the
        vertex of the angle; "site1" and "site2" are the end points of the
        bonds.

    g : nxn ndarray
        The metric tensor. If atomic positions are in real Cartesian
        coordinates, give the identitiy matrix.

    Returns
    -------
    theta : float
        Angle in degrees

    """

    vec1 = np.array(site1) - np.array(site2)
    vec2 = np.array(site3) - np.array(site2)
    p_q = np.array([vec1, vec2])

    [[pp, pq], [qp, qq]] = np.array(p_q @ g @ p_q.T)
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))

    return theta


def absolute_angle_bt_vectors(vec1, vec2, g):
    """Calculate the angle between two vectors using the dot product.

    Parameters
    ----------
    vec1, vec2, : array_like of shape (1,n) or (n,)
        The two vectors.

    g : nxn ndarray
        The metric tensor. If vectors are in real Cartesian
        coordinates, give the identitiy matrix.

    Returns
    -------
    theta : float
        Angle in degrees

    """

    p_q = np.array([vec1, vec2])

    [[pp, pq], [qp, qq]] = np.array(p_q @ g @ p_q.T)
    theta = np.degrees(np.arccos(pq/(pp**0.5 * qq**0.5)))
    return theta


def rotation_angle_bt_vectors(vec1, vec2, trans_mat=None):
    """Calculate the rotation angle from one vector to a second vector. This
        results in a rotation angle with sign giving the rotation direction
        assuming a right-handed system. Note that the order of the vectors
        matters. For rotation in image coordinates, sign of the result should
        be flipped due to the fact that "y" coordinates increases going "down"
        the image.

    Parameters
    ----------
    vec1, vec2, : array_like of shape (1,2) or (2,)
        The two vectors.

    trans_mat : 2x2 ndarray
        The transformation matrix from the vector coordinates to
        Cartesian coordinates, if not already in a Cartesian system. If equal
        to None, a Cartesian system is assumed.
        Default: None

    Returns
    -------
    theta : float
        Rotation angle in degrees.

    """

    if trans_mat is not None:
        vec1 = trans_mat @ vec1
        vec2 = trans_mat @ vec2

    theta = np.degrees(np.arctan2(vec2[1], vec2[0])
                       - np.arctan2(vec1[1], vec1[0]))

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


def IntPlAng(hkl_1, hkl_2, g):
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

    theta = np.degrees(np.arccos(np.round(pq/(pp**0.5 * qq**0.5),
                                          decimals=10)))
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
    two_theta = np.degrees(2 * np.arcsin(wavelength / (2 * d_hkl)))
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
    m_e = 9.109e-31  # electron mass (kg)
    e = 1.602e-19  # elementary charge (C)
    c = 2.997e8  # speed of light (m/s)
    h = 6.626e-34  # Plank's constant (Nms)

    wavelength = h/(2*m_e*e*V*(1+e*V/(2*m_e*c**2)))**.5
    return wavelength


# %%
"""Directory and file functions"""


def select_folder(path=None):
    """Select a folder in dialog box and return path

    Parameters
    ----------
    path : str or None
        The path to the desired folder. This option is available so this
        function can be retained in a script when the user does not always
        want to deal with clicking in a dialog box. If the path is passed
        in the function call the dialog box will not open, but the path
        can easily be commented out to activate the interactive functionality.

    Returns
    -------
    path : str
        The path to the selected folder.

    """
    if not path:
        print('Select folder')
        path = qfd.getExistingDirectory()

    return path


# %%
"""General image functions"""


def import_image(directory=None, display_image=True, images_from_stack=None,
                 emd_velox=True):
    """Select image from 'Open File' dialog box, import and (optionally) plot

    Parameters
    ----------
    directory : str or None
        The location of the image to load or the path to the folder containing
        the desired image. If only a directory is given, the "Open file"
        dialog box will still open allowing you to select an image file.
    display_image : bool
        If True, plots image (or first image if a series is imported).
        Default: True
    images_from_stack : None or 'all' or int or list-like
        If file at path contains a stack of images, this argument controls
        importing some or all of the images.
            Default: None: import only the first image of the stack.
            'all' : import all images as a 3d numpy array.
        images
    emd_velox : bool
        Whether .emd files are of Velox format. If True, open .emd files
        assuming Velox format. Otherwise, open using Berkeley .emd reader.

    Returns
    -------
    image : ndarray
        The imported image
    metadata : dict
        The metadata available in the original file

    """

    if directory is None:
        path, _ = qfd.getOpenFileName(
            caption='Select an image to load...',
            filter="Images (*.png *.jpg *.tif *.dm4 *.dm3 *.emd *.ser)")
        print(f'path to imported image: {path}')

    elif directory[-4:] in ['.dm4', '.dm3', '.emd',
                            '.ser', '.tif', '.png', '.jpg']:
        path = directory
        print(f'path to imported image: {path}')

    else:
        path, _ = qfd.getOpenFileName(
            caption='Select an image to load...',
            directory=directory,
            filter="Images (*.png *.jpg *.tif *.dm4 *.dm3 *.emd *.ser)")

    if path[-3:] in ['dm4', 'dm3']:
        dm_file = dmReader(path)
        image = (dm_file['data'])  # .astype(np.float64)
        metadata = {key: val for key, val in dm_file.items() if key != 'data'}

    elif path[-3:] == 'emd':
        if emd_velox:
            emd_file = emdVeloxReader(path)
            image = emd_file['data']
            metadata = {key: val for key, val in emd_file.items()
                        if key != 'data'}
        else:
            emd_file = emdReader(path)
            image = emd_file['data']
            metadata = {key: val for key, val in emd_file.items()
                        if key != 'data'}

    elif path[-3:] == 'ser':
        ser_file = serReader(path)
        image = ser_file['data']
        metadata = {key: val for key, val in ser_file.items() if key != 'data'}

    else:
        image = imageio.volread(path)
        metadata = image.meta

    if images_from_stack is None and len(image.shape) == 3:
        image = image[0, :, :]
    elif images_from_stack == 'all':
        pass
    elif (type(images_from_stack) == list
          or type(images_from_stack) == int):
        image = image[images_from_stack, :, :]

    if len(image.shape) == 2:
        image = image_norm(image)
        image_ = image
    if len(image.shape) == 3:
        image = np.array([image_norm(im) for im in image])
        image_ = image[0, :, :]
    if display_image is True:
        fig, axs = plt.subplots()
        axs.imshow(image_, cmap='gray')
        axs.set_xticks([])
        axs.set_yticks([])

    return image, metadata


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


def write_image_array_to_tif(image, filename, folder=None, bits=16):
    """Save an ndarray as an 8 or 16 bit TIFF image.

    Parameters
    ----------
    image : ndarray
        Input image as an ndarray
    folder : str or None
        Directory in which to save the image. If None,
    filename : str
        The file name to use for the saved image.
    bits : int
        The number of bits to use for the saved .tif. Must be 8, 16.

    Returns
    -------
    None

    """

    image = image_norm(image)
    if folder is None:
        folder = qfd.getExistingDirectory()
    if (filename[-4] != '.'):
        filename += '.tif'

    if bits == 8:
        dtype = np.uint8
    elif bits == 16:
        dtype = np.uint16
    else:
        raise Exception('"bits" must be 8 or 16')

    imwrite(os.path.join(folder, filename),
            (image*(2**bits-1)).astype(dtype), photometric='minisblack')


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

    M = moments(image, order=1)
    mu = moments_central(image, order=2)

    try:
        [x0, y0] = [M[1, 0]/M[0, 0], M[0, 1]/M[0, 0]]
        [u20, u11, u02] = [mu[2, 0]/M[0, 0],
                           mu[1, 1]/M[0, 0],
                           mu[0, 2]/M[0, 0]]
        cov = np.array([[u20, u11],
                        [u11, u02]])

    except ZeroDivisionError as err:
        raise ZeroDivisionError(no_mask_error) from err

    try:
        eigvals, eigvects = np.linalg.eig(cov)
    except np.linalg.LinAlgError as err:
        raise ArithmeticError(no_mask_error) from err

    if eigvects[0, 0] < 0:
        eigvects[:, 0] *= -1
    if eigvects[0, 1] < 0:
        eigvects[:, 1] *= -1

    ind_sort = np.argsort(eigvals)
    eigvals = np.abs(np.take_along_axis(eigvals, ind_sort, 0))
    eigvects = np.take_along_axis(eigvects, np.array([ind_sort, ind_sort]), 0)
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
    theta = np.degrees(-np.arcsin(np.cross(np.array([1, 0]),
                                           eigvects[:, major])))
    if sig_2 > sig_1:
        print('sigmas flipped')
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

    ang = np.radians(-ang)  # negative due to inverted y axis in python
    I = I_o + A*np.exp(-1/2*(
        ((np.cos(ang) * (x - x0) + np.sin(ang) * (y - y0)) / sig_1)**2
        + ((-np.sin(ang) * (x - x0) + np.cos(ang) * (y - y0)) / sig_2)**2))

    return I


# def LoG_2d(x, y, x0, y0, sig, A=1, I_o=0):
#     """Sample a round, 2D Laplacian of Gaussian function.

#     Samples a 2D Laplacian of Gaussian function at an array of points.

#     Parameters
#     ----------
#     x, y : ndarrays, must have the same shape
#         They x and y coordinates of each sampling point. If given arrays
#         generated by numpy.mgrid or numpy.meshgrid, will return an image
#         of the Gaussian.
#     x0, y0 : center of the Gaussian
#     sig : variance of the
#     A : Peak amplitude of the Gaussian
#     I_o : Constant background value

#     Returns
#     -------
#     I : the value of the function at the specified points. Will have the same
#         shape as x, y inputs

#     """

#     I = I_o + (-A*np.exp(-((x - x0)**2 + (y - y0)**2) / (2*sig**2))
#                * (((x - x0)**2 + (y - y0)**2) / (sig**4) - 2/sig**2))

#     return I


def gaussian_ellip_ss(p0, x, y, z, masks=None):
    """Sum of squares for a Gaussian function.

    Takes a parameter vector, coordinates, and corresponding data values;
    returns the sum of squares of the residuals.

    Parameters
    ----------
    p0 : array_like with shape (n,7)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig_maj, sig_rat, ang, A, I_o]
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

    if p0.shape[0] > 7:
        # model = np.ones(z.shape) * p0[-1]
        I0 = p0[-1]
        p0_ = p0[:-1].reshape((-1, 6))
        x0, y0, sig_maj, sig_rat, ang, A = np.split(p0_, 6, axis=1)
    else:
        x0, y0, sig_maj, sig_rat, ang, A, I0 = p0

    sig_min = sig_maj/sig_rat

    # Sum the functions for each peak:
    model = np.sum(A*np.exp(-1/2*(((np.cos(ang) * (x - x0)
                                    + np.sin(ang) * (y - y0))
                                   / sig_maj)**2
                                  + ((-np.sin(ang) * (x - x0)
                                      + np.cos(ang) * (y - y0))
                                     / sig_min)**2)),
                   axis=0) + I0

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def gaussian_circ_ss(p0, x, y, z, masks=None):
    """Sum of squares for a Gaussian function.

    Takes a parameter vector, coordinates, and corresponding data values;
    returns the sum of squares of the residuals.

    Parameters
    ----------
    p0 : array_like with shape (n,7)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig_maj, sig_rat, ang, A, I_o]
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

    if p0.shape[0] > 5:
        I0 = p0[-1]
        p0_ = p0[:-1].reshape((-1, 4))
        x0, y0, sig, A = np.split(p0_, 4, axis=1)
    else:
        x0, y0, sig, A, I0 = p0

    # Sum the functions for each peak:
    model = np.sum(A*np.exp(-1/2*((((x - x0)) / sig)**2
                                  + (((y - y0)) / sig)**2)),
                   axis=0) + I0

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def fit_gaussian_ellip(data, p0, masks=None, method='BFGS', bounds=None):
    """Fit an elliptical 2D Gaussain function to data.

    Fits a 2D, elliptical Gaussian to an image. Intensity values equal to zero
    are ignored.

    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak
    p0 : array_like with shape (n, 7)
        Initial guess for the n-Gaussian parameter vector where each peak
        has 6 independent parameters (x0, y0, sig_maj, sig_ratio, ang, A) the
        whole region has a constant background (I_0).

    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the
        mask for the corresponding peak and 0 elsewhere.
    method : str, the minimization solver name
        Supported solvers are: ''BFGS', 'L-BFGS-B', 'Powell', 'trust-constr'.
        Default: 'BFGS'
    bounds : list of two-tuples of length 7*n or None
        The bounds for Gaussian fitting parameters. Only works with methods
        that accept bounds (e.g. 'L-BFGS-B', but not 'BFGS'). Otherwise must
        be set to None.
        Each two-tuple is in the form: (upper, lower).
        Order of bounds must be: [x0, y0, sig_maj, sig_ratio, ang, A, I_0] * n
        Default: None

    Returns
    -------
    params : 1D array
        The best fit parameter vector found by least squares

    """

    num_gauss = int(np.ceil(p0.shape[0]/7))
    img_shape = data.shape

    I0 = p0[-1]
    p0_ = p0[:-1].reshape((num_gauss, 6))

    p0_[:, 4] *= -1

    y, x = np.indices(img_shape)
    z = data.flatten()

    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

    x0y0 = p0_[:, :2]
    if masks is None:
        image = np.zeros(img_shape)
        image[y, x] = z
        masks_labeled, _ = ndimage.label(image)

    elif (type(masks) == imageio.core.util.Array
          or type(masks) == np.ndarray):
        masks_labeled, _ = ndimage.label(masks)

    masks_to_peaks = ndimage.map_coordinates(masks_labeled,
                                             np.flipud(x0y0.T),
                                             order=0).astype(int)
    masks_labeled = np.take(masks_labeled, unmasked_data).flatten()

    masks_labeled = np.array([np.where(masks_labeled == mask_num, 1, 0)
                              for i, mask_num in enumerate(masks_to_peaks)])

    p0_ = np.append(p0_.flatten(), I0)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', category=UserWarning,
    #                             lineno=182)
    #     warnings.filterwarnings('ignore', category=RuntimeWarning,
    #                             lineno=579)
    params = minimize(gaussian_ellip_ss, p0_,
                      args=(x, y, z, masks_labeled),
                      bounds=bounds, method=method).x

    params = np.concatenate((params[:-1].reshape((-1, 6)),
                             np.ones((num_gauss, 1))*params[-1]),
                            axis=1)

    params[:, 4] *= -1
    params[:, 4] = ((params[:, 4] + np.pi/2) % np.pi) - np.pi/2

    return params


def fit_gaussian_circ(data, p0, masks=None, method='BFGS', bounds=None):
    """Fit an elliptical 2D Gaussain function to data.

    Fits a 2D, elliptical Gaussian to an image. Intensity values equal to zero
    are ignored.

    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak
    p0 : array_like with shape (n, 5)
        Initial guess for the n-Gaussian parameter vector where each peak
        has 4 independent parameters (x0, y0, sig, A) the
        whole region has a constant background (I_0).
    masks : 2d array_like of size (n, m)
        n = number of peaks to fit
        m = number of unmasked pixels
        The flattened masks for each peak. Each of the "n" rows is 1 where the
        mask for the corresponding peak and 0 elsewhere.
    method : str, the minimization solver name
        Supported solvers are: 'BFGS', 'L-BFGS-B', 'Powell', 'trust-constr'.
        Default: 'BFGS'
    bounds : list of two-tuples of length 7*n or None
        The bounds for Gaussian fitting parameters. Only works with methods
        that accept bounds (e.g. 'L-BFGS-B', but not 'BFGS'). Otherwise must
        be set to None.
        Each two-tuple is in the form: (upper, lower).
        Order of bounds must be: [x0, y0, sig, A, I_0] * n
        Default: None

    Returns
    -------
    params : 1D array
        The best fit parameter vector found by least squares

    """

    num_gauss = int(np.ceil(p0.shape[0]/5))
    img_shape = data.shape
    p0.shape
    I0 = p0[-1]
    p0_ = p0[:-1].reshape((num_gauss, 4))

    y, x = np.indices(img_shape)
    z = data.flatten()

    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

    x0y0 = p0_[:, :2]
    if masks is None:
        image = np.zeros(img_shape)
        image[y, x] = z
        masks_labeled, _ = ndimage.label(image)

    elif (type(masks) == imageio.core.util.Array
          or type(masks) == np.ndarray):
        masks_labeled, _ = ndimage.label(masks)

    masks_to_peaks = ndimage.map_coordinates(masks_labeled,
                                             np.flipud(x0y0.T),
                                             order=0).astype(int)
    masks_labeled = np.take(masks_labeled, unmasked_data).flatten()

    masks_labeled = np.array([np.where(masks_labeled == mask_num, 1, 0)
                              for i, mask_num in enumerate(masks_to_peaks)])

    p0_ = np.append(p0_.flatten(), I0)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', category=UserWarning,
    #                             lineno=182)
    #     warnings.filterwarnings('ignore', category=RuntimeWarning,
    #                             lineno=579)
    params = minimize(gaussian_circ_ss, p0_,
                      args=(x, y, z, masks_labeled),
                      bounds=bounds, method=method).x

    params = np.concatenate((params[:-1].reshape((-1, 4)),
                             np.ones((num_gauss, 1))*params[-1]),
                            axis=1)

    # Add dummy columns for sigma ratio and rotation angle:
    params = np.insert(params, 3, np.array([1, 0]*num_gauss).reshape(-1, 2).T,
                       axis=1)

    return params


def pcf_radial(dr, coords, total_area=None):
    """Calculate a radial pair correlation function from 2D or 3D data.

    Parameters
    ----------
    dr : int or float
        The step size for binning distances
    coords : array_like with shape (n, d)
        The x, y, z coordinates of each point. n is the number of points,
        d is the number of dimensions (i.e. 2 or 3)

    total_area : area or volume of region containing the points. Estimate
        made if not given.

    Returns
    -------
    pcf : 1D array
        Histogram of point-point distances with bin size dr

    """

    if total_area is None:
        diag_vect = np.max(coords, axis=0) - np.min(coords, axis=0)
        total_area = diag_vect @ np.ones(diag_vect.shape)

    n = coords.shape[0]
    rho = n / total_area

    vects = np.array([coords - i for i in coords])

    dist = np.hypot(vects[:, :, 0], vects[:, :, 1])
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
        Default: 4
    thresh : int or float
        The minimum image value that should be considered a peak. Used to
        remove low intensity background noise peaks.
        Default: 0

    Returns
    -------
    peaks : 2D array_like with shape: image.shape
        Array with 1 indicating peak pixels and 0 elsewhere.

    """

    kern_rad = int(np.floor(min_dist/2) * 0.75)
    if min_dist < 1:
        min_dist = 1
    size = 2*kern_rad + 1
    neighborhood = np.array([1 if np.hypot(i-kern_rad, j-kern_rad)
                             <= kern_rad
                             else 0
                             for j in range(size) for i in range(size)]
                            ).reshape((size, size))

    peaks = (maximum_filter(image, footprint=neighborhood) == image
             ) * (image > thresh)
    return peaks.astype(int)


def watershed_segment(image, sigma=None, buffer=0, local_thresh_factor=0.95,
                      watershed_line=True, min_dist=4):
    """Segment an image using the Watershed algorithm.

    Parameters
    ----------
    image : 2D array_like
        The image to be segmented.
    sigma : int or float
        The Laplacian of Gaussian sigma value to use for peak sharpening. If
        None, no filtering is applied.
        Default: None
    buffer : int
        The border within which peaks are ignored.
        Default: 0
    local_thresh_factor : float
        Removes background from each segmented region by thresholding.
        Threshold value determined by finding the maximum value of edge pixels
        in the segmented region and multipling this value by the
        local_thresh_factor value. The filtered image is used for this
        calculation.
        Default 0.95.
    watershed_line : bool
        Seperate segmented regions by one pixel.
        Default: True.
    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
        Default: 4.

    Returns
    -------
    masks : 2D array with same shape as image
    num_masks : int
        The number of masks
    slices : List of image slices which contain each region
    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer

    """

    img_der = copy.deepcopy(image)
    [h, w] = image.shape

    if type(sigma) in (int, float, tuple):
        img_der = image_norm(-gaussian_laplace(img_der, sigma))

    local_max, _ = ndimage.label(detect_peaks(img_der, min_dist=min_dist))

    masks = watershed(-img_der, local_max, watershed_line=watershed_line)
    slices = ndimage.find_objects(masks)
    num_masks = int(np.max(masks))

    """Refine masks with local_thresh_factor"""
    if local_thresh_factor > 0:
        masks_ref = np.zeros(image.shape)

        for i in range(num_masks):
            mask_sl = np.where(masks[slices[i][0], slices[i][1]] == i+1, 1, 0)
            img_der_sl = img_der[slices[i][0], slices[i][1]]
            edge = mask_sl - ndimage.morphology.binary_erosion(mask_sl)
            thresh = np.max(edge*img_der_sl) * local_thresh_factor
            mask_sl = np.where(mask_sl*img_der_sl > thresh, 1, 0)
            masks_ref[slices[i][0], slices[i][1]] += mask_sl * (i+1)

        masks = masks_ref

    _, peak_xy = np.unique(local_max, return_index=True)
    peak_xy = np.fliplr(np.array(np.unravel_index(peak_xy,
                                                  local_max.shape)).T[1:, :])

    peaks = pd.DataFrame.from_dict(
        {'x': list(peak_xy[:, 0]),
         'y': list(peak_xy[:, 1]),
         'label': [i+1 for i in range(num_masks)]})

    peaks = peaks[((peaks.x >= buffer) &
                   (peaks.x <= w - buffer) &
                   (peaks.y >= buffer) &
                   (peaks.y <= h - buffer))]
    peaks = peaks.reset_index(drop=True)

    return masks, num_masks, slices, peaks


def band_pass_filter(shape, high_pass=5, low_pass=None,
                     filter_edge_smoothing=0):
    """Create a high and/or low pass filter.

    Parameters
    ----------
    shape : two-tuple
        The image shape for which to make the mask.
    high_pass : int or None
        The number of reciprocal pixels below which to block with the high
        pass filter.
        Default: 5
    low_pass : int or None
        The number of reciprocal pixels above which to block with the low pass
        filter.
        Default: None
    filter_edge_smoothing : int
        Gaussian blur sigma used to smooth the hard edge of the band pass
        filter in reciprocal pixels.
        Default 0.

    Returns
    -------
    masks : 2D array with same shape as image
    num_masks : int
        The number of masks
    slices : List of image slices which contain each region
    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer

    """

    f_freq_1d = [np.fft.fftfreq(shape[0], 1),
                 np.fft.fftfreq(shape[1], 1)]
    max_dim = np.argmax(shape)
    fft_pixel_size = np.array([f_freq_1d[0][1] - f_freq_1d[0][0],
                               f_freq_1d[1][1] - f_freq_1d[1][0]])

    if not high_pass:
        high_pass = 0
    else:
        high_pass *= fft_pixel_size[max_dim]
    if not low_pass:
        low_pass = np.inf
    else:
        low_pass *= fft_pixel_size[max_dim]

    fft_freq_abs = np.linalg.norm(np.array(
        np.meshgrid(f_freq_1d[1], f_freq_1d[0])), axis=0)

    mask = np.where(((fft_freq_abs >= high_pass) &
                     (fft_freq_abs <= low_pass)), 1, 0).astype(float)

    if filter_edge_smoothing:
        mask = image_norm(ndimage.filters.gaussian_filter(
            mask,
            sigma=filter_edge_smoothing,
            truncate=4*filter_edge_smoothing))

    return mask


def get_phase_from_com(com_xy, theta, flip=True, high_low_filter=False,
                       filter_params={'beam_energy': 200e3,
                                      'conv_semi_angle': 18,
                                      'pixel_size': 0.01,
                                      'high_pass': 0.05,
                                      'low_pass': 0.85,
                                      'edge_smoothing': 0.01}):
    """Reconstruct phase from center of mass shift components.

    *** Citations...

    Parameters
    ----------
    com_xy : ndarray of shape (h,w,2)
        The center of mass shift component images as a stack.
    theta : float or int
        Rotation angle in degrees between real and reciprocal space.
    flip : bool
        Whether to transpose x and y axes.
        Default: False
    high_low_filter : bool
        Whether to perform high and/or low pass filtering as defined in the
        filter_params argument.
        Default: False
    filter_params : dict
        Dictionary of parameters used for calculating the high and/or low pass
        filters:
            {'beam_energy' : 200e3,    # electron-volts
             'conv_semi_angle' : 18,   # mrads
             'pixel_size' : 0.01,      # nm
             'high_pass' : 0.05,       # fraction of aperture passband
             'low_pass' : 0.85,        # fraction of aperture passband
             'edge_smoothing' : 0.01}  # fraction of aperture passband

    Returns
    -------
    phase : 2D array with shape (h,w)
        The reconstructed phase of the sample transmission function.

    """

    f_freq_1d_y = np.fft.fftfreq(com_xy.shape[1], 1)
    f_freq_1d_x = np.fft.fftfreq(com_xy.shape[2], 1)
    h, w = com_xy.shape[1:]

    if high_low_filter:
        lambda_ = elec_wavelength(filter_params['beam_energy']) * 1e9  # in nm
        h, w = com_xy.shape[1:]
        min_dim = np.argmin((h, w))
        fft_pixelSize = 1 / (np.array(com_xy.shape[1:])
                             * filter_params['pixel_size'])  # nm^-1

        apeture_cutoff = 2*np.sin(2 * filter_params['conv_semi_angle']  # nm^-1
                                  )/lambda_
        low_pass = (filter_params['low_pass']*apeture_cutoff
                    / fft_pixelSize[min_dim])
        high_pass = (filter_params['high_pass']*apeture_cutoff
                     / fft_pixelSize[min_dim])  # pixels

        sigma = (filter_params['edge_smoothing'] * apeture_cutoff
                 / fft_pixelSize[min_dim])

        mask = band_pass_filter((h, w), high_pass=high_pass,
                                low_pass=low_pass,
                                filter_edge_smoothing=sigma)

    else:
        mask = np.ones((h, w))

    theta = np.radians(theta)
    if not flip:
        CoMx_rot = com_xy[0]*np.cos(theta) - com_xy[1]*np.sin(theta)
        CoMy_rot = com_xy[0]*np.sin(theta) + com_xy[1]*np.cos(theta)
    if flip:
        CoMx_rot = com_xy[0]*np.cos(theta) + com_xy[1]*np.sin(theta)
        CoMy_rot = com_xy[0]*np.sin(theta) - com_xy[1]*np.cos(theta)

    com_ = np.zeros(com_xy.shape)
    com_[0] = CoMx_rot
    com_[1] = CoMy_rot

    k_p = np.array(np.meshgrid(f_freq_1d_x, f_freq_1d_y))

    _ = np.seterr(divide='ignore')
    denominator = 1/(1j*(k_p[0] + 1j*k_p[1]))
    denominator[0, 0] = 0
    _ = np.seterr(divide='warn')

    complex_image = np.fft.ifft2((mask * (np.fft.fft2(com_[1])
                                          + 1j*np.fft.fft2(com_[0])))
                                 * denominator)

    phase = (np.real(complex_image)**2 + np.imag(complex_image)**2)**0.5

    return phase


def fast_rotate_90deg(image, angle):
    """Rotate images by multiples of 90 degrees. Faster than
    scipy.ndimage.rotate().

    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.
    angle : float or int
        Rotation angle in degrees. Must be a multiple of 90.

    Returns
    -------
    rotated_image : 2D array
        The image rotated by the specified angle.

    """

    angle = angle % 360
    if angle == 90:
        image_ = np.flipud(image.T)
    elif angle == 180:
        image_ = np.flipud(np.fliplr(image))
    elif angle == 270:
        image_ = np.fliplr(image.T)
    elif angle == 0:
        image_ = image
    else:
        raise Exception('Argument "angle" must be a multiple of 90 degrees')
    return image_


def fft_amplitude_area(image, xy_fft, r, blur, thresh=0.5,
                       fill_holes=True, buffer=10):
    """Create mask based on Bragg spot filtering (via FFT) of image.

    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.
    xy_fft : ndarray
        The Bragg spot coordinates in the FFT. Must be shape (n,2).
    r : int, float or list-like of ints or floats
        The radius (or radii) of Bragg spot pass filters. If a sclar, the same
        radius is applied to all Bragg spots. If list-like, must be of shape
        (n,).
    blur : int or float
        The gaussian sigma used to blur the amplitude image.
    thresh : float
        The relative threshold for creating the mask from the blured amplitude
        image.
        Default: 0.5
    fill_holes : bool
        If true, interior holes in the mask are filled.
        Default: True
    buffer : int
        Number of pixels to erode from the mask after binarization. When used
        as a mask for restricting atom column detection, this prevents
        searching too close to the edge of the area.
        Default: 10

    Returns
    -------
    mask : 2D array
        The final amplitude mask.

    """

    if not (type(r) == int) | (type(r) == float):
        if xy_fft.shape[0] != r.shape[0]:
            raise Exception("If'r' is not an int or float, its length "
                            "must match the first dimension of xy_fft.")

    fft = np.fft.fftshift(np.fft.fft2(image))
    mask = np.zeros(fft.shape)
    xy = np.mgrid[:mask.shape[0], : mask.shape[1]]
    xy = np.array([xy[1], xy[0]]).transpose((1, 2, 0))
    if (type(r) == int) | (type(r) == float):
        for xy_ in xy_fft:
            mask += np.where(np.linalg.norm(xy - xy_, axis=2) <= r, 1, 0)

    else:
        for i, xy_ in enumerate(xy_fft):
            mask += np.where(np.linalg.norm(xy - xy_, axis=2) <= r[i], 1, 0)

    image_complex = np.fft.ifft2(np.fft.fftshift(fft * mask))
    amplitude = (np.real(np.real(image_complex))**2
                 + np.imag(np.real(image_complex))**2)**0.5
    amplitude = image_norm(gaussian_filter(amplitude, sigma=blur))
    mask = np.where(amplitude > thresh, 1, 0)
    if fill_holes:
        mask = ndimage.morphology.binary_fill_holes(mask)
    mask = np.where(mask, 1, 0)
    mask = ndimage.morphology.binary_dilation(mask, iterations=buffer)

    return mask


def std_local(image, r):
    """Get local standard deviation of an image
    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.
    r : int
        Kernel radius. STD is calculated in a square kernel of size 2*r + 1.

    Returns
    -------
    std : 2D array
        The image local standard deviation of the image.
    """

    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*r+1, 2*r+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")
    var = (s2/ns - (s/ns)**2)
    var = np.where(var < 0, 0, var)

    return var ** 0.5


def binary_find_largest_rectangle(array):
    """Gets the slice object of the largest rectangle of 1s in a 2D binary
    array. Modified version. Original by Andrew G. Clark

    Parameters
    ----------
    array : ndarray of shape (h,w)
        The binary image.

    Returns
    -------
    xlim : list-like of length 2
        The x limits (columns) of the largest rectangle.
    ylim : list-like of length 2
        The y limits (columns) of the largest rectangle.
    sl : numpy slice object
        The slice object which crops the image to the largest rectangle.


    The MIT License (MIT)

    Copyright (c) 2020 Andrew G. Clark

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE."""

    # first get the sums of successive vertical pixels
    vert_sums = (np.zeros_like(array)).astype('float')
    vert_sums[0] = array[0]
    for i in range(1, len(array)):
        vert_sums[i] = (vert_sums[i-1] + array[i]) * array[i]

    # declare some variables for keeping track of the largest rectangle
    max_area = -1
    pos_at_max_area = (0, 0)
    height_at_max_area = -1
    x_end = 0

    # go through each row of vertical sums and find the largest rectangle
    for i in range(len(vert_sums)):
        positions = []  # a stack
        heights = []  # a stack
        for j in range(len(vert_sums[i])):
            h = vert_sums[i][j]
            if len(positions) == 0 or h > heights[-1]:
                heights.append(h)
                positions.append(j)
            elif h < heights[-1]:
                while len(heights) > 0 and h < heights[-1]:
                    h_tmp = heights.pop(-1)
                    pos_tmp = positions.pop(-1)
                    area_tmp = h_tmp * (j - pos_tmp)
                    if area_tmp > max_area:
                        max_area = area_tmp
                        # this is the bottom left
                        pos_at_max_area = (pos_tmp, i)
                        height_at_max_area = h_tmp
                        x_end = j
                heights.append(h)
                positions.append(pos_tmp)
        while len(heights) > 0:
            h_tmp = heights.pop(-1)
            pos_tmp = positions.pop(-1)
            area_tmp = h_tmp * (j - pos_tmp)
            if area_tmp > max_area:
                max_area = area_tmp
                pos_at_max_area = (pos_tmp, i)  # this is the bottom left
                height_at_max_area = h_tmp
                x_end = j

    top_left = (int(pos_at_max_area[0]), int(pos_at_max_area[1]
                                             - height_at_max_area) + 1)
    width = int(x_end - pos_at_max_area[0])
    height = int(height_at_max_area - 1)
    xlim = [top_left[0], top_left[0] + width]
    ylim = [top_left[1], top_left[1] + height]
    sl = np.s_[ylim[0]:ylim[1], xlim[0]:xlim[1]]

    return xlim, ylim, sl


def fft_equxy(image, hanning_window=False):
    """Gets FFT with equal x & y pixel sizes
    image : 2D array
        The image.
    hanning_window : bool
        Whether to apply a hanning window to the image before taking the FFT.
        Default: False

    """

    h, w = image.shape
    m = (min(h, w) // 2) * 2
    U = int(m/2)
    image_square = image[int(h/2)-U: int(h/2)+U,
                         int(w/2)-U: int(w/2)+U]
    if hanning_window:
        hann = np.outer(np.hanning(m), np.hanning(m))
        image_square *= hann
    fft = np.fft.fft2(image_square)  # *hann)
    fft = (abs(np.fft.fftshift(fft)))
    fft = image_norm(fft)

    return fft
