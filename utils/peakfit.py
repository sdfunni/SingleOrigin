"""Module containing peak finding and measuring utility functions."""

from copy import deepcopy
import warnings

import numpy as np

import pandas as pd

from scipy.ndimage import (
    label,
    find_objects,
    gaussian_filter,
    gaussian_laplace,
    maximum_filter,
    center_of_mass,
    map_coordinates,
)
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
from skimage.morphology import dilation

import skimage
from skimage.segmentation import watershed
from skimage.feature import hessian_matrix_det
from skimage.morphology import binary_erosion

from SingleOrigin.utils.image import image_norm, get_circular_kernel
from SingleOrigin.utils.mathfn import (
    line,
    plane_2d,
    gaussian_1d,
    gaussian_2d
)


# %%


"""Error message string(s)"""
no_mask_error = (
    "Float division by zero during moment estimation. No image intensity "
    + "columns, this means that no pixel region was found for fitting of at "
    + "least one atom column. This situation may result from: \n"
    + "1) Too high a 'bkgd_thresh_factor' value resulting in no pixels "
    + "remaining for some atom columns. Check 'self.fit_masks' to see if some "
    + "atom columns do not have mask regions. \n "
    + "2) Due to mask regions splitting an atom column as a result of too "
    + "small or too large a value used for 'grouping_filter' or 'diff_filter'."
    + " Check 'self.group_masks' and 'self.fit_masks' to see if this may be "
    + "occuring. Too small a 'fitting_filter' value may cause noise peaks to "
    + "be detected, splitting an atom column. Too large a value for either "
    + "filter may cause low intensity peaks to be ignored by the watershed "
    + "algorithm. Masks may  be checked with the 'show_masks()' method. \n"
    + "3) Reference lattice may extend to regions of the image without "
    + "detectable atom columns. Use or alter an roi_mask to restrict the "
    + "reference lattice to avoid these parts of the image."
)

# %%


def gaussian_1d_ss(p0, x, y):
    """Sum of squares for a Gaussian function.

    Parameters
    ----------
    p0 : array_like with shape (3,)
        Gaussian parameter vector [x0, sigma, A].

    x : 1d array
        x-coordinates of the data.

    y : 1d array
        y-coordinates of the data.

    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals

    """

    x0, sig, A = p0
    # Sum the functions for each peak:
    model = gaussian_1d(x, x0, sig, A)

    # Subtract from data to get residuals:
    R = y - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def fit_gaussian_1d(
        x, y,
        p0,
        method='BFGS',
        bounds=None
):
    """Fit a 1D Gaussain function to data.

    Fits a 1D Gaussian to an image. Intensity values equal to zero are ignored.

    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak

    p0 : array_like with shape (3,)
        Initial guess for the Gaussian parameter vector (x0, sigma, A).

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

    unmasked_data = np.nonzero(y)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

    params = minimize(
        gaussian_1d_ss,
        p0,
        args=(x, y),
        bounds=bounds,
        method=method
    ).x

    return params


def get_feature_size(image):
    """
    Get nominal feature size in the image using automatic scale selection.

    Finds feature size for an image based on the highest maximum of the
    determinant of the Hessian (as applied to the central 1024x1024 region
    of the image if larger than 1k). Returns half-width of the determined
    feature size.

    Parameters
    ----------
    image : 2d array
        The image.

    Returns
    -------
    sigma : scalar
        The nominal feature size half-width.

    """

    h, w = image.shape
    if h * w > 1024**2:
        crop_factor = 1024/np.sqrt(h*w)
        crop_h = int(h * crop_factor / 2)
        crop_w = int(w * crop_factor / 2)

        image = image[int(h/2)-crop_h:int(h/2)+crop_h,
                      int(w/2)-crop_w: int(w/2)+crop_w]

    min_scale = 2
    max_scale = 30
    scale_step = 1

    scale = np.arange(min_scale, max_scale, scale_step)

    trim = int(np.min(image.shape) * 0.1)
    hess_max = np.array([
        np.max(hessian_matrix_det(image, sigma=i)[trim:-trim, trim:-trim])
        for i in scale
    ])

    spl = make_interp_spline(scale, hess_max, k=2)
    scale_interp = np.linspace(min_scale, max_scale, 1000)
    hess_max_interp = spl(scale_interp).T
    scale_max = scale_interp[np.argmax(hess_max_interp)]

    sigma = scale_max/2

    return sigma


def detect_peaks(
        image,
        min_dist=4,
        thresh=0,
        return_DataFrame=False,
):
    """
    Detect peaks in an image using a maximum filter with a minimum separation
    distance and threshold.

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

    return_DataFrame : bool
        Whether to return the peaks DataFrame in addition to the peak_map.

    Returns
    -------
    peak_map : 2D array_like with shape: image.shape
        Array with 1 indicating peak pixels and 0 elsewhere.

    peaks : DataFrame (optional)
        DataFrame containing the peak coordinates, maximum intensities and
        corresponding label in peak_map.

    """
    if min_dist < 1:
        min_dist = 1
    kern_rad = int(np.floor(min_dist))

    neighborhood = get_circular_kernel(kern_rad)

    # size = 2*kern_rad + 1
    # neighborhood = np.array(
    #     [1 if np.hypot(i - kern_rad, j - kern_rad) <= min_dist
    #      else 0
    #      for j in range(size) for i in range(size)]
    # ).reshape((size, size))

    peak_map = np.where(
        maximum_filter(image, footprint=neighborhood) == image, 1, 0
    ) * (image > thresh)

    if return_DataFrame:
        peak_map_labeled, num_peaks = label(peak_map)
        peak_xy = np.around(np.fliplr(np.array(
            center_of_mass(
                peak_map,
                peak_map_labeled,
                np.arange(1, num_peaks+1)
            )
        ))).astype(int)
        peaks = pd.DataFrame.from_dict({
            'x': list(peak_xy[:, 0]),
            'y': list(peak_xy[:, 1]),
            'max': np.array(image[peak_xy[:, 1], peak_xy[:, 0]]).astype(float),
            'label': [i+1 for i in range(peak_xy.shape[0])]
        })

        return peak_map.astype(int), peaks

    else:
        return peak_map.astype(int)


def watershed_segment(
        image,
        roi=None,
        sigma=None,
        filter_type='log',
        buffer=0,
        bkgd_thresh_factor=0.95,
        peak_bkgd_thresh_factor=0,
        watershed_line=True,
        min_dist=10,
        min_pixels=9,
        mean_std_thresh_factor=None,
):
    """
    Segment an image using the Watershed algorithm.

    Parameters
    ----------
    image : 2D array_like
        The image to be segmented.

    roi : 2D array_like
        Binary mask with 1 indicating the region where peaks should be found,
        0 elsewhere. Must be the same shape as image. "watershed_line" must be
        True or it will be set to True in the function.

    sigma : scalar or None
        The Gaussian or Laplacian of Gaussian sigma value to use for peak
        sharpening/smoothing. If None, no filtering is applied.
        Default: None

    filter_type : str
        'gauss' or 'log'. Whether to applly a Gaussian or a Laplacian of
        Gaussian filter to the image before running watershed segmentation.
        Default: 'log'

    buffer : int
        The border within which peaks are ignored.
        Default: 0

    bkgd_thresh_factor : float
        Removes background from each segmented region by thresholding.
        Threshold value determined by finding the maximum value of edge pixels
        in the segmented region and multipling this value by the
        bkgd_thresh_factor value. The filtered image is used for this
        calculation.
        Default 0.95.

    peak_bkgd_thresh_factor : float
        Alternative local thresholding method that thresholds between the
        watershed region edge maximum and the peak maximum. It is more
        appropriate if center-of-mass measurements will be made on the
        segmentation regions. Only active if bkgd_thresh_factor <= 0.
        Default: 0.

    mean_std_thresh_factor : float or None
        Thresholds at the mean value of the peak region plus this number of
        standard deviations of the values in the peak region. Negative values
        are valid if wanting to threshold below the mean. Has worked well for
        EWPC peaks using 0.5. Only active if bkgd_thresh_factor and
        peak_bkgd_thresh_factor are < 0.
        Default: None

    watershed_line : bool
        Seperate segmented regions by a 1 pixel wide line of zero-value pixels.
        Default: True.

    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
        Default: 10.

    min_pixels : int
        The minimum number of pixels allowed in a mask region. If less than
        this value, the mask and associated peak are discarded.
        Default: 9.

    Returns
    -------
    masks : 2D array with same shape as image

    num_masks : int
        The number of masks

    slices : List of image slices which contain each region in mask label
        order.

    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer / roi.

    """

    img_der = deepcopy(image)
    [h, w] = image.shape

    if sigma is not None:
        if filter_type == 'log':
            img_der = image_norm(-gaussian_laplace(img_der,
                                                   sigma, truncate=2))
        if filter_type == 'gauss':
            img_der = image_norm(
                gaussian_filter(img_der, sigma, truncate=2))

    peak_map, peaks = detect_peaks(
        img_der,
        min_dist=min_dist,
        return_DataFrame=True
    )

    local_max, n_peaks = label(peak_map)

    masks = watershed(-img_der, local_max, watershed_line=watershed_line)

    if buffer > 0:
        buffer_mask = np.zeros(image.shape)
        buffer_mask[buffer:-buffer, buffer:-buffer] = 1
        local_max = local_max * buffer_mask

    if roi is not None:
        local_max = local_max * roi

    # Remove masks outside the roi & relabel
    masks_in_roi = np.unique(local_max)[1:]

    masks_ = np.zeros(masks.shape)

    for i, lab in enumerate(masks_in_roi):
        masks_[masks == lab] = i+1

    masks = masks_.astype(int)

    # Remove peaks outside the roi
    peaks = peaks[np.isin(peaks.label.to_numpy(), masks_in_roi)]

    # Get the new labels for peaks
    labels = map_coordinates(
        masks,
        np.flipud(peaks.loc[:, 'x':'y'].to_numpy().T),
        order=0,
    ).astype(int)
    peaks.label = labels

    slices = find_objects(masks)
    num_masks = int(np.max(masks))

    # Refine masks with an optional thresholding
    if bkgd_thresh_factor > 0:
        masks_ref = np.zeros(image.shape)

        for i in range(0, num_masks):
            mask_sl = np.where(masks[slices[i][0], slices[i][1]] == i+1, 1, 0)
            img_der_sl = img_der[slices[i][0], slices[i][1]]

            edge = mask_sl - binary_erosion(mask_sl)
            thresh = np.max(edge * img_der_sl) * (bkgd_thresh_factor)
            mask_sl = np.where(mask_sl*img_der_sl >= thresh, 1, 0)

            # Check for and remove residual areas outside thresholded peak
            mask_labeled, n_areas = label(mask_sl, structure=np.ones((3, 3)))
            if n_areas > 1:
                peak_max = np.max(img_der_sl)
                peak_label = mask_labeled[img_der_sl == peak_max]
                mask_sl = np.where(mask_labeled == peak_label, 1, 0)

            # Check for minimum number of pixels to reasonably be a peak
            n_pixels = np.count_nonzero(mask_sl)
            if n_pixels >= min_pixels:
                masks_ref[slices[i][0], slices[i][1]] += mask_sl * (i+1)

        masks = masks_ref

    elif peak_bkgd_thresh_factor > 0:
        masks_ref = np.zeros(image.shape)

        for i in range(0, num_masks):
            mask_sl = np.where(masks[slices[i][0], slices[i][1]] == i+1, 1, 0)
            img_der_sl = img_der[slices[i][0], slices[i][1]]
            edge = mask_sl - binary_erosion(mask_sl)
            edge_max = np.max(edge * img_der_sl)
            peak_max = np.max(img_der_sl)
            thresh = peak_bkgd_thresh_factor * (peak_max - edge_max) + edge_max
            mask_sl = np.where(mask_sl*img_der_sl >= thresh, 1, 0)

            # Check for and remove resitual areas outside thresholded peak
            mask_labeled, n_areas = label(mask_sl, structure=np.ones((3, 3)))
            if n_areas > 1:
                peak_label = mask_labeled[img_der_sl == peak_max]
                mask_sl = np.where(mask_labeled == peak_label, 1, 0)

            # Check for minimum number of pixels to reasonably be a peak
            n_pixels = np.count_nonzero(mask_sl)
            if n_pixels >= min_pixels:
                masks_ref[slices[i][0], slices[i][1]] += mask_sl * (i+1)

        masks = masks_ref.astype(int)

        # Remove peaks discarded by thresholding
        masks_remaining = np.unique(masks)[1:]
        peaks = peaks[np.isin(peaks.label.to_numpy(), masks_remaining)]
        peaks.reset_index(inplace=True, drop=True)

        # Get the new labels for peaks
        labels = map_coordinates(
            masks,
            np.flipud(peaks.loc[:, 'x':'y'].to_numpy().T),
            order=0,
        ).astype(int)
        peaks.label = labels

        slices = find_objects(masks)
        num_masks = int(np.max(masks))

        peaks.sort_values(by='label', ignore_index=True, inplace=True)

    elif mean_std_thresh_factor is not None:
        masks_ref = np.zeros(image.shape)

        for i in range(0, num_masks):
            mask_sl = np.where(masks[slices[i][0], slices[i][1]] == i+1, 1, 0)
            img_der_sl = img_der[slices[i][0], slices[i][1]]
            # edge = mask_sl - binary_erosion(mask_sl)
            # edge_min = np.min(edge * img_der_sl)
            vals = img_der_sl[mask_sl == 1]
            # print(vals)
            std = np.std(vals)
            avg = np.mean(vals)

            peak_max = np.max(img_der_sl)
            thresh = mean_std_thresh_factor * std + avg

            mask_sl = np.where(mask_sl*img_der_sl >= thresh, 1, 0)

            # Check for and remove resitual areas outside thresholded peak
            mask_labeled, n_areas = label(mask_sl, structure=np.ones((3, 3)))
            if n_areas > 1:
                peak_label = mask_labeled[img_der_sl == peak_max]
                mask_sl = np.where(mask_labeled == peak_label, 1, 0)

            # Check for minimum number of pixels to reasonably be a peak
            n_pixels = np.count_nonzero(mask_sl)
            if n_pixels >= min_pixels:
                masks_ref[slices[i][0], slices[i][1]] += mask_sl * (i+1)

        masks = masks_ref.astype(int)

        # Remove peaks discarded by thresholding
        masks_remaining = np.unique(masks)[1:]
        peaks = peaks[np.isin(peaks.label.to_numpy(), masks_remaining)]
        peaks.reset_index(inplace=True, drop=True)

        # Get the new labels for peaks
        labels = map_coordinates(
            masks,
            np.flipud(peaks.loc[:, 'x':'y'].to_numpy().T),
            order=0,
        ).astype(int)
        peaks.label = labels

        slices = find_objects(masks)
        num_masks = int(np.max(masks))

        peaks.sort_values(by='label', ignore_index=True, inplace=True)

    return masks, num_masks, slices, peaks


def img_equ_ellip(image):
    """
    Calculate the equivalent ellipse

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    eigvals : squared magnitudes of the major and minor semi-axes, in that
        order

    eigvects : matrix containing the unit vectors of the major and minor
        semi-axes (in that order) as row vectors

    x0, y0 : coordinates of the ellipse center


    """

    M = skimage.measure.moments(image, order=1)
    mu = skimage.measure.moments_central(image, order=2)

    try:
        [x0, y0] = [M[0, 1]/M[0, 0], M[1, 0]/M[0, 0]]

        [u20, u11, u02] = [
            mu[2, 0]/M[0, 0],
            mu[1, 1]/M[0, 0],
            mu[0, 2]/M[0, 0]
        ]

        cov = np.array(
            [[u20, u11],
             [u11, u02]]
        )

    except ZeroDivisionError as err:
        raise ZeroDivisionError(no_mask_error) from err

    try:
        eigvals, eigvects = np.linalg.eig(cov)
    except np.linalg.LinAlgError as err:
        raise ArithmeticError(no_mask_error) from err

    # Exchange vector components so each column vector is [x, y]:
    eigvects = np.flipud(eigvects)

    if eigvects[0, 0] < 0:
        eigvects[:, 0] *= -1
    if eigvects[0, 1] < 0:
        eigvects[:, 1] *= -1

    ind_sort = np.flip(np.argsort(eigvals))  # Sort large to small

    eigvals = np.abs(np.take_along_axis(
        eigvals,
        ind_sort,
        0
    ))

    eigvects = np.array([eigvects[:, ind] for ind in ind_sort])

    return eigvals, eigvects, x0, y0


def img_ellip_param(image):
    """
    Find parameters of the equivalent ellipse
    Calls img_equ_ellip and transforms result to a more intuitive form

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    params : list of scalars
        [x0, y0, eccen, theta, sig_maj, sig_min]:
            x0, y0 : coordinates of the ellipse center
            eccen : eccentricity of the ellipse (standard mathmatical
                definition)
            theta : rotation angle of the major semi-axis relative to
                horizontal
            in degrees (positive is counterclockwise)
            sig_maj : magnitude of the major semi-axis
            sig_min : magnitude of the minor semi-axis

    """

    eigvals, eigvects, x0, y0 = img_equ_ellip(image)
    major = np.argmax(eigvals)
    minor = np.argmin(eigvals)
    sig_maj = np.sqrt(eigvals[major])
    sig_min = np.sqrt(eigvals[minor])
    eccen = np.sqrt(1-eigvals[minor]/eigvals[major])
    theta = np.degrees(np.arcsin(
        np.cross(np.array([1, 0]),
                 eigvects[major]).item()
    ))

    params = [x0, y0, eccen, theta, sig_maj, sig_min]

    return params


def moments(x, y, z, order=1):
    """
    Calculate image moments from indices and weighting (intensity).

    Parameters
    ----------
    x, y : list-like of scalars
         Coordinates of pixels.

    z : list-like of scalars
        Intensity values of each pixel.

    order : int
        The max moment order to be calculated.
        Default: 1.

    Returns
    -------
    M : array
        Moments as an array. To select a specific moment from the array, index
        by the [x, y] moment order desired (e.g. M[1, 0] is the x mean).

    """

    M = np.array([[np.sum(x**i * y**j * z)
                   for j in range(order + 1)]
                  for i in range(order + 1)])

    return M


def moments_central(x, y, z, order=2):
    """
    Calculate image central moments from indices (x, y) and intensity (z).

    Parameters
    ----------
    x, y : list-like of scalars
         Coordinates of pixels.

    z : list-like of scalars
        Intensity values of each pixel.

    order : int
        The max moment order to be calculated.
        Default: 1.

    Returns
    -------
    M : array
        Central moments as an array. To select a specific moment from the
        array, index by the [x, y] moment order desired (e.g. M[1, 0] is the
        x mean).

    """

    M = moments(x, y, z, order=1)

    x_, y_ = [M[1, 0]/M[0, 0], M[0, 1]/M[0, 0]]

    mu = np.array([[np.sum((x - x_)**i * (y - y_)**j * z)
                   for j in range(order + 1)]
                  for i in range(order + 1)])

    return mu


def equ_ellip_fromxyz(x, y, z):
    """
    Calculate the equivalent ellipse from x,y coordinates and intensity
    values.

    Parameters
    ----------
    x, y : arrays of scalars
         Coordinates of pixels.

    z : list-like of scalars
        Intensity values of each pixel.

    Returns
    -------
    eigvals : squared magnitudes of the major and minor semi-axes, in that
        order
    eigvects : matrix containing the unit vectors of the major and minor
    semi-axes (in that order) as row vectors
    x0, y0 : coordinates of the ellipse center

    """

    M = moments(x, y, z, order=1)
    mu = moments_central(x, y, z, order=2)

    try:
        [x0, y0] = [M[1, 0]/M[0, 0], M[0, 1]/M[0, 0]]

        [u20, u11, u02] = [
            mu[2, 0]/M[0, 0],
            mu[1, 1]/M[0, 0],
            mu[0, 2]/M[0, 0]
        ]

        cov = np.array(
            [[u20, u11],
             [u11, u02]]
        )

    except ZeroDivisionError as err:
        raise ZeroDivisionError(no_mask_error) from err

    try:
        eigvals, eigvects = np.linalg.eig(cov)
    except np.linalg.LinAlgError as err:
        raise ArithmeticError(no_mask_error) from err

    # Exchange vector components so each column vector is [x, y]:
    # eigvects = np.flipud(eigvects)

    if eigvects[0, 0] < 0:
        eigvects[:, 0] *= -1
    if eigvects[0, 1] < 0:
        eigvects[:, 1] *= -1

    ind_sort = np.flip(np.argsort(eigvals))  # Sort large to small

    eigvals = np.abs(np.take_along_axis(
        eigvals,
        ind_sort,
        0
    ))

    eigvects = np.array([eigvects[:, ind] for ind in ind_sort])

    return eigvals, eigvects, x0, y0


def ellip_param_fromxyz(x, y, z, subtract_min=True):
    """
    Find parameters of the equivalent ellipse for coordinates and intensity
    values.

    Calls equ_ellip_fromxyz and transforms result to a more intuitive form.

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    params : list of scalars
        [x0, y0, eccen, theta, sig_maj, sig_min]:
            x0, y0 : coordinates of the ellipse center
            eccen : eccentricity of the ellipse (standard mathmatical
                definition)
            theta : rotation angle of the major semi-axis relative to
                horizontal
            in degrees (positive is counterclockwise)
            sig_maj : magnitude of the major semi-axis
            sig_min : magnitude of the minor semi-axis

    """

    if subtract_min:
        z_ = z - np.min(z)
    else:
        z_ = z

    eigvals, eigvects, x0, y0 = equ_ellip_fromxyz(x, y, z_)
    major = np.argmax(eigvals)
    minor = np.argmin(eigvals)
    sig_maj = np.sqrt(eigvals[major])
    sig_min = np.sqrt(eigvals[minor])
    eccen = np.sqrt(1-eigvals[minor]/eigvals[major])
    theta = np.degrees(np.arcsin(
        np.cross(np.array([1, 0]),
                 eigvects[major]).item()
    ))

    params = [x0, y0, eccen, theta, sig_maj, sig_min]

    return params


def gaussian_ellip_ss(p0, x, y, z):
    """Sum of squares for a Gaussian function.

    Takes a parameter vector, coordinates, and corresponding data values;
    returns the sum of squares of the residuals.

    Parameters
    ----------
    p0 : array_like with shape (n,7)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig_maj, sig_rat, ang, A, b]

    x, y : 1D array_like, must have the same shape
        The flattened arrays of x and y coordinates of image pixels

    z : 1D array_like, must have the same shape as x and y
        The flattened array of image values at the x, y coordinates

    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals

    """

    if p0.shape[0] > 7:
        b = p0[-1]
        p0_ = p0[:-1].reshape((-1, 6))
        x0, y0, sig_maj, sig_rat, ang, A = np.split(p0_, 6, axis=1)
    else:
        x0, y0, sig_maj, sig_rat, ang, A, b = p0

    ang *= -1

    # Sum the functions for each peak:
    model = gaussian_2d(x, y, x0, y0, sig_maj, sig_rat, ang, A, b)

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def gaussian_circ_ss(p0, x, y, z):
    """Sum of squares for a Gaussian function.

    Takes a parameter vector, coordinates, and corresponding data values;
    returns the sum of squares of the residuals.

    Parameters
    ----------
    p0 : array_like with shape (n,5)
        n = number of peaks to fit
        Array containing the Gaussian function parameter vector(s):
            [x0, y0, sig, A, b]

    x, y : 1D array_like, must have the same shape
        The flattened arrays of x and y coordinates of image pixels

    z : 1D array_like, must have the same shape as x and y
        The flattened array of image values at the x, y coordinates

    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals

    """

    if p0.shape[0] > 5:
        b = p0[-1]
        p0_ = p0[:-1].reshape((-1, 4))
        x0, y0, sig, A = np.split(p0_, 4, axis=1)
    else:
        x0, y0, sig, A, b = p0

    # Sum the functions for each peak:
    model = gaussian_2d(x, y, x0, y0, sig, 1, 0, A, b)

    # Subtract from data to get residuals:
    R = np.atleast_2d(z) - model
    r_sum_sqrd = np.sum(R**2)

    return r_sum_sqrd


def fit_gaussians(
        xy,
        z,
        p0,
        method='BFGS',
        bounds=None,
        circular=True,
):
    """Fit 2D Gaussain function(s) to data simultaneously while including
    overlap of each fitted Gaussian with all the others in the group.

    Parameters
    ----------
    xy : list of n arrays of shape (m, 2)
        Each array represents the pixel coordinates of a peak region.

    z : array of shape (m,)
        The intensity values of the pixels indexed by xy.

    p0 : array_like with shape (n*6 + 1,) OR (n*4 +1)
        Initial guess for the n-Gaussian parameter vector where each peak
        has 6 independent parameters (x0, y0, sig_maj, sig_ratio, ang, A) the
        whole region has a constant background (b) which is the last item in
        the array.

    method : str, the minimization solver name
        Supported solvers are: ''BFGS', 'L-BFGS-B', 'Powell', 'trust-constr'.
        Default: 'BFGS'

    bounds : list of two-tuples of length 7*n or None
        The bounds for Gaussian fitting parameters. Only works with methods
        that accept bounds (e.g. 'L-BFGS-B', but not 'BFGS'). Otherwise must
        be set to None.
        Each two-tuple is in the form: (upper, lower).
        Order of bounds must be: [x0, y0, sig_maj, sig_ratio, ang, A, b] * n
        Default: None

    circular : bool
        Whether to use a circular Gaussian for fitting. If False, uses an
        elliptical Gaussian.
        Default: True

    Returns
    -------
    params : 1D array
        The best fit parameter vector found by least squares

    """

    if circular:
        n_params = 4
    else:
        n_params = 6

    num_gauss = int((p0.shape[0] - 1) / n_params)

    x, y = xy

    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            lineno=182
        )

        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            lineno=579
        )

    if circular:
        params = minimize(
            gaussian_circ_ss,
            p0,
            args=(x, y, z),
            bounds=bounds,
            method=method
        ).x

    else:
        params = minimize(
            gaussian_ellip_ss,
            p0,
            args=(x, y, z),
            bounds=bounds,
            method=method
        ).x

    params = np.concatenate(
        (params[:-1].reshape((num_gauss, n_params)),
         np.ones((num_gauss, 1))*params[-1]),
        axis=1
    )
    if circular:
        params = np.insert(params, [-2, -2], [1, 0], axis=1)

    else:
        params[:, 4] *= -1
        params[:, 4] = ((params[:, 4] + 90) % 180) - 90

    return params


def group_fitting_data(
        selected_labels,
        slices,
        masks,
        xy_peaks,
        peak_masks_dforder,
        xycoords,
):
    """
    Get y,x pixel indices for each labeled mask in the list labels.

    Parameters
    ----------
    selected_labels : list-like of length (m)
        Selected fitting labels (i.e. peak masks) to group for simultaneous
        fitting.

    slices : list of slices
        List of slices for all the peak masks in label order.

    masks : 2d array
        Labeled masks for atom column peak fitting.

    xy_peaks : array of shape (n,2)
        The [x,y] coordinates of the individual peaks to be fit.

    peak_masks_dforder : array of shape (n,)
        Mask label number for each peak in xy_peaks.

    xycoords : 3d array of shape (2, h, w)
        The [x, y] coordinates of all pixels in the full image shape.

    Returns
    -------
    datagroup : list
        List of the grouped data for the group of peaks to be fit
        simultaneously:

            selected_labels : list-like of length (m)
                Same as the input.

            dfinds : list
                Indices of the DataFrame corresponding to each label in
                selected_labels

            peak_coords : array of shape (m, 2)
                Coordinates of the peaks corresponding to selected_labels.

            mask_coords : list of arrays of shape (2, ?)
                List is over each peak in the group in the same order as
                selected_labels. The arrays contain the coordinates of each
                pixel in the corresponding peak's mask.

    """

    # This function works on a set of peaks & masks which have been grouped
    # together for simultaneous fitting.
    # There may be more than one mapped to a single peak mask so here we check
    # for duplicates and handle it appropriately.

    # Get the dataframe index (indices) which have been mapped to each mask.
    dfinds = [np.nonzero(peak_masks_dforder == lab)[0]
              for lab in selected_labels]

    # "counts" reveals if a mask maps to more than one dataframe index.
    counts = np.array([len(i) for i in dfinds])
    if np.max(counts) > 1:

        dup_inds = np.argwhere(counts > 1).flatten()

        dups = np.array([dfinds[i] for i in dup_inds])

        dup_unique, dup_counts = np.unique(dups, axis=0, return_counts=True)

        for i, dup in enumerate(dup_unique):
            matches = np.nonzero(np.all(dups == dup, axis=1))[0]

            if dup_counts[i] > 1:
                for j, m in enumerate(matches):
                    dfinds[dup_inds[m]] = [dfinds[dup_inds[m]][j]]

    dfinds = np.concatenate(dfinds)

    peak_coords = xy_peaks[dfinds]

    # Slicing, then getting coordinates is faster than argwhere on the whole:
    # Slice the coordinates array

    ind_slices = [xycoords[:, slices[lab-1][0], slices[lab-1][1]]
                  for lab in selected_labels]

    # Slice the mask array
    mask_slices = [np.where(masks[slices[lab-1]] == lab, 1, 0)
                   for lab in selected_labels]

    # Get list of coordinates for the mask
    mask_coords = [ind_sl[:, mask_slices[i] != 0]
                   for i, ind_sl in enumerate(ind_slices)]

    datagroup = [selected_labels, dfinds, peak_coords, mask_coords]

    return datagroup


def fit_gaussian_group(
        image,
        peak_coords,
        mask_coords,
        pos_bound_dist=None,
        use_circ_gauss=False,
        use_bounds=False,
):
    """Master function for simultaneously fitting one or more Gaussians to a
    piece of data.

    Parameters
    ----------
    image : 2d array
        The full image.

    peak_coords : array of shape (n,2)
        The [x,y] coordinates of the individual peaks to be fit.

    mask_coords : list of arrays of shape (m, 2)
        List of arrays. Each array contains the [x,y] coordinates of each
        pixel in one of the masks corresponding to a peak.

    pos_bound_dist : scalar or None
        The +/- distance in pixels used to bound the x, y position of
        each atom column fit from its initial guess location. If 'None',
        position bounds are not used.
        Default: None

    use_circ_gauss : bool
        Whether to force circular Gaussians for initial fitting of atom
        columns. If True, applies all bounds to ensure physically
        realistic parameter values. In some instances, especially for
        significantly overlapping columns, using circular Guassians is
        more robust in preventing obvious atom column location errors.
        In general, however, atom columns may be elliptical, so using
        elliptical Guassians should be preferred.
        Default: False

    use_bounds : bool
        Whether to apply bounds to minimization algorithm. This may be
        needed if unphysical results are obtained (e.g. negative
        background, negative amplitude, highly elliptical fits, etc.).
        The bounded version of the minimization runs slower than the
        unbounded.
        Default: False

    Returns
    -------
    params : array of shape (m, 7)
        The parameters describing each fitted Gaussian.

    """

    [x0, y0] = peak_coords.T

    num = peak_coords.shape[0]

    z = [image[coords[1], coords[0]] for coords in mask_coords]

    # Create empty parameter lists
    sig_1 = []
    sig_2 = []
    sig_rat = []
    theta = []
    b = []
    A0 = []

    for i in range(num):
        # Estimate initial parameters for each peak using moments
        _, _, _, theta_, sig_1_, sig_2_ = ellip_param_fromxyz(
            mask_coords[i][0],
            mask_coords[i][1],
            z[i])

        # # Find the intensity maximum as initial x,y position guess

        # Make corrections for mis-estimations by the moments calculation
        sig_replace = 3
        if sig_1_ <= 1:
            sig_1_ = sig_2_ = sig_replace
        elif sig_2_ <= 1:
            sig_2_ = sig_replace

        if sig_1_/sig_2_ > 3:
            sig_1_ = sig_2_
            theta_ = 0

        # Add parameters to parameter lists
        # x0 += [xy0[0]]
        # y0 += [xy0[1]]
        sig_1 += [sig_1_]
        sig_2 += [sig_2_]
        sig_rat += [sig_1_ / sig_2_]
        theta += [theta_]
        b += [(np.average(z[i]) - np.std(z[i]))]
        A0 += [np.max(z[i]) - b[i]]

    if use_circ_gauss:
        # Set bounds and group parameters for circular guassian fitting
        if use_bounds:
            bounds = [
                (None, None),
                (None, None),
                (1, None),
                (0, 1.2),
            ] * num + [(0, None)]
            for i in range(num):
                if pos_bound_dist == np.inf:
                    break
                bounds[i*4] = (x0[i] - pos_bound_dist,
                               x0[i] + pos_bound_dist)
                bounds[i*4 + 1] = (y0[i] - pos_bound_dist,
                                   y0[i] + pos_bound_dist)

            method = 'L-BFGS-B'

        else:
            bounds = None
            method = 'BFGS'

        p0 = np.array(
            [x0, y0, np.mean([sig_1, sig_2], axis=0), A0]
        ).T

        p0 = np.append(p0.flatten(), np.mean(b))

    else:
        # Set bounds and group parameters for elliptical guassian fitting
        if use_bounds:
            bounds = [
                (None, None),
                (None, None),
                (1, None),
                (1, None),
                (None, None),
                (0, 1.2),
            ] * num + [(0, None)]

            for i in range(num):
                if pos_bound_dist == np.inf:
                    break
                bounds[i*6] = (x0[i] - pos_bound_dist,
                               x0[i] + pos_bound_dist)
                bounds[i*6 + 1] = (y0[i] - pos_bound_dist,
                                   y0[i] + pos_bound_dist)

            method = 'L-BFGS-B'

        else:
            bounds = None
            method = 'BFGS'

        p0 = np.array(
            [x0, y0, sig_1, sig_rat, theta, A0]
        ).T

        p0 = np.append(p0.flatten(), np.mean(b))

    # Concatenate x, y, z arrays & ensure no duplicate pixels are included.
    xy = np.stack([np.concatenate([mask_coords[i][0] for i in range(num)]),
                   np.concatenate([mask_coords[i][1] for i in range(num)])])
    z = np.concatenate(z)

    inds = np.sort(np.unique(xy, axis=1, return_index=True)[1])
    xy = xy[:, inds]
    z = z[inds]

    params = fit_gaussians(
        xy, z,
        p0,
        method=method,
        bounds=bounds,
        circular=use_circ_gauss,
    )

    params = np.array(
        [params[:, 0],
         params[:, 1],
         params[:, 2],
         params[:, 2]/params[:, 3],
         params[:, 4],
         params[:, 5],
         params[:, 6]]
    ).T

    return params


def sum_squares(p0, args, fn):
    """Sum of squared errors between data and a model given a parameter vector.

    Parameters
    ----------
    p0 : 1d array
        The parameter vector.

    args : list
        The independent and dependent variables of the data as taken by 'fn'.

    fn : defined function name
        Function that calculates dependent values from a model given the
        independent variables and parameter vector.

    Returns
    -------
    r_sum_sqrd : float
        The sum of the squares of the residuals.

    """

    # Sum the functions for each peak:
    z = args[-1]
    model = fn(*args[:-1], p0)

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def plane_fit(
        data,
        p0,
):
    """Fit a plane to 3D coordinate data or 2D intensity data.

    Parameters
    ----------
    data: 2D array
        The intensity values or z-coordinates in a regularly sampled array.

    p0 : 3-list
        Initial guess for the plane fitting parameters: the x & y
        slopes and the z intercept.

    Returns
    -------
    params : array
        The fitted parameters.

    """

    y, x = np.indices(data.shape)
    z = data.flatten()

    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

    args = [x, y, z]

    params = minimize(
        sum_squares,
        p0,
        args=(args, plane_2d),
        method='L-BFGS-B',
    ).x

    return params


def line_fit(
    xdata,
    ydata,
    p0,
):
    """Fit a line to 2D data.

    Parameters
    ----------
    xdata, ydata: 1D arrays
        The x and y coordinates of the data.

    p0 : 3-list
        Initial guess for the line fitting parameters: m, b.

    Returns
    -------
    params : array
        The fitted parameters.

    """

    x = xdata.flatten()
    y = ydata.flatten()
    args = [x, y]

    params = minimize(
        sum_squares,
        p0,
        args=(args, line),
        method='Powell',
    ).x

    return params


def dp_peaks_com(
        data,
        sigma,
        filter_type='gauss',
        peak_bkgd_thresh_factor=0.1,
        detection_thresh=0.1,
        buffer=5,
        fit_bkgd=True,
):
    """
    Find peaks in a DP and measure positions with CoM.

    Parameters
    ----------
    data : 2d array
        DP in which to find and measure peaks.

    sigma : scalar
        Gaussian width for bluring the pattern prior to running watershed
        segmentation. All measurements are done on the raw data.

    filter_type : str
        'gauss' or 'log'. Whether to use a regular Gaussian or a Laplacian
        of Gaussian filter for the bluring.
        Default: 'gauss'

    peak_bkgd_thresh_factor : scalar
        Local thresholding method that thresholds between the
        watershed region edge maximum and the peak maximum. It is more
        appropriate if center-of-mass measurements will be made on the
        segmentation regions.
        Default: 0.1

    detection_threshold : scalar
        Peaks with a maximum lower than this will not be detected.

    buffer : int
        The edge border outside which peaks are ignored.
        Default: 5

    fit_bkgd : bool
        Whether to fit and subtract a planar background before CoM
        measurement. If False, a constant background is subtracted equal to
        the minimum intensity value in the peak region.

    Returns
    -------
    peaks : array
        The [x, y] positions of all detected peaks.

    """

    h, w = data.shape
    data_norm = image_norm(data)

    masks, num_masks, _, peaks = watershed_segment(
        data_norm,
        sigma=sigma,
        filter_type=filter_type,
        bkgd_thresh_factor=0,
        peak_bkgd_thresh_factor=peak_bkgd_thresh_factor,
        buffer=buffer,
        min_dist=sigma,
        min_pixels=sigma * 2,
    )
    # Remove edge pixels:
    if buffer > 0:
        peaks = peaks[
            ((peaks.x >= buffer) &
             (peaks.x <= w - buffer) &
             (peaks.y >= buffer) &
             (peaks.y <= h - buffer))
        ].reset_index(drop=True)

    # Update peaks DataFrame with unfiltered data values (std, max, min)
    for i, peak in peaks.iterrows():
        mask = np.where(masks == peak.label, 1, 0)
        peak_masked = mask * data_norm

        peaks.at[i, 'stdev'] = np.std(peak_masked[peak_masked > 0]
                                      ).astype(float)
        peaks.at[i, 'max'] = np.max(peak_masked[peak_masked > 0]
                                    ).astype(float)
        peaks.at[i, 'mean'] = np.mean(peak_masked[peak_masked > 0]
                                      ).astype(float)

        mask_bkgd = dilation(mask, footprint=np.ones((3, 3))) - mask
        bkgd_masked = data_norm * mask_bkgd
        peaks.at[i, 'min'] = np.mean(bkgd_masked[bkgd_masked > 0]
                                     ).astype(float)

    peaks = peaks[peaks.loc[:, 'max'] > detection_thresh
                  ].reset_index(drop=True)

    xy = peaks.loc[:, 'x':'y'].to_numpy(dtype=float)
    y, x = np.indices(data.shape)

    # Measure peak center of mass w/ background subtraction
    for i, xy_ in enumerate(xy):
        mask_num = masks[int(xy_[1]), int(xy_[0])]
        mask = np.where(masks == mask_num, 1, 0)
        masked_peak = data_norm*mask
        if fit_bkgd:
            bkgd_mask = dilation(mask) - mask
            bkgd_int = bkgd_mask * data_norm

            params = plane_fit(
                bkgd_int,
                p0=[0, 0, np.mean(bkgd_int)]
            )
            bkgd = plane_2d(x, y, params)
            bkgd_sub = (data_norm - bkgd) * mask
            masked_peak = np.where(bkgd_sub > 0, bkgd_sub, 0)

        else:
            min_ = np.min(masked_peak[masked_peak > 0])
            masked_peak = (masked_peak - min_) * mask

        masked_peak[masked_peak < 0] = 0

        if np.sum(masked_peak) > 0:
            com = np.flip(center_of_mass(masked_peak))
            peaks.loc[i, ['x_com', 'y_com']] = com

    peaks.dropna(ignore_index=True, inplace=True)

    return peaks


def fit_dipole(patt, coords, masks, peaks, sigma):
    """
    Measure dipole in an EWIC pattern using + & - Gaussians.

    Parameters
    ----------
    patt : 2 array
        EWPC or EWIC pattern in which to find and measure peaks.

    coords : 3d array of shape (2, h, w)
        The y, x coordinates of all pixels in the EWIC.

    masks : 2d array
        The masked dipole peak areas labeled 1 & 2, where the peaks are
        expected and 0 elsewhere.

    peaks : array of shape (2, 2)
        The x, y coordinates of the 2 dipole peaks.

    sigma : scalar
        The expected gaussian peak width.

    Returns
    -------
    params : array
         The [x, y] positions of all detected peaks.

    """

    dipole_fit = []

    for i in range(2):
        yx = coords[:, masks == i+1]
        sign = np.sign(patt[peaks[i][1], peaks[i][0]])

        ewpc = np.all(sign > 0)

        z = patt[*yx]

        nobnd = (None, None)

        if ewpc:
            bounds = [nobnd, nobnd, nobnd, (0, np.inf), (0, 0)]

        else:
            if sign > 0:
                bounds = [nobnd, nobnd, nobnd, (0, np.inf), (-np.inf, 0)]
            else:
                bounds = [nobnd, nobnd, nobnd, (-np.inf, 0), (0, np.inf)]

        p0 = np.array([
            *peaks[i],
            sigma,
            np.max(z*sign)*sign,
            np.mean(z) - np.std(z) * sign
        ])

        params = fit_gaussians(
            np.flip(yx, axis=0),
            z,
            p0,
            method='L-BFGS-B',
            bounds=bounds,
            circular=True,
        ).squeeze()

        params = np.concatenate([params[:3], params[5:]])

        dipole_fit += [params.squeeze()]

    dipole_fit = np.array(dipole_fit)

    return dipole_fit
