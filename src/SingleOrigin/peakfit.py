from copy import deepcopy
import warnings

import numpy as np
from numpy import exp, sin, cos

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


from scipy.ndimage import (
    label,
    find_objects,
    gaussian_filter,
    gaussian_laplace,
    maximum_filter,
    center_of_mass,
)
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
from skimage.morphology import dilation


from skimage.segmentation import watershed
from skimage.feature import hessian_matrix_det
from skimage.morphology import binary_erosion, erosion
from skimage.draw import polygon2mask
from skimage.measure import (moments, moments_central)


from SingleOrigin.image import image_norm


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


def get_feature_size(image):
    """
    Gets nominal feature size in the image using automatic scale selection.

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
    """Detect peaks in an image using a maximum filter with a minimum
    separation distance and threshold.

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
    if min_dist < 1:
        min_dist = 1
    kern_rad = int(np.floor(min_dist))
    size = 2*kern_rad + 1
    neighborhood = np.array(
        [1 if np.hypot(i - kern_rad, j - kern_rad) <= min_dist
         else 0
         for j in range(size) for i in range(size)]
    ).reshape((size, size))

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
        sigma=None,
        filter_type='log',
        buffer=0,
        bkgd_thresh_factor=0.95,
        peak_bkgd_thresh_factor=0,
        watershed_line=True,
        min_dist=5,
        min_pixels=9,
):
    """Segment an image using the Watershed algorithm.

    Parameters
    ----------
    image : 2D array_like
        The image to be segmented.

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

    watershed_line : bool
        Seperate segmented regions by a 1 pixel wide line of zero-value pixels.
        Default: True.

    min_dist : int or float
        The minimum distance allowed between detected peaks. Used to create
        a circular neighborhood kernel for peak detection.
        Default: 5.

    min_pixels : int
        The minimum number of pixels allowed in a mask region. If less than
        this value, the mask and associated peak are discarded.
        Default: 9.

    Returns
    -------
    masks : 2D array with same shape as image

    num_masks : int
        The number of masks

    slices : List of image slices which contain each region

    peaks : DataFrame with the coordinates and corresponding mask label for
        each peak not outside the buffer

    """

    img_der = deepcopy(image)
    [h, w] = image.shape

    if sigma is not None:
        if filter_type == 'log':
            img_der = image_norm(-gaussian_laplace(img_der, sigma))
        if filter_type == 'gauss':
            img_der = image_norm(gaussian_filter(img_der, sigma))

    peak_map, peaks = detect_peaks(
        img_der,
        min_dist=min_dist,
        return_DataFrame=True
    )

    local_max, n_peaks = label(peak_map)

    masks = watershed(-img_der, local_max, watershed_line=watershed_line)

    slices = find_objects(masks)
    num_masks = int(np.max(masks))

    # Refine masks with an optional threshold
    if bkgd_thresh_factor > 0:
        masks_ref = np.zeros(image.shape)

        for i in range(0, num_masks):
            mask_sl = np.where(masks[slices[i][0], slices[i][1]] == i+1, 1, 0)
            img_der_sl = img_der[slices[i][0], slices[i][1]]
            edge = mask_sl - binary_erosion(mask_sl)
            thresh = np.max(edge * img_der_sl) * (bkgd_thresh_factor)
            mask_sl = np.where(mask_sl*img_der_sl >= thresh, i+1, 0)
            n_pixels = np.count_nonzero(mask_sl)
            if n_pixels >= min_pixels:
                masks_ref[slices[i][0], slices[i][1]] += mask_sl

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
            mask_sl = np.where(mask_sl*img_der_sl >= thresh, i+1, 0)
            n_pixels = np.count_nonzero(mask_sl)
            if n_pixels >= min_pixels:
                masks_ref[slices[i][0], slices[i][1]] += mask_sl

        masks = masks_ref

    peaks = peaks[
        ((peaks.x >= buffer) &
         (peaks.x <= w - buffer) &
         (peaks.y >= buffer) &
         (peaks.y <= h - buffer))
    ]

    # Reduce all return elements to common remaining peaks/masks
    unique_masks = np.unique(masks)

    remaining = np.intersect1d(
        unique_masks,
        peaks.label.to_numpy()
    ).astype(int)

    masks = np.where(np.isin(masks, remaining), masks, 0)

    peaks = peaks[np.isin(peaks.label.to_numpy(), remaining)]
    peaks = peaks.reset_index(drop=True)

    slices = [slices[i] for i in range(len(slices)) if i in (remaining - 1)]

    num_masks = remaining.shape[0]

    return masks, num_masks, slices, peaks


def get_mask_polygon(
        data,
        vertices=None,
        buffer=0,
        invert=False,
        show_mask=True,
        return_vertices=False
):
    """Get a binary mask for an arbitrary shaped polygon.

    Parameters
    ----------
    data : 2D array or 2-tuple
        The data or data shape the mask will be applied to.

    vertices : n x 2 array or None
        The x, y coordinates of the polygon vertices. If None, will prompt to
        graphically pick vertices with mouse clicks.
        Default: None.

    buffer : int
        Edge border outside of which the mask will be cropped off regardless of
        the vertices chosen.
        Default: 0.

    invert : bool
        Whether to invert the mask so that pixel outside the specified polygon
        are selected as the mask region (if True).
        Default: False.

    show_mask : bool
        Whether to plot the resuting mask for verification.
        Default: True.

    return_vertices : bool
        Whether to return the chosen vertices as an n x 2 array.
        Default: False.

    Returns
    -------
    mask : 2d array
        The mask.

    vertices : n x 2 array (optional)
        The chosen vertices.

    """

    if isinstance(data, tuple):
        data = np.zeros(data)

    if vertices is not None:
        vertices = np.fliplr(np.array(vertices))
    if vertices is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(data, cmap='gist_gray')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            'Left click to add vertices.\n' +
            'Right click or backspace key to remove last.\n' +
            'Center click or enter key to complete.\n' +
            'Must select in clockwise order.'
        )

        vertices = plt.ginput(
            n=-1,
            timeout=0,
            show_clicks=True,
            mouse_add=MouseButton.LEFT,
            mouse_pop=MouseButton.RIGHT,
            mouse_stop=MouseButton.MIDDLE
        )

        plt.close()
        vertices = np.fliplr(np.array(vertices))

    mask = polygon2mask(data.shape, vertices)

    if buffer:
        mask = erosion(
            mask,
            footprint=np.ones((3, 3))
        )

    if invert:
        mask = np.where(mask == 1, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gist_gray')

    ax.imshow(
        mask,
        alpha=0.2,
        cmap='Reds'
    )

    if return_vertices:
        return mask, vertices
    else:
        return mask


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
    eigvects : matrix containing the unit vectors of the major and minor
    semi-axes (in that order) as row vectors
    x0, y0 : coordinates of the ellipse center

    """

    M = moments(image, order=1)
    mu = moments_central(image, order=2)

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

    return x0, y0, eccen, theta, sig_maj, sig_min


def gaussian_2d(x, y, x0, y0, sig_maj, sig_rat, ang, A=1, b=0):
    """Sample a 2D, ellpitical Gaussian function or group of Gaussians.

    Parameters
    ----------
    x, y : ndarrays, must have the same shape
        The x and y coordinates of each sampling point. If given arrays
        generated by numpy.mgrid or numpy.meshgrid, will return an image
        of the Gaussian.

    x0, y0 : scalars
        Center coordinates of Gaussian(s).

    sig_maj : scalar(s)
        Sigma of the major axix of the Gaussian(s).

    sig_ratio : scalar(s)
        Ratio of major to minor axis sigmas of the Gaussian(s).

    ang : scalar(s)
        Rotation angle of the major axis from horizontal for the Gaussian(s).
        In degrees.

    A : scalar(s)
        Peak amplitude of the Gaussian(s).

    b : scalar
        Constant background value for the Gaussian(s).

    Returns
    -------
    z : the value of the function at the specified points. Will have the same
        shape as x, y inputs

    """

    ang = np.radians(-ang)  # negative due to inverted y axis in python
    sig_min = sig_maj / sig_rat

    z = np.sum(A * exp(-0.5 * (
        ((cos(ang) * (x - x0) + sin(ang) * (y - y0)) / sig_maj)**2
        + ((-sin(ang) * (x - x0) + cos(ang) * (y - y0)) / sig_min)**2)),
        axis=0) + b

    return z


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

    if p0.shape[0] > 5:
        b = p0[-1]
        p0_ = p0[:-1].reshape((-1, 4))
        x0, y0, sig, A = np.split(p0_, 4, axis=1)
    else:
        x0, y0, sig, A, b = p0

    # Sum the functions for each peak:
    model = gaussian_2d(x, y, x0, y0, sig, 1, 0, A, b)

    # Subtract from data to get residuals:
    R = z - model
    r_sum_sqrd = (R @ R.T).flatten()

    return r_sum_sqrd


def fit_gaussians(
        data,
        p0,
        method='BFGS',
        bounds=None,
        circular=True,
):
    """Fit an elliptical 2D Gaussain function to data.

    Fits a 2D, elliptical Gaussian to an image. Intensity values equal to zero
    are ignored.

    Parameters
    ----------
    data : ndarray
        Image containing a Gaussian peak

    p0 : array_like with shape (n*6 + 1,)
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

    img_shape = data.shape

    y, x = np.indices(img_shape)
    z = data.flatten()

    unmasked_data = np.nonzero(z)
    z = np.take(z, unmasked_data)
    x = np.take(x.flatten(), unmasked_data)
    y = np.take(y.flatten(), unmasked_data)

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
        params = np.insert(params, [-1, -1], [1, 0], axis=1)

    else:
        params[:, 4] *= -1
        params[:, 4] = ((params[:, 4] + 90) % 180) - 90

    return params


def pack_data_prefit(
        data,
        slices,
        masks,
        xy_peaks,
        peak_mask_index,
        peak_groups,
        pos_bound_dist=None,
        use_circ_gauss=False,
        use_bounds=False,
):
    """Function to group data for the parallelized fitting process.

    Parameters
    ----------
    data : ndarray
        The data in which peaks are to be fit.

    slices : list of slice objects
        The slices to take out of "data" for each peak fitting. May
        contain more than one peak to be fit simultaneously.

    masks : ndarray of ints
        The data masks to isolate peak regions for fitting. Must have the
        same shape as data.

    xy_peaks : (n,2) shape array
        The [x,y] coordinates of the individual peaks to be fit.

    peak_mask_index : list of length (n,)
        The mask number in "masks" corresponding to each coordinate in
        "xy_peaks".

    peak_groups : list of lists
        For each slice, the index (indices) of the corresponding peak(s)
        in xy_peaks.

    storage_index : list of indices
        For each peak (or group of peaks) the index (or indices) of a
        storage object to which the fitting results will belong.

    Returns
    -------
    grouped_data : list
        A list of packaged information for the fit_columns() function.
        Each sublist contains:
            [data array slice,
             mask array slice,
             peak mask numbers to be fit,
             indices of the slice corner nearest the origin,
             initial peak coordinates]

    """

    packed_data = [[
        data[slices[counter]],
        np.where(np.isin(masks[slices[counter]], peak_mask_index[inds]),
                 masks[slices[counter]], 0),
        peak_mask_index[inds],
        [slices[counter][-1].start,
         slices[counter][-2].start
         ],
        xy_peaks[inds, :].reshape((-1, 2)),
        pos_bound_dist,
        use_circ_gauss,
        use_bounds,
    ]
        for counter, inds
        in enumerate(peak_groups)
    ]

    return packed_data


def fit_gaussian_group(
        data,
        masks,
        mask_nums,
        xy_start,
        xy_peaks,
        pos_bound_dist=None,
        use_circ_gauss=False,
        use_bounds=False,
):
    """Master function for simultaneously fitting one or more Gaussians to a
    piece of data.

    Parameters
    ----------
    data : 2d array
        The data to which the Gaussian(s) are to be fit.

    masks : 2d array of ints
        The data masks to isolate peak regions for fitting. Must have the
        same shape as data.

    mask_nums : list of ints
        The mask numbers (labeled in "masks") corresponding , in order, to each
        peak in xy_peaks.

    xy_start : two tuple
        The global coordinates of the origin of the data (i.e. its upper left
        corner). Used to shift the fitting results from coordinate system of
        "data" to that of a parent dataset.

    xy_peaks : array of shape (n,2)
        The [x,y] coordinates of the individual peaks to be fit.

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
    params : array of shape (n,2)
        The fitted parameters for the Gaussian(s).

    """

    num = xy_peaks.shape[0]

    img_msk = data * np.where(masks > 0, 1, 0)

    if num == 1:
        [x0, y0] = (xy_peaks - xy_start).flatten()
        _, _, _, theta, sig_1, sig_2 = img_ellip_param(img_msk)

        sig_replace = 3
        if sig_1 <= 1:
            sig_1 = sig_2 = sig_replace
        elif sig_2 <= 1:
            sig_2 = sig_replace

        if sig_1/sig_2 > 3:
            sig_1 = sig_2
            theta = 0

        sig_rat = sig_1/sig_2

        b = (
            np.average(img_msk[img_msk != 0])
            - np.std(img_msk[img_msk != 0])
        )
        A0 = np.max(img_msk) - b

        if use_circ_gauss:
            if use_bounds:
                bounds = [
                    (x0-pos_bound_dist/2, x0+pos_bound_dist),
                    (y0-pos_bound_dist/2, y0+pos_bound_dist),
                    (1, 1.2),
                    (0, None),
                ] * num + [(0, None)]
                method = 'L-BFGS-B'

            else:
                bounds = None
                method = 'BFGS'

            p0 = np.array(
                [x0, y0, A0, np.mean([sig_1, sig_2]), b]
            )

        else:
            if use_bounds:
                bounds = [
                    (x0 - pos_bound_dist/2, x0 + pos_bound_dist),
                    (y0 - pos_bound_dist/2, y0 + pos_bound_dist),
                    (1, None),
                    (1, None),
                    (None, None),
                    (0, 1.2),
                ] * num + [(0, None)]
                method = 'L-BFGS-B'

            else:
                bounds = None
                method = 'BFGS'

            p0 = np.array([x0, y0, sig_1, sig_rat, theta, A0, b])

    if num > 1:
        x0y0 = xy_peaks - xy_start
        x0 = x0y0[:, 0]
        y0 = x0y0[:, 1]

        sig_1 = []
        sig_2 = []
        sig_rat = []
        theta = []
        b = []
        A0 = []

        for i, mask_num in enumerate(mask_nums):
            mask = np.where(masks == mask_num, 1, 0)
            masked_sl = data * mask
            _, _, _, theta_, sig_1_, sig_2_ = (
                img_ellip_param(masked_sl))

            sig_replace = 3
            if sig_1_ <= 1:
                sig_1_ = sig_2_ = sig_replace
            elif sig_2_ <= 1:
                sig_2_ = sig_replace

            if sig_1_/sig_2_ > 3:
                sig_1_ = sig_2_
                theta_ = 0

            sig_1 += [sig_1_]
            sig_2 += [sig_2_]
            sig_rat += [sig_1_ / sig_2_]
            theta += [theta_]
            b += [(np.average(masked_sl[masked_sl != 0])
                   - np.std(masked_sl[masked_sl != 0]))]
            A0 += [np.max(masked_sl) - b[i]]

        if use_circ_gauss:
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

    params = fit_gaussians(
        img_msk,
        p0,
        method=method,
        bounds=bounds,
        circular=use_circ_gauss,
    )

    params = np.array(
        [params[:, 0] + xy_start[0],
         params[:, 1] + xy_start[1],
         params[:, 2],
         params[:, 2]/params[:, 3],
         params[:, 4],
         params[:, 5],
         params[:, 6]]
    ).T

    return params


def plane_2d(x, y, params):
    """
    Calculate z values for a plane at specified x and y positions.

    Parameters
    ----------
    x, y : ndarrays
        The x and y coordinates at which to calculate the z height of the
        plane. Must be the same shape.

    params : 3-typle of scalars
        The x and y slopes and z-intercept of the plane.

    Returns
    -------
    z : ndarray
        The z values of the plane at the x, y coordinates. Will be the same
        shape as x & y.

    """

    [mx, my, b] = params

    z = mx*x + my*y + b

    return z


def sum_squares(p0, args, fn):

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


def line(x, params):
    """
    Calculate z values for a plane at specified x and y positions.

    Parameters
    ----------
    x, y : ndarrays
        The x and y coordinates at which to calculate the z height of the
        plane. Must be the same shape.

    params : 3-typle of scalars
        The x and y slopes and z-intercept of the plane.

    Returns
    -------
    z : ndarray
        The z values of the plane at the x, y coordinates. Will be the same
        shape as x & y.

    """

    [m, b] = params

    y = m*x + b

    return y


def line_fit(
    xdata,
    ydata,
    p0,
):

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

    """

    h, w = data.shape
    masks, num_masks, _, peaks = watershed_segment(
        data,
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

    # # Update peaks DataFrame with unfiltered data values (std, max, min)
    for i, peak in peaks.iterrows():
        mask = np.where(masks == peak.label, 1, 0)
        peak_masked = mask * data

        peaks.at[i, 'stdev'] = np.std(peak_masked[peak_masked > 0]
                                      ).astype(float)
        peaks.at[i, 'max'] = np.max(peak_masked[peak_masked > 0]
                                    ).astype(float)
        peaks.at[i, 'mean'] = np.mean(peak_masked[peak_masked > 0]
                                      ).astype(float)

        mask_bkgd = dilation(mask, footprint=np.ones((3, 3))) - mask
        bkgd_masked = data * mask_bkgd
        peaks.at[i, 'min'] = np.mean(bkgd_masked[bkgd_masked > 0]
                                     ).astype(float)

    peaks = peaks[peaks.loc[:, 'max'] > detection_thresh
                  ].reset_index(drop=True)

    xy = peaks.loc[:, 'x':'y'].to_numpy(dtype=float)
    xy.shape

    y, x = np.indices(data.shape)

    # Measure peak center of mass w/ or w/out background subtraction
    for i, xy_ in enumerate(xy):
        mask_num = masks[int(xy_[1]), int(xy_[0])]
        mask = np.where(masks == mask_num, 1, 0)
        masked_peak = data*mask
        if fit_bkgd:
            bkgd_mask = dilation(mask) - mask
            bkgd_int = bkgd_mask * data

            params = plane_fit(
                bkgd_int,
                p0=[0, 0, np.mean(bkgd_int)]
            )
            bkgd = plane_2d(x, y, params)
            bkgd_sub = (data - bkgd) * mask
            masked_peak = np.where(bkgd_sub > 0, bkgd_sub, 0)

        else:
            min_ = np.min(masked_peak[masked_peak > 0])
            masked_peak = (masked_peak - min_) * mask

        masked_peak[masked_peak < 0] = 0

        if np.sum(masked_peak) > 0:
            com = np.flip(center_of_mass(masked_peak))
            peaks.loc[i, ['x_com', 'y_com']] = com

    peaks = peaks.dropna()

    return peaks
