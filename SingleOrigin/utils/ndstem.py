"""Module containing utility functions for higher dimensional STEM datasets (e.g. 4D STEM)."""

import psutil

from joblib import Parallel, delayed

import copy

import numpy as np
from numpy.linalg import norm

from scipy.fft import fftshift
from scipy.ndimage import (
    gaussian_filter,
    center_of_mass,
)

from sklearn.cluster import KMeans

from SingleOrigin.utils.image import getAllPixelDists, line_profile
from SingleOrigin.utils.plot import quickplot
from SingleOrigin.utils.fourier import center_dp_ewic, align_dp_cc

from SingleOrigin.utils.environ import is_running_in_jupyter

if is_running_in_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# %%


def get_com_rotation(comx, comy, minimize_divergence=True):
    """
    Get detector rotation from a center of mass vector image.

    Improved version of the py4DSTEM function: x and y axes match matplotlib
    plotting conventions (x increase to the right, y increases down). Given
    the axes convention,
    Furthermore, divergence is ** minimized ** (instead of maximized), which is
    correct for center-of-mass deflections of an electron beam in atomic scale
    electric fields. Minimizing the divervence (i.e. maximum convergence) is
    able to differentiate a 180 degree rotation from correct, which curl
    minimization is not. 180 degree flips will only flip the sign of resulting
    differential phase contrast images, but this can still cause some
    confusion. Regardless, curl minimization is still implimented as an option,
    but may be removed in future versions.

    Parameters
    ----------
    comx, comy : ndarray
        The center of mass components in x and y directions. For large
        datasets, it is accurate and more efficient to calculate on a small
        slice of comx & comy (e.g. instead of measuring rotation on a complete
        2k or 4k scan, measure on a 512x512 slice.)

    minimize_divergence : bool
        Whether to minimize divergence. If False, minimizes curl instead.
        Default: True

    Returns
    -------
    measured_rotation : scalar
        The rotation angle (in degrees) that must be applied to the CoM vectors
        to correct for detector rotation. This correction is also the detector
        rotation angle relative to the scan vectors.

    measured_transpose : bool
        Whether a transpose (swap of comx and comy) must also be
        applied to correct the observed rotation.

    """

    if minimize_divergence:
        rotation_angles_deg = np.around(np.arange(-179, 181, 1), 1)
    else:
        rotation_angles_deg = np.around(np.arange(-89, 91, 1), 1)

    rotation_angles_rad = np.radians(
        rotation_angles_deg)[:, None, None]

    # Untransposed
    com_x = (
        np.cos(rotation_angles_rad) * comx[None]
        - np.sin(rotation_angles_rad) * comy[None]
    )
    com_y = (
        np.sin(rotation_angles_rad) * comx[None]
        + np.cos(rotation_angles_rad) * comy[None]
    )

    if minimize_divergence:
        com_grad_x_x = (
            com_x[:, 1:-1, 2:] - com_x[:, 1:-1, :-2]
        )
        com_grad_y_y = (
            com_y[:, 2:, 1:-1] - com_y[:, :-2, 1:-1]
        )
        rotation_div = np.mean(
            (com_grad_x_x + com_grad_y_y), axis=(-2, -1)
        )

    else:
        com_grad_x_y = (
            com_x[:, 2:, 1:-1] - com_x[:, :-2, 1:-1]
        )
        com_grad_y_x = (
            com_y[:, 1:-1, 2:] - com_y[:, 1:-1, :-2]
        )
        rotation_curl = np.mean(
            np.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
        )

    # Transposed
    com_x_t = (
        np.cos(rotation_angles_rad) * comy[None]
        - np.sin(rotation_angles_rad) * comx[None]
    )
    com_y_t = (
        np.sin(rotation_angles_rad) * comy[None]
        + np.cos(rotation_angles_rad) * comx[None]
    )

    if minimize_divergence:
        com_grad_x_x = (
            com_x_t[:, 1:-1, 2:] - com_x_t[:, 1:-1, :-2]
        )
        com_grad_y_y = (
            com_y_t[:, 2:, 1:-1] - com_y_t[:, :-2, 1:-1]
        )
        rotation_div_transpose = np.mean(
            (com_grad_x_x + com_grad_y_y), axis=(-2, -1)
        )

    else:
        com_grad_x_y = (
            com_x_t[:, 2:, 1:-1] - com_x_t[:, :-2, 1:-1]
        )
        com_grad_y_x = (
            com_y_t[:, 1:-1, 2:] - com_y_t[:, 1:-1, :-2]
        )
        rotation_curl_transpose = np.mean(
            np.abs(com_grad_y_x - com_grad_x_y), axis=(-2, -1)
        )

    rotation_angles_rad = np.squeeze(rotation_angles_rad)
    rotation_angles_deg = rotation_angles_deg

    # Find lowest curl/ maximum div value
    if minimize_divergence:
        # Minimize Divergence
        ind_min = np.argmin(rotation_div).item()
        ind_trans_min = np.argmin(rotation_div_transpose).item()

        if rotation_div[ind_min] <= rotation_div_transpose[ind_trans_min]:
            measured_rotation = rotation_angles_deg[ind_min]
            # rotation_best_rad = rotation_angles_rad[ind_min]
            measured_transpose = False
        else:
            measured_rotation = rotation_angles_deg[ind_trans_min]
            # rotation_best_rad = rotation_angles_rad[ind_trans_min]
            measured_transpose = True

        rotation_div = rotation_div
        rotation_div_transpose = rotation_div_transpose
    else:
        # Minimize Curl
        ind_min = np.argmin(rotation_curl).item()
        ind_trans_min = np.argmin(rotation_curl_transpose).item()

        rotation_curl = rotation_curl
        rotation_curl_transpose = rotation_curl_transpose
        if rotation_curl[ind_min] <= rotation_curl_transpose[ind_trans_min]:
            measured_rotation = rotation_angles_deg[ind_min]
            # measured_rotation_rad = rotation_angles_rad[ind_min]
            measured_transpose = False
        else:
            measured_rotation = rotation_angles_deg[ind_trans_min]
            # measured_rotation_rad = rotation_angles_rad[ind_trans_min]
            measured_transpose = True

    return measured_rotation, measured_transpose


def get_phase_from_CoM(
    comx,
    comy,
    theta,
    flip,
    regLowPass=0,
    regHighPass=1e-6,
    paddingfactor=2,
    stepsize=1, n_iter=10,
    phase_init=None
):
    """
    Function copied from old version of py4DSTEM (0.12.9)

    Calculate the phase of the sample transmittance from the diffraction
    centers of mass. A bare bones description of the approach taken here is
    below - for detailed iscussion of the relevant theory, see, e.g.::

        Ishizuka et al, Microscopy (2017) 397-405
        Close et al, Ultramicroscopy 159 (2015) 124-137
        Wadell and Chapman, Optik 54 (1979) No. 2, 83-96

    The idea here is that the deflection of the center of mass of the electron
    beam in the diffraction plane scales linearly with the gradient of the
    phase of the sample transmittance. When this correspondence holds, it is
    therefore possible to invert the differential equation and extract the
    phase itself.* The primary assumption made is that the sample is well
    described as a pure phase object (i.e. the real part of the transmittance
    is 1). The inversion is performed in this algorithm in Fourier space,
    i.e. using the Fourier transform property that derivatives in real space
    are turned into multiplication in Fourier space.

    *Note: because in DPC a differential equation is being inverted - i.e. the
    fundamental theorem of calculus is invoked - one might be tempted to call
    this "integrated differential phase contrast".  Strictly speaking, this
    term is redundant - performing an integration is simply how DPC works.
    Anyone who tells you otherwise is selling something.

    Parameters
    ----------
    comx (2D array): the diffraction space centers of mass x coordinates

    comy (2D array): the diffraction space centers of mass y coordinates

    theta (float): the rotational offset between real and diffraction space
        coordinates

    flip (bool): whether or not the real and diffraction space coords
        contain a relative flip

    regLowPass (float): low pass regularization term for the Fourier
        integration operators

    regHighPass (float): high pass regularization term for the Fourier
        integration operators

    paddingfactor (int): padding to add to the CoM arrays for boundry
        condition handling. 1 corresponds to no padding, 2 to doubling the
        array size, etc.

    stepsize (float): the stepsize in the iteration step which updates the
        phase

    n_iter (int): the number of iterations

    phase_init (2D array): initial guess for the phase

    Returns
    -------
    phase : *(2D array)* the phase of the sample transmittance, in radians

    error: *(1D array)* the error - RMSD of the phase gradients compared to the
        CoM - at each iteration step

    """

    flip = not flip
    assert isinstance(flip, (bool, np.bool_))
    assert isinstance(paddingfactor, (int, np.integer))
    assert isinstance(n_iter, (int, np.integer))

    # Coordinates
    R_Nx, R_Ny = comx.shape
    R_Nx_padded, R_Ny_padded = R_Nx*paddingfactor, R_Ny*paddingfactor

    qx = np.fft.fftfreq(R_Nx_padded)
    qy = np.fft.rfftfreq(R_Ny_padded)
    qr2 = qx[:, None]**2 + qy[None, :]**2

    # Inverse operators
    denominator = qr2 + regHighPass + qr2**2*regLowPass
    _ = np.seterr(divide='ignore')
    denominator = 1./denominator
    denominator[0, 0] = 0
    _ = np.seterr(divide='warn')
    f = 1j * -0.25*stepsize

    qxOperator = f*qx[:, None]*denominator
    qyOperator = f*qy[None, :]*denominator

    # Perform rotation and flipping
    if not flip:
        comx_rot = comx*np.cos(theta) - comy*np.sin(theta)
        comy_rot = comx*np.sin(theta) + comy*np.cos(theta)
    if flip:
        comx_rot = comx*np.cos(theta) + comy*np.sin(theta)
        comy_rot = comx*np.sin(theta) - comy*np.cos(theta)

    # Initializations
    phase = np.zeros((R_Nx_padded, R_Ny_padded))
    update = np.zeros((R_Nx_padded, R_Ny_padded))
    dx = np.zeros((R_Nx_padded, R_Ny_padded))
    dy = np.zeros((R_Nx_padded, R_Ny_padded))
    mask = np.zeros((R_Nx_padded, R_Ny_padded), dtype=bool)
    mask[:R_Nx, :R_Ny] = True
    maskInv = ~mask

    print(mask.shape)
    if phase_init is not None:
        phase[:R_Nx, :R_Ny] = phase_init

    # Iterative reconstruction
    error = []
    for i in range(n_iter):

        # Update gradient estimates using measured CoM values
        dx[mask] -= comx_rot.ravel()
        dy[mask] -= comy_rot.ravel()
        dx[maskInv] = 0
        dy[maskInv] = 0

        # Calculate reconstruction update
        update = np.fft.irfft2(np.fft.rfft2(
            dx)*qxOperator + np.fft.rfft2(dy)*qyOperator)

        # Apply update
        phase += stepsize*update

        # Measure current phase gradients
        dx = (np.roll(phase, (-1, 0), axis=(0, 1)) -
              np.roll(phase, (1, 0), axis=(0, 1))) / 2.
        dy = (np.roll(phase, (0, -1), axis=(0, 1)) -
              np.roll(phase, (0, 1), axis=(0, 1))) / 2.

        # Estimate error from cost function, RMS deviation of gradients
        xDiff = dx[mask] - comx_rot.ravel()
        yDiff = dy[mask] - comy_rot.ravel()
        error += [np.sqrt(np.mean((xDiff-np.mean(xDiff)) **
                                  2 + (yDiff-np.mean(yDiff))**2))]

        # Halve step size if error is increasing
        if i > 0:
            if error[i] > error[i-1]:
                stepsize /= 2

    phase = phase[:R_Nx, :R_Ny]

    return phase, error


def make_vAnnularDet(
        shape,
        center,
        inner=5,
        outer=None,
        radial_angles=None,
):
    """
    Get a virtual image from a 4D STEM dataset using a circular/annular
    detector.

    Parameters
    ----------
    shape : 2-tyuple
        The shape of the diffraction pattern dimensions.

    center : 2-list
        The x, y coorinates of the center of the desired virtual detector in
        diffraction space.

    inner : float
        The inner radius of the detector in pixels. For a bright field image,
        set: inner = 0 and pass an outer radius.
        Default: 5

    outer : float
        The outer radius of the detector in pixels. If None, integrates to
        edges of diffraction space.
        Default: None

    radial_angles : numpy array of shape (n, 2) or None
        Rows are start and stop angles (in degrees) to limit the radial extent
        of the annular detector. Any number of segments may be used. Uses the
        unit circle convention for radial angle with 0 degrees pointing to the
        right and increasing counterclockwise. The second angle in each pair
        must be counterclockwise from the first.
        Default: None

    Returns
    -------
    detector : 2D array
        The virtual detector.

    """

    r = getAllPixelDists(shape, center)

    detector = np.where(r >= inner, 1, 0)

    if outer is not None:
        detector *= np.where(r <= outer, 1, 0)

    if radial_angles is not None:
        # Make sure these are numpy arrays:
        center = np.array(center)
        radial_angles = np.array(radial_angles)

        # Map pixel angles from the center:
        xmid = (shape[1] - 1) / 2
        xshift = 2 * (xmid - center[0])

        center_ = center + np.array([xshift, 0])

        y, x = np.indices(detector.shape) - np.flip(center_)[:, None, None]

        x = np.fliplr(x)
        theta = np.degrees(np.arctan2(y, x))
        theta = (theta + 180) % 360
        # theta[center[1], center[0]] = 0

        theta_shifted = (theta + 180) % 360 - 180

        if len(radial_angles.shape) < 2:
            radial_angles = radial_angles[None, :]

        # Get the angle limiting mask:
        angmask = np.zeros(detector.shape)

        for lims in radial_angles:
            print(lims)
            if lims[1] > lims[0]:
                angmask += np.where(
                    ((theta >= lims[0]) & (theta <= lims[1])),
                    1, 0
                )

            else:
                # Use the shifted theta if segment straddles 0 degrees
                lims = (lims + 180) % 360 - 180
                angmask += np.where(
                    ((theta_shifted >= lims[0]) & (theta_shifted <= lims[1])),
                    1, 0
                )

        # Apply the mask:
        detector = detector * angmask

    return detector


def vAnnularDetImage(
        data,
        inner=5,
        outer=None,
        center=None,
        radial_angles=None,
        return_detector=False,
):
    """
    Get a virtual image from a 4D STEM dataset using a circular/annular
    detector.

    Parameters
    ----------
    data : 4d array
        The 4D STEM data.

    center : 2-list
        The x, y coorinates of the center of the desired virtual detector in
        diffraction space (i.e. dims 2 & 3) or None. If None, the center of
        mass of the diffraction pattern is taken as the center.
        Default: None

    inner : float
        The inner radius of the detector in pixels. For a bright field image,
        set: inner = 0 and pass an outer radius.
        Default: 5

    outer : float
        The outer radius of the detector in pixels. If None, integrates to
        edges of diffraction space.
        Default: None

    radial_angles : numpy array of shape (n, 2) or None
        Rows are start and stop angles (in degrees) to limit the radial extent
        of the annular detector. Any number of segments may be used. Uses the
        unit circle convention for radial angle with 0 degrees pointing to the
        right and increasing counterclockwise. The second angle in each pair
        must be counterclockwise from the first.
        Default: None

    return_detector : bool
        Whether to return the virtual detector in addition to the virtual
        image.

    Returns
    -------
    vImage : 2d array
        The virtual image.

    detector : 2d array
        The virtual detector.

    """

    if center is None:
        meandp = np.mean(data, axis=(0, 1))
        # center = np.flip(np.unravel_index(np.argmax(meandp), meandp.shape))
        center = np.flip(center_of_mass(meandp))

    elif len(list(center)) != 2:
        raise Exception('"cent" must be None or array-like of shape (2,)')

    detector = make_vAnnularDet(
        data.shape[-2:],
        center,
        inner=inner,
        outer=outer,
        radial_angles=radial_angles,
    )

    vImage = np.sum(data * detector[None, None, ...], axis=(-2, -1))

    if return_detector:
        return vImage, detector
    else:
        return vImage


def kmeans_segment(data, n_clusters, window=None, pwr=1, plot=True,
                   scanaxes=None, pattaxes=None, roi=None):
    """
    Segment scan using kmeans over the DP/EWPC patterns.

    Parameters
    ----------
    data : ndarray of n+2 dimensions
        Datacube to segment over the first n dimensions with the last 2
        dimensions being the data slices used to determine the clusters. This
        is applicable to datatypes such as 4D STEM, time series stack of
        diffraction patterns (3D dataset), etc. May also supply EWPC data.

    n_clusters : int
        Number of regions to segment image.

    window : int or None
        The wingow size used to block off the central peak for EWPC data. This
        needs to be blocked off because it dominates the patterns and disrupts
        desired segmentation. For diffraction data, pass None.
        Default: None

    pwr : scalar
        Power law scaling applied to data prior to segmentation. Used to
        de-emphasize high intensity information by using pwr < 1.
        Default: 1

    plot : bool
        Whether to plot the segmentation map.
        Default: True.

    scanaxes, pattaxes : tuples of ints or None
        The dimensions of the data that correspond to the scan and diffraction
        axes, respectively. By default it is assumed that the last two axes
        are diffraction space and first 1 or 2 dimension are scan or time.
        Default: None.

    roi : ndarray
        Array along the scan axes where 1 is pixel to be clustered and 0 will
        be ignored.

    Returns
    -------
    labels : ndarray of n dimension
        The cluster labels over the first n dimensions (e.g. a spatial or time
        scan). First cluster label is 1, increasing by integers. If roi was
        passed, pixels outside the roi are 0.

    """

    if scanaxes is None:
        scandims = data.shape[:-2]
    elif type(scanaxes) is tuple:
        scandims = tuple([data.shape[axis] for axis in scanaxes])
    else:
        raise Exception('pattdims must be a tuple')

    if pattaxes is None:
        pattdims = data.shape[-2:]
    elif type(pattaxes) is tuple:
        pattdims = tuple([data.shape[axis] for axis in pattaxes])
    else:
        raise Exception('pattdims must be a tuple')

    if window is not None:
        # Block out central peak:
        data[..., 64-window:64+window+1, 64-window:64+window+1] = 0

    if roi is not None:
        roi_inds = np.argwhere(roi == 1)
        data = copy.deepcopy(data)
        data = data[roi == 1]

    # Prepare kmeans analysis
    X_train = data.reshape((-1, np.prod(pattdims)))
    X_train -= (np.min(X_train) - 1e-8)  # [:, None]
    X_train = X_train**pwr
    X_train /= np.max(X_train)  # [:, None]

    clustering = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init="auto",
        # algorithm='elkan',
    ).fit(X_train)

    if roi is None:
        labels = clustering.labels_.reshape(scandims) + 1

    else:
        labels = np.zeros(scandims)
        labels[roi_inds[:, 0], roi_inds[:, 1]] = clustering.labels_ + 1

    if plot:
        quickplot(labels, cmap='viridis')

    return labels


def scan_line_integration(
        data,
        int_width,
        image=None,
        vdet_args={'inner': 0, 'outer': 5},
        start=None,
        vector=None,
):
    """
    Integrate diffraction patterns along a line in scan space with a given
    integration width. Produces a "profile" of diffraction patterns. Line is
    chosen by graphical picking on a virtual image.

    Parameters
    ----------
    data : ndarray of 2+n dimensions
        The input nD STEM data where the first two dimensions are the scan
        axes and the last n dimensions are the data.

    int_width : scalar
        The width perpendicular to the line over which to integrate the data.

    image : array or None
        The image to diplay for selecting the integration line. If None, a
        default method is applied to create a virtual image but it only applies
        to 4D STEM data. If your data is 3D (e.g. EELS SI), pass some image.
        Default: None.

    vdet_args : dict
        Arguments to be passed to vAnnularDetImage to generate an virtual
        image for graphical picking. Only applies to 4D data.

    start : array-like or None
        The x, y start point of the integration line, if previously determined.
        Default: None.

    vector : array-like or None
        The vector from the start to the end point of the integration line.
        Default: None.

    Returns
    -------
    line_integration : 3d array
        The resulting integrated patterns along the line.

    start : array
        Coordinates of the line start point.

    vect : array
        Vector from start to end point of the line.

    """

    if image is None:
        image = vAnnularDetImage(data, **vdet_args)

    scandims = data.shape[:2]
    signaldims = data.shape[2:]

    line_integration, start, vector = line_profile(
        data,
        int_width,
        image=image,
        scandims=scandims,
        signaldims=signaldims,
        start=start,
        vector=vector,
    )

    return line_integration, start, vector


def center_dataset_ewic(
        data,
        xy0,
        shift_limit=None,
        return_shifts=False,
        nearest_pixel=False,
        hann_window=True,
):
    """
    Center diffraction patterns in an nD STEM dataset by minimizing the
    imaginary cepstrum for each pattern individually.

    Parameters
    ----------
    data : ndarray
        The dataset with the last two dimensions being the diffraction axes.

    xy0 : array like of shape (2,)
        The (x, y) initial guess for the diffraction pattern center.

    shift_limit : scalar
        The maximum allowed shift along x and y from the initial guess in
        pixels.

    return_shifts : bool
        If true, will return the centered DPs and shifts applied. If false,
        only returns the centered DPs.
        Default: False.

    nearest_pixel : bool
        Whether to round to the nearest pixel when centering. For EWIC
        polarization analysis, it is better to center exactly, but subpixel
        centering with the Fourier shift method produces diffraction patterns
        with obvious artifacts. The artifacts do not adversely affect EWIC
        analysis, but are undesirable for other analyses.
        Default: False

    hann_window : bool
        Whether to apply a Hann window before each FFT operation. Significantly
        recudes intensity outside the middle of the dataset.

    Returns
    -------
    data_cent : ndarray
        The centered dataset.
    """

    n_jobs = psutil.cpu_count(logical=True)

    center = np.flip(data.shape[-2:]) // 2

    coords = fftshift(
        np.flip(np.indices(data.shape[-2:]), axis=0),
        axes=(-2, -1)
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(center_dp_ewic)(
            dp,
            xy0,
            coords,
            return_shifts,
            nearest_pixel,
            hann_window,
            shift_limit,
        ) for dp in tqdm(data.reshape(-1, *data.shape[-2:]))
    )

    if return_shifts:
        shifts = np.array([cent - center for _, cent in results]
                          ).reshape((*data.shape[:2], 2))
        results = [dp for dp, _ in results]

    data_cent = np.array(results).reshape(data.shape)

    if return_shifts:
        return data_cent, shifts
    else:
        return data_cent


def descan_dataset(
        data,
        return_shifts=False,
        hann_window=False,
):
    """
    Align DPs in an nD STEM dataset with each other via the average DP.
    Cross correlation is used to estimate the shift for each DP, then they
    are Fourier shifted this amount.

    Parameters
    ----------
    data : ndarray
        The dataset with the last two dimensions being the diffraction axes.

    return_shifts : bool
        If true, will return the centered DPs and the original measured
        centers. If false, only returns the centered DPs.
        Default: False.

    hann_window : bool
        Whether to apply a Hann window before each FFT operation. Significantly
        recudes intensity outside the middle of the dataset.

    Returns
    -------
    data_cent : ndarray
        The centered dataset.
    """

    n_jobs = psutil.cpu_count(logical=True)

    h, w = data.shape[:2]
    origin = np.array([w, h]) // 2

    # Geth the average pattern from the center 1/10th of the scan.
    # All patterns will be aligned with this pattern.
    center = np.mean(data[origin[1] - h//20: origin[1] + h//20,
                          origin[0] - w//20: origin[0] + w//20],
                     axis=(0, 1))

    # Test for maximum shifts at the corner pixels
    corners = np.array([data[i, j] for i in [0, -1] for j in [0, -1]])
    cornershift = np.array([
        align_dp_cc(corner, center, return_shift=True)[1]
        for corner in corners
    ])

    # Blur the mean center pattern by the maximum shift found. This ensures
    # That smooth and reliable shifts are found for all the patterns.
    sigma = np.around(np.max(norm(cornershift, axis=1)))
    data = data.reshape((-1, *data.shape[-2:]))

    dpmean = gaussian_filter(center, sigma=sigma)

    print('Shifting patterns...')
    results = Parallel(n_jobs=n_jobs)(
        delayed(align_dp_cc)(
            dp, dpmean, return_shifts
        ) for i, dp in tqdm(enumerate(data))
    )

    if return_shifts:
        shifts = np.array([shift for _, shift in results]
                          ).reshape((h, w, 2))
        results = [dp for dp, _ in results]

    data_descanned = np.array(results).reshape((h, w, *data.shape[-2:]))

    if return_shifts:
        return data_descanned, shifts
    else:
        return data_descanned
