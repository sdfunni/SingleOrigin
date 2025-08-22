"""Module containing base pair correlation function methods."""

import copy

import numpy as np

import pandas as pd

from scipy.ndimage import (
    map_coordinates,
    gaussian_filter,
)

from SingleOrigin.utils.image import (
    nearestKDE_2D,
    linearKDE_2D,
)
from SingleOrigin.utils.peakfit import (
    watershed_segment,
    img_ellip_param,
    fit_gaussians
)

from SingleOrigin.utils.environ import is_running_in_jupyter

if is_running_in_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# %%


def pcf_radial(
        dr,
        coords,
        total_area=None
):
    """
    Calculate a radial pair correlation function from 2D or 3D data.

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


def get_vpcf(
        xlim,
        ylim,
        coords1,
        coords2=None,
        d=0.05,
        area=None,
        method='linearKDE'
):
    """
    Get a 2D pair (or pair-pair) correlation function for a dataset.

    Parameters
    ----------
    xlim, ylim : 2-tuple of floats or ints
        The limits of the vPDF along each dimension. Must include 0 in both
        x and y. The limits will determine the size of the vPCF array and the
        time required to calculate.

    coords1 : array of shape (n, 2)
        The (x,y) coordinates of the data points.

    coords2 : None or array of shape (n, 2)
        If dcoords 2 is None, a vPCF is calculated for coords1 with respect to
        itself. If a second data array is passed as coords2, a pair-pair vPCF
        is found i.e. the vPCF of coords1 data with respect to coords2. coords1
        and coords2 do not need to have the same number of data points.
        Default: None

    d : scalar
        The pixel size of the vPCF in the same units as the coords1/coords2.

    area : scalar
        The area containing the data points. Used to calculate the density
        for normalizing the vPCF values. If None, the rectangle containing
        the extreme points in coords1 is taken as the area. This may be wrong
        if the data does not come from a retangular area or the rectangle has
        been rotated relative to the cartesian axes.
        Default: None

    method : 'bin' or 'linearKDE'
        The method to use for calculating the v_pcf. If 'bin', uses a direct
        histogram binning function in two dimensions. If 'linearKDE',
        linearly divides the count for each data point among the 2x2 nearest
        neighbor pixels. Examples:
            1: A point exactly at the center of a pixel will have its full
            weight placed in that pixel and none in any others.
            2: A point at the common corner of 4 pixels will have 1/4 weight
            assigned to each.
        Discussion: 'bin' is about 4x faster in execution while 'linearKDE' is
        more quantitatively correct. Practically, the results will be very
        similar. 'bin' should only be preferred if the function must be called
        many times and speed is critical. This option may be removed in a
        future version in favor of always using the weighted method.
        Default: 'linearKDE'

    Returns
    -------
    v_pcf : ndarray of shape (int((ylim[1]-ylim[0])/d),
                              int((xlim[1]-xlim[0])/d))
        The vPCF.

    origin : array-like of shape (2,)
        The x, y coordinates of the vPCF origin, given that the y axis points
        down.

    """

    if ((xlim[0] > 0) or (ylim[0] > 0) or (xlim[1] < 0) or (ylim[1] < 0)):
        raise Exception(
            "x and y limits must include the origin, i.e. (0,0)"
        )

    if area is None:
        area = (np.max(coords1[:, 0])
                - np.min(coords1[:, 0])) * \
            (np.max(coords1[:, 1]) - np.min(coords1[:, 1]))

    # Get the point-to-point vectors
    if coords2 is None:
        coords2 = coords1

        # Skip 0 length vectors for a partial vPCF
        vects = np.array([
            np.delete(coords1, i, axis=0) - xy for i, xy in enumerate(coords2)
        ])
    else:
        # Keep all vectors for a pair-pair vPCF
        vects = np.array([coords1 - i for i in coords2])

    vects = vects.reshape((-1, 2))

    n_sq = coords1.shape[0] * coords2.shape[0]
    denominator = n_sq / area

    if method == 'bin':

        H, xedges, yedges = nearestKDE_2D(
            vects,
            xlim,
            ylim,
            d,
            return_binedges=True
        )

    elif method == 'linearKDE':

        H, xedges, yedges = linearKDE_2D(
            vects,
            xlim,
            ylim,
            d,
            return_binedges=True
        )

    else:
        raise Exception(
            "'method' must be either 'bin', 'linearKDE'."
        )

    # Flip so y axis is positive going up
    H = np.flipud(H)

    # Find the origin
    origin = np.array([
        np.argwhere(np.isclose(xedges, -d/2)).item(),
        yedges.shape[0] - np.argwhere(np.isclose(yedges, -d/2)).item() - 2
    ])

    H[origin[1], origin[0]] = 0

    vpcf = H/(denominator * d**2)  # Normalize vPCF by number density

    return vpcf, origin


def get_vpcf_peak_params(
    vpcf,
    sigma=10,
    buffer=10,
    method='moments',
    sigma_group=None,
    thresh_factor=1,
):
    """
    Calculate shape of peaks in a pair-pair vPCF.

    Parameters
    ----------
    vpcf : 2d array
        The vPCF.

    sigma : scalar
        The Gaussian sigma for bluring peaks prior to identifying
        mask areas for each peak by he watershed algorithm. In units of
        vPCF pixels.

    buffer : int
        The edge width of the vPCF outside which detected peaks should not be
        measured.

    method : 'momenets' or 'gaussian'
        Method to calculate shape and location of peaks. 'moments' uses
        image moments calculations to measure the peaks while 'gaussian'
        fits with a 2D Gaussian. Methods are roughly equivalent and give
        the parameters of the ellipse that best describes the peak shape.
        Gaussian fitting, however, is more unstable and somewhat slower.
        The primary reason to use 2D Gaussians is in the case of peaks
        with overalapping tails when simultaneous fitting is needed for
        accurate measurements; otherwise, moments should be preferred.
        Default: 'moments'

    sigma_group : scalar or None
        The maximum separation distance used for grouping close peaks for
        simultaneous fitting. This is necessary if peaks are close enough
        to have overlapping tails. Must use Gaussian fitting to account for
        overlap when determining peak shapes. If None, not used.
        Default: None.

    thresh_factor : scalar
        Adjusts the minimum amplitude for considering an identified local
        maximum to be a peak for the purposes of finding its shape. By
        default peaks with less than 10% of the amplitude of the largest
        peak are not considered for fitting. A thresh_factor of 2 would
        raise this cutoff to 20% while 0.5 would lower it to 5%.
        Default: 1.

    Returns
    -------
    None

    """

    xy_bnd = 10    # Position bound limit for gaussian fitting

    vpcf_peaks = pd.DataFrame(columns=['x_fit', 'y_fit',
                                       'sig_maj', 'sig_min',
                                       'theta', 'peak_max', 'ecc'])
    if sigma is not None:
        pcf_sm = gaussian_filter(
            vpcf,
            sigma=sigma,
            truncate=3,
            mode='constant',
        )

    else:
        pcf_sm = copy.deepcopy(vpcf)
        sigma = 2

    masks_indiv, n_peaks, _, peaks = watershed_segment(
        pcf_sm,
        min_dist=sigma,
        # max_thresh_factor=0.5,
        # bkgd_thresh_factor=0,
        sigma=None,
        buffer=buffer,
        watershed_line=False,
    )

    peaks['peak_max'] = vpcf[
        peaks.loc[:, 'y'].to_numpy(dtype=int),
        peaks.loc[:, 'x'].to_numpy(dtype=int)
    ]

    thresh = np.max(peaks.loc[:, 'peak_max']) * 0.1 * thresh_factor

    peaks = peaks[(peaks.loc[:, 'peak_max'] > thresh)
                  ].reset_index(drop=True)
    # n_peaks = peaks.shape[0]
    xy_peak = peaks.loc[:, 'x':'y'].to_numpy(dtype=int)
    labels = peaks.loc[:, 'label'].to_numpy(dtype=int)

    if sigma_group is not None:
        if method != 'gaussian':
            print('Using Gaussian method to account for peak overlap.')
            method = 'gaussian'

        pcf_sm = gaussian_filter(
            vpcf,
            sigma=sigma_group,
            truncate=3
        )

        group_masks, _, _, _ = watershed_segment(
            pcf_sm,
            min_dist=sigma_group,
            bkgd_thresh_factor=0,
            sigma=None,
            buffer=0,
            watershed_line=True
        )

        group_masks_to_peaks = map_coordinates(
            group_masks,
            np.flipud(xy_peak.T),
            order=0
        ).astype(int)

        labels = np.unique(group_masks_to_peaks).astype(int)

        group_masks = np.where(
            np.isin(group_masks, labels),
            group_masks,
            0
        )

    if method == 'moments':
        for i, peak_num in tqdm(enumerate(labels)):
            pcf_masked = np.where(masks_indiv == peak_num, 1, 0
                                  )*vpcf
            peak_max = np.max(pcf_masked)
            x_fit, y_fit, ecc, theta, sig_maj, sig_min = img_ellip_param(
                pcf_masked
            )

            vpcf_peaks.loc[i, 'x_fit':] = [
                x_fit,
                y_fit,
                sig_maj,
                sig_min,
                -theta,
                peak_max,
                ecc,
            ]

    elif method == 'gaussian':
        for i in tqdm(labels):
            if sigma_group is None:
                mask = np.where(masks_indiv == i, 1, 0)
            else:
                mask = np.where(group_masks == i, 1, 0)

            pcf_masked = mask * vpcf

            match = np.argwhere([mask[y, x] for x, y in xy_peak])

            # match = np.array([[y, x] for x, y in xy_peak])

            mask_peaks = peaks.loc[match.flatten(), :]

            if sigma_group is not None:

                pcf_masked *= np.where(
                    np.isin(masks_indiv, peaks.loc[:, 'label']),
                    1, 0)

            p0 = []
            bounds = []
            for j, (ind, row) in enumerate(mask_peaks.iterrows()):
                mask_num = masks_indiv[int(row.y), int(row.x)]
                mask = np.where(masks_indiv == mask_num, 1, 0)
                peak_masked = mask * vpcf

                x0, y0, ecc, theta, sig_maj, sig_min = img_ellip_param(
                    peak_masked
                )

                p0 += [
                    x0,
                    y0,
                    sig_maj,
                    sig_maj/sig_min,
                    np.max(pcf_masked),
                    theta
                ]

                bounds += [(x0 - xy_bnd, x0 + xy_bnd),
                           (y0 - xy_bnd, y0 + xy_bnd),
                           (1, None),
                           (1, 5),
                           (0, None),
                           (0, None),
                           ]

            p0 = np.array(p0 + [0])
            bounds += [(0, 0)]

            params = fit_gaussians(
                pcf_masked,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                shape='ellip'
            )

            # params = params[:, :-1]
            params[:, 3] = params[:, 2] / params[:, 3]
            params[:, 4] = np.degrees(params[:, 4])
            params[:, -1] = np.sqrt(1 - params[:, 3]**2
                                    / params[:, 2]**2)

            next_ind = vpcf_peaks.shape[0]
            for k, p in enumerate(params):
                vpcf_peaks.loc[next_ind + k, :] = p

    vpcf_peaks = vpcf_peaks.infer_objects()

    return vpcf_peaks
