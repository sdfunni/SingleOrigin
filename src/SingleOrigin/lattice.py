import numpy as np
from numpy.linalg import norm, lstsq

import pandas as pd

from matplotlib import pyplot as plt

from scipy.optimize import minimize

from SingleOrigin.plot import quickplot

# %%


def make_lattice(
        basis,
        origin,
        min_order=0,
        max_order=10,
        xlim=None,
        ylim=None
):
    """
    Generate list of lattice points in fractional and cartesian coordinates.

    Parameters
    ----------
    basis : array of shape (2,2)
        The array of basis vectors as row vectors [x, y] in cartesin
        coordinates.

    origin : array of shape (2,)
        The origin point of the lattice.

    min_order : int
        The minimum order of points allowed in the lattice. e.g. if 2, zeroith 
        and first order points are excluded.
        Default: 0

    max_order : int
        The maximum order of points allowed in the lattice. e.g if 5, only
        points up to order 5 are included in the search and fitting.
        Default: 10

    xlim, ylim : 2-tuples or None
        The minimum and maximum limits of allowed cartesian coordinates in the
        x & y directions.
        Default: None.

    Returns
    -------
    xy : ndarray
        The [x, y] cartesian coordinates of the lattice points.

    M : ndarray
        The [u, v] fractional coordinates of the lattice points.

    """

    M = np.array(
        [[i, j]
         for i in range(-max_order, max_order+1)
         for j in range(-max_order, max_order+1)
         if (np.abs(i) >= min_order or np.abs(j) >= min_order)]
    )

    xy = M @ basis + origin

    if xlim is not None:
        xlimited = ((xy[:, 0] > xlim[0]) & (xy[:, 0] < xlim[1]))
        xy = xy[xlimited]
        M = M[xlimited]
    if ylim is not None:
        ylimited = ((xy[:, 1] > ylim[0]) & (xy[:, 1] < ylim[1]))
        xy = xy[ylimited]
        M = M[ylimited]

    return xy, M


def register_lattice_to_peaks(
        basis,
        origin,
        xy_peaks,
        basis1_order=1,
        basis2_order=1,
        fix_origin=False,
        min_order=0,
        max_order=10,
):
    """
    Find peaks close to a an lattice defined by approximate basis vectors and
    then refine the basis vectors to best match the selected peaks.

    Parameters
    ----------
    basis : array of shape (2,2)
        The array of approximate basis vectors. Each vector is an array row.

    origin : array of shape (2,)
        The (initial) origin point.

    xy_peaks : array of shape (n,2)
        The list of all peaks from which to determine those that best match the
        initial lattice.

    basis1_order, basis2_order : ints
        The order of the peaks at the corresponding basis vectors.
        Default: 1

    fix_origin : bool
        Whether the origin should be fixed or allowed to vary when refining the
        lattice to best fit the peak positions.

    min_order : int
        The minimum order of points allowed in the lattice. e.g. if 2, first
        order points are excluded from the search and fitting.
        Default: 1

    max_order : int
        The maximum order of points allowed in the lattice. e.g if 5, only
        points up to order 5 are included in the search and fitting.

    Returns
    -------
    basis_vects : array of shape (2,2)
        The refined basis vectors.

    origin : array of shape (2,)
        The refined origin or original origin if fix_origin==True.

    lattice : pandas.DataFrame object
        The dataframe of selected peak positions, their corresponding lattice
        indices, and positions of the refined lattice points.

    """

    basis = basis / np.array([basis1_order, basis2_order], ndmin=2).T

    xy_ref, lattice_indices = make_lattice(
        basis, origin, min_order=min_order, max_order=max_order,
    )

    # Match lattice points to peaks; make DataFrame
    vects = np.array([xy_peaks - xy_ for xy_ in xy_ref])
    inds = np.argmin(norm(vects, axis=2), axis=1)

    lattice = pd.DataFrame({
        'h': lattice_indices[:, 0],
        'k': lattice_indices[:, 1],
        'x_ref': xy_ref[:, 0],
        'y_ref': xy_ref[:, 1],
        'x_fit': [xy_peaks[ind, 0] for ind in inds],
        'y_fit': [xy_peaks[ind, 1] for ind in inds],
        'mask_ind': inds
    })

    # Remove peaks that are too far from initial lattice points
    toler = max(0.05*np.min(norm(basis, axis=1)), 2)
    lattice = lattice[norm(
        lattice.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
        - lattice.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
        axis=1
    ) < toler
    ].reset_index(drop=True)

    # Refine the basis vectors
    M = lattice.loc[:, 'h':'k'].to_numpy(dtype=float)
    xy = lattice.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)

    p0 = np.concatenate((basis.flatten(), origin))

    params = fit_lattice(p0, xy, M, fix_origin=fix_origin)

    # Update lattice basis and coordinates
    basis_vects = params[:4].reshape((2, 2))

    if not fix_origin:
        origin = params[4:]

    lattice[['x_ref', 'y_ref']] = (
        lattice.loc[:, 'h':'k'].to_numpy(dtype=float)
        @ basis_vects
        + origin
    )

    return basis_vects, origin, lattice


def plot_basis(
        image,
        basis_vects,
        origin,
        lattice=None,
        return_fig=False,
        scaling=None,
        vmin=None,
        vmax=None,
):
    """
    Plot a lattice and its basis vectors on the corresponding image.

    Parameters
    ----------
    image : 2d array
        The underlying image as a numpy array.

    basis_vects : array of shape (2,2)
        The array of basis vectors. Each vector is an array row.

    origin : array of shape (2,)
        The (initial) origin point.

    lattice : pandas.DataFrame object
        The dataframe of selected peak positions, their corresponding lattice
        indices, and positions of the refined lattice points.

    return_fig: bool
        Whether to return the fig and axes objects so they can be modified.
        Default: False

    vmin, vmax : scalars
        The min and max values for the image display colormap range.

    Returns
    -------
    fig, axs : figure and axes objects (optional)
        The resulting matplotlib figure and axes objects for possible
        modification by the user.

    """

    fig, ax = plt.subplots(figsize=(10, 10))
    quickplot(
        image,
        cmap='gist_gray',
        scaling=scaling,
        figax=ax,
        vmax=vmax,
        vmin=vmin,
    )

    if lattice is not None:
        ax.scatter(
            lattice.loc[:, 'x_ref'].to_numpy(dtype=float),
            lattice.loc[:, 'y_ref'].to_numpy(dtype=float),
            marker='+',
            c='red'
        )
    ax.scatter(origin[0], origin[1], marker='+', c='white')

    ax.arrow(
        origin[0],
        origin[1],
        basis_vects[0, 0],
        basis_vects[0, 1],
        fc='red',
        ec='white',
        width=0.1,
        length_includes_head=True,
        head_width=2,
        head_length=3,
        label='1',
    )
    ax.arrow(
        origin[0],
        origin[1],
        basis_vects[1, 0],
        basis_vects[1, 1],
        fc='green',
        ec='white',
        width=0.1,
        length_includes_head=True,
        head_width=2,
        head_length=3,
        label='2'
    )

    if return_fig:
        return fig, ax


def disp_vect_sum_squares(p0, xy, M, weights=None):
    """Objective function for 'fit_lattice()'.

    Parameters
    ----------
    p0 : list-like of shape (6,)
        The current basis guess of the form: [a1x, a1y, a2x, a2y, x0, y0].
        Where [a1x, a1y] is the first basis vector, [a2x, a2y] is the second
        basis vector and [x0, y0] is the origin.

    xy : array-like of shape (n, 2)
        The array of measured [x, y] lattice coordinates.

    M : array-like of shape (n, 2)
        The array of fractional coorinates or reciprocal lattice indice
        corresponding to the xy coordinates. Rows correspond to [u, v] or
        [h, k] coordinates depending on whether the data is in real or
        reciproal space.

    Returns
    -------
    sum_sq : scalar
        The sum of squared errors given p0.

    """

    dir_struct_matrix = p0[:-2].reshape((-1, 2))
    origin = p0[-2:]
    if weights is None:
        weights = 1

    err_xy = norm(xy - (M @ dir_struct_matrix + origin), axis=1)
    sum_sq = np.sum((err_xy * weights)**2)

    return sum_sq


def fit_lattice(p0, xy, M, fix_origin=False, weights=None):
    """Find the best fit of a rigid lattice to a set of points.

    Parameters
    ----------
    p0 : list-like of shape (6,)
        The initial basis guess of the form: [a1x, a1y, a2x, a2y, x0, y0].
        Where [a1x, a1y] is the first basis vector, [a2x, a2y] is the second
        basis vector and [x0, y0] is the origin.

    xy : array-like of shape (n, 2)
        The array of measured [x, y] lattice coordinates.

    M : array-like of shape (n, 2)
        The array of fractional coorinates or reciprocal lattice indices
        corresponding to the xy coordinates. Rows correspond to [u, v] or
        [h, k] coordinates depending on whether the data is in real or
        reciproal space.

    fix_origin : bool
        Whether to fix the origin (if True) or allow it to be refined
        (if False). Generally, should be false unless data is from an FFT,
        then the origin is known and should be fixed.
        Default: False

    Returns
    -------
    params : list-like of shape (6,)
        The refined basis, using the same form as p0.

    """

    p0 = np.array(p0).flatten()
    x0y0 = p0[-2:]

    if fix_origin:
        params = (lstsq(M, xy - x0y0, rcond=-1)[0]).flatten()

    else:
        params = minimize(
            disp_vect_sum_squares,
            p0,
            args=(xy, M, weights),
            method='BFGS',
        ).x

    return params


def measure_lattice_from_peaks(
        peaks,
        basis,
        origin,
        toler,
        max_order,
        shape,
        fix_origin=False,
):
    """
    Match nearest peaks to a reference lattice, discarding unmatched peaks
    and missing lattice points.


    """

    if fix_origin:
        min_order = 1
    else:
        min_order = 0

    # basis_new = basis
    # origin_new = origin

    """match first order peaks & update basis"""
    xy, M = make_lattice(basis,  origin, max_order=1, min_order=min_order)

    # Distance matrix: axis=0 -> lattice; axis=1 -> peaks.
    # Find lattice points with peak nearby
    dists = norm(np.array([peaks - i for i in xy]), axis=2)
    mininds_latt = np.argmin(dists, axis=1)
    matches = np.array([[i, j] for i, j in enumerate(mininds_latt)
                        if dists[i, j] < toler])
    M = M[matches[:, 0]]
    # xy = xy[matches[:, 0]]
    peaks_1 = peaks[matches[:, 1]]

    # Update basis and origin
    params = fit_lattice(
        np.concatenate((basis.flatten(), origin)),
        peaks_1,
        M,
        fix_origin=fix_origin,
    )

    basis_new = params[:4].reshape((2, 2))
    if not fix_origin:
        origin_new = params[4:]
    else:
        origin_new = origin

    """match ALL peaks & update basis"""
    if max_order is None:
        diag = np.sum(np.array(shape)**2)**0.5
        max_order = np.ceil(diag / np.min(norm(basis_new, axis=1))).astype(int)

    xy, M = make_lattice(
        basis_new,
        origin_new,
        max_order=max_order,
        min_order=min_order,
        xlim=[0, shape[1]],
        ylim=[0, shape[0]],
    )

    # Distance matix
    dists = norm(np.array([peaks - i for i in xy]), axis=2)
    mininds_latt = np.argmin(dists, axis=1)
    matches = np.array([[i, j] for i, j in enumerate(mininds_latt)
                        if dists[i, j] < toler])
    M = M[matches[:, 0]]
    xy = xy[matches[:, 0]]
    peaks_matched = peaks[matches[:, 1]]

    # Update basis and origin
    params = fit_lattice(
        np.concatenate((basis_new.flatten(), origin_new)),
        peaks_matched,
        M,
        fix_origin=fix_origin,
    )

    if not fix_origin:
        basis_new = params.reshape((3, 2))
    else:
        basis_new = params.reshape((2, 2))
        basis_new = np.concatenate((basis_new, origin), axis=0)

    return basis_new


def rotate_2d_basis(basis, theta):
    """Actively rotate a set (or sets) of 2d basis vectors about the
    out-of-plane axis.

    Parameters
    ----------
    basis : ndarray
        The basis vectors or sets of basis vectors. Last two dimensions must
        be the basis vectors as row vectors.

    theta : scalar
        Rotation angle. Basis will be actively rotated by this angle.

    Returns
    -------
    params : list-like of shape (6,)
        The refined basis, using the same form as p0.

    """

    if len(basis.shape) == 2:
        basis = basis[None, None, ...]
    elif len(basis.shape) == 3:
        basis = basis[None, ...]

    theta = np.radians(theta)
    rot_mat = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    # Apply rotation
    rotated_basis = np.array([[a @ rot_mat for a in row] for row in basis])

    return rotated_basis.squeeze()
