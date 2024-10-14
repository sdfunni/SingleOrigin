"""SingleOrigin is a module for atomic column position finding intended for
    high resolution scanning transmission electron microscope images.
    Copyright (C) 2023  Stephen D. Funni

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

import numpy as np
from numpy.linalg import norm

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib_scalebar.scalebar import ScaleBar

# %%


def pick_points(
        image,
        n_picks,
        xy_peaks,
        origin=None,
        graphical_picking=True,
        window_size=None,
        timeout=None,
        cmap='gray',
        vmin=None,
        vmax=None,
        quickplot_kwargs={},
):
    """
    Plot points on an image for selection by index or graphical picking.

    Used for selecting coordinate points (x,y positions) within an image for
    subsequent analysis tasks. Typically, these are data peaks that have been
    detected and must be located accurately by a fitting algorithm.

    Parameters
    ----------
    image : 2d array
        The underlying image as a numpy array.

    n_picks : int
        The number of points to be chosen.

    xy_peaks : array of shape (m,2)
        The x,y coordinates of the full list of points from which to select the
        picks.

    origin : 2-tuple of scalars or None
        The origin point. This will be plotted for reference, but has no other
        function. If None, no origin is plotted.
        Default: None

    graphical_picking: bool
        Whether to allow graphical picking with mouse clicks (if True). If
        False, points are plotted with index labels according to their row
        index in the xy_peaks array. This allows for subsequent programatic
        selection instead of graphical picking.
        Default: True

    window_size : scalar
        The size of the region to plot, centered around the middle of the
        image. Useful to zoom in to the center of an FFT when most of the
        information is contained in a small area.

    timeout : scalar or None
        Number of seconds to allow for graphical picking before closing the
        plot window. If None, will not time out.
        Default: None

    Returns
    -------
    picks_xy : array of shape (n,2)
        The x,y coordinates of the 'n' chosen data points.

    """

    h, w = image.shape
    U = np.min([int(h/2), int(w/2)])
    fig, ax = plt.subplots(figsize=(10, 10))
    if window_size is not None:
        ax.set_ylim(bottom=U+window_size/2, top=U-window_size/2)
        ax.set_xlim(left=U-window_size/2, right=U+window_size/2)

    quickplot(image, cmap=cmap, vmin=vmin, vmax=vmax, figax=ax,
              **quickplot_kwargs)

    # ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(xy_peaks[:, 0], xy_peaks[:, 1], c='red', s=8)
    if origin is not None:
        ax.scatter(origin[0], origin[1], c='white', s=16)

    if graphical_picking:
        if timeout is None:
            timeout = 0
        picks_xy = np.array(plt.ginput(n_picks, timeout=timeout))

        vects = np.array([xy_peaks - i for i in picks_xy])
        inds = np.argmin(norm(vects, axis=2), axis=1)
        picks_xy = xy_peaks[inds, :]

        plt.close('all')

    else:
        inds = np.arange(xy_peaks.shape[0])
        for i, ind in enumerate(inds):
            ax.annotate(ind, (xy_peaks[i, 0], xy_peaks[i, 1]), color='white')

        picks_xy = None

    return picks_xy


def quickplot(
        im,
        cmap='inferno',
        figsize=(6, 6),
        hide_ticks=True,
        pixel_size=None,
        pixel_unit=None,
        scalebar_len=None,
        figax=False,
        tight_layout=True,
        scaling=None,
        vmin=None,
        vmax=None,
        zerocentered=False,
):
    """Convienience image plotting function.

    Parameters
    ----------
    im : 2D array
        The image.

    cmap : str
        The colormap to use.
        Default: 'inferno'

    figsize : 2-tuple
        Figsize to use.
        Default: (6, 6)

    hide_ticks : bool
        Whether to hide tickmarks on edges of image plot.
        Default: True

    pixel_size : scalar
        The physical size of pixels in the image. Used to plot a scalabar.
        If None, no scalbar is plotted.
        Default: None

    pixel_unit : str
        The unit length of the pixel size calibration. If pixel size passed,
        but pixel_unit is None, will use "a.u.".
        Default: None

    scalebar_len : scalar
        Fix the size of plotted scalebar. If None, optimal size found by
        function.
        Default: None

    figax : bool or matplotlib.Axes object
        Whether to return the figure and axes objects for modification by user.
        OR the Axes object to plot the spectrum into.
        Default: False

    tight_layout : bool
        Whether to apply tight layout to the plot.
        Default: True

    scaling : float or str
        Linear (None), log ('log') or power law (float on interval [0, 1])
        scaling of the image intensity.
        Default: None

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
        The resulting figure and axes objects for possible further
        modifications.

    """

    if zerocentered:
        max_ = np.nanmax(np.abs(im))
        vmin = -max_
        vmax = max_

    if scaling == 'log':
        # if vmin is None:
        #     vmin = 0
        norm = LogNorm(vmin=vmin, vmax=vmax)

        im -= np.min(im) - 1

    elif isinstance(scaling, (float, int)):
        norm = PowerNorm(scaling, vmin=vmin, vmax=vmax)
    else:
        norm = None

    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad(color='black')

    if isinstance(figax, bool):
        fig, ax = plt.subplots(1, figsize=figsize, tight_layout=tight_layout)

    else:
        ax = figax

    if scaling is not None:
        ax.imshow(im, cmap=cmap, norm=norm)
    else:
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if pixel_size is not None:
        if pixel_unit is None:
            pixel_unit = 'a.u.'
        if scalebar_len is None:
            sb_sizes = np.array([10**dec * int_ for dec in range(-1, 5)
                                 for int_ in [1, 2, 4, 5]])
            fov = np.max(im.shape) * pixel_size
            scalebar_len = sb_sizes[np.argmax(sb_sizes > fov*0.1)]
            if scalebar_len >= 1:
                scalebar_len = int(scalebar_len)

        scalebar = ScaleBar(
            pixel_size,
            pixel_unit,
            font_properties={'size': 12},
            pad=0.3,
            border_pad=0.6,
            box_color='white',
            height_fraction=0.02,
            color='black',
            location='lower right',
            fixed_value=scalebar_len,
        )
        ax.add_artist(scalebar)

    if figax is True:
        return fig, ax


def plot_vDetector(
        detector,
        figax,
        contour_kwargs_dict={}):
    """
    Plot a virtual detector outline.

    Parameters
    ----------
    detector : 2d array
        Binary array representing the detector (i.e. 1 where the detector is,
        0 elsewhere).

    figax : matplotlib.Axes object
        The Axes object to plot the detector outline into.

    contour_dict : dict


    Returns
    -------
    None.

    """

    coutour_dict_default = {
        'colors': 'black',
        'linewidths': 0.5,
    }

    coutour_dict_default.update(contour_kwargs_dict)

    Zd = detector % 2

    resolution = 25

    xd = np.linspace(0, Zd.shape[1], Zd.shape[1]*resolution)
    yd = np.linspace(0, Zd.shape[0], Zd.shape[0]*resolution)
    Xd, Yd = np.meshgrid(xd[:-1], yd[:-1])

    Zd = Zd[Yd.astype(int), Xd.astype(int)]

    figax.contour(
        Xd-0.5,
        Yd-0.5,
        Zd,
        [0.5],
        **coutour_dict_default,
    )
