import os
import copy
from pathlib import Path

import numpy as np
from numpy import sin, cos
from numpy.linalg import norm

from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_dilation,
)

from scipy.signal import convolve2d
from scipy.fft import (
    fft2,
    ifft2,
    fftshift,
)

from skimage.draw import polygon, polygon2mask
from skimage.morphology import erosion

import matplotlib

from PIL import Image, ImageFont, ImageDraw
from tifffile import imwrite

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

from SingleOrigin.crystalmath import rotation_angle_bt_vectors

pkg_dir, _ = os.path.split(__file__)

# %%


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


def burn_in_scalebar(image, pixel_size, units, loc='br'):
    """Burn a scalebar into an image array

    Parameters
    ----------
   image : array like
        The image.

    pixel_size : float
        The pixel size in units of argument units.

    units : str
        The pixel size of the image or diffraction pattern: e.g. 'nm', 'um',
        '1/nm'.

    loc : str
        'tl', 'tr', 'bl', 'br'
        Default: 'br'

    Returns
    -------
    image : array like
        The image array with scalebar burned in scalebar.

    """

    img = copy.deepcopy(image)
    # Get image and scalebar parameters
    sb_sizes = np.array([10**dec * int_ for dec in range(-1, 5)
                         for int_ in [1, 2, 4, 5]])

    fact = 1
    h, w = img.shape
    size = np.min(img.shape)
    if len(np.unique(img.shape)) > 1:
        size *= 2

    font_fact = size / 1024
    pad = int(0.02 * size)

    fov = size * pixel_size
    sbar = sb_sizes[np.argmax(sb_sizes > fov*0.1)]
    if sbar > 1:
        sbar = int(sbar)
    # print(sbar)
    sbar_pixels = int(sbar/pixel_size)

    font_size = int(50 * font_fact)
    bbox_height = int(2.2*pad + 0.02*h + font_size)

    # Set units
    if units == 'nm':
        if np.log10(sbar) < 3:
            units = 'nm'
        elif np.log10(sbar) < 6:
            units = 'um'
            fact = 1e-3

    elif units == '1/nm':
        # units = r'nm$^{-1}$'
        units = '1/nm'

    elif units == 'um':
        units = 'um'

    # Get font for PIL
    font_path = os.path.join(
        matplotlib.__path__[0],
        "mpl-data/fonts/ttf/DejaVuSans.ttf",
    )
    font = ImageFont.truetype(
        font_path,
        size=font_size,
        # layout_engine=ImageFont.Layout.RAQM,
    )

    # Get box width to ensure contains string and scalebar
    img_obj = Image.fromarray(img)
    draw = ImageDraw.Draw(img_obj)
    text = f'{sbar * fact} {units}'
    text_length = draw.textlength(
        text,
        font=font,
    )

    bbox_width = np.max([int(sbar_pixels + 2*pad), text_length + 2*pad])

    bbox = np.zeros((2, 2), dtype=int)

    if loc[0] == 't':
        bbox[0, 0] = int(pad)
        bbox[0, 1] = int(pad + bbox_height)

    if loc[0] == 'b':
        bbox[0, 0] = int(h - pad - bbox_height)
        bbox[0, 1] = int(h - pad)

    if loc[1] == 'l':
        bbox[1, 0] = int(pad)
        bbox[1, 1] = int(pad + bbox_width)

    if loc[1] == 'r':
        bbox[1, 0] = int(w - pad - bbox_width)
        bbox[1, 1] = int(w - pad)

    cent_width = int(np.mean(bbox[1]))

    # Make scalebar box
    img[bbox[0, 0]:bbox[0, 1], bbox[1, 0]:bbox[1, 1]] = 1

    # Add scalebar
    img[bbox[0, 0] + pad:bbox[0, 0] + 2*pad,
        int(cent_width - sbar_pixels/2):
        int(cent_width - sbar_pixels/2) + sbar_pixels
        ] = 0

    img_obj = Image.fromarray(img)
    draw = ImageDraw.Draw(img_obj)
    draw.text(
        (cent_width, int(bbox[0, 1] - pad)),
        text,
        0,
        font=font,
        anchor='mb',
    )

    return np.array(img_obj)


def save_tif_image(image, folder, name, bits=16):
    """
    Save image in .tif format

    Parameters
    ----------
    image : array like
        The image.

    path : str
        Valid path for saving image.

    bits : int
        TIFF bit depth to use. Must be 8, 16, or 32.

    Returns
    -------
    None.

    """

    if bits == 8:
        type_ = np.uint8
    elif bits == 16:
        type_ = np.uint16
    elif bits == 32:
        type_ = np.uint32
    else:
        raise Exception(
            'bits must be 8, 16 or 32'
        )

    if name[-4:] != '.tif':
        name += '.tif'

    path = Path(os.path.join(folder, name))

    image = (image_norm(image) * (2**bits - 1)).astype(type_)
    imwrite(
        path,
        image,
        photometric='minisblack'
    )


def save_fig(fig, folder, name, dpi=100):
    """
    Save a matplotlib figure as a .tif or .png.

    Parameters
    ----------
    fig : matplotlibl figure object
        The figure.

    folder : str
        Valid file location for saving the figure image.

    name : str
        Desired name for the saved image file. If a file name extension is
        given (e.g. .tif), that file type will be used. Otherwise .png is used.


    dpi : int
        The dots-per-inch for the figure. Size of the figure (fig.figsize) and
        dpi will determine the final file size for storage so choose wisely.
        Default: 100.

    Returns
    -------
    None.

    """

    if not np.isin(name[-4:], ['.tif', '.png']).item():
        name += '.png'
    fig.savefig(
        os.path.join(folder, name),
        bbox_inches='tight',
        pad_inches=0.25,
        dpi=dpi
    )


def image_auto_bc(
    image,
    gauss_smooth=1,
    count_lim_low=None,
    count_lim_high=None,
    log_scale_hist=False,
):
    """
    Adjust the brightness and contrast of an image.

    Parameters
    ----------
    image : array like
        The image.

    gauss_smooth : scalar
        Gaussian filter width for low pass filtering to reduce effects of
        bad pixels. Should be kept low. Set to 0 if no smooting desired.
        Default: 1

    count_lim_low, count_lim_low : int
        The maximum number of pixels to saturate at the upper and lower
        contrast limits as determined by the intensity histogram test.

    log_scale_hist : bool
        If True,uses a log scaled histogram for determining min and max
        intensity limits. Log scaling works best for high dynamic range data
        like spot diffraction patterns. If False, uses linear histogram.
        The applied histogram has 1000 bins.
        Default: False

    Returns
    -------
    image : array like
        The image array with adjusted B&C.

    """
    # LP filter to minimize effects of bad pixels
    image = gaussian_filter(image, gauss_smooth, truncate=2)
    image = image_norm(image)

    if log_scale_hist:
        counts, edges = np.histogram(np.log(image.ravel()+1e-8), bins=100,)
        edges = np.exp(edges)
    else:
        counts, edges = np.histogram(image.ravel(), bins=1000,)

    if count_lim_low is not None:
        cumul_f = np.array(
            [np.sum(counts[:i+1]) for i in range(counts.shape[0])]
        )

        min_ = edges[np.where(cumul_f > count_lim_low)[0][0]]
        image[image < min_] = min_

    if count_lim_high is not None:
        cumul_r = np.array(
            [np.sum(counts[i:]) for i in range(counts.shape[0])]
        )

        max_ = edges[np.where(cumul_r > count_lim_high)[0][-1] + 1]
        image[image > max_] = max_

    image = image_norm(image)

    return image


def bin_data(data, factor):
    """
    Parameters
    ----------
    data : ndarray
        The dataset.
    factor : list
        Binning along each dimension of data.

    Returns
    -------
    image_bin : 2D array
        The binned image.

    """

    dims = list(data.shape)
    new_dims = np.array([
        [dims[i]/factor[i], factor[i]]
        for i in range(len(factor))
    ]).flatten().astype(int)

    crop = np.array([dims[i] % factor[i] for i in range(len(factor))])
    stop = -(crop/2).astype(int)
    start = crop + stop
    stop = [s if s != 0 else dims[i]+1 for i, s in enumerate(stop)]

    indices = tuple([slice(start[i], stop[i]) for i in range(len(dims))])
    data_crop = data[indices]

    data_bin = np.sum(
        data_crop.reshape(new_dims),
        axis=tuple(np.arange(1, len(new_dims), 2, dtype=int))
    )

    return data_bin


def cross_correlation(image1, image2):
    """
    Parameters
    ----------
    image1, image2: 2D arrays of the same size
        The image.

    Returns
    -------
    im_ncc : 2D arrays of the same size as input images
        The normalized cross correlation

    """

    imCC = np.abs(ifft2((fft2(image1) * np.conj(fft2(image2)))))

    return imCC


def radial_profile(image, center=None):
    """
    Calculate the average radial profile of an image about a point.

    Parameters
    ----------
    image : 2D array
        The image. May have NaNs: these will not be counted or weighted when
        normalizing the bins.
    center : 2-list
        The [x, y] center point about which to calculate the profile. Specify
        in image coordinates of [col, row] with down/right being positive.
        If None, uses the middle of the image.
        Default: None.

    Returns
    -------
    dists : 1D array
        The distance corresponding to each bin in the radial profile.
    mean_vals : 1D array
        The mean value of the imager at the corresponding distance in dists.

    """

    if center is None:
        center = [int(image.shape[1]/2),
                  int(image.shape[0]/2)]

    r = getAllPixelDists(image.shape, center)
    r = r.astype(int)
    ids = np.rint(r).astype(int).ravel()
    w = image.ravel()

    keep = ~np.isnan(w)

    ids = ids[keep]
    w = w[keep]

    count = np.bincount(ids)

    dists = np.unique(r)

    mean_data = (np.bincount(ids, w)/count)[count != 0]

    return dists, mean_data


def bandpass(image, highpass=0, lowpass=0):
    """
    High and/or low pass filter an image using Gaussian filters.

    Parameters
    ----------
    image : 2D array
        The image.
    highpass : scalar
        The width of the Gaussian highpass filter.
    lowpass : scalar
        The width of the Gaussian lowpass filter.

    Returns
    -------
    imfiltered : 2D array
        The filtered image.

    """

    imfiltered = copy.deepcopy(image)

    if lowpass > 0:
        imfiltered = gaussian_filter(image, lowpass)

    if highpass > 0:
        imfiltered = imfiltered - gaussian_filter(image, highpass)

    imfiltered = image_norm(imfiltered)

    return imfiltered


def getAllPixelDists(shape, origin):
    """
    Get euclidian distance from an origin point for each pixel in a 2D array.

    Parameters
    ----------
    shape : 2-tuple
        The array shape.
    origin : 2-list
        The x, y coordinates of the origin point.

    Returns
    -------
    r : 2D array of shape shape
        The distances.
    """

    y, x = np.indices(shape)
    r = np.sqrt((x - origin[0])**2 + (y - origin[1])**2)

    return r


def get_mask_polygon(
        image,
        vertices=None,
        buffer=0,
        invert=False,
        show_mask=True,
        return_vertices=False
):
    """Get a binary mask for an arbitrary shaped polygon.

    Parameters
    ----------
    data : 2d array or 2-tuple
        The image to be used for graphical picking of the polygon vertices or,
        if a tuple, the image shape.

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

    if isinstance(image, tuple):
        image = np.zeros(image)

    if vertices is not None:
        vertices = np.fliplr(np.array(vertices))
    if vertices is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(image, cmap='gist_gray')
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

    mask = polygon2mask(image.shape, vertices)

    if buffer:
        mask = erosion(
            mask,
            footprint=np.ones((3, 3))
        )

    if invert:
        mask = np.where(mask == 1, 0, 1)

    if show_mask:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gist_gray')

        ax.imshow(
            mask,
            alpha=0.2,
            cmap='Reds'
        )

    if return_vertices:
        vertices = np.fliplr(vertices)
        return mask, vertices
    else:
        return mask


def line_profile(
        data,
        int_width,
        image=None,
        scandims=None,
        signaldims=None,
        start=None,
        vector=None,
):
    """
    Integrate diffraction patterns along a line in scan space with a given
    integration width. Produces a "profile" of diffraction patterns. Line is
    chosen by graphical picking on a virtual image.

    Parameters
    ----------
    data : ndarray of n+2 dimensions
        The input nD STEM data

    int_width : scalar
        The width perpendicular to the line over which to integrate diffraction
        patterns.

    vdet_args : dict
        Arguments to be passed to vAnnularDetImage to generate an virtual
        image for graphical picking. Only applies to 4D data.

    image : array or None
        The image to diplay for selecting the integration line. If None, a
        default method is applied to create a virtual image.
        Default: None.

    scandims : tuple or None
        The dimensions corresponding to the scan.
        Default: None.

    signaldims : tuple or None
        The dimensions corresponding to the signal (e.g. diffraction, specta).
        Default: None.

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

    d = int_width / 2
    n_picks = 2

    if len(data.shape) == 2:
        if image is None:
            image = data
        data = data[..., None]
        scandims = data.shape[:2]
        signaldims = (1,)
        print(data.shape)

    elif len(data.shape) == 3:
        if None in [scandims, signaldims]:
            raise Exception(
                'Must specify scandims and signaldims for 3D data.'
            )
        elif len(scandims) == 1:
            raise Exception('This is already a line profile...')
        else:
            scandims = data.shape[:2]
            signaldims = (2,),

        if image is None:
            image = np.sum(data, axis=2)

    elif len(data.shape) == 4:
        scandims = data.shape[:2]
        signaldims = data.shape[2:]
        if image is None:
            image = np.max(data, axis=(2, 3))

    else:
        raise Exception('This dimensionality is not supported.')

    if ((start is None) or (vector is None)):
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(image, cmap='inferno')

        ax.set_xticks([])
        ax.set_yticks([])

        picks_xy = np.array(plt.ginput(n_picks, timeout=60))

        start = picks_xy[0]
        vector = picks_xy[1] - picks_xy[0]

        xends = [start[0], start[0] + vector[0]]
        yends = [start[1], start[1] + vector[1]]
        plt.plot(xends, yends)

    unit_vect = vector / norm(vector)

    xy = np.fliplr(np.indices(scandims).reshape(2, -1).T)
    z = data.reshape(-1, *signaldims)

    # Get all pixels within int_width along the vector
    take = np.where(np.abs(np.cross(unit_vect[None, :], xy - start, axis=1))
                    < d)[0]

    xy = xy[take, :]
    z = z[take, :]

    pixel_bins = ((xy - start) @ unit_vect.T).astype(int)

    max_bin = np.ceil(norm(vector)) + 1

    bins = np.arange(0, max_bin)

    line_integration = np.array([
        np.mean(z[np.where(pixel_bins == bin_)[0], ...], axis=0)
        for bin_ in bins
    ])

    return line_integration, start, vector


def nearestKDE_2D(coords, xlim, ylim, d, weights=None, return_binedges=False):
    """
    Apply nearest neighbor KDE to 2D coordinate set with optional weights.

    Parameters
    ----------
    coords : array of scalars (n, 2)
        the x, y coordinates of the data points.

    xlim : two-tuple of scalars
        The x limits of the resulting density estimate.

    ylim : two-tuple of scalars
        The y limits of the resulting density estimate.

    d : scalar
        The bin width.

    weights : array of scalars (n,) or None
        The values by which to weight each coordinates for the KDE. (e.g.
        the image intensity value of a pixel). If None, all data points are
        considered equal.

    return_binedges : bool
        Whether to return arrays with x & y coordinates of the bin edges.
        Default: False

    Returns
    -------
    H : array of scalars (h, w)
        The density estimate as a 2D histogram with shape defined by the
        specified xlim, ylim and d arguments.

    xedges, yedges : array of scalars with shapes (w+1,) and (h+1,)
        The bin edges of the pixels in H.

    """

    # Get bin spacing
    xedges = np.arange(xlim[0], xlim[1], d)
    yedges = np.arange(ylim[0], ylim[1], d)

    # Find edge closest to 0 and shift edges so (0,0) is exactly at the center
    # of a pixel
    x_min_ind = np.argmin(np.abs(xedges))
    y_min_ind = np.argmin(np.abs(yedges))
    xedges -= xedges[x_min_ind] + d/2
    yedges -= yedges[y_min_ind] + d/2

    # Remove vectors that fall out of the desired field of view
    coords = coords[(coords[:, 0] > np.min(xedges)) &
                    (coords[:, 0] < np.max(xedges)) &
                    (coords[:, 1] > np.min(yedges)) &
                    (coords[:, 1] < np.max(yedges))]

    H, _, _ = np.histogram2d(
        coords[:, 1],
        coords[:, 0],
        bins=[yedges, xedges]
    )

    if return_binedges:
        return H, xedges, yedges
    else:
        return H


def linearKDE_2D(
        coords,
        xlim,
        ylim,
        d,
        r=1,
        weights=None,
        return_binedges=False
):
    """
    Apply linear KDE to 2D coordinate set with optional weights.

    Parameters
    ----------
    coords : array of scalars (n, 2)
        the x, y coordinates of the data points.

    xlim : two-tuple of scalars
        The x limits of the resulting density estimate.

    ylim : two-tuple of scalars
        The y limits of the resulting density estimate.

    d : positive scalar
        The bin width.

    r : int
        The bandwidth of the KDE.

    w : int
        The width of the kernal in integer number of bins.

    weights : array of scalars (n,) or None
        The values by which to weight each coordinates for the KDE. (e.g.
        the image intensity value of a pixel). If None, all data points are
        considered equal.

    return_binedges : bool
        Whether to return arrays with x & y coordinates of the bin edges.
        Default: False

    Returns
    -------
    H : array of scalars (h, w)
        The density estimate as a 2D histogram with shape defined by the
        specified xlim, ylim and d arguments.

    xedges, yedges : arrays of scalars with shapes (w+1,) and (h+1,)
        The bin edges of the pixels in H.

    """

    # Get bin spacing
    xedges = np.arange(xlim[0], xlim[1], d).astype(float)
    yedges = np.arange(ylim[0], ylim[1], d).astype(float)

    # Find edge closest to 0 and shift edges so (0,0) is exactly at the center
    # of a pixel
    x_min_ind = np.argmin(np.abs(xedges))
    y_min_ind = np.argmin(np.abs(yedges))
    xedges -= xedges[x_min_ind] + d/2
    yedges -= yedges[y_min_ind] + d/2

    # Get bin centers
    xcents = xedges[:-1] + d/2
    ycents = yedges[:-1] + d/2

    H = np.zeros((ycents.shape[0], xcents.shape[0]))

    # If intensity weights were not passed, get equal weighting array
    if weights is None:
        weights = np.ones(coords.shape[0])

    # Get relative pixel shift values for binning
    xs = [i for i in range(-r, r, 1)]
    ys = [i for i in range(-r, r, 1)]

    # Get reference pixel for each data point
    xyC = np.ceil(coords / d) * d

    # Calculate eash pixel shifted histogram and sum together
    for j in ys:
        for i in xs:
            # Find bin indices for the current shift
            xyB = xyC + np.array([[i*d, j*d]])

            # Find distance weighting for high sampling rate:
            # Method results in total density per data point deviating slightly
            # from 1, but close with sufficient sampling (i.e. r >= 2)
            # This method is a KDE using a linear kernel with euclidian
            # distance metric.
            if r > 1:
                dW = 3/np.pi * (1 - norm(xyB - coords, axis=1) / (d*r)) / r**2

            # Find distance weighting if low sampling (i.e. r == 1):
            # Method is effectively a reverse bilineaer interpolation.
            # That is, it distributes the density from each datapoint over four
            # nearest neighbor pixels using bilinear weighting. This ensures
            # the density contribution from each data point is exactly 1.
            elif r == 1:
                dW = np.prod(1 - np.abs(xyB - coords) / (d*r), axis=1) / r**2

            else:
                raise Exception(
                    "'r' must be >= 1"
                )

            H += np.histogram2d(
                xyB[:, 1], xyB[:, 0],
                bins=[yedges, xedges],
                weights=dW * weights
            )[0]

    if return_binedges:
        return H, xedges, yedges
    else:
        return H


def get_avg_cell(
        image,
        origin,
        basis_vects,
        M,
        upsample=1,
):
    """
    Get the average repeating cell in an image based on a registered lattice.

    Parameters
    ----------
    image : 2d array
        The image containing repeated cells.

    origin : array of shape (2,)
        The (x, y) coordinates of the lattice origin.

    basis_vects : array of shape (2, 2)
        Matrix containing the lattice basis as row vectors.

    M : array of shape (n, 2)
        Array containing fractional coordinates (u, v) of the corner of each
        unit cell to be included in the average.

    upsample : scalar
        The upsampling factor to be applied by the KDE relative to the original
        pixel size. Should probably be an integer >= 1.
        Default : 1 (i.e. no upsampling)

    Returns
    -------
    avg_cell : 2d array
        The resulting average cell.

    """

    a1 = basis_vects[0]
    a2 = basis_vects[1]

    a1unit = a1 / norm(a1)
    a2unit = a2 / norm(a2)

    xy_cells = M @ basis_vects + origin

    # Get values for each unit cell, subtracting origin coordinates
    x = []
    y = []
    z = []

    for x0, y0 in xy_cells:
        xverts = [
            x0 - 1.5*a1unit[0] - 1.5*a2unit[0],
            x0 + a1[0] + 1.5*a1unit[0] - 1.5*a2unit[0],
            x0 + a1[0] + 1.5*a1unit[0] + a2[0] + 1.5*a2unit[0],
            x0 + a2[0] + 1.5*a2unit[0] - 1.5*a1unit[0]
        ]

        yverts = [
            y0 - 1.5*a1unit[1] - 1.5*a2unit[1],
            y0 + a1[1] + 1.5*a1unit[1] - 1.5*a2unit[1],
            y0 + a1[1] + 1.5*a1unit[1] + a2[1] + 1.5*a2unit[1],
            y0 + a2[1] + 1.5*a2unit[1] - 1.5*a1unit[1]
        ]

        yind, xind = polygon(yverts, xverts, image.shape)

        imint = image[yind, xind]

        unmasked_data = np.nonzero(imint)
        imint = np.take(imint, unmasked_data).flatten()
        xind = np.take(xind, unmasked_data
                       ).flatten() - x0
        yind = np.take(yind, unmasked_data
                       ).flatten() - y0

        z += imint.tolist()
        x += xind.tolist()
        y += yind.tolist()

    # Rotate coordinates so a1 is horizontal
    xy = np.vstack([x, y]).T

    theta = rotation_angle_bt_vectors([1, 0], a1)

    xy = rotate_xy(xy, theta, [0, 0])

    a1 = rotate_xy(a1, theta, [0, 0]).squeeze()

    a2 = rotate_xy(a2, theta, [0, 0]).squeeze()

    dsm = np.vstack([a1, a2])

    x = xy[:, 0]
    y = xy[:, 1]

    h = np.max([dsm[:, 1]]) - np.min(dsm[:, 1]) + 1
    w = np.max([dsm[:, 0]]) - np.min(dsm[:, 0]) + 1

    data_coords = np.vstack([x, y]).T

    avg_cell = linearKDE_2D(
        data_coords,
        xlim=(np.min(dsm[:, 0]), np.min(dsm[:, 0]) + w),
        ylim=(np.min(dsm[:, 1]), np.min(dsm[:, 1]) + h),
        d=1/upsample,
        r=upsample,
        weights=z)

    return avg_cell


def fast_rotate_90deg(image, angle):
    """Rotate images by multiples of 90 degrees. Faster than
    scipy.ndimage.rotate().

    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.

    angle : scalar
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


def rotation_matrix(angle, origin):
    """Get a 2D origin-shifted rotation matrix for an arbitrary rotation.

    Parameters
    ----------
    angle : scalar
        The angle (in degrees) by which to rotate the image.

    origin : 2-tuple
        The point (x, y) about which to preform the rotation.

    Returns
    -------
    tmat : array of shape (3, 3)
        The origin-shifted rotation matrix.
    """
    theta = np.radians(angle)

    tmat = np.array(
        [[cos(theta), sin(theta), 0],
         [-sin(theta), cos(theta), 0],
         [0, 0, 1]]
    )
    tau = np.array(
        [[1, 0, origin[0]],
         [0, 1, origin[1]],
         [0, 0, 1]]
    )
    tau_ = np.array(
        [[1, 0, -origin[0]],
         [0, 1, -origin[1]],
         [0, 0, 1]]
    )
    tmat = tau @ tmat @ tau_

    return tmat


def rotate_xy(coords, angle, origin):
    """Apply a rotation to a set of coordinates.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        The the (x, y) coordinates.

    angle : scalar
        The angle (in degrees) by which to rotate the coordinates.

    origin : 2-tuple
        The point (x, y) about which to preform the rotation.

    Returns
    -------
    coords_ : array of shape (n, 2)
        The rotated coordinates.
    """

    rmat = rotation_matrix(angle, origin)

    coords = np.array(coords, ndmin=2)

    # Append "1" to each coordinate vector for use with origin-shifted
    # transformation matrix.
    coords = np.append(coords, np.ones(coords.shape[:-1] + (1,)), axis=-1)

    # Apply the transformation
    coords_ = (coords @ rmat.T)[..., :-1]

    return coords_


def rotate_image_kde(
        image,
        angle,
        bandwidth=0.5,
        reshape_method='original',
        fill_value=np.nan
):
    """Rotate an image to arbitrary angle & interpolate using KDE.

    Apply an aribtrary image rotation and interpolate new pixel values using
    a linear kernel density estimate (KDE). Check result carefully. May
    produce artifacts.

    Parameters
    ----------
    image : ndarray of shape (h,w)
        The image.

    angle : scalar
        The angle by which to rotate the image.

    bandwidth : scalar
        The bandwidth of the linear kernel in pixels.

    reshape_method : str
        Final frame shape after rotation: 'fulldata', 'original', or
        'only_data'.
        'original' returns an image with the same shape as the input image,
        with corners that were rotated out of the image cut off.
        'fulldata' returns a image with shape >= original shape, but without
        any pixels cut off.
        'only_data' crops the final image to the largest region of the rotated
        frame that contains only data and no NaNs.
        Default : 'original'

    fill_value : scalar
        Value to fill areas in the final output image that are not inside
        the area of the rotated frame. User can specify any value, but likely
        should be 0 or np.nan.
        Default: np.nan.

    Returns
    -------
    image_rot : 2D array
        The rotated image
    """

    h, w = image.shape

    # Rotate about the center of the image
    origin = np.array([w-1, h-1]) / 2

    # Original pixel coordinates
    coords = np.array(np.meshgrid(np.arange(0, w), np.arange(0, h))
                      ).transpose(1, 2, 0).reshape((-1, 2))

    # Transformed pixel coordinates
    coords_ = rotate_xy(coords, angle, origin)

    # Flattened array of intensity values (i.e. weights for KDE data points)
    weights = image.ravel()

    # Shift coordinates to be in quadrant 1
    minxy = np.min(coords_, axis=0)
    coords_ -= minxy

    if angle % 90 != 0:
        # Get binary mask for rotated frame area
        vertices = np.array([
            coords_[np.argmin(coords_[:, 0])],
            coords_[np.argmin(coords_[:, 1])],
            coords_[np.argmax(coords_[:, 0])],
            coords_[np.argmax(coords_[:, 1])]
        ])

    else:
        # If a 90 degree rotation full frame contained in original image size
        # You shouldn't be using this function!
        vertices = np.array([0, 0], [1, 0], [1, 1], [0, 1])

    if reshape_method == 'fulldata':
        # Expand resulting image frame

        h, w = int(np.floor(h - 2*minxy[1])), int(np.floor(w - 2*minxy[0]))
        coords = np.array(
            np.meshgrid(np.arange(0, w), np.arange(0, h))
        ).transpose(1, 2, 0).reshape((-1, 2))

        xlim = (np.min(coords_[:, 0]), np.max(coords_[:, 0]) + 1)
        ylim = (np.min(coords_[:, 1]), np.max(coords_[:, 1]) + 1)

    elif (reshape_method == 'original') or (reshape_method == 'only_data'):
        # Select coordinates remaining inside original image frame
        coords_ += minxy
        vertices += minxy

        inds = np.argwhere(
            ((coords_[:, 0] >= 0) &
             (coords_[:, 0] <= w) &
             (coords_[:, 1] >= 0) &
             (coords_[:, 1] <= h)
             )
        )

        xlim = (0, w + 1)
        ylim = (0, h + 1)

        coords_ = np.squeeze(coords_[inds])
        weights = np.squeeze(weights[inds])

    # elif :
    #     xlim = (0, w + 1)
    #     ylim = (0, h + 1)

    else:
        raise Exception('reshape_method must be one of: "fulldata", '
                        '"original", or "only_data".')

    # Create rotated frame mask:
    mask = polygon2mask(
        ((np.ceil(np.max(coords[:, 1])) + 1).astype(int),
         (np.ceil(np.max(coords[:, 0])) + 1).astype(int)),
        np.fliplr(vertices))
    # mask = erosion(mask, footprint=np.ones((3, 3)))

    image_rot = linearKDE_2D(
        coords_,
        xlim,
        ylim,
        d=1,
        r=bandwidth,
        weights=weights,
        return_binedges=False,
    )
    image_rot = np.where(mask == 0, fill_value, image_rot)

    if reshape_method == 'only_data':
        xlim, ylim, crop_sl = binary_find_largest_rectangle(mask)
        image_rot = image_rot[ylim[0]:ylim[1], xlim[0]:xlim[1]]

    else:
        image_rot = image_rot[1:-1, 1:-1]

    return image_rot


def get_circular_kernel(r):
    """Get circular kernel

    Parameters
    ----------
    r : int
        Kernel radius.

    Returns
    -------
    kernel : 2D array
        The circular kernel.
    """

    kern_rad = int(np.floor(r))
    size = 2*kern_rad + 1
    kernel = np.array(
        [1 if np.hypot(i - kern_rad, j - kern_rad) <= r
         else 0
         for j in range(size) for i in range(size)]
    ).reshape((size, size))

    return kernel


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

    kernel = get_circular_kernel(r)

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


def binary_find_smallest_rectangle(array):
    """
    Get the smallest rectangle (with horizontal and vertical sides) that
    contains an entire ROI defined by a binary array.

    This is useful for cropping to the smallest area without losing any useful
    data. Unless the region is already a rectangle with horiaontal and vertical
    sides, there will be remaining areas that are not part of the ROI. If only
    ROI area is desired in the final rectangle, use
    "binary_find_largest_rectangle."


    Parameters
    ----------
    array : ndarray of shape (h,w)
        The binary image.

    Returns
    -------
    xlim : list-like of length 2
        The x limits (columns) of the smallest rectangle.

    ylim : list-like of length 2
        The y limits (columns) of the smallest rectangle.

    sl : numpy slice object
        The slice object which crops the image to the smallest rectangle.
    """

    xinds = np.where(np.sum(array.astype(int), axis=1) > 0, 1, 0
                     ).reshape((-1, 1))
    yinds = np.where(np.sum(array.astype(int), axis=0) > 0, 1, 0
                     ).reshape((1, -1))

    newroi = (xinds @ yinds).astype(bool)

    xlim, ylim, sl = binary_find_largest_rectangle(newroi)

    return xlim, ylim, sl


def fft_amplitude_mask(
        image,
        xy_fft,
        r,
        blur,
        thresh=0.5,
        fill_holes=True,
        buffer=10
):
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

    if not (type(r) is int) | (type(r) is float):
        if xy_fft.shape[0] != r.shape[0]:
            raise Exception("If'r' is not an int or float, its length "
                            "must match the first dimension of xy_fft.")

    fft = fftshift(fft2(image))
    mask = np.zeros(fft.shape)
    xy = np.mgrid[:mask.shape[0], : mask.shape[1]]
    xy = np.array([xy[1], xy[0]]).transpose((1, 2, 0))
    if (type(r) is int) | (type(r) is float):
        for xy_ in xy_fft:
            mask += np.where(norm(xy - xy_, axis=2) <= r, 1, 0)

    else:
        for i, xy_ in enumerate(xy_fft):
            mask += np.where(norm(xy - xy_, axis=2) <= r[i], 1, 0)

    amplitude = np.abs(ifft2(fftshift(fft * mask)))
    amplitude = image_norm(gaussian_filter(amplitude, sigma=blur))
    mask = np.where(amplitude > thresh, 1, 0)
    if fill_holes:
        mask = binary_fill_holes(mask)
    mask = np.where(mask, 1, 0)
    mask = binary_dilation(mask, iterations=buffer)

    return mask
