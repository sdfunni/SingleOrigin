"""Module containing image manipulation functions."""

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
    label,
)

from scipy.signal import convolve2d
from scipy.fft import (
    fft2,
    ifft2,
    fftshift,
)

from skimage.draw import polygon, polygon2mask
from skimage.morphology import erosion

from tifffile import imwrite

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

from SingleOrigin.utils.system import is_running_in_jupyter
from SingleOrigin.utils.crystalmath import rotation_angle_bt_vectors
from SingleOrigin.utils.plot import quickplot
from SingleOrigin.utils.mathfn import gaussian_2d

pkg_dir, _ = os.path.split(__file__)

# %%

jupyter_warning = (
    'Graphical picking for this method does not work in a notebook' +
    ' environment. Please pass desired coordinates as the appropriate ' +
    'argument. These may be determined manually or selected using the ' +
    'SingleOrigin.pick_points() method which has notebook-compatible '
    'graphical picking.'
)


def image_norm(image):
    """
    Norm an image to 0-1

    Parameters
    ----------
    image : ndarray
         Input image as an ndarray

    Returns
    -------
    image_normed : ndarray with the same shape as 'image'
        The normed image

    """

    [min_, max_] = [np.nanmin(image), np.nanmax(image)]
    image_normed = (image - min_)/(max_ - min_)

    return image_normed


def burn_in_scalebar(
        image,
        pixelSize,
        pixelUnit,
        loc='br',
        length=None,
        pad=None,
        contrast=None,
        label=True,
):
    """
    Burn a scalebar into an image array

    Parameters
    ----------
    image : array like
        The image.

    pixelSize : float
        The pixel size in units of argument units.

    pixelUnit : str
        The pixel size of the image or diffraction pattern: e.g. 'nm', 'um',
        '1/nm'.

    loc : str
        'tl', 'tr', 'bl', 'br'
        Default: 'br'

    length : scalar or None
        Length of the scalebar in pixelUnits. If None, a length will be
        automatically determined.
        Default: None.

    contrast : int or None
        1, 0 or None. Determins if scalebar patch is bright (1) or dark (0).
        If only adding a scalebar (without label) the bar will be this
        contrast. If adding a label, the box will be this contrast while the
        text and bar will be the opposite. If None, contrast will
        automatically be determined.
        Default: None.

    label : bool
        Whether to place label the scalebar with length and units. If False,
        the scalebar marker will still drawn, but without text.
        Default: True.

    Returns
    -------
    image : array like
        The image array with scalebar burned in scalebar.

    """

    # Copy image and decide on scale bar length and contrast
    img = copy.deepcopy(image)
    h, w = img.shape[:2]
    if pad is None:
        pad = int(0.01 * h)

    size = np.min(img.shape[:2])
    if np.max(img.shape[:2]) >= 2*size:
        size *= 2

    font_fact = size / 1024

    if length is None:
        sb_sizes = np.array([10**dec * int_ for dec in range(-1, 5)
                             for int_ in [1, 2, 4, 5]])

        fov = size * pixelSize
        length = sb_sizes[np.argmax(sb_sizes > fov*0.1)]
        if length >= 1:
            length = int(length)

    sbar_pixels = int(length/pixelSize)

    font_size = 5 * pad

    bbox_height = int(10*pad)

    # Set units
    if pixelUnit == 'nm':
        if np.log10(length) < 3:
            pixelUnit = r'nm'
        elif np.log10(length) < 6:
            pixelUnit = 'um'
            length *= 1e-3

    elif pixelUnit == '1/nm':
        pixelUnit = r'$\mathrm{nm^{-1}}$'
        # pixelUnit = '1/nm'
    elif pixelUnit == '1/A':
        pixelUnit = r'$\mathrm{\AA^{-1}}$'
        # pixelUnit = '1/nm'

    elif pixelUnit == 'um':
        pixelUnit = r'$\mathrm{\mu m}$'

    # Work out label box area
    bbox_width = int(sbar_pixels + 2*pad)

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

    if contrast is None:
        intavg = np.mean(img[bbox[0, 0]:bbox[0, 1], bbox[1, 0]:bbox[1, 1]])
        intmid = np.mean([np.max(img), np.min(img)])
        if intavg > intmid:
            contrast = 0
        else:
            contrast = 1

    if not label:
        contrast = int((not bool(contrast)))

    # Add scalebar box and label
    if label == True:
        # Make scalebar box
        img[bbox[0, 0]:bbox[0, 1], bbox[1, 0]:bbox[1, 1]] = contrast

        # Build the label text image
        string = f'{length} ' + pixelUnit

        # Alloted size of the text:
        tlims = np.array([5*pad, bbox_width])

        plt.close()
        plt.ioff()
        fig, ax = plt.subplots(figsize=(tlims[1] / tlims[0], 1), dpi=tlims[0],
                               layout='constrained',
                               )
        ax.axis('off')

        if contrast:
            color = 'black'
        else:
            color = 'white'
            fig.set_facecolor('black')
            ax.set_facecolor('black')

        text_obj = ax.text(
            0.5, 0.5,
            string,
            fontsize=font_size,
            ha='center',
            va='center',
            color=color,
        )

        # Check text fit:
        tbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())

        # Access the dimensions from the bounding box
        tdims = np.array([tbox.height, tbox.width])
        tx0, ty0 = tbox.x0, tbox.y0

        if np.any(tdims > tlims*1):
            font_size *= np.min(tlims*1 / tdims)

            ax.cla()
            text_obj = ax.text(
                0.5, 0.5,
                string,
                fontsize=font_size,
                ha='center',
                va='center',
                color=color,
            )

        ax.axis('off')

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        text = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        text = image_norm(
            np.max(text.reshape(height, width, 4)[..., :3], axis=2))

        plt.close()
        plt.ion()
        img[bbox[0, 0] + 4*pad: bbox[0, 1] - pad,
            bbox[1, 0]: bbox[1, 0] + text.shape[1],
            ] = text

        shift = pad

    else:
        shift = pad * 6

    # Finally, add scalebar
    img[bbox[0, 0] + shift:bbox[0, 0] + 3*pad + shift,
        int(cent_width - sbar_pixels/2):
        int(cent_width - sbar_pixels/2) + sbar_pixels
        ] = int((not bool(contrast)))

    return img

    # img = copy.deepcopy(image)
    # # Get image and scalebar parameters
    # sb_sizes = np.array([10**dec * int_ for dec in range(-1, 5)
    #                      for int_ in [1, 2, 4, 5]])

    # fact = 1
    # h, w = img.shape[:2]
    # size = np.min(img.shape[:2])
    # if np.max(img.shape[:2]) >= 2*size:
    #     size *= 2

    # font_fact = size / 1024
    # pad = int(0.02 * size)

    # fov = size * pixelSize
    # sbar = sb_sizes[np.argmax(sb_sizes > fov*0.1)]
    # if sbar > 1:
    #     sbar = int(sbar)
    # # print(sbar)
    # sbar_pixels = int(sbar/pixelSize)

    # font_size = int(50 * font_fact)
    # if font_size < 1:
    #     font_size = 1

    # bbox_height = int(2.2*pad + 0.02*h + font_size)

    # # Set units
    # if pixelUnit == 'nm':
    #     if np.log10(sbar) < 3:
    #         pixelUnit = 'nm'
    #     elif np.log10(sbar) < 6:
    #         pixelUnit = 'um'
    #         fact = 1e-3

    # elif pixelUnit == '1/nm':
    #     # pixelUnit = r'nm$^{-1}$'
    #     pixelUnit = '1/nm'

    # elif pixelUnit == 'um':
    #     pixelUnit = 'um'

    # # Get font for PIL
    # font_path = os.path.join(
    #     mpl.__path__[0],
    #     "mpl-data/fonts/ttf/DejaVuSans.ttf",
    # )

    # font = ImageFont.truetype(
    #     font_path,
    #     size=font_size,
    # )

    # # Get box width to ensure contains string and scalebar
    # img_obj = Image.fromarray(img)
    # draw = ImageDraw.Draw(img_obj)
    # text = f'{sbar * fact} {pixelUnit}'
    # text_length = draw.textlength(
    #     text,
    #     font=font,
    # )

    # bbox_width = np.max([int(sbar_pixels + 2*pad), text_length + 2*pad])

    # bbox = np.zeros((2, 2), dtype=int)

    # if loc[0] == 't':
    #     bbox[0, 0] = int(pad)
    #     bbox[0, 1] = int(pad + bbox_height)

    # if loc[0] == 'b':
    #     bbox[0, 0] = int(h - pad - bbox_height)
    #     bbox[0, 1] = int(h - pad)

    # if loc[1] == 'l':
    #     bbox[1, 0] = int(pad)
    #     bbox[1, 1] = int(pad + bbox_width)

    # if loc[1] == 'r':
    #     bbox[1, 0] = int(w - pad - bbox_width)
    #     bbox[1, 1] = int(w - pad)

    # cent_width = int(np.mean(bbox[1]))

    # # Make scalebar box
    # img[bbox[0, 0]:bbox[0, 1], bbox[1, 0]:bbox[1, 1]] = 1

    # # Add scalebar
    # img[bbox[0, 0] + pad:bbox[0, 0] + 2*pad,
    #     int(cent_width - sbar_pixels/2):
    #     int(cent_width - sbar_pixels/2) + sbar_pixels
    #     ] = 0

    # img_obj = Image.fromarray(img)
    # draw = ImageDraw.Draw(img_obj)
    # draw.text(
    #     (cent_width, int(bbox[0, 1] - pad)),
    #     text,
    #     0,
    #     font=font,
    #     anchor='mb',
    # )

    # return np.array(img_obj)


def save_bw_image(image, folder, name, bits=16):
    """
    Save image with greyscale contrast.

    Parameters
    ----------
    image : array like
        The image. For image stacks, the first dimension must be the stack
        index.

    folder : str
        Folder location for saving the image.

    name : str
        File name for saving the image. Optionally ending in .tif or .png. If
        filetype extension is included, this file type will be used, otherwise
        file will save as a .tif.

    bits : int
        The bit depth to use. Must be 8, 16, or 32.

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

    if ~np.isin(name[-4:], ['.tif', '.png']):
        name += '.tif'

    path = Path(os.path.join(folder, name))

    image = (image_norm(image) * (2**bits - 1)).astype(type_)
    imwrite(
        path,
        image,
        photometric='minisblack'
    )


def save_rgb_image(
        image,
        folder,
        name,
        bits=8,
        cmap=None,
        alpha=False,
        pixelSize=None,
        pixelUnit=None,
        loc='br',
        sblabel=True,
        length=None,
        sbcontrast=1,
):
    """
    Save an rgb(a) image, using a colormap if greyscale.

    Parameters
    ----------
    image : array like
        The image. If an image stack, first dimension must be the stack
        index. If already in rgb(a) format, last dimension must be the color
        channels. Contrast for all channels should be withing the range [0, 1]
        or final saved image contrast will not be correct.

    folder : str
        Folder location for saving the image.

    name : str
        File name for saving the image. Optionally ending in .tif or .png. If
        filetype extension is included, this file type will be used, otherwise
        file will save as a .tif.

    bits : int
        The bit depth to use. Must be 8 or 16.

    cmap : str or None
        The matplotlib colormap to apply for greyscale images.
        Default: 'inferno'

    alpha : bool
        Whether to include alpha channel. Only applies if a greyscale image is
        passed. Previously converted rgb(a) image will be saved as passed.

    pixelSize : float
        The pixel size in units of argument units.

    pixelUnit : str
        The pixel size of the image or diffraction pattern: e.g. 'nm', 'um',
        '1/nm'.

    Returns
    -------
    None.

    """

    if bits == 8:
        type_ = np.uint8
    elif bits == 16:
        type_ = np.uint16
    else:
        raise Exception(
            'bits must be 8, 16'
        )

    bitscale = 2**bits - 1

    if ~np.isin(name[-4:], ['.tif', 'png']):
        name += '.tif'

    if (alpha is True) or (image.shape[-1] == 4):
        extrasamples = ['unassalpha']
    else:
        extrasamples = None

    if ~np.isin(image.shape[-1], [3, 4]):
        # Apply cmap if needed
        if cmap is None:
            cmap = mpl.colormaps['inferno']

        elif isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]

        else:
            pass

        rgb = cmap(image_norm(image))

        if alpha is False:
            rgb = rgb[..., :-1]

    else:
        rgb = image

    # Scale image for bit depth & convert to correct dtype
    rgb = np.around(rgb * bitscale).astype(type_)

    if pixelSize is not None and pixelUnit is not None:
        # Make a separate scalebar corrected for bitdepth
        sb = burn_in_scalebar(
            np.ones(image.shape[:2]) * np.nan,
            pixelSize=pixelSize,
            pixelUnit=pixelUnit,
            length=length,
            loc=loc,
            label=sblabel,
            contrast=sbcontrast,
        )[..., None] * np.ones(rgb.shape) * bitscale

        # Burn scalebar into image
        rgb = np.where(sb >= 0, sb, rgb).astype(type_)

    path = Path(os.path.join(folder, name))

    imwrite(
        path,
        rgb,
        photometric='rgb',
        extrasamples=extrasamples
    )


def save_fig(fig, folder, name, dpi=100, format='tif'):
    """
    Wrapper function for matplotlib.pyplot.savefig()

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

    format : str
        Image format for saving the figure. Any file types supported by
        matplotlib.pyplot.savefig() are valid.
        Default: 'tif'


    Returns
    -------
    None.

    """

    if '.' not in name[-5:]:
        name += '.' + format
    else:
        format = name.split('.')[-1]

    fig.savefig(
        os.path.join(folder, name),
        bbox_inches='tight',
        pad_inches=0.25,
        dpi=dpi,
        format=format
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


def cross_correlation(im_ref, im_meas):
    """
    Parameters
    ----------
    im_ref : 2D array
        The reference image.

    im_meas: nd array
        The image or array of images to correlate with the reference. Last
        two dimensions must match the shape of im_ref

    Returns
    -------
    imCC : 2D arrays of the same size as input images
        The fftshifted cross correlation image(s).

    """

    dims_exp = len(im_meas.shape) - 2

    imCC = fftshift(np.abs(ifft2(
        (fft2(im_ref[*(None,) * dims_exp, ...]) * np.conj(fft2(im_meas)))
    )),
        axes=(-2, -1)
    )

    return imCC


def radial_average(data, center=None, max_dist=None):
    """
    Calculate the average radial profile for an image/dataset about a point.

    Parameters
    ----------
    data : 2D+ array
        The image or dataset. The radial average will be taken over the final
        two dimensions. For a 4D STEM dataset, the scan dimensions should be
        the first two dimensions. For a series or stack, the frame index should
        be the first dimension. May have NaNs: these will not be counted or
        weighted when normalizing the bins.

    center : 2-list or None
        The [x, y] center point about which to calculate the profile. Specify
        in image coordinates of [col, row] with down/right being positive.
        If None, uses the middle of the image.
        Default: None.

    max_dist : int or None
        Maximum radial distance for which to calculate the radial average.

    Returns
    -------
    radial_avg : array
        The radial average(s) of the dataset as the final dimension. Will have
        one less dimension than the input data.

    """

    if center is None:
        center = [int(data.shape[-2]/2),
                  int(data.shape[-1]/2)]

    # Get bin ID for each pixel relative to the radial center
    # This uses np.rint() so that bins are centered at integer number of pixels
    # i.e. first bin is 0 (-0.5 to 0.5), 2nd bin is 1 (0.5 to 1.5), etc.
    ids = np.rint(getAllPixelDists(data.shape[-2:], center).ravel()
                  ).astype(int)

    if len(data.shape) < 2:
        raise Exception(
            'Cannot calculate radial average for data with < 2 dimensions.'
        )

    def ravg(frame, ids):
        # Ravel image and remove NaNs from image and bin ID arrays
        w = frame.ravel()
        keep = ~np.isnan(w)
        ids = ids[keep]
        w = w[keep]

        # Get number of pixels in each distance bin for normalizing
        counts = np.bincount(ids)
        # Remove zeros to prevent division by zero errors
        counts[counts == 0] = 1

        # Get array of bin centers in pixels
        # dists = np.arange(counts.shape[0])

        # Get the radial average, normalizing by number of pixels in each bin
        radial_avg = (np.bincount(ids, w)/counts)

        return radial_avg

    radial_avg = [
        ravg(frame, ids) for frame in data.reshape((-1, *data.shape[-2:]))
    ]

    minlen = np.min([r.shape[0] for r in radial_avg])

    if max_dist is None:
        max_dist = minlen

    else:
        max_dist = np.min([max_dist, minlen]).astype(int)

    radial_avg = np.array([
        ra[:max_dist] for ra in radial_avg
    ]).reshape((*data.shape[:-2], -1)).squeeze()

    return radial_avg


def bandpass(image, highpass=1, lowpass=0):
    """
    High and/or low pass filter an image using Gaussian filters.

    Parameters
    ----------
    image : 2D array
        The image.
    highpass : scalar or None
        The width of the Gaussian highpass filter.
    lowpass : scalar or None
        The width of the Gaussian lowpass filter.

    Returns
    -------
    imfiltered : 2D array
        The filtered image.

    """

    imfiltered = copy.deepcopy(image)

    h, w = image.shape
    x_coords, y_coords = np.flip(np.indices(image.shape), axis=0)

    if lowpass > 0:
        imfiltered = lowpass_filter(image, lowpass)

        # imfiltered = gaussian_filter(image, lowpass)

    if highpass > 0:
        imfiltered = imfiltered - lowpass_filter(image, highpass)

    imfiltered = image_norm(imfiltered)

    return imfiltered


def lowpass_filter(image, sigma, get_abs=True):

    h, w = image.shape
    x_coords, y_coords = np.flip(np.indices(image.shape), axis=0)

    # Get uncentered low pass filter
    lowPass = image_norm(np.abs(fft2(gaussian_2d(
        x_coords,
        y_coords,
        x0=w/2,
        y0=h/2,
        sig_maj=sigma,
        sig_rat=1,
        ang=0,
        A=1,
        b=0,
    ))))

    # Get lowpass filtered image
    imfiltered = ifft2(fft2(image) * lowPass)

    if get_abs:
        imfiltered = np.abs(imfiltered)

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


def get_mask_circle(shape, center, r):
    """
    Make a circluar mask.

    Parameters
    ----------
    shape : 2-tuple
        The array shape.

    center : 2-list
        The x, y coordinates of the circle center.

    r : scalar
        The circle radius.

    Returns
    -------
    mask : 2D array of shape shape
        The mask. 1 inside the circle, 0 outside.
    """

    dists = getAllPixelDists(shape, center)
    mask = np.where(dists <= r, 1, 0)

    return mask


def pick_mask_circle(image, pick='diameter', points=None, plot=True):
    """
    Make a circluar mask.

    Parameters
    ----------
    image : 2D array
        The image from which to pick the mask.

    pick : str ('diameter' or 'center-edge')
        Whether to pick two points as the diameter of the circle or to pick
        the center and one edge point

    points : 2x2 array or None
        The (x, y) coordinates of the points used for creating the circle
        according to which pick type is passed. (Each row is a point.) If
        None, graphical picking is enabled, which is not functional in a
        notebook environment, however it works in Spyder.
        Default: None.

    plot : bool
        Whether to plot the resuting mask for verification.
        Default: True.

    Returns
    -------
    mask : 2D array of shape shape
        The mask. 1 inside the circle, 0 outside.
    """

    if points is None:

        if is_running_in_jupyter():
            raise Warning(jupyter_warning)

        fig, ax = plt.subplots(figsize=(8, 10))
        quickplot(image, cmap='gist_gray', figax=ax)

        ax.set_title(
            'Left click to add points.\n' +
            'Right click or backspace key to remove last.\n' +
            'Center click or enter key to complete.\n' +
            'Must select only 2 points.'
        )

        points = plt.ginput(
            n=-1,
            timeout=0,
            show_clicks=True,
            mouse_add=MouseButton.LEFT,
            mouse_pop=MouseButton.RIGHT,
            mouse_stop=MouseButton.MIDDLE
        )

        plt.close()
        points = np.array(points)

    if pick == 'diameter':
        center = np.mean(points, axis=0)
        r = np.sum(np.diff(points, axis=0)**2) ** 0.5 / 2

    else:
        center = points[0]
        r = np.sum(np.diff(points, axis=0)**2) ** 0.5

    dists = getAllPixelDists(image.shape, center)
    mask = np.where(dists <= r, 1, 0)

    if plot:
        fig, ax = quickplot(image, cmap='gist_gray', figax=True)

        ax.imshow(
            mask,
            alpha=0.2,
            cmap='Reds'
        )

    return mask


def get_mask_polygon(
        data,
        vertices=None,
        buffer=0,
        invert=False,
        plot=True,
        return_vertices=False
):
    """
    Get a binary mask for an arbitrary shaped polygon.

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

    plot : bool
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

        if is_running_in_jupyter():
            raise Warning(jupyter_warning)

        fig, ax = quickplot(data, cmap='gist_gray', figax=True)

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

    mask = polygon2mask(data.shape, vertices).astype(int)

    if buffer:
        mask = erosion(
            mask,
            footprint=np.ones((3, 3))
        ).astype(int)

    if invert:
        mask = np.where(mask == 1, 0, 1)

    if plot:
        fig, ax = quickplot(data, cmap='gist_gray', figax=True)

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
        end=None,
        plot=True,
):
    """
    Integrate data along a line with a given integration width. Line is
    chosen by graphical picking on a virtual image or specifying the start
    and end points.

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

    start, end : array-like or None
        The x, y start and end points of the integration line, if previously
        determined. If not passed, graphical picking will be used, which does
        not work in notebooke environments, however it does work in Spyder.
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

    elif len(data.shape) == 3:
        if scandims is None and signaldims is None:
            print("Assuming the last dimension is the signal...")
            scandims = data.shape[:2]
            signaldims = data.shape[2]

            # raise Exception(
            #     'Must specify scandims or signaldims for 3D data.'
            # )
        elif len(scandims) == 1:
            raise Exception('This is already a line profile...')
        else:
            scandims = [data.shape[i] for i in scandims]
            signaldims = tuple(data.shape[i] for i in signaldims)

        if image is None:
            image = np.sum(data, axis=2)

    elif len(data.shape) == 4:
        scandims = data.shape[:2]
        signaldims = data.shape[2:]
        if image is None:
            image = np.max(data, axis=(2, 3))

    else:
        raise Exception('This dimensionality is not supported.')

    if plot or start is None:
        fig, ax = plt.subplots(figsize=(10, 10))

        quickplot(image, cmap='inferno', figax=ax)

        xends = [start[0], end[0]]
        yends = [start[1], end[1]]
        plt.plot(xends, yends)

        ax.set_xticks([])
        ax.set_yticks([])

    if ((start is None) or (end is None)):
        if is_running_in_jupyter():
            raise Warning(jupyter_warning)

        picks_xy = np.array(plt.ginput(n_picks, timeout=60))

        start = picks_xy[0]
        end = picks_xy[1]

        xends = [start[0], end[0]]
        yends = [start[1], end[1]]
        plt.plot(xends, yends)

    vector = np.array(end) - np.array(start)

    unit_vect = vector / norm(vector)

    xy = np.fliplr(np.indices(scandims).reshape(2, -1).T)

    z = data.reshape((-1,) + signaldims)

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

    return line_integration, start, end


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
        The bandwidth of the KDE in bins.

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
    if r == 1:
        xys = np.array([-1, 0], dtype=int)
        # Get reference pixel for each data point
        xyC = np.ceil(coords / d) * d

    elif r > 1:
        r_ = r//1

        yxs = np.arange(-r_, r_ + 1, 1, dtype=int)
        xyC = np.around(coords)

    # Calculate eash pixel shifted histogram and sum together
    for j in xys:
        for i in xys:

            # Find distance weighting for high sampling rate:
            # Method results in total density per data point deviating slightly
            # from 1, but close with sufficient bandwidth (i.e. r >= 2)
            # This method is a KDE using a linear kernel with euclidian
            # distance metric.
            if r > 1:
                shift = np.array([xys[i], xys[j]]) * d
                xyB = xyC + shift
                if norm(shift) > r:
                    continue
                dW = 3/np.pi * (1 - norm(xyB - coords, axis=1) / (d*r)) / r**2
                dW = np.where(dW < 0, 0, dW)
            # Find distance weighting if low sampling (i.e. r == 1):
            # Method is effectively a reverse bilineaer interpolation.
            # That is, it distributes the density from each datapoint over four
            # nearest neighbor pixels using bilinear weighting. This ensures
            # the density contribution from each data point is exactly 1.
            elif r == 1:
                shift = np.array([xys[i], xys[j]]) * d
                xyB = xyC + shift
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
    """
    Rotate images by multiples of 90 degrees. Faster than
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
    """
    Get a 2D origin-shifted rotation matrix for an arbitrary rotation.

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
    """
    Apply a rotation to a set of coordinates.

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
        bandwidth=1,
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
    """
    Get circular kernel

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


def erode(mask, iterations=1):

    footprint = get_circular_kernel(1)

    new_mask = copy.deepcopy(mask)

    for _ in range(iterations):
        new_mask = erosion(
            new_mask,
            footprint=footprint
        )

    return new_mask


def std_local(image, r):
    """
    Get local standard deviation of an image

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
    """
    Gets the slice object of the largest rectangle of 1s in a 2D binary
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
    DEALINGS IN THE SOFTWARE.

    """

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


def nlargest_objects(array, n_obj=1):
    """
    Find the n largest objects in a binary array and return a binary mask.


    Parameters
    ----------
    array : ndarray of shape (h,w)
        The binary image.

    n_obj : int
        The number of objects desired in the mask.

    Returns
    -------
    mask : ndarray
        The binary mask covering the n largest objects with 1 where the
        objects are located and 0 elsewhere.

    """
    binaryarray = np.where(array > 0, 1, 0)
    labels = label(binaryarray)[0]
    _, counts = np.unique(labels, return_counts=True)
    counts[0] = 0
    select = np.argsort(counts)[-n_obj:]
    mask = np.where(np.isin(labels, select), 1, 0)

    return mask


def fft_amplitude_mask(
        image,
        xy_fft,
        r,
        blur,
        thresh=0.5,
        fill_holes=True,
        buffer=10
):
    """
    Create mask based on Bragg spot filtering (via FFT) of image.

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


def measure_distance(
        endPoints,
        image,
        n_measures=1,
        lock_hor_vert=False,
        pixelSize=None,
        pixelUnit=None,
        decimals=2,
        # return_points=False,
        figax=True,
):
    # TODO : update docstring
    """
    Make a circluar mask.

    Parameters
    ----------
    image : 2D array
        The image from which to pick the mask.

    n_measures : int
        The number of measurements to make. 2 points will be selecte for each
        linear measure.

    lock_hor_vert : bool
        Whether to lock measurements into horizontal or vertical orientations
        (depending on which is closer to the original measruement).

    pixelSize : scalar or None

    pixelUnit : scalar or None

    figax : bool or matplotlib axes object
        If a bool, whether to return the figure and axes objects for
        modification by the user. If an Axes object, the Axes to plot into.

    Returns
    -------
    mask : 2D array of shape shape
        The mask. 1 inside the circle, 0 outside.

    """

    fig, ax = plt.subplots(tight_layout=True)
    quickplot(image, cmap='gist_gray', figax=ax)

    endPoints = np.array(endPoints).reshape((-1, 2, 2))

    for points in endPoints:
        if lock_hor_vert:
            vect = points[1] - points[0]
            orientation = np.argmin(np.abs(vect))  # 1=horizontal 0=vertical
            points[:, orientation] = np.mean(points[:, orientation])

        ax.annotate(
            "",
            points[0],
            points[1],
            arrowprops=dict(arrowstyle='|-|'),
            textcoords=ax.transData,
        )

        midpoint = np.mean(points, axis=0)
        vector = np.diff(points, axis=0)[0]

        # Get rotation angle for the text
        angle = rotation_angle_bt_vectors([1, 0], vector * np.array([1, -1]))
        angle -= 180*int(angle/90)

        dist = np.around(np.linalg.norm(vector) * pixelSize, decimals=decimals)

        bbox = dict(fc="white", ec="black",)
        text = ax.text(
            midpoint[0], midpoint[1],
            f"{dist:.{decimals}f} {pixelUnit}",
            ha="center",
            va="center",
            bbox=bbox,
            rotation=angle,
            size='small',
            transform=ax.transData
        )

        bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
        data_bbox = ax.transData.inverted().transform(bbox)

        unitvect = vector / np.linalg.norm(vector)
        lenbbox = np.abs(unitvect @ (data_bbox[0] - data_bbox[1]))

        if lenbbox > (np.linalg.norm(vector) * 0.75):
            shift = (np.flip(unitvect) @ (data_bbox[0] - data_bbox[1]) * 2
                     ) * np.flip(unitvect)

            if shift[0] > 0:
                shift[0] *= -1
            if shift[1] < 0:
                shift[1] *= -1

            text.set_position(midpoint + shift)

    if figax:
        return fig, ax
