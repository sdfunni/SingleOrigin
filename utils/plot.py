"""Module containing utility plotting functions."""

import numpy as np
from numpy.linalg import norm

import matplotlib as mpl
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
from matplotlib import patches

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.cm import ScalarMappable


# %%


def pick_points(
        n_picks,
        image,
        xy_peaks=None,
        origin=None,
        window_size=None,
        cmap='gray',
        quickplot_kwargs={},
        figax=False,
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

    n_picks : int or 'any'
        The number of points to be chosen. If any, picking will only be ended
        when a key is pressed.

    xy_peaks : array of shape (m,2) or None
        The x,y coordinates of the full list of points from which to select the
        picks. If None, returns click positions.

    origin : 2-tuple of scalars or None
        The origin point. This will be plotted for reference, but has no other
        function. If None, no origin is plotted.
        Default: None

    window_size : scalar
        The size of the region to plot, centered around the middle of the
        image. Useful to zoom in to the center of an FFT when most of the
        information is contained in a small area.

    cmap : str
        The matplotlib color map to use.
        Default: 'gray'

    quickplot_kwargs : dict
        Any key word arguments to pass to quickplot.

    figax : bool or matplotlib axes object
        If a bool, whether to return the figure and axes objects for
        modification by the user. If an Axes object, the Axes to plot into.

    Returns
    -------
    picks_xy : array of shape (n,2)
        The x,y coordinates of the 'n' chosen data points or click positions.

    fig, ax : matplotlib Figure and Axes objects
        Optional: if figax is True.

    """

    if isinstance(figax, bool):
        h, w = image.shape
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig, ax = figax[0], figax[1]

    if image is not None:
        quickplot(image, cmap=cmap, figax=ax, **quickplot_kwargs)

    if n_picks == 'any':
        print('Select any number of peaks.',
              'End picking by pressing any key.')

    if xy_peaks is not None:
        ax.scatter(xy_peaks[:, 0], xy_peaks[:, 1], c='red', s=8)

    if window_size is not None:
        ax.set_ylim(bottom=h//2 + window_size/2, top=h//2 - window_size/2)
        ax.set_xlim(left=w//2 - window_size/2, right=w//2 + window_size/2)

    if origin is not None:
        ax.scatter(
            origin[0], origin[1],
            marker='P', ec='white', fc='black', s=100
        )

    picks_xy = []
    plotpoints = []

    def on_click(event):
        plt.gca()
        if event.inaxes == ax:
            if event.button is MouseButton.LEFT:

                pick = [event.xdata, event.ydata]
                if (n_picks != 'any') & (len(picks_xy) > n_picks):
                    print('Too many picks. ',
                          'Remove previous picks with right click(s) if you ',
                          'need to repick.')
                else:
                    if xy_peaks is not None:
                        # find nearest
                        vects = xy_peaks - np.array([pick])
                        ind = np.argmin(norm(vects, axis=1))
                        pick = list(xy_peaks[ind, :])

                        picks_xy.append(pick)
                    else:
                        picks_xy.append(pick)

                    plotpoints.append(ax.scatter(
                        pick[0], pick[1],
                        marker='P', fc='black', ec='white', s=50, zorder=10)
                    )

                    fig.canvas.draw()

            elif event.button is MouseButton.RIGHT:
                del picks_xy[-1]
                plotpoints[-1].remove()
                fig.canvas.draw()
                del plotpoints[-1]

    def on_key(event):
        if event.key:
            plt.disconnect(cid)
            plt.disconnect(kid)
            plt.close('all')

            if (n_picks != 'any') & (len(picks_xy) != n_picks):
                print('!!! Incorrect number of picks. Please try again.')

    directions = [
        f'Select {n_picks} points.',
        'Right click to remove.',
        'Enter when finished.'
    ]

    print(*directions)
    ax.set_title(' '.join(directions), fontsize=20)

    cid = plt.connect('button_press_event', on_click)
    kid = plt.connect('key_press_event', on_key)

    if figax is True:
        return picks_xy, fig, ax
    else:
        return picks_xy


def quickplot(
        im,
        cmap='magma',
        figsize=(6, 6),
        hide_ticks=True,
        pixel_size=None,
        pixel_unit=None,
        scalebar_len=None,
        scalebar_loc='lower right',
        scalebarfont=12,
        figax=False,
        returnplot=False,
        tight_layout=True,
        scaling=None,
        vmin=None,
        vmax=None,
        zerocentered=False,
        bad_color='black',
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

    scalebar_loc : str
        The location to put the scalebar: e.g. 'lower right'.
        Default: 'lower right'

    scalebarfont : scalar
        Fontsize for the scalebar.
        Default: 12

    figax : bool or matplotlib axes object
        If a bool, whether to return the figure and axes objects for
        modification by the user. If an Axes object, the Axes to plot into.

    returnplot : bool
        Whether to return the image plotting instance. Useful for creating a
        colorbar.
        Default: False

    tight_layout : bool
        Whether to apply tight layout to the plot.
        Default: True

    scaling : float or str
        Linear (None), log ('log') or power law (scalar) scaling of the image
        intensity.
        Default: None

    vmin, vmax : scalars
        The contrast limits for the colormap.

    zerocentered : bool
        Whether to center 0 contrast at the middle of the colormap range.
        Pass for autocentering of the cmap. Manually center by passing
        vmin = -vmax.

    bad_color : str
        Color to use for "bad" values (e.g. NaNs).
        Default: 'black'

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
        norm = LogNorm(vmin=vmin, vmax=vmax)

        im -= np.min(im) - 1

    elif isinstance(scaling, (float, int)):
        norm = PowerNorm(scaling, vmin=vmin, vmax=vmax)
    else:
        norm = None
    if len(im.shape) == 2:
        cmap = mpl.colormaps.get_cmap(cmap)
        cmap.set_bad(color=bad_color)
    elif len(im.shape) == 3 and np.isin(im.shape[-1], [3, 4]):
        cmap = None
    else:
        raise Exception('Image type not allowed.')

    if isinstance(figax, bool) or figax is None:
        fig, ax = plt.subplots(1, figsize=figsize, tight_layout=tight_layout)

    else:
        ax = figax

    if scaling is not None:
        implot = ax.imshow(im, cmap=cmap, norm=norm)
    else:
        implot = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)

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
            font_properties={'size': scalebarfont},
            pad=0.3,
            sep=3,
            border_pad=0.3,
            box_color='white',
            height_fraction=0.02,
            color='black',
            location=scalebar_loc,
            fixed_value=scalebar_len,
        )
        ax.add_artist(scalebar)

    return_objs = ()

    if figax is True:
        return_objs += (fig, ax)
    if returnplot is True:
        return_objs += (implot,)
    if len(return_objs) == 1:
        return_objs = return_objs[0]
    elif len(return_objs) == 0:
        return_objs = None

    return return_objs


def quickcbar(
        cax,
        cmapping,
        label=None,
        vmin=None,
        vmax=None,
        orientation='vertical',
        ticks=None,
        tick_params={},
        label_params={},
        tick_loc=None,
):
    """
    Put a colorbar into an matplotlib Axis.
    """

    if isinstance(cmapping, mpl.image.AxesImage):
        # can pass straight to colorbar
        mappable = cmapping

    elif isinstance(cmapping, str):
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                  cmap=cmapping)

    tick_params_default = {
        'labelsize': 14,
    }
    tick_params_default.update(tick_params)

    label_params_default = {
        'fontsize': 16,
    }
    label_params_default.update(label_params)

    cbar = plt.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
        ticks=ticks,
    )
    if label is not None:
        cbar.set_label(label=label, **label_params_default)
    cbar.ax.tick_params(**tick_params_default)
    if tick_loc is not None:
        if np.isin(tick_loc, ['left', 'right']):
            cax.yaxis.set_ticks_position(tick_loc)
        elif np.isin(tick_loc, ['top', 'bottom']):
            cax.yaxis.set_ticks_position(tick_loc)


def plot_vDetector(
        detector,
        figax,
        contour_kwargs_dict={}
):
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
        Arguments to pass to matplotlib.pyplot.contour. This function is used
        to draw the virtual detector outline.

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


def plot_stack_with_slider(
        stack,
        scaling=None,
        cmap='inferno',
        zerocentered=False,
):
    """
    Plot an image stack with slider to view images through the stack
    interactively.

    Parameters
    ----------
    stack : 3d array
        The image stack. The first dimension must be the stack index.`

    scaling : float or str or None
        The scaling to apply to to the displayed images. Same as quickplot().

    cmap : str
        The color map to use.

    zerocentered : bool
        Whether to center 0 contrast at the middle of the colormap range.
        Pass for autocentering of the cmap. Manually center by passing
        vmin = -vmax.

    Returns
    -------
    None.

    """

    def frame(num, stack):
        return stack[num]

    fig, ax = plt.subplots(1)

    # Adjust the subplots region to leave  space for the sliders and buttons
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Draw the initial plot
    quickplot(
        frame(0, stack),
        scaling=scaling,
        figax=ax,
        cmap=cmap,
        zerocentered=zerocentered,
    )

    # Define an axes area and draw a slider in it
    frame_slider_ax = fig.add_axes(
        [0.25, 0.15, 0.65, 0.03],
        facecolor='lightgoldenrodyellow'
    )

    frame_slider = Slider(
        ax=frame_slider_ax,
        label='Frame',
        valmin=0,
        valmax=stack.shape[0]-1,
        valstep=1,
        valinit=0,
    )

    # Define an action for modifying the line when any slider's value changes

    def update(val):
        quickplot(
            frame(frame_slider.val, stack),
            scaling=scaling,
            figax=ax,
            cmap=cmap,
            zerocentered=zerocentered,
        )
        fig.canvas.draw()

    frame_slider.on_changed(update)


def sharexy(axs):
    """
    Share x and y axes among a group of Axes objects. Allows for some but not
    all Axes objects to share x and y axes when zooming.

    Parameters
    ----------
    axes : list or
        The image stack. The first dimension must be the stack index.`

    Returns
    -------
    None.

    """
    target = axs.flat[0]
    for ax in axs.flat[1:]:
        target._shared_axes['x'].join(target, ax)
        target._shared_axes['y'].join(target, ax)


def plot_4dstem_explorer_click(
        vImage,
        data1,
        data2=None,
        vImage_kwargs={},
        pattern1_kwargs={},
        pattern2_kwargs={},
        mark_center=True,
        center1=None,
        center2=None,
        orientation='horizontal',
):
    """
    Plot an image stack with slider to view images through the stack
    interactively.

    Parameters
    ----------
    data4d : 4d array
        The 4D STEM dataset.

    vImage : 2d array
        An image to display representing the scan axes.

    scaling : float or str or None
        The scaling to apply to to the displayed images. Same as quickplot().

    cmapImage, cmap4D : str
        The color maps to use for the image and patterns, respectively.

    Returns
    -------
    None.

    """

    if center1 is None:
        center1 = np.flip(data1.shape[-2:]) // 2
    if data2 is not None and center2 is None:
        center2 = np.flip(data2.shape[-2:]) // 2

    vImage_kwargs_default = {'cmap': 'inferno'}
    vImage_kwargs_default.update(vImage_kwargs)

    pattern1_kwargs_default = {
        'cmap': 'inferno',
        'scaling': 0.2,
    }
    pattern1_kwargs_default.update(pattern1_kwargs)

    pattern2_kwargs_default = {
        'cmap': 'inferno',
        'scaling': 0.2,
    }
    pattern2_kwargs_default.update(pattern2_kwargs)

    def frame(yx, data):
        # if data2 is None:
        return data[*yx]
        # else:
        #     return data1[*yx], data2[*yx]

    if data2 is not None:
        n_plots = 3
    else:
        n_plots = 2

    if orientation == 'horizontal':
        fig, axs = plt.subplots(1, n_plots, tight_layout=True)
    elif orientation == 'vertical':
        if n_plots == 2:
            fig, axs = plt.subplots(2, 1, tight_layout=True)
        else:
            fig = plt.figure()
            gs = fig.add_gridspec(nrows=2, ncols=2)
            axs = [
                fig.add_subplot(gs[0, :]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
            ]
    yx = np.array([0, 0])
    # Plot the real space scan image
    quickplot(
        vImage,
        figax=axs[0],
        **vImage_kwargs_default,
    )

    # Draw the initial plot
    quickplot(
        frame(yx, data1),
        figax=axs[1],
        **pattern1_kwargs_default,
    )
    if mark_center:
        axs[1].scatter(center1[0], center1[1], marker='+', color='red')

    if data2 is not None:
        # Draw the initial plot
        quickplot(
            frame(yx, data2),
            figax=axs[2],
            **pattern2_kwargs_default,
        )
        if mark_center:
            axs[2].scatter(center2[0], center2[1], marker='+', color='black')

    def on_click(event, yx):
        if event.inaxes == axs[0]:
            if event.button is MouseButton.LEFT:
                yx = np.array([int(event.ydata), int(event.xdata)])

                # axs[1].cla()
                quickplot(
                    frame(yx, data1),
                    figax=axs[1],
                    **pattern1_kwargs_default,
                )

                if data2 is not None:
                    # axs[2].cla()
                    quickplot(
                        frame(yx, data2),
                        figax=axs[2],
                        **pattern2_kwargs_default,
                    )

                axs[0].cla()
                quickplot(
                    vImage,
                    figax=axs[0],
                    **vImage_kwargs_default,
                )

                rect = patches.Rectangle(
                    np.flip(yx) - 0.5, 1, 1,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                )

                # Add the patch to the Axes
                axs[0].add_patch(rect)
                fig.canvas.draw()

    plt.connect('button_press_event', on_click)

    plt.show()


def dark_mode(fig, axs, extras=None, cbars=None):
    """
    Set figure background to black and axes spines, ticks and labels to white.

    """

    axs = np.array(axs).flatten().tolist()
    if extras is not None:
        axs = axs + list(extras)

    for ax in axs:
        plt.setp(ax.spines.values(), color='white')
        ax.tick_params(color='white', labelcolor='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    if cbars is not None:
        for cbar in cbars:
            plt.setp(cbar.ax.spines.values(), color='white')
            cbar.ax.tick_params(color='white', labelcolor='white')
            cbar.ax.xaxis.label.set_color('white')
            cbar.ax.yaxis.label.set_color('white')

    fig.set_facecolor('black')
