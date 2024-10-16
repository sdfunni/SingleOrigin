import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, to_rgb, hsv_to_rgb

from SingleOrigin.plot import quickplot
from SingleOrigin.image import bin_data

pkg_dir, _ = os.path.split(__file__)


# %%


def plot_eds_spectrum(
        spectrum,
        elem_to_index=[],
        figax=False,
        text_offset=300,
        xlim=None,
        ylim=None,
        intensity_min=None,
        line_labeling='emission',
        seplim=None,
        axis_ratio=None,
):
    """
    Plot EDS spectrum loaded using temutils.image.emdVelox.

    Parameters
    ----------
    spectrum : dict
        Dictionary containing the spectrum counts, energy bin edges and
        centers as three numpy arrays.

    elem_to_index : list of strings
        Element abreviations for peaks in the spectrum that should be labeled.

    figax : bool or matplotlib.Axes object
        Whether to return the figure and axes objects for modification by user.
        OR the Axes object to plot the spectrum into.
        Default: False

    text_offset : scalar
        The vertical offset (in counts above the peak) to display the peak
        index label.
        Default: 200

    xlim, ylim : 2-list
        The x and y limits of the plotting window in data coordinates. If None,
        will size to show all data.
        Default: None.

    intensity_min : scalar
        Th minimum intensity peak that should be labeled. Peaks below this
        intensity will not be labeled even if other peaks from the same
        element are labeled.

    line_labeling : str ('elem' or 'emission')
        If 'elem', only the element will be used to label corresponding lines.
        If 'emission', the full X-ray emission nomenclature will be used.
        Default: 'emission'.

    seplim : scalar or None
        The minimum energy separation between plotted lines for each element's
        emissions. If two lines are closer than this, only the lower energy
        line will be plotted.

    axis_ratio : scalar or None
        If scalar, the aspect ratio of the hieght to width of the plot window
        in the Axes. NOT the ratio of data coordinate sizes like
        Axes.set_aspect(). If None, the aspect ratio is not fixed.
        Default: None.

    Returns
    -------
    None.

    """

    counts = spectrum['Counts']
    bins = spectrum['eV_bins']
    eV = spectrum['eV_cent']

    # For line separations from a single element, if less, combine
    if seplim is None:
        seplim = 3 * np.abs(np.diff(eV[:2]))   # 3 times the dispersion
    # Get minimum itensity for labeling lines, if not passed
    if intensity_min is None:
        intensity_min = 0.3 * np.mean(counts)

    # Load EDS energy table
    eds_table = pd.read_excel(
        os.path.join(pkg_dir, 'EDS_Energies.xlsx'))

    # Get EDS peak info for the elements to index
    # Simplify labeling if some lines absent and combine peaks if close
    peak_dict = {
        elem: {
            i: v.item()
            for (i, v)
            in eds_table[(eds_table.Element == elem)].loc[:, 'Kα1':].items()
            if not pd.isna(v.item())
        }
        for elem in elem_to_index}

    for elem, peaks in peak_dict.items():
        sublines = np.array(list(peaks.keys()))
        energies = np.array(list(peaks.values()))
        lines, lineinds = np.unique(
            [line[:2] for line in sublines],
            return_inverse=True
        )

        for i, line in enumerate(lines):
            inds = np.argwhere(lineinds == i)
            line_eVs = energies[inds].flatten()

            if line_eVs.shape[0] > 1:
                deV = np.abs(np.diff(line_eVs))

                if deV < seplim:
                    peaks = {k: v for k, v in peaks.items() if line not in k}

                    peaks[line] = np.min(line_eVs)

            else:
                peaks = {k: v for k, v in peaks.items() if line not in k}
                peaks[line] = line_eVs[0]

        sublines = np.array(list(peaks.keys()))
        energies = np.array(list(peaks.values()))

        shells, shellinds = np.unique(
            [line[:1] for line in sublines],
            return_inverse=True
        )

        for i, shell in enumerate(shells):
            inds = np.argwhere(shellinds == i)
            shell_eVs = energies[inds].flatten()

            if shell_eVs.shape[0] == 2:
                deV = np.abs(np.diff(shell_eVs))

                if deV < seplim:
                    peaks = {k: v for k, v in peaks.items()
                             if shell not in k}

                    peaks[shell] = np.min(shell_eVs)

            if shell_eVs.shape[0] == 1:
                peaks = {k: v for k, v in peaks.items() if shell not in k}
                peaks[shell] = shell_eVs[0]

        peak_dict[elem] = peaks

    # Plot the histogram
    if xlim is None:
        xlim = np.array([0, np.max(eV)]) / 1e3
        print(xlim)
    else:
        xlim = np.array(xlim)
    if ylim is None:
        ylim = np.array([0, np.max(counts) * 1.1])

    if isinstance(figax, bool):
        fig, ax = plt.subplots(1, figsize=(14, 7), layout='constrained')

    else:
        ax = figax

    if np.max(counts) < 2e3:
        ct_factor = 1
        ct_label = 'Counts'
    elif np.max(counts) < 2e6:
        ct_factor = 1e-3
        ct_label = r'Counts ($x \mathbf{10^-3}$)'
    else:
        ct_factor = 1e-6
        ct_label = r'Counts ($x \mathbf{10^-6}$)'

    ylim *= ct_factor
    ax.stairs(
        counts * ct_factor,
        edges=bins / 1e3,
        fill=True,
        ec='darkgreen',
        fc='green',
        lw=1,
    )

    # Plot emission lines
    for elem, lines in peak_dict.items():
        lines_offset = len(lines) * text_offset
        for line, energy in lines.items():
            line_ind = np.argmin(np.abs(eV - energy))
            peak_int = counts[line_ind]

            if ((energy / 1e3 > xlim[0]) & (energy / 1e3 < xlim[1])):

                if peak_int > intensity_min:
                    if peak_int * ct_factor > 0.95 * ylim[1]:
                        y = 0.5 * ylim[1] * ct_factor

                    else:
                        y = (peak_int + lines_offset) * ct_factor

                    ax.plot([energy/1e3, energy/1e3],
                            [0, peak_int * 0.9 * ct_factor],
                            c='black'
                            )

                    if line_labeling == 'emission':
                        label = elem + '-' + line
                    else:
                        label = elem

                    ax.text(
                        energy / 1e3, y,
                        label,
                        ha='center',
                        size=18,
                        weight='bold',
                        c='black'
                    )

                    lines_offset -= text_offset

    # Finalize plotting
    ax.set_xbound(*xlim)
    ax.set_ybound(*ylim)

    if axis_ratio is not None:
        lim_ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        ax.set_aspect(aspect=axis_ratio / lim_ratio)

    ax.set_xlabel('Energy (keV)', weight='bold', size=20)
    ax.set_ylabel(ct_label, weight='bold', size=20)
    # fig.canvas.draw()
    # fig.tight_layout()

    if figax is True:
        return fig, ax


def plot_eds_maps(
        data,
        nrows,
        ncols,
        orientation='horizontal',
        elem_colors={},
        bin_factor=1,
        figsize=5,
        haadf_text_kwargs={},
        elem_text_kwargs={},
):
    """
    Plot EDS elemental maps loaded using temutils.image.emdVelox.

    Parameters
    ----------
    data : dict
        Dictionary containing elemental maps and HAADF image or image stack.

    nrows, ncols : int
        Number of rows and colums to use for plotting the elemental maps. This
        does not include the HAADF image.

    orientation : str ('horizontal' or 'vertical')
        If 'horizontal', the elemental maps are plotted to the right of the
        HAADF image. If 'vertical', they are plotted below the HAADF image.

    elem_colors : dict
        Dictionary of matplotlib color strings (as the values) and keyed by
        corresponding elemental symbols. All symbols must be elemental maps
        in data. However, not all maps present in data must be included in the
        dictionary. If elem_colors is not an empty dict, only elemental maps
        corresponding to elements in elem_colors will be plotted.

    bin_factor : int
        Binning factor for the elemental maps.

    figsize : scalar
        Width of the figure if orientation is 'vertical', or height of the
        figure if 'horizontal'.

    haadf_text_kwargs : dict
        Dictionary of kwargs to be passed to plt.text for the HAADF image
        plot label.

    elem_text_kwargs
        Dictionary of kwargs to be passed to plt.text for the elemental map
        plot labels.

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    ax : dict
        Dictionary of matplotlib.Axes objects in the figure keyed by element
        symbol or 'HAADF'.


    """

    # Set text kwargs dicts
    haadf_text_default = {
        'x': 0.03,
        'y': 0.97,
        'color': 'white',
        'size': 24,
        'ha': 'left',
        'va': 'top',
    }

    haadf_text_default.update(haadf_text_kwargs)

    elem_text_default = {
        'x': 0.03,
        'y': 0.97,
        'color': 'white',
        'size': 24,
        'ha': 'left',
        'va': 'top',
    }

    elem_text_default.update(elem_text_kwargs)

    # Get plot rows, cols
    if orientation == 'horizontal':
        ncols += nrows
        nhaadf = nrows

    if orientation == 'vertical':
        nrows += ncols
        nhaadf = ncols

    # Prep HAADF image
    if len(data['HAADF'].shape) > 2:
        haadf = np.mean(data['HAADF'], axis=0)
    else:
        haadf = data['HAADF']

    if isinstance(figsize, (float, int)):
        aspect = haadf.shape[1] / haadf.shape[0]

        if orientation == 'horizontal':
            figsize = (aspect * ncols/nrows * figsize,
                       figsize
                       )
        elif orientation == 'vertical':
            figsize = (figsize,
                       1 / aspect * nrows/ncols * figsize,
                       )

    # Prep elemental maps
    if len(elem_colors) == 0:
        print('No elem_colors dict passed, plotting greyscale maps.')
        elems = [k for k in data.keys()
                 if not np.isin(k, ['HAADF', 'Spectrum'])]
        elems.sort()
        elem_colors = {k: 'grey' for k in elems}
    else:
        elems = list(elem_colors.keys())

    color_hsv = {
        el: rgb_to_hsv(to_rgb(color))
        for el, color in elem_colors.items()
    }

    data_ = {}
    for k in elems:
        data_[k] = bin_data(data[k], [bin_factor, bin_factor])

    max_ = np.max([np.max(data_[el]) for el in elem_colors.keys()])
    cmaps = {}
    for el, color in color_hsv.items():
        max_ = np.max(data_[el])
        hs = np.ones(data_[el].shape + (2,)) * color[:2].reshape((1, 1, -1))
        v = np.expand_dims(data_[el]/max_, axis=-1)
        cmaps[el] = hsv_to_rgb(np.concatenate((hs, v), axis=-1))

    fig = plt.figure(figsize=figsize, layout='tight')
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
    )

    axs = {}

    axs['haadf'] = fig.add_subplot(gs[0:nhaadf, 0:nhaadf])

    quickplot(haadf, cmap='gist_gray', figax=axs['haadf'])
    axs['haadf'].text(
        s='HAADF',
        transform=axs['haadf'].transAxes,
        **haadf_text_default,
    )

    for i, el in enumerate(elems):
        if orientation == 'horizontal':
            r, c = [i % nhaadf, nhaadf + i // nhaadf]
            axs[el] = fig.add_subplot(gs[r, c])

        if orientation == 'vertical':
            r, c = [nhaadf + i // nhaadf, i % nhaadf]
            axs[el] = fig.add_subplot(gs[r, c])

        axs[el].set_xticks([])
        axs[el].set_yticks([])

        axs[el].text(
            s=el,
            transform=axs[el].transAxes,
            **elem_text_default,
        )

        axs[el].imshow(cmaps[el])

    return fig, axs
