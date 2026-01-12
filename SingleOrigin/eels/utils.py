"""Module with utilities for analyzing and visualizing EELS data."""

import copy

import numpy as np

from scipy.optimize import minimize, least_squares

from scipy.ndimage import gaussian_filter

from scipy.fft import (rfft, irfft)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import psutil

from joblib import Parallel, delayed

from exspy.components import EELSCLEdge

from SingleOrigin.utils.fourier import hann1d_taper
from SingleOrigin.utils.environ import is_running_in_jupyter
from SingleOrigin.utils.peakfit import fit_gaussian_1d
from SingleOrigin.utils.mathfn import gaussian_1d

if is_running_in_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# %%


def get_energy_inds(select_eVs, eV):
    """
    Get the array index for the nearest energy bin(s) to the specified
    energy (energies).

    Parameters
    ----------
    select_eVs : scalar or 1D array-like
        The energies for which to find the matching index (indices).

    eV : scalar or array of scalars
        Energy losses for each bin in the spectrum axis.

    Returns
    -------
    inds : list
        The indices of the specified energy bins.

    """

    select_eVs = list(select_eVs)

    inds = []
    for select in select_eVs:
        inds += [np.argmin(np.abs(eV - select))]

    return inds


def plot_spectrum(
        counts,
        eV,
        energy_range=None,
        background_window=None,
        norm_scaling='max',
        counts_shift=0,
        figax=False,
        figsize=None,
        tight_layout=True,
        plot_kwargs={},
        arb_units=False,
        tight_max_min=False,
        xlabel=True,
        ylabel=True,
        fontsize=14,
):
    """
    Plot a spectrum from a SingleOrigin.eels.EELS_SIdata object.

    Parameters
    ----------
    counts : 1d array
        The spectrum counts.

    eV : 1d array
        The energy loss values of the spectrum bins.

    energy_range : 2-list or None.
        The start and stop energy values to display. If None, the entire
        spectrum will be plotted.
        Default: None.

    background_window : 2-list or None
        The start and stop energy values of the window for background
        subtraction. If passed, a background will be subtracted before
        plotting.
        Default: None

    norm_scaling : 'max', 'total', 2-list or None
        If 'max', normalizes the spectrum maximum to 1 in the plotted window.
        If 'total', normalizes so that the total counts in the spectrum is 1.
        If a 2-list, the mean counts over this energy range is normalized to 1.
        If None, no normailzation performed. Only None allows plotting of true
        counts on the y-axis; all others use arbitrary units.
        Default: 'max'

    counts_shift : scalar or None
        Count offset to apply to the spectrum. For plotting multiple spectra
        on one plot. For plots with a normalization active, use a fractional
        offset, e.g. 0.3. If None, no offset applied.
        Default: None.

    figax : matplotlib Axes object or bool
        If a Axes, plots in this Axes. If bool, whether to return the Figure
        and Axes objects.

    figsize : 2-tuple or None
        The figure size.
        Default: None.x

    tight_layout : bool
        Whether to use tight_layout.
        Default: True

    plot_kwargs : dict
        Dictionary of keyword arguments to pass to matplotlib.pyplot.plot().
        Default: {}.

    arb_units : bool
        Whether to force arbitrary units for the y-axis.
        Default: False.

    tight_max_min : bool
        Whether to force a tight minimum and maximum of the y-limits around
        the plotted count range. y-axis may not include zero if True.
        Default: False.

    xlabel, ylabel : bool
        Whether to display the x-axis and y-axis labels, respectively.
        Default: True.

    fontsize : scalar
        The font size to use for axis labels and tick labels.
        Default: 14.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
        Optional: only returned if figax is True.

    """

    if background_window is not None:
        counts = subtract_background(
            counts,
            eV,
            background_window
        )
    deV = eV[1] - eV[0]
    eV_ = np.concatenate([eV, [eV[-1] + deV]])

    if isinstance(figax, bool):
        fig, ax = plt.subplots(1, figsize=figsize, tight_layout=tight_layout)

    else:
        ax = figax

    if norm_scaling is not None:
        if norm_scaling == 'max':
            counts /= np.nanmax(counts)
        elif norm_scaling == 'total':
            counts /= np.sum(counts)
        elif isinstance(norm_scaling, list):
            start_ind = np.argmin(
                np.abs(eV - norm_scaling[0]))
            stop_ind = np.argmin(np.abs(eV - norm_scaling[1]))
            counts /= np.mean(counts[start_ind:stop_ind])

        else:
            counts /= norm_scaling
        arb_units = True
        decades = 0

    else:
        maxCounts = np.nanmax(counts)
        # decades = 10**int(np.log10(maxCounts) // 3 * 3)
        decades = 0

    if counts_shift != 0:
        counts += counts_shift

    if energy_range is not None:
        ax.set_xlim(energy_range)

        start_ind = np.argmin(np.abs(eV - energy_range[0]))
        stop_ind = np.argmin(np.abs(eV - energy_range[1]))
        counts = counts[start_ind:stop_ind]
        eV = eV[start_ind:stop_ind]

    maxCounts = np.nanmax(counts)
    ax.set_ylim(0, maxCounts/10**decades * 1.1)

    ax.stairs(
        counts/10**decades,
        eV_,
        **plot_kwargs
    )

    if xlabel:
        ax.set_xlabel('Energy Loss (eV)', weight='bold', fontsize=fontsize)

    ax.tick_params(axis='x', labelsize=fontsize)

    if ylabel:
        if arb_units:
            ax.set_ylabel('Counts (a.u.)', weight='bold', fontsize=fontsize)
            ax.set_ylim(-0.1, 1.1 + counts_shift)
            ax.set_yticks([])
        else:
            ax.set_ylabel(f'Counts x 10$^{decades}$', weight='bold',
                          fontsize=fontsize)
    else:
        ax.set_yticks([])

    pad = maxCounts/10**decades * 0.05
    ax.set_ylim(-pad, maxCounts/10**decades + pad)

    if figax is True:
        return fig, ax


def plot_eels_fit(
        eV,
        models,
        bkgd_prms,
        weights,
        whitelines,
        labels=None,
        total_window=None,
        figax=False,
        colors=None,
):
    """
    Plot an EELS spectrum fit.

    Parameters
    ----------
    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    models : list of 1D arrays
        EELS edge models.

    bkgd_prms : array of shape (n, 2)
        [A, r] parameters of the power law background fit(s).

    weights : list of scalars
        Weights of each EELS edge.

    whitelines : list of 2D arrays or None
        For each model, either a 2D array or None (if no white lines). Array(s)
        have shape: (n_lines, 3).

    total_window : array of shape (n, 2)
        The start, stop eVs of the background fit(s). Must be the same shape as
        bkgd_prms. If None, background fit will be plotted over the whole
        energy range.
        Default: None.

    figax : matplotlib Axes object or bool
        If a Axes, plots in this Axes. If bool, whether to return the Figure
        and Axes objects.

    colors : list or None
        Colors to use for plotting the fitted edge models. These will be
        applied in energy loss order, from least to greatest. If None, a
        default list of colors will be used.
        Default: None.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
        Optional: only returned if figax is True.

    """

    if colors is None:
        colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:purple',
                  'tab:pink', 'tab:olive', 'tab:cyan', 'tab:brown', 'tab:grey']
    if total_window is None:
        total_window = np.array([[eV[0], eV[-1]] * len(models)])

    bkgdmask = [np.ones(eV.shape)] * len(models)

    if total_window is not None:
        # bkgdinds = get_energy_inds(total_window.flatten()).reshape((-1, 2))
        for i, wind in enumerate(total_window):
            bkgdmask[i] = np.where(((eV > wind[0]) & (eV < wind[1])),
                                   1, np.nan)
    fits = [[power_law(eV, *bp) * bkgdmask[i]]
            for i, bp in enumerate(bkgd_prms)]

    for g, fit in enumerate(fits):
        for i, model in enumerate(models[g]):
            m = model * weights[g][i]
            if whitelines is not None:
                if whitelines[g][i] is not None:
                    for line in whitelines[g][i]:
                        m += gaussian_1d(eV, *line)

            maxval = np.nanmax(m)
            startind = np.argmax(m > maxval*1e-6) - 1

            m[:startind] = np.nan

            fit += [fit[-1] + m]

    if isinstance(figax, bool):
        fig, ax = plt.subplots()
    else:
        ax = figax

    k = 0
    for i, fit in enumerate(fits):
        for j, comp in enumerate(fit):
            if j == 0:
                c = 'black'
                label = None
            else:
                c = colors[k]
                if labels is not None:
                    label = labels[i][j-1]
                else:
                    label = None

                k += 1

            ax.plot(eV, fit[j], c=c, label=label)

    ax.set_xlabel('Energy Loss (eV)', weight='bold')
    ax.set_ylabel('Counts (arb. units)', weight='bold')

    ax.legend()

    if figax is True:
        return fig, ax


def power_law(eV, A, r, b=0):
    """
    (Inverse) power law mathematical function for fitting EELS backgrounds.
    A * eV ^ (-r)

    Parameters
    ----------
    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    A : scalar
        The prefactor value.

    r : scalar
        The (positive) power value.

    b : scalar
        Background offset.

    Returns
    -------
    I_b : scalar or array
        Intensity value(s) corresponding to eV. Same shape as 'eV'.

    """

    ndims = len(A.shape)
    A, r = [np.array(A), np.array(r)]
    I_b = A[..., None] * (eV[*(None,)*ndims, ...]) ** (-r[..., None]) + b

    I_b = np.squeeze(I_b)

    return I_b


def exponential(eV, A, b, C=0, d=0, e=0):
    """
    Exponential model mathematical function for fitting EELS backgrounds.
    A * exp(b * eV)

    Parameters
    ----------
    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    A : scalar
        The prefactor value.

    b : scalar
        The (positive) power value.

    Returns
    -------
    I_b : scalar or array
        Intensity value(s) corresponding to eV. Same shape as 'eV'.

    """

    ndims = len(A.shape)
    C, d = [np.array(C), np.array(d)]
    I_b = A[..., None] * np.exp((eV[*(None,)*ndims, ...]) * (-b[..., None])) \
        + C[..., None] * np.exp((eV[*(None,)*ndims, ...]) * (-d[..., None])) \
        + e

    I_b = np.squeeze(I_b)

    return I_b


def pl_ss(params, spectrum, eV):
    """
    Objective function for fitting an inverse power law to EELS background.

    Parameters
    ----------
    params : 2-list of scalars
        The fitting parameters: A, r.

    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    Returns
    -------
    chi_sq : scalar
        The sum squared difference between the data and fit given the
        parameters.

    """

    fit = power_law(eV, *params)

    chi_sq = np.nansum((spectrum - fit) ** 2)

    return chi_sq


def exp_ss(params, spectrum, eV):
    """
    Objective function for fitting an exponential function to EELS background.

    Parameters
    ----------
    params : 2-list of scalars
        The fitting parameters: A, r.

    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    Returns
    -------
    chi_sq : scalar
        The sum squared difference between the data and fit given the
        parameters.

    """

    fit = exponential(eV, *params)

    chi_sq = np.nansum((spectrum - fit) ** 2)

    return chi_sq


def fit_background(
        spectrum,
        eV,
        window,
        model='pwr',
        plot=False,
        fitmask=None
):
    """
    Fit a single EELS background given a fitting window.

    Parameters
    ----------
    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    window : 2-list
        The start and stop energy of the fitting window.

    model : str
        The background model to use. 1 or 2 term power law or exponential
        functions are allowed, each with or without a constant offset.
        A one-term power law background without constant offset is the typical
        background function and should be preferred. For some spectra in the
        low loss region or if improper gain / dark references were used, other
        options may provide a better fit. One-term functions can be specified
        by 'pwr' or 'exp'; add a following '2' for two-term functions. Add a
        'c' as the last character to include a constant offset in the fit.
        e.g.: 'pwr2c' will give a two-term power law function with constant
        offset.
        Default: 'pwr'

    plot : bool
        Whether to plot the spectrum, fit and background-subtracted spectrum.
        Default: True.

    fitmask : 1D array or None
        0 elements indicate energy bins that should not be included in the fit,
        1 elsewhere. This can be used to remove specific parts of the spectrum
        from background fitting in the case of detector artifacts.
        Default: None.

    Returns
    -------
    params : array of shape (2,)
        The [A, r] inverse power law parameters of the fit.

    """

    d_eV = eV[1] - eV[0]
    start_ind = np.nanargmin(np.abs(eV - window[0]))
    stop_ind = np.nanargmin(np.abs(eV - window[1]))

    spectrum_ = copy.deepcopy(spectrum[start_ind:stop_ind])
    if fitmask is not None:
        spectrum_[fitmask[start_ind:stop_ind] == 0] = np.nan

    eV_ = eV[start_ind:stop_ind]

    if model[:3] == 'pwr':
        # Get initial guess from two window method:
        d = stop_ind - start_ind
        d_half = d // 2
        w1 = [start_ind, start_ind + d_half]
        w2 = [stop_ind - d_half, stop_ind]
        E1 = window[0]
        E2 = window[1]

        I1 = np.nansum(spectrum[w1[0]:w1[1]]) * d_eV
        I2 = np.nansum(spectrum[w2[0]:w2[1]]) * d_eV

        if I1 > I2:
            r0 = 2 * np.log(I1/I2) / np.log(E2/E1)
        else:
            r0 = 2

        A0 = (1 - r0) * (I1 + I2) / (E2**(1-r0) - E1**(1-r0))
        if A0 <= 0:
            A0 = 1

        p0 = [A0, r0]
        bounds = [(0, np.inf), (1, 20)]

        if model[:4] == 'pwr2':
            p0 += [0, p0[-1]]
            bounds = [bnd * 2 for bnd in bounds]

        if model[-1] == 'b':
            p0 += [0]
            bounds = [bounds[0] + (0,), bounds[1] + (np.inf,)]

        params = minimize(
            pl_ss,
            p0,
            bounds=bounds,
            args=(spectrum_, eV_),
            method='L-BFGS-B',
        ).x

        bkgd_fit = power_law(eV, *params)

    if model[:3] == 'exp':
        p0 = [np.nanmax(spectrum_), 0.01]
        bounds = [(0, None), (0, 1)]

        if model[:4] == 'exp2':
            p0 += [0, p0[-1]]
            bounds *= 2

        if model[-1] == 'b':
            p0 += [0]
            bounds += [(0, None)]

        params = minimize(
            exp_ss,
            p0,
            bounds=None,
            args=(spectrum_, eV_),
            method='Powell',
        ).x

        bkgd_fit = exponential(eV, *params)

    spec_bkgdrmv = spectrum - bkgd_fit

    if plot:
        fig, ax = plt.subplots()
        ax.fill_between(eV, spectrum,
                        step='mid', color='cadetblue', alpha=0.85)
        ax.plot(eV, bkgd_fit, color='darkred', lw=0.5)
        ax.plot(eV, spec_bkgdrmv, color='teal')
        rect = Rectangle([window[0], 0],
                         width=window[1]-window[0],
                         height=np.max(spectrum)*1.1,
                         alpha=0.2, color='darkred', )
        outline = Rectangle([window[0], np.max(spectrum)*-0.1],
                            width=window[1]-window[0],
                            height=np.max(spectrum)*1.1,
                            alpha=0.8, color='darkred',  fill=False)

        ax.plot([eV[0], eV[-1]], [0, 0], lw=1, color='black')
        ax.add_patch(rect)
        ax.add_patch(outline)
        ax.set_ylim(np.max(spectrum) * -0.05, np.max(spectrum))
        ax.set_xlim(eV[0]-1, eV[-1])
        ax.set_ylabel('Counts', fontweight='bold')
        ax.set_xlabel('Energy Loss (eV)', fontweight='bold')

        plt.title("Background Fit")

    if plot:
        return params, fig, ax

    else:
        return params


def subtract_background(
        spectrum,
        eV,
        window,
        model='pwr',
        return_params=False,
):
    """
    Fit a single EELS background given a fitting window and return the
    background-subtracted spectrum.

    Parameters
    ----------
    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    window : 2-list
        The start and stop energy of the fitting window.

    model : str
        The background model to use. 1 or 2 term power law or exponential
        functions are allowed, each with or without a constant offset.
        A one-term power law background without constant offset is the typical
        background function and should be preferred. For some spectra in the
        low loss region or if improper gain / dark references were used, other
        options may provide a better fit. One-term functions can be specified
        by 'pwr' or 'exp'; add a following '2' for two-term functions. Add a
        'c' as the last character to include a constant offset in the fit.
        e.g.: 'pwr2c' will give a two-term power law function with constant
        offset.
        Default: 'pwr'

    return_params : bool
        Whether to return the parameters of the power law background fit.

    Returns
    -------
    spectrum_sub : array
        The background-subtracted spectrum.

    params : array of shape (2,)
        The A, r parameters of the background fit.

    """

    params = fit_background(spectrum, eV, window, model=model, plot=False)

    if model[:3] == 'pwr':
        bkgd = power_law(eV, *params)
    elif model[:3] == 'exp':
        bkgd = exponential(eV, *params)

    spectrum_sub = spectrum - bkgd

    start_ind = np.argmin(np.abs(eV - window[0]))
    spectrum_sub[..., :start_ind] = 0
    if return_params:
        return spectrum_sub, params
    else:
        return spectrum_sub


def integrate_edge(spectrum, eV, int_window, bkgd_window=None):
    """
    Integrate an EELS edge intensity over a specified range with optional
    background subtraction.

    Parameters
    ----------
    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    int_window : 2-list
        The start and stop energy of the signal integration window.

    bkgd_window : 2-list or None
        The start and stop energy of the background fitting window. If
        None, no background is subtracted.

    Returns
    -------
    eels_map : array
        The background-subtracted spectrum image.

    """

    if bkgd_window is not None:
        spectrum_sub = subtract_background(
            spectrum,
            eV,
            bkgd_window
        )

    else:
        spectrum_sub = spectrum

    start_ind = np.argmin(np.abs(eV - int_window[0]))
    stop_ind = np.argmin(np.abs(eV - int_window[1]))
    edge_int = np.sum(spectrum_sub[start_ind:stop_ind])

    return edge_int


def subtract_background_SI(si, eV, window, lba=None):
    """
    Fit EELS background for each pixel in a spectrum image, given a fitting
    window and return the background-subtracted spectrum image.

    Parameters
    ----------
    si : str or None
        Which spectrum image to use: 'SI' (single EELS), 'SI_ll' (dual EELS
        low loss) or 'SI_hl' (dual EELS high loss). If None, will default
        to 'SI_hl' for dual EELS or 'SI' in the case of single EELS.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    window : 2-list
        The start and stop energy of the fitting window.

    lba : int or None.
        Local background averaging (LBA) sigma. Used for locally averaging
        spectra for background fitting. Will produce smoother maps but
        inherently reduces the quantitative, and in some cases, the qualitative
        correctness of the analysis.
        *** Generally, NOT RECOMMENDED.
        If None, no LBA is applied.
        Default: None.

    Returns
    -------
    data_sub : array
        The background-subtracted spectrum image.

    """

    h, w, _ = si.shape

    xy = np.array([[i, j] for i in range(h) for j in range(w)])

    n_jobs = psutil.cpu_count(logical=True)

    if lba is not None:
        si_fitting = gaussian_filter(si, sigma=lba, axes=(0, 1))

    else:
        si_fitting = si

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_background)(
            si_fitting[i, j], eV, window
        ) for i, j in tqdm(xy)
    )

    params = np.array(results).reshape(h, w, -1)

    si_sub = si - power_law(eV, params[..., 0], params[..., 1])

    start_ind = np.argmin(np.abs(eV - window[0]))
    si_sub[..., :start_ind] = 0

    return si_sub, params


def get_thickness(spectrum, eV, zlp_cutoff=3):
    """
    Calculate the thickness of a sample relative to the mean free path of beam
    electrons, using the log-ratio method.

    Parameters
    ----------
    spectrum : 1D array
        The counts for each EEL bin. Must contain the zero loss peak.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

     zlp_cutoff : scalar or None
         The cutoff energy between the ZLP and low loss spectrum. If None,
         will be found as the first minimum in the spectrum above the ZLP.


    Returns
    -------
    thickness : scalar
        The thickness relative to the mean free path.

    """

    if (np.min(eV) < 0) & (np.max(eV) > 0):
        pass
    else:
        raise Exception('Low loss spectra does not include ZLP.')

    zlp_cutoff = np.argmin(np.abs(eV - zlp_cutoff))

    I0 = np.sum(spectrum[:zlp_cutoff])
    It = np.sum(spectrum)

    thickness = np.log(It/I0)

    return thickness


def get_zlp_cutoff(spec, eV):
    """
    Find spectrum minima between the ZLP and plasmon scattering.

    Parameters
    ----------
    spec : ndarray
        EEL spectrum or spectrum image for which to find the cutoff between the
        ZLP and plasmon scattering.

    eV : 1d array
        The energy of each bin in the spectrum.

    Returns
    -------
    zlp_cutoff : scalar
        The energy of the counts minimum after the ZLP.

    """

    scan_dims = len(spec.shape) - 1
    if scan_dims > 0:
        spec_total = np.sum(spec, axis=tuple([i for i in range(scan_dims)]))
    else:
        spec_total = spec

    spec_de = np.array([
        spec_total[i] - spec_total[i+1]
        for i in range(spec_total.shape[0] - 1)
    ])

    spec_de[:np.argmax(spec_total)] = 1

    zlp_cutoff = eV[np.argmax(spec_de < 0) - 1]

    return zlp_cutoff


def fit_zlp(spec, eV, bounds=None):
    """
    Fit the ZLP in and EELS spectrum with a 1d gaussian.

    Parameters
    ----------
    spec : 1d array
        The counts for each energy bin.

    eV : 1d array
        The center energy of each bin.

    bounds : None or a 3-list of 2-tuples
        If None, no bounds are used for fitting. Otherwise is the upper and
        lower limits for the fitting parameters: current ZLP position, ZLP
        width, ZLP intensity maximum.

    Returns
    -------
    params : 3-list
        The fitted gaussian parameters.

    """

    p0 = [eV[np.argmax(spec)],
          0.5,
          np.max(spec)
          ]

    params = fit_gaussian_1d(
        eV,
        spec,
        p0,
        method='L-BFGS-B',
        bounds=bounds
    )

    return params


def get_zlp_fwhm(spec, eV):
    """
    Measure the FWHM of the ZLP.

    Parameters
    ----------
    spec : 1d array
        The counts for each energy bin.

    eV : 1d array
        The center energy of each bin.

    Returns
    -------
    fwhm : scalar
        The full width at half-maximum of the ZLP.

    """

    if len(spec.shape) > 1:
        spec = np.sum(
            spec,
            axis=tuple([i for i in range(len(spec.shape) - 1)])
        )

    max_ = np.nanmax(spec)
    maxind = np.nanargmax(spec)

    _eV = eV[:maxind]
    _spec = spec[:maxind]

    lowhm = np.interp([max_/2], _spec, _eV)

    eV_ = eV[maxind:]
    spec_ = spec[maxind:]

    highhm = np.interp([max_/2], np.flip(spec_), np.flip(eV_))

    fwhm = highhm - lowhm

    return fwhm


def fourier_ratio_deconvolution(hl, ll, lleV, hann_taper=None):
    """
    Remove multiple scattering from a spectrum by deconvolution of the low
    loss spectrum. Subtract background first!!!

    Parameters
    ----------
    hl : ndarray
        EEL spectrum or spectrum image from which the multiple scattering
        should be removed. Backgroudn subtraction should be done first.

    ll : ndarray
        Low loss EEL spectrum or spectrum image. Must include the ZLP.

    lleV : 1d array
        The energy of each bin in the low loss spectrum.

    hann_taper : scalar or None
        The number of bins over which to apply a Hann taper on each end of the
        spectra. This prevents wrap-around effects from producing artifacts in
        the final spectra. If None, 5% of the spectrum length is tapered on
        each end.

    Returns
    -------
    hl_deconv : ndarray
        The single scattering (i.e. deconvolved) high loss spectrum.

    """

    zlp_cutoff = get_zlp_cutoff(ll, lleV)
    cutoff_ind = np.argmin(np.abs(lleV - zlp_cutoff))

    spec_len = hl.shape[-1]
    scan_dims = len(hl.shape) - 1

    if hann_taper is None:
        hann_taper = spec_len*0.05

    hann_taper = hann1d_taper(spec_len, n_taper=hann_taper
                              )[*(None,)*scan_dims, ...]

    hl *= hann_taper
    ll *= hann_taper

    I0 = np.sum(ll[..., :cutoff_ind], axis=-1)[..., None]

    fwhm = get_zlp_fwhm(
        np.sum(ll, axis=tuple([i for i in range(scan_dims)])),
        lleV)

    zlp = gaussian_1d(lleV, 0, fwhm*0.6, 1)[*(None,)*scan_dims, ...]

    z = rfft(zlp, n=2*spec_len,)
    jk = rfft(hl, n=2*spec_len, axis=-1)
    jl = rfft(ll, n=2*spec_len, axis=-1)

    hl_deconv = (irfft(z * jk / jl, axis=-1) * I0)[..., :spec_len]

    return hl_deconv


# def fourier_log_deconvolution(spec, eV, hann_taper=None):
#     """
#     Remove multiple scattering from a spectrum by deconvolution of the low
#     loss spectrum. Subtract background first!!!

#     Parameters
#     ----------
#     hl : ndarray
#         EEL spectrum or spectrum image from which the multiple scattering
#         should be removed. Backgroudn subtraction should be done first.

#     ll : ndarray
#         Low loss EEL spectrum or spectrum image. Must include the ZLP.

#     lleV : 1d array
#         The energy of each bin in the low loss spectrum.

#     hann_taper : scalar or None
#         The number of bins over which to apply a Hann taper on each end of the
#         spectra. This prevents wrap-around effects from producing artifacts in
#         the final spectra. If None, 5% of the spectrum length is tapered on
#         each end.

#     Returns
#     -------
#     hl_deconv : ndarray
#         The single scattering (i.e. deconvolved) high loss spectrum.

#     """

#     zlp_cutoff = get_zlp_cutoff(spec, eV)
#     cutoff_ind = np.argmin(np.abs(eV - zlp_cutoff))

#     spec_len = spec.shape[-1]
#     scan_dims = len(spec.shape) - 1

#     if hann_taper is None:
#         hann_taper = spec_len*0.05

#     hann_taper = hann1d_taper(spec_len, n_taper=hann_taper
#                               )[*(None,)*scan_dims, ...]

#     spec *= hann_taper

#     I0 = np.sum(ll[..., :cutoff_ind], axis=-1)[..., None]

#     fwhm = get_zlp_fwhm(
#         np.sum(spec, axis=tuple([i for i in range(scan_dims)])),
#         eV)

#     zlp = gaussian_1d(eV, 0, fwhm*0.6, 1)[*(None,)*scan_dims, ...]
#     zlp /= np.sum(zlp)

#     z = rfft(I0*zlp, n=2*spec_len,)
#     jk = rfft(spec, n=2*spec_len, axis=-1)

#     hl_deconv = irfft(z * np.log(jk/z))

#     return spec_deconv


def second_diff(counts, dbins=1, sigma=1):
    """
    Calculate second difference of a spectrum.

    Parameters
    ----------
    counts : 1D array
        The counts for each EEL bin in the spectrum.

    dbins : int
        Number of bins to shift the spectrum when taking the second difference.

    sigma : scalar
        Gaussian sigma used to smooth the spectrum before taking the second
        difference.

    Returns
    -------
    sd : 1D array
        The second difference of the input spectrum.

    """

    spec = gaussian_filter(counts, sigma)

    sd = spec[2*dbins:] - 2*spec[dbins:-dbins] + spec[:-2*dbins]

    return sd


def modelfit_res(p0, spectrum, eV, components):
    """
    Objective function for fitting multiple reference spectra to a target
    spectrum using linear least squares.

    Parameters
    ----------
    p0 : 2-list of scalars
        The fitting parameters: A, r (power law background); edge weights;
        x0, sigma, amplitude for each white line gaussian fit.

    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    components : spectrum : 1D array
        The intensity of each .

    Returns
    -------
    r : 1D array
        The residuals array for the spectrum and current fit.

    """

    model = np.zeros(eV.shape)

    edges = np.array(components['edges'])
    n_edges = len(edges)
    m0 = p0[:n_edges]

    model += np.nansum(m0[..., None] * np.array(edges), axis=0)

    if 'whitelines' in components.keys():
        for i in range(len(components['whitelines'])):
            wl0 = p0[n_edges + 3*i: n_edges + 3*(i + 1)]
            model += gaussian_1d(eV, *wl0)
    r = spectrum - model
    return r


def modelfit(
        spectrum,
        eV,
        edges,
        E0,
        alpha,
        beta,
        GOS='dirac',
        energy_shifts=None,
        whitelines=None,
        bkgd_window=None,
        fit_window=None,
        return_components=True,
        return_parameter_keys=False,
        return_nanmask=True,
        plot=True,
        figax=False,
        model_colors=None,
):
    """
    Fit model components (background, edge(s) and white lines) to a single
    EELS spectrum.

    Parameters
    ----------
    spectrum : 1D array
        The spectrum counts.

    eV : 1D array
        Energy losses for each data point / bin of 'spectrum'.

    edges : list of 1D arrays or list of strings
        Model edge(s) or edge label(s) to be fit to 'spectrum'. Multiple edges
        can be simultaneously fit to a single spectrum assuming close or
        overlapping edges and/or the post edge backgrounds of lower energy
        edges model the spectrum well up to subsequent edges.

    E0 : scalar
        The accelerating voltage in kV.

    alpha : scalar
        The convergence semi-angle in mrad.

    beta : scalar
        The collection semi-angle in mrad.

    GOS : str
        The edge (or generalized oscillator strength) model type to use:
        'dft' or 'dirac'. This function uses the edge model calculation in
        exspy. Not all edges for all elements are included in the exspy
        element dictionary, but the underlying models are present in both
        databases. As a result, the library  may need to be modified by the
        user for less common edges. The file is 'elements.py' and can be
        found in the exspy library in your environment.
        Default: 'dirac'

    energy_shifts : list of scalars or None
        The energy shift to apply to each model edge to better match the
        experimental edge onset. If None, no offset(s) applied. If more
        than one edge is being fit, pass 0 in this list for any edges that
        should not be shifted. Order must match the order of 'edges'.
        Default: None

    whitelines : list of scalars or None
        The approximate energy loss of white line peaks to be fit with
        gaussians. Prevents model edge intensity from being fit to near edge
        structure not accounted for in the isolated atom model. The gaussian
        fits are also useful for measuring energy shifts of the white lines
        with oxidation state / local environment.
        Default: None.

    bkgd_window : 2-list or None
        Start and stop energy loss for power law background fitting. If None,
        no attempt is made to account for the background; it is assumed that
        a background has been pre-subtracted.
        Default: None.

    fit_window : 2-list or None
        Start and stop energy loss for fitting the model to the experimental
        spectrum. If None, the model is fit up to the highest energy loss in
        the spectrum. If subsequent edges are not to be simultaneously fit,
        the window should end before any additional edges.
        Default: None.

    return_components : bool
        Whether to return the edge models (i.e. the spectrum components)
        Default: True

    return_parameter_keys
        Whether to return the a list of strings identifying the elements of the
        fitting parameter list.
        Default: False

    return_nanmask : bool
        Wether to return the masks used to isolate the total fitting window.
        Default: True.

    plot : bool
        Whether to plot the spectrum and model fits (background and edges).
        Default: True.

    figax : matplotlib Axes object or bool
        If a Axes, plots in this Axes. If bool, whether to return the Figure
        and Axes objects.
        Default: False.

    model_colors : list or None
        Colors to use for plotting the fitted edge models. These will be
        applied in energy loss order, from least to greatest. If None, a
        default list of colors will be used.
        Default: None.

    Returns
    -------
    params : 1D array
        The fitted parameters in the following order: 2-parameter power law
        background (if fit), single weight for each edge and 3-parameter
        gaussian(s) for white line(s) (if fit).

    components : list of arrays (optionally)
        The edge models used for fitting (i.e. the spectrum components).

    pkeys : list of str (optionally)
        The keys describing each element of the 'params' vector.

    nanmask : array (optionally)
        The masks used to isolate the total iftting window.

    figax : matplotlib Axes object (optionally)


    """

    edges = list(edges)
    if isinstance(edges[0], str):
        # Calculate the edge models
        models = []
        if energy_shifts is None:
            energy_shifts = [0] * len(edges)
        elif isinstance(energy_shifts, (int, float)):
            energy_shifts = [energy_shifts]

        for i, edge in enumerate(edges):
            elem, shell = edge.split('-')
            models += [get_edge_model(
                elem,
                shell,
                eV=eV,
                shift=energy_shifts[i],
                GOS=GOS,
                E0=E0,
                alpha=alpha,
                beta=beta,
            )]

    else:
        models = edges

    components = {'edges': np.array(models)}

    bounds = [*[(0, np.inf)]*(len(models))]
    p0 = [1]*len(models)
    pkeys = ['r', 'A'] + [f'edge{i}' for i in range(len(models))]

    if bkgd_window is not None:
        spectrum_, b0 = subtract_background(
            spectrum, eV, window=bkgd_window, return_params=True,
        )

        # pkeys = ['r', 'A'] + pkeys

    else:
        spectrum_ = copy.deepcopy(spectrum)
        b0 = np.array([0, 0])

    if fit_window is not None:
        if bkgd_window is None:
            start = fit_window[0]
        else:
            start = bkgd_window[0]
        stop = fit_window[1]

        spectrum_[((eV < start) | (eV > stop))] = np.nan

    nanmask = np.invert(np.isnan(spectrum_))
    spectrum_ = spectrum_[nanmask]
    eV_ = eV[nanmask]

    if whitelines is not None:

        components['whitelines'] = whitelines
        for i, x0 in enumerate(whitelines):
            p0 = np.concatenate(
                (p0, [x0, 1, np.nanmax(spectrum)/5])
            )

            pkeys += [f'x0_{i}', f'sig_{i}', f'A_{i}']

            bounds += [(x0-5, x0+5), (0.5, 10), (0, np.inf)]

    bounds = ([bound[0] for bound in bounds],
              [bound[1] for bound in bounds])
    components_ = copy.deepcopy(components)
    components_['edges'] = components_['edges'][:, nanmask]

    result = least_squares(
        modelfit_res,
        p0,
        args=(spectrum_, eV_, components_),
        bounds=bounds,
    )

    params = result.x
    # vecR = result.fun

    params = np.concatenate([b0, params])

    if plot:
        if whitelines is not None:
            nlines = len(whitelines)
            wl_params = params[-3*nlines:].reshape((-1, 3))
            onsets = eV[np.argmax(np.array(models) > 0, axis=1)]
            wledges = np.array([np.max(np.argwhere(line > onsets - 5))
                                for line in whitelines])

            whitelines = []
            for i in range(len(components['edges'])):
                if i in wledges:
                    inds = np.argwhere(wledges == i).squeeze()
                    whitelines += [wl_params[inds]]
                    if len(whitelines[-1].shape) == 1:
                        whitelines[-1] = whitelines[-1][None, ...]
                else:
                    whitelines += [None]

            whitelines = [whitelines]

        else:
            nlines = 0

        if plot is True:
            fig, ax = plt.subplots(1)
        elif isinstance(plot, mpl.axes.Axes):
            ax = plot

        if bkgd_window is None or fit_window is None:
            total_window = [eV[0], eV[-1]]
            if bkgd_window is not None:
                total_window[0] = bkgd_window[0]
            if fit_window is not None:
                total_window[1] = fit_window[1]

        else:
            total_window = [bkgd_window[0], fit_window[1]]

        ax.plot(eV, spectrum, zorder=0, c='tab:blue')
        plot_eels_fit(
            eV=eV,
            models=[components['edges']],
            labels=[edges],
            bkgd_prms=[params[:2]],
            weights=[params[2:len(params) - nlines*3]],
            whitelines=whitelines,
            total_window=np.array([total_window]),
            figax=ax,
            colors=model_colors,
        )

    ret = [params]

    if return_components:
        ret += [components]
    if return_parameter_keys:
        ret += [pkeys]
    if return_nanmask:
        ret += [nanmask]
    # ret += [vecR]
    if figax is True:
        ret += [[fig, ax]]

    return ret


def get_edge_model(
        element,
        shell,
        E0,
        alpha,
        beta,
        eV,
        shift=0,
        GOS='dirac',
):
    """
    Calculate an edge model.

    Parameters
    ----------
    element : str
        The elemental abbreviation.

    shell : str
        The shell to model (i.e. 'K', 'L', 'M', 'N', 'O', 'P'...)

    E0 : scalar
        The accelerating voltage in kV.

    alpha : scalar
        The convergence semi-angle in mrad.

    beta : scalar
        The collection semi-angle in mrad.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    shift : scalar
        Energy value by which to shift the edge.
        Default: 0.

    GOS : str
        The edge model type to use: 'dft' or 'dirac'. This function uses
        the edge model calculation in exspy. Not all edges for all
        elements are included in the exspy element dictionary, but the
        underlying models are present in both GOS's. As a result, the
        library  may need to be modified by the user for less common edges.
        It is called 'elements.py' and can be found in the exspy library
        in your environment.

    Returns
    -------
    params : 1D array
        The fitted parameters in the following order: 2-parameter power law
        background (if fit), single weight for each edge and 3-parameter
        gaussian(s) for white line(s) (if fit).

    pkeys : list of str
        The keys describing each parameter

    """

    microscope_parameters = {
        'E0': E0, 'alpha': alpha, 'beta': beta, 'energy_scale': (eV[1] - eV[0])
    }

    subshell_dict = {
        'K': 'K',
        'L': ['L3', 'L2', 'L1'],
        'M': ['M5', 'M4', 'M3', 'M2',],
        'N': ['N5', 'N4'],  # 'N3', 'N2', 'N1'],
        'O': ['O5', 'O4', 'O3', 'O2', 'O1'],
    }

    model = np.zeros(eV.shape)
    edge_found = False

    for subshell in subshell_dict[shell]:
        try:
            model_sub = EELSCLEdge(
                element_subshell=element + '_' + subshell,
                GOS=GOS,
            )

            model_sub.set_microscope_parameters(**microscope_parameters)
            model += model_sub.function(eV - shift)

            edge_found = True

        except ValueError:
            print(f'{subshell} not found, skipping')

    if not edge_found:
        raise Exception(
            'No subshells found for requested shell.'
            'Use a different shell or add to the exspy dictionary in your'
            'Python environment. The file can be found in the folder: '
            '<your_env>/lib/python3.??/site-packages/exspy/_misc/elements.py'
        )

    elif np.sum(model) == 0:
        raise Exception(
            'Edge found but onset is above the spectrum range.'
        )

    return model


def get_edge_crosssection(
        element,
        shell,
        E0,
        alpha,
        beta,
        eV,
        int_window,
        bkgd_window=None,
        GOS='dirac',
        return_model=True,
):

    model = get_edge_model(
        element,
        shell,
        E0,
        alpha,
        beta,
        eV,
        GOS=GOS,
    )

    csect = integrate_edge(model, eV, int_window, bkgd_window=bkgd_window)

    if return_model:
        return csect, model
    else:
        return csect
