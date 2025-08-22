"""Module with utilities for analyzing and visualizing EELS data."""

import copy

import numpy as np

from scipy.optimize import minimize, least_squares

from scipy.ndimage import gaussian_filter

from scipy.fft import (rfft, irfft)

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
    eelsdata : SingleOrigin.eels.EELS_SIdata
        The dataset of interest.

    roi : 2d array
        Array valued 1 where spectra should be integrated and zero elsewhere.

    energy_range : 2-list
        The start and stop energy values to display

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

    use_aligned : bool
        Whether to plot the ZLP aligned spectrum. If False, plots the
        unaligned spectrum.
        Default: True.

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
        Whether to show the x-axis and y-axis labels, respectively.
        Default: True.

    fontsize : scalar
        The font size to use for axis labels and tick labels.
        Default: 14.

    Returns
    -------

    """

    if background_window is not None:
        counts = remove_eels_background(
            counts,
            eV,
            background_window
        )

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

    ax.plot(
        eV,
        counts/10**decades,
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


def plot_eels_fit(eV, bkgd_prms, models, weights, whitelines, figax=False):
    """
    Plot an EELS spectrum fit.

    Parameters
    ----------
    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    bkgd_prms : 2-list
        [A, r] parameters of the power law backgroun fit.

    models : list of 1D arrays
        EELS edge models.

    weights : list of scalars
        Weights of each EELS edge.

    whitelines : list of 2D arrays or None
        For each model, either a 2D array or None (if no white lines). Array(s)
        have shape: (n_lines, 3).

    Returns
    -------
    I_b : scalar or array
        Intensity value(s) corresponding to eV. Same shape as 'eV'.

    """

    colors = ['black', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green']
    fits = [power_law(eV, *bkgd_prms)]
    for i, model in enumerate(models):
        m = model * weights[i]
        if whitelines[i] is not None:
            for line in whitelines[i]:
                m += gaussian_1d(eV, *line)
        m = np.where(m < 0.001 * np.nanmax(m), np.nan, m)

        fits += [fits[-1] + m]

    if isinstance(figax, bool):
        fig, ax = plt.subplots()
    else:
        ax = figax

    for i, fit in enumerate(fits):
        ax.plot(
            eV,
            fit,
            # label='background',
            c=colors[i]
        )

    if figax is True:
        return figax


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

    Returns
    -------
    I_b : scalar or array
        Intensity value(s) corresponding to eV. Same shape as 'eV'.

    """

    ndims = len(A.shape)
    I_b = A[..., None] * (eV[*(None,)*ndims, ...]) ** (-r[..., None]) + b

    I_b = np.squeeze(I_b)

    return I_b


def pl_ss(params, spectrum, eV):
    """
    Objective function for fitting an inverse power law to EELS background.
    Calls : power_law(eV, A, r)

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

    chi_sq = np.sum((spectrum - fit) ** 2)

    return chi_sq


def fit_eels_background(spectrum, eV, window, plot=False):
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

    plot : bool
        Whether to plot the spectrum, fit and background-subtracted spectrum.
        Default: True.

    Returns
    -------
    params : array of shape (2,)
        The [A, r] inverse power law parameters of the fit.

    """

    d_eV = eV[1] - eV[0]
    start_ind = np.argmin(np.abs(eV - window[0]))
    stop_ind = np.argmin(np.abs(eV - window[1]))

    # Get initial guess from two window method:
    d = stop_ind - start_ind
    d_half = d // 2
    w1 = [start_ind, start_ind + d_half]
    w2 = [stop_ind - d_half, stop_ind]
    E1 = window[0]
    E2 = window[1]
    I1 = np.sum(spectrum[w1[0]:w1[1]]) * d_eV
    I2 = np.sum(spectrum[w2[0]:w2[1]]) * d_eV

    if I1 > I2:
        r0 = 2 * np.log(I1/I2) / np.log(E2/E1)
    else:
        r0 = 2
    A0 = (1 - r0) * (I1 + I2) / (E2**(1-r0) - E1**(1-r0))

    spectrum_ = spectrum[start_ind:stop_ind]
    eV_ = eV[start_ind:stop_ind]

    p0 = [A0, r0]

    params = minimize(
        pl_ss,
        p0,
        bounds=None,
        args=(spectrum_, eV_),
        method='Powell',
    ).x

    bkgd_fit = power_law(eV, *params)
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
        return fig, ax

    else:
        return params


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


def remove_eels_background(spectrum, eV, window, return_params=False):
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

    Returns
    -------
    spectrum_sub : array
        The background-subtracted spectrum.

    """

    params = fit_eels_background(spectrum, eV, window, plot=False)

    spectrum_sub = spectrum - power_law(eV, *params)

    start_ind = np.argmin(np.abs(eV - window[0]))
    spectrum_sub[..., :start_ind] = 0
    if return_params:
        return spectrum_sub, params
    else:
        return spectrum_sub


def integrate_edge(counts, eV, int_window, bkgd_window=None):
    """
    Integrate an EELS edge intensity over a specified range with optional
    background subtraction.

    Parameters
    ----------
    int_window : 2-list
        The start and stop energy of the signal integration window.

    bkgd_window : 2-list or None
        The start and stop energy of the background fitting window. If
        None, no background is subtracted.

    SI : str or None
        Which spectrum image to use: 'SI' (single EELS), 'SI_ll' (dual EELS
        low loss) or 'SI_hl' (dual EELS high loss). If None, will default
        to 'SI_hl' for dual EELS or 'SI' in the case of single EELS.

    Returns
    -------
    eels_map : array
        The background-subtracted spectrum image.

    """

    if bkgd_window is not None:
        counts_sub = remove_eels_background(
            counts,
            eV,
            bkgd_window
        )

    else:
        counts_sub = counts

    start_ind = np.argmin(np.abs(eV - int_window[0]))
    stop_ind = np.argmin(np.abs(eV - int_window[1]))
    edge_int = np.sum(counts_sub[start_ind:stop_ind])
    return edge_int


def si_remove_eels_background(si, eV, window, lba=None):
    """
    Fit EELS background for each pixel in a spectrum image, given a fitting
    window and return the background-subtracted spectrum image.

    Parameters
    ----------
    spectrum : 1D array
        The counts for each EEL bin.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

    window : 2-list
        The start and stop energy of the fitting window.

    lba : int or None.
        Local background averaging (LBA) sigma. Used for locally averaging
        spectra for background fitting. Will produce smoother maps but
        inherently reduces the quantitative, and in some cases, the
        qualitative correctness of the analysis. NOT RECOMMENDED. If None,
        no LBA is applied.
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
        delayed(fit_eels_background)(
            si_fitting[i, j], eV, window
        ) for i, j in tqdm(xy)
    )

    params = np.array(results).reshape(h, w, -1)

    si_sub = si - power_law(eV, params[..., 0], params[..., 1])

    start_ind = np.argmin(np.abs(eV - window[0]))
    si_sub[..., :start_ind] = 0

    return si_sub


def get_thickness(spectrum, eV, zlp_cutoff=3, extrapolate=True):
    """
    Calculate the thickness of a sample relative to the mean free path of beam
    electrons, using the log-ratio method.

    Parameters
    ----------
    spectrum : 1D array
        The counts for each EEL bin. Must contain the zero loss peak.

    eV : scalar or array of scalars
        Energy losses for each data point / bin.

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
    loss spectrum.

    Parameters
    ----------
    hl : ndarray
        EEL spectrum or spectrum image from which the multiple scattering
        should be removed.

    ll : ndarray
        Los loss EEL spectrum or spectrum image. Must include the ZLP.

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

    zlp = gaussian_1d(lleV, 0, fwhm, 1)[*(None,)*scan_dims, ...]

    z = rfft(zlp, n=2*spec_len,)
    jk = rfft(hl, n=2*spec_len, axis=-1)
    jl = rfft(ll, n=2*spec_len, axis=-1)

    hl_deconv = (irfft(z * jk / jl, axis=-1) * I0)[..., :spec_len]

    return hl_deconv


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


def eels_residuals(p0, spectrum, eV, components):
    """
    Objective function for fitting multiple reference spectra to a target
    spectrum using linear least squares.

    Parameters
    ----------
    params : 2-list of scalars
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

    if 'white_lines' in components.keys():
        for i in range(len(components['white_lines'])):
            wl0 = p0[n_edges + 3*i: n_edges + 3*(i + 1)]
            model += gaussian_1d(eV, *wl0)
    r = spectrum - model
    return r


def eels_multifit(
        spectrum,
        eV,
        edges,
        E0=None,
        alpha=None,
        beta=None,
        GOS='dirac',
        energy_shifts=None,
        white_lines=None,
        bkgd_window=None,
        fit_window=None,
        return_parameter_keys=True,
        return_nanmask=True,
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

    white_lines : list of scalars
        The approximate energy loss of white line peaks to be fit with
        gaussians. Prevents model edge intensity from being fit to near edge
        structure not accounted for in the isolated atom model. The gaussian
        fits are also useful for measuring energy shifts of the white lines
        with oxidation state / local environment.

    bkgd_window : 2-list or None
        Start and stop energy loss for power law background fitting. If None,
        no attempt is made to account for the background; it is assumed that
        a background has been pre-subtracted.

    fit_window : 2-list or None
        Start and stop energy loss for fitting the model to the experimental
        spectrum. If None, the model is fit up to the highest energy loss in
        the spectrum. If subsequent edges are not to be simultaneously fit,
        the window should end before any additional edges.

    Returns
    -------
    params : 1D array
        The fitted parameters in the following order: 2-parameter power law
        background (if fit), single weight for each edge and 3-parameter
        gaussian(s) for white line(s) (if fit).

    pkeys : list of str
        The keys describing each parameter

    """

    edges = list(edges)
    if isinstance(edges[0], str):
        # Calculate the edge models
        edges_ = []
        if energy_shifts is None:
            energy_shifts = [0] * len(edges)
        else:
            energy_shifts = list(energy_shifts)

        for i, edge in enumerate(edges):
            elem, shell = edge.split('-')
            edges_ += [get_edge_model(
                elem,
                shell,
                eV=eV,
                shift=energy_shifts[i],
                GOS=GOS,
                E0=E0,
                alpha=alpha,
                beta=beta,
            )]

        edges = edges_

    components = {'edges': np.array(edges)}

    bounds = [*[(0, np.inf)]*(len(edges))]
    p0 = [1]*len(edges)
    pkeys = [f'edge{i}' for i in range(len(edges))]

    if bkgd_window is not None:
        spectrum_, b0 = remove_eels_background(
            spectrum, eV, window=bkgd_window, return_params=True,
        )

        pkeys = ['r', 'A'] + pkeys

    else:
        spectrum_ = copy.deepcopy(spectrum)

    if fit_window is not None:
        if bkgd_window is not None:
            start = bkgd_window[0]
        else:
            start = fit_window[0]
        stop = fit_window[1]

        spectrum_[((eV < start) | (eV > stop))] = np.nan

    nanmask = np.invert(np.isnan(spectrum_))
    spectrum_ = spectrum_[nanmask]
    eV = eV[nanmask]
    components['edges'] = components['edges'][:, nanmask]

    if white_lines is not None:

        components['white_lines'] = white_lines
        for i, x0 in enumerate(white_lines):
            p0 = np.concatenate(
                (p0, [x0, 1, np.nanmax(spectrum)/5])
            )

            pkeys += [f'x0_{i}', f'sig_{i}', f'A_{i}']

            bounds += [(x0-3, x0+3), (0.5, 20), (0, np.inf)]

    bounds = ([bound[0] for bound in bounds],
              [bound[1] for bound in bounds])

    params = least_squares(
        eels_residuals,
        p0,
        args=(spectrum_, eV, components),
        bounds=bounds,
    ).x

    if bkgd_window is not None:
        params = np.concatenate([b0, params])

    ret = [params]

    if return_parameter_keys:
        ret += [pkeys]
    if return_nanmask:
        ret += [nanmask]

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

    for subshell in subshell_dict[shell]:
        try:
            model_sub = EELSCLEdge(
                element_subshell=element + '_' + subshell,
                GOS=GOS,
            )
            model_sub.set_microscope_parameters(**microscope_parameters)
            model += model_sub.function(eV - shift)
        except ValueError:
            print(f'{subshell} not found, skipping')

    if np.sum(model) == 0:
        raise Exception('No subshells found for requested shell.'
                        'Use a different shell or add to the exspy dictionary'
                        'in your Python environment.')

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
