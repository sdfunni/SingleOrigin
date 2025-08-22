"""Classes for EELS dataset analysis."""

import inspect
import copy

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseButton

import psutil

from joblib import Parallel, delayed

from ncempy.io.dm import fileDM

import hyperspy.api as hs

from SingleOrigin.utils.environ import is_running_in_jupyter
from SingleOrigin.eels.utils import (
    # power_law,
    get_zlp_fwhm,
    get_energy_inds,
    fit_zlp,
    si_remove_eels_background,
    fourier_ratio_deconvolution,
    get_zlp_cutoff,
    get_thickness,
    get_edge_model,
    eels_multifit,
    plot_spectrum,
    plot_eels_fit,
)
from SingleOrigin.utils.read import emdVelox
from SingleOrigin.utils.plot import quickplot, quickcbar
# from SingleOrigin.mathfn import gaussian_1d

if is_running_in_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# %%
class dset():
    """
    Class for organizing image and EELS data from DM datasets.

    Parameters
    ----------
    file : DM file object
        The DM file object created using fileDM() from ncempy.

    dsetIndex : int
        The index of the dataset to read from the file.

    Attributes...
    -------
    data : ndarray
        The data.

    pixelUnit : ndarray of str
        The units of each axis of the data.

    pixelSize : ndarray of scalars
        The size of pixels along each axis in pixelUnits.

    pixelOrigin : ndarray of scalars
        The offset value of the first pixel along each dimension. Usually 0 for
        image dimensions. Offsets are typical for spectrum dimensions: the
        first energy bin is typically not 0 energy.

    axes : dictionary of {str: array}
        Dictionary of the axes labels ('x', 'y', 'eV', etc.) and corresponding
         oordinates of each pixel along that dimension.

    """

    def __init__(self, path, dsetIndex=0):
        if path.parts[-1][-3:] == 'emd':
            data, meta = emdVelox(path)
            keys = list(data.keys())

            if dsetIndex == 'image':
                detectors = ['HAADF', 'DF-S']
                keyind = np.argmax(np.isin(detectors, keys))

                self.array = data[detectors[keyind]]
                if len(self.array.shape) == 3:
                    self.array = np.sum(self.array, axis=0)
                self.dwellTime = float(meta['Scan']['FrameTime'])

            self.pixelUnit = ['nm'] * 2
            self.pixelSize = [float(meta['BinaryResult']['PixelSize']['width']
                                    ) * 1e9] * 2
            self.pixelOrigin = [0, 0]
            self.beamEnergy = float(meta['Optics']['AccelerationVoltage']
                                    ) / 1e3
            self.cameraLength = float(meta['Optics']['CameraLength']) * 1e3

            if dsetIndex[:2] == 'SI':
                ind = int(dsetIndex[-1])
                self.array = data['EELS_SI']['SI'][ind]
                self.aligned = False
                self.pixelUnit += ['eV']
                self.pixelSize += [data['EELS_SI']['BinSize']]
                self.pixelOrigin += [np.min(data['EELS_SI']['eV'][ind])]
                self.dwellTime = data['EELS_SI']['DwellTimes'][ind]
                self.apertureSize = meta['EnergyFilter'][
                    'EntranceApertureDiameter']

            self.axes = {}

            xydim_index = 0

            for dim in range(len(self.array.shape)):
                if self.pixelUnit[dim] != 'eV':
                    label = ['x', 'y'][xydim_index]
                    xydim_index += 1
                elif self.pixelUnit[dim] == 'eV':
                    label = 'eV'
                else:
                    label = 'frame'

                self.axes[label] = self.pixelSize[dim] * np.arange(
                    self.array.shape[dim]) + self.pixelOrigin[dim]

        else:
            file = fileDM(path)
            dsetdict = file.getDataset(dsetIndex)

            self.array = dsetdict['data']
            self.pixelUnit = np.array(dsetdict['pixelUnit'])
            self.pixelSize = np.array(dsetdict['pixelSize']).astype(float)
            self.pixelOrigin = np.array(dsetdict['pixelOrigin']).astype(float)
            self.aligned = False

            if dsetIndex > 1:

                hsdata = hs.load(path)[dsetIndex]
                metaTEM = hsdata.__dict__[
                    '_metadata']['Acquisition_instrument']['TEM']
                metaEELS = metaTEM['Detector']['EELS']

                self.beamEnergy = metaTEM['beam_energy']
                self.cameraLength = metaTEM['camera_length']
                self.apertureSize = metaEELS['aperture_size']
                self.dwellTime = metaEELS['dwell_time']

            if ((len(self.array.shape) == 3 and ~np.isin('eV', self.pixelUnit))
                    or len(self.array.shape) >= 4):
                dtype = 'stack'
            elif len(self.array.shape) == 1:
                dtype = 'spectrum'
            else:
                dtype = 'SI'

            if 'eV' in self.pixelUnit:
                if dtype == 'stack':
                    raise Exception('stacked EELS frames not yet supported')
                elif dtype == 'spectrum':
                    order = [0]
                else:
                    order = [1, 2, 0]
                # For EELS data, put energy axis last:
                self.array = self.array.transpose(tuple(order))
                self.pixelUnit = self.pixelUnit[order]
                self.pixelSize = self.pixelSize[order]
                self.pixelOrigin = self.pixelOrigin[order]

            self.axes = {}

            xydim_index = 0

            for dim in range(len(self.array.shape)):
                if self.pixelUnit[dim] != 'eV' and dtype != 'stack':
                    label = ['x', 'y'][xydim_index]
                    xydim_index += 1
                elif self.pixelUnit[dim] == 'eV':
                    label = 'eV'
                else:
                    label = 'frame'

                self.axes[label] = np.arange(
                    -self.pixelOrigin[dim] * self.pixelSize[dim],
                    (-self.pixelOrigin[dim] * self.pixelSize[dim]
                     + self.array.shape[dim] * self.pixelSize[dim]),
                    self.pixelSize[dim],
                )

    def rot_scan90(self, k=1, axes=(0, 1)):
        """
        Rotate the scan dimensions of the dataset by multiples of 90 degrees
        in a self-consistent way.

        Parameters
        ----------
        k : int
            Multiple of 90 degree rotations to apply. Positive gives counter-
            clockwise roation, negative is clockwise.
            Default: 1.

        axes : 2-tuple of ints
            The scan axes of the dataset. This should be (0, 1) according to
            the convention of this module.
            Default: (0, 1)

        Returns
        -------
        None.

        """

        self.array = np.rot90(self.array, k=k, axes=axes)
        if hasattr(self, 'array_aligned'):
            self.array_aligned = np.rot90(self.array_aligned, k=k, axes=axes)

        if k % 2 == 1:
            swap = {'x': 'y', 'y': 'x', 'eV': 'eV'}
            self.axes = {swap[k]: self.axes[k] for k in self.axes.keys()}

    def show_spectrum(
            self,
            roi=None,
            energy_range=None,
            aligned=True,
            arb_units=True,
            figax=True,
            figsize=(12, 5),
            waterfall_shifts=None,
    ):
        """
        Plot a spectrum or spectra.

        Parameters
        ----------
        spectrum : str or None
            Which spectrum to plot from the dataset. i.e. for dual EELS:
            'SI_hl' or 'SI_ll'.

        roi : ndarray or None
            2d array with dimensions the same as the scan dimensions of the
            dataset (0, 1). Pixels with values > 0 will be integrated to form
            a spectrum or spectra. For multiple spectrum ROIs, each unique
            value (label) in the roi array will be used as a separate mask
            to generate a separate spectrum.
            Default: None.

        energy_range : 2-list like or None
            The minimum and maximum energy losses to plot. If None, plots the
            whole available energy range.
            Default: None.

        aligned : bool
            Whether to plot the ZLP aligned spectrum. If False, plots the
            unaligned one.
            Default: True

        arb_units : bool
            Whether to plot the y-axis of the spectrum with arbitrary units
            (a.u.). If False, plots the actual counts.

        figax : bool
            Wether to return the figure and axes objects for modification by
            the user.
            Default: False

        figsize : tuple or None
            Size of the resulting figure.

        waterfall_shifts : scalar or list of scalars or None
            How much to shift the spectra if multiple masks are passed in the
            roi. If values > 1, will be intrepreted as a shift in absolute
            counts. For values 0 < x <= 1, shift(s) will be fractions of the
            maximum number of counts.

        Returns
        -------
        I_b : scalar or array
            Intensity value(s) corresponding to eV. Same shape as 'eV'.

        """

        color_list = ['red', 'blue', 'green', 'orange', 'purple', 'lightblue',
                      'magenta']

        # Check for aligned dataset
        if self.aligned:
            data_array = self.array_aligned

        else:
            data_array = self.array

        eV = self.axes['eV']

        # Check if data is an SI
        if roi is None:
            roi = np.ones(data_array.shape[:2])

        labels = np.unique(roi)
        labels = labels[labels > 0]

        data_array = np.array([
            np.mean(data_array[roi == lab], axis=0)
            for lab in labels
        ])

        if energy_range is not None:
            start_ind = np.argmin(np.abs(self.axes['eV'] - energy_range[0]))
            stop_ind = np.argmin(np.abs(self.axes['eV'] - energy_range[1]))
            data_array = data_array[..., start_ind:stop_ind]
            eV = eV[start_ind:stop_ind]

        maxCounts = np.max(data_array, axis=-1)
        decades = (np.log10(maxCounts).astype(int) // 3 * 3)

        if data_array.shape[0] > 1:
            if waterfall_shifts is None:
                waterfall_shifts = np.array([
                    i * 0.1 for i in range(data_array.shape[0])
                ])
            elif isinstance(waterfall_shifts, (int, float)):
                waterfall_shifts = np.array([
                    i * waterfall_shifts
                    for i in range(data_array.shape[0])
                ])
            else:
                waterfall_shifts = np.array(waterfall_shifts)

            if waterfall_shifts[0] < 10:
                waterfall_shifts = waterfall_shifts * maxCounts

            data_array += waterfall_shifts[:, None]

        fig, axs = plt.subplots(
            1,
            # width_ratios=width_ratios,
            figsize=figsize,
            layout="compressed"
        )

        axs.set_xlabel('Energy Loss (eV)', weight='bold', size=16)
        axs.tick_params(axis='x', labelsize=16)

        if arb_units:
            max_y = np.max(data_array)
            axs.set_ylabel('Counts (a.u.)', weight='bold', size=16)
            axs.set_ylim(0, max_y * 1.1)
            axs.set_yticks([])

            for i, spec in enumerate(data_array):
                axs.plot(eV, spec, color=color_list[i])

        else:
            data_array /= 10**decades
            max_y = np.max(data_array)
            axs.set_ylabel(f'Counts x 10$^{decades}$', weight='bold')
            axs.set_ylim(0, max_y/10**decades * 1.1)
            axs.set_yticks([])

            for i, spec in enumerate(data_array):
                axs.plot(eV, data_array, color=color_list[i])

        asp = np.diff(axs.get_xlim())[0] / np.diff(axs.get_ylim())[0]
        asp /= np.abs(np.diff(axs.get_xlim())[0] / np.diff(axs.get_ylim())[0])

        if figax:
            return fig, axs

    def get_summed_spectra(self, roi=None):
        """
        Sum spectra of an SI dataset.

        Parameters
        ----------
        roi : 2d array or None
            2d array with dimensions the same as the scan dimensions of the
            dataset (0, 1). Pixels with values > 0 will be integrated to form
            a spectrum or spectra. For multiple spectrum ROIs, each unique
            value (label) in the roi array will be used as a separate mask
            to generate a separate spectrum.
            Default: None.

        Returns
        -------
        eels.dset object
            Copy of self with the dataset summed over the scan dimensions
            (within the roi, if provided) yielding one or more summed spectra.
            The resulting spectrum array will have 3 dimensions of shape
            (1, n_labels, channels) where n_labels is the number of roi labels,
            channels is the number of energy channels.

        """

        if roi is None:
            roi = np.ones(self.array.shape[:2])

        dset_ = copy.deepcopy(self)

        labels = np.unique(roi)
        labels = labels[labels > 0]

        dset_.array = np.array([[
            np.sum(self.array[roi == lab], axis=0)
            for lab in labels
        ]])

        if self.aligned:
            dset_.array_aligned = np.array([[
                np.sum(self.array[roi == lab], axis=0)
                for lab in labels
            ]])

        dset_.axes['y'] = np.array([0], dtype=int)
        dset_.axes['x'] = labels
        dset_.pixelSize[:2] = 1
        dset_.pixelUnit[:2] = 'NA'

        return dset_


class EELS_SIdata():
    """
    Class for loading, viewing & analizing dm3/4 combined EELS SI datasets.

    Enables streamlined analysis of EELS spectrum image datasets, including
    dual EELS.

    """

    def __init__(
            self,
            path,
    ):

        if path.parts[-1][-3:] == 'emd':

            self.survey = None
            self.adf = dset(path, dsetIndex='image')
            self.zlp_spec = None
            spec_list = ['SI_0', 'SI_1', 'SI_2', 'SI_3', 'SI_4']
            for spec in spec_list:
                setattr(self, spec, dset(path, dsetIndex=spec))
                if getattr(self, spec).pixelOrigin[-1] < 0:
                    self.zlp_spec = spec

        else:
            file = fileDM(path)
            numdsets = file.numObjects - 1

            self.survey = dset(path, 0)
            self.adf = dset(path, 1)
            self.scanSize = self.adf.array.shape
            self.pixelSize = self.adf.pixelSize[0]
            self.pixelUnit = self.adf.pixelUnit[0]
            if numdsets == 3:
                self.SI = dset(path, 2)

            if numdsets == 4:
                self.SI_ll = dset(path, 2)
                self.SI_hl = dset(path, 3)
                if self.SI_hl.pixelOrigin[-1] < 0:
                    self.zlp_spec = 'SI_ll'

        self.aligned = False
        self.microscope_params = None
        self.quant_results = {}

    # TODO: use dset.show_spectrum here with some modifications to reduce code.
    def show_spectrum(
            self,
            spectrum=None,
            roi=None,
            energy_range=None,
            aligned=True,
            arb_units=True,
            figax=True,
            figsize=(12, 5),
            width_ratios=[1, 2],
            waterfall_shifts=None,
    ):
        """
        Show a spectrum alongside the ADF signal from the scan.

        Parameters
        ----------
        spectrum : str or None
            Which spectrum to plot from the dataset. i.e. for dual EELS:
            'SI_hl' or 'SI_ll'.

        roi : ndarray or None
            2d array with dimensions the same as the scan dimensions of the
            dataset (0, 1). Pixels with values > 0 will be integrated to form
            a spectrum or spectra. For multiple spectrum ROIs, each unique
            value (label) in the roi array will be used as a separate mask
            to generate a separate spectrum.
            Default: None.

        energy_range : 2-list like or None
            The minimum and maximum energy losses to plot. If None, plots the
            whole available energy range.
            Default: None.

        aligned : bool
            Whether to plot the ZLP aligned spectrum. If False, plots the
            unaligned one.
            Default: True

        arb_units : bool
            Whether to plot the y-axis of the spectrum with arbitrary units
            (a.u.). If False, plots the actual counts.

        figax : bool
            Wether to return the figure and axes objects for modification by
            the user.
            Default: False

        figsize : tuple or None
            Size of the resulting figure.

        waterfall_shifts : scalar or list of scalars or None
            How much to shift the spectra if multiple masks are passed in the
            roi. If values > 1, will be intrepreted as a shift in absolute
            counts. For values 0 < x <= 1, shift(s) will be fractions of the
            maximum number of counts.

        Returns
        -------
        I_b : scalar or array
            Intensity value(s) corresponding to eV. Same shape as 'eV'.

        """

        color_list = ['red', 'blue', 'green', 'orange', 'purple', 'lightblue',
                      'magenta']

        if spectrum is None:
            if hasattr(self, 'SI_hl'):
                si = self.SI_hl
            else:
                si = self.SI

        else:
            si = getattr(self, spectrum)

        if self.aligned:
            data_array = si.array_aligned

        else:
            data_array = si.array

        eV = si.axes['eV']
        if roi is None:
            counts = np.array([np.sum(data_array, axis=(0, 1))])
        else:
            counts = []
            labels = np.unique(roi)
            labels = labels[labels > 0]
            for lab in labels:
                counts += [np.sum(data_array[roi == lab], axis=0)]

            counts = np.array(counts)

        if energy_range is not None:
            start_ind = np.argmin(np.abs(si.axes['eV'] - energy_range[0]))
            stop_ind = np.argmin(np.abs(si.axes['eV'] - energy_range[1]))
            counts = counts[..., start_ind:stop_ind]
            eV = eV[start_ind:stop_ind]

        maxCounts = np.max(counts)
        decades = int(np.log10(maxCounts) // 3 * 3)

        if counts.shape[0] > 1:
            if waterfall_shifts is None:
                waterfall_shifts = np.array([
                    i * 0.1 for i in range(counts.shape[0])
                ])
            elif isinstance(waterfall_shifts, (int, float)):
                waterfall_shifts = np.array([
                    i * waterfall_shifts
                    for i in range(counts.shape[0])
                ])
            else:
                # waterfall_shifts = np.zeros(counts.shape[0])
                waterfall_shifts = np.array(waterfall_shifts)

            if waterfall_shifts[0] < 10:
                waterfall_shifts *= maxCounts

            counts += waterfall_shifts[:, None]

        fig, axs = plt.subplots(
            1, 2,
            width_ratios=width_ratios,
            figsize=figsize,
            layout="compressed"
        )

        axs[0].imshow(self.adf.array, cmap='gray', zorder=0)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        if roi is not None:
            labels = np.unique(roi)
            labels = labels[labels > 0]

            rgb = np.zeros((*roi.shape, 4))

            for i, lab in enumerate(labels):
                color = np.array(mpl.colors.to_rgba(color_list[i], alpha=0.2))
                rgb[roi == lab] = color

            axs[0].imshow(
                rgb,
                zorder=1
                # alpha=0.2,
                # cmap='Reds'
            )

        axs[1].set_xlabel('Energy Loss (eV)', weight='bold', size=16)
        axs[1].tick_params(axis='x', labelsize=16)

        if arb_units:
            max_y = np.max(counts)
            axs[1].set_ylabel('Counts (a.u.)', weight='bold', size=16)
            axs[1].set_ylim(0, max_y * 1.1)
            axs[1].set_yticks([])

            for i, spec in enumerate(counts):
                axs[1].plot(eV, spec, color=color_list[i])

        else:
            counts /= 10**decades
            max_y = np.max(counts)
            axs[1].set_ylabel(f'Counts x 10$^{decades}$', weight='bold')
            axs[1].set_ylim(0, max_y/10**decades * 1.1)
            axs[1].set_yticks([])

            for i, spec in enumerate(counts):
                axs[1].plot(eV, counts, color=color_list[i])

        asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
        asp /= np.abs(np.diff(axs[0].get_xlim())
                      [0] / np.diff(axs[0].get_ylim())[0])

        if figax:
            return fig, axs

    def rotate_scan90(self, k=1, axes=(0, 1)):
        """
        Rotate the SI scan in multiples of 90 degrees. Will be applied to all
        datasets in the SI object.

        Parameters
        ----------
        k : int
            90 degrees times this number is the rotation that will be applied.

        axes : 2-tuple of ints
            The dimensions of the scan axes.
            Default: (0, 1).

        Returns
        -------
        None.

        """

        dsetlist = [
            getattr(self, attr)
            for attr in dir(self)
            if 'SI' in attr or 'adf' == attr
        ]

        for d in dsetlist:
            if inspect.ismethod(d):
                print('skipping')
                continue
            d.rot_scan90(k=k, axes=axes)

    def alignZLP(self):
        """
        Align the spectra by the ZLP for each pixel.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        # Get array and energy axes
        if self.zlp_spec is None:
            raise Exception('No ZLP spectrum found.')

        else:
            zl = getattr(self, self.zlp_spec)

        zldata = zl.array
        eVzl = zl.axes['eV']

        # Get mean and check that a ZLP is present
        if not (np.nanmin(eVzl) < 0 and np.nanmax(eVzl) > 0):
            raise Exception('ZLP not detected')

        # Check ZLP width and mask data (threshold may need to be modified...)
        fwhm_initial = np.around(get_zlp_fwhm(zldata, eVzl), decimals=4)

        print('\n', 'FWHM before:', fwhm_initial, 'eV')

        startstop = get_energy_inds([-30, 30], zl.axes['eV'])

        start, stop = startstop

        zl_masked = zldata[..., start:stop]
        eVzl_masked = eVzl[start:stop]

        # Set up for iteration over each spectrum
        scanInds = np.indices(zldata.shape[:-1]).reshape((2, -1)).T

        bounds = [(np.min(eVzl), np.max(eVzl)),
                  (1e-3, 5),
                  ((1, None))]

        print('Measuring shifts...')

        zl_masked = zl_masked.reshape((-1, zl_masked.shape[-1]))

        n_jobs = psutil.cpu_count(logical=True)

        fits = np.array(Parallel(n_jobs=n_jobs)(
            delayed(fit_zlp)(
                spec, eVzl_masked, bounds
            ) for spec in tqdm(zl_masked)
        ))

        shifts = -fits[:, 0]
        shifts = np.array(shifts).reshape(zldata.shape[:-1])

        print('Aligning spectra...')
        self.shifts = shifts
        self.aligned = True

        spec_list = [attr for attr in dir(self) if attr[:2] == 'SI']

        for dset in spec_list:
            print
            spec = getattr(self, dset)
            eV = spec.axes['eV']
            spec_aligned = np.array([
                np.interp(
                    eV,
                    eV + shifts[ind[0], ind[1]],
                    spec.array[ind[0], ind[1]]
                )
                for ind in tqdm(scanInds)]).reshape(spec.array.shape)
            if self.zlp_spec == dset:
                # Check ZLP width and mask data
                fwhm_final = np.around(get_zlp_fwhm(spec_aligned, eV),
                                       decimals=4)

                print('\n', 'FWHM after:', fwhm_final, 'eV', '\n')
            spec.array_aligned = spec_aligned
            spec.aligned = True

    def get_eels_intensity_map(
            self,
            int_window,
            bkgd_window=None,
            SI=None,
            lba=None
    ):
        """
        Make an EELS elemental map.

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

        lba : int or None.
            Local background averaging (LBA) sigma. Used for locally averaging
            spectra for background fitting. Will produce smoother maps but
            inherently reduces the quantitative, and in some cases, the
            qualitative correctness of the analysis. NOT RECOMMENDED. If None,
            no LBA is applied.
            Default: None.

        Returns
        -------
        eels_map : array
            The background-subtracted spectrum image.

        """

        if SI is None and self.dualEELS:
            si = self.SI_hl
        elif not self.dualEELS:
            si = self.SI
        else:
            si = getattr(self, SI)

        eV = si.axes['eV']

        if self.aligned:
            data_array = si.array_aligned
        else:
            data_array = si.array

        if bkgd_window is not None:
            si_sub_bkgd = si_remove_eels_background(
                data_array,
                eV,
                bkgd_window,
                lba=lba,
            )

        else:
            si_sub_bkgd = data_array

        start_ind = np.argmin(np.abs(eV - int_window[0]))
        stop_ind = np.argmin(np.abs(eV - int_window[1]))
        eels_map = np.sum(si_sub_bkgd[..., start_ind:stop_ind], axis=(-1))

        eels_map = np.nan_to_num(eels_map, nan=0, posinf=0, neginf=0)
        eels_map = np.where(eels_map < 0, 0, eels_map)

        return eels_map

    def get_eels_thickness_map(self, zlp_cutoff=None):
        """
        Make an EELS thickness map using the log-ratio method.

        Parameters
        ----------
        zlp_cutoff : scalar or None
            The cutoff energy between the ZLP and low loss spectrum. If None,
            will be found as the first minimum in the spectrum above the ZLP.

        Returns
        -------
        thickness : array
            The EELS thickness map

        """

        if self.dualEELS:
            si = self.SI_ll
        else:
            si = self.SI

        if hasattr(si, 'array_aligned'):
            data = si.array_aligned
        else:
            data = si.array

        eV = si.axes['eV']
        if zlp_cutoff is None:
            zlp_cutoff = get_zlp_cutoff(data, eV)

        thickness = np.array([[
            get_thickness(spec, eV, zlp_cutoff) for spec in row]
            for row in data])

        return thickness

    def fourier_ratio_deconvolve_SI(self, hann_taper=None):
        """
        Deconvolve the low loss spectrum to get a single scattering spectrum
        image using the fourier ratio deonvolution method. Result saved as the
        data array in the object.

        Parameters
        ----------
        hann_taper : scalar or None
            The Hann taper to apply to the ends of the spectrum to prevent
            artifacts from fourier filtering. If None, 5% of the spectrum
            length will be tapered on each end.

        Returns
        -------
        None.

        """
        if self.dualEELS:
            pass
        else:
            raise Exception('Must have dual EELS dataset to use Fourier' +
                            'ratio deconvolution.')

        self.SI_hl.array_aligned = fourier_ratio_deconvolution(
            self.SI_hl.array_aligned,
            self.SI_ll.array_aligned,
            # self.SI_hl.axes['eV'],
            self.SI_ll.axes['eV'],
            hann_taper=hann_taper)

    def remove_background(self, si, window):
        """
        Fit and subtract background for each pixel in the dataset. Overwrites
        the data array in the object.

        Parameters
        ----------
        si : str
            The spectrum image to background subtract.

        window : 2-list
            The start and stop energy of the fitting window.

        Returns
        -------
        None.

        """

        si = getattr(self, si)
        eV = si.axes['eV']

        if self.aligned:
            si.array_aligned = si_remove_eels_background(
                si.array_aligned,
                eV,
                window
            )
        else:
            si.array = si_remove_eels_background(
                si.array,
                eV,
                window
            )

    def set_microscope_params(self, E0, alpha, beta):
        """
        Set microscope parameters for subsequent analysis.

        Parameters
        ----------
        E0 : scalar
            The accelerating voltage in kV.

        alpha : scalar
            The convergence semi-angle in mrad.

        beta : scalar
            The collection semi-angle in mrad.

        Returns
        -------
        None.

        """

        self.microscope_params = {'E0': E0, 'alpha': alpha, 'beta': beta}

    def multifit_SI(
            self,
            si,
            edges,
            energy_shifts=None,
            white_lines=None,
            bkgd_window=None,
            fit_window=None,
            GOS='dirac',
    ):
        """
        Fit one or more EELS edges with a single background fit. Stores results
        in "quant_results" as a dictionary. These results can be plotted using
        self.plot_elemental_map() or self.plot_whiteline_map(). They can also
        be explored using self.eels_quant_explorer().

        Parameters
        ----------
        si : str
            Which spectrum image to fit ('SI', 'SI_hl', or 'SI_ll').

        edges : list of 1D arrays
            Model edge(s) to be fit to 'spectrum'. Multiple edges can be
            simultaneously fit to a single spectrum assuming close or
            overlapping edges and/or the post edge backgrounds of lower energy
            edges model the spectrum well up to subsequent edges.

        energy_shifts : list of scalars or None
            The amount to shift each model edge to better match the
            experimental edge onset. If None, no offset(s) applied.

        white_lines : list of scalars
            The approximate energy loss of white line peaks to be fit with
            gaussians. Prevents model edge intensity from being fit to near
            edge structure not accounted for in the isolated atom model. The
            gaussian fits are also useful for measuring energy shifts of the
            white lines with oxidation state / local environment.

        bkgd_window : 2-list or None
            Start and stop energy loss for power law background fitting. If
            None, no attempt is made to account for the background; it is
            assumed that a background has been pre-subtracted.

        fit_window : 2-list or None
            Start and stop energy loss for fitting the model to the
            experimental spectrum. If None, the model is fit up to the highest
            energy loss in the spectrum. If subsequent edges are not to be
            simultaneously fit, the window should end before any additional
            edges.

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
        None.

        """
        # Setup
        if isinstance(edges, str):
            edges = [edges]
        elif isinstance(edges, (list, np.ndarray)):
            pass
        else:
            raise Exception(
                '"edges" must be a string or list-like of strings.'
            )
        si_ = getattr(self, si)
        eV = si_.axes['eV']
        if self.aligned:
            data_array = si_.array_aligned
        else:
            data_array = si_.array
        data_array = data_array.reshape((-1, data_array.shape[-1]))

        # Ensure white lines are in energy order
        white_lines = np.sort(white_lines)

        # Get edge models
        if self.microscope_params is None:
            raise Exception('Must define microscope parameters first')

        if energy_shifts is None:
            energy_shifts = [0] * len(edges)
        else:
            energy_shifts = list(energy_shifts)

        edges_ = []
        onsets = []
        for i, edge in enumerate(edges):
            elem, shell = edge.split('-')
            edges_ += [get_edge_model(
                elem,
                shell,
                eV=eV,
                shift=energy_shifts[i],
                GOS=GOS,
                **self.microscope_params,
            )]

            onsets += [eV[np.argmax(edges_[-1] > 0)]]
        # Order edges by onset:
        indsort = np.argsort(onsets)
        onsets = np.array(onsets)[indsort].tolist()
        edges_ = [edges_[ind] for ind in indsort]
        edges = [edges[ind] for ind in indsort]
        print(indsort, edges)

        onsets += [np.inf]

        n_jobs = psutil.cpu_count(logical=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(eels_multifit)(
                spec,
                eV,
                edges_,
                white_lines=white_lines,
                bkgd_window=bkgd_window,
                fit_window=fit_window,
                return_parameter_keys=True,
                return_nanmask=True,

            ) for spec in tqdm(data_array)
        )

        # Rehsape results into parameter maps
        pkeys = results[0][1]
        nanmask = np.where(results[0][2], 1, np.nan)

        pmaps = np.array([result[0] for result in results]
                         ).reshape((si_.array.shape[:2]) + (len(pkeys),))
        if bkgd_window is not None:
            bkdg_ind_offset = 2
        else:
            bkdg_ind_offset = 0

        edges_ind_offset = bkdg_ind_offset + len(edges)

        # Build results dictionary
        quant = {}
        quant['SI'] = si
        quant['bkgd_window'] = bkgd_window
        quant['fit_window'] = fit_window
        quant['edge_order'] = edges
        if bkgd_window is None:
            quant['bkgd_fit'] = np.zeros(self.scanSize + (2,))
        else:
            quant['bkgd_fit'] = pmaps[..., :2]

        for i, edge in enumerate(edges):
            quant[edge] = {}
            quant[edge]['map'] = pmaps[..., i + bkdg_ind_offset]
            quant[edge]['model'] = edges_[i] * nanmask
            if white_lines is not None:
                lines = []
                for j, line in enumerate(white_lines):
                    if (line > onsets[i]) and (line < onsets[i+1]):
                        lines += [
                            pmaps[...,
                                  edges_ind_offset + 3*j:
                                      edges_ind_offset + 3*(j + 1)]
                        ]

                quant[edge]['white_line_eV'] = white_lines

                quant[edge]['white_lines'] = np.array(lines
                                                      ).transpose((1, 2, 0, 3))
            else:
                quant[edge]['white_lines'] = None

        self.quant_results[('_').join(edges)] = quant
        return nanmask

    def plot_elemental_map(
            self,
            edge,
            fit=None,
            cmap='inferno',
            figax=True,
            return_elemmap=False,
    ):
        """
        Plot an elemental intensity map acquired using the self.multifit_SI
        method.

        Parameters
        ----------
        edge : str
            Edge for which to map the intensity.

        fit : str or None
            The string identifying the spectral fit of one or more elemenets
            with a single background. If None, the first instance of a fit
            with the specified edge will be used. Only needed if more than
            one fit of the same edge has been performed.
            Default: None.

        cmap : str or 3- or 4-list
            If a string, will be used as a colormap. If a list is passed, the
            color represented will be used as a uniform color gradient for the
            map.

        figax : matplotlib Axes object or bool
            If a Axes, plots in this Axes. If bool, whether to return the
            Figure and Axes objects.

        return_elemmap : bool
            Whether to return the data array containing the elemental map.
            Default: False.

        Returns
        -------
        fig, axs : Matplotlib Figure and Axes object (optional)

        elemmap : ndarray (optional)
            The elemental intensity map.

        """
        if fit is None:
            # Find first fit with the specified edge
            matchingfits = [
                key for key in self.quant_results.keys()
                if np.isin(edge, list(self.quant_results[key].keys())).item()
            ]

            if len(matchingfits) > 1:
                print('Multiple fits found. Check for intended results or '
                      + 'specify desired fit.')
            fit = matchingfits[0]

        elemmap = self.quant_results[fit][edge]['map']

        if isinstance(figax, bool):
            fig, ax = plt.subplots(1)

        else:
            ax = figax

        quickplot(elemmap, cmap=cmap, figax=ax)

        return_objs = []
        if figax is True:
            return_objs += [fig, ax]
        if return_elemmap is True:
            return_objs += [elemmap]

        if len(return_objs) > 0:
            return return_objs

    def plot_whiteline_map(
        self,
        edge,
        fit=None,
        param='energy',
        whichlines='all',
        refenergy=0,
        cmap='inferno',
        plot_cbar=True,
        figax=True,
        return_array=False,
    ):
        """
        Plot a map of a white line parameter acquired using the
        self.multifit_SI method.

        Parameters
        ----------
        edge : str
            Edge for which to map the intensity.

        fit : str or None
            The string identifying the spectral fit of one or more elemenets
            with a single background. If None, the first instance of a fit
            with the specified edge will be used. Only needed if more than
            one fit of the same edge has been performed.
            Default: None.

        param : str
            The white line fitting parameter to plot: 'energy', 'width' or
            'intensity'.

        cmap : str or 3- or 4-list
            If a string, will be used as a colormap. If a list is passed, the
            color represented will be used as a uniform color gradient for the
            map.

        whichlines : str, int or list of ints
            Which white line map(s) to plot, indexed in energy order for the
            specified edge.

        refenergy : scalar or ndarray of scalars
            Reference energy/energies for plotting a white line energy shift
            rather than an absolute energy value.

        figax : matplotlib Axes object or bool
            If a Axes, plots in this Axes. If bool, whether to return the
            Figure and Axes objects.

        return_array : bool
            Whether to return the data array containing the map(s) of the white
            line parameter.
            Default: False.

        Returns
        -------
        fig, axs : Matplotlib Figure and Axes object (optional)

        elemmap : ndarray (optional)
            The the white line parameter map(s).

        """
        if fit is None:
            # Find first fit with the specified edge
            matchingfits = [
                key for key in self.quant_results.keys()
                if np.isin(edge, list(self.quant_results[key].keys())).item()
            ]

            if len(matchingfits) > 1:
                print('Multiple fits found. Check for intended results or '
                      + 'specify desired fit.')
            fit = matchingfits[0]

        param_ind = {'energy': 0, 'width': 1, 'intensity': 2}[param]

        maps = self.quant_results[fit][edge]['white_lines'][..., param_ind]
        maps = maps.transpose((2, 0, 1))
        print(maps.shape)

        if whichlines != 'all':
            if isinstance(whichlines, (int, list)):
                maps = maps[whichlines]
            else:
                raise Exception(
                    '"whichlines" must be "all", an int or list of ints.'
                )

        if len(maps.shape) <= 2:
            maps = maps[..., None]

        if isinstance(figax, bool):
            nplots = maps.shape[0]
            fig, axs = plt.subplots(1, nplots, layout='constrained')
            fig.suptitle(f'{edge} White Lines')

        else:
            axs = figax

        axs = np.array(axs, ndmin=1)

        for i, ax in enumerate(axs):
            mean = np.mean(maps[i])
            std = np.mean([np.std(map_) for map_ in maps])
            vmin, vmax = np.around([mean - std*3, mean + std*3], decimals=1)

            cbar = quickplot(
                maps[i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                figax=ax,
                returnplot=True,
            )
            nominal_eV = self.quant_results[fit][edge]['white_line_eV'][i]
            ax.set_title(f'{nominal_eV} eV')
            if plot_cbar:
                quickcbar(
                    cax=None,
                    cmapping=cbar,
                    label='eV',
                    vmin=None,
                    vmax=None,
                    orientation='horizontal',
                    ticks=np.around([vmin, mean, vmax], decimals=1),
                    tick_params={},
                    label_params={},
                    tick_loc=None,
                )

        if figax is True:
            return fig, ax

    def plot_EELS_SI_explorer(
            self,
            edge,
            si=None,
            vImage=None,
            orientation='horizontal',
            figax=False,
    ):
        """
        Explore a set of measurements made on a 4D STEM dataset through an
        interactive plot.

        Parameters
        ----------
        vImage : 2d array
            The image to display for representing the scan region. Any image
            will work, but best to pass one that is meaningful for the
            measurement you are exploring. (e.g. if looking at a Bragg lattice
            measurement, pass one of the lattice parameter or strain maps.)

        measurementType : str ('ewpc', 'ewic', or 'bragg')
            The type of measurement to display.

        scaling : float or str
            The contrast scaling argument to pass to quickplot(). Floats are
            power scaling; 'log' activates logrithmic scaling.

        cmapImage : str
            The color map to use for the vImage subplot.
            Default: 'inferno'

        cmapPattern : str or None
            The color map to use for the pattern (DP, EWPC, EWIC) subplot.
            If None, uses 'grey' for EWPC, 'bwr' for EWIC measurements and
            'inferno' for Bragg.
            Default: 'inferno'



        Returns
        -------
        None.

        """

        vImage_kwargs_default = {'cmap': 'grey'}
        # vImage_kwargs_default.update(vImage_kwargs)

        # pattern_kwargs_default = {
        #     'cmap': {'ewpc': 'gray', 'ewic': 'RdBu_r', 'bragg': 'inferno',
        #              None: 'inferno'
        #              }[measurementType],
        #     'scaling': {'ewpc': 0.2, 'ewic': 1, 'bragg': 0.2, None: 0.2
        #                 }[measurementType]
        # }
        # pattern_kwargs_default.update(pattern_kwargs)

        if edge is not None:
            matchingfits = [
                key for key in self.quant_results.keys()
                if np.isin(edge, list(self.quant_results[key].keys())).item()
            ]

            if len(matchingfits) > 1:
                print('Multiple fits found. Check for intended results or '
                      + 'specify desired fit key:')
                print(list(self.quant_results.keys()))
            fit = matchingfits[0]

            if si is None:
                si = self.quant_results[fit]['SI']

        si = getattr(self, si)

        if self.aligned:
            data = getattr(si, 'array_aligned')
        else:
            data = getattr(si, 'array')

        eV = si.axes['eV']

        if hasattr(self, 'quant_results'):
            plotfit = True
            edge_list = self.quant_results[fit]['edge_order']
            bkgd_prms = self.quant_results[fit]['bkgd_fit']
            elemmaps = []
            models = []
            whitelines = []
            for edge in edge_list:
                elemmaps += [self.quant_results[fit][edge]['map']]
                models += [self.quant_results[fit][edge]['model']]
                whitelines += [self.quant_results[fit][edge]['white_lines']]

            elemmaps = np.array(elemmaps).transpose((1, 2, 0))
            models = np.array(models)
            whitelines = np.array(whitelines).transpose((1, 2, 0, 3, 4))
            # print(whitelines.shape)

        else:
            plotfit = False

        patt_h, patt_w = data.shape[-2:]

        if vImage is None:
            vImage = self.adf.array

        def spectrum(yx, data):
            return data[*yx]
        if orientation == 'horizontal':
            fig, axs = plt.subplots(
                1, 2, constrained_layout=True, width_ratios=[1, 3],
                height_ratios=[1], figsize=(8, 3),
            )
        elif orientation == 'vertical':
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 2, constrained_layout=True)

        # Plot the real space scan image
        quickplot(
            vImage,
            scaling=None,
            figax=axs[0],
            **vImage_kwargs_default
        )
        axs[0].add_patch(plt.Rectangle(np.array([0, 0]) - 0.5, 1, 1,
                                       ec='red',
                                       fc=(0, 0, 0, 0),
                                       lw=2,
                                       ))

        # Draw the initial plot
        plot_spectrum(spectrum([0, 0], data), eV, figax=axs[1],
                      norm_scaling=None)

        plot_eels_fit(
            eV,
            bkgd_prms[0, 0],
            models=models,
            weights=elemmaps[0, 0],
            whitelines=whitelines[0, 0],
            figax=axs[1],
        )

        def on_click(event):
            if event.inaxes == axs[0]:
                if event.button is MouseButton.LEFT:
                    yx = np.around([event.ydata, event.xdata]).astype(int)
                    axs[1].cla()

                    plot_spectrum(spectrum(yx, data), eV, figax=axs[1],
                                  norm_scaling=None)

                    plot_eels_fit(
                        eV,
                        bkgd_prms[*yx],
                        models=models,
                        weights=elemmaps[*yx],
                        whitelines=whitelines[*yx],
                        figax=axs[1],
                    )

                    axs[0].cla()

                    quickplot(
                        vImage,
                        scaling=None,
                        figax=axs[0],
                        **vImage_kwargs_default
                    )
                    axs[0].add_patch(plt.Rectangle(np.flip(yx) - 0.5, 1, 1,
                                                   ec='red',
                                                   fc=(0, 0, 0, 0),
                                                   lw=2,
                                                   ))
                    fig.canvas.draw()
                    self.yx = yx

        def on_press(event):
            if event.key == 'up':
                self.yx += np.array([-1, 0])
            elif event.key == 'down':
                self.yx += np.array([1, 0])
            elif event.key == 'right':
                self.yx += np.array([0, 1])
            elif event.key == 'left':
                self.yx += np.array([0, -1])
            else:
                print('key not supported')
            yx = self.yx

            axs[1].cla()

            plot_spectrum(spectrum(yx, data), eV, figax=axs[1],
                          norm_scaling=None)

            plot_eels_fit(
                eV,
                bkgd_prms[*yx],
                models=models,
                weights=elemmaps[*yx],
                whitelines=whitelines[*yx],
                figax=axs[1],
            )

            axs[0].cla()

            quickplot(
                vImage,
                scaling=None,
                figax=axs[0],
                **vImage_kwargs_default
            )
            axs[0].add_patch(plt.Rectangle(np.flip(yx) - 0.5, 1, 1,
                                           ec='red',
                                           fc=(0, 0, 0, 0),
                                           lw=2,
                                           ))

            fig.canvas.draw()

        # plt.connect("motion_notify_event", on_move)
        plt.connect('button_press_event', on_click)
        plt.connect('key_press_event', on_press)

        plt.show()
        if figax:
            return fig, axs
    # END
