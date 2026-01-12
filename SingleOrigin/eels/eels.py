"""Classes for EELS dataset analysis."""

import inspect
import copy
from pathlib import Path

import numpy as np

from scipy.ndimage import gaussian_filter

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
    subtract_background_SI,
    fourier_ratio_deconvolution,
    get_zlp_cutoff,
    get_thickness,
    get_edge_model,
    modelfit,
    plot_spectrum,
    plot_eels_fit,
)
from SingleOrigin.utils.read import emdVelox
from SingleOrigin.utils.image import bin_data
from SingleOrigin.utils.plot import quickplot, quickcbar
# from SingleOrigin.mathfn import gaussian_1d

if is_running_in_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# %%
class EELSdset():
    """
    Base EELS data class for a single energy range.

    Loads and contains EELS SI or simultaneous ADF image and associated
    metadata.

    Parameters
    ----------
    path : str
        The path to the data file.

    dsetIndex : int or str
        The index of the dataset to read from the file for DM or, for Velox
        EELS datasets, the detector (e.g. 'HAADF', 'DF-S') or spectrum label
        (e.g. 'SI_0', 'SI_1', ... ).

    Attributes
    ----------
    array : ndarray
        The data array.

    array_aligned : ndarray
        The data array after aligning each spectrum by the ZLP.

    aligned : bool
        True if ZLP alignment has been performed, False otherwise.

    pixelUnit : ndarray of str
        The units of each axis of the data.

    pixelSize : ndarray of scalars
        The size of pixels along each axis in pixelUnits.

    pixelOrigin : ndarray of scalars
        The offset value of the first pixel along each dimension. Usually 0 for
        image dimensions. Offsets are typical for spectrum dimensions: the
        first energy bin is typically not 0 energy.

    axes : dictionary of {str: array}
        Dictionary with the dataset axis names as keys ('x', 'y', 'eV', etc.)
        and arrays of corresponding pixel coordinates along that dimension as
        values.

    """

    def __init__(self, path, dsetIndex=0):

        if isinstance(path, str):
            path = Path(path)

        # For Velox .emd files:
        if path.parts[-1][-3:] == 'emd':
            data, meta = emdVelox(path)
            # keys = list(data.keys())

            detectors = ['HAADF', 'DF-S']
            if np.isin(dsetIndex, detectors):
                # keyind = np.argmax(np.isin(detectors, keys))

                self.array = data[dsetIndex]
                if len(self.array.shape) == 3:
                    self.array = np.sum(self.array, axis=0)
                self.dwellTime = float(meta['Scan']['FrameTime'])

            elif dsetIndex == 'a':
                self.array = data['a']

                if len(self.array.shape) == 3:
                    self.array = np.sum(self.array, axis=0)
                # self.dwellTime = float(meta['Scan']['FrameTime'])

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
                    label = ['y', 'x'][xydim_index]
                    xydim_index += 1
                elif self.pixelUnit[dim] == 'eV':
                    label = 'eV'
                else:
                    label = 'frame'

                self.axes[label] = self.pixelSize[dim] * np.arange(
                    self.array.shape[dim]) + self.pixelOrigin[dim]

        # For Gatan Digitalmicrograph .dm3/.dm4 files
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
                    label = ['y', 'x'][xydim_index]
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

    def rotate90(self, k=1, axes=(0, 1)):
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

    def flatten_dset(self, axis):
        """
        Flatten the dataset to make a line scan.

        Parameters
        ----------
        axis : int
            The scan axis to flatten.

        Returns
        -------
        None.

        """

        self.array = np.nansum(self.array, axis=axis, keepdims=True)
        if hasattr(self, 'array_aligned'):
            self.array_aligned = np.nansum(
                self.array_aligned, axis=axis, keepdims=True
            )

        axesdict = {0: 'y', 1: 'x'}
        for ax in axis:
            self.axes[axesdict[ax]] = np.array(self.axes[axesdict[ax]][0])

    def crop_dset(self, xlim, ylim):
        """
        Crop a dataset.

        Parameters
        ----------
        xlim, ylim : 2-lists or None
            The [start, stop] scan pixels along each axis for the cropping.
            If None, the dimension will not be cropped.

        Returns
        -------
        None.

        """

        if xlim is None:
            xlim = [0, None]
        if ylim is None:
            ylim = [0, None]

        self.array = self.array[ylim[0]: ylim[1], xlim[0]: xlim[1]]
        if hasattr(self, 'array_aligned'):
            self.array_aligned = \
                self.array_aligned[ylim[0]: ylim[1], xlim[0]: xlim[1]]

        axesdict = {'y': ylim, 'x': xlim}
        for ax, lim in axesdict.items():
            self.axes[ax] = self.axes[ax][lim[0]: lim[1]]

    def bin_dset(self, binfactor):
        """
        Bin a dataset along the scan dimensions.

        Parameters
        ----------
        binfactor : int
            Number of pixels in each direction to combine.

        Returns
        -------
        None.

        """
        ndims = len(self.array.shape)
        factor = [binfactor] * 2 + [1] * (ndims - 2)
        self.array = bin_data(self.array, factor=factor)
        if hasattr(self, 'array_aligned'):
            self.array_aligned = bin_data(self.array_aligned,
                                          factor=[binfactor, binfactor, 1])
        newshape = self.array.shape

        for i, ax in enumerate(['y', 'x']):
            end = newshape[i]
            self.axes[ax] = self.axes[ax][::binfactor][:end]

        self.pixelSize *= binfactor

    def show_spectra(
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
            Which SI to plot from the dataset. i.e. for dual EELS:
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


        """

        color_list = ['red', 'blue', 'green', 'orange', 'purple', 'lightblue',
                      'magenta']

        # Check for aligned dataset
        if self.aligned:
            data_array = copy.deepcopy(self.array_aligned)

        else:
            data_array = copy.deepcopy(self.array)

        eV = self.axes['eV']

        if roi is None:
            roi = np.ones(data_array.shape[:-1])

        if energy_range is not None:
            start_ind = np.argmin(np.abs(self.axes['eV'] - energy_range[0]))
            stop_ind = np.argmin(np.abs(self.axes['eV'] - energy_range[1]))
            data_array = data_array[..., start_ind:stop_ind]
            eV = eV[start_ind:stop_ind]

        if len(data_array.shape) > 1:
            labels = np.unique(roi)
            labels = labels[labels > 0]

            data_array = np.array([
                np.mean(data_array[roi == lab], axis=0)
                for lab in labels
            ])

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

        else:
            data_array = data_array[None, ...]
            maxCounts = np.max(data_array)
            decades = (np.log10(maxCounts).astype(int) // 3 * 3)

        if isinstance(figax, bool):
            fig, axs = plt.subplots(
                1,
                # width_ratios=width_ratios,
                figsize=figsize,
                layout="compressed"
            )

        elif isinstance(figax, mpl.axes.Axes):
            axs = figax

        axs.set_xlabel('Energy Loss (eV)', weight='bold', size=16)
        axs.tick_params(axis='x', labelsize=16)

        if arb_units:
            axs.set_ylabel('Counts (a.u.)', weight='bold', size=16)
            axs.set_ylim(0, 1.1)
            axs.set_yticks([])

            for i, spec in enumerate(data_array):
                max_y = np.nanmax(spec)
                axs.plot(eV, spec / max_y, color=color_list[i])

        else:
            data_array /= 10**decades
            # max_y = np.max(data_array)
            # print(max_y/10**decades * 1.1)

            axs.set_ylabel(f'Counts x 10$^{decades}$', weight='bold')
            # axs.set_ylim(0, max_y/10**decades * 1.1)
            # axs.set_yticks([])

            for i, spec in enumerate(data_array):
                axs.plot(eV, spec, color=color_list[i])

        if figax is True:
            return fig, axs

    def get_summed_spectra(self, roi=None):
        """
        Sum spectra of an SI dataset.

        Parameters
        ----------
        roi : 2d array of ints or None
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


class EELSgroup():
    """
    Super-class for loading, organizing, analizing and visualizing multi-EELS
    data.

    Loads dual (or higher) EELS SI datasets as a single object for integrated
    & streamlined processing. Each energy range is stored as a separate
    SingleOrigin.eels.eels.EELSdset object. Currently works with Gatan DM
    collated SI files and Velox .emd files containing EELS data.

    Parameters
    ----------
    path : Path or str
        Path to the file. May be DM combined EELS dataset or Velox
        emd file containing EELS data.

    Attributes
    ----------
    SI_... : dset objects
        SI or SI_ll & SI_hl for DM files (depending on whether single or
        dual EELS was collected). SI_0 - SI_4 (for Velox with 5 strip Zebra
        detector).

    adf : dset object
        The simultaneous ADF scan.

    survey : dset object
        The survey scan (only for DM).

    aligned : bool
        True if the datasets have been aligned by the ZLP using the alignZLP()
        method.

    quant_results : dict
        The results of elemental quantification using the modelfit_SI method.

    shifts : array
        The energy shifts applied to each spectrum to align the ZLPs.

    zlp_spec : str or None.
        Label of the dset that contains the ZLP. Determined automatically on
        loading. None if no ZLP is found.

    """

    def __init__(
            self,
            path,
    ):

        if path.parts[-1][-3:] == 'emd':
            dsetlabels = list(emdVelox(path)[0].keys())
            if 'a' in dsetlabels:
                self.survey = EELSdset(path, dsetIndex='a')
            else:
                self.survey = None

            self.adf = EELSdset(path, dsetIndex='DF-S')
            self.zlp_spec = None
            spec_list = ['SI_0', 'SI_1', 'SI_2', 'SI_3', 'SI_4']
            for spec in spec_list:
                setattr(self, spec, EELSdset(path, dsetIndex=spec))
                if getattr(self, spec).pixelOrigin[-1] < 0:
                    self.zlp_spec = spec

        else:
            file = fileDM(path)
            numdsets = file.numObjects - 1

            self.survey = EELSdset(path, 0)
            self.adf = EELSdset(path, 1)
            self.zlp_spec = None
            if numdsets == 3:
                self.SI = EELSdset(path, 2)

            if numdsets == 4:
                self.SI_ll = EELSdset(path, 2)
                self.SI_hl = EELSdset(path, 3)
                if self.SI_hl.pixelOrigin[-1] < 0:
                    self.zlp_spec = 'SI_ll'

        self.scanSize = self.adf.array.shape
        self.pixelSize = self.adf.pixelSize[0]
        self.pixelUnit = self.adf.pixelUnit[0]
        self.aligned = False
        self.microscope_params = None
        self.quant_results = {attr: {}
                              for attr in dir(self)
                              if attr[:2] == 'SI'}

    def show_spectra(
            self,
            spectrum,
            roi=None,
            energy_range=None,
            aligned=True,
            arb_units=True,
            figax=True,
            figsize=(12, 5),
            width_ratios=[1, 2],
            waterfall_shifts=None,
    ):
        """x
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

        width_ratios : 2-list
            Ratio of the image to spectrum plot widths.
            Default: [1, 2]

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

        si = getattr(self, spectrum)

        fig, axs = plt.subplots(
            1, 2,
            width_ratios=width_ratios,
            figsize=figsize,
            layout="compressed"
        )

        quickplot(self.adf.array, cmap='gray', figax=axs[0])

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
            )

        si.show_spectra(
            roi=roi,
            energy_range=energy_range,
            aligned=aligned,
            arb_units=arb_units,
            figax=axs[1],
            figsize=None,
            waterfall_shifts=waterfall_shifts,
        )

        if figax:
            return fig, axs

    def rotate90(self, k=1, axes=(0, 1)):
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
                continue
            d.rotate90(k=k, axes=axes)

        self.scanSize = self.adf.array.shape

        if hasattr(self, 'shifts;'):
            self.shifts = np.rot90(self.shifts, k=k)

    def flatten_SI(self, axis):
        """
        Rotate the SI scan in multiples of 90 degrees. Will be applied to all
        datasets in the SI object.

        Parameters
        ----------
        axis : int
            The scan axis along which to sum the SI.

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
                continue
            d.flatten_dset(axis)

        self.scanSize = self.adf.array.shape

        if hasattr(self, 'shifts;'):
            self.shifts = np.nanmean(self.shifts, axes=axis)

    def crop_SI(self, xlim, ylim):
        """
        Crop an SI dataset. The crop will be applied to all datasets in the SI
        object excluding the survey image (if present).

        Parameters
        ----------
        xlim, ylim : 2-lists or None
            The [start, stop] scan pixels along each axis for the cropping.
            If None, the dimension will not be cropped.

        Returns
        -------
        None.

        """

        if xlim is None:
            xlim = [0, None]
        if ylim is None:
            ylim = [0, None]

        dsetlist = [
            getattr(self, attr)
            for attr in dir(self)
            if 'SI' in attr or 'adf' == attr
        ]

        for d in dsetlist:
            if inspect.ismethod(d):
                continue
            d.crop_dset(xlim, ylim)

        self.scanSize = self.adf.array.shape

        if hasattr(self, 'shifts'):
            self.shifts = self.shifts[ylim[0]: ylim[1], xlim[0]: xlim[1]]

    def bin_SI(self, binfactor):
        """
        Bin an SI dataset. The bining will be applied to all datasets in the SI
        object excluding the survey image (if present).

        Parameters
        ----------
        binfactor : int
            Number of pixels in each bin (n x n pixels).

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
                continue
            d.bin_dset(binfactor)

        self.scanSize = self.adf.array.shape
        self.pixelSize = self.adf.pixelSize

        if hasattr(self, 'shifts'):
            self.shifts = None

# TODO: Make line profile function
    # def lineprofile(
    #     self,
    #     data,
    #     int_width,
    #     image=None,
    #     scandims=None,
    #     signaldims=None,
    #     start=None,
    #     end=None,
    #     plot=True,
    # ):

    #     """
    #     Take a line profile on an SI dataset. The profile will be applied to
    #     all datasets in the SI object excluding the survey image (if present)

    #     Parameters
    #     ----------
    #     binfactor : int
    #         Number of pixels in each bin (n x n pixels).

    #     Returns
    #     -------
    #     None.

    #     """

    #     dsetlist = [
    #         getattr(self, attr)
    #         for attr in dir(self)
    #         if 'SI' in attr or 'adf' == attr
    #     ]

    #     for d in dsetlist:
    #         if inspect.ismethod(d):
    #             continue
    #         d.bin_dset(binfactor)

    def alignZLP(self, sigma=None):
        """
        Align the spectra by the ZLP.

        Parameters
        ----------
        sigma : scalar or None
            Width of the gaussian filter (in eV) to apply prior to measuring
            ZLP position. Helps prevent errors due to noisy ZLP spectra. If
            None, no smoothing is applied./
            Default: None.

        Returns
        -------
        None.

        """

        # Get array and energy axes
        if self.zlp_spec is None:
            raise Exception('No ZLP spectrum found.')

        else:
            zl = getattr(self, self.zlp_spec)

        zldata = copy.deepcopy(zl.array)
        eVzl = zl.axes['eV']

        if sigma is not None:
            zldata = gaussian_filter(
                zldata,
                sigma=sigma/zl.pixelSize[-1],
                axes=(-1,))

        # Get mean and check that a ZLP is present
        if not (np.nanmin(eVzl) < 0 and np.nanmax(eVzl) > 0):
            raise Exception('ZLP not detected')

        # Check ZLP width and mask data (threshold may need to be modified...)
        fwhm_initial = np.around(get_zlp_fwhm(zl.array, eVzl), decimals=4)

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

        print('Aligning spectra...')

        zl_masked = zl_masked.reshape((-1, zl_masked.shape[-1]))

        n_jobs = psutil.cpu_count(logical=True)

        fits = np.array(Parallel(n_jobs=n_jobs)(
            delayed(fit_zlp)(
                spec, eVzl_masked, bounds
            ) for spec in tqdm(zl_masked)
        ))

        shifts = -fits[:, 0]
        shifts = np.array(shifts).reshape(zldata.shape[:-1])

        self.shifts = shifts
        self.aligned = True

        spec_list = [attr for attr in dir(self) if attr[:2] == 'SI']

        for dset in spec_list:
            spec = getattr(self, dset)
            eV = spec.axes['eV']
            spec_aligned = np.array([
                np.interp(
                    eV,
                    eV + shifts[ind[0], ind[1]],
                    spec.array[ind[0], ind[1]]
                )
                for ind in scanInds]).reshape(spec.array.shape)
            if self.zlp_spec == dset:
                # Check ZLP width and mask data
                fwhm_final = np.around(get_zlp_fwhm(spec_aligned, eV),
                                       decimals=4)

                print('\n', 'FWHM after:', fwhm_final, 'eV', '\n')
            spec.array_aligned = spec_aligned
            spec.aligned = True

    def get_eels_intensity_map(
            self,
            si,
            int_window,
            bkgd_window=None,
            lba=None
    ):
        """
        Integrate over an energy range with optional background subtraction.

        Parameters
        ----------
        si : str or None
            Which spectrum image to use: 'SI' (single EELS), 'SI_ll' (dual EELS
            low loss) or 'SI_hl' (dual EELS high loss). If None, will default
            to 'SI_hl' for dual EELS or 'SI' in the case of single EELS.

        int_window : 2-list
            The start and stop energy of the signal integration window.

        bkgd_window : 2-list or None
            The start and stop energy of the background fitting window. If
            None, no background is subtracted.

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

        si = getattr(self, si)

        eV = si.axes['eV']

        if self.aligned:
            data_array = si.array_aligned
        else:
            data_array = si.array

        if bkgd_window is not None:
            si_sub_bkgd, params = subtract_background_SI(
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

        if self.zlp_spec is not None:
            si = getattr(self, self.zlp_spec)
        else:
            raise Exception('No ZLP found. Cannot calculate thickness.')

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
        if self.zlp_spec is None:
            raise Exception('Must have an SI with ZLP to use Fourier' +
                            'ratio deconvolution.')

        if self.aligned is False:
            raise Exception('Must run alignZLP() method first.')

        spec_list = [attr for attr in dir(self) if attr[:2] == 'SI']

        zlp = getattr(self, self.zlp_spec)

        for dset in spec_list:
            if dset == self.zlp_spec:
                continue
            else:
                si = getattr(self, dset)

            si.array_aligned = fourier_ratio_deconvolution(
                si.array_aligned,
                zlp.array_aligned,
                zlp.axes['eV'],
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
            si.array_aligned, params = subtract_background_SI(
                si.array_aligned,
                eV,
                window
            )
        else:
            si.array, params = subtract_background_SI(
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

    def modelfit_SI(
            self,
            si,
            edges,
            energy_shifts=None,
            whitelines=None,
            bkgd_window=None,
            fit_window=None,
            GOS='dirac',
    ):
        """
        Fit one or more EELS edges with a single background fit. Stores results
        in "quant_results" as a dictionary. These results can be #ted using
        self.plot_elemental_map() or self.plot_whiteline_map(). They can also
        be explored using self.explore().

        Parameters
        ----------
        si : str
            Which spectrum image to fit ('SI', 'SI_hl', or 'SI_ll').

        edges : list of strings
            List of edges to be fit. Must follow the format: 'Elem-Shell'.
            For example: 'C-K' or 'La-M'.

        energy_shifts : list of scalars or None
            The energy shift to apply to each model edge to better match the
            experimental edge onset. If None, no offset(s) applied. If more
            than one edge is being fit, pass 0 in this list for any edges that
            should not be shifted. Order must match the order of 'edges'.
            Default: None

        whitelines : list of scalars
            The approximate energy loss of white line peaks to be fit with
            gaussians. Prevents model edge intensity from being fit to near
            edge structure not accounted for in the isolated atom model. The
            gaussian fits are also useful for measuring energy shifts of the
            white lines with oxidation state / local bonding environment.
            Default: None

        bkgd_window : 2-list or None
            Start and stop energy loss for power law background fitting. If
            None, no attempt is made to account for the background; it is
            assumed that a background has been pre-subtracted.
            Default: None

        fit_window : 2-list or None
            Start and stop energy loss for fitting the model to the
            experimental spectrum. If None, the model is fit up to the highest
            energy loss in the spectrum. If subsequent edges are not to be
            simultaneously fit, the window should end before any additional
            edges.
            Default: None

        GOS : str
            The edge (or generalizec oscillator strength) model type to use:
            'dft' or 'dirac'. This function uses the edge model calculation in
            exspy. Not all edges for all elements are included in the exspy
            element dictionary, but the underlying models are present in both
            databases. As a result, the library  may need to be modified by the
            user for less common edges. It is called 'elements.py' and can be
            found in the exspy library in your environment.
            Default: 'dft'

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

        size = data_array.shape
        data_array = data_array.reshape((-1, data_array.shape[-1]))
        # Ensure white lines are in energy order
        if whitelines is not None:
            whitelines = np.sort(whitelines)

        # Get edge models
        if self.microscope_params is None:
            raise Exception('Must define microscope parameters first')

        if energy_shifts is None:
            energy_shifts = [0] * len(edges)
        else:
            energy_shifts = list(energy_shifts)

        models = []
        onsets = []
        for i, edge in enumerate(edges):
            elem, shell = edge.split('-')
            models += [get_edge_model(
                elem,
                shell,
                eV=eV,
                shift=energy_shifts[i],
                GOS=GOS,
                **self.microscope_params,
            )]

            onsets += [eV[np.argmax(models[-1] > 0)]]

        # Order edges by onset:
        indsort = np.argsort(onsets)
        onsets = np.array(onsets)[indsort].tolist()
        models = [models[ind] for ind in indsort]
        edges = [edges[ind] for ind in indsort]

        onsets = np.concatenate((onsets, [np.inf]))

        if whitelines is not None:
            whitelines = np.sort(whitelines)
            # Match whitelines to appropriate edge
            wledges = [edges[-np.argmax(np.flip(line > onsets - 5))]
                       for line in whitelines]

        n_jobs = psutil.cpu_count(logical=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(modelfit)(
                spec,
                eV,
                models,
                whitelines=whitelines,
                bkgd_window=bkgd_window,
                fit_window=fit_window,
                return_parameter_keys=True,
                return_nanmask=True,
                **self.microscope_params,
                plot=False,
            ) for spec in tqdm(data_array)
        )

        # Rehsape results into parameter maps
        pkeys = results[0][2]
        nanmask = np.where(results[0][3], 1, np.nan)
        pmaps = np.array([result[0] for result in results]
                         ).reshape(size[:2] + (len(pkeys),))

        # vecres = np.array([result[-1] for result in results]
        #                   ).reshape(size[:2] + (-1,))
        # n = np.nansum(nanmask)
        # fitstd = (sumsqrs / n)**0.5

        wl_ind_offset = 2 + len(edges)

        # Check for previous results and clean up

        if 'backgrounds' in self.quant_results[si]:

            keylist = list(self.quant_results[si]['backgrounds'].keys())

            for bkgd in keylist:

                if len(set(edges) & set(bkgd.split('_'))) > 0:
                    # Remove previous background fit
                    del self.quant_results[si]['backgrounds'][bkgd]
                    # Remove all edges associated with the background
                    for edge in bkgd.split('_'):
                        del self.quant_results[si][edge]

        else:
            self.quant_results[si] = {'backgrounds': {}}

        # Build new results
        # Store the background fit information
        if bkgd_window is None:
            bkgd_fit = np.zeros(self.scanSize + (2,))
            bkgd_window = [0, None]
        else:
            bkgd_fit = pmaps[..., :2]

        self.quant_results[si]['backgrounds'][('_').join(edges)] = {
            'bkgd_window': bkgd_window,
            'params': bkgd_fit,
        }

        for i, edge in enumerate(edges):
            self.quant_results[si][edge] = {
                'fit_window': fit_window,
                'model': models[i] * nanmask,
                'weights': pmaps[..., i + 2],

            }

            self.quant_results[si][edge]['whitelines'] = None

        if whitelines is not None:
            # n_wl = len(whitelines)
            for edge in edges:
                if edge not in wledges:
                    continue
                wlinds = np.argwhere(
                    edge == np.array(wledges)).squeeze().reshape((-1,))
                self.quant_results[si][edge]['whitelines'] = pmaps[...,
                                                                   wl_ind_offset + 3*wlinds[0]:
                                                                   wl_ind_offset + 3 *
                                                                       (wlinds[-1] + 1)
                                                                   ].reshape(size[:2] + (len(wlinds), 3))

                self.quant_results[si][edge]['whiteline_eV'] = whitelines

        # return vecres

    def get_elemental_map(
            self,
            edge,
            si=None,
    ):
        """
        Plot an elemental intensity map acquired using the self.modelfit_SI
        method.

        Parameters
        ----------
        edge : str
            Edge for which to map the intensity.

        si : str or None
            The string identifying the SI with the desired elemental map. Only
            needed if a given edge has been fit in more than one SI (uncommon).
            Default: None.

        Returns
        -------

        elemmap : ndarray
            The quantitative elemental intensity map.

        """
        if si is None:
            # Find first fit with the specified edge
            matchingSI = [
                si for si in self.quant_results
                if edge in self.quant_results[si]
            ]

            if len(matchingSI) > 1:
                print('Edge fit found in multiple SI datasets. Showing the' +
                      'first instance. Specify desired SI if necessary.')
            si = matchingSI[0]

        elemmap = self.quant_results[si][edge]['weights']

        return elemmap

    def plot_elemental_map(
            self,
            edge,
            cmap='inferno',
            figax=True,
            return_map=False,
            si=None,
    ):
        """
        Plot an elemental intensity map acquired using the self.modelfit_SI
        method.

        Parameters
        ----------
        edge : str
            Edge for which to map the intensity.

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

        si : str or None
            The string identifying the SI with the desired elemental map. Only
            needed if a given edge has been fit in more than one SI (uncommon).
            Default: None.

        Returns
        -------
        fig, axs : Matplotlib Figure and Axes object (optional)

        elemmap : ndarray (optional)
            The elemental intensity map.

        """
        if si is None:
            # Find first fit with the specified edge
            matchingSI = [
                si for si in self.quant_results
                if edge in self.quant_results[si]
            ]

            if len(matchingSI) > 1:
                print('Edge fit found in multiple SI datasets. Showing the' +
                      'first instance. Specify desired SI if necessary.')
            si = matchingSI[0]

        elemmap = self.quant_results[si][edge]['weights']

        if isinstance(figax, bool):
            fig, ax = plt.subplots(1)

        else:
            ax = figax

        quickplot(elemmap, cmap=cmap, figax=ax)

        return_objs = []
        if figax is True:
            return_objs += [fig, ax]
        if return_map is True:
            return_objs += [elemmap]

        if len(return_objs) > 0:
            return return_objs

    def plot_whiteline_map(
        self,
        edge,
        param='energy',
        whichlines='all',
        refenergy=0,
        weightthresh=0.01,
        cmap='inferno',
        plot_cbar=True,
        figax=True,
        return_array=False,
        si=None,
    ):
        """
        Plot a map of a white line parameter acquired using the
        self.modelfit_SI method.

        Parameters
        ----------
        edge : str
            Edge for which to map the intensity.

        param : str
            The white line fitting parameter to plot: 'energy', 'width' or
            'intensity'.
            Default: 'energy'

        whichlines : str, int or list of ints
            Which white line map(s) to plot, indexed in energy order for the
            specified edge.
            Default: 'all'

        refenergy : scalar or ndarray of scalars
            Reference energy/energies for plotting a white line energy shift
            rather than an absolute energy value.
            Default: 0

        weightthresh : scalar or None
            Relative edge weight below which to cut off mapping of the white
            line parameter. Allows removing of parts of the map where the
            element is not actually present.
            Default: 0.01

        cmap : str or 3- or 4-list
            If a string, will be used as a colormap. If a list is passed, the
            color represented will be used as a uniform color gradient for the
            map (as RBG[A] value).
            Default: 'inferno'

        figax : matplotlib Axes object or bool
            If an Axes, plots in this Axes. If bool, whether to return the
            Figure and Axes objects.

        return_array : bool
            Whether to return the data array containing the map(s) of the white
            line parameter.
            Default: False.

        si : str or None
            The string identifying the SI with the desired elemental map. Only
            needed if a given edge has been fit in more than one SI (uncommon).
            Default: None.

        Returns
        -------
        fig, axs : Matplotlib Figure and Axes object (optional)

        elemmap : ndarray (optional)
            The the white line parameter map(s).

        """
        if si is None:
            # Find first fit with the specified edge
            matchingSI = [
                si for si in self.quant_results
                if edge in self.quant_results[si]
            ]

            if len(matchingSI) > 1:
                print('Edge fit found in multiple SI datasets. Showing the' +
                      'first instance. Specify desired SI if necessary.')
            si = matchingSI[0]

        param_ind = {'energy': 0, 'width': 1, 'intensity': 2}[param]

        elemWeights = self.quant_results[si][edge]['weights']
        weightsmask = np.where(
            elemWeights > np.max(elemWeights) * weightthresh, 1, 0)

        maps = self.quant_results[si][edge]['whitelines'][..., param_ind]
        maps = maps.transpose((2, 0, 1))

        # Mask off parts with minimal edge weight
        maps = np.where(weightsmask > 0, maps, np.nan)

        if whichlines != 'all':
            if isinstance(whichlines, (int, list)):
                maps = maps[whichlines]
            else:
                raise Exception(
                    '"whichlines" must be "all", an int or list of ints.'
                )

        if len(maps.shape) <= 2:
            maps = maps[..., None]

        if isinstance(figax, bool) or (figax is None):
            nplots = maps.shape[0]
            fig, axs = plt.subplots(1, nplots, layout='constrained')
            fig.suptitle(f'{edge} White Lines')

        else:
            axs = figax

        axs = np.array(axs, ndmin=1)

        for i, ax in enumerate(axs):
            mean = np.nanmean(maps[i])
            std = np.nanmean([np.nanstd(map_) for map_ in maps])
            vmin, vmax = np.around([mean - std, mean + std], decimals=2)

            cbar = quickplot(
                maps[i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                figax=ax,
                returnplot=True,
            )

            nominal_eV = self.quant_results[si][edge]['whiteline_eV'][i]
            ax.set_title(f'{nominal_eV} eV')
            if plot_cbar:
                quickcbar(
                    cax=None,
                    cmapping=cbar,
                    label='eV',
                    vmin=None,
                    vmax=None,
                    orientation='horizontal',
                    ticks=np.around([vmin, mean, vmax], decimals=2),
                    tick_params={},
                    label_params={},
                    tick_loc=None,
                )

        return_obj = []
        if figax is True:
            return_obj += [fig, ax]

        if return_array:
            return_obj += [maps]

        if len(return_obj) > 0:
            return return_obj

    def explore(
            self,
            si=None,
            vImage=None,
            orientation='horizontal',
            figax=False,
            xlims=None,
            ylims=None,
            model_colors=None,
    ):
        """
        Explore an EELS dataset and any associated edge fits through an
        interactive plot.

        Parameters
        ----------
        si : str or None
            The spectrum image dataset to explore. Must give either 'edge' or
            'si'.
            Default: None.

        vImage : 2D array or None
            The virtual image to display for navigating the dataset. Must have
            the same dimensions as the EELS scan. If None, displays the
            simultaneous ADF image.
            Default: None.

        orientation : str
            'horizontal' or 'vertical'. Whether the virtual image and spectrum
            plots are arranged horizontally or vertically.
            Default: 'horizontal'

        figax : matplotlib Axes object or bool
            If a Axes, plots in this Axes. If bool, whether to return the
            Figure and Axes objects.
            Default: False.

        xlims : 2-list or None
            Energy range to show. If None, eneire available range is shown.
            Default: None.

        ylims : 2-list or None
            Counts range to show. If None, eneire available range is shown.
            Default: None.

        model_colors : list or None
            Colors to use for plotting the fitted edge models. These will be
            applied in energy loss order, from least to greatest. If None, a
            default list of colors will be used.
            Default: None.

        Returns
        -------
        fig : matplotlib figure object

        axs : list of matplotlib Axes objects

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

        if si is None:
            if len(self.quant_results) == 1:
                si = list(self.quant_results.keys())[0]
            else:
                raise Exception('Must specify which EELSdset to explore.')

        params = [[{} for x in range(self.scanSize[1])]
                  for y in range(self.scanSize[0])]

        if len(self.quant_results[si]) > 0:
            plotfit = True

            bkgd_prms = np.array([
                v['params']
                for v in self.quant_results[si]['backgrounds'].values()
            ])

            bkgd_windows = np.array([
                v['bkgd_window']
                for v in self.quant_results[si]['backgrounds'].values()
            ])
            # for bkgd in bkgd_windows:
            #     if bkgd is None:
            #         bkgd =

            fit_windows = np.array([
                self.quant_results[si][k.split('_')[0]]['fit_window']
                for k in self.quant_results[si]['backgrounds'].keys()
            ])

            total_window = np.hstack(
                (bkgd_windows[:, 0][:, None], fit_windows[:, 1][:, None]),
            )

            models = [
                [self.quant_results[si][edge]['model']
                 for edge in bkgd.split('_')]
                for bkgd in self.quant_results[si]['backgrounds'].keys()
            ]

            edges = [
                [edge for edge in bkgd.split('_')]
                for bkgd in self.quant_results[si]['backgrounds'].keys()
            ]

            weights = [
                [self.quant_results[si][edge]['weights']
                 for edge in bkgd.split('_')]
                for bkgd in self.quant_results[si]['backgrounds'].keys()
            ]

            whitelines = [
                [self.quant_results[si][edge]['whitelines']
                 for edge in bkgd.split('_')]
                for bkgd in self.quant_results[si]['backgrounds'].keys()
            ]

            for i in range(self.scanSize[0]):
                for j in range(self.scanSize[1]):

                    params[i][j] = {
                        'bkgd_prms': bkgd_prms[:, i, j],
                    }

                    params[i][j]['weights'] = [[e[i, j] for e in group]
                                               for group in weights]

                    params[i][j]['whitelines'] = [[
                        e[i, j] if e is not None else e for e in group]
                        for group in whitelines]

        else:
            plotfit = False

        si = getattr(self, si)

        if self.aligned:
            data = getattr(si, 'array_aligned')
        else:
            data = getattr(si, 'array')

        eV = si.axes['eV']

        patt_h, patt_w = data.shape[-2:]

        self.yx = [0, 0]

        if xlims is not None:
            xinds = get_energy_inds(xlims, eV)
            ymax = np.nanmax(data[..., xinds[0]:xinds[1]])

        else:
            ymax = np.nanmax(data)

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

        if plotfit:
            plot_eels_fit(
                eV,
                models=models,
                labels=edges,
                **params[0][0],
                total_window=total_window,
                figax=axs[1],
                colors=model_colors,
            )

        if xlims is not None:
            axs[1].set_xlim(*xlims)
        if ylims is None:
            axs[1].set_ylim(0, ymax)
        else:
            axs[1].set_ylim(*ylims)

        def on_click(event):
            if event.inaxes == axs[0]:
                if event.button is MouseButton.LEFT:
                    yx = np.around([event.ydata, event.xdata]).astype(int)
                    axs[1].cla()

                    plot_spectrum(spectrum(yx, data), eV, figax=axs[1],
                                  norm_scaling=None)

                    if plotfit:
                        plot_eels_fit(
                            eV,
                            models=models,
                            labels=edges,
                            **params[yx[0]][yx[1]],
                            total_window=total_window,
                            figax=axs[1],
                            colors=model_colors,
                        )
                    if xlims is not None:
                        axs[1].set_xlim(*xlims)
                    if ylims is None:
                        axs[1].set_ylim(0, ymax)
                    else:
                        axs[1].set_ylim(*ylims)

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
                yx = self.yx + np.array([-1, 0])
            elif event.key == 'down':
                yx = self.yx + np.array([1, 0])
            elif event.key == 'right':
                yx = self.yx + np.array([0, 1])
            elif event.key == 'left':
                yx = self.yx + np.array([0, -1])
            else:
                print('key not supported')

            if np.all(yx >= 0):

                try:
                    counts = spectrum(yx, data)

                except IndexError:
                    pass

                else:
                    axs[1].cla()
                    plot_spectrum(counts, eV, figax=axs[1],
                                  norm_scaling=None)

                    if plotfit:
                        plot_eels_fit(
                            eV,
                            models=models,
                            labels=edges,
                            **params[yx[0]][yx[1]],
                            total_window=total_window,
                            figax=axs[1],
                            colors=model_colors,
                        )

                    if xlims is not None:
                        axs[1].set_xlim(*xlims)
                    if ylims is None:
                        axs[1].set_ylim(0, ymax)
                    else:
                        axs[1].set_ylim(*ylims)

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

        plt.connect('button_press_event', on_click)
        plt.connect('key_press_event', on_press)

        plt.show()
        if figax:
            return fig, axs
