"""SingleOrigin is a module for atomic column position finding intended for
    high probe_fwhm scanning transmission electron microscope images.
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
    along with this program.  If not, see https://www.gnu.org/licenses

    This module impliments general EWPC method from: Padgett, E. et al. The
    exit-wave power-cepstrum transform for scanning nanobeam electron
    diffraction: robust strain mapping at subnanometer resolution and
    subpicometer precision. Ultramicroscopy 214, (2020).
    """

import psutil
from tqdm import tqdm
import copy
import time

from joblib import Parallel, delayed

import numpy as np
from numpy.linalg import (
    norm,
    inv,
    solve,
)

import pandas as pd

# from scipy.ndimage import gaussian_filter

from sklearn.cluster import KMeans
from skimage.morphology import erosion

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LogNorm, PowerNorm

from SingleOrigin.image import image_norm, get_avg_cell
from SingleOrigin.peakfit import (
    detect_peaks,
    get_feature_size,
    watershed_segment,
    dp_peaks_com,
)
from SingleOrigin.plot import (
    pick_points,
    quickplot,
)
from SingleOrigin.lattice import (
    rotate_2d_basis,
    fit_lattice,
    register_lattice_to_peaks,
    plot_basis,
    measure_lattice_from_peaks,
)
from SingleOrigin.fourier import (
    fft_square,
    cepstrum,
    find_ewpc_peak,
    find_ewic_peak,
    hann_2d,
)
from SingleOrigin.crystalmath import (
    metric_tensor,
    get_astar_2d_matrix,
    rotation_angle_bt_vectors,
    absolute_angle_bt_vectors,
)
from SingleOrigin.cell_transform import UnitCell
from SingleOrigin.ndstem import vAnnularDetImage


class ReciprocalLattice:

    """
    Class for lattice paremeter analysis.

    - Individual image or diffraction pattern (DP) space must be the final two
    dimensions (i.e. -2, -1).
    - Real space scanning must be the second-to-last pair of dimensions
    (i.e. -4, -3) or, in the case of a line scan, the third-to-last dimension.
    - Additional dimensions may define other experimental parameters such as:
    time, temperature, voltage, etc. These must be the first dimension(s).


    Parameters
    ----------
    data : ndarray
        The STEM data to analize.

    scan_pixel_size : scalar, 2-tuple of scalars or None
        The calibrated scan pixel size for the instrument, experimental
        parameters, etc. This (these) values are applied to axes -4, -3
        (typically the first two axes).
        Default: None.

    dp_pixel_size : scalar, 2-tuple of scalars or None
        The calibrated diffraction pattern pixel size for the instrument,
        experimental parameters, etc. This is usually the diffraction image,
        so should be  in 1/Angstroms. This (these) values are applied to
        axes -2, -1.
        Default: None.

    dp_rotation : scalar
        The relative rotation angle from the horizontal detector direction to
        the horizontal scan direction (clockwise is + ). Alternatively, the
        rotation required to align a direction in the diffraction pattern
        with the same direction in the scanned image (counter-clockwise is + ).
        Pass the angle in degrees.
        Default: 0.

    origin : 2-tuple or None
        The initial guess for the origin (for direct experimental data) or the
        defined origin position (for FFT processed data) in the final two
        dimensions.
        Default: None.


    show_mean_image : bool
        Whether to display the mean image after initiation of the data
        object. Verify that this is representative of the underlying data
        as it will be used to hone some analysis parameters prior to applying
        the analysis to the full data.
        Default: True.

    get_ewpc : bool
        Whether to calculate the exit wave power cepstrum from the diffraction
        space data supplied in data.
        Default: True

    upsample_factor : int
        Factor by which to upsample the dataset for EWPC calculaiton.

    Attributes
    ----------
    h, w : ints
        The heiht and width of the image/DP in the final two dimensions.

    mean_image : 2d array
        The mean taken along the first len(data.shape) - 2 axes.

    A bunch of other stuff...

    """

    # TODO : Add new / update attributess in docstring

    # TODO : Incorporate pixel size calibrations for scan dims

    def __init__(
        self,
        data,
        datatype='dp',
        # lattice_dims=(-2, -1),
        # scan_dims=(0, 1),
        t_dim=None,
        scan_pixel_size=None,
        dp_pixel_size=None,
        dp_rotation=0,
        origin=None,
        show_mean_image=True,
        calc_ewpc=True,
        calc_ewic=False,
        upsample_factor=1,
        roi=None,
    ):
        ndims = len(data.shape)
        self.data = data
        self.datatype = datatype

        self.dp_h, self.dp_w = np.array(self.data.shape[-2:])
        self.scan_h, self.scan_w = self.data.shape[-4:-2]

        if roi is None:
            self.roi = np.ones((self.scan_h, self.scan_w))
        else:
            self.roi = np.where(roi > 0, 1, 0)

        if ndims == 2:
            self.data_mean = data
            if datatype == 'dp':
                self.data = data[None, None, :, :]
            if datatype == 'image':
                self.data = data[:, :, None, None]
        elif ndims == 3:
            self.data_mean = np.mean(data, axis=0)
            if datatype == 'dp':
                self.data = data[None, :, :, :]
            if datatype == 'image':
                raise Exception('Data shape not yet supported.')
                # self.data = data[:, :, :, None, None]
        else:
            self.data_mean = np.mean(data[roi == 1], axis=0)

        self.scan_pixel_size = scan_pixel_size
        self.dp_pixel_size = dp_pixel_size
        if dp_pixel_size is not None:
            self.ewpc_pixel_size = 1 / (
                self.dp_pixel_size * data.shape[-1])
        self.dp_rotation = dp_rotation
        self.origin = origin
        self.ref_region = None
        self.upsample_factor = upsample_factor
        self.origin_shift = 0
        self.ewpc = None
        self.ewpc_mean = None
        self.ewic = None
        self.ewic_mean = None

        if calc_ewic or calc_ewpc:
            self.origin = np.array(self.data.shape[-2:]) / 2

            if calc_ewic:
                print('Calculating EWIC. This may take a moment...')
                calc_ewpc = True

                self.ewic = cepstrum(
                    self.data,
                    method='imaginary',
                    upsample_factor=upsample_factor,
                )

                self.ewic_mean = np.mean(
                    self.ewic,
                    axis=(0, 1),
                )

            if calc_ewpc:
                print('Calculating EWPC. This may take a moment...')
                self.ewpc = cepstrum(
                    self.data,
                    method='power',
                    upsample_factor=upsample_factor,
                )

                self.ewpc_mean = np.mean(
                    self.ewpc,
                    axis=(0, 1),
                )
                im_display = self.ewpc_mean

        else:
            im_display = self.data_mean

        if upsample_factor is not None:
            self.origin_shift = self.scan_w * (upsample_factor - 1) / 2
            self.dp_h *= upsample_factor
            self.dp_w *= upsample_factor

        if show_mean_image:
            fig, ax = plt.subplots(1)
            quickplot(im_display, cmap='gray', scaling='log', figax=ax)
            if self.origin is not None:
                ax.scatter(*self.origin, c='red', marker='+')

    def kmeans_segmentation_ewpc(self, n_clusters, window=4):
        """
        Segment scan using kmeans over the ewpc patterns.

        Parameters
        ----------
        n_clusters : int
            Number of regions to segment image.

        window : int
            The wingow size used to block off the central peak. This needs to
            be blocked off because it dominates the patterns and disrupts
            desired segmentation.
            Default: 4

        Returns
        -------
        None.

        """

        ewpc = copy.deepcopy(self.ewpc)

        # Block out central peak:
        ewpc[:, :, 64-window:64+window+1, 64-window:64+window+1] = 0

        # Prepare kmeans analysis
        X_train = ewpc.reshape((-1, 128*128))**0.5
        X_train -= np.min(X_train)
        X_train /= np.max(X_train)

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X_train)
        self.kmeans_labels = kmeans.labels_.reshape((self.scan_h, self.scan_w))
        plt.imshow(self.kmeans_labels)

    def initalize_ewpc_analysis(
        self,
        pick_basis_order=(1, 1),
        use_only_basis_peaks=False,
        measure_basis_order=(1, 1),
        r_min=None,
        r_max=None,
        graphical_picking=True,
        pick_labels=None,
        min_order=0,
        max_order=10,
        thresh=0.05,
        window_size='auto',
        roi=None,
    ):
        """
        Initialize EWPC analysis by registring a lattice to the mean image.

        Parameters
        ----------
        pick_basis_order : 2-tuple of ints
            The order of peak selected for each of the basis vactors.

        use_only_basis_peaks : bool
            Whether to use only basis dirction peaks for measurement. If
            True, measures only two peaks along the basis directions. If False,
            measures all peaks in the registered lattice, subject to 'r_min'
            and 'r_max' arguments.
            Default: False

        measure_basis_order : 2-tuple of ints
            The orders of the two basis peaks to be measured if use_only_basis
            is True. By default, the first order basis vector peaks.
            Default: (1, 1)

        r_max : scalar or None
            If use_only_basis=False, the outer radius (from the origin) within
            which peaks will be measured for lattice determination. If None,
            all detected peaks will be used.
            Default: None

        r_min : scalar or None
            If use_only_basis=False, the inner radius (from the origin) beyond
            which peaks will be measured for lattice determination. If None,
            all detected peaks will be used.
            Default: None

        graphical_picking : bool
            Whether to select the basis peaks graphically.
            Default: True

        pick_labels : 2-tuple or None
            If not picking graphically, the index labels corresponding to the
            two basis peaks chosen as labeled in the plot of a previous call
            of this function and with the same remaining arguments. If picking
            graphically, set to None.
            Defalut: None

        min_order, max_order : ints
            The minimum/maximum order of peaks to locate, use for lattice
            registration and to retain in the lattice DataFrame. For example,
            if first order peaks are affected by direct beam diffuse
            scattering, set min_order=2 to ignore the first order peaks.
            Limiting the maximum order may prevent errors stemming from
            detection of noise peaks where real peaks are too dim or
            non-existant.
            Default: 0, 10

        thresh : scalar
            The threshold cutoff for detecting peaks relative to the highest
            peak, vmax (not considering the central EWPC peak). An individual
            peak must have a maximum > thresh * vmax to be detected.

        window_size : scalar
            Width of field of view in the picking window in pixels.
            Default: 'auto'

        Returns
        -------
        None.

        """

        timeout = None

        if roi is not None:
            if roi.shape != self.data.shape[:2]:
                raise Exception(
                    'roi must be a numpy.array with the same shape as the real'
                    ' space scan dimensions.')
            else:
                print('applying ROI')
                self.ewpc_mean = np.mean(self.ewpc[roi == 1], axis=0)
                self.roi = roi
                self.data_mean = np.mean(self.data[roi == 1], axis=0)

                plt.close('all')

        peaks = detect_peaks(self.ewpc_mean, min_dist=2, thresh=0)

        # Find the maximum of the 2nd highest peak (ignores the central peak)
        self.ewpc_vmax = np.unique(peaks*self.ewpc_mean)[-2]

        # Get feature size after applying vmax as a maximum threshold.
        # This prevents the central peak from dominating the size determination
        # vmax will also be used as the upper limit of the imshow cmap.
        ewpc_thresh = np.where(
            self.ewpc_mean > self.ewpc_vmax,
            self.ewpc_vmax,
            self.ewpc_mean
        )

        min_dist = get_feature_size(ewpc_thresh) * 1

        # Detect peaks and get x,y coordinates
        peaks = detect_peaks(
            self.ewpc_mean,
            min_dist=min_dist,
            thresh=thresh*self.ewpc_vmax,
        )

        xy_peaks = np.fliplr(np.argwhere(peaks))

        # Remove origin peak
        xy_peaks = xy_peaks[norm(xy_peaks - self.origin, axis=1) > 2]

        if window_size == 'auto':
            window_size = int(np.min(norm(xy_peaks - self.origin, axis=1)
                                     ) * np.max(pick_basis_order)) * 5

            if window_size > np.min([self.dp_h, self.dp_w]):
                window_size = np.min([self.dp_h, self.dp_w])

        if graphical_picking or (pick_labels is None):
            basis_picks = pick_points(
                self.ewpc_mean,
                n_picks=2,
                xy_peaks=xy_peaks,
                origin=self.origin,
                graphical_picking=graphical_picking,
                window_size=window_size,
                timeout=timeout,
                vmax=self.ewpc_vmax,
            )

        else:
            basis_picks = xy_peaks[np.array(pick_labels), :]

        if graphical_picking or (pick_labels is not None):

            basis_vects = basis_picks - self.origin

            basis_vects, _, lattice = register_lattice_to_peaks(
                basis_vects,
                self.origin,
                xy_peaks=xy_peaks,
                basis1_order=pick_basis_order[0],
                basis2_order=pick_basis_order[1],
                fix_origin=True,
                min_order=min_order,
                max_order=max_order,
            )

            self.basis_real_px_mean = basis_vects

            if use_only_basis_peaks:
                hk = lattice.loc[:, 'h':'k'].to_numpy()
                lattice = lattice[
                    ((hk == (measure_basis_order[0], 0)).all(axis=1) |
                     (hk == (0, measure_basis_order[1])).all(axis=1))
                ]

            else:
                if r_max is not None:
                    lattice = lattice[norm(
                        lattice.loc[:, 'x_ref':'y_ref'].to_numpy() -
                        self.origin,
                        axis=1)
                        < r_max]
                if r_min is not None:
                    lattice = lattice[norm(
                        lattice.loc[:, 'x_ref':'y_ref'].to_numpy() -
                        self.origin,
                        axis=1)
                        > r_min]

                lattice = lattice[lattice.loc[:, 'y_ref'] <= self.origin[1]]

            fig, ax = plot_basis(
                image=self.ewpc_mean,
                basis_vects=self.basis_real_px_mean,
                origin=self.origin,
                lattice=lattice,
                return_fig=True,
                vmax=self.ewpc_vmax,
            )

            ax.scatter(lattice.loc[:, 'x_ref'], lattice.loc[:, 'y_ref'],
                       marker='o', color='white', s=100, facecolors='none')

            if r_min:
                inner = Circle(self.origin, r_min, color='black', fill=False)
                ax.add_patch(inner)
            if r_max:
                outer = Circle(self.origin, r_max, color='black', fill=False)
                ax.add_patch(outer)

            self.lattice = lattice

    def get_ewpc_basis(
        self,
        window_size=7,
        fit_only_roi=True,
    ):
        """
        Get basis vectors for each scan pixel from the registered EWPC lattice.

        Parameters
        ----------
        window_size : scalar
            Window size of the mask used to find initial guess for peak
            position in each EWPC pattern.
            Default: 7

        Returns
        -------
        None.


        """

        t = [time.time()]
        m = self.ewpc_mean.shape[0] / self.upsample_factor
        p0 = np.concatenate([self.basis_real_px_mean.flatten(), self.origin]
                            ) / self.upsample_factor

        lattice = copy.deepcopy(self.lattice)

        xy_ref = np.around(lattice.loc[:, 'x_ref':'y_ref'].to_numpy()
                           ).astype(int)
        M = lattice.loc[:, 'h':'k'].to_numpy()

        dxy = window_size//2

        masks = np.zeros((xy_ref.shape[0], self.dp_h, self.dp_w))

        for i, mask in enumerate(masks):
            mask[xy_ref[i, 1]-dxy: xy_ref[i, 1] + dxy + 1,
                 xy_ref[i, 0]-dxy: xy_ref[i, 0] + dxy + 1] = 1

        x0y0 = np.array([[
            [np.flip(np.unravel_index(np.argmax(im * mask),
                                      (self.dp_h, self.dp_w)))
             for mask in masks
             ]
            for im in row]
            for row in self.ewpc
        ]) / self.upsample_factor

        def find_mult_ewpc_peaks(
                coords,
                log_dp,
                xy_bound,
        ):

            results = [find_ewpc_peak(coord, log_dp, xy_bound)
                       for coord in coords]

            return results

        hann = hann_2d(self.data.shape[-1])
        minval = np.min(self.data, axis=(-2, -1))
        scan_coords_roi = [[i, j]
                           for i in range(self.scan_h)
                           for j in range(self.scan_w)
                           if self.roi[i, j] == 1]
        scan_coords_all = [[i, j]
                           for i in range(self.scan_h)
                           for j in range(self.scan_w)]

        t += [time.time()]
        print(f'Step 1 (Find peaks): {(t[-1]-t[-2]):.{2}f} sec')

        n_jobs = psutil.cpu_count(logical=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(find_mult_ewpc_peaks)(
                x0y0[i, j],
                np.log(self.data[i, j] - minval[i, j] + 0.1) * hann,
                dxy,
            ) for i, j in tqdm(scan_coords_roi)
        )

        t += [time.time()]
        print(f'Step 2 (Measure peaks): {(t[-1]-t[-2]):.{2}f} sec')

        basis_vects_roi = np.array([
            fit_lattice(
                p0,
                peaks,
                M,
                fix_origin=True,
            )[:4].reshape((2, 2))
            for peaks in results])

        basis_vects = np.zeros((self.scan_h, self.scan_w, 2, 2), dtype=float,
                               ) * np.nan

        for i, basis in enumerate(basis_vects_roi):
            basis_vects[*scan_coords_roi[i]] = basis.reshape((2, 2))

        # rot_mat = np.array([
        #     [np.cos(theta), np.sin(theta)],
        #     [-np.sin(theta), np.cos(theta)]
        # ])

        # # Apply rotation
        # basis_vects = np.array([
        #     [basis @ rot_mat for basis in row]
        #     for row in basis_vects])

        self.basis_real_px = basis_vects

        # Get the mean real space basis vectors
        self.basis_real_px_mean = np.mean(
            self.basis_real_px[self.roi == 1], axis=0
        )

        # Get reciprocal space basis (i.e. for the diffraction patterns)
        self.basis_recip_px = np.array([
            [inv(basis.T) * m
             if not np.any(np.isnan(basis))
             else np.array([np.nan]*4).reshape((2, 2))
             for basis in row]
            for row in self.basis_real_px])

        # Get the mean reciprocal space basis
        self.basis_recip_px_mean = np.mean(
            self.basis_recip_px[self.roi == 1], axis=0
        )

        if self.dp_pixel_size is not None:
            self.apply_true_units()

        t += [time.time()]
        print(f'Step 3 (Register lattice): {(t[-1]-t[-2]):.{2}f} sec')

    def apply_true_units(self, dp_pixel_size=None):
        """
        Applies calibrated detector units to the dataset by calculating lattice
        parameter maps.

        Parameters
        ----------
        dp_pixel_size : scalar
            Pixel size calibration of the detector.

        Returns
        -------
        None.

        """

        if dp_pixel_size is not None:
            self.dp_pixel_size = dp_pixel_size
            self.ewpc_pixel_size = 1 / (
                self.dp_pixel_size * self.data.shape[-1] * self.upsample_factor
            )

        else:
            if self.dp_pixel_size is None:
                raise Exception('dp_pixel_size must be passed or previously ' +
                                'determined using calibrate_dp()')

        # Make lattice parameter maps
        self.lattice_maps = {}
        # Find lattice parameter distances
        self.lattice_maps['a1'] = norm(
            np.squeeze(self.basis_real_px[:, :, 0, :]),
            axis=-1
        ) * self.ewpc_pixel_size

        self.lattice_maps['a2'] = norm(
            np.squeeze(self.basis_real_px[:, :, 1, :]),
            axis=-1
        ) * self.ewpc_pixel_size

        # Find lattice parameter angle
        self.lattice_maps['gamma'] = np.array([
            [absolute_angle_bt_vectors(basis[0], basis[1], np.identity(2))
             if not np.all(basis == 0)
             else 0
             for basis in row]
            for row in self.basis_real_px
        ])

        # Find angle between local & mean basis vectors
        theta1 = np.array([
            [rotation_angle_bt_vectors(
                basis[0],
                self.basis_real_px_mean[0]
            )
                if not np.all(basis == 0)
                else np.nan
                for basis in row]
            for row in self.basis_real_px
        ])
        theta1 = (theta1 + 180) % 360 - 180

        theta2 = np.array([
            [rotation_angle_bt_vectors(
                basis[1],
                self.basis_real_px[1]
            )
                if not np.all(basis == 0)
                else np.nan
                for basis in row]
            for row in self.basis_real_px
        ])
        theta2 = (theta2 + 180) % 360 - 180

        # Find lattice rotation relative to mean basis
        self.lattice_maps['theta'] = (theta1 + theta2) / 2

    def overlay_mean_basis(self, im=None):
        """
        Overlay the mean basis vectors on a virtual dark field image.

        Function is for sanity checking the measured mean basis vectors
        relative to the scan axes. They are not plotted to scale on the image,
        but are to scale with respect to each other.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        if im is None:
            im = vAnnularDetImage(
                self.data,
                inner=5, outer=None,
            )

        # Rotation direction flipped because x, y reference is left handed
        # in Python.
        basis = rotate_2d_basis(self.basis_real_px_mean, -self.dp_rotation)

        fig, ax = plot_basis(
            im,
            basis,
            [self.scan_w/2, self.scan_h/2],
            return_fig=True)

        fig.suptitle('Basis orientation overlaid on scan dimensions image'
                     + '\n Vectors not to scale.')

    def plot_lattice_parameter_maps(
        self,
        lattice_vect_origin=(0.1, 0.9),
        lattice_vect_scale=0.1,
        kmeans_roi=None,
        roi_erode=1,
        figsize=(10, 10),
        return_fig=False
    ):
        """
        Plot calculated strain maps.

        Parameters
        ----------
        lattice_vect_origin : 2-tuple of scalars (0 to 1)
            Fractional axes coordinates (x, y) to locate the lattice vectors
            plot in the a1 plot (upper left axes). (0,1, 0.9) plots the lattice
            vectors with the origin in the bottom left corner of the a1 plot.
            Default: (0.1, 0.9)

        lattice_vect_scale : scalar
            Scale of the lattce basis vectors relative to the plot size. Note:
            the basis vectors are plotted to visualize the crystal orientation
            and are not to scale with respect to the scan pixel size; they are,
            however, to scale with respect to each other.
            lattice_vect_scale=0.1 will scale the vectors so their mean length
            is 10% of the plot size.
            Default: 0.1

        kmeans_roi : int or None
            The k-means cluster label of the region of interest.

        roi_erode : int
            Radius of the erosion footprint, equivalent to erroding this number
            of pixels from the original ROI. Useful for removing pixels near
            the edge of the ROI that are incorrectly classified or may contain
            information from the neighboring region.

        figsize : 2-tuple of scalars
            Size of the resulting figure.
            Default: (10, 10)

        return_fig: bool
            Whether to return the fig and axes objects so they can be modified.
            Default: False

        Returns
        -------
        fig, axs : figure and axes objects (optional)
            The resulting matplotlib figure and axes objects for possible
            modification by the user.

        """

        if kmeans_roi is not None:
            mask = np.where(self.kmeans_labels == kmeans_roi, True, False)
            mask = erosion(mask, footprint=np.ones((roi_erode*2 + 1,)*2))

        else:
            mask = np.ones((self.scan_h, self.scan_w)).astype(bool)

        labels = [r'$a_1$', r'$a_2$',
                  r'$\gamma$', r'$\theta$']
        units = [r'$\AA$', r'$\AA$', r'$\circ$', r'$\circ$']

        label_shift = np.min([self.scan_w, self.scan_h]) * 0.1

        fig, axs = plt.subplots(
            2, 2,
            sharex=True,
            sharey=True,
            figsize=figsize,
            tight_layout=True,
            # layout='constrained',
        )
        axs = axs.flatten()
        plots = []

        for i, comp in enumerate(self.lattice_maps.keys()):
            decimals = 2 if i < 2 else 1

            masked = np.where(mask, self.lattice_maps[comp], np.nan)

            vmid = np.around(np.nanmean(masked), decimals)
            vstd = np.around(np.nanstd(masked), decimals)
            vmin = np.around(np.nanmin(masked), decimals)
            vmax = np.around(np.nanmax(masked), decimals)

            vdiff = np.min([3*vstd, vmid-vmin, vmax-vmid])

            vmin = vmid - vdiff
            vmax = vmid + vdiff

            if i == 0:
                size = np.mean(norm(self.basis_real_px_mean, axis=1))

                plot_basis_vects = self.basis_real_px_mean * \
                    lattice_vect_scale * np.min([self.scan_h, self.scan_w]) \
                    / size

                axis_origin = [self.scan_w * lattice_vect_origin[0],
                               self.scan_h * lattice_vect_origin[1]]
                axs[i].arrow(
                    axis_origin[0],
                    axis_origin[1],
                    plot_basis_vects[0, 0],
                    plot_basis_vects[0, 1],
                    fc='black',
                    ec='black',
                    width=0.2,
                    length_includes_head=True,
                )
                axs[i].arrow(
                    axis_origin[0],
                    axis_origin[1],
                    plot_basis_vects[1, 0],
                    plot_basis_vects[1, 1],
                    fc='black',
                    ec='black',
                    width=0.2,
                    length_includes_head=True,
                )
                axs[i].text(axis_origin[0]+plot_basis_vects[0, 0] * 1.3,
                            axis_origin[1]+plot_basis_vects[0, 1] * 1.3,
                            r'$a_1$',
                            size=16,)
                axs[i].text(axis_origin[0]+plot_basis_vects[1, 0] * 1.3,
                            axis_origin[1]+plot_basis_vects[1, 1] * 1.3,
                            r'$a_2$',
                            size=16,)

            plots += [axs[i].imshow(
                np.where(mask, self.lattice_maps[comp], np.nan),
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
            )]

            axs[i].set_xticks([])
            axs[i].set_yticks([])

            cbar = plt.colorbar(
                plots[i],
                orientation='vertical',
                shrink=0.3,
                aspect=10,
                ticks=[
                    vmin,
                    vmax,
                    np.around(np.mean([vmin, vmax]), decimals+1)
                ],
                pad=0.02,

            )

            cbar.set_label(label=units[i], fontsize=20, fontweight='bold',
                           rotation='horizontal')

            axs[i].text(label_shift, label_shift,
                        labels[i], ha='left', va='top',
                        size=24, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, lw=0)
                        )

        if return_fig:
            return fig, axs

    def get_strain(
        self,
        ref=None,
        roi_erode=1,
        rotation=None,
    ):
        """
        Get strain from the EWPC registered lattice points.

        Parameters
        ----------
        ref : 2x2 array, 2-tuple of 2-lists, int or None
            How do determine the zero strain reference.

            If a 2x2 array, the reference basis vectors to be considered as
            zero strain. Reference vectors must be in units of detector pixels
            and rotationally aligned to the diffraction patterns.

            If a 2-tuple of 2-lists, the reference region coordinate limits of
            the real space scan from which to measure strain as:
            ([xleft, xright], [ytop, ybottom]). The average lattice
            parameters in this region are taken as the zero strain reference.

            If k-means segmentation has been run, passing an integer
            selects one of the k-means labels as the reference region.

            If None, The average basis vectors from the entire dataset are used
            as the zero strain reference.

            Default: None

        roi_erode : int
            Number of pixels to erode from the ROI edge. Only applies if
            k-means segmentation has been run and an int is passed as "ref".
            Useful because 

        rotation : scalar
            Reference frame rotation for strain relative to the scan axes, in
            degrees. Positive is counterclockwise ; negative is clockwise.

        Returns
        -------
        None.

        """

        basis = rotate_2d_basis(self.basis_real_px, -self.dp_rotation)

        if ref is None:
            ref = self.basis_real_px_mean
            self.ref_region = None

        elif isinstance(ref, tuple):
            self.ref_region = np.array([[ref[1][0], ref[1][1]],
                                        [ref[0][0], ref[0][1]]])
            sl = np.s_[ref[1][0]: ref[1][1], ref[0][0]: ref[0][1]]

            ref = np.nanmean(basis[sl], axis=(0, 1))

        elif isinstance(ref, int):
            mask = np.where(self.kmeans_labels == ref, True, False)
            mask = erosion(mask, footprint=np.ones((roi_erode*2 + 1,)*2))
            ref = np.mean(
                self.basis_real_px[mask],
                axis=0,
            )

        beta = np.array([
            [solve(ref, local_basis)
             for local_basis in row]
            for row in basis
        ])

        exx = (beta[:, :, 0, 0] - 1) * 100
        eyy = (beta[:, :, 1, 1] - 1) * 100
        exy = -(beta[:, :, 0, 1] + beta[:, :, 1, 0]) / 2 * 100
        theta = np.degrees((beta[:, :, 0, 1] - beta[:, :, 1, 0]) / 2)

        if rotation is None or rotation == 0:
            self.strain_basis = np.identity(2) \
                * np.min([self.scan_h, self.scan_w]) * 0.05

        else:
            strain = np.stack([exx,
                               eyy,
                               exy],
                              axis=-1)

            rotation = np.radians(rotation)
            c = np.cos(rotation)
            s = np.sin(rotation)
            strain_transform = np.array([[c**2, s**2, 2*s*c],
                                         [s**2, c**2, -2*s*c],
                                         [-s*c, s*c, c**2 - s**2]])
            strain = np.array([[strain_transform @ strain_local
                              for strain_local in row]
                               for row in strain])
            [exx, eyy, exy] = strain.transpose((2, 0, 1))

            self.strain_basis = np.array([
                [np.cos(-rotation), np.sin(-rotation)],
                [-np.sin(-rotation), np.cos(-rotation)]
            ])

        self.strain_map = {
            'exx': exx,
            'eyy': eyy,
            'exy': exy,
            'theta': theta
        }

    def plot_strain_maps(
        self,
        normal_strain_lim=None,
        shear_strain_lim=None,
        theta_lim=None,
        kmeans_roi=None,
        roi_erode=1,
        plot_strain_axes=True,
        strain_axes_size=10,
        strain_axes_origin=(0.25, 0.1),
        figsize=(10, 10),
        figax=False
    ):
        """
        Plot calculated strain maps.

        Parameters
        ----------
        normal_strain_lim : 2-tuple of scalars
            The lower and upper limits for displaying x and y normal strain
            maps. If None, will use the zero-centered absolute maximum of all
            values.
            Default: None

        shear_strain_lim : 2-tuple of scalars
            The lower and upper limits for displaying shear strain maps.
            If None, will use the zero-centered absolute maximum of all
            values.
            Default: None

        theta_lim : 2-tuple of scalars
            The lower and upper limits for displaying lattice rotation.
            If None, will use the zero-centered absolute maximum of all
            values.
            Default: None

        kmeans_roi : int
            The k-means ROI number for which to plot the strain. Only applies
            if k-means segmentation has been run. If None, all data is plotted.
            Default: None.

        plot_strain_axes : bool
            Whether to plot vectors showing the axes of the displayed strain
            fields, i.e. the x and y strain directions.

        strain_axes_size : scalar
            Length of the strain axis arrows in pixels.

        strain_axes_origin : 2-tuple
            The origin location of the displayed strain axes vectors
            (if plot_strain_axes is True) relative to the (x, y) plot axes,
            e.g. (0.1, 0.1) is the upper left corner, 10% in from each edge.
            Default: (0.1, 0.1)

        figsize : 2-tuple of scalars
            Size of the resulting figure.
            Default: (10, 10)

        figax: bool
            Whether to return the fig and axes objects so they can be modified.
            Default: False

        Returns
        -------
        fig, axs : figure and axes objects (optional)
            The resulting matplotlib figure and axes objects for possible
            modification by the user.

        """

        if kmeans_roi is not None:
            mask = np.where(self.kmeans_labels == kmeans_roi, True, False)
            mask = erosion(mask, footprint=np.ones((roi_erode*2 + 1,)*2))

        else:
            mask = np.ones((self.scan_h, self.scan_w)).astype(bool)

        limlist = [normal_strain_lim] * 2 + [shear_strain_lim, theta_lim]
        keys = ['exx', 'eyy', 'exy', 'theta']
        if np.all(self.strain_basis == np.array([[1, 0], [0, 1]])):
            labels = [r'$\epsilon _{xx}$', r'$\epsilon _{yy}$',
                      r'$\epsilon _{xy}$', r'$\theta$']
        else:
            labels = [r'$\epsilon _{11}$', r'$\epsilon _{22}$',
                      r'$\epsilon _{12}$', r'$\theta$']

        # Prepare strain limits for each component
        vminmax = {}
        for i, key in enumerate(keys):
            if limlist[i] is None:
                s = self.strain_map[key][~np.isnan(self.strain_map[key])]
                bins, edges = np.histogram(
                    s,
                    bins=100
                )

                thresh = 0.05 * np.max(bins)
                bins_thresh = edges[:-1][bins > thresh]
                max_ = np.around(np.max(np.abs(bins_thresh)), decimals=1)

                vminmax[key] = (-max_, max_)

            else:
                vminmax[key] = tuple(limlist[i])

        fig, axs = plt.subplots(
            2, 2,
            sharex=True,
            sharey=True,
            figsize=figsize,
            layout='constrained'
        )
        axs = axs.flatten()
        plots = []

        strain_basis = self.strain_basis * strain_axes_size

        for i, comp in enumerate(keys):
            if (comp in ['exx', 'eyy']) & plot_strain_axes:
                axis_origin = [self.scan_w * strain_axes_origin[0],
                               self.scan_h * strain_axes_origin[1]]

                strain_arrow = FancyArrowPatch(
                    axis_origin - strain_basis[i] / 2,
                    axis_origin + strain_basis[i] / 2,
                    arrowstyle="<|-|>",
                    mutation_scale=strain_axes_size,
                    fc='black',
                )

                axs[i].add_patch(strain_arrow)

            units = r'$\circ$' if comp == 'theta' else '%'
            plots += [axs[i].imshow(
                np.where(mask, self.strain_map[comp], np.nan),
                cmap='RdBu_r',
                vmin=vminmax[comp][0],
                vmax=vminmax[comp][1])]

            axs[i].set_xticks([])
            axs[i].set_yticks([])

            if not np.isin(None, vminmax[comp]):
                ticks = vminmax[comp] + (np.mean(vminmax[comp]),)
            else:
                ticks = None

            cbar = plt.colorbar(
                plots[i],
                # cax=cbax,
                orientation='horizontal',
                shrink=0.3,
                aspect=10,
                ticks=ticks,
                pad=0.02,

            )

            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(label=units, fontsize=20, fontweight='bold')

            pad_ = np.min([self.scan_w, self.scan_h]) * 0.05
            axs[i].text(pad_, pad_,
                        labels[i], ha='left', va='top', size=24,
                        bbox=dict(facecolor='white', alpha=0.8, lw=0)
                        )

            if self.ref_region is not None:
                dx = self.ref_region[0, 1] - self.ref_region[0, 0]
                dy = self.ref_region[1, 0] - self.ref_region[1, 1]
                corner = [self.ref_region[0, 0], self.ref_region[1, 1] - 1]

                rect = Rectangle(corner, dx, dy, fill=False)
                axs[i].add_patch(rect)
        if figax:
            return fig, axs

    def plot_dp_basis(self, dp_origin=None, basis_factor=[1, 1]):
        """Plot the mean DP with measured basis vectors.

        Parameters
        ----------
        dp_origin : 2-list or None
            The coordinates of the direct beam or desired origin point. If
            None, will use the pixel with maximum intensity as the origin.
            Default: None
        basis_factor : 2-list
            The factors by which to scale the two basis vectors to check
            alignment with higher order spots.
            Default: [1, 1]

        Returns
        -------
        None.

        """

        basis_factor = np.array(basis_factor, ndmin=2).T

        if dp_origin is None:
            dp_origin = np.flip(np.unravel_index(np.argmax(self.data_mean),
                                                 self.data.shape[-2:]))

        fig, ax = plot_basis(
            np.log(self.data_mean),
            self.basis_recip_px_mean * basis_factor,
            dp_origin,
            return_fig=True)
        ax.legend()

    def plot_ewpc_basis(self, basis_factor=[1, 1]):
        """Plot the mean EWPC with measured basis vectors.

        Parameters
        ----------
        basis_factor : 2-list
            The factors by which to scale the two basis vectors to check
            alignment with higher order spots.
            Default: [1, 1]

        Returns
        -------
        None.

        """

        basis_factor = np.array(basis_factor, ndmin=2).T

        fig, ax = plot_basis(
            self.ewpc_mean,
            self.basis_real_px_mean * basis_factor,
            self.origin,
            return_fig=True,
            vmax=self.ewpc_vmax,
            vmin=None)
        ax.legend()

    def calibrate_dp(self, g1, g2, cif_path=None, apply_to_instance=True):
        """Calibrate detector pixel size from EWPC or DP Bragg basis.

        Parameters
        ----------
        g1, g2 : array-like of shape (3,)
            The [h,k,l] indices of the plane spacings described by the basis
            vectors chosen for the EWPC analysis. Best to use two vectors in
            the same family (e.g. [100] & [010]). Otherwise, be careful to
            specify with the same handedness and order as the two measured
            vectors.

        cif_path : str
            Path to the .cif file describing the evaluated structure. If None,
            will prompt to select a .cif file.
            Default: None.

        apply_to_instance : bool
            Whether to apply the calibration to the current instance as the
            detector pixel size. Otherwise simply returns the calibration
            matrix.
            Default: True.

        Returns
        -------
        beta : array of shape (2,2)
            The transformation matrix from detector pixels to inverse
            angstroms. Index [0,0] is the x-coordinate pixel size, [1,1] is the
            y-coordinate pixel size and. [0,1] & [1,0] indices represent
            shearing/rotation and should be neglidible (typically ~ 1e-5).
            Additionally, x and y pixel sizes should be very close (differing
            by ~ 1e-5 Angstroms^-1). Large values in either case indicates a
            measurement error or that substantial distortion of the diffraction
            pattern(s) is present.

        """

        alpha_meas = copy.copy(self.basis_recip_px_mean)

        uc = UnitCell(cif_path)

        lattice_params = [uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma]

        g = metric_tensor(*lattice_params)

        a_star_2d = get_astar_2d_matrix(g1, g2, g)
        hand = np.sign(np.cross(a_star_2d[0], a_star_2d[1]))

        # Make sure basis systems have the same handedness
        if np.cross(alpha_meas[0], alpha_meas[1]) != hand:
            a_star_2d = np.fliplr(a_star_2d)

        theta1 = np.radians(rotation_angle_bt_vectors(
            alpha_meas[0], a_star_2d[0])
        )
        theta2 = np.radians(rotation_angle_bt_vectors(
            alpha_meas[1], a_star_2d[1])
        )
        theta = (theta1 + theta2) / 2

        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        a_star_t = a_star_2d @ rot_mat

        # Get the transform from detector pixels to 1/Angstroms
        # (i.e. the detector calibration)
        beta = solve(alpha_meas, a_star_t)

        if apply_to_instance:
            self.dp_pixel_size = np.mean(np.diag(beta))
            self.ewpc_pixel_size = 1 / (
                self.dp_pixel_size * self.data.shape[-1])

            self.apply_true_units()

        return beta

    def initalize_bragg_lattice(
        self,
        a1_order=1,
        a2_order=1,
        sigma=None,
        buffer=5,
        detection_thresh='auto',
        peak_bkgd_thresh_factor=0.1,
        fit_bkgd=True,
        max_order=5,
        show_fit=True,
        verbose=True,
        logscale=True,
        pwrscale=None,
    ):
        """Find peaks in a reciprocal lattice image (i.e. FFT or diffraction
        pattern) and get the reciprocal lattice parameters.

        Peaks locations are determined by center-of-mass applied to the
        surrounding region as found using the watershed method. A best-fit
        lattice is then registered to the set of located peaks. This procedure
        results in a high precision measurement of the lattice parameters from
        reciprocal images.

        Parameters
        ----------
        a1_order, a2_order : ints
            The order of the first peak along the a1* and a2* directions.
            (e.g.: if the first spot corresponds to an (002) reflection, then
             set equal to 2.)

        sigma : scalar or 'auto'
            The Laplacian of Gaussian sigma value to use for sharpening of the
            peaks for the peak finding algorithm. If None, automatic feature
            size detection will be used.
            Usually a value between 2
            and 10 will work well.

        buffer : scalar
            Distance defining the edge border outside which peaks will be
            ignored.

        detection_thresh : scalar or str
            The threshold (in % max data intensity) of peak maxima to
            consider as a possible reciprocal lattice spot. Used to remove
            low intensity peaks. If 'auto' is passed, determined as the mean
            intensity of the data.
            Default: 'auto'

        peak_bkgd_thresh_factor : scalar
            Thresholding factor applied to remove background pixels from the
            watershed region of each peak. Threshold value for each peak
            determined as:
                peak_bkgd_thresh_factor * (peak_max - edge_max) + edge_max.
            Where edge_max is the maximum value of the edge pixels in the
            watershed region and peak_max is the maximum of the region. This
            This value should be [0 : ~0.8]. The method ensures that the
            threshold is above the local background and prevents an asymmetric
            region being used for center-of-mass determination.
            Default: 0.1.

        max_order : int
            The maximum order of peak to be used for fitting the Bragg lattice.
            Default: 5.

        show_fit : bool
            Whether to show the fitted Bragg lattice.
            Default: True

        verbose : bool
            Whether to print lattice fitting information to the console.
            Default: True

        logscale : bool
            Whether to log scale the displayed pattern intensities for
            diffraction patterns. No effect on measurements.
            Default: True.

        pwrscale : scalar or None
            If a scalar and logscale is False, will apply a power scaling
            (e.g. 0.5) to the displayed pattern intensity for diffraction
            patterns. No effect on measurements. If None, no scaling.
            Default: None.

        Returns
        -------
        ...

        """

        if self.datatype == 'image':
            n_picks = 2
            fix_origin = True
            log_fft = np.log(fft_square(self.data_mean))
            im_meas = im_display = image_norm(log_fft + 1e-6)
            U = im_display.shape[0] // 2
            origin = np.array([U, U])
            imnorm = None

        elif self.datatype == 'dp':
            if self.origin is None:
                fix_origin = False
                n_picks = 3
            else:
                fix_origin = True
                n_picks = 2
            im_meas = self.data_mean
            if logscale:
                imnorm = LogNorm()
            elif pwrscale is not None:
                imnorm = PowerNorm(pwrscale)
            else:
                imnorm = None
        else:
            raise Exception(
                "'datatype' must be 'image' or 'dp'."
            )

        if detection_thresh == 'auto':
            detection_thresh = np.mean(im_meas)

        self.peak_bkgd_thresh_factor = peak_bkgd_thresh_factor
        self.detection_thresh = detection_thresh
        self.buffer = buffer

        h, w = im_meas.shape

        if sigma is None:
            self.sigma = 0

        elif sigma == 'auto':
            peak_map = detect_peaks(
                im_meas,
                min_dist=4,
                thresh=0
            )

            # Find the max of the 2nd highest peak (ignores the central peak)
            vmax = np.unique(peak_map*im_meas)[-2]

            # Get feature size after applying vmax as a maximum threshold.
            # This prevents the central peak from dominating the size
            # determination vmax will also be used as the upper limit of the
            # imshow cmap.

            image_thresh = np.where(
                im_meas > vmax,
                vmax,
                im_meas
            )

            self.sigma = get_feature_size(image_thresh)
            if self.sigma < 2:
                self.sigma = 2

        else:
            self.sigma = sigma

        # Measure peaks with CoM
        peaks = dp_peaks_com(
            im_meas,
            sigma=self.sigma,
            filter_type='gauss',
            peak_bkgd_thresh_factor=peak_bkgd_thresh_factor,
            detection_thresh=detection_thresh,
            buffer=buffer,
            fit_bkgd=fit_bkgd,
        )

        xy = peaks.loc[:, 'x_com':'y_com'].to_numpy(dtype=float)

        # Pick basis peaks
        basis_picks_xy = pick_points(
            im_meas,
            n_picks,
            xy,
            origin=self.origin,
            graphical_picking=True,
            window_size=None,
            timeout=30,
            cmap='gray',
            vmin=None,
            vmax=None,
            quickplot_kwargs={'scaling': 0.2},
        )

        # Match peaks to basis click points or passed a_star/origin data
        vects = np.array([xy - i for i in basis_picks_xy])
        inds = np.argmin(norm(vects, axis=2), axis=1)
        basis_picks_xy = xy[inds, :]

        a1_pick = basis_picks_xy[-2, :]
        a2_pick = basis_picks_xy[-1, :]

        if self.datatype == 'dp':
            if self.origin is None:
                self.origin = basis_picks_xy[0, :]

        # Generate reciprocal lattice
        a1_star = (a1_pick - self.origin) / a1_order
        a2_star = (a2_pick - self.origin) / a2_order

        a_star = np.array([a1_star, a2_star])

        self.a_star, self.origin, self.recip_latt = register_lattice_to_peaks(
            a_star,
            self.origin,
            xy,
            basis1_order=1,
            basis2_order=1,
            fix_origin=fix_origin,
            max_order=max_order,
            min_order=0,
        )

        # Recalculate reference lattice points
        self.a1_star = self.a_star[0]
        self.a2_star = self.a_star[1]

        self.basis_recip_px_mean = self.a_star

        # Remove lattice points outside image bounds
        self.recip_latt = self.recip_latt[(
            (self.recip_latt.x_ref >= 0) &
            (self.recip_latt.x_ref <= self.dp_w-1) &
            (self.recip_latt.y_ref >= 0) &
            (self.recip_latt.y_ref <= self.dp_h-1)
        )]

        theta = absolute_angle_bt_vectors(
            self.a1_star,
            self.a2_star,
            np.identity(2)
        )

        ratio = norm(self.a1_star)/norm(self.a2_star)

        if verbose:
            print(f'Reciproal lattice angle (degrees): {theta:.{3}f}')
            print(f'Reciproal vector ratio (a1*/a2*): {ratio:.{5}f}')

        # Plot refined basis
        if show_fit:
            fig, ax = plot_basis(
                im_meas,
                self.a_star,
                self.origin,
                lattice=self.recip_latt,
                return_fig=True,
                scaling=0.2,
                vmin=None,
                vmax=None,
            )

            plt.title('Reciprocal Lattice Fit')

    def get_bragg_basis(
        self,
        toler,
        max_order=10,
        fix_origin=False,
        fit_only_roi=True,
    ):
        """
        Get basis vectors for each scan pixel from the registered EWPC lattice.

        Parameters
        ----------
        window_size : scalar
            Window size of the mask used to find initial guess for peak
            position in each EWPC pattern.
            Default: 7

        Returns
        -------
        None.


        """

        t = [time.time()]
        m = self.data_mean.shape[0] / self.upsample_factor

        scan_coords_roi = [[i, j]
                           for i in range(self.scan_h)
                           for j in range(self.scan_w)
                           if self.roi[i, j] == 1]

        t += [time.time()]
        print(f'Step 1 (Setup): {(t[-1]-t[-2]):.{2}f} sec')
        print('Detecting peaks in each DP...')

        n_jobs = psutil.cpu_count(logical=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(dp_peaks_com)(
                self.data[i, j],
                sigma=self.sigma,
                peak_bkgd_thresh_factor=self.peak_bkgd_thresh_factor,
                detection_thresh=self.detection_thresh,
                buffer=self.buffer,
                fit_bkgd=False,
            )
            for i, j in tqdm(scan_coords_roi)
        )

        results = [result.loc[:, 'x_com':'y_com'].to_numpy()
                   for result in results]

        t += [time.time()]
        print(f'Step 2 (Find peaks): {(t[-1]-t[-2]):.{2}f} sec')
        print('Measuring reciprocal lattice in each DP...')

        results = Parallel(n_jobs=n_jobs)(
            delayed(measure_lattice_from_peaks)(
                peaks,
                self.a_star,
                self.origin,
                toler=toler,
                max_order=max_order,
                shape=(self.dp_h, self.dp_w),
                fix_origin=fix_origin,
            )
            for peaks in results
        )

        results = np.array(results)

        t += [time.time()]
        print(f'Step 3 (Measure lattice by pixel): {(t[-1]-t[-2]):.{2}f} sec')

        recip_basis = np.zeros((self.scan_h, self.scan_w, 2, 2), dtype=float,
                               ) * np.nan
        origins = np.zeros((self.scan_h, self.scan_w, 2), dtype=float,
                           ) * np.nan

        for i, basis in enumerate(results):
            recip_basis[*scan_coords_roi[i]] = basis[:2]
            origins[*scan_coords_roi[i]] = basis[2]

        self.basis_recip_px = recip_basis

        # Get the mean reciprocal space basis vectors
        self.basis_recip_px_mean = np.mean(
            self.basis_recip_px[self.roi == 1], axis=0
        )

        # Get real space basis from reciprocal basis
        self.basis_real_px = np.array([
            [inv(basis.T / m)
             if not np.any(np.isnan(basis))
             else np.array([np.nan]*4).reshape((2, 2))
             for basis in row]
            for row in self.basis_recip_px])

        # Get the mean reciprocal space basis
        self.basis_real_px_mean = np.mean(
            self.basis_real_px[self.roi == 1], axis=0
        )

        if self.dp_pixel_size is not None:
            self.apply_true_units()

        t += [time.time()]
        print(f'Step 3 (Register lattice): {(t[-1]-t[-2]):.{2}f} sec')

    def initalize_superlattice(
        self,
        n_superlatt=1,
        superlatt_order=(1,),
        min_order=1,
        max_order=1,
        known_qvects=None,
        show_fit=True,
        verbose=True,
        logscale=True,
        pwrscale=None,
    ):
        """Find superlattice peaks relative to previously located Bragg peaks.

        Uses previously detected peaks in diffraction pattern, so make sure
        the desired superlattice peaks were found in the initialize_lattice
        step.

        Parameters
        ----------
        n_superlatt : int
            The number of independent superlattice modulation vectors to
            measure.
            Default: 1

        superlatt_order : tuple of ints of length n_superlatt
            The order of each superlattice reflection that will be picked.
            Default: (1,)

        max_order, min_order : int
            The maximum/minimum order of superlattice reflections to be
            measured around each Bragg peak.
            Default: 2 / 1

        show_fit : bool
            Whether to plot the final fitted superlattice and basis vectors.
            Default: True

        verbose : bool
            Whether to print resulting basis vector angles and length ratios to
            the console.
            Default: True

        Returns
        -------
        ...

        """

        xy = self.all_peaks.loc[:, 'x':'y'].to_numpy()

        if self.datatype == 'image':
            n_picks = n_superlatt
            U = self.h // 2
            im_meas = fft_square(self.data_mean)
            U = im_meas.shape[0] // 2
            origin = np.array([U, U])
            imnorm = None

        elif self.datatype == 'dp':
            n_picks = 2 * n_superlatt
            im_meas = self.data_mean
            if logscale:
                imnorm = LogNorm()
            elif pwrscale is not None:
                imnorm = PowerNorm(pwrscale)
            else:
                imnorm = None

        else:
            raise Exception(
                "'datatype' must be 'image' or 'dp'."
            )

        if known_qvects is None:
            if min_order < 1:
                min_order = 1

            if ((len(superlatt_order) != n_superlatt)
                    & (superlatt_order == (1,))):
                superlatt_order *= n_superlatt

            elif len(superlatt_order) != n_superlatt:
                raise Exception('len(superlatt_order) must equal n_superlatt')

            h, w = im_meas.shape

            # Get vmin for plotting (helps with dead camera pixels)
            # vmin = np.max([np.mean(im_meas) - 1*np.std(im_meas), 0])

            if n_picks > 0:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(im_meas, cmap='plasma', norm=imnorm)
                ax.scatter(xy[:, 0], xy[:, 1], c='black', marker='+')
                ax.scatter(
                    self.recip_latt.loc[:, 'x_ref'],
                    self.recip_latt.loc[:, 'y_ref'],
                    ec='white',
                    fc='none',
                    marker='o',
                    s=100,
                )

                if self.datatype == 'image':
                    ax.scatter(origin[0], origin[1], c='white', marker='+')
                    fig.suptitle(
                        'Pick peaks for the q1, q2, etc. superlattice vectors',
                        fontsize=12,
                        c='black',
                    )

                elif self.datatype == 'dp':
                    fig.suptitle(
                        'Pick a Bragg peak followed by a corresponding ' +
                        'superlattice peak. Repeat for each superlattice.',
                        fontsize=12,
                        c='black',
                    )

                ax.set_xticks([])
                ax.set_yticks([])

                if self.datatype == 'image':
                    ax.set_xlim(np.min(xy[:, 0]) - 100, np.max(xy[:, 0]) + 100)
                    ax.set_ylim(np.max(xy[:, 1]) + 100, np.min(xy[:, 1]) - 100)

                basis_picks_xy = np.array(plt.ginput(n_picks, timeout=60))

                plt.close('all')

            # Match peaks to  click points to get reciprocal superspace basis
            vects = np.array([xy - i for i in basis_picks_xy])
            inds = np.argmin(norm(vects, axis=2), axis=1)
            basis_picks_xy = xy[inds, :]

            if self.datatype == 'dp':

                origin = basis_picks_xy[::2, :]
                suplat = basis_picks_xy[1::2, :]

            a_4_star = (suplat - origin) / np.array(superlatt_order)[:, None]

        else:
            a_4_star = known_qvects @ self.a_star

        super_latt_indices = np.array([
            i for i in range(1, max_order+1) if np.abs(i) >= min_order
        ])

        # Make array of all superlattice reflection vectors to max_order
        q_vects = a_4_star[:, :, None] * super_latt_indices
        q_vects = np.concatenate([
            q_vects[:, :, i]
            for i in range(q_vects.shape[-1])
        ])

        # Record the reflection order for each vector
        q_order = np.identity(n_superlatt)[:, :, None] * super_latt_indices
        q_order = np.concatenate([
            q_order[:, :, i]
            for i in range(q_order.shape[-1])
        ])

        # Get all superlattice reflection positions from all Bragg peaks
        xy_bragg = self.recip_latt[['x_ref', 'y_ref']].to_numpy()
        q_pos = np.concatenate([
            q_vects + bragg_peak
            for bragg_peak in xy_bragg
        ])

        # Match reciprocal superlattice points to peaks; make DataFrame
        if known_qvects is None:
            vects = np.array([xy - xy_ for xy_ in q_pos])
            inds = np.argmin(norm(vects, axis=2), axis=1)

        else:
            inds = np.arange(q_pos.shape[0])
            xy = np.zeros(q_pos.shape)
            xy[:] = np.nan

        h_inds = self.recip_latt.loc[:, 'h'].tolist()
        h_inds = np.concatenate([[h_ind] * q_vects.shape[0]
                                 for h_ind in h_inds])
        k_inds = self.recip_latt.loc[:, 'k'].tolist()
        k_inds = np.concatenate([[k_ind] * q_vects.shape[0]
                                 for k_ind in k_inds])

        q_keys = [f'q{i+1}' for i in range(n_superlatt)]
        q_order = np.concatenate([q_order] * xy_bragg.shape[0]).T

        self.recip_suplatt = pd.DataFrame({
            'h': h_inds,
            'k': k_inds,
            'x_ref': q_pos[:, 0],
            'y_ref': q_pos[:, 1],
            'x_com': [xy[ind, 0] for ind in inds],
            'y_com': [xy[ind, 1] for ind in inds],
        })

        for i, col in enumerate(q_order):
            self.recip_suplatt[q_keys[i]] = col

        if known_qvects is None:
            # Remove peaks that are too far from initial reciprocal lattice
            self.recip_suplatt = self.recip_suplatt[norm(
                self.recip_suplatt.loc[:, 'x_com':'y_com'].to_numpy(
                    dtype=float)
                - self.recip_suplatt.loc[:, 'x_ref':'y_ref'
                                         ].to_numpy(dtype=float),
                axis=1
            ) < 0.1*np.max(norm(a_4_star, axis=1))
            ].reset_index(drop=True)

            # Refine reciprocal basis vectors
            M_star = self.recip_suplatt.loc[:, 'h': 'k'].to_numpy(dtype=float)

            xy_bragg = M_star @ self.a_star + self.origin

            q_vect_xy = self.recip_suplatt.loc[:, 'x_com': 'y_com'
                                               ].to_numpy(dtype=float) \
                - xy_bragg

            q_order = self.recip_suplatt.loc[:, q_keys[0]:q_keys[-1]
                                             ].to_numpy(dtype=float)

            p0 = np.concatenate((a_4_star.flatten(), np.array([0, 0])))

            params = fit_lattice(p0, q_vect_xy, q_order, fix_origin=True)

            # Save data and report key values
            self.a_4_star = params[:n_superlatt*2].reshape((n_superlatt, 2))

            self.recip_suplatt[['x_ref', 'y_ref']] = (
                q_order @ self.a_4_star + xy_bragg
            )

            if n_superlatt > 1:
                theta_super = [absolute_angle_bt_vectors(
                    self.a_4_star[i],
                    self.a_4_star[int((i+1) % n_superlatt)],
                    np.identity(2))
                    for i in range(n_superlatt)
                ]

            self.theta_bragg_to_super = [rotation_angle_bt_vectors(
                self.a1_star, self.a_4_star[i], np.identity(2))
                for i in range(n_superlatt)
            ]

            ratios = np.around(
                [norm(self.a_4_star[i]) /
                 norm(self.a_star, axis=1)
                 for i in range(n_superlatt)],
                decimals=5
            )

            if verbose:
                if n_superlatt > 1:
                    print(
                        'Rotation angles between superlattice vectors ',
                        f' (degrees): {theta_super}.'
                    )

                print(
                    'Superlattice rotation angle from a1_star (degrees): ',
                    f'{self.theta_bragg_to_super}.'
                )
                print(f'a1 : q norm ratios: {ratios}')

        else:
            # Remove superlattice points outside image bounds

            self.a_4_star = a_4_star
            self.recip_suplatt = self.recip_suplatt[(
                (self.recip_suplatt.x_ref >= 0) &
                (self.recip_suplatt.x_ref <= self.dp_w-1) &
                (self.recip_suplatt.y_ref >= 0) &
                (self.recip_suplatt.y_ref <= self.dp_h-1)
            )]

        # Plot refined basis
        if show_fit:
            fig2, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(im_meas, cmap='plasma', norm=imnorm)
            ax.scatter(
                self.recip_suplatt.loc[:, 'x_ref'].to_numpy(dtype=float),
                self.recip_suplatt.loc[:, 'y_ref'].to_numpy(dtype=float),
                marker='+',
                c='red',
                label='Superlattice Fit'
            )
            ax.scatter(
                self.recip_latt.loc[:, 'x_ref'].to_numpy(dtype=float),
                self.recip_latt.loc[:, 'y_ref'].to_numpy(dtype=float),
                marker='+',
                c='black',
                label='Bragg Lattice Fit'
            )
            ax.scatter(self.origin[0], self.origin[1], marker='+', c='white')

            ax.arrow(
                self.origin[0],
                self.origin[1],
                self.a1_star[0],
                self.a1_star[1],
                fc='black',
                ec='black',
                width=1,
                length_includes_head=True,
                # head_width=10,
                # head_length=15
            )
            ax.arrow(
                self.origin[0],
                self.origin[1],
                self.a2_star[0],
                self.a2_star[1],
                fc='black',
                ec='black',
                width=1,
                length_includes_head=True,
                # head_width=10,
                # head_length=15
            )

            for q in self.a_4_star:
                ax.arrow(
                    self.origin[0],
                    self.origin[1],
                    q[0],
                    q[1],
                    fc='red',
                    ec='red',
                    width=1,
                    length_includes_head=True,
                    # head_width=10,
                    # head_length=15
                )

            if self.datatype == 'image':
                ax.set_xlim(np.min(self.recip_suplatt.loc[:, 'x_ref']) - 100,
                            np.max(self.recip_suplatt.loc[:, 'x_ref']) + 100)
                ax.set_ylim(np.max(self.recip_suplatt.loc[:, 'y_ref']) + 100,
                            np.min(self.recip_suplatt.loc[:, 'y_ref']) - 100)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend()
            plt.title('Superlattice Fit')

# TODO: Add functions to analize bragg lattice over series or scan
# TODO: Add functions to analize superlattice lattice over series or scan
    def initalize_ewic_analysis(
        self,
        roi=None,
        graphical_picking=True,
        use_avg_ewic_cell=False,
        n_cells=None,
        cell_averaging_shift=[0, 0],
        pick_labels=None,
        thresh=0,
        window_size='auto',
    ):
        """
        Initialize polarization analysis by choosing positive and negative
        peaks in the imaginary cepstral transform of the dataset.

        Parameters
        ----------
        roi : 2d array
            ...

        graphical_picking : bool
            Whether to select the basis peaks graphically.
            Default: True

        use_avg_ewic_cell : bool
            Whether to preform the analysis on a unit cell averaged from the
            EWIC. If false, total EWIC is used. Using an average cell

        n_cells : array of shape (2,) or None
            The number of unit cells to include along each basis vector
            direction on each sides of the origin. Total unit cells along an
            axis will usually be double this number. If "cell_averaging_shift"
            is given so that the origin is within a cell along a given basis
            direction, 2n + 1 cells will be averaged along that direction.

        cell_averaging_shift : array of shape (2,) or None
            The fractional coordinates to shift the unit cell origin for
            averaging. This is helpful if the polarization peaks in the EWIC
            lay on/near the unit cell edges.
            Default: None.

        pick_labels : 2-tuple or None
            If not picking graphically, the index labels corresponding to the
            two basis peaks chosen as labeled in the plot of a previous call
            of this function and with the same remaining arguments. If picking
            graphically, set to None.
            Defalut: None

        thresh : scalar
            The threshold cutoff for detecting peaks relative to the highest
            peak, vmax (not considering the central EWPC peak). An individual
            peak must have a maximum > thresh * vmax to be detected.

        window_size : scalar
            Width of field of view in the picking window in pixels.
            Default: 'auto'

        Returns
        -------
        None.

        """

        timeout = 30

        self.origin = np.array([self.dp_w // 2, self.dp_h // 2])

        # Find the maximum of the 2nd highest peak (ignores the central peak)
        self.ewic_mean = np.mean(self.ewic[self.roi == 1], axis=0)
        self.ewpc_mean = np.mean(self.ewpc[self.roi == 1], axis=0)

        peaks = detect_peaks(self.ewic_mean, min_dist=2, thresh=0)
        self.ewic_vmax = np.unique(peaks * self.ewic_mean)[-2]

        # Get feature size after applying vmax as a maximum threshold.
        ewic_thresh = np.where(
            np.abs(self.ewic_mean) > self.ewpc_vmax,
            np.sign(self.ewic_mean)*self.ewpc_vmax,
            self.ewic_mean
        )

        min_dist = get_feature_size(ewic_thresh)

        # Calculate average EWIC cells if using
        if use_avg_ewic_cell:

            # Prepare cell coordinates
            self.origin = np.flip(self.ewic.shape[-2:]) / 2

            urange = np.arange(-n_cells[0], n_cells[0])
            vrange = np.arange(-n_cells[1], n_cells[1]+1)

            M = np.array([[u, v] for u in urange for v in vrange]
                         ) - cell_averaging_shift

            # Blank array for probe positions outside the ROI
            blank = np.zeros(np.flip(np.ceil(np.linalg.norm(
                self.basis_mean_real_px, axis=1)
            ).astype(int) * self.upsample_factor + 2)) * np.nan

            scan_coords_roi = [
                [i, j]
                for i in range(self.scan_h)
                for j in range(self.scan_w)
                if self.roi[i, j] == 1
            ]

            scan_coords_all = [
                [i, j]
                for i in range(self.scan_h)
                for j in range(self.scan_w)
            ]

            # Get average EWIC cells for probe positions in the ROI
            print('Finding average EWIC cells')
            n_jobs = psutil.cpu_count(logical=True)
            ewic_cells = Parallel(n_jobs=n_jobs)(
                delayed(get_avg_cell)(
                    ewic,
                    self.origin,
                    self.basis_vects_real_px[
                        self.roi == 1][i] * self.upsample_factor,
                    M,
                    upsample=1,
                ) for i, ewic in tqdm(enumerate(self.ewic[self.roi == 1]))
            )

            ewic_cells = [
                ewic_cells[np.where((np.array(scan_coords_roi) == (i, j)
                                     ).all(axis=1))[0].item()]
                if [i, j] in scan_coords_roi
                else blank
                for i, j in scan_coords_all
            ]

            # Get average EWPC cells for probe positions in the ROI
            print('Finding average EWPC cells')
            ewpc_cells = Parallel(n_jobs=n_jobs)(
                delayed(get_avg_cell)(
                    ewpc,
                    self.origin,
                    self.basis_vects_real_px[
                        self.roi == 1][i] * self.upsample_factor,
                    M,
                    upsample=1,
                ) for i, ewpc in tqdm(enumerate(self.ewpc[self.roi == 1]))
            )

            ewpc_cells = [
                ewpc_cells[np.where((np.array(scan_coords_roi) == (i, j)
                                     ).all(axis=1))[0].item()]
                if [i, j] in scan_coords_roi
                else blank
                for i, j in scan_coords_all
            ]

            # Crop  cells to smallest size before making a numpy array
            minshape = np.min(np.array([[*px.shape]
                                        for px in ewic_cells]), axis=0)

            ewic_cells = np.array(
                [cell[:minshape[0], :minshape[1]] for cell in ewic_cells]
            ).reshape(*self.data.shape[:2], *minshape)

            ewpc_cells = np.array(
                [cell[:minshape[0], :minshape[1]] for cell in ewpc_cells]
            ).reshape(*self.data.shape[:2], *minshape)

            self.ewic_cells = ewic_cells
            self.ewpc_cells = ewpc_cells

            self.ewic_mean = np.mean(ewic_cells[self.roi == 1], axis=0)
            self.ewpc_mean = np.mean(ewpc_cells[self.roi == 1], axis=0)

            self.origin = np.array(cell_averaging_shift) * self.ewpc_pixel_size

        # Detect peaks and get x,y coordinates
        pospeaks = detect_peaks(
            self.ewic_mean,
            min_dist=min_dist,
            thresh=0
        )
        # posmax = np.unique(pospeaks * self.ewic_mean)[-2]

        negpeaks = detect_peaks(
            -self.ewic_mean,
            min_dist=min_dist,
            thresh=0
        )
        # negmax = np.unique(negpeaks * self.ewic_mean)[-2]

        xy_peaks = np.fliplr(np.argwhere(pospeaks + negpeaks))

        if use_avg_ewic_cell:
            window_size = None

        elif window_size == 'auto':
            window_size = np.max(self.basis_mean_real_px) * 4

            if window_size > np.min([self.dp_h, self.dp_w]):
                window_size = np.min([self.dp_h, self.dp_w])

        if graphical_picking or (pick_labels is None):
            dipole_picks = pick_points(
                self.ewic_mean,
                n_picks=2,
                xy_peaks=xy_peaks,
                origin=None,
                graphical_picking=graphical_picking,
                window_size=window_size,
                timeout=timeout,
                cmap='bwr',
                vmax=self.ewic_vmax,
                vmin=-self.ewic_vmax,
                # quickplot_kwargs={'zerocentered': True}
            )

        else:
            dipole_picks = xy_peaks[np.array(pick_labels), :]

        self.dipole_peaks_mean = dipole_picks

    def get_ewic_dipole(
        self,
        window_size=5,
        # fit_only_roi=True,
        ewic_ratio_thresh=0.9
    ):
        """
        Get basis vectors for each scan pixel from the registered EWPC lattice.

        Parameters
        ----------
        window_size : scalar
            Window size of the mask used to find initial guess for peak
            position in each EWPC pattern.
            Default: 7

        ewic_ratio_thresh : scalar
            Thresholding value used to determine whether an individual EWIC
            pattern exhibits a true dipole. 

        Returns
        -------
        None.

        """

        # Initial setup
        t = [time.time()]
        print(self.dipole_peaks_mean)
        dxy = window_size//2
        hann = hann_2d(self.data.shape[-1])
        minval = np.min(self.data, axis=(-2, -1))

        # xy_dipole = self.dipole_peaks_mean

        # Make a mask for selected dipole peaks
        posmask, _, _, pospeaks = watershed_segment(
            (self.ewic_mean),
            # sigma=1,
            bkgd_thresh_factor=0.1,
            min_dist=1
        )
        negmask, _, _, negpeaks = watershed_segment(
            -(self.ewic_mean),
            # sigma=1,
            bkgd_thresh_factor=0.1,
            min_dist=2
        )

        posnum = [posmask[i, j]
                  for [j, i] in self.dipole_peaks_mean
                  if posmask[i, j] != 0
                  ][0]
        negnum = [negmask[i, j]
                  for [j, i] in self.dipole_peaks_mean
                  if negmask[i, j] != 0
                  ][0]

        mask = (np.where(posmask == posnum, 1, 0) +
                np.where(negmask == negnum, 1, 0))

        # Find max & min pixels in the dipole peak regions for each DP.
        # Positive peak will be the first coord in each set, then
        # negative.

        x0y0 = np.array([[
            np.fliplr([
                np.unravel_index(np.argmax(im * mask), im.shape),
                np.unravel_index(np.argmin(im * mask), im.shape)
            ])
            for im in row]
            for row in self.ewic
        ]) / self.upsample_factor

        # Map EWIC peak to EWPC ratio. Apply threshold to determine
        # whether local dipole is present. Only measure EWIC peaks where
        # this analysis determines a local dipole is present, otherwise
        # consider the local dipole to be 0.

        ratios = np.zeros(self.ewic.shape[:2])

        for ij in range(self.scan_h * self.scan_w):
            i, j = [ij//self.scan_w, ij % self.scan_w]
            ewic = self.ewic[i, j].astype(np.float64)[mask == 1]
            ewpc = self.ewpc[i, j][mask == 1]
            if self.roi[i, j] == 0:
                continue

            posint = np.max(ewic)
            if posint > 0:
                posratio = posint / ewpc[ewic == posint][0]
            else:
                posratio = 0

            negint = np.min(ewic)
            if negint < 0:
                negratio = -negint / ewpc[ewic == negint][0]
            else:
                negratio = 0

            ratios[i, j] = np.mean([posratio, negratio])

        self.ewic_ratios = ratios

        # List of scan coords that have a dipole and are in the ROI:
        scan_coords_thresh = [
            [i, j]
            for i in range(self.scan_h)
            for j in range(self.scan_w)
            if ((self.roi[i, j] == 1) & (ratios[i, j] > ewic_ratio_thresh))
        ]

        # Define function for measureing each EWIC pattern
        def find_ewic_dipole_peaks(
                coords,
                log_dp,
                xy_bound,
        ):
            """
            Find the positive and negative peak positions for a dipole
            in the EWIC.
            """

            sign = [1, -1]
            results = [find_ewic_peak(coord, log_dp, xy_bound, sign[i])
                       for i, coord in enumerate(coords)]

            return results

        t += [time.time()]
        print(f'Step 1 (Find peaks): {(t[-1]-t[-2]):.{2}f} sec')

        n_jobs = psutil.cpu_count(logical=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(find_ewic_dipole_peaks)(
                x0y0[i, j],
                np.log(self.data[i, j] - minval[i, j] + 0.1) * hann,
                dxy,
            ) for i, j in tqdm(scan_coords_thresh)
        )

        t += [time.time()]
        print(f'Step 2 (Measure peaks): {(t[-1]-t[-2]):.{2}f} sec')

        local_dipole_thresh = np.array([
            peaks[0] - peaks[1]
            for peaks in results
        ])
        self.local_dipole = np.zeros((self.scan_h, self.scan_w, 2),
                                     dtype=float) * np.nan

        for i, dipole in enumerate(local_dipole_thresh):
            self.local_dipole[*scan_coords_thresh[i]] = dipole

        return results, x0y0


# %%
