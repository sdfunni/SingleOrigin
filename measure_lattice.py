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

# import warnings
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

from sklearn.cluster import MiniBatchKMeans
from skimage.morphology import erosion, disk

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle

from SingleOrigin.utils import (
    detect_peaks,
    get_feature_size,
    pick_points,
    register_lattice_to_peaks,
    plot_basis,
    fft_square,
    find_cepstrum_peak,
    hann_2d,
    fit_lattice,
    metric_tensor,
    get_astar_2d_matrix,
    rotation_angle_bt_vectors
)

from SingleOrigin.cell_transform import UnitCell


class MeasureLattice:
    #TODO: Modify to handle more data types, e.g. HR STEM image, FFT, single DP, etc.
    
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
    datacube : ndarray
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
        Whether to display the mean image after initiation of the DataCube
        object. Verify that this is representative of the underlying data
        as it will be used to hone some analysis parameters prior to applying
        the analysis to the full DataCube.
        Default: True.

    get_ewpc : bool
        Whether to calculate the exit wave power cepstrum from the diffraction
        space data supplied in datacube.
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

    # TODO : Incorporate .cif structure for absolute strain reference

    def __init__(
        self,
        datacube,
        scan_pixel_size=None,
        dp_pixel_size=None,
        dp_rotation=0,
        origin=None,
        show_mean_image=True,
        get_ewpc=True,
        upsample_factor=1,
    ):
        self.datacube = datacube
        self.ndims = len(datacube.shape)
        self.data_mean = np.mean(datacube,
                                 axis=tuple(i for i in range(self.ndims-2)))

        self.im_h, self.im_w = np.array(self.datacube.shape[-2:])
        self.scan_h, self.scan_w = self.datacube.shape[-4:-2]
        self.scan_pixel_size = scan_pixel_size
        self.dp_pixel_size = dp_pixel_size
        if dp_pixel_size is not None:
            self.ewpc_pixel_size = 1 / (
                self.dp_pixel_size * datacube.shape[-1])
        self.dp_rotation = dp_rotation
        self.origin = origin
        self.ref_region = None
        self.upsample_factor = upsample_factor
        self.origin_shift = 0

        if get_ewpc:
            print('Calculating EWPC. This may take a moment...')
            minval = np.min(datacube)
            self.ewpc = fft_square(
                np.log(datacube - minval + 0.1),
                hann_window=True,
                upsample_factor=upsample_factor)

            self.ewpc_mean = np.mean(
                self.ewpc,
                axis=tuple(i for i in range(self.ndims-2))
            )
            display_image = self.ewpc_mean
        else:
            display_image = self.data_mean

        if upsample_factor is not None:
            self.origin_shift = self.scan_w * (upsample_factor - 1) / 2
            self.im_h *= upsample_factor
            self.im_w *= upsample_factor

        if show_mean_image:
            fig, ax = plt.subplots(1)
            ax.imshow(np.log(display_image), cmap='gray')
            if self.origin is not None:
                ax.scatter(*self.origin, c='red', marker='+')

    def kmeans_segmentation_ewpc(self, n_clusters, window=4):

        ewpc = copy.deepcopy(self.ewpc)

        # Block out central peak:
        ewpc[:, :, 64-window:64+window+1, 64-window:64+window+1] = 0

        # Prepare kmeans analysis
        X_train = ewpc.reshape((-1, 128*128))**0.5
        X_train -= np.min(X_train)
        X_train /= np.max(X_train)

        kmeans = MiniBatchKMeans(n_clusters=n_clusters)
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
            thresh=0.05
    ):
        """
        Initialize EWPC analysis by registring a lattice to the mean image.

        Parameters
        ----------
        pick_basis_order : 2-tuple of ints
            The order of peak selected for each of the basis vactors.
        use_only_basis : bool
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
        scaling : str ('pwr' or 'log') or None
            Whether to scale the displayed image for better visualization.
            Uses power scaling ('pwr') or logrithmic ('log').
            Default: None
        power : scalar
            The exponent to use for power scaling of the pattern for plotting.
            Default: 0.2
        thresh : scalar
            The threshold cutoff for detecting peaks relative to the highest
            peak, vmax (not considering the central EWPC peak). An individual
            peak must have a maximum > thresh * vmax to be detected.

        Returns
        -------
        None.

        """

        timeout = None

        self.origin = np.array([self.im_w/2, self.im_h/2])

        peaks = detect_peaks(self.ewpc_mean, min_dist=2, thresh=0)

        # Find the maximum of the 2nd highest peak (ignores the central peak)
        self.ewpc_vmax = np.unique(peaks*self.ewpc_mean)[-2]

        # Get feature size after applying vmax as a maximum threshold.
        # This prevents the central peak from dominating the size determination
        # vmax will also be used as the upper limit of the imshow cmap.
        ewpc_thresh = np.where(
            self.ewpc_mean > self.ewpc_vmax,
            0,
            self.ewpc_mean
        )

        min_dist = get_feature_size(ewpc_thresh) * 1.5

        # Detect peaks and get x,y coordinates
        peaks = detect_peaks(
            self.ewpc_mean,
            min_dist=min_dist,
            thresh=thresh*self.ewpc_vmax,
        )

        xy_peaks = np.fliplr(np.argwhere(peaks))

        if graphical_picking or (pick_labels is None):
            basis_picks = pick_points(
                ewpc_thresh,
                n_picks=2,
                xy_peaks=xy_peaks,
                origin=self.origin,
                graphical_picking=graphical_picking,
                window_size=None,
                timeout=timeout
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

            self.basis = basis_vects

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

                lattice = lattice[lattice.loc[:, 'y_ref'] > self.origin[1]]

            fig, ax = plot_basis(
                image=ewpc_thresh,
                basis_vects=self.basis,
                origin=self.origin,
                lattice=lattice,
                return_fig=True,
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

        theta = np.radians(self.dp_rotation)
        t = [time.time()]
        m = self.ewpc_mean.shape[0] / 2
        p0 = np.concatenate([self.basis.flatten(), self.origin]
                            ) / self.upsample_factor

        lattice = copy.deepcopy(self.lattice)

        xy_ref = np.around(lattice.loc[:, 'x_ref':'y_ref'].to_numpy()
                           ).astype(int)
        M = lattice.loc[:, 'h':'k'].to_numpy()

        dxy = window_size//2

        masks = np.zeros((xy_ref.shape[0], self.im_h, self.im_w))

        for i, mask in enumerate(masks):
            mask[xy_ref[i, 1]-dxy: xy_ref[i, 1] + dxy + 1,
                 xy_ref[i, 0]-dxy: xy_ref[i, 0] + dxy + 1] = 1

        t += [time.time()]
        print(f'Step 1 (Initial setup): {(t[-1]-t[-2]) :.{2}f} sec')

        x0y0 = np.array([[
            [np.flip(np.unravel_index(np.argmax(im * mask),
                                      (self.im_h, self.im_w)))
             for mask in masks
             ]
            for im in row]
            for row in self.ewpc
        ]) / self.upsample_factor

        t += [time.time()]
        print(f'Step 2 (Find peaks): {(t[-1]-t[-2]) :.{2}f} sec')

        def find_mult_ewpc_peaks(
                coords,
                log_dp,
                xy_bound,
        ):

            results = [find_cepstrum_peak(coord, log_dp, xy_bound)
                       for coord in coords]

            return results

        hann = hann_2d(self.datacube.shape[-1])
        minval = np.min(self.datacube)

        args_packed = [[x0y0[i, j],
                        np.log(self.datacube[i, j] - minval + 0.1) * hann,
                        dxy]
                       for i in range(self.scan_h)
                       for j in range(self.scan_w)]

        t += [time.time()]
        print(f'Step 3 (Pack data): {(t[-1]-t[-2]) :.{2}f} sec')

        n_jobs = psutil.cpu_count(logical=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(find_mult_ewpc_peaks)(
                *args,
            ) for args in tqdm(args_packed)
        )

        t += [time.time()]
        print(f'Step 4 (Measure peaks): {(t[-1]-t[-2]) :.{2}f} sec')

        basis_vects = np.array([
            fit_lattice(
                p0,
                np.array(xy),
                M,
                fix_origin=True,
            )[:4].reshape((2, 2))
            for xy in results
        ]).reshape((self.scan_h, self.scan_w, 2, 2))

        rot_mat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        # Apply rotation
        basis_vects = np.array([
            [basis @ rot_mat for basis in row]
            for row in basis_vects])

        self.basis_vects_real_px = basis_vects

        # Get reciprocal space basis (i.e. for the diffraction patterns)
        self.basis_vects_recip_px = np.array([
            [inv(basis.T) * m for basis in row]
            for row in basis_vects])

        # Get the mean real space basis vectors
        self.basis_mean_real_px = np.mean(
            self.basis_vects_real_px, axis=(0, 1)
        )
        # Get the mean reciprocal space basis
        self.basis_mean_recip_px = np.mean(
            self.basis_vects_recip_px, axis=(0, 1)
        )

        if self.dp_pixel_size is not None:

            # Make lattice parameter maps
            lattice_maps = {}
            # Find lattice parameter distances
            lattice_maps['a1'] = norm(
                np.squeeze(self.basis_vects_real_px[:, :, 0, :]),
                axis=-1
            ) * self.ewpc_pixel_size

            lattice_maps['a2'] = norm(
                np.squeeze(self.basis_vects_real_px[:, :, 1, :]),
                axis=-1
            ) * self.ewpc_pixel_size

            # Find lattice parameter angle
            lattice_maps['gamma'] = np.array([
                [np.abs(rotation_angle_bt_vectors(basis[0], basis[1]))
                 for basis in row]
                for row in self.basis_vects_real_px
            ])

            # Find angle between local & mean basis vectors
            theta1 = np.array([
                [rotation_angle_bt_vectors(
                    basis[0],
                    self.basis_mean_real_px[0]
                )
                    for basis in row]
                for row in self.basis_vects_real_px
            ])

            theta2 = np.array([
                [rotation_angle_bt_vectors(
                    basis[1],
                    self.basis_mean_real_px[1]
                )
                    for basis in row]
                for row in self.basis_vects_real_px
            ])

            # Find lattice rotation relative to mean basis
            lattice_maps['theta'] = (theta1 + theta2) / 2

            self.lattice_maps = lattice_maps

        t += [time.time()]
        print(f'Step 5 (Register lattice): {(t[-1]-t[-2]) :.{2}f} sec')

    def overlay_mean_basis(self):
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

        ewpc_df = self.ewpc[:, :, int(self.origin[1]), int(self.origin[0])]
        fig, ax = plot_basis(
            ewpc_df,
            self.basis_mean_real_px,
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
            return_fig=False):
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
            vmin = np.around(np.nanmin(
                np.where(mask, self.lattice_maps[comp], np.nan)),
                decimals)
            vmax = np.around(np.nanmax(
                np.where(mask, self.lattice_maps[comp], np.nan)),
                decimals)

            if i == 0:
                size = np.mean(norm(self.basis_mean_real_px, axis=1))

                plot_basis_vects = self.basis_mean_real_px * \
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
                axs[i].text(axis_origin[0]+plot_basis_vects[0, 0] * 1.2,
                            axis_origin[1]+plot_basis_vects[0, 1] * 1.2,
                            r'$a_1$',
                            size=16,)
                axs[i].text(axis_origin[0]+plot_basis_vects[1, 0] * 1.2,
                            axis_origin[1]+plot_basis_vects[1, 1] * 1.2,
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

            # cbar.ax.tick_params(labelsize=16)
            cbar.set_label(label=units[i], fontsize=20, fontweight='bold',
                           rotation='horizontal')
            axs[i].text(self.scan_w/50, self.scan_h/50,
                        labels[i], ha='left', va='top',
                        size=24, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, lw=0)
                        )

            # if self.ref_region is not None:
            #     dx = self.ref_region[0, 1] - self.ref_region[0, 0]
            #     dy = self.ref_region[1, 0] - self.ref_region[1, 1]
            #     corner = [self.ref_region[0, 0], self.ref_region[1, 1] - 1]

            #     rect = Rectangle(corner, dx, dy, fill=False)
            #     axs[i].add_patch(rect)

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
        window_size : scalar
            Window size of the mask used to find initial guess for peak
            position in each EWPC pattern.
            Default: 7

        ref : 2x2 array, 2d slice object, int or None
            The reference basis vectors or reference region from which to
            measure strain. Reference vectors must be in units of detector
            pixels and rotationally aligned to the average pattern.
            If a slice, is passed, it is intrepreted to be a region of the real
            space scan dimensions. The average lattice parameters in this
            region are taken as the zero strain reference.
            If k-means segmentation has been run, passing an integer is
            selects one of the k-means labels as the reference region.
            If None, The average vectors from the entire dataset are used as
            the zero strain reference.
            Default: None

        rotation : scalar
            Reference frame rotation for strain relative to the scan axes, in
            degrees. Positive is clockwise; negative is counterclockwise.

        Returns
        -------
        None.

        """

        if ref is None:
            ref = self.basis_mean_real_px
            self.ref_region = None

        elif (type(ref) == tuple):
            self.ref_region = np.array([[ref[1].start, ref[1].stop],
                                        [ref[0].start, ref[0].stop]])
            ref = np.mean(self.basis_vects_real_px[ref], axis=(0, 1))

        elif type(ref) == int:
            mask = np.where(self.kmeans_labels == ref, True, False)
            mask = erosion(mask, footprint=np.ones((roi_erode*2 + 1,)*2))
            ref = np.mean(
                self.basis_vects_real_px[mask],
                axis=0,
            )

        beta = np.array([
            [solve(ref, local_basis)
             for local_basis in row]
            for row in self.basis_vects_real_px
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
                [np.cos(rotation), np.sin(rotation)],
                [-np.sin(rotation), np.cos(rotation)]
            ]) * np.min([self.scan_h, self.scan_w]) * 0.05

        self.strain_map = {
            'exx': exx,
            'eyy': eyy,
            'exy': exy,
            'theta': theta
        }

        print('Done')

    def plot_strain_maps(
            self,
            normal_strain_lim=None,
            shear_strain_lim=None,
            theta_lim=None,
            kmeans_roi=None,
            roi_erode=1,
            plot_strain_axes=True,
            strain_axes_origin=(0.1, 0.1),
            figsize=(10, 10),
            return_fig=False):
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
        plot_strain_axes : bool
            Whether to plot vectors showing the axes of the displayed strain
            fields, i.e. the x and y strain directions.
        strain_axes_origin : 2-tuple
            The origin location of the displayed strain axes vectors
            (if plot_strain_axes is True) relative to the (x, y) plot axes,
            e.g. (0.1, 0.1) is the upper left corner, 10% in from each edge.
            Default: (0.1, 0.1)
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

        limlist = [normal_strain_lim] * 2 + [shear_strain_lim, theta_lim]
        keys = ['exx', 'eyy', 'exy', 'theta']
        labels = [r'$\epsilon _{xx}$', r'$\epsilon _{yy}$',
                  r'$\epsilon _{xy}$', r'$\theta$']

        # TODO: center limits at zero by default

        vminmax = {
            key: tuple(limlist[i]) if limlist[i] is not None
            else (None, None)  # find the min and max
            for i, key in enumerate(keys)
        }

        fig, axs = plt.subplots(
            2, 2,
            sharex=True,
            sharey=True,
            figsize=figsize,
            # tight_layout=True,
            layout='constrained',
        )
        axs = axs.flatten()
        plots = []
        for i, comp in enumerate(keys):
            if (i == 0) & plot_strain_axes:
                print('plotting strain axes')
                axis_origin = [self.scan_w * strain_axes_origin[0],
                               self.scan_h * strain_axes_origin[1]]
                axs[i].arrow(
                    axis_origin[0],
                    axis_origin[1],
                    self.strain_basis[0, 0],
                    self.strain_basis[0, 1],
                    fc='black',
                    ec='black',
                    width=0.1,
                    length_includes_head=True,
                    # head_width=2,
                    # head_length=3,
                    label='1',
                )
                axs[i].arrow(
                    axis_origin[0],
                    axis_origin[1],
                    self.strain_basis[1, 0],
                    self.strain_basis[1, 1],
                    fc='black',
                    ec='black',
                    width=0.1,
                    length_includes_head=True,
                    # head_width=2,
                    # head_length=3,
                )
                axs[i].text(axis_origin[0]+self.strain_basis[0, 0] * 1.2,
                            axis_origin[1]+self.strain_basis[0, 1] * 1.2,
                            'xx')
                axs[i].text(axis_origin[0]+self.strain_basis[1, 0] * 1.2,
                            axis_origin[1]+self.strain_basis[1, 1] * 1.2,
                            'yy')

            units = r'$\circ$' if comp == 'theta' else '%'
            plots += [axs[i].imshow(
                np.where(mask, self.strain_map[comp], np.nan),
                cmap='RdBu_r',
                vmin=vminmax[comp][0],
                vmax=vminmax[comp][1])]

            axs[i].set_xticks([])
            axs[i].set_yticks([])

            cbar = plt.colorbar(
                plots[i],
                # cax=cbax,
                orientation='horizontal',
                shrink=0.3,
                aspect=10,
                ticks=vminmax[comp] + (np.mean(vminmax[comp]),),
                pad=0.02,

            )

            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(label=units, fontsize=20, fontweight='bold')
            axs[i].text(self.scan_w/50, self.scan_h/50,
                        labels[i], ha='left', va='top', size=24,
                        # bbox={'alpha' : 0.2,
                        #       'color' : 'white',
                        #       'fill' : True},
                        )

            if self.ref_region is not None:
                dx = self.ref_region[0, 1] - self.ref_region[0, 0]
                dy = self.ref_region[1, 0] - self.ref_region[1, 1]
                corner = [self.ref_region[0, 0], self.ref_region[1, 1] - 1]

                rect = Rectangle(corner, dx, dy, fill=False)
                axs[i].add_patch(rect)

        return fig, axs

    def plot_dp_basis(self, dp_origin=None, basis_factor=[1,1]):
        
        basis_factor = np.array(basis_factor, ndmin=2).T

        if dp_origin is None:
            dp_origin = np.flip(np.unravel_index(np.argmax(self.data_mean),
                                                 self.datacube.shape[-2:]))

        theta = -np.radians(self.dp_rotation)
        rot_matrix = np.array([[np.cos(theta), np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]])

        fig, ax = plot_basis(
            np.log(self.data_mean),
            self.basis_mean_recip_px @ rot_matrix * basis_factor,
            dp_origin,
            return_fig=True)
        ax.legend()

    def plot_real_basis(self, basis_factor=[1,1]):
        
        

        fig, ax = so.plot_basis(
            self.ewpc_mean,
            self.basis_mean_real_px,
            self.origin,
            return_fig=True,
            vmax=self.ewpc_vmax,
            vmin=None)
        ax.legend()

    def calibrate_dp(self, g1, g2, cif_path):
        """Calibrate detector pixel size from EWPC basis vectors.

        Parameters
        ----------
        g1, g2 : array-like of shape (3,)
            The [h,k,l] indices of the plane spacings described by the basis
            vectors chosen for the EWPC analysis.
        cif_path : str
            Path to the .cif file describing the evaluated structure.

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

        alpha_meas = copy.copy(self.basis_mean_recip_px)
        if np.cross(alpha_meas[0], alpha_meas[1]) < 0:
            alpha_meas[0] *= -1

        uc = UnitCell(cif_path)

        lattice_params = [uc.a, uc.b, uc.c, uc.alpha, uc.beta, uc.gamma]

        g = metric_tensor(*lattice_params)

        a_star_2d = get_astar_2d_matrix(g1, g2, g)

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

        return beta
