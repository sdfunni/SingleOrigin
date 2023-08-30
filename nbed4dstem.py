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

import warnings
import psutil
from tqdm import tqdm
import copy
import time

from joblib import Parallel, delayed

import numpy as np
from numpy.linalg import (
    norm,
    solve,
)

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle

from SingleOrigin.utils import (
    watershed_segment,
    get_feature_size,
    pick_points,
    register_lattice_to_peaks,
    plot_basis,
    fft_square,
    find_cepstrum_peak,
    hann_2d,
    fit_lattice,
)


class DataCube:
    """
    Class for handling analysis of STEM data with 3 or more dimensions.

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
        parameters, etc. This (these) values are applied to axes -4, -3.
        Default: None.

    image_pixel_size : scalar, 2-tuple of scalars or None
        The calibrated scan pixel size for the instrument, experimental
        parameters, etc. This (these) values are applied to axes -2, -1.
        Default: None.

    origin : 2-tuple or None
        The initial guess for the origin (for direct experimental data) or the
        defined origin position (for FFT processed data) in the final two
        dimensions.
        Default: None.

    fix_origin : bool
        If True, origin is allowed to vary for best fit of a lattice. If False,
        the origin not allowed to vary. Fixing the origin is appropriate
        for FFT processed data, while for experimental data, it should be
        allowed to vary during fitting. The later case is a result, for
        example, of beam shifting during scanning in 4D STEM. In the former
        case, the origin is always at (h/2, w/2).
        Default: False.

    origin_type : str (either 'exp' or 'fft')
        The type of the origin:
            'exp', or experimental, is unknown prior to measurement;
            'fft', for images processed by shifted FFT where the origin is
            (h/2, w/2).
        For 'exp', the origin attribute will be set to None initially, unless
        one is passed when initiating the DataCube object.
        For 'fft', the correct origin will be determined from datacube.shape.
        Ensure the datacube has not been manipulated in any way to change the
        correct origin for 'fft'.
        Default: 'fft'

    show_mean_image : bool
        Whether to display the mean image after initiation of the DataCube
        object. Verify that this is representative of the underlying data
        as it will be used to hone some analysis parameters prior to applying
        the analysis to the full DataCube.
        Default: True.

    Attributes
    ----------
    h, w : ints
        The heiht and width of the image/DP in the final two dimensions.

    mean_image : 2d array
        The mean taken along the first len(data.shape) - 2 axes.

    """

    # TODO : Incorporate pixel size calibrations

    # TODO : Incorporate .cif structure for absolute strain reference

    def __init__(
        self,
        datacube,
        scan_pixel_size=None,
        image_pixel_size=None,
        origin=None,
        fix_origin=False,
        origin_type='fft',
        show_mean_image=True,
        get_ewpc=True,
    ):
        self.datacube = datacube  # copy.deepcopy(datacube)
        self.ndims = len(datacube.shape)
        self.data_mean = np.mean(datacube,
                                 axis=tuple(i for i in range(self.ndims-2)))

        self.im_h, self.im_w = self.datacube.shape[-2:]
        self.scan_h, self.scan_w = self.datacube.shape[-4:-2]
        self.scan_pixel_size = scan_pixel_size
        self.image_pixel_size = image_pixel_size
        self.origin = origin
        self.fix_origin = fix_origin
        self.origin_type = origin_type
        self.ref_region = None

        if get_ewpc:
            print('Calculating EWPC. This may take a moment...')
            minval = np.min(datacube)
            self.ewpc = np.abs(fft_square(
                np.log(datacube - minval + 0.1),
                hanning_window=True))  # .astype(np.float16)
            self.ewpc_mean = np.mean(
                self.ewpc,
                axis=tuple(i for i in range(self.ndims-2))
            )
            display_image = self.ewpc_mean
        else:
            display_image = self.data_mean

        if self.origin_type == 'fft':
            self.origin = (int(self.im_h/2), int(self.im_w/2))
            if np.max(display_image) != display_image[self.origin]:
                warnings.warn(
                    "Determined origin is not correct for origin_type='fft'." +
                    "Check that data has not been manipulated, e.g. via " +
                    "cropping. Otherwise, should assign: origin_type='exp'."
                )

        if show_mean_image:
            fig, ax = plt.subplots(1)
            ax.imshow(np.log(display_image), cmap='gray')
            if self.origin is not None:
                ax.scatter(*self.origin, c='red', marker='+')

    def initalize_cepstral_analysis(
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
            scaling=None,
            power=0.2,
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
        """

        timeout = None
        sigma = get_feature_size(self.ewpc_mean)

        if scaling is None:
            display_image = self.ewpc_mean
        else:
            if scaling == 'pwr':
                display_image = self.ewpc_mean ** power
            elif scaling == 'log':
                display_image = np.log(self.ewpc_mean)
            else:
                raise Exception(
                    'Scaling must be: None, "pwr", or "log".')

        if self.origin is None:
            n_picks = 3
        else:
            n_picks = 2

        masks, num_masks, slices, peaks = watershed_segment(
            self.ewpc_mean,
            sigma=sigma,
            min_dist=np.ceil(sigma * 2),
            local_thresh_factor=0,
        )

        self.masks = masks
        xy_peaks = peaks.loc[:, 'x':'y'].to_numpy()

        if graphical_picking or (pick_labels is None):
            basis_picks = pick_points(
                display_image,
                n_picks,
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
                fix_origin=self.fix_origin,
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
                image=display_image,
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

    def get_cepstral_strain(
            self,
            window_size=7,
            ref=None,
            rotation=0,
    ):
        """
        Get strain from the EWPC registered lattice points.

        Parameters
        ----------
        window_size : scalar
            Window size of the mask used to find initial guess for peak
            position in each EWPC pattern.
            Default: 7
        ref : 2x2 array, 2d slice object, or None
            The reference basis vectors or reference region from which to
            measure strain. Reference vectors must be in units of detector
            pixels and rotationally aligned to the average pattern.
            If a slice, is passed, it is intrepreted to be a region of the real
            space scan dimensions. The average lattice parameters in this
            region are taken as the zero strain reference.
            If None, The average vectors from the entire dataset are used as
            the zero strain reference.
            Default: None
        rotation : scalar
            Rotation angle (in degrees) between the real space scan axes and
            the detector axes. Positive is clockwise. This must incorporate
            both normal image-detector rotation angle as well as any applied
            scan rotation. Note that the scan rotation value may be opposite in
            sign compared to the resulting image rotation.


        """

        t = [time.time()]

        p0 = self.basis.flatten().tolist() + list(self.origin)

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
        ])

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

        hann = hann_2d(self.im_h)
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
            fit_lattice(p0, xy, M, fix_origin=True)[:4].reshape((2, 2))
            for xy in results
        ]).reshape((self.scan_h, self.scan_w, 2, 2))

        t += [time.time()]
        print(f'Step 5 (Register lattice): {(t[-1]-t[-2]) :.{2}f} sec')

        if ref is None:
            ref = np.mean(basis_vects, axis=(0, 1))

        elif type(ref[0]) == slice:
            self.ref_region = np.array([[ref[1].start, ref[1].stop],
                                        [ref[0].start, ref[0].stop]])
            ref = np.mean(basis_vects[ref], axis=(0, 1))

        beta = np.array([
            [solve(ref, local_basis)
             for local_basis in row]
            for row in basis_vects
        ])  # .reshape((self.scan_h, self.scan_w, 2, 2))

        exx = (beta[:, :, 0, 0] - 1) * 100
        eyy = (beta[:, :, 1, 1] - 1) * 100
        exy = -(beta[:, :, 0, 1] + beta[:, :, 1, 0]) / 2 * 100
        theta = np.degrees((beta[:, :, 0, 1] - beta[:, :, 1, 0]) / 2)

        t += [time.time()]
        print(f'Step 6 (Calculate strain): {(t[-1]-t[-2]) :.{2}f} sec')

        if rotation is not None:
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

        self.strain_map = {
            'exx': exx,
            'eyy': eyy,
            'exy': exy,
            'theta': theta
        }

        t += [time.time()]
        print(f'Step 7 (Apply rotation): {(t[-1]-t[-2]) :.{2}f} sec')

        print('Done')

    def plot_strain_maps(
            self,
            normal_strain_lim=None,
            shear_strain_lim=None,
            theta_lim=None,
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

        limlist = [normal_strain_lim] * 2 + [shear_strain_lim, theta_lim]
        keys = ['exx', 'eyy', 'exy', 'theta']
        labels = [r'$\epsilon _{xx}$', r'$\epsilon _{yy}$',
                  r'$\epsilon _{xy}$', r'$\theta$']

        # TODO: center limits at zero by default

        vminmax = {key: tuple(limlist[i]) if limlist[i] is not None
                   else (None, None)
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
            label = r'$\circ$' if comp == 'theta' else '%'
            plots += [axs[i].imshow(
                self.strain_map[comp],
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
            cbar.set_label(label=label, fontsize=20, fontweight='bold')
            axs[i].text(self.scan_w/50, self.scan_h/50,
                        labels[i], ha='left', va='top', size=24)

            if self.ref_region is not None:
                dx = self.ref_region[0, 1] - self.ref_region[0, 0]
                dy = self.ref_region[1, 0] - self.ref_region[1, 1]
                corner = [self.ref_region[0, 0], self.ref_region[1, 1] - 1]

                rect = Rectangle(corner, dx, dy, fill=False)
                axs[i].add_patch(rect)

        return fig, axs
