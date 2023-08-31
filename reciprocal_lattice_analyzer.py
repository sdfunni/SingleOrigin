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
    along with this program.  If not, see https://www.gnu.org/licenses"""


import numpy as np
from numpy.linalg import norm

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from scipy.ndimage import (
    center_of_mass,
    gaussian_filter,
    gaussian_laplace,
)

from skimage.transform import (downscale_local_mean, rescale)

from SingleOrigin.utils import (
    image_norm,
    watershed_segment,
    fit_lattice,
    absolute_angle_bt_vectors,
)


class ReciprocalImage:
    """
    Class for analizing reciprocal lattice images (i.e. an FFT or diffraction
    pattern.)

    Parameters
    ----------
    image : 2D array
        The reciprocal lattice image to analize. An appropriate FFT can be
        taken from a lattice iamge using the 'fft_square()' function in
        SingleOrigin.
    pixel_size : float or None
        The pixel size of the reciprocal lattice image. For diffraction
        patterns, stored in the metadata for .dm4, .emd files, etc. For FFTs,
        pass the image pixel size for this argument and set fft_or_dp='fft'
        and the FFT pixel size will be automatically calculated. If None, all
        analysis done in units of pixels. Must be in Angstroms or 1/Angstroms,
        respectively.
        Default: None.
    fft_or_dp : str ('fft' or 'dp')
        Whether the supplied image is an FFT ('fft') or a diffraction pattern
        ('dp'). If an FFT, the origin of the reciprocal lattice will be
        determined as the maximum valued pixel and not allowed to vary during
        lattice refinement. In that case, the user picks only the two
        reciprocal vector spots. Otherwise, the user picks the origin first,
        (i.e. the direct beam) followed by the reciprocal basis vector spots
        (2nd and 3rd).
        Default: 'fft'.
    """

    def __init__(
            self,
            image,
            pixel_size,
            pixel_units=r'/AA',
            origin=None,
            fft_or_dp='fft',
    ):
        self.image = image
        self.h, self.w = self.image.shape
        self.fft_or_dp = fft_or_dp
        if self.fft_or_dp == 'fft':
            self.pixel_size = 1 / (self.h * pixel_size)
            self.pixel_units = rf'{pixel_units}$^{-1}$'
        else:
            self.pixel_size = pixel_size
            self.pixel_units = pixel_units
        self.latt_dict = {}
        self.recip_latt = None
        self.origin = origin
        self.a1_star = None
        self.a2_star = None
        self.a_star = None

    def get_recip_basis(
        self,
        a1_order=1,
        a2_order=1,
        sigma=3,
        min_dist=5,
        buffer=5,
        thresh_factor_std=1,
        thresh_abs=0.1,
        peak_thresh_factor=0.1,
        max_order=5,
        origin=None,
        a_star=None,
        show_fit=True,
        verbose=True,
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
        sigma : int or float
            The Laplacian of Gaussian sigma value to use for sharpening of the
            peaks for the peak finding algorithm. Usually a value between 2
            and 10 will work well.
        thresh_factor_std : scalar
            Relative adjustment of the threshold level for detecting peaks.
            Greater values will detect fewer peaks; smaller values, more.
            The thresholding is done based on the standard deviation of the
            peak and its surrounding area. Dim peaks will have low standard
            deviations.
            Default: 1.
        thresh_abs : scalar
            The threshold (in % max image intensith) of peak amplitude to
            consider as a possible reciprocal lattice spot. Used to remove
            noise peaks with high local standard devaition from being detected
            as Bragg peaks.

        Returns
        -------
        ...

        """

        if self.fft_or_dp == 'fft':
            U = self.h // 2
            origin = np.array([U, U])
            n_picks = 2
            pwr_factor = 0.1
            fix_origin = True

        elif self.fft_or_dp == 'dp':
            if origin is None:
                n_picks = 3
            elif a_star is None:
                n_picks = 2
            else:
                n_picks = 0

            fix_origin = False
            pwr_factor = 0.5
        else:
            raise Exception(
                "'fft_or_dp' must be 'fft' or 'dp'."
            )

        # Downsample for speed
        # if np.max([self.h, self.w]) > 2000:
        #     print('Downsampling...')
        #     factor = int(np.max([self.h, self.w])/1e3)
        #     image_ds = downscale_local_mean(self.image, (factor, factor))
        #     if sigma > 0:
        #         image_der = image_norm(-gaussian_laplace(
        #             gaussian_filter(image_ds, 1), sigma/factor))
        #     else:
        #         image_der = self.image

        # else:
        if sigma > 0:
            image_der = image_norm(-gaussian_laplace(
                gaussian_filter(self.image, 1), sigma))
        else:
            image_der = self.image
        masks, num_masks, _, spots = watershed_segment(
            image_der,
            local_thresh_factor=0,
            max_thresh_factor=peak_thresh_factor,
            buffer=buffer,
            min_dist=min_dist
        )

        # Upsample
        # if np.max([self.h, self.w]) > 2000:
        #     masks = rescale(masks, (factor, factor), order=0)
        #     spots.loc[:, 'x':'y'] *= factor

        # Remove edge pixels:
        if buffer > 0:
            spots = spots[
                ((spots.x >= buffer) &
                 (spots.x <= self.w - buffer) &
                 (spots.y >= buffer) &
                 (spots.y <= self.h - buffer))
            ].reset_index(drop=True)

        spots['stdev'] = [
            np.std(image_der[int(y-sigma):int(y+sigma+1),
                             int(x-sigma):int(x+sigma+1)])
            for [x, y]
            in np.around(spots.loc[:, 'x':'y']).to_numpy(dtype=int)
        ]

        thresh = 0.003 * thresh_factor_std

        if thresh > 0 or thresh_abs > 0:
            spots = spots[
                (spots.loc[:, 'stdev'] > thresh) &
                (spots.loc[:, 'max'] > thresh_abs)
            ].reset_index(drop=True)

        xy = spots.loc[:, 'x':'y'].to_numpy(dtype=float)
        for i, xy_ in enumerate(xy):
            mask_num = masks[int(xy_[1]), int(xy_[0])]
            mask = np.where(masks == mask_num, 1, 0)
            com = np.flip(center_of_mass(self.image*mask))
            spots.loc[i, ['x', 'y']] = com

        xy = spots.loc[:, 'x':'y'].to_numpy(dtype=float)

        if n_picks > 0:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow((self.image)**(pwr_factor), cmap='gray')
            ax.scatter(xy[:, 0], xy[:, 1], c='red', marker='+')
            if self.fft_or_dp == 'fft':
                ax.scatter(origin[0], origin[1], c='white', marker='+')
                fig.suptitle(
                    'Pick peaks for the a1* and a2* reciprocal basis vectors' +
                    ' \n in that order).',
                    fontsize=12,
                    c='black',
                )

            elif self.fft_or_dp == 'dp':
                fig.suptitle(
                    'Pick peaks for the origin, a1* and a2* reciprocal basis' +
                    ' \n vectors (in that order).',
                    fontsize=12,
                    c='black',
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if self.fft_or_dp == 'fft':
                ax.set_xlim(np.min(xy[:, 0]) - 100, np.max(xy[:, 0]) + 100)
                ax.set_ylim(np.max(xy[:, 1]) + 100, np.min(xy[:, 1]) - 100)

            basis_picks_xy = np.array(plt.ginput(n_picks, timeout=30))

            plt.close('all')

        else:
            basis_picks_xy = a_star + origin

        # Match peaks to basis click points or passed a_star/origin data
        vects = np.array([xy - i for i in basis_picks_xy])
        inds = np.argmin(norm(vects, axis=2), axis=1)
        basis_picks_xy = xy[inds, :]

        a1_pick = basis_picks_xy[-2, :]
        a2_pick = basis_picks_xy[-1, :]

        if ((self.fft_or_dp == 'dp') & (origin is None)):
            origin = basis_picks_xy[0, :]

        # Generate reciprocal lattice
        a1_star = (a1_pick - origin) / a1_order
        a2_star = (a2_pick - origin) / a2_order

        a_star = np.array([a1_star, a2_star])

        recip_latt_indices = np.array(
            [[i, j]
             for i in range(-max_order, max_order+1)
             for j in range(-max_order, max_order+1)]
        )

        xy_ref = recip_latt_indices @ a_star + origin

        # Match reciprocal lattice points to peaks; make DataFrame
        vects = np.array([xy - xy_ for xy_ in xy_ref])
        inds = np.argmin(norm(vects, axis=2), axis=1)

        self.recip_latt = pd.DataFrame({
            'h': recip_latt_indices[:, 0],
            'k': recip_latt_indices[:, 1],
            'x_ref': xy_ref[:, 0],
            'y_ref': xy_ref[:, 1],
            'x_fit': [xy[ind, 0] for ind in inds],
            'y_fit': [xy[ind, 1] for ind in inds],
            'mask_ind': inds
        })

        # Remove peaks that are too far from initial reciprocal lattice
        self.recip_latt = self.recip_latt[norm(
            self.recip_latt.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
            - self.recip_latt.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
            axis=1
        ) < 0.1*np.max(norm(a_star, axis=1))
        ].reset_index(drop=True)

        # Refine reciprocal basis vectors
        M_star = self.recip_latt.loc[:, 'h':'k'].to_numpy(dtype=float)
        xy = self.recip_latt.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)

        p0 = np.concatenate((a_star.flatten(), origin))

        params = fit_lattice(p0, xy, M_star, fix_origin=fix_origin)

        # Save data and report key values
        self.a1_star = params[:2]
        self.a2_star = params[2:4]
        if len(params) == 6:
            self.origin = params[4:]
        else:
            self.origin = origin

        self.a_star = np.array([self.a1_star, self.a2_star])

        self.recip_latt[['x_ref', 'y_ref']] = (
            self.recip_latt.loc[:, 'h':'k'].to_numpy(dtype=float)
            @ self.a_star
            + origin
        )

        theta = absolute_angle_bt_vectors(
            self.a1_star,
            self.a2_star,
            np.identity(2)
        )

        ratio = norm(self.a1_star)/norm(self.a2_star)

        if verbose:
            print(f'Reciproal lattice angle (degrees): {theta :.{3}f}')
            print(f'Reciproal vector ratio (a1*/a2*): {ratio :.{5}f}')

        # Plot refined basis
        if show_fit:
            fig2, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.image**0.2, cmap='plasma')
            ax.scatter(
                self.recip_latt.loc[:, 'x_ref'].to_numpy(dtype=float),
                self.recip_latt.loc[:, 'y_ref'].to_numpy(dtype=float),
                marker='+',
                c='red'
            )
            ax.scatter(self.origin[0], self.origin[1], marker='+', c='white')

            ax.arrow(
                origin[0],
                origin[1],
                a1_star[0],
                a1_star[1],
                fc='red',
                ec='white',
                width=0.1,
                length_includes_head=True,
                head_width=2,
                head_length=3
            )
            ax.arrow(
                origin[0],
                origin[1],
                a2_star[0],
                a2_star[1],
                fc='green',
                ec='white',
                width=0.1,
                length_includes_head=True,
                head_width=2,
                head_length=3
            )

            if self.fft_or_dp == 'fft':
                ax.set_xlim(np.min(self.recip_latt.loc[:, 'x_ref']) - 100,
                            np.max(self.recip_latt.loc[:, 'x_ref']) + 100)
                ax.set_ylim(np.max(self.recip_latt.loc[:, 'y_ref']) + 100,
                            np.min(self.recip_latt.loc[:, 'y_ref']) - 100)

            ax.set_xticks([])
            ax.set_yticks([])
            plt.title('Reciprocal Lattice Fit')

    def get_reciprocal_vector_length(self, from_origin=True):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow((self.image)**(1), cmap='gray')
        ax.scatter(self.peaks[:, 0], self.peaks[:, 1], c='black', marker='+')
        if from_origin:
            ax.scatter(self.origin[0], self.origin[1], c='white', marker='+')
            fig.suptitle(
                'Pick peak for the desired vector from reciprocal origin.',
                fontsize=12,

                c='black',
            )
            n_picks = 1
        else:
            fig.suptitle(
                'Pick 2 peaks between which to measure the reciprocal vector.',
                fontsize=12,
                c='black',
            )
            n_picks = 2

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(np.min(self.peaks[:, 0]) - 100,
                    np.max(self.peaks[:, 0]) + 100)
        ax.set_ylim(np.max(self.peaks[:, 1]) + 100,
                    np.min(self.peaks[:, 1]) - 100)

        picks_xy = np.array(plt.ginput(n_picks, timeout=30))

        if from_origin:
            picks_xy = np.vstack([picks_xy, self.origin])

        g_mag = norm(picks_xy[0] - picks_xy[1]) * self.pixel_size

        return g_mag

    def plot_scattering_ring(self, dist=None, origin=None):

        if origin is None:
            origin = self.origin

        if dist is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.image, cmap='gray')
            ax.scatter(origin[0], origin[1], c='white', marker='+')
            fig.suptitle(
                'Pick peaks for the a1* and a2* reciprocal basis vectors \n' +
                '(in that order).',
                fontsize=12,

                c='black',
            )

            ax.set_xticks([])
            ax.set_yticks([])

            # ax.set_xlim(
            #     np.min(self.peaks[:, 0]) - 100,
            #     np.max(self.peaks[:, 0]) + 100
            # )
            # ax.set_ylim(
            #     np.max(self.peaks[:, 1]) + 100,
            #     np.min(self.peaks[:, 1]) - 100
            # )

            picks_xy = np.array(plt.ginput(1, timeout=30))

            dist = float(norm(picks_xy - origin))
            print(type(dist))
        plt.close('all')

        # Plot refined basis
        if type(dist) == float:
            dist = np.array([dist])
        fig2, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image, cmap='gray')

        ax.scatter(self.origin[0], self.origin[1], marker='+', c='white')
        for d in dist:
            ax.add_artist(Circle(
                origin,
                radius=d,
                ec='red',
                fill=False,
                lw=0.3,
                alpha=1,
                transform=ax.transData,
            ))

        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(
            'Diffraction Ring: '
            + f'{np.around(np.array(dist)*self.pixel_size, decimals=3)}'
            + f'{self.pixel_units}}}')

    def get_orientation_and_phase(self, cifs, max_peak_order=5):
        """Determine likely orientation and (optionally) phase produced the
        diffraction pattern or FFT.

        Given a reciprocal lattice fit and list of candidate phase .cifs,
        determins most likely orientation and phase. ReciprocalImage object
        must have a reciprocal lattice fit generated by the get_recip_basis()
        method first.

        Parameters
        ----------
        cifs : string or list of strings
            The path(s) to the candidate phase .cif files. If only one file
            path is passed, phase will be assumed to be correct...
        max_peak_order : int
            Maximum peak order calculated for candidate orientations.

        Returns
        -------
        ...

        TODO : WRITE THIS FUNCTION.
        """
