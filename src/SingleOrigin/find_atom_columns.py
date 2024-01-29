"""SingleOrigin is a module for atomic column position finding intended for
    high resolution scanning transmission electron microscope images.
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

import copy
import time
from tqdm import tqdm
import psutil
# import types

from joblib import Parallel, delayed

import numpy as np
from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Rectangle,
    Circle,
    Ellipse,
    Polygon,
)
from matplotlib.legend_handler import HandlerPatch
from matplotlib import colors as colors
from matplotlib.colors import Normalize
from matplotlib.backend_bases import MouseButton
from matplotlib.cm import ScalarMappable


from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib_scalebar.scalebar import ScaleBar

from scipy.optimize import minimize
from scipy.ndimage import (
    gaussian_filter,
    gaussian_laplace,
    rotate,
    map_coordinates,
    center_of_mass,
    binary_fill_holes,
    binary_erosion,
)

from skimage.draw import polygon2mask, polygon
from skimage.transform import (downscale_local_mean, rescale)
from skimage.morphology import erosion

from SingleOrigin.utils import (
    image_norm,
    linearKDE_2D,
    img_ellip_param,
    gaussian_2d,
    fit_gaussian_ellip,
    pack_data_prefit,
    fit_gaussian_group,
    detect_peaks,
    watershed_segment,
    std_local,
    fft_square,
    get_vpcf,
    fit_lattice,
    get_feature_size,
    rotation_matrix,
    rotate_xy,
    rotation_angle_bt_vectors,
    quickplot,
)

# %%


class HRImage:
    """Object class used for analysis of atomic resolution HR STEM images.

    This is a parent class to hold a HRSTEM image and one or more
    AtomColumnLattice objects to analize the structure(s) in the image.
    This data structure organizes the processing workflow and
    results visualization for an image containing one or more lattices.

    Parameters
    ----------
    image : 2D array
        The STEM image to analize.

    pixel_size_cal : scalar or None
        The calibrated pixel size from the instrument. Usually stored in
        the metadata for .dm4, .emd files, etc.
        Default: None.

    Attributes
    ----------
    h, w : ints
        The height and width of the image.

    latt_dict : dict
        Dictionary of AtomColumnLattice objects added to the image using the
        "add_lattice()" method.

    Methods
    -------
    add_lattice(
        self,
        name,
        unitcell,
        origin_atom_column=None
        ):
        Initiatites a new AtomColumnLattice object associated with the
        HRImage object.

    rotate_image_and_data(
        self,
        lattice_to_align,
        align_basis='a1',
        align_dir='horizontal'
        ):
        Rotates the image and data to align a basis vector to horizontal or
        vertical direction.

    plot_atom_column_positions(
        self,
        filter_by='elem',
        sites_to_fit='all',
        fit_or_ref='fit',
        outlier_disp_cutoff=None,
        plot_masked_image=False
        ):
        Plot fitted or reference atom colum positions.

    plot_disp_vects(
        self,
        filter_by='elem',
        sites_to_plot='all',
        titles=None,
        xlim=None,
        ylim=None,
        scalebar=True,
        scalebar_len_nm=2,
        arrow_scale_factor = 1,
        outlier_disp_cutoff = None,
        max_colorwheel_range_pm=None,
        plot_fit_points=False,
        plot_ref_points=False
        ):
        Plot dislacement vectors between reference and fitted atom colum
        positions.

    """

    def __init__(
            self,
            image,
            pixel_size_cal=None,
    ):
        self.image = copy.deepcopy(image.astype(np.float32))
        self.h, self.w = self.image.shape
        self.pixel_size_cal = pixel_size_cal
        self.latt_dict = {}

    def add_lattice(
            self,
            name,
            unitcell,
            # probe_fwhm=0.8,
            origin_atom_column=None,
    ):
        """Initiatites a new AtomColumnLattice object associated with the
        HRImage object.

        Parameters
        ----------
        name : str
            A unique name for the new lattice. Will be used as the dictionary
            key for the lattice in self.latt_dict.

        unitcell : UnitCell class object
            An appropriate UnitCell object of the SingleOrigin module with
            project_uc_2d() method applied. See examples for how to generate
            this.

        origin_atom_column : int
            The DataFrame row index (in unitcell.at_cols) of the atom column
            that is later picked by the user to register the reference lattice.
            If None, the closest atom column to the unit cell origin is
            automatically chosen.
            Default: None.

        Returns
        -------
        new_lattice: The new AtomicColumnLattice object.

        """
        new_lattice = AtomicColumnLattice(
            self.image,
            unitcell,
            # probe_fwhm,
            origin_atom_column=origin_atom_column,
            pixel_size_cal=self.pixel_size_cal
        )
        self.latt_dict[name] = new_lattice

        return new_lattice

    def rotate_image_and_data(
            self,
            lattice_to_align,
            align_basis='a1',
            align_dir='right',
    ):
        """Rotates the image and data to align a basis vector to image edge.

        Rotates the image so that a chosen crystalloraphic basis vector
        is horizontal or vertical. Adjusts the reference lattice(s) and
        atomic column fit data accordingly. Useful for displaying data for
        presentation.

        Parameters
        ----------
        lattice_to_align : str
            The name of the lattice that will be aligned by the image rotation.

        align_basis : str ('a1' or 'a2')
            The basis vector to align.
            Default 'a1'.

        align_dir : str ('right' or 'left' or 'up' or 'down')
            Direction to align the chosen basis vector.
            Default 'right'.

        Returns
        -------
        rot_ : HRImage object with rotated image.

        lattice_dict : Dictionary of associated lattices with rotated data.
            Some of the attributes do not rotate in a meaningful way and are,
            therefore, set to None (e.g. mask arrays are destroyed by the
            interpolation necessary for arbitrary rotations.)

        """

        rot_ = HRImage(self.image, pixel_size_cal=self.pixel_size_cal)

        if align_basis == 'a1':
            align_vect = self.latt_dict[lattice_to_align].a1
        elif align_basis == 'a2':
            align_vect = self.latt_dict[lattice_to_align].a2

        self.latt_dict[lattice_to_align]

        '''Find the rotation angle and direction'''
        angle = np.degrees(np.arctan2(align_vect[1], align_vect[0]))
        if align_dir == 'right':
            pass
        elif align_dir == 'up':
            angle += np.pi / 2
        elif align_dir == 'left':
            angle += np.pi
        elif align_dir == 'down':
            angle += 3 * np.pi / 2
        else:
            raise Exception(
                "align_dir must be 'right' or 'left' or 'up' or 'down'"
            )

        print('Rotation angle:', angle)

        rot_.image = rotate(rot_.image, angle)
        [rot_.h, rot_.w] = rot_.image.shape
        rot_.fft = fft_square(rot_.image)

        for key, lattice in self.latt_dict.items():
            key += '_rot'
            rot_.latt_dict[key] = copy.deepcopy(lattice)
            lattice_rot = rot_.latt_dict[key]

            lattice_rot.image = rot_.image
            lattice_rot.roi_mask = rotate(lattice_rot.roi_mask,
                                          angle)

            lattice_rot.at_cols_uncropped = None

            '''Translation of image center due to increased image array size
                resulting from the rotation'''
            origin_shift = np.flip((
                (np.array(lattice_rot.image.shape, ndmin=2)-1)/2
                - (np.array(self.image.shape, ndmin=2)-1)/2),
                axis=1
            )
            '''Find the origin-shifted rotation matrix for transforming atomic
                column position data'''

            xy_fit = lattice_rot.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy(
                dtype=float)

            rotation_origin = (np.array(self.image.shape)-1)/2

            lattice_rot.at_cols[['x_fit', 'y_fit']] = \
                rotate_xy(xy_fit, angle, rotation_origin) \
                + origin_shift

            xy_ref = lattice_rot.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(
                dtype=float)

            lattice_rot.at_cols[['x_ref', 'y_ref']] = \
                rotate_xy(xy_ref, angle, rotation_origin) \
                + origin_shift

            [lattice_rot.x0, lattice_rot.y0] = (rotate_xy(
                np.array([[lattice_rot.x0, lattice_rot.y0]]),
                angle,
                rotation_origin
            ) + origin_shift).squeeze()

            '''Transform data'''
            tmat = rotation_matrix(angle, rotation_origin)
            print(tmat)

            lattice_rot.dir_struct_matrix = (
                lattice_rot.dir_struct_matrix
                @ tmat[0:2, 0:2].T
            )

            lattice_rot.a1 = lattice_rot.dir_struct_matrix[0, :]
            lattice_rot.a2 = lattice_rot.dir_struct_matrix[1, :]
            '''***Logic sequence to make basis vectors ~right, ~up'''

            lattice_rot.a1_star = (
                np.linalg.inv(lattice_rot.dir_struct_matrix).T
            )[0, :]

            lattice_rot.a2_star = (
                np.linalg.inv(lattice_rot.dir_struct_matrix).T
            )[1, :]

            lattice_rot.at_cols.theta += angle
            lattice_rot.at_cols.theta -= np.trunc(
                lattice_rot.at_cols.theta.to_numpy(dtype=float).astype('float')
                / 90) * 180
            lattice_rot.angle = angle

        return rot_

    def plot_atom_column_positions(
            self,
            filter_by='elem',
            sites_to_plot='all',
            fit_or_ref='fit',
            outlier_disp_cutoff=None,
            plot_masked_image=False,
            xlim=None,
            ylim=None,
            scalebar_len_nm=2,
            color_dict=None,
            legend_dict=None,
            scatter_kwargs_dict={},
            figsize=(13, 10)
    ):
        """Plot fitted or reference atom colum positions.

        Parameters
        ----------
        filter_by : str
            'at_cols' column name to use for filtering to plot only a subset
            of the atom colums.
            Default 'elem'

        sites_to_plot : str ('all') or list of strings
            The criteria for the sites to print, e.g. a list of the elements
            to plot: ['Ba', 'Ti']
            Default 'all'

        fit_or_ref : str ('fit' or 'ref')
            Which poisitions to plot, the
            Default: 'fit'

        outlier_disp_cutoff : None or scalar
            Criteria for removing outlier atomic column fits from the
            plot (in Angstroms). The maximum difference between the fitted
            position and the corresponding reference lattice point. All
            positions with greater errors than this value will be removed.
            If None, all column positions will be plotted.
            Default None.

        plot_masked_image : bool
            Whether to plot the masked image (shows only the regions used
            for fitting). If False, the unmasked image is plotted.
            Default: False

        xlim : None or list-like shape (2,)
            The x axis limits to be plotted as (min, max). If None, the whole
            image is displayed.
            Default: None

        ylim : None or list-like shape (2,)
            The y axis limits to be plotted as (min, max). If None, the whole
            image is displayed.
            Default: None

        scalebar_len_nm : scalar or None
            The desired length of the scalebar in nm. If None, no scalebar is
            plotted.
            Default: 2.

        color_dict : None or dict
            Dict of (atom column site label:color) (key:value) pairs. Colors
            will be used for plotting positions and the legend. If None, a
            standard color scheme is created from the 'RdYlGn' colormap.

        legend_dict : None or dict
            Dict of string names to use for legend labels. Keys must correspond
            to the atom column site labels to be plotted.

        scatter_kwargs_dict : dict
            Dict of additional key word args to be passed to  pyplot.scatter.
            Do not include "c" or "color" as these are specified by the
            color_dict argument. Default kwards specified in the function are:
                s=25, edgecolor='black', linewidths=0.5. These or other
            pyplot.scattter parameters can be modified through this dictionary.
            Default: {}

        figsize : 2-tuple
            The figure size.

        Returns
        -------
        fig : the matplotlib figure object.
        axs : the matplotlib axes object.

        """

        scatter_kwargs_default = {
            's': 5,
            'edgecolor': 'black',
            'linewidths': 0.5,
        }
        scatter_kwargs_default.update(scatter_kwargs_dict)

        if fit_or_ref == 'fit':
            xcol, ycol = 'x_fit', 'y_fit'
        elif fit_or_ref == 'ref':
            xcol, ycol = 'x_ref', 'y_ref'

        if self.pixel_size_cal:
            pixel_size = self.pixel_size_cal
            flag_estimated_pixel_size = False
        else:
            pixel_size = np.mean([latt.pixel_size_est
                                  for latt in self.latt_dict.values()])
            flag_estimated_pixel_size = True

        if (type(outlier_disp_cutoff) == float or
                type(outlier_disp_cutoff) == int):
            outlier_disp_cutoff /= pixel_size * 100

        fig, axs = plt.subplots(
            ncols=1,
            figsize=figsize,
            tight_layout=True,
            sharex=True,
            sharey=True,
        )

        if plot_masked_image is True:
            peak_masks = np.sum(np.array([lattice.peak_masks
                                         for lattice in
                                         self.latt_dict.values()]),
                                axis=0)
            axs.imshow(self.image * peak_masks, cmap='gray')
        else:
            axs.imshow(self.image, cmap='gray')
        axs.set_xticks([])
        axs.set_yticks([])

        if xlim:
            xlim = np.sort(xlim)
            axs.set_xlim(xlim)
        if ylim:
            ylim = np.flip(np.sort(ylim))
            axs.set_ylim(ylim)

        if color_dict is None:
            elems = np.sort(np.unique(np.concatenate(
                [lattice.unitcell_2D.loc[:, filter_by].unique()
                 for lattice in self.latt_dict.values()])))
            if not (sites_to_plot == 'all') or (sites_to_plot[0] == 'all'):
                elems = elems[np.isin(elems, sites_to_plot)]
            num_colors = elems.shape[0]
            if num_colors == 1:
                color_dict = {elems[0]: 'red'}
            else:
                cmap = plt.cm.RdYlGn
                color_dict = {k: cmap(v/(num_colors-1)) for v, k in
                              enumerate(elems)}

        sites_used = []
        for key, lattice in self.latt_dict.items():
            if (sites_to_plot == 'all') or (sites_to_plot[0] == 'all'):
                filtered = lattice.at_cols.copy()
            else:
                filtered = lattice.at_cols[
                    lattice.at_cols.loc[:, filter_by].isin(sites_to_plot)
                ].copy()

            if ((outlier_disp_cutoff is not None) &
                    (outlier_disp_cutoff is not np.inf)):

                filtered = filtered[norm(
                    filtered.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
                    - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
                    axis=1)
                    < outlier_disp_cutoff].copy()

            sub_latts = filtered.loc[:, filter_by].unique()
            sub_latts.sort()

            for site in sub_latts:
                if np.isin(site, sites_used):
                    label = None
                else:
                    if legend_dict is not None:
                        label = legend_dict[site]
                    else:
                        label = site
                sites_used += [site]

                sublattice = filtered[filtered[filter_by] == site].copy()
                axs.scatter(
                    sublattice.loc[:, xcol],
                    sublattice.loc[:, ycol],
                    color=color_dict[site],
                    label=label,
                    **scatter_kwargs_default
                )

            arrow_scale = np.min([norm(lattice.a1), norm(lattice.a2)]) / 20

            axs.arrow(
                lattice.x0,
                lattice.y0,
                lattice.a1[0],
                lattice.a1[1],
                fc='white',
                ec='black',
                width=arrow_scale,
                length_includes_head=True,
                head_width=arrow_scale * 3,
                head_length=arrow_scale * 5
            )
            axs.arrow(
                lattice.x0,
                lattice.y0,
                lattice.a2[0],
                lattice.a2[1],
                fc='white',
                ec='black',
                width=arrow_scale,
                length_includes_head=True,
                head_width=arrow_scale * 3,
                head_length=arrow_scale * 5
            )

        axs.legend(
            loc='lower left',
            bbox_to_anchor=[1.02, 0],
            facecolor='grey',
            fontsize=16,
            markerscale=3,
        )

        if scalebar_len_nm:
            scalebar = ScaleBar(
                pixel_size/10,
                'nm', location='lower right',
                pad=0.4,
                fixed_value=scalebar_len_nm,
                font_properties={'size': 20},
                box_color='lightgrey', width_fraction=0.02,
                sep=2,
                border_pad=2
            )

            axs.add_artist(scalebar)

        if flag_estimated_pixel_size:
            axs.set_title(
                'Warning: scalebar length is based on the pixel size ' +
                'estimated from the reference lattice. The user should ' +
                'specify the calibrated pixel size if known.',
                fontsize=12
            )

        return fig, axs

    def plot_disp_vects(
            self,
            filter_by='elem',
            sites_to_plot='all',
            outlier_disp_cutoff=None,
            xlim=None,
            ylim=None,
            scalebar_len_nm=2,
            arrow_scale_factor=1,
            max_colorwheel_range_pm=None,
            label_dict=None,
            plot_fit_points=False,
            plot_ref_points=False
    ):
        """Plot dislacement vectors between reference and fitted atom colum
            positions.

        Parameters
        ----------
        filter_by : str
            'at_cols' column to use for filtering to plot only a subset
            of the atom colums.
            Default 'elem'

        sites_to_plot : str ('all') or list of strings
            The criteria for the sites to print, e.g. a list of the elements
            to plot: ['Ba', 'Ti']
            Default 'all'

        outlier_disp_cutoff : None or scalar
            Criteria for removing outlier atomic column fits from the
            plot (in Angstroms). The maximum difference between the fitted
            position and the corresponding reference lattice point. All
            positions with greater errors than this value will be removed.
            If None, all column positions will be plotted.
            Default None.

        xlim : None or list-like shape (2,)
            The x axis limits to be plotted. If None, the whole image is
            displayed.
            Default: None

        ylim : None or list-like shape (2,)
            The y axis limits to be plotted. If None, the whole image is
            displayed.
            Default: None

        scalebar_len_nm : scalar or None
            The desired length of the scalebar in nm. If None, no scalebar is
            plotted.
            Default: 2.

        arrow_scale_factor : scalar
            Relative scaling factor for the displayed displacement vectors.
            Default: 1.

        max_colorwheel_range_pm : scalar or None
            The maximum absolute value of displacement included in the
            colorwheel range. Displacement vectors longer than this value will
            have the same color intensity. Specified in picometers.

        label_dict : None or a dict
            Keys must be lattice site types in the 'filter_by' column of the
            at_cols datarame(s). Values should be a string to replace this
            with when labeling the plots.
        plot_fit_points : bool
            Whether to indicate the fitted positions with colored dots.
            Default: False

        plot_ref_points : bool
            Whether to indicate the reference positions with colored dots.
            Default: False

        Returns
        -------
        fig : the matplotlib figure object.
        axs : the list of matplotlib axes objects: each sublattice plot and
            the legend.

        """

        if self.pixel_size_cal:
            pixel_size = self.pixel_size_cal
            flag_estimated_pixel_size = False
        else:
            pixel_size = np.mean(
                [latt.pixel_size_est for latt in self.latt_dict.values()]
            )
            flag_estimated_pixel_size = True

        if outlier_disp_cutoff is None:
            outlier_disp_cutoff = np.inf
        else:
            outlier_disp_cutoff /= pixel_size * 100

        # Get combined data an list of unique sublattices
        combined = pd.concat(
            [latt.at_cols for latt in self.latt_dict.values()],
            ignore_index=True
        )

        sublatt_list = combined.loc[:, filter_by].unique().tolist()

        # Remove atom columns with large displacements
        if ((outlier_disp_cutoff is not None) &
                (outlier_disp_cutoff is not np.inf)):

            combined = combined[norm(
                combined.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
                - combined.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
                axis=1)
                < outlier_disp_cutoff].copy()

        # Find the max colorwheel range if not specified
        if max_colorwheel_range_pm is None:
            dxy = (combined.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
                   - combined.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float))
            mags = norm(dxy, axis=1) * pixel_size * 100
            avg = np.mean(mags)
            std = np.std(mags)
            max_colorwheel_range_pm = int(np.ceil((avg + 3*std)/5) * 5)

        # Set up Figure and gridspec
        if sites_to_plot == 'all':
            sites_to_plot = sublatt_list
            sublatt_list.sort()
        elif type(sites_to_plot) != list:
            raise Exception('"sites_to_plot" must be either "all" or a list')
        else:
            sites_found = np.isin(sites_to_plot, sublatt_list)
            sites_not_found = np.array(sites_to_plot)[~sites_found]
            sites_to_plot = np.array(sites_to_plot)[sites_found]
            if len(sites_not_found) >= 1:
                print('!!! Note: Specified sites not found for plotting: \n',
                      f'{sites_not_found} \n')

        n_plots = len(sites_to_plot)

        if n_plots > 12:
            raise Exception('The number of plots exceeds the limit of 12.')

        if n_plots <= 3:
            nrows = 1
            ncols = n_plots

        elif n_plots <= 8:
            nrows = 2
            ncols = np.ceil(n_plots/2).astype(int)

        elif n_plots <= 12:
            nrows = 3
            ncols = np.ceil(n_plots/3).astype(int)

        figsize = (ncols * 5 + 3, 5 * nrows + 3)

        fig = plt.figure(figsize=figsize)

        width_ratios = [1] * ncols

        if n_plots % ncols == 0:
            ncols_ = ncols + 1
            width_ratios += [0.4]
        else:
            ncols_ = ncols

        # print(nrows, ncols_)
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=ncols_,
            width_ratios=width_ratios,
            height_ratios=[1 for _ in range(nrows)],
            wspace=0.05
        )

        axs = []
        for ax, site in enumerate(sites_to_plot):
            if label_dict is not None:
                if np.isin(site, list(label_dict.keys())):
                    label = label_dict[site]
            else:
                label = site

            row = ax // ncols
            col = ax % ncols
            if ax > 0:
                axs += [fig.add_subplot(
                    gs[row, col],
                    sharex=axs[0],
                    sharey=axs[0]
                )]
            else:
                axs += [fig.add_subplot(gs[row, col])]

            axs[ax].imshow(self.image, cmap='gray')

            if xlim:
                xlim = np.sort(xlim)
                axs[ax].set_xlim(xlim)
            else:
                xlim = [0, self.w]

            if ylim:
                ylim = np.flip(np.sort(ylim))
                axs[ax].set_ylim(ylim)
            else:
                ylim = [self.h, 0]

            h = ylim[0] - ylim[1]

            axs[ax].set_xticks([])
            axs[ax].set_yticks([])

            if ax == 0 and scalebar_len_nm is not None:
                scalebar = ScaleBar(
                    pixel_size/10,
                    'nm', location='lower right',
                    pad=0.4,
                    fixed_value=scalebar_len_nm,
                    font_properties={'size': 10},
                    box_color='lightgrey',
                    width_fraction=0.02,
                    sep=2
                )
                axs[ax].add_artist(scalebar)

            sub_latt = combined[combined.loc[:, filter_by] == site]

            axs[ax].text(
                xlim[0] + 0.02*h,
                ylim[1] - 0.02*h,
                label,
                color='black',
                size=24,
                weight='bold',
                va='bottom',
                ha='left'
            )

            # title.set_path_effects([path_effects.Stroke(linewidth=3,
            #                                             foreground='black'),
            #                         path_effects.Normal()])

            hsv = np.ones((sub_latt.shape[0], 3))
            dxy = (sub_latt.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
                   - sub_latt.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float))

            disp_pm = (norm(dxy, axis=1) * pixel_size * 100)
            normed = disp_pm / max_colorwheel_range_pm
            print(rf'Displacement statistics for {site}:',
                  f'average: {np.mean(disp_pm)  :.{2}f} (pm)',
                  f'standard deviation: {np.std(disp_pm)  :.{2}f} (pm)',
                  f'maximum: {np.max(disp_pm)  :.{2}f} (pm)',
                  f'minimum: {np.min(disp_pm)  :.{2}f} (pm)',
                  '\n',
                  sep='\n')
            hsv[:, 2] = np.where(normed > 1, 1, normed)
            hsv[:, 0] = (np.arctan2(dxy[:, 0], dxy[:, 1])
                         + np.pi/2)/(2*np.pi) % 1
            rgb = colors.hsv_to_rgb(hsv)

            if plot_fit_points:
                axs[ax].scatter(
                    sub_latt.loc[:, 'x_fit'],
                    sub_latt.loc[:, 'y_fit'],
                    color='blue',
                    s=1)
            if plot_ref_points:
                axs[ax].scatter(
                    sub_latt.loc[:, 'x_ref'],
                    sub_latt.loc[:, 'y_ref'],
                    color='red',
                    s=1)

            axs[ax].quiver(
                sub_latt.loc[:, 'x_fit'],
                sub_latt.loc[:, 'y_fit'],
                dxy[:, 0],
                dxy[:, 1],
                color=rgb,
                angles='xy',
                scale_units='xy',
                scale=0.1/arrow_scale_factor,
                headlength=10,
                headwidth=5,
                headaxislength=10,
                edgecolor='white',
                linewidths=0.5,
                width=0.003,
            )

        def colour_wheel(samples=1024, clip_circle=True):
            xx, yy = np.meshgrid(
                np.linspace(-1, 1, samples),
                np.linspace(-1, 1, samples)
            )

            v = np.sqrt(xx ** 2 + yy ** 2)
            if clip_circle:
                v[v > 1] = 1
            v[(v > 0.98) & (v < 1)] = 0
            h = ((np.arctan2(xx, yy) + np.pi/2) / (np.pi * 2)) % 1
            hsv = np.ones((samples, samples, 3))
            hsv[:, :, 0] = h
            hsv[:, :, 1][v == 1] = 0
            hsv[:, :, 2] = v

            rgb = colors.hsv_to_rgb(hsv)

            alpha = np.expand_dims(np.where(v == 0, 0, 1), 2)
            hsv = np.concatenate((hsv, alpha), axis=2)

            return rgb

        rgb = colour_wheel()

        if ncols == ncols_:
            gs_legend = gs[-1, -1].subgridspec(5, 5)
            legend = fig.add_subplot(gs_legend[2, 2])
        else:
            gs_legend = gs[-1, -1].subgridspec(3, 3)
            legend = fig.add_subplot(gs_legend[1, 1])
        legend.text(
            0.5,
            -1.2,
            f'Displacement\n(0 - {max_colorwheel_range_pm} pm)',
            transform=legend.transAxes,
            horizontalalignment='center',
            fontsize=12,
            fontweight='bold'
        )

        legend.imshow(rgb)
        legend.set_xticks([])
        legend.set_yticks([])
        legend.axis('off')
        legend.axis('image')

        fig.subplots_adjust(
            hspace=0,
            wspace=0,
            top=0.9,
            bottom=0.01,
            left=0.01,
            right=0.99
        )

        if flag_estimated_pixel_size:
            fig.suptitle(
                'Warning: scalebar length and vector magnitudes are based\n' +
                'on the pixel size estimated from the reference lattice.\n' +
                'The user should specify the calibrated pixel size, '
                'if known, when defining the HRImage object.',
                fontsize=12,
                c='red'
            )

        return fig, axs


class AtomicColumnLattice:
    """Object class for quantification of atomic columns in HR STEM images.

    Class with methods for locating atomic columns in a HR STEM image using
    a reference lattice genenerated from a .cif file.
        -Requires minimal parameter adjustmets by the user to achieve accurate
    results.
        -Provides a fast fitting algorithm with automatic parallel processing.
        -Automatically groups close atomic columns for simultaneous fitting.

    Parameters
    ----------
    image : 2D array
        The HR STEM image array to analize.

    unitcell : UnitCell class object
        An appropriate UnitCell object of the SingleOrigin module with
        project_uc_2d() method applied. See examples for how to generate this.

    origin_atom_column : int
        The DataFrame row index (in unitcell.at_cols) of the atom column
        that is later picked by the user to register the reference lattice.
        If None, the closest atom column to the unit cell origin is
        automatically chosen.
        Default: None.

    Attributes
    ----------
    basis_offset_frac : The input basis_offset_frac, fractional coordinates

    basis_offset_pix : The basis offset in image pixel coordinates.

    at_cols : DataFrame containing the he reference lattice and fitting data
        (including positions) for the atomic columns in the image.

    at_cols_uncropped : The reference lattice atom columns before removing
        positions close to the image edges (as defined by the "buffer" arg in
        the "fit_atom_columns()" method).

    unitcell_2D : DataFrame containing the projected crystallographic unit
        cell atom column positions.

    a_2d : The direct structure matrix of the projected structure
        from  the .cif in the Cartesian reference frame with units of
        Angstroms. Premultiplication of position vectors in fractional
        coordinates (u, v) by this matrix converts them into realspace
        coordinates. Additionally, its columns are basis vectors in Cartesian
        Angstrom coordinates: row 0 is the x-component; row 1 is the
        y-component.

    x0, y0 : The image coordinates of the origin of the reference lattice
        (in pixels).

    peak_masks : The last set of masks used for fitting  atom columns. Has
        the same shape as image.

    roi_mask : ndarray or None
        Binary array with the same shape as image. 1 where lattice is present.
        0 otherwise. Used to restrict the image area where a reference lattice
        will be generated. If none, reference lattice will extend over the
        entire image.

    a1_star, a2_star : The reciprocal basis vectors in FFT pixel coordinates.

    a1, a2 : The real space basis vectors in image pixel coordinates.

    dir_struct_matrix : The direct structure matrix (i.e. transformation
        matrix from fractional to image pixel coordinates.)

    pixel_size_est : The estimated pixel size using the reference lattice basis
        vectors and lattice parameter values from the .cif file.

    Methods
    -------
    fft_get_basis_vect(
        self,
        a1_order=1,
        a2_order=1,
        sigma=5
        ):
        Find basis vectors for the image from the FFT.

    get_roi_mask_std(
        self,
        r=4,
        sigma=8,
        thresh=0.5,
        fill_holes=True,
        buffer=10,
        show_mask=True
        ):
        Get mask for specific region of image based on local standard
        deviation.

    select_origin():
        Select origin for the reference lattice. Used by
        define_reference_lattice() method.

    define_reference_lattice(
        self,
        LoG_sigma=None
        ):
        Registers a reference lattice to the image.

    fit_atom_columns(
        self,
        buffer=0,
        local_thresh_factor=1,
        peak_sharpening_filter='auto',
        peak_grouping_filter='auto',
        filter_by='elem',
        sites_to_fit='all',
        watershed_line=True,
        use_LoG_fitting=False
        ):
        Algorithm for fitting 2D Gaussians to HR STEM image.

    refine_reference_lattice(
        self,
        filter_by='elem',
        sites_to_use='all',
        outlier_disp_cutoff=None
        ):
        Refines the reference lattice on fitted column positions.

    get_fitting_residuals(self):
        Plots image residuals from the atomic column fitting.

    """

    def __init__(
            self,
            image,
            unitcell,
            # probe_fwhm=0.8,
            origin_atom_column=None,
            pixel_size_cal=None,
    ):

        self.image = image
        # self.probe_fwhm = probe_fwhm
        (self.h, self.w) = image.shape
        self.unitcell_2D = unitcell.at_cols
        self.a_2d = unitcell.a_2d
        self.at_cols = pd.DataFrame()
        self.at_cols_uncropped = pd.DataFrame()
        self.x0, self.y0 = np.nan, np.nan
        self.peak_masks = np.zeros(image.shape)
        self.roi_mask = np.ones(image.shape)
        self.a1, self.a2 = None, None
        self.a1_star, self.a2_star = None, None
        self.dir_struct_matrix = None
        self.sigma = None
        self.pixel_size_est = None
        self.pixel_size_cal = None
        self.residuals = None

        if origin_atom_column is None:
            origin_atom_column = np.argmin(norm(
                self.unitcell_2D.loc[:, 'x':'y'].to_numpy(dtype=float),
                axis=1))
        self.origin_atom_column = origin_atom_column
        self.basis_offset_frac = self.unitcell_2D.loc[
            origin_atom_column, 'u':'v'].to_numpy(dtype=float)
        self.basis_offset_pix = self.unitcell_2D.loc[
            origin_atom_column, 'x':'y'].to_numpy(dtype=float)
        self.use_LoG_fitting = False

    def get_est_pixel_size(self):
        if (self.a1 is None) or (self.a2 is None):
            raise Exception(
                "self.a1 and self.a2 must be defined to estimate pixel size."
                + "Get basis vectors first.")
        pixel_size_est = np.average(
            [norm(self.a_2d[:, 0]) / norm(self.a1),
             norm(self.a_2d[:, 1]) / norm(self.a2)]
        )
        return pixel_size_est

    def fft_get_basis_vect(
            self,
            a1_order=1,
            a2_order=1,
            sigma=5,
            thresh_factor=1,
            spot_numbers=None
    ):
        """Measure crystal basis vectors from the image FFT.

        Displays FFT peaks for graphical picking or accepts indcies of
        desired peaks. Following selection of two peaks representing the
        reciprocal basis vectors, finds peaks related by approximate vector
        additions, refines a reciprocal basis from these positions and
        transforms the reciprocal basis into a real space basis.

        Parameters
        ----------
        a1_order, a2_order : ints
            Order of first peaks visible in the FFT along the two reciprocal
            lattice basis vector directions. If some FFT peaks are weak or
            absent (such as forbidden reflections), specify the order of the
            first peak that is clearly visible.

        sigma : scalar
            The Laplacian of Gaussian sigma value to use for sharpening of the
            FFT peaks. Usually a value between 2 and 10 will work well.

        thresh_factor : scalar
            Relative adjustment of the threshold level for detecting peaks.
            Greater values will detect fewer peaks; smaller values, more.
            Default: 1.

        spot_numbers : 2-tuple of ints or None
            Labeled spot numbers of the FFT peaks found using fft_get_peaks()
            method. Must be in the order (a1_star, a2_star). This method allows
            selection of FFT peaks without requiring graphical picking. This is
            useful for Jupyter notebooks when it is preferred to plot in the
            notebook rather than in a seperate window or if processing multiple
            similar images where the FFT peaks will be identically labeled.
            If None, graphical picking is chosen.
            Default: None.

        Returns
        -------
        None.

        """

        '''Find rough reciprocal lattice'''

        if spot_numbers is None:
            basis_picks_xy, m, U = self.fft_get_peaks(
                sigma=sigma,
                thresh_factor=thresh_factor,
                graphical_picking=True
            )
            xy = self.recip_latt.loc[:, 'x':'y'].to_numpy()

        elif (len(spot_numbers) == 2) & (type(spot_numbers) == tuple):
            xy = self.recip_latt.loc[:, 'x':'y'].to_numpy()
            basis_picks_xy = xy[spot_numbers, :]
            print(basis_picks_xy)
            U = int(self.fft.shape[0] / 2)
            m = (min(self.h, self.w) // 2) * 2

        origin = np.array([U, U])

        '''Generate reference lattice and find corresponding peak regions'''
        a1_star = (basis_picks_xy[0, :] - origin) / a1_order
        a2_star = (basis_picks_xy[1, :] - origin) / a2_order

        a_star = np.array([a1_star, a2_star])

        recip_latt_indices = np.array(
            [[i, j] for i in range(-5, 6) for j in range(-5, 6)]
        )
        xy_ref = recip_latt_indices @ a_star + origin

        vects = np.array([xy - xy_ for xy_ in xy_ref])
        inds = np.argmin(norm(vects, axis=2), axis=1)

        df = {
            'h': recip_latt_indices[:, 0],
            'k': recip_latt_indices[:, 1],
            'x_ref': xy_ref[:, 0],
            'y_ref': xy_ref[:, 1],
            'x_fit': [xy[ind, 0] for ind in inds],
            'y_fit': [xy[ind, 1] for ind in inds],
            'mask_ind': inds
        }

        recip_latt = pd.DataFrame(df)

        recip_latt = recip_latt[norm(
            recip_latt.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
            - recip_latt.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
            axis=1
        ) < 0.1*np.max(norm(a_star, axis=1))
        ].reset_index(drop=True)

        M_star = recip_latt.loc[:, 'h':'k'].to_numpy(dtype=float)
        xy = recip_latt.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)

        p0 = np.concatenate((a_star.flatten(), origin))

        params = fit_lattice(p0, xy, M_star, fix_origin=True)

        a1_star = params[:2]
        a2_star = params[2:4]
        a_star = np.array([a1_star, a2_star])

        dir_struct_matrix = np.linalg.inv(a_star.T) * m

        recip_latt[['x_ref', 'y_ref']] = (
            recip_latt.loc[:, 'h':'k'].to_numpy(dtype=float) @ a_star + origin
        )
        self.recip_latt = recip_latt
        plt.close('all')

        recip_vects = norm(xy - origin, axis=1)
        min_recip_vect = np.min(recip_vects[recip_vects > 0])
        window = min(min_recip_vect*10, U)

        fig2, ax = plt.subplots(figsize=(10, 10))
        ax.imshow((self.fft)**(0.1), cmap='gray')
        ax.scatter(
            recip_latt.loc[:, 'x_fit'].to_numpy(dtype=float),
            recip_latt.loc[:, 'y_fit'].to_numpy(dtype=float)
        )

        ax.arrow(
            origin[0],
            origin[1],
            a1_star[0],
            a1_star[1],
            fc='red',
            ec='red',
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
            ec='green',
            width=0.1,
            length_includes_head=True,
            head_width=2,
            head_length=3
        )

        ax.set_ylim(bottom=U+window, top=U-window)
        ax.set_xlim(left=U-window, right=U+window)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Reciprocal Lattice Fit')

        self.a1_star = a1_star
        self.a2_star = a2_star
        self.a1 = dir_struct_matrix[0, :]
        self.a2 = dir_struct_matrix[1, :]
        self.dir_struct_matrix = dir_struct_matrix
        self.basis_offset_pix = self.basis_offset_frac @ self.dir_struct_matrix
        self.pixel_size_est = self.get_est_pixel_size()
        self.recip_latt = recip_latt

    def fft_get_peaks(
            self,
            sigma=5,
            thresh_factor=1,
            graphical_picking=False
    ):
        """Find peaks in the image FFT.

        Finds peaks in the image FFT, plots and numbers peaks for selection
        with fft_get_basis_vect().

        Parameters
        ----------
        sigma : scalar
            The Laplacian of Gaussian sigma value to use for sharpening of the
            FFT peaks. Usually a value between 2 and 10 will work well.

        thresh_factor : scalar
            Relative adjustment of the threshold level for detecting peaks.
            Greater values will detect fewer peaks; smaller values, more.
            Default: 1.

        graphical_picking : bool
            Whether to use graphical picking of FFT peaks. If False, FFT is
            plotted with numbered peaks. The desired peak numbers can be
            passed to fft_get_basis_vect() instead of graphical picking.
            Default: False.

        Returns
        -------
        None unless graphical_picking=True, then:

        basis_picks_xy : 2 x 2 list
            The FFT coordinates of the chosen peaks.

        m : int
            Size of the real space image before FFT is taken (same value for
            x and y because image cropped to square before taking FFT).
            Needed to find correct recprocal pixel size.

        U : int
            Center of the shifted FFT in pixels (same value for x and y
            because FFT is forced to be square).

        """

        self.fft = image_norm(fft_square(copy.deepcopy(self.image),
                                         hann_window=True))
        m = (min(self.h, self.w) // 2) * 2
        U = int(m/2)
        if m > 1024:
            self.fft = self.fft[U-512:U+512, U-512:U+512]
            U = 512
        origin = np.array([U, U])

        fft_der = image_norm(-gaussian_laplace(self.fft, sigma))

        # Downsample for speed
        if np.max([self.h, self.w]) > 2000:
            fft_der_ds = downscale_local_mean(fft_der, (2, 2))
        else:
            fft_der_ds = fft_der
        masks, num_masks, _, spots = watershed_segment(
            fft_der_ds,
            local_thresh_factor=0,
            buffer=2*sigma
        )

        # Resample
        if np.max([self.h, self.w]) > 2000:
            masks = rescale(masks, (2, 2), order=0)
            spots.loc[:, 'x':'y'] *= 2

        spots['stdev'] = [
            np.std(self.fft[int(y-sigma):int(y+sigma+1),
                            int(x-sigma):int(x+sigma+1)])
            for [x, y]
            in np.around(spots.loc[:, 'x':'y']).to_numpy(dtype=int)
        ]

        thresh = 0.003 * thresh_factor
        spots_ = spots[(spots.loc[:, 'stdev'] > thresh)].reset_index(drop=True)
        xy = spots_.loc[:, 'x':'y'].to_numpy(dtype=float)

        recip_vects = norm(xy - origin, axis=1)
        min_recip_vect = np.min(recip_vects[recip_vects > 0])
        window = min(min_recip_vect*5, U)

        self.recip_latt = spots_

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(bottom=U+window, top=U-window)
        ax.set_xlim(left=U-window, right=U+window)
        ax.imshow((self.fft)**(0.1), cmap='gray')
        ax.scatter(xy[:, 0], xy[:, 1], c='red', s=8)
        ax.scatter(origin[0], origin[1], c='white', s=16)
        ax.set_xticks([])
        ax.set_yticks([])

        if graphical_picking:
            basis_picks_xy = np.array(plt.ginput(2, timeout=15))

            vects = np.array([xy - i for i in basis_picks_xy])
            inds = np.argmin(norm(vects, axis=2), axis=1)
            basis_picks_xy = xy[inds, :]

            return basis_picks_xy, m, U

        else:
            inds = spots_.index.tolist()
            for i, label in enumerate(inds):
                plt.annotate(label, (xy[i, 0], xy[i, 1]))

    def specify_basis_vectors(self, a1, a2):
        """Specify basis vectors manually.

        Useful for images that are too small for basis vector estimation from
        the FFT (such as small simulated images) or if the basis vectors are
        known. Input basis vectors do not need to be exact. They will be
        refined during the subsequent fitting processes.

        Parameters
        ----------
        a1, a2 : array-like of shape (2,)
            The basis vectors, [x,y] in pixels. Follows the matpoltlib
            convention of positive x and y being in the 'right' and 'down'
            directions, respectively.

        Returns
        -------
        None.

        """
        self.dir_struct_matrix = np.array([a1, a2])
        self.a1 = np.array(a1)
        self.a2 = np.array(a2)
        self.basis_offset_pix = self.basis_offset_frac @ self.dir_struct_matrix
        self.pixel_size_est = self.get_est_pixel_size()

    def get_roi_mask_std(
        self,
        r=4,
        thresh=0.5,
        fill_holes=True,
        buffer=10,
        show_mask=True
    ):
        """Get mask for specific region of image based on local standard
        deviation.

        Create a mask for a desired image region where the local standard
        deviation is above a threshold. The mask is saved in the
        AtomColumnLattice object and used during reference lattice generation
        to limnit the extent of the lattice. Useful if vacuum, off-axis grains
        or padded edges are in the image frame.

        Parameters
        ----------
        r : int
            Kernel radius. STD is calculated in a square kernel of size
            2*r + 1.
            Default: 4.

        thresh : scalar
            Thresholding level for binarizing the result into a mask.
            Default: 0.5.

        fill_holes : bool
            If true, interior holes in the mask are filled.
            Default: True

        buffer : int
            Number of pixels to erosion from the edges of the mask. Prevents
            retention of reference lattice points that are outside the actual
            lattice region.
            Default: 10.

        show_mask : bool
            Whether to plot the mask for verification.
            Default: True

        Returns
        -------
        None.

        """

        if self.sigma is None:
            self.sigma = get_feature_size(self.image)
            sigma = self.sigma

        image_std = image_norm(gaussian_filter(
            std_local(self.image, r),
            sigma=sigma)
        )
        new_mask = np.where(image_std > thresh, 1, 0)
        if fill_holes:
            new_mask = binary_fill_holes(new_mask)
        if buffer:
            new_mask = erosion(
                new_mask,
                footprint=np.ones((3, 3))
            )

        self.roi_mask *= new_mask

        if show_mask:
            self.show_roi_mask()

    def get_roi_mask_polygon(
            self,
            vertices=None,
            buffer=0,
            invert=False,
            show_mask=True,
            return_vertices=False
    ):
        """Create mask for a polygon-shaped ROI in an image.

        Create a mask for a desired image ROI from polygon vertices. The
        mask is saved in the AtomColumnLattice object and used during
        reference lattice generation to limnit the extent of the lattice.
        Useful for images with interfaces, multiple grains, etc.

        Parameters
        ----------
        vertices : ndarray or None
            Array of shape (n,2). Each row is the [x,y] coordinate for the
            respective vertice. Vertices must be in clockwise order. If None,
            polygon will be created by graphical picking.
            Default: None.

        buffer : int
            Number of pixels to erosion from the edges of the mask. Prevents
            retention of reference lattice points that are outside the actual
            lattice region.
            Default: 0.

        invert : bool
            Whether to invert the mask. If False, the area inside the polygon
            is the selected area. If True, the area outside the polygon is
            selected instead.
            Default: False

        show_poly : bool
            Whether to plot the polygon overlaid on the image for confirmation.
            Default: True

        return_vertices : bool
            Whether to return the list of vertices. Useful if the same set of
            manually selected vertices will be applied to a second lattice or
            image.
            Default: False

        Returns
        -------
        vertices or None.

        """

        if vertices is not None:
            vertices = np.fliplr(np.array(vertices))
        if vertices is None:
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.imshow(self.image, cmap='gist_gray')
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(
                'Left click to add vertices.\n' +
                'Right click or backspace key to remove last.\n' +
                'Center click or enter key to complete.\n' +
                'Must select in clockwise order.'
            )

            vertices = plt.ginput(
                n=-1,
                timeout=0,
                show_clicks=True,
                mouse_add=MouseButton.LEFT,
                mouse_pop=MouseButton.RIGHT,
                mouse_stop=MouseButton.MIDDLE
            )

            plt.close()
            vertices = np.fliplr(np.array(vertices))

        new_mask = polygon2mask(self.image.shape, vertices)
        if buffer:
            new_mask = erosion(
                new_mask,
                footprint=np.ones((3, 3))
            )

        if invert:
            new_mask = np.where(new_mask == 1, 0, 1)

        self.roi_mask *= new_mask

        if show_mask:
            self.show_roi_mask()
        if return_vertices:
            return vertices

    def show_roi_mask(self):
        """Show the current region mask

        """

        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gist_gray')

        ax.imshow(
            self.roi_mask,
            alpha=0.2,
            cmap='Reds'
        )

        ax.set_title('Region of interest:', fontsize=16)

    def select_origin(
            self,
            window_size=200,
            window_center=None,
            interactive_selection=False
    ):
        """Select origin for the reference lattice.

        User chooses appropriate atomic column to establish the reference
        lattice origin. Used by the define_reference_lattice() method.

        Parameters
        ----------
        window_size : int
            Crop image to h/zoom_factor x w/zoom_factor for selection of
            an atom column to use for locating the reference lattice origin.
            Default: 200.

        interactive_selection : bool
            Whether to enable interactive selection of an atom column by
            graphical clicking. If False, generates the plot but does not wait
            for mouse clicks.

        Returns
        -------
        None.

        """

        if window_center is None:
            window_center = [int(self.w/2), int(self.h/2)]

        half_window = int(window_size/2)
        bottom = window_center[0] + half_window
        top = window_center[0] - half_window
        left = window_center[1] - half_window
        right = window_center[1] + half_window

        if 'LatticeSite' in list(self.unitcell_2D.columns):
            lab = 'LatticeSite'
        else:
            lab = 'elem'

        fig, ax = plt.subplots(figsize=(10, 10))
        message = ('Pick an atom column of the reference atom column type'
                   + ' (white outlined position)')
        ax.set_title(str(message), fontdict={'color': 'red'}, wrap=True)
        ax.set_ylim(bottom=bottom, top=top)
        ax.set_xlim(left=left, right=right)
        ax.imshow(self.image, cmap='gray')

        self.unitcell_2D['x_ref'] = ''
        self.unitcell_2D['y_ref'] = ''
        self.unitcell_2D[['x_ref', 'y_ref']] = (
            self.unitcell_2D.loc[:, 'u':'v'].to_numpy(dtype=float)
            @ self.dir_struct_matrix
        )

        x_coords = list([
            *self.dir_struct_matrix[:, 0],
            np.sum(self.dir_struct_matrix[:, 0]).item(), 0])

        y_coords = list([
            *self.dir_struct_matrix[:, 1],
            np.sum(self.dir_struct_matrix[:, 1]).item(), 0])

        uc_w = (np.max(x_coords) - np.min(x_coords))
        uc_h = (np.max(y_coords) - np.min(y_coords))

        xy0 = [
            left + uc_w*0.2 - np.min(x_coords),
            bottom - np.max(y_coords) - uc_h*0.2
        ]

        box_params = [
            left + 0.1*uc_w,
            bottom - 0.1*uc_h,
            uc_w*1.2,
            -uc_h*1.2
        ]

        box = Rectangle(
            (box_params[0],
             box_params[1]),
            box_params[2],
            box_params[3],
            edgecolor='black',
            facecolor='lightslategrey',
            alpha=1
        )

        ax.add_patch(box)

        cell_verts = np.array(
            [xy0,
             xy0 + self.a1,
             xy0 + self.a1 + self.a2,
             xy0 + self.a2]
        )

        cell = Polygon(cell_verts, fill=False, ec='black', zorder=1)
        ax.add_patch(cell)

        site_list = list(set(self.unitcell_2D[lab]))
        site_list.sort()

        color_code = {
            k: v for v, k in
            enumerate(np.sort(self.unitcell_2D.loc[:, lab].unique()))
        }

        color_list = [
            color_code[site] for site in self.unitcell_2D.loc[:, lab]
        ]

        ax.scatter(
            self.unitcell_2D.loc[:, 'x_ref'].to_numpy() + xy0[0],
            self.unitcell_2D.loc[:, 'y_ref'].to_numpy() + xy0[1],
            c=color_list,
            cmap='RdYlGn',
            s=10,
            zorder=10
        )

        ax.arrow(
            xy0[0],
            xy0[1],
            self.a1[0],
            self.a1[1],
            fc='black',
            ec='black',
            width=0.1,
            length_includes_head=True,
            head_width=2,
            head_length=3,
            zorder=8
        )
        ax.arrow(
            xy0[0],
            xy0[1],
            self.a2[0],
            self.a2[1],
            fc='black',
            ec='black',
            width=0.1,
            length_includes_head=True,
            head_width=2,
            head_length=3,
            zorder=8
        )

        ref_atom = self.basis_offset_pix + xy0
        ax.scatter(
            ref_atom[0],
            ref_atom[1],
            c='white',
            ec='black',
            s=70,
            zorder=9
        )

        cmap = plt.cm.RdYlGn
        color_index = [
            Circle((30, 7), 3, color=cmap(c)) for c in
            np.linspace(0, 1, num=len(color_code))
        ]

        def make_legend_circle(
                legend,
                orig_handle,
                xdescent,
                ydescent,
                width,
                height,
                fontsize
        ):
            p = orig_handle
            return p

        ax.legend(
            handles=color_index,
            labels=list(color_code.keys()),
            handler_map={
                Circle: HandlerPatch(patch_func=make_legend_circle),
            },
            fontsize=20,
            loc='lower left',
            bbox_to_anchor=[1.02, 0],
            facecolor='grey'
        )

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        if interactive_selection:
            pt = plt.ginput(1, timeout=30)
            plt.close('all')
            ax.scatter(pt[0][0], pt[0][1], marker='+', c='red')
            return pt[0]

    def get_min_atom_col_dist(self):
        """Find the minimum expected distance between atom columns in the image
        based on the projected structure parameters and pixel size.

        Parameters
        ----------
        None.

        Returns
        -------
        min_dist : scalar
            The minimum image distance between atom columns in the porojected
            structure.

        """
        unit_cell_uv = self.unitcell_2D.loc[:, 'u':'v'].to_numpy(dtype=float)

        # Expand lattice by +/- one unit cell and transform to pixel units:
        unit_cell_xy = np.concatenate(
            ([unit_cell_uv + [i, j]
              for i in range(-1, 2)
              for j in range(-1, 2)])
        ) @ self.dir_struct_matrix

        dists = norm(np.array(
            [unit_cell_xy - pos for pos in unit_cell_xy]),
            axis=2
        )
        min_dist = (np.amin(dists, initial=np.inf, where=dists > 0) - 1)

        return min_dist

    def define_reference_lattice(
            self,
            LoG_sigma=None,
            zoom_factor=1,
            origin=None,
            mask=None,
            plot_ref_lattice=False,
            buffer=None,
    ):
        """Register reference lattice to image.

        User chooses appropriate atomic column to establish the reference
        lattice origin. Rough scaling and orientation are initially defined
        as derived from fft_get_basis_vect() method and then pre-refined by
        local peak detection.

        Parameters
        ----------
        LoG_sigma : scalar
            The Laplacian of Gaussian sigma value to use for peak sharpening.
            If None, calculated using get_feature_size().
            Default None.

        zoom_factor : scalar
            Factor used to determine the plotting window for graphical picking
            of the approximate position of the origin atom column. By default
            the longer lattice basis vector is multiplied by 10 to determine
            the window size. The zoom factor makes this field of view smaller
            if larger than 1 and larger if less than 1.
            Default: 1.

        origin : 2-tuple of scalars
            The approximate position of the origin atom column, previously
            determined. If origin is given, graphical picking will not be
            prompted
            Default: None.

        mask : ndarray of bool or None
             Array of the same shape as self.image. True (or 1) where image is
             of the desired lattice, False (or 0) otherwise. If None, no mask
             is used.
             Default: None.
         plot_ref_lattice : bool
             Whether to plot the reference lattice after registration. For
             verification of proper alignment. This can help avoid wasted time
             running the fitting step if alignment was not accurate.
         buffer : scalar or None
             Apply an ROI mask to restrict the reference lattice to stop this
             distance from the image edges. For more complex ROI masks, use
             the various ROI mask functions directly. If None, reference
             lattice is only restricted if an ROI mask already exists. Without
             an ROI mask, the reference lattice will only stop at the image
             edges.
             Default: None.

        Returns
        -------
        None.

        """

        self.pixel_size_est = self.get_est_pixel_size()

        if buffer is not None:
            self.get_roi_mask_polygon(
                vertices=np.array([[buffer, buffer],
                                   [self.w-buffer, buffer],
                                   [self.w-buffer, self.h-buffer],
                                   [buffer, self.h-buffer]]))

        if self.sigma is None:
            self.sigma = get_feature_size(self.image)

        if LoG_sigma is None:
            LoG_sigma = self.sigma * 0.75

        if mask is not None:
            self.roi_mask = mask

        if origin is None:
            window_size = np.max([norm(self.a1),
                                  norm(self.a2)]
                                 ) * 10 / zoom_factor
            if self.roi_mask is not None:
                window_center = np.array(
                    center_of_mass(self.roi_mask)
                ).astype(int)
            else:
                window_center = np.array([self.h, self.w]) / 2
            (x0, y0) = self.select_origin(
                window_size=window_size,
                window_center=window_center,
                interactive_selection=True
            )

        elif len(origin) == 2:
            (x0, y0) = origin

        print('pick coordinates:', np.around([x0, y0], decimals=2), '\n')

        img_LoG = image_norm(-gaussian_laplace(self.image, LoG_sigma))
        min_dist = self.get_min_atom_col_dist() / 2
        xy_peaks = np.fliplr(np.argwhere(
            detect_peaks(img_LoG, min_dist=min_dist) > 0
        ))

        [x0, y0] = xy_peaks[np.argmin(norm(xy_peaks - [x0, y0], axis=1))]

        print('detected peak coordinates:', [x0, y0], '\n')

        self.x0 = x0 - self.basis_offset_pix[0]
        self.y0 = y0 - self.basis_offset_pix[1]

        a1 = self.a1
        a2 = self.a2
        # x0 = self.x0
        # y0 = self.y0
        h = self.h
        w = self.w

        print('Creating reference lattice...')

        def vect_angle(a, b):
            theta = np.arccos(a @ b.T/(norm(a) * norm(b)))
            return theta

        # Vectors from origin to image corners
        d = [
            np.array([-x0, -y0]),
            np.array([-x0, h - y0]),
            np.array([w - x0, h - y0]),
            np.array([w - x0, -y0])
        ]

        # Find  +/- basis vectors that are closest angularly to each
        # origin-to-image-corner vector.

        a1p = np.argmin([(vect_angle(a1, d[i])) for i, _ in enumerate(d)])
        a1n = np.argmin([(vect_angle(-a1, d[i])) for i, _ in enumerate(d)])
        a2p = np.argmin([(vect_angle(a2, d[i])) for i, _ in enumerate(d)])
        a2n = np.argmin([(vect_angle(-a2, d[i])) for i, _ in enumerate(d)])

        a1_start = int(norm(d[a1n])**2 / (a1 @ d[a1n].T)) - 1
        a1_stop = int(norm(d[a1p])**2 / (a1 @ d[a1p].T)) + 2
        a2_start = int(norm(d[a2n])**2 / (a2 @ d[a2n].T)) - 1
        a2_stop = int(norm(d[a2p])**2 / (a2 @ d[a2p].T)) + 2

        latt_cells = np.array([
            [i, j]
            for i in range(a1_start, a1_stop)
            for j in range(a2_start, a2_stop)
            for _ in range(self.unitcell_2D.shape[0])
        ])

        at_cols = pd.concat(
            [self.unitcell_2D]
            * int(latt_cells.shape[0] / self.unitcell_2D.shape[0]),
            ignore_index=True
        )

        at_cols[['u', 'v']] += latt_cells

        xy_ref = (
            at_cols.loc[:, 'u':'v'].to_numpy(dtype=float)
            @ self.dir_struct_matrix
            + np.array([self.x0, self.y0])
        )

        at_cols['x_ref'] = xy_ref[:, 0]
        at_cols['y_ref'] = xy_ref[:, 1]

        at_cols = at_cols[(
            (at_cols.x_ref >= 0) &
            (at_cols.x_ref <= w-1) &
            (at_cols.y_ref >= 0) &
            (at_cols.y_ref <= h-1)
        )]

        if self.roi_mask is None:
            self.roi_mask = np.ones((self.h, self.w))

        at_cols = at_cols[self.roi_mask[
            np.around(at_cols.y_ref.to_numpy()).astype(int),
            np.around(at_cols.x_ref.to_numpy()).astype(int)
        ] == 1
        ]

        at_cols.reset_index(drop=True, inplace=True)
        empty = pd.DataFrame(
            index=np.arange(0, at_cols.shape[0]),
            columns=['x_fit', 'y_fit', 'sig_1', 'sig_2',
                     'theta', 'peak_int', 'bkgd_int', 'total_col_int']
        )

        at_cols = pd.concat([at_cols, empty], axis=1)
        self.at_cols = pd.DataFrame(columns=at_cols.columns)

        '''Refine reference lattice on imgLoG peaks'''

        print('Performing rough reference lattice refinement...')

        init_inc = np.max(np.abs(at_cols.loc[:, 'u':'v'].to_numpy()))/10

        if init_inc < 3:
            init_inc = 3

        origin_ind = at_cols[
            (at_cols.u == self.basis_offset_frac[0]) &
            (at_cols.v == self.basis_offset_frac[1])
        ].index[0]

        at_cols_orig_type = at_cols[
            (at_cols.x == at_cols.at[origin_ind, 'x']) &
            (at_cols.y == at_cols.at[origin_ind, 'y'])
        ].copy()

        at_cols_orig_type = at_cols_orig_type[(
            (at_cols_orig_type.x_ref >= 20) &
            (at_cols_orig_type.x_ref <= w-21) &
            (at_cols_orig_type.y_ref >= 20) &
            (at_cols_orig_type.y_ref <= h-21)
        )]

        t = [time.time()]

        for i, mult in enumerate([1, 3, 9]):
            lim = mult * init_inc

            # print(f'Refinement iteration {i+1}')

            # if lim > init_inc:
            at_cols_orig_type[['x_ref', 'y_ref']] = (
                at_cols_orig_type.loc[:, 'u':'v'].to_numpy(dtype=float)
                @ self.dir_struct_matrix
                + np.array([self.x0, self.y0])
            )

            filtered = at_cols_orig_type[
                (np.abs(at_cols_orig_type.u) <= lim) &
                (np.abs(at_cols_orig_type.v) <= lim)
            ].copy()

            M = filtered.loc[:, 'u':'v'].to_numpy(dtype=float)
            xy_ref = filtered.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float)

            inds = np.array(
                [np.argmin(norm(xy_peaks - xy, axis=1)) for xy in xy_ref]
            )

            xy = np.array([xy_peaks[ind] for ind in inds])

            def disp_vect_sum_squares(p0, M, xy):

                dir_struct_matrix = p0[:4].reshape((2, 2))
                origin = p0[4:]

                R = norm(
                    xy - M @ dir_struct_matrix - origin,
                    axis=1
                )

                sum_sq = (R @ R.T).item()
                return sum_sq

            p0 = np.concatenate(
                (self.dir_struct_matrix.flatten(),
                 np.array([self.x0, self.y0]))
            )

            params = minimize(disp_vect_sum_squares, p0, args=(M, xy)).x

            self.a1 = params[:2]
            self.a2 = params[2:4]

            self.dir_struct_matrix = params[:4].reshape((2, 2))

            self.x0 = params[4]
            self.y0 = params[5]

            t += [time.time()]
            # print(f'{int((t[-1]-t[-2]) // 60)} min '
            #       + f'{(t[-1]-t[-2]) % 60 :.{2}f} sec')

        at_cols[['x_ref', 'y_ref']] = (
            at_cols.loc[:, 'u':'v'].to_numpy(dtype=float)
            @ self.dir_struct_matrix
            + np.array([self.x0, self.y0])
        )

        at_cols = at_cols[(
            (at_cols.x_ref >= 0) &
            (at_cols.x_ref <= w-1) &
            (at_cols.y_ref >= 0) &
            (at_cols.y_ref <= h-1)
        )]

        if self.roi_mask is not None:
            at_cols = at_cols[self.roi_mask[
                np.around(at_cols.y_ref.to_numpy()).astype(int),
                np.around(at_cols.x_ref.to_numpy()).astype(int)
            ] == 1]

        self.at_cols_uncropped = copy.deepcopy(at_cols)

        plt.close('all')

        if plot_ref_lattice:
            fig, ax = plt.subplots()
            ax.imshow(self.image, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

            elems = np.sort(np.unique(
                [self.unitcell_2D.loc[:, 'elem'].unique()]
            ))

            num_colors = elems.shape[0]
            if num_colors == 1:
                color_dict = {elems[0]: 'red'}
            else:
                cmap = plt.cm.RdYlGn
                color_dict = {k: cmap(v/(num_colors-1)) for v, k in
                              enumerate(elems)}

            scatter_kwargs_default = {
                's': 5,
                'edgecolor': 'black',
                'linewidths': 0.5,
            }

            for elem in elems:

                sublattice = self.at_cols_uncropped[
                    self.at_cols_uncropped['elem'] == elem
                ].copy()
                ax.scatter(
                    sublattice.loc[:, 'x_ref'],
                    sublattice.loc[:, 'y_ref'],
                    color=color_dict[elem],
                    label=elem,
                    **scatter_kwargs_default
                )

            arrow_scale = np.min([norm(self.a1), norm(self.a2)]) / 20

            ax.arrow(
                self.x0,
                self.y0,
                self.a1[0],
                self.a1[1],
                fc='white',
                ec='black',
                width=arrow_scale,
                length_includes_head=True,
                head_width=arrow_scale * 3,
                head_length=arrow_scale * 5
            )
            ax.arrow(
                self.x0,
                self.y0,
                self.a2[0],
                self.a2[1],
                fc='white',
                ec='black',
                width=arrow_scale,
                length_includes_head=True,
                head_width=arrow_scale * 3,
                head_length=arrow_scale * 5
            )

    def fit_atom_columns(
            self,
            buffer=0,
            local_thresh_factor=0.95,
            pos_toler=None,
            peak_sharpening_filter='auto',
            peak_grouping_filter='auto',
            select_sites_by='elem',
            sites_to_fit='all',
            watershed_line=True,
            parallelize=True,
            use_circ_gauss=False,
            use_bounds=False,
            use_background_param=True,
            pos_bound_dist=None
    ):
        """Algorithm for fitting 2D Gaussians to HR STEM image.

        1) Laplacian of Gaussian filter applied to image to isolate individual
        atom column areas. Watershed method used to generate individual mask
        regions.
        2) Gaussian filter used to blur image and create mask areas of closely
        spaced peaks for simultaneous fitting of the group.
        3) Runs fitting algorithm. First attempt for each atomic column group
        uses the unbounded "BFSG" solver. In the event the first attempt
        returns physically unrealistic parameter values, the bounded
        "L-BFSG-B" solver is used.

        Requires reference lattice or initial guesses for all atom columns in
        the image (i.e. self.at_cols_uncropped must be a Pandas DataFrame with
        values in 'x_ref' and 'y_ref' columns). This can be achieved by running
        self.get_reference_lattice or generating initial peak positions using
        another peak finding algorithm and assigning the result to
        self.at_cols_uncropped.

        ***Other peak finding options to be implimented in the future.

        Stores fitting parameters for each atom column in self.at_cols.

        Parameters
        ----------
        buffer : int
            Distance defining the image border used to ignore atom columns
            whose fits my be questionable.

        local_thresh_factor : scalar
            Removes background from each segmented region by thresholding.
            Threshold value determined by finding the maximum value of edge
            pixels in the segmented region and multipling this value by the
            local_thresh_factor value. The LoG-filtered image (with
            sigma=peak_sharpening_filter) is used for this calculation.
            Default: 0.95.

        pos_toler : scalar or None
            The maximum allowed distance between the registered refernce
            lattice and a detected peak, in Angstroms. If greater than this
            distance, the reference lattice position will be used for the
            initial peak fitting guess. If no peak is present at the reference
            lattice position, the fitting algorithm will raise an error.
            Conversly, if a neighboring atom column is identified as belonging
            to both its correct lattice point and to a point belonging to a
            missing neighbor, two gaussians will be fit to a single atom
            column. In general, this parameter should be set as low as possible
            while ensuring that atom columns with large displacements are still
            assigned to their appriate reference lattice point. If None,
            the position error allowed will be half the specified probe size.
            Default: None.


        peak_sharpening_filter : scalar or 'auto'
            The Laplacian of Gaussian sigma value to use for peak sharpening
            for defining peak regions via the Watershed segmentation method.
            Should be approximately pixel_size_est / probe_fwhm / 2.
            If 'auto', sigma found using get_feature_size().
            Default 'auto'.

        peak_grouping_filter : scalar, 'auto' or None
            The Gaussian sigma value to use for peak grouping by blurring,
            then creating image segment regions with watershed method.
            Should be approximately pixel_size_est / probe_fwhm * 0.5.
            If 'auto', sigma found using get_feature_size().
            If simultaneous fitting of close atom columns is not desired, set
            to None.
            Default: 'auto'.

        select_sites_by : str
            'at_cols' column to use for filtering to fit only a subset
            of the atom colums.
            Default: 'elem'

        sites_to_fit : str ('all') or list of strings
            The criteria for the sites to fit, e.g. a list of the elements to
            fit: ['Ba', 'Ti']
            Default: 'all'

        watershed_line : bool
            Seperate segmented regions by one pixel.
            Default: True.

        parallelize : bool
            Whether to use parallel CPU processing. Will use all available
            physical cores if set to True. If False, will use serial
            processing.
            Default: True

        use_circ_gauss : bool
            Whether to force circular Gaussians for initial fitting of atom
            columns. If True, applies all bounds to ensure physically
            realistic parameter values. In some instances, especially for
            significantly overlapping columns, using circular Guassians is
            more robust in preventing obvious atom column location errors.
            In general, however, atom columns may be elliptical, so using
            elliptical Guassians should be preferred.
            Default: False

        use_bounds : bool
            Whether to apply bounds to minimization algorithm. This may be
            needed if unphysical results are obtained (e.g. negative
            background, negative amplitude, highly elliptical fits, etc.).
            The bounded version of the minimization is slower than the
            unbounded.
            Default: False

        use_background_param : bool
            Whether to use the background parameter when fitting each atom
            column or group of columns. If False, background value is forced
            to be 0.
            Default: True

        pos_bound_dist : 'auto' or scalar or None
            The +/- distance in Angstroms used to bound the x, y position of
            each atom column fit from its initial guess location. If 'auto',
            half the minimum seperation between atom columns in the reference
            lattice is used. If 'None', position bounds are not used.
            Argument is only functional when 'use_bounds' is set to True.
            Default: None

        Returns
        -------
        None.

        """

        print('Preparing to fit atom columns...')

        """Handle various settings configurations, prepare for masking
        process, and check for exceptions"""
        t = [time.time()]
        self.buffer = buffer
        if not use_background_param:
            use_bounds = True

        if self.at_cols_uncropped.shape[0] == 0:
            raise Exception(
                "Must define lattice before running fitting routine"
            )

        if self.at_cols.shape[0] == 0:
            at_cols = self.at_cols_uncropped.copy()
        else:
            at_cols = self.at_cols.copy()
            at_cols = pd.concat(
                [at_cols, self.at_cols_uncropped.loc[
                    [i for i in self.at_cols_uncropped.index.tolist()
                     if i not in at_cols.index.tolist()],
                    :]
                 ]
            )

        if peak_grouping_filter == 'auto':
            peak_grouping_filter = self.sigma * 0.75

        elif (
            (type(peak_grouping_filter) == float or
             type(peak_grouping_filter) == int)
            and peak_grouping_filter > 0
        ) or peak_grouping_filter is None:
            pass

        else:
            raise Exception(
                '"peak_grouping_filter" must be "auto", a positive scalar '
                + 'or None.'
            )

        if peak_sharpening_filter == 'auto':
            peak_sharpening_filter = self.sigma * 0.75

        elif (
            (type(peak_sharpening_filter) is float or
             type(peak_sharpening_filter) is int) and
            peak_sharpening_filter > 0
        ):

            pass

        else:
            raise Exception(
                '"peak_sharpening_filter" must be "auto" or a positive scalar.'
            )

        img_LoG = image_norm(
            -gaussian_laplace(
                self.image,
                peak_sharpening_filter,
                truncate=2
            )
        )
        self.img_LoG = img_LoG

        if peak_grouping_filter is not None:
            img_gauss = image_norm(
                gaussian_filter(
                    self.image,
                    peak_grouping_filter,
                    truncate=2
                )
            )

        if sites_to_fit != 'all':
            at_cols = at_cols[
                at_cols.loc[:, select_sites_by].isin(sites_to_fit)
            ]

            if at_cols.shape[0] == 0:
                raise Exception(
                    "Filtering by 'sites_to_fit' resulted in no atom " +
                    "columns remaining. Check arguments."
                )

        """Find minimum distance (in pixels) between atom columns for peak
        detection neighborhood"""

        min_dist = self.get_min_atom_col_dist()

        if pos_bound_dist == 'auto':
            pos_bound_dist = min_dist/2
        elif pos_bound_dist is None:
            pos_bound_dist = np.inf
        elif (np.isin(type(pos_bound_dist), [float, int]).item() &
              (pos_bound_dist > 0)):
            pos_bound_dist = pos_bound_dist / self.pixel_size_est
        else:
            raise Exception(
                "'pos_bound_dist' must be 'auto', a positive scalar or None."
            )

        t += [time.time()]
        print(f'Step 1 (Initial checks): {(t[-1]-t[-2]) :.{2}f} sec')

        """Use Watershed segmentation with LoG filtering to generate fitting
        masks"""
        peak_masks, num_peak_masks, slices_LoG, xy_peak = watershed_segment(
            img_LoG,
            local_thresh_factor=local_thresh_factor,
            watershed_line=watershed_line,
            min_dist=min_dist * 0.75
        )

        t += [time.time()]
        print(f'Step 2 (fitting masks): {(t[-1]-t[-2]) :.{2}f} sec')

        """Use Watershed segmentation with Gaussian blur to group columns for
        simultaneous fitting"""
        if peak_grouping_filter is not None:
            group_masks, _, _, _ = watershed_segment(
                img_gauss,
                local_thresh_factor=0,
                watershed_line=watershed_line,
                min_dist=min_dist
            )

            t += [time.time()]
            print(f'Step 3 (grouping masks): {(t[-1]-t[-2]) :.{2}f} sec')

        else:
            group_masks = peak_masks
            t += [time.time()]
            print('Step 3 (grouping masks) was skipped.')

        """Match the reference lattice points to local peaks in img_LoG.
        These points will be initial position guesses for fitting"""
        xy_peak = xy_peak.loc[:, 'x':'y'].to_numpy(dtype=float)
        xy_ref = at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float)
        inds = np.array(
            [np.argmin(norm(xy_peak - xy, axis=1)) for xy in xy_ref]
        )
        xy_peak = np.array([xy_peak[ind, :] for ind in inds])
        slices_LoG = [slices_LoG[ind] for ind in inds]

        """If the difference between detected peak position and reference
        position is greater than the position tolerance, the reference is taken
        as the initial guess."""

        """*** OR now we will just throw them out???"""
        if pos_toler is None:
            pos_toler = peak_sharpening_filter
        else:
            pos_toler /= self.pixel_size_est

        pos_errors = (norm(xy_peak - xy_ref, axis=1)
                      > pos_toler
                      ).reshape((-1, 1))
        pos_errors = np.concatenate((pos_errors, pos_errors), axis=1)
        xy_peak = np.where(pos_errors, xy_ref, xy_peak)

        """Find corresponding mask (from both LoG and Gauss filtering) for each
        peak"""
        peak_masks_to_peaks = map_coordinates(
            peak_masks,
            np.flipud(xy_peak.T),
            order=0
        ).astype(int)

        if peak_grouping_filter is not None:
            group_masks_to_peaks = map_coordinates(
                group_masks,
                np.flipud(xy_peak.T),
                order=0
            ).astype(int)
        else:
            group_masks_to_peaks = peak_masks_to_peaks

        peak_masks_used = np.unique(peak_masks_to_peaks)
        group_masks_used = np.unique(group_masks_to_peaks)

        """Throw out masks which do not correspond to at least one reference
            lattice point"""
        peak_masks = np.where(
            np.isin(peak_masks, peak_masks_used),
            peak_masks,
            0
        )
        self.peak_masks = np.where(peak_masks >= 1, 1, 0)

        if peak_grouping_filter is not None:
            group_masks = np.where(
                np.isin(group_masks, group_masks_used),
                group_masks,
                0
            )
            self.group_masks = np.where(group_masks >= 1, 1, 0)
        else:
            group_masks = peak_masks
            self.group_masks = self.peak_masks

        """Find sets of reference columns for each grouping mask"""
        peak_groups = [
            np.argwhere(group_masks_to_peaks == mask_num).flatten()
            for mask_num in group_masks_used if mask_num != 0
        ]

        group_sizes, counts = np.unique(
            [len(group) for group in peak_groups],
            return_counts=True
        )

        if peak_grouping_filter:
            print('Atomic columns grouped for simultaneous fitting:')
            for i, size in enumerate(group_sizes):
                print(f'{counts[i]}x {size}-column groups')

        """Find min & max slice indices for each group of fitting masks"""
        group_fit_slices = [
            [slices_LoG[ind] for ind in inds]
            for inds in peak_groups
        ]

        group_fit_slices = [np.array(
            [[sl[0].start, sl[0].stop,
              sl[1].start, sl[1].stop]
             for sl in group]).T
            for group in group_fit_slices
        ]

        group_slices = [np.s_[
            np.min(group[0]): np.max(group[1]),
            np.min(group[2]): np.max(group[3])]
            for group in group_fit_slices
        ]

        """Pack image slices and metadata together for the fitting routine"""

        args_packed = pack_data_prefit(
            data=self.image,
            slices=group_slices,
            masks=peak_masks,
            xy_peaks=xy_peak,
            peak_mask_index=peak_masks_to_peaks,
            peak_groups=peak_groups,
            pos_bound_dist=pos_bound_dist,
            use_circ_gauss=use_circ_gauss,
            use_bounds=use_bounds,
            use_background_param=use_background_param,
        )

        self.args_packed = args_packed

        at_cols_inds = at_cols.index.to_numpy(dtype=float)
        self.at_cols_inds = at_cols_inds

        unpacking_inds = [at_cols_inds[inds] for inds in peak_groups]

        self.unpacking_inds = unpacking_inds

        t += [time.time()]
        print(f'Step 4 (Final prep): {(t[-1]-t[-2]) % 60 :.{2}f} sec')

        """Run fitting routine"""
        print('Fitting atom columns...')

        if parallelize is True:
            """Large data set: use parallel processing"""
            print('Using parallel processing')
            n_jobs = psutil.cpu_count(logical=True)

            results_ = Parallel(n_jobs=n_jobs)(
                delayed(fit_gaussian_group)(
                    *args,
                ) for args in tqdm(args_packed)
            )

        else:
            """Small data set: use serial processing"""
            print('Using serial processing')

            results_ = [fit_gaussian_group(
                *args,
            ) for args in tqdm(args_packed)]

        results = np.concatenate([np.concatenate(
            (result,
             unpacking_inds[i].reshape(-1, 1)),
            axis=1
        )
            for i, result in enumerate(results_)]
        )

        t += [time.time()]
        print(f'Step 5 (Fitting): {(t[-1]-t[-2]) % 60 :.{2}f} sec')

        """Post-process results"""

        col_labels = ['x_fit', 'y_fit', 'sig_1', 'sig_2',
                      'theta', 'peak_int', 'bkgd_int', 'total_col_int']
        if not col_labels[0] in at_cols.columns:
            empty = pd.DataFrame(
                index=at_cols.index.tolist(),
                columns=col_labels
            )

            at_cols = at_cols.join(empty)

        results = pd.DataFrame(
            data=results[:, :-1],
            index=results[:, -1].astype(int),
            columns=col_labels[:-1]
        ).sort_index()

        results['total_col_int'] = (
            2 * np.pi * results.peak_int.to_numpy(dtype=float)
            * results.sig_1.to_numpy(dtype=float)
            * results.sig_2.to_numpy(dtype=float)
        )

        at_cols.update(results)
        sigmas = at_cols.loc[:, 'sig_1':'sig_2'].to_numpy(dtype=float)
        theta = at_cols.loc[:, 'theta'].to_numpy(dtype=float)
        sig_maj_inds = np.argmax(sigmas, axis=1)
        sig_min_inds = np.argmin(sigmas, axis=1)

        sig_maj = sigmas[
            [i for i in range(sigmas.shape[0])],
            list(sig_maj_inds)
        ]

        sig_min = sigmas[
            [i for i in range(sigmas.shape[0])],
            list(sig_min_inds)
        ]

        theta += np.where(sig_maj_inds == 1, 90, 0)
        theta = ((theta + 90) % 180) - 90
        at_cols['sig_1'] = sig_maj
        at_cols['sig_2'] = sig_min
        at_cols['theta'] = theta

        '''Convert values from dtype objects to ints, floats, etc:'''
        at_cols = at_cols.infer_objects()

        '''Crop with buffer '''
        at_cols = at_cols[
            ((at_cols.x_ref >= 0) &
             (at_cols.x_ref <= self.w) &
             (at_cols.y_ref >= 0) &
             (at_cols.y_ref <= self.h))
        ].copy()

        self.at_cols_uncropped.update(at_cols)
        self.at_cols_uncropped = self.at_cols_uncropped.infer_objects()
        self.at_cols = self.at_cols_uncropped.dropna(axis=0)

        if buffer:
            roi_mask_buffer = binary_erosion(
                self.roi_mask,
                iterations=buffer
            )

        else:
            roi_mask_buffer = self.roi_mask

        self.at_cols = self.at_cols[roi_mask_buffer[
            np.around(self.at_cols.y_ref.to_numpy()).astype(int),
            np.around(self.at_cols.x_ref.to_numpy()).astype(int)
        ] == 1
        ]

        t += [time.time()]
        print(f'Step 6 (Post-processing): {(t[-1]-t[-2]) % 60 :.{2}f} sec',
              '\n Done.')

    def show_masks(
            self,
            mask_to_show='fitting',
            display_masked_image=True,
            return_fig=False,
    ):
        """View the fitting or grouping masks. Useful for troubleshooting
            fitting problems.

        Parameters
        ----------
        mask_to_show : str ('fitting' or 'grouping')
            Which mask to show.
            Default: 'fitting'

        display_masked_image : bool
            Whether to show the image masked by the specified mask (if True)
            or the mask alone (if False).
            Default: True

        Returns
        -------
        fig : matplotlib figure object

        """

        if mask_to_show == 'fitting':
            mask = self.peak_masks
        elif mask_to_show == 'grouping':
            mask = self.group_masks
        else:
            raise Exception(
                "The argument 'mask_to_show' must be either: "
                "'fitting' or 'grouping'."
            )
        if mask is None:
            raise Exception(
                'Mask not defined. If you have run "rotate_image_and_data()" '
                + 'the mask was removed. '
            )

        fig, ax = plt.subplots()
        if display_masked_image:
            ax.imshow(self.image * mask)
        else:
            ax.imshow(mask)
        ax.set_xticks([])
        ax.set_yticks([])

        if return_fig:
            return fig, ax

    def refine_reference_lattice(
            self,
            filter_by='elem',
            sites_to_use='all',
            outlier_disp_cutoff=None
    ):
        """Refines the reference lattice on fitted column positions.

        Refines the referene lattice origin and basis vectors to minimize
        the sum of the squared errors between the reference and fitted
        positions. Prints residual lattice distortion values and estimated
        pixel size to the console.

        Parameters
        ----------
        filter_by : str
            The DataFrame column used to filter for selecting a subset of
            atom columns. Typically 'elem' unless a DataFrame column is added
            by the user such as to label the lattice site naming convention,
            e.g. A, B, O sites in perovskites.
            Default 'elem'.

        sites_to_use : str or array_like of strings
            The sites to use for refinement. 'all' or a list of the site
            labels.
            Default 'all'.

        outlier_disp_cutoff : None or scalar
            Criteria for removing outlier atomic column fits from the
            reference lattice refinement (in pm). The maximum difference
            between the fitted position and the corresponding reference
            lattice point. All positions  with greater errors than this value
            will be removed. If None, 100 pm will be used.
            Default None.

        Returns
        -------
        None.

        """

        if sites_to_use == ('all' or ['all']):
            filtered = self.at_cols.copy()
        else:
            if type(sites_to_use) == list:
                filtered = self.at_cols[
                    self.at_cols.loc[:, filter_by].isin(sites_to_use)
                ].copy()

            elif type(sites_to_use) == str:
                filtered = self.at_cols[
                    self.at_cols.loc[:, filter_by] == sites_to_use
                ].copy()

            else:
                raise Exception('"sites_to_use" must be a string or a list')

        if filtered.shape[0] == 0:
            raise Exception(
                'No atom columns found to use for '
                + 'refinement with arguments given'
            )

        if outlier_disp_cutoff is None:
            outlier_disp_cutoff = 1 / self.pixel_size_est

        else:
            outlier_disp_cutoff /= self.pixel_size_est * 100

        filtered = filtered[norm(
            filtered.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
            - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
            axis=1
        ) < outlier_disp_cutoff].copy()

        M = filtered.loc[:, 'u':'v'].to_numpy(dtype=float)
        xy = filtered.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)

        p0 = np.concatenate(
            (self.dir_struct_matrix.flatten(),
             np.array([self.x0, self.y0]))
        )

        params = fit_lattice(p0, xy, M)

        self.a1 = params[:2]
        self.a2 = params[2:4]

        self.dir_struct_matrix = params[:4].reshape((2, 2))

        print('Origin shift:', params[4:] - np.array([self.x0, self.y0]))
        self.x0 = params[4]
        self.y0 = params[5]
        print('Optimized basis vectors:', self.dir_struct_matrix)

        self.basis_offset_pix = self.basis_offset_frac @ self.dir_struct_matrix

        self.at_cols[['x_ref', 'y_ref']] = (
            self.at_cols.loc[:, 'u':'v'].to_numpy(dtype=float)
            @ self.dir_struct_matrix + np.array([self.x0, self.y0])
        )

        self.pixel_size_est = self.get_est_pixel_size()

        theta_ref = np.degrees(np.arccos(
            self.dir_struct_matrix[0, :]
            @ self.dir_struct_matrix[1, :].T
            / (norm(self.dir_struct_matrix[0, :])
               * norm(self.dir_struct_matrix[1, :].T))
        ))

        shear_distortion_res = np.radians(90 - theta_ref)

        scale_distortion_res = 1 - (
            (norm(self.a1)
             * norm(self.a_2d[:, 1])) /
            (norm(self.a2)
             * norm(self.a_2d[:, 0]))
        )

        print('')
        print(
            'Residual distortion of reference lattice basis vectors from .cif:'
        )
        print(f'Scalar component: {scale_distortion_res * 100 :.{4}f} %')
        print(f'Shear component: {shear_distortion_res :.{6}f} (radians)')
        print(f'Estimated Pixel Size: {self.pixel_size_est * 100 :.{3}f} (pm)')

    def get_fitting_residuals(self):
        """Calculates fitting residuals for each atom column. Plots all
        residuals and saves the normalized sum of squares of the residuals for
        each column in the 'at_cols' dataframe.

        Parameters
        ----------
        None.

        Returns
        -------
        fig : matplotlib figure object

        """

        def get_group_residuals(
                args,
                params,
                buffer,
                image_dims
        ):

            [img_sl,
             mask_sl,
             mask_nums,
             xy_start,
             xy_peak,
             inds,
             _] = args

            # Isolate atom column areas
            masks = np.where(np.isin(mask_sl, mask_nums), mask_sl, 0)
            # Shift origin for slice
            params[:, :2] -= xy_start

            # Mask data
            data = np.where(masks > 0, img_sl, 0).flatten()

            # Get indices of data and remove masked-off areas
            data_inds = np.nonzero(data)
            y, x = np.indices(masks.shape)
            x = np.take(x.flatten(), data_inds)
            y = np.take(y.flatten(), data_inds)
            data = np.take(data, data_inds)
            masks = np.take(masks.flatten(), data_inds)

            # Initalize model array
            model = np.zeros(masks.shape)
            # Add background intensity for each column region
            for i, mask_num in enumerate(mask_nums):
                model = np.where(masks == mask_num, params[i, -1], model)
            # Add Gaussian peak shapes
            for peak in params:
                model += gaussian_2d(x, y, *peak[:-1], I_o=0)

            # Calculate residuals
            r = data - model

            # Calculate normed sum of squares for each column, add to
            # dataframe. If the fitted position is beyond the buffer area
            # of the image, remove that column from the total residuals.
            RSS_norm = []
            params[:, :2] += xy_start
            for i, mask_num in enumerate(mask_nums):
                n = np.count_nonzero(masks == mask_num)
                r_i = r[masks == mask_num]
                RSS_norm += [np.sum(r_i**2) / n]

                if ((params[i, 0] <= buffer) |
                    (params[i, 0] >= image_dims[1] - buffer) |
                    (params[i, 1] <= buffer) |
                        (params[i, 1] >= image_dims[0] - buffer)):
                    r[masks == mask_num] = 0

            x += xy_start[0]
            y += xy_start[1]

            return x, y, r, RSS_norm

        group_residuals = [
            get_group_residuals(
                args, self.at_cols_uncropped.loc[
                    args[5], 'x_fit':'bkgd_int'
                ].to_numpy(dtype=float),
                self.buffer,
                self.image.shape
            )
            for args in self.args_packed
        ]

        self.residuals = np.zeros(self.image.shape)

        for counter, [x, y, R, RSS_norm] in enumerate(group_residuals):
            self.residuals[y, x] += R
            for i, ind in enumerate(self.args_packed[counter][5]):
                self.at_cols_uncropped.loc[ind, 'RSS_norm'] = RSS_norm[i]

        self.at_cols['RSS_norm'] = np.nan
        self.at_cols.update(self.at_cols_uncropped)

        cmap_lim = np.max(np.abs(self.residuals))

        fig, axs = plt.subplots(ncols=1, figsize=(10, 10), tight_layout=True)
        axs.set_xticks([])
        axs.set_yticks([])
        res_plot = axs.imshow(
            self.residuals,
            cmap='bwr',
            norm=Normalize(vmin=-cmap_lim, vmax=cmap_lim)
        )

        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(res_plot, cax=cax)

        axs.scatter(self.at_cols.loc[:, 'x_fit'],
                    self.at_cols.loc[:, 'y_fit'],
                    color='black', s=4)

        r = self.residuals.flatten()
        data = self.image * self.peak_masks
        data[:self.buffer, :] = 0
        data[-self.buffer:, :] = 0
        data[:, :self.buffer] = 0
        data[:, -self.buffer:] = 0

        data = data.flatten()
        inds = data.nonzero()
        r = np.take(r, inds)
        data = np.take(data, inds)

        n = data.shape[1]
        m = self.at_cols.shape[0] * 7
        v = n - m

        R_sq = 1 - np.sum(r**2) / np.sum((data - np.mean(data))**2)

        chi_sq = np.sum((r**2) / v)

        print(f'chi-squared of all atom column fits: {chi_sq :.{4}e} \n')
        print(f'R-squared of all atom column fits: {R_sq :.{6}f} \n')

        return fig

    def get_avg_unitcell(self, upsample=1, supercell=(1, 1), plot=True):
        """
        Get the average unitcell based on the registered lattice.

        Parameters
        ----------
        upsample : int
            Factor by which to upsample the output unit cell.
            *** Note: upsampling is not currently implimented in a rigurous way
            and results may be less than desirable. Not advised to upsample
            more than 2x.

        supercell : two-tuple
            The number of basic unit cells to include in the output averaged
            unitcell along the two basis vectors.

        plot : bool
            Whether to plot the result.

        Returns
        -------
        H : array
            The resulting average unitcell. Also stored in self.avg_unitcell.

        """

        # Get U,V combinations for all unit cells
        uv = np.unique(
            np.floor(self.at_cols.loc[:, 'u':'v'].to_numpy()),
            axis=0
        )

        # Lattice vector information
        dsm = self.dir_struct_matrix * np.array([supercell]).T
        a1 = dsm[0]
        a2 = dsm[1]

        a1unit = a1 / norm(a1)
        a2unit = a2 / norm(a2)

        # Get origin point for each lattice cell
        x0y0 = uv @ dsm + [self.x0, self.y0]

        # Mask off the image
        im = self.image * self.roi_mask

        # Get values for each unit cell, subtracting origin coordinates
        x = []
        y = []
        z = []

        print('Getting unit cell data')
        for x0, y0 in tqdm(x0y0):
            xverts = [
                x0 - 1.5*a1unit[0] - 1.5*a2unit[0],
                x0 + a1[0] + 1.5*a1unit[0] - 1.5*a2unit[0],
                x0 + a1[0] + 1.5*a1unit[0] + a2[0] + 1.5*a2unit[0],
                x0 + a2[0] + 1.5*a2unit[0] - 1.5*a1unit[0]
            ]

            yverts = [
                y0 - 1.5*a1unit[1] - 1.5*a2unit[1],
                y0 + a1[1] + 1.5*a1unit[1] - 1.5*a2unit[1],
                y0 + a1[1] + 1.5*a1unit[1] + a2[1] + 1.5*a2unit[1],
                y0 + a2[1] + 1.5*a2unit[1] - 1.5*a1unit[1]
            ]

            yind, xind = polygon(yverts, xverts, im.shape)

            imint = im[yind, xind]

            unmasked_data = np.nonzero(imint)
            imint = np.take(imint, unmasked_data).flatten()
            xind = np.take(xind, unmasked_data
                           ).flatten() - x0
            yind = np.take(yind, unmasked_data
                           ).flatten() - y0

            z += imint.tolist()
            x += xind.tolist()
            y += yind.tolist()

        # Rotate coordinates so a1 is horizontal
        xy = np.vstack([x, y]).T

        theta = rotation_angle_bt_vectors([1, 0], a1)

        xy = rotate_xy(xy, theta, [0, 0])

        a1 = rotate_xy(a1, theta, [0, 0]).squeeze()

        a2 = rotate_xy(a2, theta, [0, 0]).squeeze()

        dsm = np.vstack([a1, a2])

        x = xy[:, 0]
        y = xy[:, 1]

        h = np.max([dsm[:, 1]]) - np.min(dsm[:, 1]) + 1
        w = np.max([dsm[:, 0]]) - np.min(dsm[:, 0]) + 1

        data_coords = np.vstack([x, y]).T

        print('Getting density estimate')
        H = image_norm(linearKDE_2D(
            data_coords,
            xlim=(np.min(dsm[:, 0]), np.min(dsm[:, 0]) + w),
            ylim=(np.min(dsm[:, 1]), np.min(dsm[:, 1]) + h),
            d=1/upsample,
            r=upsample,
            weights=z))

        if plot:
            quickplot(H, pixel_size=self.pixel_size_est/10, pixel_unit='nm')

        self.avg_unitcell = H

        return H

    def get_vpcfs(
        self,
        xlim,
        ylim,
        d=0.05,
        area=None,
        filter_by='elem',
        sublattice_list=None,
        get_only_partial_vpcfs=False,
        affine_transform=False,
    ):
        """Calculates pair-pair vPCFs for all sublattices in the
        AtomColumnLattice object or a specified subset.

        The vPCFs are stored in the "vpcfs" dictionary within the
        AtomColumnLattice object. Peak shape measurement and plotting are
        included as separate methods. xlim and ylim work best if one of the
        lattice vectors is nearly parallel to the x-axis. For an image with the
        lattice rotated relative to the image edges, use
        HRImage.rotate_image_and_data() following atom column fitting and prior
        to vPCF analysis.

        Parameters
        ----------
        xlim, ylim : scalar array of shape (2,)
            The limits of the vPDF along each dimension in lattice basis vector
            multiples (i.e. fractional coordinates). self.a1 is assumed to be
            horizontal and and self.a2 is assumed to be vertical. Must include
            0 in both x and y. The limits will determine the size of the vPCF
            array and the time required to calculate it.

        d : scalar
            The pixel size of the vPCF in units of Angstroms.
            Default: 0.05

        area : scalar
            The area containing the data points. Used to calculate the density
            for normalizing the vPCF values. If None, the area of the roi_mask
            is used. If you did not explicitly define the roi_mask, it is the
            image size minus any edge buffer applied.
            Default: None

        filter_by : str
            The DataFrame column used to determine the sublattice.
            Default: 'elem'

        sublattice_list : list of strings or None
            List of sublattices for which to calculate vPCFs. If None,
            find all pair-pair vPCFs.

        get_only_partial_vpcfs : bool
            Only get partial vPCFs if True. Otherwise get all unique
            combinations of partial and pair-pair vPCFs.
            Default: False

        affine_transform : bool
            If True, applies an affine transformation to the data so that
            the basis vectors match the lattice parameters from the CIF file
            used to generate the reference lattice. This is useful if
            significant scan or drift related distortion remains present in
            the image. Absolute measurement accuracy (i.e. distance between
            peaks and peak widths) will be a function of how well matched the
            CIF file is to the actual sample imaged. Realtive information
            (i.e. size of one peak compared to another) will be mostly
            unaffected.

        Returns
        -------
        None

        """

        if self.pixel_size_cal is not None:
            pixel_size = self.pixel_size_cal
        else:
            pixel_size = self.pixel_size_est

        self.vpcfs = {}
        sites = [site for site in
                 pd.unique(self.at_cols.loc[:, filter_by])]
        sites.sort()
        if sublattice_list is not None:
            sites = [site for site in sites
                     if np.isin(site, sublattice_list).item()]

        if get_only_partial_vpcfs:
            pair_pairs = [[site1, site2]
                          for count, site1 in enumerate(sites)
                          for site2 in sites[count:] if site1 == site2]
        else:
            pair_pairs = [[site1, site2]
                          for count, site1 in enumerate(sites)
                          for site2 in sites[count:]]

        at_cols = self.at_cols.copy()

        # Get area from the ROI mask (in A**2):
        area = np.count_nonzero(self.roi_mask) * self.pixel_size_est**2

        a1_mag = norm(self.a_2d[0, :])
        a2_mag = norm(self.a_2d[1, :])

        for pair in pair_pairs:
            pair_pair_str = f'{pair[0]}-{pair[1]}'
            print(f'Calculating {pair_pair_str} vPCF')
            sub1 = at_cols[
                (at_cols[filter_by] == pair[0])
            ].loc[:, 'x_fit': 'y_fit'].to_numpy()

            sub2 = at_cols[
                (at_cols[filter_by] == pair[1])
            ].loc[:, 'x_fit': 'y_fit'].to_numpy()

            if affine_transform:
                print('Appling affine transformation to data...')
                sub1 = (sub1 @ self.dir_struct_matrix
                        / norm(self.dir_struct_matrix, axis=1)**2
                        ) @ self.a_2d
                sub2 = (sub2 @ self.dir_struct_matrix
                        / norm(self.dir_struct_matrix, axis=1)**2
                        ) @ self.a_2d
            elif not affine_transform:
                sub1 *= pixel_size
                sub2 *= pixel_size

            self.vpcfs[pair_pair_str], origin = get_vpcf(
                xlim=np.array(xlim)*a1_mag,
                ylim=np.array(ylim)*a2_mag,
                coords1=sub1,
                coords2=sub2,
                d=d,
                area=area,
                method='weighted',
            )

        self.vpcfs['metadata'] = {'origin': origin,
                                  'pixel_size': d,
                                  'filter_by': filter_by,
                                  'pair_pair': not get_only_partial_vpcfs,
                                  'affine_transform': affine_transform,
                                  }

    def get_vpcf_peak_params(
        self,
        sigma=10,
        buffer=10,
        method='moments',
        sigma_group=None,
        thresh_factor=1,
    ):
        """Calculates shape of peaks in each pair-pair vPCF.

        Calculates peak equivalent ellipse shapes and locations for all vPCFs
        stored in the "vpcfs" dictionary within the AtomColumnLattice object.
        Results are saved in the "vpcf_peaks" dictionary and can be plotted
        with the "plot_vpcfs" method.

        Parameters
        ----------
        sigma : scalar
            The Gaussian sigma for bluring peaks prior to identifying
            mask areas for each peak by he watershed algorithm. In units of
            vPCF pixels.

        method : 'momenets' or 'gaussian'
            Method to calculate shape and location of peaks. 'moments' uses
            image moments calculations to measure the peaks while 'gaussian'
            fits with a 2D Gaussian. Methods are roughly equivalent and give
            the parameters of the ellipse that best describes the peak shape.
            Gaussian fitting, however, is more unstable and slightly slower.
            The primary reason to use 2D Gaussians is in the case of peaks
            with overalapping tails when simultaneous fitting is needed for
            accurate measurements.
            In future the moments method will return higher order moments along
            the major and minor axes to measure additional statistical
            parameters of the distributions.
            Default: 'moments'

        sigma_group : scalar or None
            The maximum separation distance used for grouping close peaks for
            simultaneous fitting. This is necessary if peaks are close enough
            to have overlapping tails. Must use Gaussian fitting to account for
            overlap when determining peak shapes. If None, not used.
            Default: None.

        thresh_factor : scalar
            Adjusts the minimum amplitude for considering an identified local
            maximum to be a peak for the purposes of finding its shape. By
            default peaks with less than 10% of the amplitude of the largest
            peak are not considered for fitting. A thresh_factor of 2 would
            raise this cutoff to 20% while 0.5 would lower it to 5%.
            Default: 1.

        Returns
        -------
        None

        """

        xy_bnd = 10    # Position bound limit for gaussian fitting

        self.vpcf_peaks = {}
        for key, vpcf in self.vpcfs.items():
            if key == 'metadata':
                continue

            print(f'Calculating peaks for {key} vPCF')
            self.vpcf_peaks[key] = pd.DataFrame(columns=['x_fit', 'y_fit',
                                                         'sig_maj', 'sig_min',
                                                         'theta', 'ecc',
                                                         'peak_max'])
            if sigma is not None:
                pcf_sm = gaussian_filter(
                    vpcf,
                    sigma=sigma,
                    truncate=3,
                    mode='constant',
                )

            else:
                pcf_sm = copy.deepcopy(vpcf)
                sigma = 2

            masks_indiv, n_peaks, _, peaks = watershed_segment(
                pcf_sm,
                min_dist=sigma,
                local_thresh_factor=0,
                sigma=None,
                buffer=buffer,
                watershed_line=False,
            )

            peaks['peak_max'] = vpcf[
                peaks.loc[:, 'y'].to_numpy(dtype=int),
                peaks.loc[:, 'x'].to_numpy(dtype=int)
            ]

            thresh = np.max(peaks.loc[:, 'peak_max']) * 0.1 * thresh_factor

            peaks = peaks[(peaks.loc[:, 'peak_max'] > thresh)
                          ].reset_index(drop=True)
            # n_peaks = peaks.shape[0]
            xy_peak = peaks.loc[:, 'x':'y'].to_numpy(dtype=int)
            labels = peaks.loc[:, 'label'].to_numpy(dtype=int)

            if sigma_group is not None:
                if method != 'gaussian':
                    print('Using Gaussian method to account for peak overlap.')
                    method = 'gaussian'

                pcf_sm = gaussian_filter(
                    vpcf,
                    sigma=sigma_group,
                    truncate=3
                )

                group_masks, _, _, _ = watershed_segment(
                    pcf_sm,
                    min_dist=sigma_group,
                    local_thresh_factor=0,
                    sigma=None,
                    buffer=0,
                    watershed_line=True
                )

                group_masks_to_peaks = map_coordinates(
                    group_masks,
                    np.flipud(xy_peak.T),
                    order=0
                ).astype(int)

                labels = np.unique(group_masks_to_peaks).astype(int)

                group_masks = np.where(
                    np.isin(group_masks, labels),
                    group_masks,
                    0
                )

            if method == 'moments':
                for i, label in tqdm(enumerate(labels)):
                    pcf_masked = np.where(masks_indiv == label, 1, 0
                                          )*self.vpcfs[key]
                    peak_max = np.max(pcf_masked)
                    x_fit, y_fit, ecc, theta, sig_maj, sig_min = \
                        img_ellip_param(
                            pcf_masked
                        )

                    self.vpcf_peaks[key].loc[i, 'x_fit':] = [
                        x_fit,
                        y_fit,
                        sig_maj,
                        sig_min,
                        theta,
                        ecc,
                        peak_max,
                    ]

            elif method == 'gaussian':
                for i in labels:
                    if sigma_group is None:
                        mask = np.where(masks_indiv == i, 1, 0)
                    else:
                        mask = np.where(group_masks == i, 1, 0)

                    pcf_masked = mask * self.vpcfs[key]

                    match = np.argwhere([mask[y, x] for x, y in xy_peak])

                    # match = np.array([[y, x] for x, y in xy_peak])

                    mask_peaks = peaks.loc[match.flatten(), :]
                    if sigma_group is not None:

                        pcf_masked *= np.where(
                            np.isin(masks_indiv, peaks.loc[:, 'label']),
                            1, 0)

                    p0 = []
                    bounds = []
                    for j, (ind, row) in enumerate(mask_peaks.iterrows()):
                        mask_num = masks_indiv[int(row.y), int(row.x)]
                        mask = np.where(masks_indiv == mask_num, 1, 0)
                        peak_masked = mask * self.vpcfs[key]

                        x0, y0, ecc, theta, sig_maj, sig_min = img_ellip_param(
                            peak_masked
                        )

                        p0 += [
                            x0,
                            y0,
                            sig_maj,
                            sig_maj/sig_min,
                            np.max(pcf_masked),
                            0
                        ]

                        bounds += [(x0 - xy_bnd, x0 + xy_bnd),
                                   (y0 - xy_bnd, y0 + xy_bnd),
                                   (1, None),
                                   (1, None),
                                   (None, None),
                                   (0, None),
                                   ]

                    p0 = np.array(p0 + [0])
                    bounds += [(0, 0)]

                    params = fit_gaussian_ellip(
                        pcf_masked,
                        p0,
                        masks=None,
                        method='L-BFGS-B',
                        bounds=bounds
                    )

                    params = params[:, :-1]
                    params[:, 3] = params[:, 2] / params[:, 3]
                    params[:, 4] = np.degrees(params[:, 4])
                    params[:, -1] = np.sqrt(1 - params[:, 3]**2
                                            / params[:, 2]**2)

                    next_ind = self.vpcf_peaks[key].shape[0]
                    for k, p in enumerate(params):
                        self.vpcf_peaks[key].loc[next_ind + k, :] = p

            self.vpcf_peaks[key].infer_objects()

    def plot_vpcfs(
        self,
        vpcfs_to_plot='all',
        plot_equ_ellip=True,
        vpcf_cmap='Greys',
        ellip_scale_factor=10,
        ellip_colormap_param='sig_maj',
        colormap_range=None,
        unit_cell_box=True,
        unit_cell_box_color='black',
        scalebar_len=1,

    ):
        """
        Plot vPCFs.

        Parameters
        ----------
        vpcfs_to_plot : str or list of strings
            If 'all', plots all available vPCFs. If a list, plots vPCFs
            whose dictionary key matches the elements of the list.
            Default: 'all'

        plot_equ_ellip : bool
            Whether to plot the equivalent ellipses found by
            get_vpcf_peak_params().
            Default: True

        vpcf_cmap : str
            The colormap to use for displaying the vPCF data.
            Default: 'Greys'

        ellip_color_scale_param : str
            The equilalent ellipse parameter to use for ellipse coloring.
            Options are: 'sig_maj' (the major axis length) or 'ecc'
            (ellipse eccentiricity). Generallly, 'sig_maj' is most informative;
            it describes the half-width of the peak in the widest direction.
            Default: 'sig_maj'

        ellip_scale_factor : scalar
            Factor by which to scale the plotted ellipse size relative to its
            actual size. Generally, vPCF peaks are sharp and will appear very
            small. Scaling their value several times allows the plotted
            ellipses (which illustrate the shape of the underlying peak) to be
            easily seen in a figure.
            Default: 5

        colormap_range : listlike of shape (2,) or None
            The (minimum, maximum) values for the range of ellipse colors.
            Sizes of ellipse are still plotted proportionally, but colormap
            saturates below and above these values. If None, the minimum and
            maximum values of the ellipse scale parameter
            Default: None

        unit_cell_box : bool
            Whether to plot the unit cell box on the vPCF.
            Default: True

        unit_cell_box_color : str
            Line color for the unit cell box. Must be a matplotlib color.
            Default: 'black'

        scalebar_len : scalar
            The length of the scalebar to plot in Angstroms.
            Default: 1

        Returns
        -------
        fig : the matplotlib figure object.

        axs : the list of matplotlib axes objects: each vpcf plot and
            the scalebar.

        """

        prop = ellip_colormap_param
        cmap = plt.cm.plasma
        origin = self.vpcfs['metadata']['origin']
        d = self.vpcfs['metadata']['pixel_size']
        filter_by = self.vpcfs['metadata']['filter_by']

        shape_params_all = np.concatenate(
            [self.vpcf_peaks[key].loc[:, prop].to_numpy()
             for key in self.vpcf_peaks.keys()]
        )
        if prop == 'sig_maj':
            unit_conv = 100 * d
        elif prop == 'ecc':
            unit_conv = 1
        else:
            raise Exception('argument "ellip_color_scale_param" must be '
                            + 'either "sig_maj" or "ecc".')

        if colormap_range is None:
            [min_, max_] = np.array([
                np.floor(np.min(shape_params_all) * unit_conv),
                np.ceil(np.max(shape_params_all) * unit_conv)
            ])

        else:
            colormap_range = np.array(colormap_range, dtype=float).flatten()

            if (colormap_range.shape[0] == 2 and
                    np.isin(colormap_range.dtype, [float, int]).item()):
                [min_, max_] = colormap_range

            else:
                raise Exception(
                    '"colormap_range" must be: listlike of shape (2,) or None'
                )

        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

        # Set up Figure and gridspec
        if vpcfs_to_plot == 'all':
            vpcfs_to_plot = list(self.vpcfs.keys())
            vpcfs_to_plot.remove('metadata')
            sites = [site for site in
                     pd.unique(self.at_cols.loc[:, filter_by])]
            sites.sort()

            if self.vpcfs['metadata']['pair_pair']:
                nrows = ncols = len(sites)
                gs_inds = [[row, col]
                           for row in range(nrows)
                           for col in range(row, nrows)]
            else:
                nplots = len(vpcfs_to_plot)
                ncols = np.ceil(nplots**0.5).astype(int)
                nrows = (nplots // ncols + np.ceil(nplots % ncols)).astype(int)
                gs_inds = [[row, col]
                           for row in range(nrows)
                           for col in range(ncols)]

        else:
            vpcfs_found = np.isin(vpcfs_to_plot, list(self.vpcfs.keys()))
            vpcfs_not_found = np.array(vpcfs_to_plot)[~vpcfs_found]
            vpcfs_to_plot = np.array(vpcfs_to_plot)[vpcfs_found]
            if len(vpcfs_not_found) > 0:
                print('!!! Note: Specified vPCFs not found for plotting: \n',
                      f'{vpcfs_not_found} \n')
            nplots = len(vpcfs_to_plot)
            ncols = np.ceil(nplots**0.5).astype(int)
            nrows = (nplots // ncols + np.ceil(nplots % ncols)).astype(int)
            gs_inds = [[row, col]
                       for row in range(nrows)
                       for col in range(ncols)]

        h, w = self.vpcfs[list(self.vpcfs.keys())[0]].shape
        width_ratio = (w * ncols) / (h * nrows)
        fig = plt.figure(figsize=(10 * width_ratio + 0.5, 10))
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=ncols + 1,
            width_ratios=[width_ratio] * ncols + [0.05],
            wspace=0.05
        )

        axs = [fig.add_subplot(gs[row, col]) for [row, col] in gs_inds]

        axs_cbar = fig.add_subplot(gs[nrows-1, ncols])

        font = 14

        for i, key in enumerate(vpcfs_to_plot):
            row, col = gs_inds[i]
            axs[i].imshow(self.vpcfs[key], cmap=vpcf_cmap, zorder=0)

            if plot_equ_ellip:
                peaks = self.vpcf_peaks[key]
                '''Ellipse Plotting'''
                elips = [Ellipse(
                    xy=[peak.x_fit, peak.y_fit],
                    width=2 * peak.sig_maj * ellip_scale_factor,
                    height=2 * peak.sig_min * ellip_scale_factor,
                    angle=peak.theta,
                    facecolor=cmap((peak[prop] * unit_conv - min_) /
                                   (max_ - min_)),
                    lw=2,
                    zorder=2,
                    alpha=0.65)
                    for i, peak in peaks.iterrows()
                ]

                for elip in elips:
                    axs[i].add_artist(elip)

            axs[i].set_xticks([])
            axs[i].set_yticks([])

            axs[i].scatter(
                origin[0],
                origin[1],
                c='red',
                marker='+')

            cell_box = np.array(
                [origin + np.sum(corner * self.a_2d / d, axis=1)
                 * np.array([1, -1])
                 for corner in corners]
            )

            label = key
            axs[i].text(
                0.05, 1.02,
                label, color='black', size=font,
                ha='left',
                va='bottom',
                weight='bold',
                transform=axs[i].transAxes
            )
            if unit_cell_box:
                rectangle = Polygon(
                    cell_box,
                    fill=False,
                    ec=unit_cell_box_color,
                    zorder=1,
                    lw=0.25)

                axs[i].add_artist(rectangle)

            if (row == 0 and col == 0):
                scalebar = ScaleBar(
                    d/10,
                    'nm',
                    font_properties={'size': 12},
                    pad=0.3,
                    border_pad=0.6,
                    box_color='dimgrey',
                    height_fraction=0.02,
                    color='white',
                    location='lower right',
                    fixed_value=scalebar_len,
                )

                axs[i].add_artist(scalebar)

            for axis in ['top', 'bottom', 'left', 'right']:
                axs[i].spines[axis].set_linewidth(1.5)
        cbar_ell = fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=min_, vmax=max_), cmap=cmap),
            cax=axs_cbar,
            orientation='vertical',
            shrink=0.6,
            aspect=12,
        )

        cbar_ell.set_label(label='Major axis length (pm)', fontsize=10)

        return fig, axs, axs_cbar

    def get_vector_from_vpcf(
        self,
        vpcf,
        number_of_peaks_to_pick=1,
        plot_equ_ellip=True,
    ):

        [sub1, sub2] = vpcf.split('-')

        fig, axs, _ = self.plot_vpcfs(
            vpcfs_to_plot=[vpcf],
            plot_equ_ellip=plot_equ_ellip,
            vpcf_cmap='Greys',
            ellip_colormap_param='sig_maj',
            ellip_scale_factor=10,
            unit_cell_box=True,
            unit_cell_box_color='black'
        )

        fig.suptitle(
            (f'*** Click on {number_of_peaks_to_pick} peaks '
             + 'to plot their corresponding atom column spacings. ***'),
            fontsize=16,
            c='black',
            fontweight='bold'
        )

        pts = np.array(plt.ginput(number_of_peaks_to_pick, timeout=30))

        # # axs[0].scatter(pts[0][0], pts[0][1], marker='+', c='black')

        peaks = self.vpcf_peaks[vpcf].loc[:, 'x_fit':'y_fit'
                                          ].to_numpy(dtype=float)
        peak_inds = np.array([np.argmin(norm(peaks - pt, axis=1))
                              for pt in pts])
        peak_coords = np.array(peaks[peak_inds, :])

        axs[0].scatter(
            peak_coords[:, 0],
            peak_coords[:, 1],
            marker='+',
            c='green',
            s=100,
            zorder=10,
        )
        plt.draw()
        # plt.close('all')

        vects = (
            peak_coords - self.vpcfs['metadata']['origin']
        ) * self.vpcfs['metadata']['pixel_size']

        if self.vpcfs['metadata']['affine_transform']:
            # Find transformation matrix from image pixels to CIF lattice
            # params in Angstroms
            t_mat = (
                self.dir_struct_matrix/norm(self.dir_struct_matrix, axis=1)**2
            ) @ self.a_2d

            vect_im = vects @ np.linalg.inv(t_mat)
        else:
            pixel_size = self.pixel_size_cal
            if pixel_size is None:
                pixel_size = self.pixel_size_est
            vect_im = vects / pixel_size

        return vect_im, sub1, sub2

    def plot_distances_from_vpcf_peak(
        self,
        vpcf,
        r=0.5,
        locate_by_fit_or_ref='ref',
        plot_fit_or_ref='fit',
        number_of_peaks_to_pick=1,
        dist_along_vector=None,
        deviation_or_absolute='deviation',
        plot_equ_ellip=True,
        return_nn_list=False,
        xlim=None,
        ylim=None,
        scalebar_len=1,
        outlier_disp_cutoff=None
    ):
        """Finds vector from vPCF and plots the inter-column distances
        in the image that correspond to that vector.

        Parameters
        ----------
        vpcf : str
            One of the vPCFs calculated previously.

        r : scalar
            Radius error tolerance for finding matching distances.
            Default: 0.5

        fit_or_ref : 'fit' or 'ref'
            Whether to use the reference lattice or the fitted atom column
            positions for determining matching distances. (Plotted distances
            will be based on fitted positions, regardless.) 'ref' will be more
            reliable for finding all the peaks in a sublattice, while 'fit'
            is appropriate for finding distances contributing to one part
            of a split peak.
            Default: 'ref'.

        number_of_peaks_to_pick : int
            The number of peaks that will be chosen in the vPCF. More than 2
            will probably produce a plot that is too complicated.
            Default: 1.

        dist_along_vector : None or array-like of shape (2,) or (n,2)
            If an array is passed, the distances plotted will be the inter atom
            column vector components along this vector. For example, if it is
            desired to show the projected atom column seperation along a
            certain basis vector, pass that basis vector here. If more than
            one vector is to be picked, this argument must be of shape (n,2)
            for 'n' picked vectors.
            Default: None

        deviation_or_absolute : 'deviation' or 'absolute'
            Whether to use distance deviations from reference lattice or
            absolute distances for the bond distance colormap.

        return_nn_list : bool
            Whether to return the list of DataFrames of near neighbor
            information.
            Default: False

        outlier_disp_cutoff : None or scalar
            Criteria for removing outlier atomic column fits from the
            plot (in Angstroms). The maximum difference between the fitted
            position and the corresponding reference lattice point. All
            positions with greater errors than this value will be removed.
            If None, all column positions will be plotted.
            Default None.

        Returns
        -------
        NN_list : list of DataFrames (optional)
            The list of DataFrames with the near neighbor information. Only
            returned if return_nn_list=True. Otherwise returns None.
            Dataframes are ordered according to the order vPCF peaks are
            chosen.

        """

        if not np.isin(vpcf, list(self.vpcfs.keys())).item():
            vpcfs_list = [v for v in list(self.vpcfs.keys())
                          if v != 'metadata']
            raise Exception(
                'Specified vpcf does not exist. Available vpcfs: \n' +
                f'{vpcfs_list}')

        vects, str1, str2 = self.get_vector_from_vpcf(
            vpcf,
            number_of_peaks_to_pick=number_of_peaks_to_pick,
            plot_equ_ellip=plot_equ_ellip,
        )

        # print(vects)

        if dist_along_vector is not None:
            dist_along_vector = dist_along_vector.reshape(-1, 2)
            if dist_along_vector.shape[0] != number_of_peaks_to_pick:
                raise Exception(
                    'argument "dist_along_vector" must a vector for each peak '
                    + 'that will be picked. That is, if '
                    + 'number_of_peaks_to_pick=n, "dist_along_vector" must '
                    + 'have n vectors.'
                )

        pixel_size = self.pixel_size_cal
        if pixel_size is None:
            pixel_size = self.pixel_size_est

        filter_by = self.vpcfs['metadata']['filter_by']

        if ((outlier_disp_cutoff is not None) &
                (outlier_disp_cutoff is not np.inf)):
            # if (type(outlier_disp_cutoff) == float or
            #         type(outlier_disp_cutoff) == int):
            outlier_disp_cutoff /= pixel_size * 100

            at_cols = self.at_cols[norm(
                self.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
                - self.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float),
                axis=1)
                < outlier_disp_cutoff].copy()

        else:
            at_cols = self.at_cols.copy()

        sub1 = at_cols[at_cols.loc[:, filter_by] == str1
                       ].loc[:, 'x_ref':'y_fit'].reset_index(drop=True)
        sub2 = at_cols[at_cols.loc[:, filter_by] == str2
                       ].loc[:, 'x_ref':'y_fit'].reset_index(drop=True)

        if locate_by_fit_or_ref == 'ref':
            xy1 = sub1.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float)
            xy2 = sub2.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float)
        elif locate_by_fit_or_ref == 'fit':
            xy1 = sub1.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)
            xy2 = sub2.loc[:, 'x_fit':'y_fit'].to_numpy(dtype=float)

        NN = []

        for i, vect in enumerate(vects):
            NN += [pd.DataFrame(
                columns=['x_fit', 'y_fit', 'dx', 'dy', 'dl', 'dl_ref'])]

            if dist_along_vector is None:
                ref_vect = np.identity(2)

            else:
                ref_vect = np.diag(
                    dist_along_vector[i]/norm(dist_along_vector[i])
                )

            for ind1, xy in enumerate(tqdm(xy1)):
                vect_err = norm(xy2 - xy - vect, axis=1)
                min_ = np.min(vect_err)

                if min_ <= r / pixel_size:
                    ind2 = np.argmin(vect_err)

                    if plot_fit_or_ref == 'fit':
                        row = np.array([
                            sub1.loc[ind1, 'x_fit'],
                            sub1.loc[ind1, 'y_fit'],
                            (sub2.loc[ind2, 'x_fit'] - sub1.loc[ind1, 'x_fit']
                             ) * ref_vect[0, 0],
                            (sub2.loc[ind2, 'y_fit'] - sub1.loc[ind1, 'y_fit']
                             ) * ref_vect[1, 1],
                            norm((sub2.loc[ind2, 'x_fit':'y_fit'] -
                                  sub1.loc[ind1, 'x_fit':'y_fit']) @ ref_vect),
                            norm((sub2.loc[ind2, 'x_ref':'y_ref'] -
                                  sub1.loc[ind1, 'x_ref':'y_ref']) @ ref_vect)
                        ])
                    elif plot_fit_or_ref == 'ref':
                        row = np.array([
                            sub1.loc[ind1, 'x_ref'],
                            sub1.loc[ind1, 'y_ref'],
                            (sub2.loc[ind2, 'x_ref'] - sub1.loc[ind1, 'x_ref']
                             ) * ref_vect[0, 0],
                            (sub2.loc[ind2, 'y_ref'] - sub1.loc[ind1, 'y_ref']
                             ) * ref_vect[1, 1],
                            norm((sub2.loc[ind2, 'x_ref':'y_ref'] -
                                  sub1.loc[ind1, 'x_ref':'y_ref']) @ ref_vect),
                            norm((sub2.loc[ind2, 'x_ref':'y_ref'] -
                                  sub1.loc[ind1, 'x_ref':'y_ref']) @ ref_vect)
                        ])

                    NN[-1].loc[len(NN[-1].index), :] = row

        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')

        cscale = []

        if deviation_or_absolute == 'absolute':
            for i in range(len(NN)):
                cscale += [NN[i].loc[:, 'dl'].to_numpy(dtype=float)
                           * pixel_size]

            label = r'Distance ($\AA$)'
            min_ = np.floor(np.min(np.concatenate(cscale)) / 0.01) * 0.01
            max_ = np.ceil(np.max(np.concatenate(cscale)) / 0.01) * 0.01

        elif deviation_or_absolute == 'deviation':
            for i in range(len(NN)):
                cscale += [(NN[i].loc[:, 'dl'].to_numpy(dtype=float)
                            - NN[i].loc[:, 'dl_ref'].to_numpy(dtype=float))
                           * pixel_size * 100]

            label = r'$\Delta$ Distance (pm)'
            max_ = np.ceil(np.max(np.abs(np.concatenate(cscale))))
            min_ = -max_

        else:
            raise Exception(
                'argument "deviation_or_absolute" must be either: "absolute" '
                'or "deviation".'
            )

        for i, nn in enumerate(NN):

            ax.quiver(
                nn.loc[:, 'x_fit'].to_numpy(dtype=float).flatten(),
                nn.loc[:, 'y_fit'].to_numpy(dtype=float).flatten(),
                nn.loc[:, 'dx'].to_numpy(dtype=float).flatten(),
                nn.loc[:, 'dy'].to_numpy(dtype=float).flatten(),
                cscale[i],
                norm=Normalize(vmin=min_, vmax=max_),
                angles='xy',
                scale_units='xy',
                scale=1,
                headaxislength=0,
                headwidth=0,
                headlength=0,
                cmap='bwr',
                width=0.003
            )

        ax.set_xticks([])
        ax.set_yticks([])

        if xlim:
            xlim = np.sort(xlim)
            ax.set_xlim(xlim)
        if ylim:
            ylim = np.flip(np.sort(ylim))
            ax.set_ylim(ylim)

        cbar = fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=min_, vmax=max_), cmap='bwr'),
            ax=ax,
            orientation='vertical',
            shrink=0.3,
            # fraction=0.25,
            aspect=12,
            ticks=[min_, (max_ + min_)/2, max_]
        )

        cbar.set_label(label=label, fontsize=10)

        if self.pixel_size_cal is not None:
            pixel_size = self.pixel_size_cal
        else:
            pixel_size = self.pixel_size_est
        scalebar = ScaleBar(
            pixel_size/10,
            'nm',
            font_properties={'size': 12},
            pad=0.3,
            border_pad=0.6,
            box_color='dimgrey',
            height_fraction=0.02,
            color='white',
            location='lower right',
            fixed_value=scalebar_len,
        )

        ax.add_artist(scalebar)

        if return_nn_list:
            return NN, fig

        else:
            return fig
