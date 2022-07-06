"""SingleOrigin is a module for atomic column position finding intended for 
    high probe_fwhm scanning transmission electron microscope images.
    Copyright (C) 2022  Stephen D. Funni

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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle as Circle
from matplotlib.legend_handler import HandlerPatch
from matplotlib import colors as colors
from matplotlib.patches import Wedge
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib_scalebar.scalebar import ScaleBar

from scipy.optimize import minimize
from scipy import ndimage, fftpack
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)

import psutil
from joblib import Parallel, delayed
from tqdm import tqdm

from SingleOrigin.utils import (image_norm, img_ellip_param, gaussian_2d,
                                LoG_2d, fit_gaussian2D, watershed_segment)        
#%%
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
    image : 2D array_like
        The STEM image.
    unitcell : UnitCell class object
        Instance of the UnitCell() class with project_uc_2d() method applied.
    probe_fwhm : float
        The probe_fwhm of the microscope and imaging mode in Angstroms.
        Default: 0.8.
    origin_atom_column : int
        The DataFrame row index (in self.unitcell_2D) of the atom column 
        that is later picked by the user to register the reference lattice. 
        If None, the closest atom column to the unit cell origin is 
        automatically chosen.
        Default: None.
    xlim, ylim : list-like, of shape (2,)
        Image area to be an analyzed. If "None" entire input image is used. 
        Default: None
    
    Attributes
    ----------
    image : The input STEM image.
    probe_fwhm : Approximate full width half maximum of the imaging probe in 
        Angstroms. Used for auto-calculation of bluring parameters for 
        watershed segmentation.
    basis_offset_frac : The input basis_offset_frac, fractional coordinates
    basis_offset_pix : The basis offset in image pixel coordinates.
    h, w : The height and width of the image.
    at_cols : DataFrame containing the he reference lattice and fitting data 
        (including positions) for the atomic columns in the image.
    at_cols_uncropped : The reference lattice atom columns before removing
        positions close to the image edges (as defined by the "buffer" arg in 
        the "fit_atom_columns()" method).
    unitcell_2D : DataFrame containing the projected crystallographic unit 
        cell atom column positions.
    a_2d : The matrix of real space basis vectors in the Cartesian reference 
        frame with units of Angstroms.
    x0, y0 : The image coordinates of the origin of the reference lattice 
        (in pixels).
    fit_masks : The last set of masks used for fitting  atom columns. Has
        the same shape as image.
    a1_star, a2_star : The reciprocal basis vectors in FFT pixel coordinates.
    a1, a2 : The real space basis vectors in image pixel coordinates.
    trans_mat : The transformation matrix from fractional to image pixel
        coordinates.
    pixel_size : The estimated pixel size using the reference lattice basis
        vectors and lattice parameter values from the .cif file.
    
    Methods
    -------
    fft_get_basis_vect(self, a1_order=1, a2_order=1, sigma=5): 
        Find basis vectors for the image from the FFT. 
    define_reference_lattice(self, LoG_sigma=None): 
        Registers a reference lattice to the image.
    fit_atom_columns(self, buffer=0,local_thresh_factor=1, 
                     diff_filter='auto', grouping_filter='auto',
                     filter_by='elem', sites_to_fit='all',
                     watershed_line=True, use_LoG_fitting=False):
        Algorithm for fitting 2D Gaussians to HR STEM image.
    refine_reference_lattice(self, filter_by='elem', 
                             sites_to_use='all', outlier_disp_cutoff=None):
        Refines the reference lattice on fitted column positions.
    rotate_image_and_data(align_dir='horizontal', align_basis='a1'):
        Rotates the image and data to align a basis vector to image edge.
    select_origin():
        Select origin for the reference lattice. Used by 
        define_reference_lattice() method.
    plot_fitting_residuals(self):
        Plots image residuals from the atomic column fitting.
    rotate_image_and_data(self, align_dir='horizontal', align_basis='a1'):
        Rotates the image and data to align a basis vector to image edge.
    plot_atom_column_positions(self, filter_by='elem', sites_to_fit='all',
                               fit_or_ref='fit', outlier_disp_cutoff=None,
                               plot_masked_image=False):
        Plot fitted or reference atom colum positions.
    plot_disp_vects(self, filter_by='elem', sites_to_plot='all', 
                    titles=None, x_lim=None, y_lim=None,
                    scalebar=True, scalebar_len_nm=2,
                    arrow_scale_factor = 1,
                    outlier_disp_cutoff = None, max_colorwheel_range_pm=None,
                    plot_fit_points=False, plot_ref_points=False):
        Plot dislacement vectors between reference and fitted atom colum 
        positions.
        
    """
    
    def __init__(self, image, unitcell, probe_fwhm = 0.8, 
                 origin_atom_column=None, xlim=None, ylim=None):
        
        h, w = image.shape
        m = min(h,w)
        U = min(1000, int(m/2))
        crop_dim = 2*U
        image_square = image[int(h/2)-U : int(h/2)+U,
                          int(w/2)-U : int(w/2)+U]
        
        hann = np.outer(np.hanning(crop_dim),np.hanning(crop_dim))
        fft = fftpack.fft2(image_square*hann)
        fft = (abs(fftpack.fftshift(fft)))
        fft = image_norm(fft)
        self.fft = fft
        
        if xlim != None:
            image = image[:, xlim[0]:xlim[1]]
        if ylim != None:
            image = image[ylim[0]:ylim[1], :]
        self.image = image
        
        self.probe_fwhm = probe_fwhm
        self.h, self.w = self.image.shape
        self.unitcell_2D = unitcell.at_cols
        self.a_2d = unitcell.a_2d
        self.at_cols = pd.DataFrame()
        self.at_cols_uncropped = pd.DataFrame()
        self.x0, self.y0 = np.nan, np.nan
        self.fit_masks = np.zeros(image.shape)
        self.a1, self.a2 = None, None
        self.a1_star, self.a2_star = None, None
        self.trans_mat = None
        self.pixel_size = None
        self.residuals = None
        
        if origin_atom_column == None:
            origin_atom_column = np.argmin(np.linalg.norm(
                self.unitcell_2D.loc[:, 'x':'y'], axis=1))
        
        self.basis_offset_frac = self.unitcell_2D.loc[
            origin_atom_column, 'u':'v'].to_numpy()
        self.basis_offset_pix = self.unitcell_2D.loc[
            origin_atom_column, 'x':'y'].to_numpy()
        self.use_LoG_fitting = False
        
    
    def fft_get_basis_vect(self, a1_order=1, a2_order=1, sigma=5):
        """Measure crystal basis vectors from the image FFT.
        
        Finds peaks in the image FFT and displays for graphical picking. 
        After the user chooses two peaks for the reciprocal basis vectors, 
        finds peaks related by approximate vector additions, refines a
        reciprocal basis from these positions and transforms the reciprocal 
        basis into a real space basis.
        
        Parameters
        ----------
        a1_order, a2_order : ints
            Order of first peaks visible in the FFT along the two reciprocal 
            lattice basis vector directions. If some FFT peaks are weak or 
            absent (such as forbidden reflections), specify the order of the 
            first peak that is clearly visible.
        sigma : int or float
            The Laplacian of Gaussian sigma value to use for sharpening of the 
            FFT peaks. Usually a value between 2 and 10 will work well.
             
        Returns
        -------
        None.
            
        """
        
        '''Find rough reciprocal lattice'''
        
        h, w = self.fft.shape
        m = min(h,w)
        U = int(m/2)
        crop_dim = 2*U
                
        fft_der = image_norm(-gaussian_laplace(self.fft, sigma))
        masks, num_masks, slices, spots = watershed_segment(fft_der)
        spots.loc[:, 'stdev'] = ndimage.standard_deviation(fft_der, masks, 
                                            index=np.arange(1, num_masks+1))
        spots_ = spots[(spots.loc[:, 'stdev'] > 0.003)].reset_index(drop=True)
        xy = spots_.loc[:,'x':'y'].to_numpy()
        
        origin = np.array([U, U])
        
        recip_vects = np.linalg.norm(xy - origin, axis=1)
        min_recip_vect = np.min(recip_vects[recip_vects>0])
        window = min(min_recip_vect*10, U)
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('''Pick reciprocal basis vectors''',
                  fontdict = {'color' : 'red'})
        ax.set_ylim(bottom = U+window, top = U-window)
        ax.set_xlim(left = U-window, right = U+window)
        ax.imshow((self.fft)**(0.1), cmap='gray')
        ax.scatter(xy[:,0], xy[:,1], c='red', s=8)
        ax.scatter(origin[0], origin[1], c='white', s=16)
        ax.set_xticks([])
        ax.set_yticks([])
        
        basis_picks_xy = np.array(plt.ginput(2, timeout=15))
        
        vects = np.array([xy - i for i in basis_picks_xy])
        inds = np.argmin(np.linalg.norm(vects, axis=2), axis=1)
        basis_picks_xy = xy[inds, :]
        
        print('done selecting', '\n')
        
        '''Generate reference lattice and find corresponding peak regions'''
        a1_star = (basis_picks_xy[0, :] - origin) / a1_order
        a2_star = (basis_picks_xy[1, :] - origin) / a2_order
        
        trans_mat_rec = np.array([a1_star, a2_star])
        
        recip_latt_indices = np.array([[i,j] for i in range(-5,6) 
                                       for j in range(-5,6)])
        xy_ref = recip_latt_indices @ trans_mat_rec + origin
        
        vects = np.array([xy - xy_ for xy_ in xy_ref])
        inds = np.argmin(np.linalg.norm(vects, axis=2), axis=1)
        
        df = {'h': recip_latt_indices[:, 0],
              'k': recip_latt_indices[:, 1],
              'x_ref': xy_ref[:, 0],
              'y_ref': xy_ref[:, 1],
              'x_fit': [xy[ind, 0] for ind in inds],
              'y_fit': [xy[ind, 1] for ind in inds],
              'mask_ind': inds,
              'stdev': [spots_.loc[:, 'stdev'][ind] 
                        for ind in inds]}
        
        recip_latt = pd.DataFrame(df)

        recip_latt = recip_latt[np.linalg.norm(
            recip_latt.loc[:, 'x_fit':'y_fit'].to_numpy()
            -recip_latt.loc[:, 'x_ref':'y_ref'].to_numpy(), axis=1) 
            < 0.25*np.min(np.linalg.norm(trans_mat_rec, axis=1))
            ].reset_index(drop=True)
        
        def disp_vect_sum_squares(p0, M_star, xy, origin):
            
            trans_mat_rec_ = p0.reshape((2,2))
            err_xy = xy - M_star @ trans_mat_rec_ - origin
            sum_sq = np.sum(err_xy**2)
            return sum_sq
        
        M_star = recip_latt.loc[:, 'h':'k'].to_numpy()
        xy = recip_latt.loc[:, 'x_fit':'y_fit'].to_numpy()
        
        p0 = trans_mat_rec.flatten()
        
        params = minimize(disp_vect_sum_squares, p0, 
                          args=(M_star, xy, origin)).x
        
        a1_star = params[:2] 
        a2_star = params[2:] 
        
        trans_mat_rec = params.reshape((2,2))
        trans_mat = np.linalg.inv(trans_mat_rec.T) * crop_dim
        
        recip_latt.loc[:, 'x_ref':'y_ref'] = (
            recip_latt.loc[:, 'h':'k'].to_numpy() @ trans_mat_rec + origin)
        plt.close('all')
        
        fig2, ax = plt.subplots(figsize=(10,10))
        ax.imshow((self.fft)**(0.1), cmap = 'gray')
        ax.scatter(recip_latt.loc[:, 'x_fit'].to_numpy(), 
                    recip_latt.loc[:, 'y_fit'].to_numpy())
        ax.arrow(origin[0], origin[1], a1_star[0], a1_star[1],
                    fc='red', ec='red', width=0.1, 
                    length_includes_head=True,
                    head_width=2, head_length=3)
        ax.arrow(origin[0], origin[1], a2_star[0], a2_star[1],
                    fc='green', ec='green', width=0.1, 
                    length_includes_head=True,
                    head_width=2, head_length=3)

        ax.set_ylim(bottom = U+window, top = U-window)
        ax.set_xlim(left = U-window, right = U+window)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Reciprocal Lattice Fit')
        
        self.a1_star = a1_star
        self.a2_star = a2_star
        self.a1 = trans_mat[0,:]
        self.a2 = trans_mat[1,:]
        self.trans_mat = trans_mat
        self.basis_offset_pix = self.basis_offset_frac @ self.trans_mat
        self.pixel_size = np.average([np.linalg.norm(self.a_2d[0,:])
                                      /np.linalg.norm(self.a1),
                                      np.linalg.norm(self.a_2d[1,:])
                                      /np.linalg.norm(self.a2)])
        self.recip_latt = recip_latt
        
        
    def select_origin(self, crop_factor=4):
        """Select origin for the reference lattice.
        
        User chooses appropriate atomic column to establish the reference 
        lattice origin. Used by the define_reference_lattice() method.
        
        Parameters
        ----------
        None.
             
        Returns
        -------
        None.
            
        """
        
        if 'LatticeSite' in list(self.unitcell_2D.columns):
            lab = 'LatticeSite'
        else:
            lab = 'elem'
        
        [h, w] = [self.h, self.w]
        
        crop_view = (np.max(np.abs(self.trans_mat)) * crop_factor)
        fig, ax = plt.subplots(figsize=(10,10))
        message=('Pick an atom column of the reference atom column type'
                 +' (white outlined position)')
        ax.set_title(str(message), fontdict={'color':'red'}, wrap=True)
        ax.set_ylim(bottom = h/2+crop_view, top = h/2-crop_view)
        ax.set_xlim(left = w/2-crop_view, right = w/2+crop_view)
        ax.imshow(self.image, cmap='gray')
        
        self.unitcell_2D['x_ref'] = ''
        self.unitcell_2D['y_ref'] = ''
        self.unitcell_2D.loc[:, 'x_ref': 'y_ref'] = self.unitcell_2D.loc[
            :, 'u':'v'].to_numpy() @ self.trans_mat
        
        uc_w = np.max(np.abs(self.trans_mat[:,0]))
        uc_h = np.max(np.abs(self.trans_mat[:,1]))
        rect_params = [w/2-crop_view + 0.1*uc_w, 
                       h/2+crop_view - 0.2*uc_h, 
                       np.max(np.abs(self.trans_mat[:,0]))*1.1, 
                       -np.max(np.abs(self.trans_mat[:,1]))*1.1]
        
        x_mean = np.mean(self.trans_mat[:,0])
        y_mean = np.mean(self.trans_mat[:,1])
        
        x0 = rect_params[2]/2 - x_mean + rect_params[0]
        y0 = rect_params[3]/2 - y_mean + rect_params[1]
        
        site_list = list(set(self.unitcell_2D[lab]))
        site_list.sort()
        
        color_code = {k:v for v, k in  enumerate(
            np.sort(self.unitcell_2D.loc[:, lab].unique()))}
        
        color_list = [color_code[site] for site in 
                      self.unitcell_2D.loc[:, lab]]
        
        box = Rectangle((rect_params[0], rect_params[1]), 
                        rect_params[2], rect_params[3], 
                        edgecolor='black', facecolor='grey', 
                        alpha = 1)
        ax.add_patch(box)
        
        ax.scatter(self.unitcell_2D.loc[:,'x_ref'].to_numpy() + x0, 
                    self.unitcell_2D.loc[:,'y_ref'].to_numpy() + y0,
                    c=color_list, cmap='RdYlGn', s=10, zorder=10)
        
        ax.arrow(x0, y0, self.a1[0], self.a1[1],
                      fc='black', ec='black', width=0.1, 
                      length_includes_head=True,
                      head_width=2, head_length=3, zorder=8)
        ax.arrow(x0, y0, self.a2[0], self.a2[1],
                      fc='black', ec='black', width=0.1, 
                      length_includes_head=True,
                      head_width=2, head_length=3, zorder=8)
        
        ref_atom = self.basis_offset_pix + np.array([x0, y0])
        ax.scatter(ref_atom[0], ref_atom[1], c='white', s=70, zorder=9)
        
        cmap = plt.cm.RdYlGn
        color_index = [Circle((30, 7), 3, color=cmap(c)) 
                       for c in np.linspace(0,1, num=len(color_code))]
        
        def make_legend_circle(legend, orig_handle,
                               xdescent, ydescent,
                               width, height, fontsize):
            p = orig_handle
            return p
        ax.legend(handles = color_index,
                  labels = list(color_code.keys()),
                  handler_map={Circle : HandlerPatch(patch_func=
                                                     make_legend_circle),},
                  fontsize=20, loc='lower left', bbox_to_anchor=[1.02, 0],
                  facecolor='grey')
        
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        pt = plt.ginput(1, timeout=30)
        
        plt.close('all')
        return pt[0]
        
    
    def define_reference_lattice(self, LoG_sigma=None, crop_factor=4):
        
        """Register reference lattice to image.
        
        User chooses appropriate atomic column to establish the reference 
        lattice origin. Rough scaling and orientation are initially defined
        as derived from fft_get_basis_vect() method and then pre-refined by
        local peak detection.
        
        Parameters
        ----------
        LoG_sigma : int or float
            The Laplacian of Gaussian sigma value to use for peak sharpening.
            If None, calculated by: pixel_size / probe_fwhm * 0.5.
            Default None.
             
        Returns
        -------
        None.
            
        """
        
               
        if 'LatticeSite' in list(self.unitcell_2D.columns):
            lab = 'LatticeSite'
        else:
            lab = 'elem'
        
        self.pixel_size = np.average([np.linalg.norm(self.a_2d[0,:])
                                      /np.linalg.norm(self.a1),
                                      np.linalg.norm(self.a_2d[1,:])
                                      /np.linalg.norm(self.a2)])
                    
        if ((LoG_sigma == None) 
            & ((type(self.probe_fwhm) == float) 
               | (type(self.probe_fwhm) == int))):
            LoG_sigma = self.probe_fwhm / self.pixel_size * 0.5 
        
        (x0, y0) = self.select_origin(crop_factor=crop_factor)

        print('pick coordinates:', np.around([x0, y0], decimals=2), '\n')
        
        filt = image_norm(-gaussian_laplace(self.image, LoG_sigma))
        neighborhood = np.ones((7,7))
        local_max = np.fliplr(np.argwhere(maximum_filter(filt,
                                    footprint=neighborhood)==filt))

        [x0, y0] = local_max[np.argmin(np.linalg.norm(local_max 
                                                    - [x0, y0], axis=1))]
        
        print('detected peak coordinates:', [x0, y0], '\n')
        
        self.x0 = x0 - self.basis_offset_pix[0]
        self.y0 = y0 - self.basis_offset_pix[1]
            
        a1 = self.a1
        a2 = self.a2
        x0 = self.x0
        y0 = self.y0
        h = self.h
        w = self.w
        
        print('Creating reference lattice...')
        def vect_angle(a, b):
            theta = np.arccos(a @ b.T/(np.linalg.norm(a) 
                                       * np.linalg.norm(b)))
            return theta
        
        d = [np.array([-x0, -y0]),
             np.array([-x0, h -y0]),
             np.array([w - x0, h - y0]),
             np.array([w - x0, -y0])]
        
        a1p = np.argmin([(vect_angle(a1, d[i])) for i,_ in enumerate(d)])
        a1n = np.argmin([(vect_angle(-a1, d[i])) for i,_ in enumerate(d)])
        a2p = np.argmin([(vect_angle(a2, d[i])) for i,_ in enumerate(d)])
        a2n = np.argmin([(vect_angle(-a2, d[i])) for i,_ in enumerate(d)])
        
        a1_start =  int(np.linalg.norm(d[a1n])**2 / (a1 @ d[a1n].T)) - 1
        a1_stop = int(np.linalg.norm(d[a1p])**2 / (a1 @ d[a1p].T)) + 2
        a2_start = int(np.linalg.norm(d[a2n])**2 / (a2 @ d[a2n].T)) - 1
        a2_stop = int(np.linalg.norm(d[a2p])**2 / (a2 @ d[a2p].T)) + 2
        
        latt_cells = np.array([[i, j] for i in range(a1_start, a1_stop)
                               for j in range(a2_start, a2_stop)
                               for _ in range(self.unitcell_2D.shape[0])])
        
        at_cols = pd.concat([self.unitcell_2D] 
                            * int(latt_cells.shape[0]
                                  /self.unitcell_2D.shape[0]),
                            ignore_index=True)
        
        at_cols.loc[:,'u':'v'] += latt_cells
        
        at_cols.loc[:,'x_ref':'y_ref'] = (at_cols.loc[:,'u':'v'].to_numpy() 
                                           @ self.trans_mat 
                                           + np.array([self.x0, self.y0]))
        
        at_cols = at_cols[((at_cols.x_ref >= 0) &
                           (at_cols.x_ref <= w ) &
                           (at_cols.y_ref >= 0) &
                           (at_cols.y_ref <= h))]
        
        at_cols.reset_index(drop=True, inplace=True)
        empty = pd.DataFrame(index=np.arange(0,at_cols.shape[0]),
                              columns = ['x_fit', 'y_fit', 'sig_1', 'sig_2',
                                         'theta', 'peak_int', 'bkgd_int', 
                                         'total_col_int'])
        
        at_cols = pd.concat([at_cols, empty], axis=1)
        self.at_cols = pd.DataFrame(columns = at_cols.columns)
        
        ch_list = np.sort(at_cols.loc[:,lab].unique()).tolist()
        ch_list = {k: v for v, k in enumerate(ch_list)}
        channels = np.array([ch_list[site] for site in 
                             at_cols.loc[:, lab]])
        at_cols.loc[:, 'channel'] = channels
        
        '''Refine reference lattice on watershed mask CoMs'''
        print('Performing rough reference lattice refinement...')
        img_LoG = image_norm(-gaussian_laplace(self.image, LoG_sigma))
        poss = self.unitcell_2D.loc[:, 'x_ref':'y_ref'].to_numpy()
        dists = np.linalg.norm(np.array([poss - pos for pos in poss]), axis=2)
        min_dist = np.amin(dists, initial=np.inf, where=dists>0)
        
        masks, num_masks, slices, peaks = watershed_segment(
            img_LoG,  buffer=0, min_dist=min_dist)
        
        coords = peaks.loc[:, 'x':'y'].to_numpy()
        print('prior basis vectors:', 
              f'[[{self.trans_mat[0,0]:.{4}f} {self.trans_mat[0,1]:.{4}f}]',
              f' [{self.trans_mat[1,0]:.{4}f} {self.trans_mat[1,1]:.{4}f}]]'
              +'\n', sep='\n')
        
        init_inc = int(np.min(np.max(np.abs(at_cols.loc[:, 'u' :'v']),
                                            axis=0))/10)
        # print(init_inc)
        
        if init_inc < 3: init_inc = 3
        # elif init_inc < 5: init_inc = 5
        origin_ind = at_cols[(at_cols.u == self.basis_offset_frac[0]) &
                             (at_cols.v == self.basis_offset_frac[1])].index[0]
        at_cols_orig_type = at_cols[(at_cols.x == at_cols.at[origin_ind,'x']) &
                                    (at_cols.y == at_cols.at[origin_ind,'y'])]
        
        for lim in [init_inc * i for i in [1,2,4]]:
            print(rf'refining to +/-{lim} unit cells')
            filtered = at_cols_orig_type[(np.abs(at_cols_orig_type.u) <= lim) & 
                                         (np.abs(at_cols_orig_type.v) <= lim)]
            
            M = filtered.loc[:, 'u':'v'].to_numpy()
            xy_ref = filtered.loc[:, 'x_ref':'y_ref'].to_numpy()
            
            vects = np.array([coords - xy for xy in xy_ref])
            
            inds = np.argmin(np.linalg.norm(vects, axis=2), axis=1)
            xy = np.array([coords[ind] for ind in inds])
            
            def disp_vect_sum_squares(p0, M, xy):
                
                trans_mat = p0[:4].reshape((2,2))
                origin = p0[4:]
                
                R = np.linalg.norm(xy - M @ trans_mat - origin, axis=1)
                sum_sq = (R @ R.T).item()
                return sum_sq
            
            p0 = np.concatenate((self.trans_mat.flatten(),
                                  np.array([self.x0, self.y0])))
            
            params = minimize(disp_vect_sum_squares, p0, args=(M, xy)).x
            
            self.a1 = params[:2]
            self.a2 = params[2:4]
            
            self.trans_mat = params[:4].reshape((2,2))
            
            self.x0 = params[4]
            self.y0 = params[5]
            
            at_cols.loc[:, 'x_ref':'y_ref'] = (
                at_cols.loc[:, 'u':'v'].to_numpy() @ self.trans_mat
                + np.array([self.x0, self.y0])
                )
            
            at_cols = at_cols[((at_cols.x_ref >= 5) &
                               (at_cols.x_ref <= w - 5) &
                               (at_cols.y_ref >= 5) &
                               (at_cols.y_ref <= h - 5))]
            
        at_cols.loc[:, 'x_ref':'y_ref'] = (
            at_cols.loc[:, 'u':'v'].to_numpy() @ self.trans_mat
            + np.array([self.x0, self.y0])
            )
        
        at_cols = at_cols[((at_cols.x_ref >= 5) &
                           (at_cols.x_ref <= w - 5) &
                           (at_cols.y_ref >= 5) &
                           (at_cols.y_ref <= h - 5))]
            
        self.at_cols_uncropped = copy.deepcopy(at_cols)
        
        
        print('refined basis vectors:', 
              f'[[{self.trans_mat[0,0]:.{4}f} {self.trans_mat[0,1]:.{4}f}]',
              f' [{self.trans_mat[1,0]:.{4}f} {self.trans_mat[1,1]:.{4}f}]]'
              +'\n', sep='\n')
                
        
    def fit_atom_columns(self, buffer=0,local_thresh_factor=0.95, 
                         diff_filter='auto', grouping_filter='auto',
                         filter_by='elem', sites_to_fit='all',
                         watershed_line=True, use_LoG_fitting=False,
                         parallelize=True):
        """Algorithm for fitting 2D Gaussians to HR STEM image.
        
        Uses Laplacian of Gaussian filter to isolate each peak by the 
        Watershed method. Gaussian filter blurs image to group closely spaced 
        peaks for simultaneous fitting. If simultaneous fitting is not 
        desired, set grouping_filter = None. Requires reference lattice or
        initial guesses for all atom columns in the image (i.e. 
        self.at_cols_uncropped must have values in 'x_ref' and 'y_ref' 
        columns). This can be achieved by running self.get_reference_lattice. 
        Stores fitting parameters for each atom column in self.at_cols.
        
        Parameters
        ----------
        buffer : int
            Distance defining the image border used to ignore atom columns 
            whose fits my be questionable.
        local_thresh_factor : float
            Removes background from each segmented region by thresholding. 
            Threshold value determined by finding the maximum value of edge 
            pixels in the segmented region and multipling this value by the 
            local_thresh_factor value. The LoG-filtered image (with 
            sigma=diff_filter) is used for this calculation. Default 0.95.
        diff_filter : int or float
            The Laplacian of Gaussian sigma value to use for peak sharpening
            for defining peak regions via the Watershed segmentation method.
            Should be approximately pixel_size / probe_fwhm / 2. If 'auto',  
            calculated using self.pixel_size and self.probe_fwhm.
            Default 'auto'. 
        grouping_filter : int or float
            The Gaussian sigma value to use for peak grouping by blurring, 
            then creating image segment regions with watershed method. 
            Should be approximately pixel_size / probe_fwhm * 0.5. If 'auto', 
            calculated using self.pixel_size and self.probe_fwhm. If
            simultaneous fitting of close atom columns is not desired, set
            to None.
            Default: 'auto'.
        filter_by : str
            'at_cols' column to use for filtering to fit only a subset
            of the atom colums.
            Default 'elem'
        sites_to_fit : str ('all') or list of strings
            The criteria for the sites to fit, e.g. a list of the elements to 
            fit: ['Ba', 'Ti']
            Default 'all'
        watershed_line : bool
            Seperate segmented regions by one pixel. Default: True.
        use_LoG_fitting : bool
            Whether to use a Laplacian of Gaussian (LoG) function instead of a 
            standard 2D Gaussian function for peak fitting. LoG fitting should
            ONLY be used for dDPC images. All other image types, including 
            HAADF, ADF, BF, ABF and iDPC should be fitted with a standard 2D 
            Gaussian. The LoG funciton is round and so does not give peak 
            shape information. Additionally, its paramters cannot be 
            integrated to calculate the peak volume. Only interpretable 
            parameters are the peak position.
            Defalut: False.
        parallelize : bool
            Whether to use parallel CPU processing. Will use all available
            physical cores if set to True. If False, will use serial 
            processing.
            
        Returns
        -------
        None.
            
        """
        
        print('Creating atom column masks...')
        
        """Handle various settings configurations, prepare for masking 
        process, and check for exceptions"""
        t0 = time.time()
        self.buffer = buffer
        self.use_LoG_fitting = use_LoG_fitting
        use_Guass_for_LoG = False
        use_LoG_for_Gauss = False
        
        if self.at_cols.shape[0] == 0:
            at_cols = self.at_cols_uncropped.copy()
        else:
            at_cols = self.at_cols.copy()
            at_cols = pd.concat([at_cols, self.at_cols_uncropped.loc[
                [i for i in self.at_cols_uncropped.index.tolist()
                 if i not in at_cols.index.tolist()], 
                :]])
            
        if grouping_filter == 'auto': 
            if ((type(self.probe_fwhm) == float) 
                | (type(self.probe_fwhm) == int)):
        
                grouping_filter = self.probe_fwhm / self.pixel_size * 0.5
            
            else:
                raise Exception('"probe_fwhm" must be defined for the class '
                                + 'instance to enable "grouping_filter" '
                                + 'auto calculation.')
                
        elif grouping_filter == None: 
            use_LoG_for_Gauss = True
            
        elif ((type(grouping_filter) == float or type(grouping_filter) == int) and
              grouping_filter > 0):
            pass
            
        else:
            raise Exception('"diff_filter" must be "auto", a positive '
                            + 'float or int value, or None.')
            
        if diff_filter == 'auto':
            if ((type(self.probe_fwhm) == float) 
                | (type(self.probe_fwhm) == int)):
                
                diff_filter = self.probe_fwhm / self.pixel_size * 0.5
            
            else:
                raise Exception('"probe_fwhm" must be defined for the class '
                                + 'instance to enable "diff_filter" '
                                + 'auto calculation.')
        
        elif diff_filter == None:
            use_Guass_for_LoG = True
            
        elif ((type(diff_filter) == float or type(diff_filter) == int) and
              diff_filter > 0):
            pass
            
        else:
            raise Exception('"diff_filter" must be "auto", a positive '
                            + 'float or int value, or None.')
        
        if use_Guass_for_LoG==True and use_LoG_for_Gauss==True:
            img_gauss = self.image
            img_LoG = self.image
            print('Unfiltered image being used for all masking. This is not '
                  + 'recommended. Check results carefully.')
        
        else:
            if use_Guass_for_LoG:
                img_LoG = image_norm(gaussian_filter(self.image, 
                                                       grouping_filter, 
                                                       truncate=4))
                
            else:
                img_LoG = image_norm(-gaussian_laplace(self.image, 
                                                       diff_filter, 
                                                       truncate=4))
            if use_LoG_for_Gauss:
                img_gauss = image_norm(-gaussian_laplace(self.image, 
                                                         diff_filter, 
                                                         truncate=4))
            else:
                img_gauss = image_norm(gaussian_filter(self.image, 
                                                         grouping_filter, 
                                                         truncate=4))
                
        if sites_to_fit != 'all':
            at_cols = at_cols[at_cols.loc[:, filter_by].isin(sites_to_fit)]
            
        """Find minimum distance (in pixels) between atom columns for peak
        detection neighborhood"""
        unit_cell_uv = self.unitcell_2D.loc[:, 'u':'v'].to_numpy()
        unit_cell_xy = np.concatenate(([unit_cell_uv + [i,j] 
                                        for i in range(-1, 2)
                                        for j in range(-1, 2)])
                                      ) @ self.trans_mat
        dists = np.linalg.norm(np.array([unit_cell_xy - pos 
                                         for pos in unit_cell_xy]), axis=2)
        min_dist = (np.amin(dists, initial=np.inf, where=dists>0) - 1) / 2
        
        """Apply Watershed segmentation to generate fitting masks"""
        fit_masks, num_fit_masks, slices_LoG, xy_peak = watershed_segment(
            img_LoG, local_thresh_factor = local_thresh_factor, 
            watershed_line=watershed_line, min_dist=min_dist)
        
        """Gaussian blur to group columns for simultaneous fitting"""
        group_masks, num_group_masks, slices_Gauss, _ = watershed_segment(
            img_gauss, local_thresh_factor = 0, watershed_line=watershed_line,
            min_dist=min_dist)
        
        """Use local peaks in img_LoG and match to reference lattice.
        These points will be initial position guesses for fitting"""
        xy_peak = xy_peak.loc[:, 'x':'y'].to_numpy()
        xy_ref = at_cols.loc[:, 'x_ref':'y_ref'].to_numpy()
        
        vects = np.array([xy_peak - xy for xy in xy_ref])
        norms = np.linalg.norm(vects, axis=2)
        inds = np.argmin(norms, axis=1)
        xy_peak = np.array([xy_peak[ind] for ind in inds])
        slices_LoG = [slices_LoG[ind] for ind in inds]
        
        """If the difference between detected peak position and reference 
        position is greater than half the probe_fwhm, the reference is taken
        as the initial guess."""
        mask = (np.min(norms, axis=1) < self.probe_fwhm/self.pixel_size*0.5
                ).reshape((-1, 1))
        mask = np.concatenate((mask, mask), axis=1)
        xy_peak = np.where(mask, xy_peak, xy_ref)
        self.xy_peak = xy_peak
        
        """Find corresponding mask (from both LoG and Gauss filtering) for each 
        peak"""
        fit_masks_to_peaks = ndimage.map_coordinates(fit_masks, 
                                                     np.flipud(xy_peak.T), 
                                                     order=0).astype(int)
        group_masks_to_peaks = ndimage.map_coordinates(group_masks, 
                                                       np.flipud(xy_peak.T), 
                                                       order=0).astype(int)
        
        LoG_masks_used = np.unique(fit_masks_to_peaks)
        Gauss_masks_used = np.unique(group_masks_to_peaks)
        
        """Throw out masks which do not correspond to at least one reference 
            lattice point"""
        fit_masks = np.where(np.isin(fit_masks, LoG_masks_used), 
                                  fit_masks, 0)
        
        group_masks = np.where(np.isin(group_masks, Gauss_masks_used), 
                                  group_masks, 0)
        self.fit_masks = np.where(fit_masks >= 1, 1, 0)
        self.group_masks = np.where(group_masks >= 1, 1, 0)
        
        """Find sets of reference columns for each grouping mask"""
        peak_groupings = [[mask_num, 
                           np.argwhere(group_masks_to_peaks==mask_num
                                       ).flatten()]
                          for mask_num in Gauss_masks_used if mask_num != 0]
        
        group_sizes, counts = np.unique([match[1].shape[0] 
                                         for match in peak_groupings],
                                        return_counts=True)
        
        if grouping_filter:
            print('Atomic columns grouped for simultaneous fitting:')
            for i, size in enumerate(group_sizes):
                print(f'{counts[i]}x {size}-column groups')
                
        at_cols_inds = at_cols.index.to_numpy()
        
        """Find min & max slice indices for each group of fitting masks"""
        group_fit_slices = [[slices_LoG[ind] for ind in inds] 
                                for [_, inds]  in peak_groupings]
        
        group_fit_slices = [np.array([[sl[0].start, sl[0].stop, 
                                       sl[1].start, sl[1].stop] 
                                      for sl in group]).T
                            for group in group_fit_slices]
        
        group_slices = [np.s_[np.min(group[0]) : np.max(group[1]),
                              np.min(group[2]) : np.max(group[3])]
                         for group in group_fit_slices]
        
        """Pack image slices and metadata together for the fitting routine"""
        args_packed = [[self.image[group_slices[counter][0],
                                   group_slices[counter][1]],
                        fit_masks[group_slices[counter][0],
                                      group_slices[counter][1]],
                        fit_masks_to_peaks[inds],
                        [group_slices[counter][1].start, 
                         group_slices[counter][0].start],
                        xy_peak[inds, :].reshape((-1, 2)), 
                        at_cols_inds[inds],
                        group_mask_num]
                       for counter, [group_mask_num, inds]  
                       in enumerate(peak_groupings)]
        
        self.args_packed = args_packed
        
        """Define column fitting function for image slices"""
        
        if use_LoG_fitting==False:
            # Use Gaussian fitting
            def fit_column(args):
                [img_sl, mask_sl, log_mask_num, xy_start, xy_peak, inds, 
                 _] = args
                num = xy_peak.shape[0]
                masks = np.where(np.isin(mask_sl, log_mask_num), mask_sl, 0)
                    
                img_msk = img_sl * np.where(masks > 0, 1, 0)
                
                if num == 1:
                    [x0, y0] = (xy_peak - xy_start).flatten()
                    _, _, _, theta, sig_1, sig_2 = img_ellip_param(img_msk)
                    I0 = (np.average(img_msk[img_msk != 0])
                          - np.std(img_msk[img_msk != 0]))
                    A0 = np.max(img_msk) - I0
     
                    p0 = np.array([x0, y0, sig_1, sig_2, np.radians(theta), 
                                    A0, I0])
                    
                    params = fit_gaussian2D(img_msk, p0, 
                                            use_LoG_fitting=False,
                                            method='BFGS')
                    
                    params = np.array([params[:,0] + xy_start[0],
                                       params[:,1] + xy_start[1],
                                       params[:,2],
                                       params[:,3],
                                       np.degrees(params[:,4]),
                                       params[:,5],
                                       params[:,6]]).T
                
                if num > 1:
                    x0y0 = xy_peak - xy_start
                    x0 = x0y0[:, 0]
                    y0 = x0y0[:, 1]
                    
                    sig_1 = []
                    sig_2 = []
                    theta = []
                    I0 = []
                    A0 = []
                    
                    for i, mask_num in enumerate(log_mask_num):
                        mask = np.where(masks == mask_num, 1, 0)
                        masked_sl = img_sl * mask
                        _, _, _, theta_, sig_1_, sig_2_ = (
                            img_ellip_param(masked_sl))
                        sig_1 += [sig_1_]
                        sig_2 += [sig_2_]
                        theta += [np.radians(theta_)]
                        I0 += [(np.average(masked_sl[masked_sl != 0])
                              - np.std(masked_sl[masked_sl != 0]))]
                        A0 += [np.max(masked_sl) - I0[i]]
                    
                    p0 = np.array([x0, y0, sig_1, sig_2, theta, A0]).T

                    p0 = np.append(p0.flatten(), np.mean(I0))

                    params = fit_gaussian2D(img_msk, p0, masks, 
                                            use_LoG_fitting=False,
                                            method='BFGS')
                    
                    if np.any(params[:,-1] < 0):
                        params = fit_gaussian2D(img_msk, p0, masks, 
                                                use_LoG_fitting=False,
                                                method='L-BFGS-B')
                    
                    params = np.array([params[:,0] + xy_start[0],
                                        params[:,1] + xy_start[1],
                                        params[:,2],
                                        params[:,3],
                                        np.degrees(params[:,4]),
                                        params[:,5],
                                        params[:,6]]).T
                
                return params
        
        elif use_LoG_fitting==True:
            # Use LoG fitting
            def fit_column(args):
                [img_sl, mask_sl, log_mask_num, xy_start, xy_peak, inds, 
                 mask_num] = args
                num = xy_peak.shape[0]
                masks = np.where(np.isin(mask_sl, log_mask_num), mask_sl, 0)
                    
                img_msk = img_sl * np.where(masks > 0, 1, 0) 
                
                if num == 1:
                    [x0, y0] = (xy_peak - xy_start).flatten()
                    _, _, _, theta, sig_1, sig_2 = img_ellip_param(img_msk)
                    I0 = (np.average(img_msk[img_msk != 0])
                          - np.std(img_msk[img_msk != 0]))
                    A0 = np.max(img_msk) - I0
     
                    p0 = np.array([x0, y0, np.mean([sig_1, sig_2]), 1, 0, 
                                    A0, I0])
                    
                    params = fit_gaussian2D(img_msk, p0, method='BFGS',
                                            use_LoG_fitting=use_LoG_fitting)
                
                if num > 1:
                    x0y0 = xy_peak - xy_start
                    x0 = x0y0[:, 0]
                    y0 = x0y0[:, 1]
                    
                    sig = []
                    sig_r = []
                    theta = []
                    I0 = []
                    A0 = []
                    
                    for i, mask_num in enumerate(log_mask_num):
                        mask = np.where(masks == mask_num, 1, 0)
                        masked_sl = img_sl * mask
                        _, _, _, theta_, sig_1_, sig_2_ = (
                            img_ellip_param(masked_sl))
                        sig += [(sig_1_+sig_2_)/2]
                        sig_r += [1]
                        theta += [0]
                        I0 += [(np.average(masked_sl[masked_sl != 0])
                              - np.std(masked_sl[masked_sl != 0]))]
                        A0 += [np.max(masked_sl) - I0[i]]
                    
                    p0 = np.array([x0, y0, sig, sig_r, theta, A0]).T
                    p0[:, 2] *= 1.75
                    p0[:, 5] *= p0[:, 2]**2
                    p0 = np.append(p0.flatten(), np.mean(I0))
                    
                    params = fit_gaussian2D(img_msk, p0, masks, 
                                            use_LoG_fitting=True,
                                            method='BFGS')
                    
                    if np.any(params[:,-1] < 0):
                        params = fit_gaussian2D(img_msk, p0, masks, 
                                                use_LoG_fitting=True,
                                                method='L-BFGS-B')
                    
                params = np.array([params[:,0] + xy_start[0],
                                    params[:,1] + xy_start[1],
                                    params[:,2],
                                    params[:,3],
                                    np.degrees(params[:,4]),
                                    params[:,5],
                                    params[:,6]]).T
                
                return params
        
        else:
            raise Exception('the arg "use_LoG_fitting" must be a bool')
        
        """Run fitting routine"""
        print('Fitting atom columns...')
        t0 = time.time()
        if parallelize == True:
            """Large data set: use parallel processing"""
            print('Using parallel processing')
            n_jobs = psutil.cpu_count(logical=False)
            
            try:
                results_ = Parallel(n_jobs=n_jobs)(
                    delayed(fit_column)(arg) for arg in tqdm(args_packed))
                
            except ZeroDivisionError as err:
                # import sys
                # sys.tracebacklimit=0
                print('\n')
                raise ZeroDivisionError(
                    "Float division by zero during Gausssian fitting. "
                    + "No pixel region found for fitting of at least one "
                    + "atom column. This may result from: \n"
                    + "1) Too high a 'local_thresh_factor' "
                    + "value resulting in no pixels remaining for some "
                    + "atom columns. Check 'self.fit_masks' to see if "
                    + "some atom columns do not have mask regions. \n"
                    + "2) Due to mask regions splitting "
                    + "an atom column as a result of too small or too "
                    + "large a value used for 'grouping_filter' or "
                    + "'diff_filter'. Check 'self.group_masks' and "
                    + "'self.fit_masks' to see if this may be "
                    + "occuring. Too small a 'fitting_filter' value may "
                    + "cause noise peaks to be detected, splitting an atom "
                    + "column. Too large a value for either filter may "
                    + "cause low intensity peaks to be ignored by the "
                    + "watershed algorithm. \n"
                    + "Masks may also be checked with the 'show_masks()' "
                    + "method.'") from None
                
            results = np.concatenate([np.concatenate(
                (result,
                 args_packed[i][5].reshape(-1, 1)), axis=1)
                for i, result in enumerate(results_)])
                
        else:
            """Small data set: use serial processing"""
            print('Using serial processing')
            try:
                results_ = [fit_column(arg) for arg in tqdm(args_packed)]
                
            except ZeroDivisionError as err:
                # import sys
                # sys.tracebacklimit=0
                print('\n')
                raise ZeroDivisionError(
                    "Float division by zero during Gausssian fitting. "
                    + "No pixel region found for fitting of at least one "
                    + "atom column. This may result from: \n"
                    + "1) Too high a 'local_thresh_factor' "
                    + "value resulting in no pixels remaining for some "
                    + "atom columns. Check 'self.fit_masks' to see if "
                    + "some atom columns do not have mask regions. \n"
                    + "2) Due to mask regions splitting "
                    + "an atom column as a result of too small or too "
                    + "large a value used for 'grouping_filter' or "
                    + "'diff_filter'. Check 'self.group_masks' and "
                    + "'self.fit_masks' to see if this may be "
                    + "occuring. Too small a 'fitting_filter' value may "
                    + "cause noise peaks to be detected, splitting an atom "
                    + "column. Too large a value for either filter may "
                    + "cause low intensity peaks to be ignored by the "
                    + "watershed algorithm. \n"
                    + "Masks may also be checked with the 'show_masks()' "
                    + "method.'") from None
               
            results = np.concatenate([np.concatenate((
                result, 
                args_packed[i][5].reshape(-1, 1)), axis=1)
                for i, result in enumerate(results_)])
                                                      
        t_elapse = time.time() - t0
        
        print(f'Done. Fitting time: {int(t_elapse // 60)} min '
              +f'{t_elapse % 60 :.{1}f} sec')
       
        """Post-process results"""
       
        col_labels = ['x_fit', 'y_fit', 'sig_1', 'sig_2',
                     'theta', 'peak_int', 'bkgd_int', 'total_col_int']
        if not col_labels[0] in at_cols.columns:
           empty = pd.DataFrame(index=at_cols.index.tolist(), 
                                columns = col_labels)
           at_cols = at_cols.join(empty)
       
        results = pd.DataFrame(data=results[:, :-1], 
                              index=results[:,-1].astype(int), 
                              columns=col_labels[:-1]).sort_index()
        results.loc[:, 'total_col_int'] = (2 * np.pi 
                                           * results.peak_int.to_numpy()
                                           * results.sig_1.to_numpy()
                                           * results.sig_2.to_numpy())
       
        at_cols.update(results)
        sigmas = at_cols.loc[:, 'sig_1':'sig_2'].to_numpy()
        theta = at_cols.loc[:, 'theta'].to_numpy()
        sig_maj_inds = np.argmax(sigmas, axis=1)
        sig_min_inds = np.argmin(sigmas, axis=1)
        sig_maj = sigmas[[i for i in range(sigmas.shape[0])], 
                         list(sig_maj_inds)]
        sig_min = sigmas[[i for i in range(sigmas.shape[0])],
                         list(sig_min_inds)]
        theta += np.where(sig_maj_inds == 1, 90, 0)
        theta = ((theta + 90) % 180) - 90
        at_cols.loc[:, 'sig_1'] = sig_maj
        at_cols.loc[:, 'sig_2'] = sig_min
        at_cols.loc[:, 'theta'] = theta
        
        '''Convert values from dtype objects to ints, floats, etc:'''
        at_cols = at_cols.infer_objects()
        
        '''Crop with buffer '''
        at_cols = at_cols[((at_cols.x_ref >= 0) &
                           (at_cols.x_ref <= self.w) &
                           (at_cols.y_ref >= 0) &
                           (at_cols.y_ref <= self.h))].copy()
        
        self.at_cols_uncropped.update(at_cols)
        self.at_cols_uncropped = self.at_cols_uncropped.infer_objects()
        self.at_cols = self.at_cols_uncropped.dropna(axis=0)
        self.at_cols = self.at_cols[((self.at_cols.x_ref >= buffer) &
                                     (self.at_cols.x_ref <= self.w - buffer) &
                                     (self.at_cols.y_ref >= buffer) &
                                     (self.at_cols.y_ref <= self.h - buffer))
                                    ].copy()
    
    def show_masks(self, mask_to_show='fitting', display_masked_image=True):
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
            mask = self.fit_masks
        elif mask_to_show == 'grouping':
            mask = self.group_masks
        else:
            raise Exception("The argument 'mask_to_show' must be either: " \
                            "'fitting' or 'grouping'.")
            
        fig, ax = plt.subplots()
        if display_masked_image:
            ax.imshow(self.image * mask)
        else:
            ax.imshow(mask)
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def refine_reference_lattice(self, filter_by='elem', 
                                 sites_to_use='all', outlier_disp_cutoff=None):
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
        outlier_disp_cutoff : None or float or int
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
                filtered = self.at_cols[self.at_cols.loc[:, filter_by]
                                        .isin(sites_to_use)].copy()
            
            elif type(sites_to_use) == str:
                filtered = self.at_cols[self.at_cols.loc[:, filter_by]
                                        == sites_to_use].copy()
            else:
                raise Exception('"sites_to_use" must be a string or a list')
                
        if filtered.shape[0] == 0:
            raise Exception('No atom columns found to use for '
                            + 'refinement with arguments given')
            
        if outlier_disp_cutoff == None:
            outlier_disp_cutoff = 1 / self.pixel_size
            
        else:
            outlier_disp_cutoff /= self.pixel_size * 100
            
        filtered = filtered[np.linalg.norm(
            filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
            - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outlier_disp_cutoff].copy()

        def disp_vect_sum_squares(p0, M, xy):
            
            trans_mat = p0[:4].reshape((2,2))
            origin = p0[4:]
            
            err_xy = xy - M @ trans_mat - origin
            sum_sq = np.sum(err_xy**2)
            return sum_sq
        
        M = filtered.loc[:, 'u':'v'].to_numpy()
        xy = filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
        
        p0 = np.concatenate((self.trans_mat.flatten(),
                             np.array([self.x0, self.y0])))
        params = minimize(disp_vect_sum_squares, p0, args=(M, xy)).x
        
        self.a1 = params[:2]
        self.a2 = params[2:4]
        
        self.trans_mat = params[:4].reshape((2,2))
        
        print('Origin shift:', params[4:] - np.array([self.x0, self.y0]))
        self.x0 = params[4]
        self.y0 = params[5]
        print('Optimized basis vectors:', self.trans_mat)
        
        self.basis_offset_pix = self.basis_offset_frac @ self.trans_mat
        
        self.at_cols.loc[:, 'x_ref':'y_ref'] = (self.at_cols.loc[:, 'u':'v']
                                                .to_numpy() @ self.trans_mat
                                                + np.array([self.x0, self.y0]))
        
        self.pixel_size = np.average([np.linalg.norm(self.a_2d[0,:])
                                      /np.linalg.norm(self.a1),
                                      np.linalg.norm(self.a_2d[1,:])
                                      /np.linalg.norm(self.a2)])
        
        theta_ref = np.degrees(
            np.arccos(self.trans_mat[0,:] 
                      @ self.trans_mat[1,:].T
                      /(np.linalg.norm(self.trans_mat[0,:]) 
                        * np.linalg.norm(self.trans_mat[1,:].T))))

        shear_distortion_res = np.radians(90 - theta_ref)

        scale_distortion_res = 1 - ((np.linalg.norm(self.a1)
                                     * np.linalg.norm(self.a_2d[1,:]))/
                                    (np.linalg.norm(self.a2)
                                     * np.linalg.norm(self.a_2d[0,:])))

        pix_size = np.average([np.linalg.norm(self.a_2d[0,:])
                                  /np.linalg.norm(self.a1),
                                  np.linalg.norm(self.a_2d[1,:])
                                  /np.linalg.norm(self.a2)])
        
        print('')
        print('Residual distortion of reference lattice basis vectors' 
              + ' from .cif:')
        print(f'Scalar component: {scale_distortion_res * 100 :.{2}f} %')
        print(f'Shear component: {shear_distortion_res :.{5}f} (radians)')
        print(f'Estimated Pixel Size: {pix_size * 100 :.{3}f} (pm)')
        
    
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
        
        if not self.use_LoG_fitting:

            def get_group_residuals(args, params, buffer, image_dims):
                [img_sl, mask_sl, mask_nums, xy_start, xy_peak, inds, 
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
                    model = np.where(masks==mask_num, params[i,-1], model)
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
                    n = np.count_nonzero(masks==mask_num)
                    r_i = r[masks == mask_num]
                    data_i = data[masks == mask_num]
                    RSS_norm += [np.sum(r_i**2) / n]
                    
                    if ((params[i, 0] <= buffer) |
                        (params[i, 0] >= image_dims[1] - buffer) |
                        (params[i, 1] <= buffer) |
                        (params[i, 1] >= image_dims[0] - buffer)):
                        r[masks==mask_num] = 0
               
                x += xy_start[0]
                y += xy_start[1]
                
                return x, y, r, RSS_norm

        elif self.use_LoG_fitting:
            
            def get_group_residuals(args, params, buffer, image_dims):
                [img_sl, mask_sl, mask_nums, xy_start, xy_peak, inds, 
                 _] = args

                masks = np.where(np.isin(mask_sl, mask_nums), mask_sl, 0)
                
                params[:, :2] -= xy_start
                
                data = np.where(masks > 0, img_sl, 0)
                
                data = data.flatten()
                y, x = np.indices(masks.shape)
                unmasked_data = np.nonzero(data)
                data = np.take(data, unmasked_data)
                x = np.take(x, unmasked_data)
                y = np.take(y, unmasked_data)
                masks = np.take(masks.flatten(), unmasked_data)
                
                model = np.zeros(masks.shape)
                for i, mask_num in enumerate(mask_nums):
                    model = np.where(masks==mask_num, params[i,-1], model)
                for peak in params:
                    model += LoG_2d(x, y, *peak[:-1], I_o=0)
                
                r = data - model
                
                RSS_norm = []
                params[:, :2] += xy_start
                for i, mask_num in enumerate(mask_nums):
                    # n = np.count_nonzero(masks==mask_num)
                    r_i = r[masks == mask_num]
                    data_i = data[masks == mask_num]
                    RSS_norm += [np.sum(r_i**2) / 
                                   np.sum((data_i - np.mean(data_i))**2)]
                    
                    if ((params[i, 0] <= buffer) |
                        (params[i, 0] >= image_dims[1] - buffer) |
                        (params[i, 1] <= buffer) |
                        (params[i, 1] >= image_dims[0] - buffer)):
                        r[masks==mask_num] = 0
                
                x += xy_start[0]
                y += xy_start[1]
                
                return x, y, r, RSS_norm

        else:
            raise Exception('"self.use_LoG_fitting" must be a bool')
        
        group_residuals = [get_group_residuals(
            args, self.at_cols_uncropped.loc[args[5], 
                                             'x_fit':'bkgd_int'].to_numpy(),
            self.buffer, self.image.shape) 
            for args in self.args_packed]
        
        self.residuals = np.zeros(self.image.shape)

        for counter, [x, y, R, RSS_norm] in enumerate(group_residuals):
            self.residuals[y, x] += R
            for i, ind in enumerate(self.args_packed[counter][5]):
                self.at_cols_uncropped.loc[ind, 'RSS_norm'] = RSS_norm[i]
        
        self.at_cols.loc[:,'RSS_norm'] = np.nan
        self.at_cols.update(self.at_cols_uncropped)
        
        cmap_lim = np.max(np.abs(self.residuals))
        
        fig,axs = plt.subplots(ncols=1,figsize=(10,10), tight_layout=True)
        axs.set_xticks([])
        axs.set_yticks([])
        res_plot = axs.imshow(self.residuals, cmap='bwr', 
                               norm=Normalize(vmin=-cmap_lim, vmax=cmap_lim))
        
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(res_plot, cax=cax)        
        
        axs.scatter(self.at_cols.loc[:,'x_fit'], 
                    self.at_cols.loc[:,'y_fit'],
                    color='black', s=4)        
       
        r = self.residuals.flatten()
        data = self.image * self.fit_masks
        data[:self.buffer, :] = 0
        data[-self.buffer:, :] = 0
        data[:, :self.buffer] = 0
        data[:, -self.buffer:] = 0
                  
        data = data.flatten()
        inds = data.nonzero()
        r = np.take(r, inds)
        data = np.take(data, inds)
        
        # model = data - res
        n = data.shape[1]
        m = self.at_cols.shape[0] * 7
        v = n - m
        
        R_sq = 1 - np.sum(r**2) / np.sum((data - np.mean(data))**2)
        
        chi_sq = np.sum((r**2) / v)
        
        print(f'chi-squared of all atom column fits: {chi_sq :.{4}e} \n')
        print(f'R-squared of all atom column fits: {R_sq :.{6}f} \n')
        
        return fig

    def rotate_image_and_data(self, align_dir='horizontal',
                              align_basis='a1'):
        """Rotates the image and data to align a basis vector to image edge.
        
        Rotates the image so that a chosen crystalloraphic basis vector 
        is horizontal or vertical. Adjusts the reference lattice and 
        atomic column fit data accordingly. Desireable for displaying data 
        for presentation.
        
        Parameters
        ----------
        align_dir : str ('horizontal' or 'vertical')
            Direction to align the chosen basis vector.
            Default 'horizontal'. 
        align_basis : str ('a1' or 'a2')
            The basis vector to align.
            Default 'a1'.
             
        Returns
        -------
        rot_: AtomicColumnLattice object with rotated image and data.
            
        """
        
        rot_ = copy.deepcopy(self)
        if align_dir == 'horizontal' or align_dir == 'vertical':
            pass
        else:
            raise Exception('align_dir must be "horizontal" or "vertical"')
        
        if align_basis =='a1':
            align_vect = self.a1
        elif align_basis =='a2':
            align_vect = self.a2
        
        '''Find the rotation angle and 
            direction'''
        angle =  np.arctan2(align_vect[1], align_vect[0])
        if align_dir == 'horizontal':
            pass
        elif align_dir == 'vertical':
            angle += np.pi/2
        else:
            raise Exception('align_dir must be "horizontal" or "vertical"')
        
        print('Rotation angle:', np.degrees(angle))
        
        rot_.image = ndimage.rotate(rot_.image, np.degrees(angle))
        [rot_.h, rot_.w] = rot_.image.shape
        rot_.fit_masks = ndimage.rotate(rot_.fit_masks, 
                                            np.degrees(angle))
        
        '''Translation of image center due to increased image array size
            resulting from the rotation'''
        trans = np.flip(((np.array(rot_.image.shape, ndmin=2)-1)/2
                         - (np.array(self.image.shape, ndmin=2)-1)/2), axis=1)
        '''Find the origin-shifted rotation matrix for transforming atomic
            column position data'''
        trans_mat = np.array([[np.cos(angle), np.sin(angle), 0],
                          [-np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
        tau = np.array([[1, 0, (self.image.shape[1]-1)/2],
                        [0, 1, (self.image.shape[0]-1)/2],
                        [0, 0, 1]])
        tau_ = np.array([[1, 0, -(self.image.shape[1]-1)/2],
                         [0, 1, -(self.image.shape[0]-1)/2],
                         [0, 0, 1]])
        trans_mat = tau @ trans_mat @ tau_
        
        xy = np.array(
            np.append(rot_.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy(),
                      np.ones((rot_.at_cols.shape[0],1)), axis=1)).T
        
        rot_.at_cols.loc[:, 'x_fit':'y_fit'] = ((trans_mat @ xy).T[:, :2] 
                                                + trans)
        
        xy_pix = np.append(rot_.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(),
                           np.ones((rot_.at_cols.shape[0],1)), axis=1).T
        
        rot_.at_cols.loc[:, 'x_ref':'y_ref'] = ((trans_mat @ xy_pix).T[:, :2] 
                                                + trans)
        
        [rot_.x0, rot_.y0] = list((np.array([rot_.x0, rot_.y0, 1], ndmin=2) 
                                  @ trans_mat.T)[0,0:2] + trans[0,:])
        
        '''Transform data'''
        rot_.trans_mat = rot_.trans_mat @ trans_mat[0:2, 0:2].T
        rot_.a1 = rot_.trans_mat[0, :]
        rot_.a2 = rot_.trans_mat[1, :]
        '''***Logic sequence to make basis vectors ~right, ~up'''
        
        rot_.a1_star = (np.linalg.inv(rot_.trans_mat).T)[0, :]
        rot_.a2_star = (np.linalg.inv(rot_.trans_mat).T)[1, :]
        rot_.at_cols.theta += np.degrees(angle)
        rot_.at_cols.theta -= np.trunc(rot_.at_cols.theta.to_numpy().astype(
            'float') / 90) * 180
        rot_.angle = angle
        
        return rot_
        
    def plot_atom_column_positions(self, filter_by='elem', sites_to_plot='all',
                                   fit_or_ref='fit', outlier_disp_cutoff=None,
                                   plot_masked_image=False, 
                                   xlim=None, ylim=None, scalebar_len_nm=2,
                                   color_dict=None, legend_dict=None,
                                   scatter_kwargs_dict={}):
        '''TODO: add kwargs for plt.scatter'''
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
        outlier_disp_cutoff : None or float or int
            Criteria for removing outlier atomic column fits from the 
            plot (in pm). The maximum difference between the fitted position 
            and the corresponding reference lattice point. All positions 
            with greater errors than this value will be removed. If None, 
            all column positions will be plotted.
            Default None.
        plot_masked_image : bool
            Whether to plot the masked image (shows only the regions used
            for fitting). If False, the unmasked image is plotted.
            Default: False
        scalebar_len_nm : int or float or None
            The desired length of the scalebar in nm. 
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
        
        Returns
        -------
        fig : the matplotlib figure object.
        axs : the matplotlib axes object.
            
        """
        
        
        scatter_kwargs_default = {'s' : 25, 
                                  'edgecolor' : 'black', 
                                  'linewidths' : 0.5}
        scatter_kwargs_default.update(scatter_kwargs_dict)
                                  
        if sites_to_plot == 'all' or ['all']:
            filtered = self.at_cols
            
        else:
            filtered = self.at_cols[self.at_cols.loc[:, filter_by]
                                    .isin(sites_to_plot)].copy()
        
        if (type(outlier_disp_cutoff) == float or 
            type(outlier_disp_cutoff) == int):
            outlier_disp_cutoff /= self.pixel_size * 100
        
        if outlier_disp_cutoff != None:
            filtered = filtered[np.linalg.norm(
                filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
                - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(),
                axis=1)
                < outlier_disp_cutoff].copy()
        
        if fit_or_ref == 'fit':
            xcol, ycol = 'x_fit', 'y_fit'
        elif fit_or_ref == 'ref':
            xcol, ycol = 'x_ref', 'y_ref'
                                
        fig, axs = plt.subplots(ncols=1,figsize=(13,10), tight_layout=True)
        
        if plot_masked_image == True:
            axs.imshow(self.image * self.fit_masks, cmap='gray')
        else:
            axs.imshow(self.image, cmap='gray')
        axs.set_xticks([])
        axs.set_yticks([])
        
        if xlim == None:
            xlim = [0, self.w]
        else:
            axs.set_xlim(xlim)
        if ylim == None:
            ylim = [0, self.h]
        else:
            axs.set_ylim(ylim)
        
        unitcell = filtered[(filtered['u'] // 1 == 0) & 
                               (filtered['v'] // 1 == 0)]
        
        if color_dict == None:
            cmap = plt.cm.RdYlGn
            num_colors = unitcell.loc[:, filter_by].unique().shape[0]
            color_dict = {k:cmap(v/(num_colors-1)) for v, k in 
                          enumerate(np.sort(unitcell.loc[:, filter_by].unique()
                                            ))}
    
        for site in color_dict:
            if legend_dict != None:
                label = legend_dict[site]
            else: label = site
            sublattice = filtered[filtered[filter_by] == site].copy()
            axs.scatter(sublattice.loc[:, xcol], sublattice.loc[:, ycol], 
                        color=color_dict[site], label=label, 
                        **scatter_kwargs_default)
            
        axs.legend(loc='lower left', bbox_to_anchor=[1.02, 0], 
                   facecolor='grey', fontsize=16)
        
        if scalebar_len_nm != None:
            scalebar = ScaleBar(self.pixel_size/10,
                                'nm', location='lower right', pad=0.4, 
                                fixed_value=scalebar_len_nm, 
                                font_properties={'size':20}, 
                                box_color='lightgrey', width_fraction=0.02,
                                sep=2, border_pad=2)
            axs.add_artist(scalebar)

        axs.arrow(self.x0, self.y0, self.a1[0], self.a1[1],
                      fc='red', ec='red', width=0.1, 
                      length_includes_head=True,
                      head_width=2, head_length=3, label=r'[001]')
        axs.arrow(self.x0, self.y0, self.a2[0], self.a2[1],
                      fc='green', ec='green', width=0.1, 
                      length_includes_head=True,
                      head_width=2, head_length=3, label=r'[110]')
        
        return fig, axs
        
    def plot_disp_vects(self, filter_by='elem', sites_to_plot='all', 
                        x_lim=None, y_lim=None,
                        scalebar=True, scalebar_len_nm=2,
                        arrow_scale_factor = 1,
                        outlier_disp_cutoff = None, 
                        max_colorwheel_range_pm=None,
                        plot_fit_points=False, plot_ref_points=False):
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
        x_lim : None or list-like shape (2,)
            The x axis limits to be plotted. If None, the whole image is 
            displayed.
            Default: None
        y_lim : None or list-like shape (2,)
            The y axis limits to be plotted. Note that the larger valued limit
            should be first or the plot will be flipped top-to-bottom.  If 
            None, the whole image is displayed.
            Default: None
        scalebar_len_nm : int or float or None
            The desired length of the scalebar in nm. If None, no scalebar is
            plotted.
            Default: 2.
        arrow_scale_factor : int or float
            Relative scaling factor for the displayed displacement vectors.
            Default: 1.
        outlier_disp_cutoff : None or float or int
            Criteria for removing outlier atomic column fits from the 
            plot (in pm). The maximum difference between the fitted position 
            and the corresponding reference lattice point. All positions 
            with greater errors than this value will be removed. If None, 
            all column positions will be plotted.
            Default None.
        max_colorwheel_range_pm : int or float or None
            The maximum absolute value of displacement included in the 
            colorwheel range. Displacement vectors longer than this value will
            have the same color intensity.
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
        
        if sites_to_plot == 'all':
            sites_to_plot = self.at_cols.loc[:, filter_by].unique().tolist()
            sites_to_plot.sort()
        elif type(sites_to_plot) != list:
            raise Exception('"sites_to_plot" must be either "all" or a list')
            
        if outlier_disp_cutoff == None:
            outlier_disp_cutoff = np.inf
            
        else:
            outlier_disp_cutoff /= self.pixel_size * 100
            
        filtered = self.at_cols[np.linalg.norm(
            self.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy()
            - self.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outlier_disp_cutoff].copy()
       
        if max_colorwheel_range_pm == None:
            filtered = filtered[filtered.loc[:, filter_by]
                                    .isin(sites_to_plot)]
            dxy = (filtered.loc[:,'x_fit':'y_fit'].to_numpy()
                   - filtered.loc[:,'x_ref':'y_ref'].to_numpy())
            mags = np.linalg.norm(dxy, axis=1) * self.pixel_size * 100
            avg = np.mean(mags)
            std = np.std(mags)
            max_colorwheel_range_pm = int(np.ceil((avg + 3*std)/5) * 5)
        
        if x_lim==None:
            x_lim=[0, self.w]
        if y_lim==None:
            y_lim=[self.h, 0]
        if y_lim[0] < y_lim[1]:
            y_lim = [y_lim[1], y_lim[0]]
        
        n_plots = len(sites_to_plot)
        if n_plots > 12:
            raise Exception('The number of plots exceeds the limit of 12.')
        
        if n_plots <= 3:
            nrows = 1
            ncols = n_plots
            width_ratios=[3] * ncols + [1]
            
        elif n_plots <= 8:
            nrows = 2
            ncols = np.ceil(n_plots/2).astype(int)
            width_ratios=[3] * ncols + [1]
            print(nrows, ncols)
            
        elif n_plots <= 12:
            nrows = 3
            ncols = np.ceil(n_plots/3).astype(int)
            width_ratios=[3] * ncols + [1]
        
        figsize=(ncols * 5 + 3, 5 * nrows + 3)
    
        fig = plt.figure(figsize=figsize)#, tight_layout=True)
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, 
                              width_ratios=width_ratios,
                              height_ratios=[3 for _ in range(nrows)], 
                              wspace=0.05)
        if x_lim == None:
            x_lim=[0, self.w] 
        if y_lim == None:
            y_lim=[self.h, 0],
        axs = []
        for ax, site in enumerate(sites_to_plot):
            row = ax // ncols
            col = ax % ncols
            axs += [fig.add_subplot(gs[row, col])]
            axs[ax].imshow(self.image, cmap='gray')

            axs[ax].set_xlim(x_lim[0], x_lim[1])
            axs[ax].set_ylim(y_lim[0], y_lim[1])
            
            axs[ax].set_xticks([])
            axs[ax].set_yticks([])

            if ax == 0 and scalebar_len_nm != None:
                scalebar = ScaleBar(self.pixel_size/10,
                                    'nm', location='lower right', pad=0.4, 
                                    fixed_value=scalebar_len_nm, 
                                    font_properties={'size':10}, 
                                    box_color='lightgrey', width_fraction=0.02,
                                    sep=2)
                axs[ax].add_artist(scalebar)
                
            sub_latt = filtered[filtered.loc[:, filter_by] == site]
            h = y_lim[0] - y_lim[1]
            title = axs[ax].text(x_lim[0] + 0.02*h, y_lim[1] + 0.02*h, 
                             rf'{"$" + sites_to_plot[ax] + "$"}', 
                             color='white', size=24,  weight='bold',
                             va='top', ha='left')
            title.set_path_effects([path_effects.Stroke(linewidth=3, 
                                                        foreground='black'),
                                    path_effects.Normal()])
            
            hsv = np.ones((sub_latt.shape[0], 3))
            dxy = (sub_latt.loc[:,'x_fit':'y_fit'].to_numpy()
                   - sub_latt.loc[:,'x_ref':'y_ref'].to_numpy())
            
            disp_pm = (np.linalg.norm(dxy, axis=1) * self.pixel_size * 100)
            normed = disp_pm / max_colorwheel_range_pm
            print(rf'Displacement statistics for {site}:',
                  f'average: {np.mean(disp_pm)  :.{2}f} (pm)',
                  f'standard deviation: {np.std(disp_pm)  :.{2}f} (pm)',
                  f'maximum: {np.max(disp_pm)  :.{2}f} (pm)',
                  f'minimum: {np.min(disp_pm)  :.{2}f} (pm)',
                  '\n',
                  sep='\n')
            hsv[:, 2] = np.where(normed>1, 1, normed)
            hsv[:, 0] = (np.arctan2(dxy[:,0], dxy[:,1]) 
                         + np.pi/2)/(2*np.pi) % 1
            rgb = colors.hsv_to_rgb(hsv)
            
            if plot_fit_points:
                axs[ax].scatter(sub_latt.loc[:,'x_fit'], sub_latt.loc[:,'y_fit'],
                            color='blue', s=1)
            if plot_ref_points: 
                axs[ax].scatter(sub_latt.loc[:,'x_ref'], sub_latt.loc[:,'y_ref'],
                            color='red', s=1)
            cb = axs[ax].quiver(sub_latt.loc[:,'x_fit'], sub_latt.loc[:,'y_fit'], 
                            dxy[:, 0], dxy[:, 1],
                            color=rgb,
                            angles='xy', scale_units='xy', 
                            scale=0.1/arrow_scale_factor,
                            headlength=10, headwidth=5, headaxislength=10,
                            edgecolor='white', linewidths=0.5)
        
        def colour_wheel(samples=1024, clip_circle=True):
            xx, yy = np.meshgrid(
                np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))
        
            v = np.sqrt(xx ** 2 + yy ** 2)
            if clip_circle == True:
                v[v>0.99] = 1
            h = ((np.arctan2(xx, yy) + np.pi/2) / (np.pi * 2)) % 1
            hsv = np.ones((samples, samples, 3))
            hsv[:,:,0] = h
            hsv[:,:,1][v == 1] = 0
            hsv[:,:,2] = v
            
            rgb = colors.hsv_to_rgb(hsv)
            
            alpha = np.expand_dims(np.where(v == 0, 0, 1), 2)
            hsv = np.concatenate((hsv, alpha), axis=2)
            
            return rgb
        
        rgb=colour_wheel()
        if col < 2:
            gs_legend = gs[row, 2].subgridspec(3, 3)
            legend = fig.add_subplot(gs_legend[1,1])
            legend.text(0.5, -.7, 
                        f'Displacement\n(0 - {max_colorwheel_range_pm} pm)', 
                        transform=legend.transAxes,
                        horizontalalignment='center', 
                        fontsize=12, fontweight='bold')
        else:
            legend = fig.add_subplot(gs[-1])    
            legend.text(0.5, -.3, 
                        f'Displacement\n(0 - {max_colorwheel_range_pm} pm)', 
                        transform=legend.transAxes,
                        horizontalalignment='center', 
                        fontsize=12, fontweight='bold')
        legend.imshow(rgb)
        legend.set_xticks([])
        legend.set_yticks([])
        r=rgb.shape[0]/2
        circle = Wedge((r,r), r-5, 0, 360, width=5, color='black')
        legend.add_artist(circle)
        legend.axis('off')
        legend.axis('image')
        
        
        fig.subplots_adjust(hspace=0, wspace=0, 
                            top=0.9, bottom=0.01, 
                            left=0.01, right=0.99)
        
        return fig, axs
        
        