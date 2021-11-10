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

from matplotlib_scalebar.scalebar import ScaleBar

from scipy.optimize import minimize
from scipy import ndimage, fftpack
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)

import psutil
from joblib import Parallel, delayed
from tqdm import tqdm

from SingleOrigin.utils import (image_norm, img_ellip_param, gaussian_2d,
                                fit_gaussian2D, watershed_segment)        
#%%
class AtomicColumnLattice:
    """Object class for quantification of atomic columns in HR STEM images.
    
    Class with methods for locating atomic columns in a HR STEM image using
    a reference lattice genenerated from a .cif file. 
    -Requires minimal parameter adjustmets by the user to achieve accurate 
    results. 
    -Provides a fast fitting algorithm with automatic parallel processing.
    -Automatically groups close atomic columns for simultaneous fitting.
    -Some attributes are assigned values or modified by the various class 
    methods.
    
    Parameters
    ----------
    image : 2D array_like
        The STEM image.
    unitcell : UnitCell class object
        Instance of the UnitCell() class with project_uc_2d() method applied.
    resolution : float
        The resolution of the microscope and imaging mode in Angstroms.
        Default 0.8.
    origin_atom_column : int
        The DataFrame row index (in 'unitcell.at_cols') of the atom column 
        that is later picked by the user to register the reference lattice. 
        If None, the closest atom column to the unit cell origin is 
        automatically chosen.
        Default: None.
    
    Attributes
    ----------
    image : The input STEM image.
    resolution : The input resolution.
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
    fitting_masks : The last set of masks used for fitting  atom columns. Has
        the same shape as image.
    a1_star, a2_star : The reciprocal basis vectors in FFT pixel coordinates.
    a1, a2 : The real space basis vectors in image pixel coordinates.
    trans_mat : The transformation matrix from fractional to image pixel
        coordinates.
    pixel_size : The estimated pixel size using the reference lattice basis
        vectors and lattice parameter values from the .cif file.
    
    Methods
    -------
    fft_get_basis_vect(a1_order=1, a2_order=1, sigma=5): 
        Find basis vectors for the image from the FFT. 
    define_reference_lattice(a1_var='u', a2_var='v', diff_filter = None,
                             sub_latt_criteria=None, sub_latt_list=None): 
        Registers a reference lattice to the image.
    fit_atom_columns(diff_filter=None, grouping_filter=None, 
                     local_thresh_factor = 0.95, buffer=20,
                     filter_by='elem', sites_to_fit='all'):
        Algorithm for fitting 2D Gaussians to HR STEM image.
    refine_reference_lattice(filter_by='elem', 
                             sites_to_use='all',  =None):
        Refines the reference lattice on fitted column positions.
    rotate_image_and_data(align_dir='horizontal', align_basis='a1'):
        Rotates the image and data to align a basis vector to image edge.
    select_origin():
        Select origin for the reference lattice. Used by 
        define_reference_lattice() method.
    """
    
    def __init__(self, image, unitcell, resolution = 0.8, 
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
        
        self.resolution = resolution
        self.h, self.w = self.image.shape
        self.unitcell_2D = unitcell.at_cols
        self.a_2d = unitcell.a_2d
        self.at_cols = pd.DataFrame()
        self.at_cols_uncropped = pd.DataFrame()
        self.x0, self.y0 = np.nan, np.nan
        self.fitting_masks = np.zeros(image.shape)
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
        
    
    def fft_get_basis_vect(self, a1_order=1, a2_order=1, sigma=5):
        """Measure crystal basis vectors from the image FFT.
        
        Finds peaks in the image FFT and displays for graphical picking. 
        After the user chooses two peaks for the reciprocal basis vectors, 
        finds peaks related by vector additions and refines reciprocal lattice 
        on these positions. 
        
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
        U = min(1000, int(m/2))
        crop_dim = 2*U
        
        fft_der = image_norm(-gaussian_laplace(self.fft, sigma))
        masks, num_masks, slices, spots = watershed_segment(fft_der)
        spots.loc[:, 'stdev'] = ndimage.standard_deviation(fft_der, masks, 
                                            index=np.arange(1, num_masks+1))
        spots_ = spots[(spots.loc[:, 'stdev'] > 0.003)].reset_index(drop=True)
        xy = spots_.loc[:,'x':'y'].to_numpy()
        
        origin = np.array([U, U])
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('''Pick reciprocal basis vectors''',
                  fontdict = {'color' : 'red'})
        ax.set_ylim(bottom = U+U/4, top = U-U/4)
        ax.set_xlim(left = U-U/4, right = U+U/4)
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

        ax.set_ylim(bottom = U+U/4, top = U-U/4)
        ax.set_xlim(left = U-U/4, right = U+U/4)
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
        
        
    def select_origin(self):
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
        crop_view = np.max(np.abs(self.trans_mat)) / np.min([h,w]) * 4
        fig, ax = plt.subplots(figsize=(10,10))
        message=('Pick an atom column of the reference atom column type'
                 +' (white outlined position)')
        ax.set_title(str(message), fontdict={'color':'red'}, wrap=True)
        ax.set_ylim(bottom = h/2+h*crop_view, top = h/2-h*crop_view)
        ax.set_xlim(left = w/2-w*crop_view, right = w/2+w*crop_view)
        ax.imshow(self.image, cmap='gray')
        
        self.unitcell_2D['x_ref'] = ''
        self.unitcell_2D['y_ref'] = ''
        self.unitcell_2D.loc[:, 'x_ref': 'y_ref'] = self.unitcell_2D.loc[
            :, 'u':'v'].to_numpy() @ self.trans_mat
        
        rect_params = [w/2+w*crop_view, 
                       h/2+h*crop_view, 
                       -w*crop_view/2, 
                       -h*crop_view/2]
        
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
                    c=color_list, cmap='RdYlGn', s=20, zorder=10)
        
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
        
    
    def define_reference_lattice(self, LoG_sigma=None):
        
        """Register reference lattice to image.
        
        User chooses appropriate atomic column to establish the reference 
        lattice origin. Rough scaling and orientation are initially defined
        as derived from fft_get_basis_vect() method and then pre-refined by
        local peak detection.
        
        Parameters
        ----------
        LoG_sigma : int or float
            The Laplacian of Gaussian sigma value to use for peak sharpening.
            If None, calculated by: pixel_size / resolution * 0.5.
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
            & ((type(self.resolution) == float) 
               | (type(self.resolution) == int))):
            LoG_sigma = self.resolution / self.pixel_size * 0.5 
        
        (x0, y0) = self.select_origin()

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
            img_LoG,  buffer=0, min_dist=min_dist/4)
        
        coords = peaks.loc[:, 'x':'y'].to_numpy()
        print('prior basis vectors:', 
              f'[[{self.trans_mat[0,0]:.{4}f} {self.trans_mat[0,1]:.{4}f}]',
              f' [{self.trans_mat[1,0]:.{4}f} {self.trans_mat[1,1]:.{4}f}]]'
              +'\n', sep='\n')
        
        init_inc = int(np.min(np.max(np.abs(at_cols.loc[:, 'u' :'v']),
                                            axis=0))/10)
        
        if init_inc < 3: init_inc = 3
        
        for lim in [init_inc * i for i in [1,2,4]]:
            filtered = at_cols[(np.abs(at_cols.u) <= lim) & 
                                (np.abs(at_cols.v) <= lim)]
            
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
            
        self.at_cols_uncropped = copy.deepcopy(at_cols)
        
        print('refined basis vectors:', 
              f'[[{self.trans_mat[0,0]:.{4}f} {self.trans_mat[0,1]:.{4}f}]',
              f' [{self.trans_mat[1,0]:.{4}f} {self.trans_mat[1,1]:.{4}f}]]'
              +'\n', sep='\n')
                
        
    def fit_atom_columns(self, buffer=0,local_thresh_factor=1, 
                         diff_filter='auto', grouping_filter='auto',
                         filter_by='elem', sites_to_fit='all'):
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
        diff_filter : int or float
            The Laplacian of Gaussian sigma value to use for peak sharpening
            for defining peak regions via the Watershed segmentation method.
            Should be approximately pixel_size / resolution / 2. If 'auto',  
            calculated using self.pixel_size and self.resolution.
            Default 'auto'. 
        grouping_filter : int or float
            The Gaussian sigma value to use for peak grouping by blurring, 
            then creating image segment regions with watershed method. 
            Should be approximately pixel_size / resolution * 0.5. If 'auto', 
            calculated using self.pixel_size and self.resolution. If
            simultaneous fitting of close atom columns is not desired, set
            to None.
            Default: 'auto'.
        local_thresh_factor : float
            Removes background from each segmented region by thresholding. 
            Threshold value determined by finding the maximum value of edge 
            pixels in the segmented region and multipling this value by the 
            local_thresh_factor value. The filtered image is used for this 
            calculation. Default 0.95.
        buffer : int
            Distance defining the image border used to ignore atom columns 
            whose fits my be questionable.
        filter_by : str
            'at_cols' column to use for filtering to fit only a subset
            of the atom colums.
            Default 'elem'
        sites_to_fit : str ('all') or list of strings
            The criteria for the sites to fit, e.g. a list of the elements to 
            fit: ['Ba', 'Ti']
            Default 'all'
             
        Returns
        -------
        None.
            
        """
        
        print('Creating atom column masks...')
        
        self.buffer = buffer
        
        if self.at_cols.shape[0] == 0:
            at_cols = self.at_cols_uncropped.copy()
        else:
            at_cols = self.at_cols.copy()
            at_cols = pd.concat([at_cols, self.at_cols_uncropped.loc[
                [i for i in self.at_cols_uncropped.index.tolist()
                 if i not in at_cols.index.tolist()], 
                :]])
            
        if diff_filter == 'auto':
            if ((type(self.resolution) == float) 
                | (type(self.resolution) == int)):
                
                diff_filter = self.resolution / self.pixel_size * 0.5
            
            else:
                raise Exception('"resolution" must be defined for the class '
                                + 'instance to enable "diff_filter" '
                                + 'auto calculation.')
        elif type(diff_filter) != float and type(diff_filter) != int:
            raise Exception('"diff_filter" must be "auto" or a positive float '
                            + 'or int value.')
            
        if grouping_filter == 'auto': 
            if ((type(self.resolution) == float) 
                | (type(self.resolution) == int)):
        
                grouping_filter = self.resolution / self.pixel_size * 0.5
            
            else:
                raise Exception('"resolution" must be defined for the class '
                                + 'instance to enable "diff_filter" '
                                + 'auto calculation.')
                
        img_LoG = image_norm(-gaussian_laplace(self.image, diff_filter, 
                                                truncate=4))
        
        if grouping_filter == None or grouping_filter == 0: 
            img_gauss = img_LoG
            
        else: img_gauss = image_norm(gaussian_filter(self.image, grouping_filter,
                                                     truncate=4))
        
        if sites_to_fit != 'all':
            at_cols = at_cols[at_cols.loc[:, filter_by].isin(sites_to_fit)]
        
        """Find minimum distance (in pixels) between atom columns for peak
        detection neighborhood"""
        poss = self.unitcell_2D.loc[:, 'u':'v'].to_numpy()
        poss = np.concatenate(([poss + [i,j] 
                               for i in range(-1, 2)
                               for j in range(-1, 2)])) @ self.trans_mat
        dists = np.linalg.norm(np.array([poss - pos for pos in poss]), axis=2)
        min_dist = (np.floor(np.amin(dists, initial=np.inf, where=dists>0) 
                            / 2.2) - 1).astype(int)
        
        # print(f'minimum distance used for peak detection: {min_dist*2 + 1} '
        #       + 'pixels \n')
        
        """Apply Watershed segmentation to generate fitting masks"""
        diff_masks, _, _, xy_peak = watershed_segment(img_LoG, 
            local_thresh_factor = local_thresh_factor, 
            watershed_line=True, min_dist=min_dist)
        
        """Gaussian blur to group columns for simultaneous fitting"""
        grouping_masks, num_grouping_masks, slices_Gauss, _ = watershed_segment(
            img_gauss, local_thresh_factor = 0, watershed_line=True,
            min_dist=min_dist)
        
        """Use local peaks in img_LoG and match to reference lattice.
        These points will be initial position guesses for fitting"""
        xy_peak = xy_peak.loc[:, 'x':'y'].to_numpy()
        xy_ref = at_cols.loc[:, 'x_ref':'y_ref'].to_numpy()
        
        vects = np.array([xy_peak - xy for xy in xy_ref])
        norms = np.linalg.norm(vects, axis=2)
        inds = np.argmin(norms, axis=1)
        xy_peak = np.array([xy_peak[ind] for ind in inds])
        
        """If the difference between detected peak position and reference 
        position is greater than the resolution, the reference is taken
        as initial guess."""
        mask = (np.min(norms, axis=1) < self.resolution/self.pixel_size*0.75
                ).reshape((-1, 1))
        mask = np.concatenate((mask, mask), axis=1)
        xy_peak = np.where(mask, xy_peak, xy_ref)
        
        """Find corresponding mask (from both LoG and Gauss filtering) for each 
            peak"""
                
        LoG_masks_to_peaks = ndimage.map_coordinates(diff_masks, 
                                                     np.flipud(xy_peak.T), 
                                                     order=0).astype(int)
        
        Gauss_masks_to_peaks = ndimage.map_coordinates(grouping_masks, 
                                                       np.flipud(xy_peak.T), 
                                                       order=0).astype(int)
        
        LoG_masks_used = np.unique(LoG_masks_to_peaks)
        Gauss_masks_used = np.unique(Gauss_masks_to_peaks)
        
        """Save all masks which correspond to at least one reference lattice
            point"""
        fitting_masks = np.where(np.isin(diff_masks, LoG_masks_used), 
                                  diff_masks, 0)
        
        grouping_masks = np.where(np.isin(grouping_masks, Gauss_masks_used), 
                                  grouping_masks, 0)
        self.fitting_masks = np.where(fitting_masks >= 1, fitting_masks, 0)
        self.grouping_masks = np.where(grouping_masks >= 1, grouping_masks, 0)
        
        """Find sets of reference columns for each grouping mask"""
        peak_groupings = [[mask_num, 
                           np.argwhere(Gauss_masks_to_peaks==mask_num
                                       ).flatten()]
                          for mask_num in Gauss_masks_used if mask_num != 0]
        
        group_sizes, counts = np.unique([match[1].shape[0] 
                                         for match in peak_groupings],
                                        return_counts=True)
        
        if np.max(group_sizes) > 1:
            print('Atomic columns grouped for simultaneous fitting:')
            for i, size in enumerate(group_sizes):
                print(f'{counts[i]}x {size}-column groups')
            print('\n', 
                  'Fitting routine will be faster without simultaneous ',
                  'fitting, but may be less accurate. If faster fitting is ',
                  'needed set "grouping_filter=None". \n',
                  sep='\n')
        
        sl_start = np.array([[slices_Gauss[i][1].start, 
                              slices_Gauss[i][0].start] 
                             for i in range(num_grouping_masks)])
        
        """Pack image slices and metadata together for the fitting routine"""
        print('Preparing data for fitting...')
        args_packed = [[self.image[slices_Gauss[mask_num-1][0],
                                   slices_Gauss[mask_num-1][1]],
                        fitting_masks[slices_Gauss[mask_num-1][0],
                                      slices_Gauss[mask_num-1][1]],
                        LoG_masks_to_peaks[inds],
                        sl_start[mask_num - 1],
                        xy_peak[inds, :].reshape((-1, 2)), 
                        at_cols.index.to_numpy()[inds],
                        mask_num]
                       for [mask_num, inds]  in tqdm(peak_groupings)]
        
        """Define column fitting function for image slices"""
        

        def fit_column(args):
            [img_sl, mask_sl, log_mask_num, xy_start, xy_peak, inds, mask_num
             ] = args
            counts = []
            num = xy_peak.shape[0]
            masks = np.zeros(mask_sl.shape)
            for mask_num in log_mask_num: 
                if mask_num == 0: continue
                mask = np.where(mask_sl == mask_num, mask_num, 0)
                # mask = np.where(ndimage.morphology.binary_dilation(
                #     ndimage.morphology.binary_erosion(mask, iterations=2,
                #                                       border_value=1), 
                #     iterations=2), mask_num, 0)
                
                masks += mask
                
            # masks = np.where(np.isin(masks, log_mask_num), masks, 0)
            img_msk = img_sl * np.where(masks > 0, 1, 0) 
            
            if num == 1:
                [x0, y0] = (xy_peak - xy_start).flatten()
                _, _, _, theta, sig_1, sig_2 = img_ellip_param(img_msk)
                I0 = (np.average(img_msk[img_msk != 0])
                      - np.std(img_msk[img_msk != 0]))
                A0 = np.max(img_msk) - I0
 
                p0 = np.array([[x0, y0, sig_1, sig_2, np.radians(theta), 
                                A0, I0]])
                
                params = fit_gaussian2D(img_msk, p0)
                
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
                masks_to_peaks = ndimage.map_coordinates(masks, 
                                                         np.flipud(x0y0.T), 
                                                         order=0).astype(int)
                
                sig_1 = []
                sig_2 = []
                theta = []
                Z0 = []
                I0 = []
                A0 = []
                
                for i, mask_num in enumerate(masks_to_peaks):
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
                
                p0 = np.array([x0, y0, sig_1, sig_2, theta, A0, I0]).T
                p0 = np.append(p0.flatten(), Z0)
                
                params = fit_gaussian2D(img_msk, p0, masks)
                
                params = np.array([params[:,0] + xy_start[0],
                                    params[:,1] + xy_start[1],
                                    params[:,2],
                                    params[:,3],
                                    np.degrees(params[:,4]),
                                    params[:,5],
                                    params[:,6]]).T
            
            return params, masks
           
        """Run fitting routine"""
        print('Fitting atom columns with 2D Gaussians...')
        t0 = time.time()
        if len(args_packed) >= 50:
            """Large data set: use parallel processing"""
            print('Using parallel processing')
            n_jobs = psutil.cpu_count(logical=False)
            
            results_ = Parallel(n_jobs=n_jobs)(delayed(fit_column)(arg) 
                                              for arg in tqdm(args_packed))
            results = np.concatenate([np.concatenate(
                (result[0], args_packed[i][5].reshape(-1, 1)), axis=1)
                for i, result in enumerate(results_)])
            fitting_masks[:,:] = 0
            for i, result in enumerate(results_):
                mask_sl = result[1]
                h, w = mask_sl.shape
                x0, y0 = args_packed[i][3]
                fitting_masks[y0:y0+h, x0:x0+w] += mask_sl
                
        else:
            """Small data set: use serial processing"""
            results_ = [fit_column(arg) for arg in tqdm(args_packed)]
        
            results = np.concatenate([np.concatenate((
                result[0], args_packed[i][5].reshape(-1, 1)), axis=1)
                for i, result in enumerate(results_)])
            fitting_masks[:,:] = 0
            for i, result in enumerate(results_):
                mask_sl = result[1]
                (h, w) = mask_sl.shape
                x0, y0 = args_packed[i][3]
                fitting_masks[y0:y0+h, x0:x0+w] += mask_sl
                                                      
        t_elapse = time.time() - t0
        
        print(f'Done. Fitting time: {int(t_elapse // 60)} min '
              +f'{t_elapse % 60 :.{1}f} sec')
       
        """Process results and return"""
       
        col_labels = ['x_fit', 'y_fit', 'sig_1', 'sig_2',
                     'theta', 'peak_int', 'bkgd_int', 'total_col_int']
        if not col_labels[0] in at_cols.columns:
           empty = pd.DataFrame(index=at_cols.index.tolist(), 
                                columns = col_labels)
           at_cols = at_cols.join(empty)
       
        results = pd.DataFrame(data=results[:, :-1], 
                              index=results[:,-1].astype(int), 
                              columns=col_labels[:-1]).sort_index()
        results.loc[:, 'total_col_int'] = (2 * np.pi * results.peak_int 
                                           * results.sig_1 * results.sig_2)
       
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
        self.at_cols_uncropped = at_cols.copy()
        
        self.at_cols = at_cols[((at_cols.x_ref >= buffer) &
                                (at_cols.x_ref <= self.w - buffer) &
                                (at_cols.y_ref >= buffer) &
                                (at_cols.y_ref <= self.h - buffer))].copy()

        self.fitting_masks = np.where(fitting_masks >= 1, 1, 0)
        self.grouping_masks = np.where(grouping_masks >= 1, 1, 0)
        
        
    def refine_reference_lattice(self, filter_by='elem', 
                                 sites_to_use='all', outliers=None):
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
        outliers : None or float or int
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
            
        if outliers == None:
            outliers = 1 / self.pixel_size
            
        else:
            outliers /= self.pixel_size * 100
            
        filtered = filtered[np.linalg.norm(
            filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
            - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outliers].copy()

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
            
        
    def plot_fitting_residuals(self):
        """Plots image residuals from the atomic column fitting.
        
        Plots image residuals. Subtracts gaussian fits from each corresponding 
        masked region.
        
        Parameters
        ----------
        None.
             
        Returns
        -------
        None.
            
        """
        
        outliers = 1/self.pixel_size
        filtered = self.at_cols_uncropped[np.linalg.norm(
            self.at_cols_uncropped.loc[:, 'x_fit':'y_fit'].to_numpy()
            - self.at_cols_uncropped.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outliers].copy()
        
        xy_peak = filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
        
        fitting_masks, n = ndimage.label(self.fitting_masks)
        fitting_masks_to_peaks = ndimage.map_coordinates(
            fitting_masks, np.flipud(xy_peak.T),  order=0).astype(int)
        for i, num in enumerate(fitting_masks_to_peaks):
            if num == 0:
                print('found zero, fixing...')
                for j in range(n):
                    mask = np.where(fitting_masks == j+1, fitting_masks, 0)
                    mask_dilated = ndimage.morphology.binary_dilation(
                        mask, iterations=1) * (j+1)
                    mask_new = ndimage.map_coordinates(
                        mask_dilated, 
                        np.flipud(xy_peak[i,:].reshape((-1,2)).T),  
                        order=0).astype(int)
                    if mask_new != 0: break
                fitting_masks_to_peaks[i] = mask_new
                print(fitting_masks_to_peaks[i])
                
            if fitting_masks_to_peaks[i] == 0:
                print('could not find matching mask')
        
        grouping_masks, m = ndimage.label(self.grouping_masks)
        grouping_masks_to_peaks = ndimage.map_coordinates(
            grouping_masks, np.flipud(xy_peak.T),  order=0).astype(int)
        
        grouping_masks_used = np.unique(grouping_masks_to_peaks)
        
        peak_groupings = [[mask_num, 
                           np.argwhere(grouping_masks_to_peaks==mask_num
                                       ).flatten()]
                          for mask_num in grouping_masks_used 
                          if mask_num != 0]
        
        print('Calculating residuals:')
        def peak_residuals(mask_num, inds, fitting_masks_to_peaks):
            masks_labeled = np.where(np.isin(fitting_masks, 
                             fitting_masks_to_peaks[inds]), 
                             fitting_masks, 0).ravel()
            y, x = np.indices(self.image.shape)
            
            unmasked_data = np.nonzero(masks_labeled)
            masks_labeled = np.take(masks_labeled, unmasked_data)
            x = np.take(x.ravel(), unmasked_data)
            y = np.take(y.ravel(), unmasked_data)
            
            peak = np.zeros(masks_labeled.shape)
            
            for ind in inds:
                ind_ = filtered.index.tolist()[ind]
                row = filtered.loc[ind_, :]
                
                peak += gaussian_2d(x, y, row.x_fit, row.y_fit,
                                    row.sig_1, row.sig_2, 
                                    row.theta, row.peak_int, 0)
                
                if fitting_masks_to_peaks[ind] != 0:
                    peak += (np.where(masks_labeled == 
                                      fitting_masks_to_peaks[ind], 1, 0) 
                             * row.bkgd_int)
                    
            return x, y, peak
            
        n_jobs = psutil.cpu_count(logical=False)
        
        results = Parallel(n_jobs=n_jobs)(delayed(peak_residuals
                                                  )(mask_num, inds, 
                                                    fitting_masks_to_peaks) 
                                          for [mask_num, inds] in 
                                          tqdm(peak_groupings))
        y, x = np.indices(self.image.shape)
        buffer_mask = np.where(
            ((y < self.buffer) | (y > self.h - self.buffer) |
             (x < self.buffer) | (x > self.w - self.buffer)), 0, 1)
        
        self.residuals = self.image * self.fitting_masks
        
        for x, y, peak in results:
            self.residuals[y, x] -= peak
            
        self.residuals *= buffer_mask
        
        cmap_lim = np.max(np.abs(self.residuals))
        
        fig,axs = plt.subplots(ncols=1,figsize=(10,10), tight_layout=True)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.imshow(self.residuals, cmap='bwr', 
                    norm=Normalize(vmin=-cmap_lim, vmax=cmap_lim))
        axs.scatter(self.at_cols.loc[:,'x_fit'], self.at_cols.loc[:,'y_fit'],
                    color='red', s=4)        
        R = self.residuals.ravel()
        
        print(f'Sum of the squared residuals: {R @ R.T :.{3}f}')
        print(f'Absolute sum of the residuals: {np.sum(R) :.{3}e}')

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
        rot_.fitting_masks = ndimage.rotate(rot_.fitting_masks, 
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
        
    def plot_atom_column_positions(self, filter_by='elem', sites_to_fit='all',
                                   fit_or_ref='fit', outliers=None,
                                   plot_masked_image=False):
        """Plot fitted or reference atom colum positions.
        
        Parameters
        ----------
        filter_by : str
            'at_cols' column to use for filtering to plot only a subset
            of the atom colums.
            Default 'elem'
        sites_to_fit : str ('all') or list of strings
            The criteria for the sites to print, e.g. a list of the elements 
            to plot: ['Ba', 'Ti']
            Default 'all'
        fit_or_ref : str ('fit' or 'ref')
            Which poisitions to plot, the 
            Default: 'fit'
        plot_masked_image : bool
            Whether to plot the masked image (shows only the regions used
            for fitting). If False ,unmasked image is plotted.
            Default: False
             
        Returns
        -------
        None.
            
        """

        if sites_to_fit == 'all' or ['all']:
            filtered = self.at_cols
            
        else:
            filtered = self.at_cols[self.at_cols.loc[:, filter_by]
                                    .isin(sites_to_fit)].copy()
            
        if outliers == None:
            outliers = 1 / self.pixel_size #Outliers defined as 1 Angstrom
            
        else:
            outliers /= self.pixel_size * 100
            
        filtered = filtered[np.linalg.norm(
            filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
            - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outliers].copy()
            
        if fit_or_ref == 'fit':
            xcol, ycol = 'x_fit', 'y_fit'
        elif fit_or_ref == 'ref':
            xcol, ycol = 'x_ref', 'y_ref'
                                
        fig,axs = plt.subplots(ncols=1,figsize=(10,15), tight_layout=False)
        
        if plot_masked_image == True:
            axs.imshow(self.image * self.fitting_masks, cmap='gray')
        else:
            axs.imshow(self.image, cmap='gray')
        axs.set_xticks([])
        axs.set_yticks([])
    
        unitcell = filtered[(filtered['u'] // 1 == 0) & 
                               (filtered['v'] // 1 == 0)]
    
        color_code = {k:v for v, k in 
                      enumerate(np.sort(unitcell.loc[:, filter_by].unique()))}
        color_list = [color_code[site] for site in filtered.loc[:, filter_by]]
    
        cmap = plt.cm.RdYlGn
        
        axs.scatter(filtered.loc[:, xcol], filtered.loc[:, ycol],
                    c=color_list, s=4, cmap=cmap, zorder=2)
    
        color_index = [Circle((30, 7), 3, color=cmap(c)) 
                       for c in np.linspace(0,1, num=len(color_code))]
    
        def make_legend_circle(legend, orig_handle,
                                xdescent, ydescent,
                                width, height, fontsize):
            p = orig_handle
            return p
    
        axs.legend(handles = color_index,
                    labels = list(color_code.keys()),
                    handler_map={Circle : HandlerPatch(patch_func=
                                                      make_legend_circle),},
                    fontsize=20, loc='lower left', bbox_to_anchor=[1.02, 0],
                    facecolor='grey')
    
        axs.arrow(self.x0, self.y0, self.a1[0], self.a1[1],
                      fc='red', ec='red', width=0.1, 
                      length_includes_head=True,
                      head_width=2, head_length=3, label=r'[001]')
        axs.arrow(self.x0, self.y0, self.a2[0], self.a2[1],
                      fc='green', ec='green', width=0.1, 
                      length_includes_head=True,
                      head_width=2, head_length=3, label=r'[110]')
        
    def plot_disp_vects(self, filter_by='elem', sites_to_plot='all', 
                        titles=None, x_crop=None, y_crop=None,
                        scalebar=True, scalebar_len_nm=2,
                        arrow_scale_factor = 1,
                        outliers = None, max_colorwheel_range_pm=None,
                        plot_fit_points=False, plot_ref_points=False):
        
        if sites_to_plot == 'all':
            sites_to_plot = self.at_cols.loc[:, filter_by].unique().tolist()
            sites_to_plot.sort()
        elif type(sites_to_plot) != list:
            raise Exception('"sites_to_plot" must be either "all" or a list')
            
        if outliers == None:
            outliers = 1 / self.pixel_size
            
        else:
            outliers /= self.pixel_size * 100
            
        filtered = self.at_cols[np.linalg.norm(
            self.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy()
            - self.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outliers].copy()
       
        if max_colorwheel_range_pm == None:
            filtered = filtered[filtered.loc[:, filter_by]
                                    .isin(sites_to_plot)]
            dxy = (filtered.loc[:,'x_fit':'y_fit'].to_numpy()
                   - filtered.loc[:,'x_ref':'y_ref'].to_numpy())
            mags = np.linalg.norm(dxy, axis=1) * self.pixel_size * 100
            avg = np.mean(mags)
            std = np.std(mags)
            max_colorwheel_range_pm = int(np.ceil((avg + 3*std)/5) * 5)
        
        if x_crop==None:
            x_crop=[0, self.w]
        if y_crop==None:
            y_crop=[self.h, 0]
        if y_crop[0] < y_crop[1]:
            y_crop = [y_crop[1], y_crop[0]]
        
        n_plots = len(sites_to_plot)
        if n_plots > 12:
            raise Exception('The number of plots exceeds the limit of 8.')
        
        if n_plots <= 3:
            nrows = 1
            ncols = n_plots
            width_ratios=[3] * ncols + [1]
            
        elif n_plots <= 8:
            nrows = 2
            ncols = np.ceil(n_plots/2).astype(int)
            width_ratios=[3] * ncols + [1]
            
        elif n_plots <= 12:
            nrows = 3
            ncols = np.ceil(n_plots/3).astype(int)
            width_ratios=[3] * ncols + [1]
            
        else:
            raise Exception('The number of plots exceeds the limit of 12.')
        
        figsize=(ncols * 5 + 2, 5 * nrows + 2)
    
        fig = plt.figure(figsize=figsize)#, tight_layout=True)
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, 
                              width_ratios=width_ratios,
                              height_ratios=[3 for _ in range(nrows)], 
                              wspace=0.05)
        if x_crop == None:
            x_crop=[0, self.w] 
        if y_crop == None:
            y_crop=[self.h, 0],
        
        for ax, site in enumerate(sites_to_plot):
            row = ax // 3
            col = ax % 3
            axs = fig.add_subplot(gs[row, col])
            axs.imshow(self.image, cmap='gray')

            axs.set_xlim(x_crop[0], x_crop[1])
            axs.set_ylim(y_crop[0], y_crop[1])
            
            axs.set_xticks([])
            axs.set_yticks([])

            if ax == 0 and scalebar == True:
                scalebar = ScaleBar(self.pixel_size/10,
                                    'nm', location='lower right', pad=0.4, 
                                    fixed_value=scalebar_len_nm, 
                                    font_properties={'size':10}, 
                                    box_color='lightgrey', width_fraction=0.02,
                                    sep=2)
                axs.add_artist(scalebar)
                
            sub_latt = filtered[filtered.loc[:, filter_by] == site]
            h = y_crop[0] - y_crop[1]
            title = axs.text(x_crop[0] + 0.02*h, y_crop[1] + 0.02*h, 
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
                axs.scatter(sub_latt.loc[:,'x_fit'], sub_latt.loc[:,'y_fit'],
                            color='blue', s=1)
            if plot_ref_points: 
                axs.scatter(sub_latt.loc[:,'x_ref'], sub_latt.loc[:,'y_ref'],
                            color='red', s=1)
            cb = axs.quiver(sub_latt.loc[:,'x_fit'], sub_latt.loc[:,'y_fit'], 
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
        else:
            legend = fig.add_subplot(gs[-1])    
        legend.imshow(rgb)
        legend.set_xticks([])
        legend.set_yticks([])
        r=rgb.shape[0]/2
        circle = Wedge((r,r), r-5, 0, 360, width=5, color='black')
        legend.add_artist(circle)
        legend.axis('off')
        legend.axis('image')
        legend.text(0.5, -.35, 
                    f'Displacement\n(0 - {max_colorwheel_range_pm} pm)', 
                    transform=legend.transAxes,
                    horizontalalignment='center', 
                    fontsize=12, fontweight='bold')
        # if col < 2:
        #     legend.set_position([1/3, 2/3, 1/3, 2/3])
        
        fig.subplots_adjust(hspace=0, wspace=0, 
                            top=0.9, bottom=0.01, 
                            left=0.01, right=0.99)
        
        
        
