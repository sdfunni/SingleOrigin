import copy
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle as Circle
from matplotlib.legend_handler import HandlerPatch

from scipy.optimize import minimize
from scipy import ndimage, fftpack
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)
from scipy.optimize import minimize

import psutil
from joblib import Parallel, delayed
import time
from tqdm import tqdm

from SingleOrigin.utils import *
        
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
    basis_offset_frac : array_like with shape (2,)
        The position in fractional coordinates of a prominent atomc column.
        This type of atom column is later picked by the user to register the
        reference lattice.
        Default [0, 0].
    
    Attributes
    ----------
    image : The input STEM image.
    resolution : The input resolution.
    basis_offset_frac : The input basis_offset_frac.
    basis_offset_pix : The basis offset in image pixel coordinates.
    h, w : The height and width of the image.
    at_cols : DataFrame containing the he reference lattice and position plus 
        other fitting data for the atomic columns in the image.
    at_cols_uncropped : The reference lattice atom columns before removing
        positions close to the image edges (as defined by the buffer arg in 
        fit_atom_columns() method).
    unitcell_2D : DataFrame containing the projected crystallographic unit 
        cell atom column positions.
    a_2d : The matrix of basis vectors in the cartesian reference frame with
        units of Angstroms.
    x0, y0 : The image coordinates of the origin of the reference lattice 
        (in pixels).
    all_masks : The last set of masks used for fitting  atom columns. Has
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
    define_reference_lattice(a1_var='u', a2_var='v', LoG_sigma = None,
                             sub_latt_criteria=None, sub_latt_list=None): 
        Registers a reference lattice to the image.
    fit_atom_columns(LoG_sigma=None, Gauss_sigma=None, 
                     edge_max_threshold = 0.95, buffer=20,
                     filter_by='elem', sites_to_fit='all'):
        Algorithm for fitting 2D Gaussians to HR STEM image.
    refine_reference_lattice(filter_by='elem', 
                             sites_to_use='all', outliers=None):
        Refines the reference lattice on fitted column positions.
    rotate_image_and_data(align_dir='horizontal', align_basis='a1'):
        Rotates the image and data to align a basis vector to image edge.
    select_origin():
        Select origin for the reference lattice. Used by 
        define_reference_lattice() method.
    """
    
    def __init__(self, image, unitcell, resolution = 0.8, 
                 basis_offset_frac=[0,0]):
        self.image = image
        self.resolution = resolution
        self.basis_offset_frac = np.array(basis_offset_frac)
        self.h, self.w = self.image.shape
        self.unitcell_2D = unitcell.at_cols
        self.a_2d = unitcell.a_2d
        self.basis_offset_pix = None
        self.at_cols = pd.DataFrame()
        self.at_cols_uncropped = pd.DataFrame()
        self.x0, self.y0 = np.nan, np.nan
        self.all_masks = np.zeros(image.shape)
        self.a1, self.a2 = None, None
        self.a1_star, self.a2_star = None, None
        self.trans_mat = None
        self.pixel_size = None
    
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
        h, w = self.image.shape
        m = min(h,w)
        U = min(1000, int(m/2))
        crop_dim = 2*U
        
        image_square = self.image[int(h/2)-U : int(h/2)+U,
                          int(w/2)-U : int(w/2)+U]
        
        hann = np.outer(np.hanning(crop_dim),np.hanning(crop_dim))
        fft = fftpack.fft2(image_square*hann)
        fft = (abs(fftpack.fftshift(fft)))
        fft = image_norm(fft)
        self.fft = fft
        
        fft_der = image_norm(-gaussian_laplace(fft, sigma))
        masks, num_masks, slices, spots = watershed_segment(fft_der)
        spots.loc[:, 'stdev'] = ndimage.standard_deviation(fft_der, masks, 
                                            index=np.arange(1, num_masks+1))
        spots_ = spots[(spots.loc[:, 'stdev'] > 0.005)].reset_index(drop=True)
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
        
        print('done selecting')
        
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
                      fc='black', ec='black', width=0.1, length_includes_head=True,
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
        
        if (np.isnan(self.x0) and np.isnan(self.y0)):
            (x0, y0) = self.select_origin()

            print('pick coordinates:', [x0, y0])
            
            filt = image_norm(-gaussian_laplace(self.image, LoG_sigma))
            neighborhood = np.ones((7,7))
            local_max = np.fliplr(np.argwhere(maximum_filter(filt,
                                        footprint=neighborhood)==filt))

            [x0, y0] = local_max[np.argmin(np.linalg.norm(local_max 
                                                        - [x0, y0], axis=1))]
            
            print('detected peak coordinates:', [x0, y0])
            
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
                              columns = ['x_fit', 'y_fit', 'sig_maj', 
                                         'sig_min', 'sig_rat', 'theta', 
                                         'peak_int', 'bkgd_int'])
        at_cols = pd.concat([at_cols, empty], axis=1)
        
        ch_list = np.sort(at_cols.loc[:,lab].unique()).tolist()
        ch_list = {k: v for v, k in enumerate(ch_list)}
        channels = np.array([ch_list[site] for site in 
                             at_cols.loc[:, lab]])
        at_cols.loc[:, 'channel'] = channels
        
        '''Refine reference lattice on watershed mask CoMs'''
        print('Performing rough reference lattice refinement...')
        img_LoG = image_norm(-gaussian_laplace(self.image, LoG_sigma))
        masks, num_masks, slices, peaks = watershed_segment(img_LoG, 
                                                            buffer=0)
        
        coords = peaks.loc[:, 'x':'y'].to_numpy()
        print('prior basis', self.trans_mat)
        
        init_inc = int(np.min(np.max(np.abs(at_cols.loc[:, 'u' :'v']),
                                            axis=0))/10)
        
        if init_inc < 3: init_inc = 3
        
        for lim in [init_inc * i for i in [1,2,4]]:
            filtered = at_cols[(np.abs(at_cols.u) <= lim) & 
                                (np.abs(at_cols.v) <= lim)]
            
            if ((type(sub_latt_criteria) == str) and 
                (type(sub_latt_list) == list)):
                filtered = (filtered[filtered.loc[:,sub_latt_criteria]
                              .isin(sub_latt_list)])
            
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
            
        self.at_cols_uncropped = copy.deepcopy(at_cols)
        print('refined basis', self.trans_mat)
                
    def fit_atom_columns(self, LoG_sigma=None, Gauss_sigma=None, 
                         edge_max_threshold = 0.95, buffer=20,
                         filter_by='elem', sites_to_fit='all'):
        """Algorithm for fitting 2D Gaussians to HR STEM image.
        
        Uses Laplacian of Gaussian filter to isolate each peak by the 
        Watershed method. Gaussian filter blurs image to group closely spaced 
        peaks for simultaneous fitting. If simultaneous fitting is not 
        desired, set Gauss_sigma = None. Requires reference lattice or
        initial guesses for all atom columns in the image (i.e. 
        self.at_cols_uncropped must have values in 'x_ref' and 'y_ref' 
        columns). This can be achieved by running self.get_reference_lattice. 
        Stores fitting parameters for each atom column in self.at_cols.
        
        Parameters
        ----------
        LoG_sigma : int or float
            The Laplacian of Gaussian sigma value to use for peak sharpening.
            Should be approximately pixel_size / resolution * 0.5. If not 
            given, calculated using self.pixel_size and self.resolution.
            Default None. 
        Gauss_sigma : int or float
            The Gaussian sigma value to use for peak grouping by blurring. 
            Should be approximately pixel_size / resolution * 0.5. If not 
            given, calculated using self.pixel_size and self.resolution. If
            simultaneous fitting of close atom columns is not desired, set
            equal to 0.
            Default None.
        edge_max_threshold : float
            Removes background from each segmented region by thresholding. 
            Threshold value determined by finding the maximum value of edge pixels
            in the segmented region and multipling this value by the 
            edge_max_threshold value. The filtered image is used for this 
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
        
        if self.at_cols.shape[0] == 0:
            at_cols = self.at_cols_uncropped.copy()
        else:
            at_cols = self.at_cols.copy()
            at_cols = pd.concat([at_cols, self.at_cols_uncropped.loc[
                [i for i in self.at_cols_uncropped.index.tolist()
                 if i not in at_cols.index.tolist()], 
                :]])
        
        if ((Gauss_sigma == None) 
            & ((type(self.resolution) == float) 
               | (type(self.resolution) == int))):
        
            Gauss_sigma = self.resolution / self.pixel_size * 0.5
            
        if ((LoG_sigma == None) 
            & ((type(self.resolution) == float) 
               | (type(self.resolution) == int))):
            LoG_sigma = self.resolution / self.pixel_size * 0.5 
        
        
        if sites_to_fit != 'all':
            at_cols = at_cols[at_cols.loc[:, filter_by].isin(sites_to_fit)]
        
        img_LoG = image_norm(-gaussian_laplace(self.image, LoG_sigma))
        
        if Gauss_sigma == None or Gauss_sigma == 0: img_gauss = img_LoG
        else: img_gauss = image_norm(gaussian_filter(self.image, Gauss_sigma))
        
        """Apply Watershed segmentation to generate masks"""
        masks_LoG, _, _, _ = watershed_segment(img_LoG, 
            edge_max_threshold = edge_max_threshold, watershed_line=True)
        
        """Gaussian blur to group columns for simultaneous fitting"""
        masks_Gauss, num_masks_Gauss, slices_Gauss, _ = watershed_segment(
            img_gauss, edge_max_threshold = 0, watershed_line=True)
        
        """Find corresponding mask (from both LoG and Gauss filtering) for each 
            peak"""
        
        xy_peak = at_cols.loc[:, 'x_ref':'y_ref'].to_numpy()
        
        LoG_masks_to_peaks = ndimage.map_coordinates(masks_LoG, 
                                                     np.flipud(xy_peak.T), 
                                                     order=0).astype(int)
        
        Gauss_masks_to_peaks = ndimage.map_coordinates(masks_Gauss, 
                                                       np.flipud(xy_peak.T), 
                                                       order=0).astype(int)
        
        LoG_masks_used = np.unique(LoG_masks_to_peaks)
        Gauss_masks_used = np.unique(Gauss_masks_to_peaks)
        
        """Save all masks which correspond to at least one reference lattice
            point"""
        all_masks = np.where(np.isin(masks_LoG, LoG_masks_used), masks_LoG, 0)
        
        """Find sets of reference columns for each Gaussian mask"""
        peak_groupings = [[mask_num, 
                           np.argwhere(Gauss_masks_to_peaks==mask_num).flatten()]
                          for mask_num in Gauss_masks_used if mask_num != 0]
        
        max_inds = np.max([match[1].shape[0] for match in peak_groupings])
        print('Maximum number of atomic columns grouped for simultaneous '
              + 'fitting:', max_inds)
        
        sl_start = np.array([[slices_Gauss[i][1].start, 
                              slices_Gauss[i][0].start] 
                             for i in range(num_masks_Gauss)])
        
        """Pack image slices and metadata together for the fitting routine"""
        args_packed = [[(self.image[slices_Gauss[mask_num-1][0],
                                    slices_Gauss[mask_num-1][1]]
                         * np.isin(masks_LoG[slices_Gauss[mask_num-1][0],
                                             slices_Gauss[mask_num-1][1]],
                                   LoG_masks_to_peaks[inds])),
                        sl_start[mask_num - 1],
                        xy_peak[inds, :].reshape((-1, 2)), 
                        at_cols.index.to_numpy()[inds],
                        mask_num]
                       for [mask_num, inds]  in peak_groupings]
       
        """Define column fitting function for image slices"""
 
        def fit_column(args):
            [img_sl, xy_start, xy_peak, inds, mask_num] = args
            num = xy_peak.shape[0]
            
            if num == 1:
                eigvals, eigvects, x0, y0 = img_equ_ellip(img_sl)
                max_ind = np.argmax(eigvals)
                min_ind = 1 if max_ind==0 else 0
                sig_max = np.sqrt(np.abs(eigvals[max_ind]))
                sig_min = np.sqrt(np.abs(eigvals[min_ind]))
                if sig_min == 0: sig_min = 1 
                sig_r = sig_max / sig_min
                theta = -np.arcsin(np.cross(np.array([1,0]), 
                                            eigvects[:, max_ind]))
                I0 = (np.average(img_sl[img_sl != 0])
                      - np.std(img_sl[img_sl != 0]))
                A0 = np.max(img_sl) - I0
 
                p0 = np.array([[x0, y0, sig_max, sig_r, theta, A0, I0]])
                
                params = fit_gaussian2D(img_sl, p0)
                
                params = np.array([params[:,0] + xy_start[0],
                                    params[:,1] + xy_start[1],
                                    params[:,2],
                                    params[:,2]/params[:,3],
                                    params[:,3],
                                    np.degrees(params[:,4]),
                                    params[:,5],
                                    params[:,6]]).T
                
                return params
            
            if num > 1:
                eigvals, _, _, _ = img_equ_ellip(img_sl)
                max_ind = np.argmax(eigvals)
                min_ind = 1 if max_ind==0 else 0
                
                x0 = xy_peak[:, 0] - xy_start[0]
                y0 = xy_peak[:, 1] - xy_start[1]
                sig_max = [np.sqrt(np.abs(eigvals[min_ind]))] * num
                sig_r = [1] * num
                theta = [0] * num
                Z0 = [np.average(img_sl[img_sl != 0])
                      - np.std(img_sl[img_sl != 0])] 
                I0 = [0] * num
                A0 = [ndimage.map_coordinates(img_sl, 
                                              np.array([y0, x0]), order=0)[0] 
                      - Z0[0] for i in range(num)]
                
                if min(A0) < 0.1:
                    Z0 += min(A0) -0.1
                    A0 = [ndimage.map_coordinates(img_sl, np.array([y0, x0]), 
                                                  order=0)[0] 
                          - Z0[0] for i in range(num)]
                
                p0 = np.array([x0, y0, sig_max, sig_r, theta, A0, I0]).T
                p0 = np.append(p0.flatten(), Z0)
                
                params = fit_gaussian2D(img_sl, p0, method='trust-constr')
                
                params = np.array([params[:,0] + xy_start[0],
                                    params[:,1] + xy_start[1],
                                    params[:,2],
                                    params[:,2]/params[:,3],
                                    params[:,3],
                                    np.degrees(params[:,4]),
                                    params[:,5],
                                    params[:,6]]).T
                
                return params
           
        """Run fitting routine"""
        print('Fitting atom columns with 2D Gaussians...')
        t0 = time.time()
        if len(args_packed) >= 1e2:
            """Large data set: use parallel processing"""
            print('Using parallel processing')
            n_jobs = psutil.cpu_count(logical=False)
            
            results = Parallel(n_jobs=n_jobs)(delayed(fit_column)(arg) 
                                              for arg in tqdm(args_packed))
            
            results = np.concatenate([np.concatenate(
                (result, args_packed[i][3].reshape(-1, 1)), axis=1)
                for i, result in enumerate(results)])
       
        else:
            """Small data set: use serial processing"""
            results = np.concatenate([np.concatenate((fit_column(arg), 
                                                      arg[3].reshape(-1, 1)), 
                                                     axis=1)
                                      for arg in tqdm(args_packed)])
           
        print('Done. Fitting time:', time.time() - t0)
       
        """Process results and return"""
       
        col_labels = ['x_fit', 'y_fit', 'sig_maj', 'sig_min', 'sig_rat',
                     'theta', 'peak_int', 'bkgd_int']
        if not col_labels[0] in at_cols.columns:
           empty = pd.DataFrame(index=at_cols.index.tolist(), 
                                columns = col_labels)
           at_cols = at_cols.join(empty)
       
        results = pd.DataFrame(data=results[:, :-1], 
                              index=results[:,-1].astype(int), 
                              columns=col_labels).sort_index()
       
        at_cols.update(results)

        '''Convert values from dtype objects to ints, floats, etc:'''
        at_cols = at_cols.infer_objects()
        
        self.at_cols = at_cols[((at_cols.x_ref >= buffer) &
                                (at_cols.x_ref <= self.w - buffer) &
                                (at_cols.y_ref >= buffer) &
                                (at_cols.y_ref <= self.h - buffer))]

        self.all_masks = np.where(all_masks >= 1, 1, 0)
        
        
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
        sites_to_use : str or list of strings
            The sites to use for refinement. 'all' or a list of the site 
            labels.
            Default 'all'.
        outliers : None or float
            Criteria for removing outlier atomic column fits from the 
            refinement. The maximum difference between the fitted position
            and the corresponding reference lattice point. All positions
            with greater errors than this value will be removed. If None, 
            half of the smallest lattice parameter will be used.
            Default None.
             
        Returns
        -------
        None.
            
        """
                    
        if sites_to_use == 'all' or ['all']:
            filtered = self.at_cols.copy()
        
        else:
            if ((type(filter_by) == list) |
                (type(filter_by) == np.array)):
                filtered = pd.DataFrame(columns=list(self.at_cols.columns))
                for site in filter_by:
                    filtered = filtered.append(self.at_cols[
                        self.at_cols[df_col_label]  == site])
                    
            if type(filter_criteria) == str:
                filtered = self.at_cols[self.at_cols[filter_by]
                                        == filter_criteria]
        '''***Should do this from pixel size... in Angstroms'''
        if outliers == None:
            outliers = np.min(np.linalg.norm(self.trans_mat, axis=1)) * 0.5
            
        filtered = filtered[np.linalg.norm(
            filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
            - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outliers]

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
        print(f'Scalar componenent: {scale_distortion_res * 100 :.{2}f} %')
        print(f'Shear componenent: {shear_distortion_res :.{5}f} (radians)')
        print(f'Estimated Pixel Size: {pix_size * 100 :.{3}f} (pm)')
            
                        
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
        rot_.all_masks = ndimage.rotate(rot_.all_masks, np.degrees(angle))
        
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
                                   fit_or_ref='fit', 
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
            
        if fit_or_ref == 'fit':
            xcol, ycol = 'x_fit', 'y_fit'
        elif fit_or_ref == 'ref':
            xcol, ycol = 'x_ref', 'y_ref'
                                
        fig,axs = plt.subplots(ncols=1,figsize=(10,10), tight_layout=True)
        
        axs.imshow(self.image * self.all_masks, cmap='gray')
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
        
        
        
        