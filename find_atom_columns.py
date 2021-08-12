import copy
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle as Circle
from matplotlib.legend_handler import HandlerPatch

from scipy import ndimage, fftpack
from scipy.ndimage.filters import (gaussian_filter, gaussian_laplace, 
                                   maximum_filter)
from scipy.optimize import minimize

from SingleOrigin.utils import *
        
#%%
class AtomicColumnLattice:
    '''
    Apply a lattice cell framework to the image for crystallographic 
    computations.
    -"image" is the image to analyze
    -unitcell_2D is a DataFrame of projected atom positions in fractional
        coordinates
    -"a_2d" basis vector matrix
    -"basis_offset_frac" is vector (in fractional coordinates) from unit cell 
        origin to the atom column type
        that will be selected as the origin reference
    -"at_cols" Depricated. Functionality has been removed to "old scripts."
    '''
    def __init__(self, image, unitcell_2D, a_2d, 
                 basis_offset_frac=[0,0], at_cols=None):
        self.image = image
        self.h, self.w = self.image.shape
        if at_cols == None:
            self.at_cols = pd.DataFrame()
        else:
            self.at_cols = at_cols        
        self.unitcell_2D = unitcell_2D
        self.a_2d = a_2d
        self.basis_offset_frac = np.array(basis_offset_frac)
        [self.x0, self.y0] = [np.nan, np.nan]
        self.all_masks = np.zeros(image.shape)
    
    def select_basis_vect_FFT(self, a1_order=1, a2_order=1, #thresh = 0.5,
                              sigma = 5):
        '''Select primary lattice directions from FFT'''
        '''Take FFT and find spot coordinates using filters and Gaussian
            fitting'''
        
        '''Find rough reciprocal lattice'''
        h, w = self.image.shape
        m = min(h,w)
        U = min(1000, int(m/2))
        crop_dim = 2*U - 1
        
        image_square = self.image[int(h/2)-U : int(h/2)+U-1,
                          int(w/2)-U : int(w/2)+U-1]
        
        hann = np.outer(np.hanning(crop_dim),np.hanning(crop_dim))
        # hann = np.ones(image_square.shape)
        fft = fftpack.fft2(image_square*hann)
        fft = (abs(fftpack.fftshift(fft))) # may need to np.log10
        # fft = fft[int(U/2-U):int(m/2+U+1),int(m/2-U):int(m/2+U+1)]
        fft = image_norm(fft)
        self.fft = fft
        
        fft_der = image_norm(-gaussian_laplace(fft, sigma))
        masks, num_masks, slices, spots = watershed_segment(fft_der)
        spots.loc[:, 'stdev'] = ndimage.standard_deviation(fft_der, masks, 
                                            index=np.arange(1, num_masks+1))
        spots_ = spots[(spots.loc[:, 'stdev'] > 0.005)]
        xy = np.array([spots_.loc[:,'x'], spots_.loc[:,'y']]).T
        # origin_ind = np.argmin(np.linalg.norm(xy - np.array([[U,U]]), axis=1))
        origin = np.array([U-1, U-1])
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('''Pick reciprocal basis vectors''',
                  fontdict = {'color' : 'red'})
        ax.set_ylim(bottom = U+U/4, top = U-U/4)
        ax.set_xlim(left = U-U/4, right = U+U/4)
        # ax.imshow(fft_der, cmap='gray')
        ax.imshow(np.log10(self.fft), cmap='gray')
        ax.scatter(xy[:,0], xy[:,1], c='red', s=8)
        ax.scatter(origin[0], origin[1], c='white', s=16)
        ax.set_xticks([])
        ax.set_yticks([])
        
        basis_picks_xy = np.array(plt.ginput(2, timeout=15))
        
        print('done selecting')
        
        '''Generate reference lattice and find corresponding peak regions'''
        a1_star = basis_picks_xy[0, :] - origin
        a2_star = basis_picks_xy[1, :] - origin
        
        alpha_rec = np.array([a1_star, a2_star])
        
        recip_latt_indices = np.array([[i,j] for i in range(-3,4) 
                                       for j in range(-3,4)])
        recip_latt_pix = recip_latt_indices @ alpha_rec + origin
        
        xy = np.array([spots.loc[:,'x'], spots.loc[:,'y']]).T
        vects = np.array([xy - rec_latt_pt for rec_latt_pt in recip_latt_pix])
        inds = np.argmin(np.linalg.norm(vects, axis=2), axis=1)
        
        data = {'h': recip_latt_indices[:, 0],
                'k': recip_latt_indices[:, 1],
                'x_ref': recip_latt_pix[:, 0],
                'y_ref': recip_latt_pix[:, 1],
                'mask_ind': inds,
                'stdev': [spots.loc[:, 'stdev'][ind] for ind in inds]}
        
        recip_latt = pd.DataFrame(data)
        
        mskd_sl_stack = []
        # masks_ref = np.zeros(fft.shape)
        
        for i in inds:
            mask_sl = np.where(masks[slices[i][0],slices[i][1]]== i+1, 1, 0)
            fft_der_sl = fft_der[slices[i][0],slices[i][1]]
            edge = mask_sl - ndimage.morphology.binary_erosion(mask_sl)
            thresh = np.max(edge*fft_der_sl) * 0.95
            mask_sl = np.where(mask_sl*fft_der_sl > thresh, 1, 0)
            # masks_ref[slices[i][0],slices[i][1]] += mask_sl * (i+1)
            mskd_sl_stack.append(fft_der_sl * mask_sl)
        
        '''save (x, y )start of each image slice and slice CoM coordinates'''
        sl_start = np.array([[slices[i][1].start, slices[i][0].start] 
                        for i in inds])
        '''Pack args into list of lists to pass to 2'''
        args_packed= [[mskd_sl_stack[i], sl_start[i]]
                      for i in range(len(mskd_sl_stack))]
        
        def fit_peaks(args):
            [img_sl, xy_start] = args
            '''Estimate initial guess for 2D Gaussian parameters'''
            # [y0, x0] = ndimage.center_of_mass(img_sl)
            eigvals, eigvects, x0, y0 = img_equ_ellip(img_sl)
            max_ind = np.argmax(eigvals)
            min_ind = 1 if max_ind==0 else 0
            sig_max = np.sqrt(np.abs(eigvals[max_ind]))
            sig_min = np.sqrt(np.abs(eigvals[min_ind]))
            sig_r = sig_max / sig_min
            theta = -np.arcsin(np.cross(np.array([1,0]), eigvects[:, max_ind]))
            Z0 = (np.average(img_sl[img_sl != 0])
                  - np.std(img_sl[img_sl != 0]))
            A0 = np.max(img_sl) - Z0
            
            '''Fit 2D Guassian'''
            p0 = np.array([x0, y0, sig_max, sig_r, theta, A0, Z0])
            
            params = fit_gaussian2D(img_sl, p0)
            
            return [params[0] + xy_start[0],
                    params[1] + xy_start[1],
                    params[2],
                    params[2]/params[3],
                    params[3],
                    np.degrees(params[4]),
                    params[5],
                    params[6]]
        
        results = np.array([fit_peaks(args) for args in tqdm(args_packed)])
        
        '''Process results and return'''
    
        col_labels = ['x_fit', 'y_fit', 'sig_maj', 'sig_min', 'sig_rat',
                      'theta', 'peak_int', 'bkgd_int']
        
        for i, lab in enumerate(col_labels):
            if lab in list(recip_latt.columns): del recip_latt[lab]
            loc = recip_latt.shape[1]
            recip_latt.insert(loc, lab, results[:, i])
            
        def disp_vect_sum_squares(p0, M_star, xy, origin):
            
            alpha = p0[:4].reshape((2,2))
            
            err_xy = xy - M_star @ alpha_rec - origin
            sum_sq = np.sum(err_xy**2)
            return sum_sq
        
        filtered = recip_latt[recip_latt.stdev > 0.001]
        
        M_star = filtered.loc[:, 'h':'k'].to_numpy()
        xy = filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
        
        p0 = alpha_rec.flatten()
        params = minimize(disp_vect_sum_squares, p0, 
                          args=(M_star, xy, origin)).x
        a1_star = params[:2] / (crop_dim*a1_order)
        a2_star = params[2:4] / (crop_dim*a2_order)
        
        alpha_rec = np.array([a1_star, a2_star])
        
        plt.close(fig)
        
        # recip_latt_fit = recip_latt_indices @ (alpha_rec * m) + origin
        fig2, ax = plt.subplots(figsize=(10,10))
        ax.imshow(np.log10(self.fft))
        ax.scatter(xy[:, 0], xy[:, 1], c='red')
        ax.arrow(origin[0], origin[1], a1_star[0]*2*U, a1_star[1]*2*U,
                   fc='red', ec='red', width=0.1, length_includes_head=True,
                   head_width=2, head_length=3)
        ax.arrow(origin[0], origin[1], a2_star[0]*2*U, a2_star[1]*2*U,
                   fc='green', ec='green', width=0.1, length_includes_head=True,
                   head_width=2, head_length=3)
        # ax.set_xlim(np.min(xy[:, 0])-50, 
        #          np.max(xy[:, 0])+50)
        # ax.set_ylim(np.max(xy[:, 1])+50,
        #          np.min(xy[:, 1])-50)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Reciprocal Lattice Fit')
        
        alpha = np.linalg.inv(alpha_rec).T
        
        self.a1_star = a1_star
        self.a2_star = a2_star
        self.a1 = alpha[0,:]
        self.a2 = alpha[1,:]
        self.alpha = alpha
        self.basis_offset_pix = self.basis_offset_frac @ self.alpha
        
        # time.sleep(5)
        # plt.close('all')
        
    def select_origin(self):
        if 'LatticeSite' in list(self.unitcell_2D.columns):
            lab = 'LatticeSite'
        else:
            lab = 'elem'
        
        [h, w] = [self.h, self.w]
        crop_view = np.max(np.abs(self.alpha)) / np.min([h,w]) * 4
        fig, ax = plt.subplots(figsize=(10,10))
        message=('Pick an atom column corresponding to reference atom column type'
                 +' (white outlined position)')
        ax.set_title(str(message), fontdict={'color':'red'}, wrap=True)
        ax.set_ylim(bottom = h/2+h*crop_view, top = h/2-h*crop_view)
        ax.set_xlim(left = w/2-w*crop_view, right = w/2+w*crop_view)
        ax.imshow(self.image, cmap='gray')
        
        a1_unit = self.a1/np.linalg.norm(self.a1)
        a2_unit = self.a2/np.linalg.norm(self.a2)
        alpha_unit = np.array([a1_unit, a2_unit])
        
        self.unitcell_2D['x_ref'] = ''
        self.unitcell_2D['y_ref'] = ''
        self.unitcell_2D.loc[:, 'x_ref': 'y_ref'] = self.unitcell_2D.loc[
            :, 'u':'v'].to_numpy() @ self.alpha
        
        
        rect_params = [w/2+w*crop_view, 
                       h/2+h*crop_view, 
                       -w*crop_view/2, 
                       -h*crop_view/2]
        
        x_space = np.abs(self.alpha[0,0] - self.alpha[1,0])                                  
        y_space = np.abs(self.alpha[0,1] - self.alpha[1,1])
        x_mean = np.mean(self.alpha[:,0])
        y_mean = np.mean(self.alpha[:,1])
        
        x0 = rect_params[2]/2 - x_mean + rect_params[0]
        y0 = rect_params[3]/2 - y_mean + rect_params[1]
        
        site_list = list(set(self.unitcell_2D[lab]))
        site_list.sort()
        
        color_code = {k:v for v, k in  enumerate(
            np.sort(self.unitcell_2D.loc[:, lab].unique()))}
        
        color_list = [color_code[site] for site in 
                      self.unitcell_2D.loc[:, lab]]
        
        colors = np.array([color_code[site] for site in 
                           self.unitcell_2D[lab]])
        
        box = Rectangle((rect_params[0], rect_params[1]), 
                        rect_params[2], rect_params[3], 
                        edgecolor='black', facecolor='grey', 
                        alpha = 1)
        ax.add_patch(box)
        
        uc = self.unitcell_2D[(self.unitcell_2D['u'] // 1 == 0) & 
                             (self.unitcell_2D['v'] // 1 == 0)]
        
        ax.scatter(self.unitcell_2D.loc[:,'x_ref'].to_numpy() + x0, 
                    self.unitcell_2D.loc[:,'y_ref'].to_numpy() + y0,
                    c=color_list, cmap='RdYlGn', s=20, zorder=10)
        
        ax.arrow(x0, y0, self.a1[0], self.a1[1],
                      fc='black', ec='black', width=0.1, length_includes_head=True,
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
        
    def refine_lattice(self, df_col_label=None, filter_criteria=None, 
                       outliers=None):
        '''Refine basis vectors by linear least squares fitting of the 
           resulting lattice (or sublattice) to peak positions measured in the
           image. Minimizes error in fractional coordinates. Optionally,
           specify the dataframe column label and filtering criteria to
           restrict this refinement to a subset of the atomic columns in the 
           image. (e.g.: only fit to a certain sublattice or atomic column in
           the unit cell.)'''
        
        if df_col_label == None:
            filtered = copy.deepcopy(self.at_cols)
        
        else:
            if ((type(filter_criteria) == list) |
                (type(filter_criteria) == np.array)):
                filtered = pd.DataFrame(columns=list(self.at_cols.columns))
                for site in filter_criteria:
                    filtered = filtered.append(self.at_cols[self.at_cols[df_col_label] 
                                            == site])
            if type(filter_criteria) == str:
                filtered = self.at_cols[self.at_cols[df_col_label]
                                        == filter_criteria]
        
        if outliers == None:
            outliers = np.min(np.linalg.norm(self.alpha, axis=1)) * 0.75
            
        filtered = filtered[np.linalg.norm(
            filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
            - filtered.loc[:, 'x_ref':'y_ref'].to_numpy(),
            axis=1)
            < outliers]
        
        print(filtered.shape)

        def disp_vect_sum_squares(p0, M, xy):
            
            alpha = p0[:4].reshape((2,2))
            origin = p0[4:]
            
            err_xy = xy - M @ alpha - origin
            sum_sq = np.sum(err_xy**2)
            return sum_sq
        
        M = filtered.loc[:, 'u':'v'].to_numpy()
        xy = filtered.loc[:, 'x_fit':'y_fit'].to_numpy()
        
        p0 = np.concatenate((self.alpha.flatten(),
                             np.array([self.x0, self.y0])))
        params = minimize(disp_vect_sum_squares, p0, args=(M, xy)).x
        
        print('Sum squared of residuals:',
              disp_vect_sum_squares(params, M, xy))
        self.a1 = params[:2]
        self.a2 = params[2:4]
        
        self.alpha = params[:4].reshape((2,2))
        
        print('Origin shift:', params[4:] - np.array([self.x0, self.y0]))
        self.x0 = params[4]
        self.y0 = params[5]
        print('Optimized basis vectors:', self.alpha)
        
        self.basis_offset_pix = self.basis_offset_frac @ self.alpha
        
        self.at_cols.loc[:, 'x_ref':'y_ref'] = (self.at_cols.loc[:, 'u':'v']
                                                .to_numpy() @ self.alpha
                                                + np.array([self.x0, self.y0]))
        
    def unitcell_template_method(self, a1_var='u', a2_var='v', buffer=20,
                                 filter_type = 'LoG', sigma = 2,
                                 edge_max_thresholding=0.95,
                                 fit_filtered_data=False,
                                 sub_latt_criteria=None, sub_latt_list=None):
        
        '''Uses the projected unit cell positions from .cif file to define a 
            "perfect" lattice and finds atom columns by fitting a 2D Gaussian
            to a window around each projected position
            - "a1_var"/"a2_var": the the dataframe column labels for
                fractional coordinates corresponding to each basis vector
            - "buffer": number of pixels around the image edge where atomic 
                columns will not be located
            - "filter_type":
                    'LoG': (Laplacian of Gaussian) is good for
                        differentiating closely spaced peaks, fast.
                    'Gauss': Gaussian kernal smoothing, fast.
                    'norm_cross_corr': normalized cross correlation, robust
                        agasinst background variations, differentiates closely
                        spaced peaks, standard TEM image processing method, 
                        very slow.
            - "sigma": sigma value to use for 'Gauss', 'LoG' filters or for 
                atom column template for 'norm_cross_corr'.
            - "window_sizes": slice area from the image in which to find 
                atomic columns.
            - "fit_filtered_data": if True, fit 2D gaussian to the masked
                filtered image, else fit to original data. If False, only use
                filtered image for producing masks. Default: False.
            - "sub_latt_criteria": dataframe column to use for sorting subset
                of atomic columns for fitting. Used to customize fitting 
                method for different sublattices
            -"sub_latt_list": list of values in the specified column
                to use for sorting. Atomic columns in the image corresponding
                to dataframe rows that match these values will be fit.
            '''
        
        if filter_type not in ['LoG', 'norm_cross_corr', 'Gauss', None]:
            raise Exception('filter type not allowed')
        # offset = np.array(offset)
        if 'LatticeSite' in list(self.unitcell_2D.columns):
            lab = 'LatticeSite'
        else:
            lab = 'elem'
        
        img_der = copy.deepcopy(self.image)
        
        if filter_type == 'Gauss':
            for _ in range(2):
                img_der = image_norm(gaussian_filter(img_der, sigma=sigma))
        
        elif filter_type == 'LoG':
            img_der = image_norm(-gaussian_laplace(img_der, sigma))
        
        if (np.isnan(self.x0) and np.isnan(self.y0)):
            (x0, y0) = self.select_origin()
            print('pick coordinates:', x0, y0)
            origin = pd.DataFrame().from_dict({'x_ref' : [x0], 'y_ref' : [y0],
                                           'label' : [1]})

            origin, _ = gauss_position_refine(origin, self.image,
                                              img_der, Gauss_smooth=0)

            print('fit:', origin.at[0,'x_fit'], origin.at[0,'y_fit'])
            
            self.x0 = origin.at[0,'x_fit'] - self.basis_offset_pix[0]
            self.y0 = origin.at[0,'y_fit'] - self.basis_offset_pix[1]
        
        a1 = self.a1
        a2 = self.a2
        x0 = self.x0
        y0 = self.y0
        h = self.h
        w = self.w
        
        '''Find maximum extents of lattice expansion required to cover the
            whole image'''
        if 'x_fit' not in list(self.at_cols.columns):
            print('Creating reference lattice...')
            def vect_angle(a, b):
                theta = np.arccos(a @ b.T/(np.linalg.norm(a) * np.linalg.norm(b)))
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
            
            at_cols = pd.concat([self.unitcell_2D] * int(latt_cells.shape[0]
                                                      /self.unitcell_2D.shape[0]),
                                ignore_index=True)
            at_cols.loc[:, 'u':'v'] += latt_cells
            
            at_cols.loc[:, 'x_ref':'y_ref'] = (at_cols.loc[:, 'u':'v'].to_numpy() 
                                               @ self.alpha 
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
            print('Reference lattice pre-refinement...')
            masks, num_masks, slices, peaks = watershed_segment(img_der, 
                                                                buffer=0)
            CoMs = peaks.loc[:, 'x':'y'].to_numpy()
            
            for lim in [5, 10, 20]:
                filtered = at_cols[(np.abs(at_cols.u) <= lim) & 
                                    (np.abs(at_cols.v) <= lim)]
                
                if ((type(sub_latt_criteria) == str) and 
                    (type(sub_latt_list) == list)):
                    filtered = (filtered[filtered.loc[:,sub_latt_criteria]
                                  .isin(sub_latt_list)])
                    print(filtered.shape)
                
                M = filtered.loc[:, 'u':'v'].to_numpy()
                xy_ref = filtered.loc[:, 'x_ref':'y_ref'].to_numpy()
                
                vects = np.array([CoMs - xy for xy in xy_ref])
                inds = np.argmin(np.linalg.norm(vects, axis=2), axis=1)
                xy = np.array([CoMs[ind] for ind in inds])
                
                def disp_vect_sum_squares(p0, M, xy):
                    
                    alpha = p0[:4].reshape((2,2))
                    origin = p0[4:]
                    
                    err_xy = xy - M @ alpha - origin
                    sum_sq = np.sum(err_xy**2)
                    return sum_sq
                
                p0 = np.concatenate((self.alpha.flatten(),
                                      np.array([self.x0, self.y0])))
                params = minimize(disp_vect_sum_squares, p0, args=(M, xy)).x
                
                self.a1 = params[:2]
                self.a2 = params[2:4]
                
                self.alpha = params[:4].reshape((2,2))
                
                at_cols.loc[:, 'x_ref':'y_ref'] = (
                    at_cols.loc[:, 'u':'v'].to_numpy() @ self.alpha
                    + np.array([self.x0, self.y0])
                    )
    
                self.x0 = params[4]
                self.y0 = params[5]
        
        else: at_cols = copy.deepcopy(self.at_cols)
         
        at_cols = at_cols[((at_cols.x_ref >= buffer) &
                         (at_cols.x_ref <= w - buffer) &
                         (at_cols.y_ref >= buffer) &
                         (at_cols.y_ref <= h - buffer))]
        
        self.at_cols = at_cols
        
        print('Fitting atom columns with 2D Gaussians...')
        if (type(sub_latt_criteria) == str) and (type(sub_latt_list) == list):
            at_cols_ = (self.at_cols[self.at_cols.loc[:,sub_latt_criteria]
                                  .isin(sub_latt_list)])
        
        else: at_cols_ = self.at_cols.copy()
        at_cols_, all_masks = gauss_position_refine(at_cols_,
                                self.image, img_der,
                                fit_filtered_data=fit_filtered_data,
                                Gauss_smooth=0,
                                edge_max_thresholding = edge_max_thresholding)
    
        self.at_cols.update(at_cols_)
        self.at_cols.loc[:, 'x_fit':'bkgd_int'] = (
            self.at_cols.loc[:, 'x_fit': 'bkgd_int'].astype(float))
        
        self.all_masks = np.where((self.all_masks + all_masks) >= 1, 1, 0)
                
    def rotate_image_and_data(self, align_dir='horizontal',
                              align_basis='a1'):
        '''Returns a new class instance with a rotated image.
            Transforms the basis vectors, crystallographic transformation 
            matrix, atomic column positions and elliptical gaussian fit 
            orientation. 
            '''
        rot_ = copy.deepcopy(self)
        # if align_dir == 'horizontal':
        #     ref_vect = np.array([1,0])
        # elif align_dir == 'vertical':
        #     ref_vect = np.array([0,1])
        # else:
        #     raise Exception('align_dir must be "horizontal" or "vertical"')
        
        if align_basis =='a1':
            align_vect = self.a1
        elif align_basis =='a2':
            align_vect = self.a2
        # else:
        #     print('align_vector assigned as "a1"')
        #     align_vect = self.alpha[0,:]
        
        '''Find the rotation angle and 
            direction'''
        # angle = np.arcsin(np.cross(align_vect, ref_vect)
        #                              / np.linalg.norm(align_vect)).item()
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
        alpha = np.array([[np.cos(angle), np.sin(angle), 0],
                          [-np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
        tau = np.array([[1, 0, (self.image.shape[1]-1)/2],
                        [0, 1, (self.image.shape[0]-1)/2],
                        [0, 0, 1]])
        tau_ = np.array([[1, 0, -(self.image.shape[1]-1)/2],
                         [0, 1, -(self.image.shape[0]-1)/2],
                         [0, 0, 1]])
        alpha = tau @ alpha @ tau_
        
        xy = np.array(np.append(rot_.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy(),
                       np.ones((rot_.at_cols.shape[0],1)), axis=1)).T
        rot_.at_cols.loc[:, 'x_fit':'y_fit'] = (alpha @ xy).T[:, :2] + trans
        
        xy_pix = np.append(rot_.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy(),
                           np.ones((rot_.at_cols.shape[0],1)), axis=1).T
        
        rot_.at_cols.loc[:, 'x_ref':'y_ref'] = (alpha @ xy_pix).T[:, :2] + trans
        
        [rot_.x0, rot_.y0] = list((np.array([rot_.x0, rot_.y0, 1], ndmin=2) 
                                  @ alpha.T)[0,0:2] + trans[0,:])
        
        '''Transform data'''
        rot_.alpha = rot_.alpha @ alpha[0:2, 0:2].T
        rot_.a1 = rot_.alpha[0, :]
        rot_.a2 = rot_.alpha[1, :]
        '''***Logic sequence to make basis vectors ~right, ~up'''
        
        rot_.a1_star = (np.linalg.inv(rot_.alpha).T)[0, :]
        rot_.a2_star = (np.linalg.inv(rot_.alpha).T)[1, :]
        rot_.at_cols.theta += np.degrees(angle)
        rot_.at_cols.theta -= np.trunc(rot_.at_cols.theta.to_numpy().astype(
            'float') / 90) * 180
        rot_.angle = angle
        
        return rot_
    
    def simult_fit_gaussian2D(self, tol=0.01):
        '''Refines Gaussian fits by simultaneous fitting all atom columns in the
            image.
            -"tol": image intensity fraction used to calcualte truncation 
            range for Gaussian calculations. Truncation is based on the maximum
            previously calculated parameter values for sigma major ('sig_maj')
            and peak intensity ('A')
        '''
        
        # Get parameter estimates from previous fitting
        p0 = self.at_cols.loc[:, 'x_fit':'bkgd_int'].to_numpy()
        p0 = np.delete(p0, 3, 1)
        
        p0[:, 4] = np.radians(p0[:, 4])
        # p0[:, 4] *= -1
        
        # Find pixel range at which to truncate Gaussians
        p_max = np.max(p0, axis=0)
        [sigma, A] = [p_max[2], p_max[-2]]
        print(A)
        x = np.arange(0, 200, 1e-3)
        y = A * np.exp(-(x / sigma)**2 / 2)
        trunc = np.ceil(np.where(y < 0.01)[0][0] * 1e-3)
        print(trunc)
        
        # p_max[:2] = [99, 99]
        # p_max[-1] = 0
        # [x0, y0, sig_maj, sig_rat, ang, A] = p_max
        
        # y, x = np.indices((200, 200))
        # x = x.ravel()
        # y = y.ravel()
        
        # z = gaussian_2d(x, y, x0, y0, sig_maj, sig_rat, ang, A=1)
        # z = np.reshape(np.where(z < 0.01, z, 0), (200,200))
        # mask = np.where(z, 1, 0)
        # edge = mask - np.where(binary_erosion(mask, border_value=1), 1, 0)
        # edge_ind = np.vstack(((np.take(x, np.nonzero(edge.ravel()))),
        #                       np.take(y, np.nonzero(edge.ravel())))).T
        # tol = np.ceil(np.max(np.linalg.norm(edge_ind - [99, 99], axis=1)))
        
        # Prepare data structures for fitting to masked image regions only
        y, x = np.indices(self.image.shape)
        z = (self.image * self.all_masks).ravel()
        
        x = x.ravel()
        y = y.ravel()
        
        unmasked_data = np.nonzero(z)
        x = np.take(x, unmasked_data)
        y = np.take(y, unmasked_data)
        z = np.take(z, unmasked_data)
        
        # Model background with bivariate spline and subtract from image data,
        # update peak intensities
        I_o = p0[:, -1]
        bkgd = SmoothBivariateSpline(p0[:, 0], p0[:, 1], I_o, 
                                   kx=5, ky=5)
        I_o_new = bkgd.__call__(p0[:, 0], p0[:, 1], grid=False)
        
        p0[:, -2] += I_o - I_o_new
        p0[:,-1] = I_o_new
        
        z -= bkgd.__call__(x, y, grid=False)
        print('min after backgroud subtraction', np.min(z))
        
        bounds = [[(atom[0]-2, atom[0]+ 2), (atom[1] - 2, atom[1] + 2), 
                   (0, 1.5*sigma), (1, 3), (-np.pi/2, np.pi/2), 
                   (0, 1 - atom[-1])]
                  for atom in p0]
        
        bounds = [item for sublist in bounds for item in sublist]
        bounds.append((trunc, trunc))
        
        p0 = np.delete(p0, -1, 1)
        p0 = p0.ravel()
        p0 = np.append(p0, trunc)
        
        # Optimize fitting
        params = minimize(multi_gauss_ss, p0, args=(x, y, z), bounds=bounds,
                          method = 'L-BFGS-B', options={'maxiter':3}).x
        
        # Adjust parameters and update the DataFrame
        params[:, 4] *= -1
        params[:, 4] = np.degrees(params[:, 4])
        params = np.insert(params, 3, params[:, 3]/params[:, 4], axis=1)
        
        return params
        
        # self.at_cols.loc[:, 'x_fit':'bkgd_int'] = params
        
        
        
        
        
        