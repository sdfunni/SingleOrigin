import SingleOrigin as so
import numpy as np

#%%
"""Specify projection axis and in-plane basis vectors"""

za = [1,1,0]            #Zone Axis direction
a2 = [-1,1,0]         #Apparent horizontal axis in projection
a3 = [0,0,1]          #Most vertical axis in projection

#%%
"""Load .cif file. Find 3D coordinates of unit cell, metric tensor and 
3D direct stucture matrix (a_3d)"""
uc = so.UnitCell(directory='')

uc.transform_basis(za, a2, a3)

"""Project Unit Cell to 2D and combine coincident/proximate columns"""
uc.project_uc_2d(proj_axis = 0, ignore_elements = []) 

uc.combine_prox_cols(toler = 1e-2)
uc.at_cols.loc[:, 'LatticeSite'] = ['B', 'O', 'A']

uc.plot_unit_cell(label_by='LatticeSite')
sites = sorted(uc.at_cols.loc[:, 'LatticeSite'].to_list())

#%%
"""Import experimental image and normalize to 0-1"""
image, metadata = so.import_image(
    directory='',
    display_image=True)

#%%
"""Define offset (in fractional coordinates from the unit cell origin) 
of an easily picked atom column"""

"""Initialize AtomicColumnLattice object"""

acl = so.AtomicColumnLattice(image, uc, probe_fwhm=0.8,
                             xlim=None, ylim=None)

"""Get real space basis vectors using the FFT
if some FFT peaks are weak or absent (such as forbidden reflections), 
specify the order of the first peak that is clearly visible"""

acl.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=2)

#%%
"""Generate a mathematical lattice from the basis vectors and unit cell 
    projection."""

acl.define_reference_lattice()

#%%
"""Fit atom columns at reference lattice points
-Automatically decides what filters to use based on resolution (input above) 
and pixel size (determined from basis vectors lengths compared to .cif lattice 
parameters
-If fitting is extremely slow due to simultaneous fitting, set 
"Gauss_sigma" to None or a small value. (See docstring for details of 'auto' 
setting.) Be careful to check results (including residuals) to verify 
accuracy."""

acl.fit_atom_columns(buffer=10, local_thresh_factor=0.5, 
                     grouping_filter='auto', diff_filter='auto',)

#%%
"""Use the fitted atomic column positions to refine the basis vectors and 
    origin. 
    -It is best to choose a sublattice with minimal displacements. 
    -It also must have only one column per projected unit cell. 
    -If no sublattice meets this criteria, specify a specific column in the 
        projected cell."""
 
acl.refine_reference_lattice(filter_by='LatticeSite', sites_to_use='B', 
                             outlier_disp_cutoff=30)

#%%
"""Plot Column positions with color indexing"""
fit, axs = acl.plot_atom_column_positions(filter_by='elem', sites_to_plot='all',
                                fit_or_ref='fit', outlier_disp_cutoff=np.inf,
                                plot_masked_image=True)

#%%
"""Plot displacements from reference lattice"""
acl.plot_disp_vects(filter_by='elem', sites_to_plot='all', 
                        x_lim=[0, acl.w], y_lim=[acl.h, 0],
                        scalebar=True, scalebar_len_nm=2,
                        max_colorwheel_range_pm=25,
                        arrow_scale_factor=2)

#%%
"""Rotate the image and data to align a desired basis vector to horizontal
    or vertical"""

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')


acl_rot.plot_atom_column_positions(filter_by='elem', sites_to_plot='all',
                                    fit_or_ref='fit', 
                                    plot_masked_image=False,
                                    scatter_kwargs_dict={'s':10}, )

#%%
"""Plot displacements from reference lattice"""
acl_rot.plot_disp_vects(filter_by='elem', sites_to_plot='all', 
                        x_lim=[0, acl_rot.w], y_lim=[acl_rot.h, 0],
                        scalebar=True, scalebar_len_nm=2,
                        max_colorwheel_range_pm=25,
                        arrow_scale_factor=2)
