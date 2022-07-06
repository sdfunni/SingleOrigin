from SingleOrigin import *
from BSSN5137_110 import LatticeSiteLabeling

#%%
"""Location of image data"""

path = '18.58.20.951_2.55Mx_HAADF_5us_RevSTEM_110_LEFT2_100pA_BSSN5137_1_SDCorr.tif'

#%%
"""Structure import from .cif and project"""

cif = 'BSSN5137.cif'

za = [1,1,0]            #Zone Axis direction
a2 = [-1,1,0]         #Apparent horizontal axis in projection
a3 = [0,0,1]          #Most vertical axis in projection

#%%
"""Find 3D coordinates of unit cell, metric tensor and 3D direct 
stucture matrix (a_3d)"""
uc = UnitCell(cif)

uc.transform_basis(za, a2, a3)

"""Project Unit Cell to 2D and combine coincident/proximate columns"""
uc.project_uc_2d(proj_axis = 0, ignore_elements = ['O']) 

uc.combine_prox_cols(toler = 1e-2)

uc.at_cols = LatticeSiteLabeling(uc.at_cols)
uc.plot_unit_cell()

#%%
"""Import experimental image and normalize to 0-1"""
image = import_image('STEM_images/' + path, display_image=True)
image = image[340:-340, 300:-300]

#%%
"""Define offset (in fractional coordinates from the unit cell origin) 
of an easily picked atom column"""

"""Initialize AtomicColumnLattice object"""

acl = AtomicColumnLattice(image, uc, resolution=0.8, origin_atom_column=3)
                          # xlim=[200, -200], ylim=[200,500])

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
parameters"""

acl.fit_atom_columns(diff_filter='auto', grouping_filter='auto',
                     local_thresh_factor=0.5, buffer=20)

#%%
"""Use the fitted atomic column positions to refine the basis vectors and 
    origin. 
    -It is best to choose a sublattice with minimal displacements. 
    -It also must have only one column per projected unit cell. 
    -If no sublattice meets this criteria, specify a specific column in the 
        projected cell."""
 
acl.refine_reference_lattice('LatticeSite', 'A1')

#%%
"""Check image residuals after fitting"""
""" *** This is what I was talking about. I need to make a function that 
does this, but better. """

acl.plot_fitting_residuals()

#%%
"""Plot Column positions with color indexing"""
acl.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                               fit_or_ref='fit', 
                               plot_masked_image=False)

#%%
"""Plot displacements from reference lattice"""
acl.plot_disp_vects(filter_by='LatticeSite', sites_to_plot='all', titles=None,
                    # x_crop=[0, acl.w], y_crop=[acl.h, 0],
                    scalebar=True, scalebar_len_nm=2, arrow_scale_factor = 2,
                    outliers=100, max_colorwheel_range_pm=None,
                    plot_fit_points=False, plot_ref_points=False)

#%%
"""Rotate the image and data to align a desired basis vector to horizontal
    or vertical"""

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')


acl_rot.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                                   fit_or_ref='fit', 
                                   plot_masked_image=False)
