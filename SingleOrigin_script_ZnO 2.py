import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from matplotlib import colors as colors
from matplotlib.patches import Wedge
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle as Circle
from matplotlib.legend_handler import HandlerPatch
import copy

from matplotlib_scalebar.scalebar import ScaleBar

from SingleOrigin import *

#%%
"""Location of image data"""

path = '/Users/funni/Downloads/dDPC.tif'

#%%
"""Structure import from .cif and project"""

cif_path = 'ZnO_mp-2133_symmetrized.cif'

za = [1,1,0]            #Zone Axis direction
a1 = [1,-1,0]         #Apparent horizontal axis in projection
a2 = [0,0,1]          #Most vertical axis in projection

#%%
"""Find 3D coordinates of unit cell, metric tensor and 3D direct 
stucture matrix (a_3d)"""
uc = UnitCell(cif_path)

uc.transform_basis(za, a1, a2)

"""Project Unit Cell to 2D and combine coincident/proximate columns"""
uc.project_uc_2d(proj_axis = 0, ignore_elements = []) 

uc.combine_prox_cols(toler = 1e-2)

uc.plot_unit_cell()

#%%
"""Import experimental image and normalize to 0-1"""
image = import_image(path, display_image=True)

#%%
"""Define offset (in fractional coordinates from the unit cell origin) 
of an easily picked atom column"""

basis_offset = uc.at_cols.loc[3, 'u':'v'].tolist()

"""Initialize AtomicColumnLattice object"""

acl = AtomicColumnLattice(image, uc, resolution=0.8,
                          basis_offset_frac=basis_offset)

"""Get real space basis vectors using the FFT
if some FFT peaks are weak or absent (such as forbidden reflections), 
specify the order of the first peak that is clearly visible"""

acl.fft_get_basis_vect(a1_order=1, a2_order=2, sigma=2)

#%%
"""Generate a mathematical lattice from the basis vectors and unit cell 
    projection."""

acl.define_reference_lattice(a1_var='u', a2_var='v')

"""Fit atom columns at reference lattice points
-Automatically decides what filters to use based on resolution (input above) 
and pixel size (determined from basis vectors lengths compared to .cif lattice 
parameters"""

acl.fit_atom_columns(edge_max_threshold=0.95, buffer=40)

#%%
"""Use the fitted atomic column positions to refine the basis vectors and 
    origin. 
    -It is best to choose a well defined sublattice with minimal 
    displacements. """
 
acl.refine_reference_lattice('elem', 'Zn')


#%%
"""Check image residuals after fitting"""
""" *** This is what I was talking about. I need to make a function that 
does this, but better. """

residuals = copy.deepcopy(acl.image)
y, x = np.indices(acl.image.shape)
for row in acl.at_cols.itertuples():
    peak = gaussian_2d(x, y, row.x_fit, row.y_fit ,row.sig_maj,
                             row.sig_rat, -row.theta, row.peak_int, 
                             0)
    residuals -= peak

plt.figure()
plt.imshow(residuals*acl.all_masks)
plt.scatter(acl.at_cols.loc[:,'x_fit'], acl.at_cols.loc[:,'y_fit'],
            c=color_list, s=4, cmap='RdYlGn')
#%%
"""Plot Column positions with color indexing"""
acl.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                               fit_or_ref='fit', 
                               plot_masked_image=False)

#%%
"""Rotate the image and data to align a desired basis vector to horizontal
    or vertical"""

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')

acl_rot.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                                   fit_or_ref='fit', 
                                   plot_masked_image=False)
    
