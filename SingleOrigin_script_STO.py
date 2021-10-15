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

path = '/Users/funni/Downloads/RevSTEM_SDCorr_rot.tif'

#%%
"""Structure import from .cif and project"""

cif_path = 'STO.cif'

za = [1,0,0]            #Zone Axis direction
a2 = [0,1,0]         #Apparent horizontal axis in projection
a3 = [0,0,1]          #Most vertical axis in projection

#%%
"""Find 3D coordinates of unit cell, metric tensor and 3D direct 
stucture matrix (a_3d)"""
uc = UnitCell(cif_path)

uc.transform_basis(za, a2, a3)

"""Project Unit Cell to 2D and combine coincident/proximate columns"""
uc.project_uc_2d(proj_axis = 0, ignore_elements = ['O']) 

uc.combine_prox_cols(prox_toler = 1e-2)

uc.plot_unit_cell()

#%%
"""Import experimental image and normalize to 0-1"""
image = import_image(path, display_image=True)

#%%
"""Define offset (in fractional coordinates from the unit cell origin) 
of an easily picked atom column"""

basis_offset = [0, 0]

"""Initialize AtomicColumnLattice object"""

acl = AtomicColumnLattice(image, uc.at_cols, uc.a_2d, resolution=0.8,
                          basis_offset_frac=basis_offset)

"""Get real space basis vectors using the FFT
if some FFT peaks are weak or absent (such as forbidden reflections), 
specify the order of the first peak that is clearly visible"""

acl.FFT_get_basis_vect(a1_order=1, a2_order=1, sigma=2)

#%%
"""Generate a mathematical lattice from the basis vectors and unit cell 
    projection."""

acl.get_reference_lattice(a1_var='u', a2_var='v')

"""Fit atom columns at reference lattice points
-Automatically decides what filters to use based on resolution (input above) 
and pixel size (determined from basis vectors lengths compared to .cif lattice 
parameters"""

acl.fit_atom_columns(edge_max_threshold=1.5, buffer=40)

#%%
"""Use the fitted atomic column positions to refine the basis vectors and 
    origin. 
    -It is best to choose a sublattice with minimal displacements. 
    -It also must have only one column per projected unit cell. 
    -If no sublattice meets this criteria, specify a specific column in the 
        projected cell."""
 
acl.refine_reference_lattice()#'elem', 'Sr')


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
plt.scatter(filtered.loc[:,'x_fit'], filtered.loc[:,'y_fit'],
            c=color_list, s=4, cmap='RdYlGn')
#%%
"""Plot Column positions with color indexing"""
fig,axs = plt.subplots(ncols=1,figsize=(10,10), tight_layout=True)
axs.imshow(acl.image * acl.all_masks, cmap='gray')
axs.set_xticks([])
axs.set_yticks([])

unitcell = acl.at_cols[(acl.at_cols['u'] // 1 == 0) & 
                       (acl.at_cols['v'] // 1 == 0)]

color_code = {k:v for v, k in 
              enumerate(np.sort(unitcell.loc[:,'elem'].unique()))}
color_list = [color_code[site] for site in acl.at_cols.loc[:,'elem']]

cmap = plt.cm.RdYlGn

axs.scatter(acl.at_cols.loc[:,'x_fit'], acl.at_cols.loc[:,'y_fit'],
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

axs.arrow(acl.x0, acl.y0, acl.a1[0], acl.a1[1],
              fc='red', ec='red', width=0.1, length_includes_head=True,
              head_width=2, head_length=3, label=r'[001]')
axs.arrow(acl.x0, acl.y0, acl.a2[0], acl.a2[1],
              fc='green', ec='green', width=0.1, length_includes_head=True,
              head_width=2, head_length=3, label=r'[110]')

#%%
"""Rotate the image and data to align a desired basis vector to horizontal
    or vertical"""

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')


acl_rot.plot_atom_column_positions(filter_by='elem', sites_to_fit='all',
                                   fit_or_ref='fit', 
                                   plot_masked_image=False)
