import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from matplotlib import colors as colors
from matplotlib.patches import Wedge
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle as Circle
from matplotlib.legend_handler import HandlerPatch
from CifFile import ReadCif
import copy

from matplotlib_scalebar.scalebar import ScaleBar

from SingleOrigin import *

#%%
"""Location of image data"""

path = '/Users/funni/Downloads/RevSTEM_SDCorr_rot.tif'

#%%
"""Structure import from .cif and project"""

cif = 'STO.cif'

za = [1,0,0]            #Zone Axis direction
a2 = [0,1,0]         #Apparent horizontal axis in projection
a3 = [0,0,1]          #Most vertical axis in projection

cif_data = copy.deepcopy(ReadCif(cif))

#%%
"""Find 3D coordinates of unit cell, metric tensor and 3D direct 
stucture matrix (a_3d)"""
uc = UnitCell(cif_data)

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

acl.fit_atom_columns(#LoG_sigma=None, Gauss_sigma=None,
                      edge_max_threshold=1.5, buffer=40)

#%%
"""Use the fitted atomic column positions to refine the basis vectors and 
    origin. 
    -It is best to choose a sublattice with minimal displacements. 
    -It also must have only one column per projected unit cell. 
    -If no sublattice meets this criteria, specify a specific column in the 
        projected cell."""
 
acl.refine_reference_lattice()#'elem', 'Sr')

"""Re-check angle between basis vectors"""


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
# plt.imshow(acl.image*acl.all_masks)
plt.scatter(filtered.loc[:,'x_fit'], filtered.loc[:,'y_fit'],
            c=color_list, s=4, cmap='RdYlGn')
#%%
"""Plot Column positions with color indexing"""
fig,axs = plt.subplots(ncols=1,figsize=(10,10))
axs.imshow(acl.image * acl.all_masks)
axs.set_xticks([])
axs.set_yticks([])

filtered = acl.at_cols

unitcell = filtered[(filtered['u'] // 1 == 0) & (filtered['v'] // 1 == 0)]

color_code = {k:v for v, k in 
              enumerate(np.sort(unitcell.loc[:,'elem'].unique()))}
color_list = [color_code[site] for site in filtered.loc[:,'elem']]

cmap = plt.cm.RdYlGn

color_index = [Circle((30, 7), 3, color=cmap(c)) 
               for c in np.linspace(0,1, num=len(color_code))]

# def make_legend_circle(legend, orig_handle,
#                         xdescent, ydescent,
#                         width, height, fontsize):
#     p = orig_handle
#     return p
# axs.legend(handles = color_index,
#             labels = color_code.keys(),
#             handler_map={Circle : HandlerPatch(patch_func=
#                                               make_legend_circle),},
#             fontsize=20, loc='lower left', bbox_to_anchor=[1.02, 0],
#             facecolor='grey')

axs.scatter(filtered.loc[:,'x_fit'], filtered.loc[:,'y_fit'],
            c=color_list, s=4, cmap='RdYlGn')

axs.legend()

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

#%%
"""Plot Column positions with color indexing"""
"""*** Make function"""
xlim = (485, 785)
ylim = (533, 233)

if 'LatticeSite' in list(acl.at_cols.columns):
    lab = 'LatticeSite'
else:
    lab = 'elem'
            
fig,axs = plt.subplots(ncols=1,figsize=(10,10))
axs.imshow(acl_rot.image, cmap='gray')
axs.set_xticks([])
axs.set_yticks([])

axs.set_xlim(xlim[0], xlim[1])
axs.set_ylim(ylim[0], ylim[1])

uc = acl_rot.at_cols[(acl_rot.at_cols['u'] // 1 == 0) &
                    (acl_rot.at_cols['v'] // 1 == 0)]

sub_latt = acl_rot.at_cols

color_code = {k:v for v, k in 
              enumerate(np.sort(uc.loc[:, lab].unique()))}
color_list = [color_code[site] for site in sub_latt.loc[:, lab]]

# cmap = plt.cm.RdYlGn
cmap = plt.cm.viridis
axs.scatter(sub_latt.loc[:,'x_fit'], sub_latt.loc[:,'y_fit'],
            c=color_list, cmap=cmap, s=4, zorder=2)
color_index = [Circle((30, 7), 3, color=cmap(c)) 
               for c in np.linspace(0,1, num=len(color_code))]

def make_legend_circle(legend, orig_handle,
                       xdescent, ydescent,
                       width, height, fontsize):
    p = orig_handle
    return p
axs.legend(handles = color_index,
           labels = title,
           handler_map={Circle : HandlerPatch(patch_func=
                                              make_legend_circle),},
           fontsize=20, loc='lower left', bbox_to_anchor=[1.02, 0],
           facecolor='grey')
           
axs.arrow(acl_rot.x0, acl_rot.y0, acl_rot.a1[0], acl_rot.a1[1],
              fc='white', ec='white', width=1, length_includes_head=True,
              head_width=4, head_length=5, label=r'[110]')
axs.arrow(acl_rot.x0, acl_rot.y0, acl_rot.a2[0], acl_rot.a2[1],
              fc='white', ec='white', width=1, length_includes_head=True,
              head_width=4, head_length=5, label=r'[001]')

axs.text(acl_rot.x0 + 25, acl_rot.y0 + 5, r'$\frac{1}{2} [1\bar1 0]$', 
         color='white', size=18, zorder=11)
axs.text(acl_rot.x0 - 20, acl_rot.y0 - np.hypot(*acl.a2) - 5, r'$ [001]$', color='white', 
         size=18, zorder=11)

scalebar = ScaleBar(len_per_pix/10, 'nm', pad=0.4,
                    font_properties={'size':18}, location='lower right',
                    fixed_value=0.5)
scalebar.box_alpha = 0.7
axs.add_artist(scalebar)
# fig.tight_layout()

#%%
"""Plot Column positions with displacement vectors"""
"""*** Make a function"""
# sites = ['A1', 'A2_1', 'B2']
cbar_range = 25
fig = plt.figure(figsize=(18,7), tight_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[3,3,1],
                      height_ratios=[3,], wspace=0.05)


for ax, site in enumerate(sites):
    axs = fig.add_subplot(gs[ax])
    axs.imshow(acl_rot.image, cmap='gray')
    
    x_crop = [0, acl_rot.image.shape[1]]
    y_crop = [acl_rot.image.shape[0], 0]
    
    
    axs.set_xlim(x_crop[0], x_crop[1])
    axs.set_ylim(y_crop[0], y_crop[1])
    
    axs.set_xticks([])
    axs.set_yticks([])
    
    uc = acl_rot.at_cols[(acl_rot.at_cols['u'] // 1 == 0) & 
                         (acl_rot.at_cols['v'] 
                          // 1 == 0)]
    
    color_code = {k:v for v, k in 
                  enumerate(np.sort(uc.loc[:,'elem'].unique()))}
    color_list = [color_code[site] for site in 
                  acl_rot.at_cols.loc[:,'elem']]
    
    min_M = acl_rot.at_cols.v.min()
    upper_left = acl_rot.at_cols[(acl_rot.at_cols.v == min_M + 4) 
                                & (acl_rot.at_cols.elem == site)]
    
    if ax == 2:
        scalebar = ScaleBar(a_2d[0,0].item()/np.linalg.norm(acl_rot.a1)/10,
                            'nm', location='lower right', pad=0.4, 
                            fixed_value=2, font_properties={'size':10}, 
                            box_color='lightgrey', width_fraction=0.02, sep=2)
        axs.add_artist(scalebar)
        
    sub_latt = acl_rot.at_cols[acl_rot.at_cols.elem == site]

    axs.text(x_crop[0]+10, y_crop[1]-20, sites[ax], color='black', size=12,
              weight='bold')
    
    
    hsv = np.ones((sub_latt.shape[0], 3))
    dxy = (sub_latt.loc[:,'x_fit':'y_fit'].to_numpy()
           - sub_latt.loc[:,'x_ref':'y_ref'].to_numpy())
    
    norms=np.linalg.norm(dxy, axis=1)*len_per_pix*100/cbar_range
    print(np.max(norms)*cbar_range)
    hsv[:, 2] = np.where(norms>1, 1, norms)
    hsv[:, 0] = (np.arctan2(dxy[:,0], -dxy[:,1]) + np.pi/2)/(2*np.pi) % 1
    rgb = colors.hsv_to_rgb(hsv)

    cb = axs.quiver(sub_latt.loc[:,'x_ref'], sub_latt.loc[:,'y_ref'], 
                    sub_latt.loc[:,'x_fit']-sub_latt.loc[:,'x_ref'],
                    (sub_latt.loc[:,'y_fit']-sub_latt.loc[:,'y_ref']),
                    color=rgb,
                    angles='xy', scale_units='xy', scale=0.05,
                    headlength=20, headwidth=10, headaxislength=20,
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
legend = fig.add_subplot(gs[-1])
legend.imshow(rgb)
legend.set_xticks([])
legend.set_yticks([])
r=rgb.shape[0]/2
circle = Wedge((r,r), r-5, 0, 360, width=5, color='black')
legend.add_artist(circle)
legend.axis('off')
legend.axis('image')
legend.text(0.5, -.35, 'Displacement\n(0 - 25 pm)', transform=legend.transAxes,
            horizontalalignment='center', fontsize=7, fontweight='bold')

fig.subplots_adjust(hspace=0, wspace=0, 
                    top=0.9, bottom=0.01, 
                    left=0.01, right=0.99)