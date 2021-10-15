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

path = 'ZnO_dDPC.tif'

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

# axs.set_xlim(xlim[0], xlim[1])
# axs.set_ylim(ylim[0], ylim[1])

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

# fig.savefig(save_folder + 'Figure_6.pdf')


#%%
plt.figure()
plt.hist(np.linalg.norm((sub_latt.loc[:,'x_':'y_'].to_numpy()
                                - sub_latt.loc[:,'x_pix':'y_pix'].to_numpy()),
                                axis=1)*len_per_pix*100, bins=50)
plt.xlim(0,50)
plt.title(f'Distribution of {site} displacement magnitudes')
#%%
"""Plot A2_1 NN distance vectors"""
"""*** ?? Make function ??"""

NN1 = {}
NN2 = {}
for site in sites:
    sublatt = acl_rot.at_cols[(acl_rot.at_cols.elem 
                               == site)].reset_index()
    NN1[site] = pd.DataFrame(columns = ['x_fit', 'y_fit', 'dx', 'dy'])
    NN2[site] = pd.DataFrame(columns = ['x_fit', 'y_fit', 'dx', 'dy'])
    
    missed = [0,0]
    for ind, row in tqdm(sublatt.iterrows()):
        M1, M2 = row.u, row.v
        a1_nn = sublatt[(np.round(sublatt.u, 3) == np.round(M1 + 1, 3)) & 
                        (np.round(sublatt.v, 3) == np.round(M2, 3))]
        a2_nn = sublatt[((np.round(sublatt.u, 3) == np.round(M1 + 1/3, 3)) & 
                         (np.round(sublatt.v, 3) == np.round(M2 + 0.5, 3))) |
                         ((np.round(sublatt.u, 3) == np.round(M1 + 2/3, 3)) & 
                         (np.round(sublatt.v, 3) == np.round(M2 + 0.5, 3)))]
        if a1_nn.shape[0] == 1:
            NN1[site].at[ind,'dx'] = (a1_nn.x_fit 
                                      - sublatt.at[ind, 'x_fit']).item()
            NN1[site].at[ind,'dy'] = (a1_nn.y_fit 
                                      - sublatt.at[ind, 'y_fit']).item()
            NN1[site].at[ind,'x_fit'] = sublatt.at[ind, 'x_fit']
            NN1[site].at[ind,'y_fit'] = sublatt.at[ind, 'y_fit']
        else: missed[0] += 1
        if a2_nn.shape[0] == 1:
            NN2[site].at[ind,'dx'] = (a2_nn.x_fit 
                                      - sublatt.at[ind, 'x_fit']).item()
            NN2[site].at[ind,'dy'] = (a2_nn.y_fit 
                                      - sublatt.at[ind, 'y_fit']).item()
            NN2[site].at[ind,'x_fit'] = sublatt.at[ind, 'x_fit']
            NN2[site].at[ind,'y_fit'] = sublatt.at[ind, 'y_fit']
        else: missed[1] += 1
    
    NN1[site] = NN1[site].dropna().astype(np.float64)
    NN2[site] = NN2[site].dropna().astype(np.float64)

print(missed)
#%%
# sites=['B2', 'A2_1']
fig = plt.figure(figsize=(6*len(sites),6), tight_layout=True)
gs = gridspec.GridSpec(nrows=1, ncols=len(sites), 
                       width_ratios=[3 for _ in range(len(sites))])
axs= [i for i in range(len(sites))]
for i, site in enumerate(sites):
    
    axs[i] = fig.add_subplot(gs[i])
    axs[i].imshow(acl_rot.image, cmap='gray')
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].text(80, -70, title[i], color='black', size=12, weight='bold')
    if i == 0:
        scalebar = ScaleBar(a_2d[0,0].item()/np.linalg.norm(acl_rot.a1)/10,
                            'nm', location='lower right', fixed_value=5,
                            font_properties={'size': 10}, 
                            box_color='lightgrey',  width_fraction=0.015)
        # # scalebar.box_alpha = 0.7
        # axs[i].add_artist(scalebar)
    
        # axs[i].arrow(100, 2400,
        #           acl_rot.a1[0]*5, -acl_rot.a1[1]*5,
        #           fc='white', ec='white', width=5, length_includes_head=True,
        #           head_width=20, head_length=30)
        # axs[i].text(270, 2350, r'[$\bar1$10]', 
        #             {'color': 'white', 'fontsize': 14})
        # axs[i].arrow(100, 2400,
        #           -acl_rot.a2[0]*5, -acl_rot.a2[1]*5,
        #           fc='white', ec='white', width=5, length_includes_head=True,
        #           head_width=20, head_length=30)
        # axs[i].text(30, 2250, r'[001]', {'color': 'white', 'fontsize': 14})
    
    axs[i].quiver(NN2[site].loc[:,'x_fit'], NN2[site].loc[:,'y_fit'],
                    NN2[site].loc[:,'dx'],(NN2[site].loc[:,'dy']),
                    (np.linalg.norm(
                        NN2[site].loc[:,'dx':'dy'].to_numpy(dtype=np.float64),
                        axis=1)
                        - np.linalg.norm(acl_rot.a2))*len_per_pix*100,
                    # norm=Normalize(vmin=-200, vmax=200),
                    angles='xy', scale_units='xy', scale=1,
                    headaxislength=0, headwidth=0, headlength=0, cmap='bwr',
                    width=0.003)
    # if site == 'A2_1':
    axs[i].quiver(NN1[site].loc[:,'x_fit'], NN1[site].loc[:,'y_fit'], 
                    NN1[site].loc[:,'dx'], NN1[site].loc[:,'dy'],
                    (np.linalg.norm(NN1[site].loc[:,'dx':'dy']
                                    .to_numpy(dtype=np.float64),
                                   axis=1) 
                     - np.linalg.norm(acl_rot.a1))*len_per_pix*100,
                    # norm=Normalize(vmin=-200, vmax=200),
                    angles='xy', scale_units='xy', scale=1,
                    headaxislength=0, headwidth=0, headlength=0, cmap='bwr',
                    width=0.003)
    
# axins = axs[1].inset_axes([1.05, 0.0, 1.0, 1.0])
# axins.imshow(acl_rot.image, cmap='gray')
# cb = axins.quiver(NN2[site].loc[:,'x_fit'], NN2[site].loc[:,'y_fit'],
#                    NN2[site].loc[:,'dx'],(NN2[site].loc[:,'dy']),
#                    (np.linalg.norm(
#                        NN2[site].loc[:,'dx':'dy'].to_numpy(dtype=np.float64),
#                        axis=1)
#                     - np.linalg.norm(acl_rot.a2))*len_per_pix*100,
#                    norm=Normalize(vmin=-35, vmax=35),
#                    angles='xy', scale_units='xy', scale=1,
#                    headaxislength=0, headwidth=0, headlength=0, cmap='bwr', 
#                    width=0.005)
# cb = axins.quiver(NN1[site].loc[:,'x_fit'], NN1[site].loc[:,'y_fit'], 
#                    NN1[site].loc[:,'dx'], NN1[site].loc[:,'dy'],
#                    (np.linalg.norm(NN1[site].loc[:,'dx':'dy']
#                                    .to_numpy(dtype=np.float64),
#                                    axis=1) 
#                     - np.linalg.norm(acl_rot.a1))*len_per_pix*100,
#                    norm=Normalize(vmin=-35, vmax=35),
#                    angles='xy', scale_units='xy', scale=1,
#                    headaxislength=0, headwidth=0, headlength=0, cmap='bwr',
#                    width=0.005)
# axins.text(x_crop[0]+20, y_crop[1]-30, title[2], 
#            color='black', size=12, weight='bold')
# axins.set_xticks([])
# axins.set_yticks([])

# axins.set_xlim(x_crop[0], x_crop[1])
# axins.set_ylim(y_crop[0], y_crop[1])

# axins.arrow(305, y_crop[0]-50,
#           0, -80,
#           fc='white', ec='white', width=3, length_includes_head=True,
#           head_width=6, head_length=15)
# axins.arrow(472, y_crop[0]-50,
#           0, -80,
#           fc='white', ec='white', width=3, length_includes_head=True,
#           head_width=6, head_length=15)
# axins.arrow(638, y_crop[0]-50,
#           0, -80,
#           fc='white', ec='white', width=3, length_includes_head=True,
#           head_width=6, head_length=15)
# scalebar = ScaleBar(a_2d[0,0].item()/np.linalg.norm(acl_rot.a1)/10,'nm',
#                     location='lower right', fixed_value=2,
#                     font_properties={'size': 10}, box_color='lightgrey',
#                     width_fraction=0.015)

# axins.add_artist(scalebar)
# rect, lines = axs[1].indicate_inset([x_crop[0], y_crop[1], 500, 500], axins,
#                                     edgecolor='black', linewidth=1, alpha=1)
# for line in lines:
#     line.set_linewidth(1)
    
# for axis in ['top','bottom','left','right']:
#   axins.spines[axis].set_linewidth(1)
  
# cbar_ax = axs[1].inset_axes([2.1, 0., 0.05, 1])
# cbar = fig.colorbar(cb, cax=cbar_ax, shrink=0.5,aspect=15)
# cbar.set_label(label = r'$\Delta$ NN distance (pm)', #weight = 'bold',
#                 fontsize = 10)
# cbar.ax.tick_params(labelsize=10)

# plt.savefig(save_folder + 'Figure_7.eps')
