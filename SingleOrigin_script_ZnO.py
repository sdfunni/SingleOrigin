import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import pickle
import imageio
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib import colors as colors
from scipy.optimize import minimize
from matplotlib.patches import Wedge
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import spines
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as Circle
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerPatch
from matplotlib.cm import ScalarMappable
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Ellipse
from CifFile import ReadCif
import copy

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import SI_LENGTH_RECIPROCAL

from SingleOrigin import *

#%%
# Data used for AdvMat Paper:
# exp_img = '1928 20210722 HAADF 2.70 Mx Zn0_3 HAADF_SDCorr.tif'
exp_img = '1926 20210726 DF4 7.60 Mx ZnO_3 iDPC_SDCorr.tif'

#%%
'''Basis atom offset from cell origin'''
offset = [0, 0]


'''Structure import from .cif and project'''
'''*** User inputs ***'''
'''BSSN'''
# cif = 'BSSN5137.cif'
cif = 'ZnO.cif'
# cif = 'ANT55_010_pc.cif'
# cif = 'PMN.cif'
za = np.array([1,1,0])            #Zone Axis direction
a2_p = np.array([-1,1,0])         #Apparent horizontal axis in projection
a3_p = np.array([0,0,1])          #Most vertical axis in projection

za_str = ''.join([str(i) for i in np.array(za)])
# pix_per_unit = 6.0625

cif_data= copy.deepcopy(ReadCif(cif))

#%%
'''Find 3D coordinates of unit cell, metric tensor and 3D direct 
stucture matrix (a_3d)'''
uc = UnitCell(cif_data)
uc.transform_basis(za, a2_p, a3_p)

'''Project Unit Cell to 2D and combine coincident/proximate columns'''
uc.project_uc_2d(proj_axis = 0)

uc.combine_prox_cols(prox_toler = 1e-2)

a_2d = uc.a_2d
at_cols = uc.at_cols
#%%
# at_cols = LatticeSiteLabeling(at_cols)
# at_cols.sort_values(by='LatticeSite', inplace=True)
at_cols = at_cols.sort_values(by='elem')
at_cols.reset_index(drop=True, inplace=True)
at_cols.reset_index(drop=False, inplace=True)
# at_cols.loc[:, 'LatticeSite'] = ['O', 'B', 'A']
sites = pd.unique(at_cols.elem).tolist()
title = sites

#%%
# '''Plot Unit Cell Projection for Verification'''
'''***Should make this a function'''
lab = 'elem'
fig,axs = plt.subplots(ncols=1,figsize=(10,10))
axs.set_aspect(1)
p = patches.Polygon(np.array([[0, 0],
                              [a_2d[0,0], 0],
                              [a_2d[0,0]+a_2d[0,1], a_2d[1,1]],
                              [a_2d[0,1], a_2d[1,1]]]),
                    fill = False, ec = 'black', lw = 0.5)
axs.add_patch(p)
axs.set_facecolor('grey')
ColList = np.sort(pd.unique(at_cols[lab])).tolist()
cmap = plt.get_cmap('RdYlGn')
c = {ColList[x] : np.array([cmap(x/(len(ColList)-1))],
                                ndmin=2) for x in range(len(ColList))}

cart_axs = list(set(at_cols.columns.tolist()) & {'x', 'y', 'z'})
cart_axs.sort()
for site in ColList:
    axs.scatter(at_cols.loc[at_cols[lab] == site].loc[:, cart_axs[0]],
                at_cols.loc[at_cols[lab] == site].loc[:, cart_axs[1]],
                c=c[site], vmin = 0, vmax = 1, s=300)

for ind in at_cols.index:
    axs.annotate(rf'${ at_cols.at[ind, lab] }$',
                  (at_cols['x'][ind] + 1/8, at_cols['y'][ind] + 1/8), 
                  fontsize=20)
axs.set_xticks([])
axs.set_yticks([])

#%%
'''Import experimental image and normalize to 0-1'''
TEM_micro = imageio.imread('STEM_Images/' + exp_img).astype('float64')
TEM_micro = image_norm(TEM_micro)
TEM_micro = TEM_micro[140:-140, 140:-140]

TEM_micro = image_norm(-gaussian_laplace(TEM_micro, 1.7)) #dDPC image
#%%
plt.figure()
plt.imshow(TEM_micro, cmap='gray')

#%%
'''Create a crystal lattice  object
    -Select an atom column to register the crystallographic origin.
    -Must specify an offset if the selected atomic column is not located at
    the unit cell origin: default is (0,0).
    -Use FFT peaks to find basis vectors for the image.'''
    
acl = AtomicColumnLattice(TEM_micro, at_cols, a_2d, at_cols=None)
acl.select_basis_vect_FFT(a1_order=1, a2_order=2, sigma=5)

#%%
'''Generate a mathematical lattice from the basis vectors and unit cell 
    projection
   -First fit prominent columns (ideally a sublattice with minimal 
    displacements) with 2D Gaussians, using projected lattice positions as
    initial guesses. Use this sublattice to refine the basis
    vectors before finding all lattice positions.'''

acl.unitcell_template_method(a1_var='u', a2_var='v',
                             offset = at_cols.loc[2, 'u':'v'].tolist(), 
                             buffer=20, filter_type='LoG', sigma=2.5,
                             fit_filtered_data=False,
                             edge_max_thresholding = 0)

'''Check angle between basis vectors'''
theta = np.degrees(np.arccos(acl.alpha[0,:] @ acl.alpha[1,:].T
                             /(np.linalg.norm(acl.alpha[0,:]) 
                               * np.linalg.norm(acl.alpha[1,:].T))))

#%%
'''Use the fitted atomic column positions to refine the basis vectors and 
    origin. 
    -It is best to choose a sublattice with minimal displacements. 
    -It also must have only one column per projected unit cell. 
    -If no sublattice meets this criteria, specify a specific column in the 
        projected cell.'''
 
acl.refine_lattice('elem', 'Zn')

'''Re-check angle between basis vectors'''
theta_ref = np.degrees(np.arccos(acl.alpha[0,:] @ acl.alpha[1,:].T
                                 /(np.linalg.norm(acl.alpha[0,:]) 
                                   * np.linalg.norm(acl.alpha[1,:].T))))

shear_distortion_res = np.radians(90 - theta_ref)

scale_distortion_res = 1 - ((np.linalg.norm(acl.a1)
                             * np.linalg.norm(acl.a_2d[1,:]))/
                            (np.linalg.norm(acl.a2)
                             * np.linalg.norm(acl.a_2d[0,:])))

len_per_pix = np.average([np.linalg.norm(acl.a_2d[0,:])
                          /np.linalg.norm(acl.a1),
                          np.linalg.norm(acl.a_2d[1,:])
                          /np.linalg.norm(acl.a2)])

#%%
'''Plot Column positions with color indexing'''
fig,axs = plt.subplots(ncols=1,figsize=(10,10))
axs.imshow(acl.image * acl.all_masks)
# axs.imshow(acl.image, cmap='gray')
# axs.imshow(masked_img, cmap = 'gray')
axs.set_xticks([])
axs.set_yticks([])

filtered = acl.at_cols

uc = filtered[(filtered['u'] // 1 == 0) & (filtered['v'] // 1 == 0)]

color_code = {k:v for v, k in 
              enumerate(np.sort(uc.loc[:,'elem'].unique()))}
color_list = [color_code[site] for site in filtered.loc[:,'elem']]

axs.scatter(filtered.loc[:,'x_fit'], filtered.loc[:,'y_fit'],
            c=color_list, s=4)

axs.arrow(acl.x0, acl.y0, acl.a1[0], acl.a1[1],
              fc='red', ec='red', width=0.1, length_includes_head=True,
              head_width=2, head_length=3, label=r'[001]')
axs.arrow(acl.x0, acl.y0, acl.a2[0], acl.a2[1],
              fc='green', ec='green', width=0.1, length_includes_head=True,
              head_width=2, head_length=3, label=r'[110]')

#%%
'''Rotate the image and data to align a desired basis vector to horizontal
    or vertical'''

acl_rot = acl.rotate_image_and_data(align_basis='a1', align_dir='horizontal')

#%%
'''Plot Column positions with color indexing'''
'''*** Make function'''
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
fig.tight_layout()

#%%
'''Plot Column positions with displacement vectors'''
'''*** Make a function'''
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
    # print(np.max(norms)*cbar_range)
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
'''Plot A2_1 NN distance vectors'''
'''*** ?? Make function ??'''

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


#%%
'''Find the modulation wave vector'''
g_2d = np.array([[a_2d[0,0]**2, 0],
                 [0, a_2d[1,1]**2]])
a=12.5876
b=a
c=3.9622
alpha=90
beta=90
gamma=90

[h,k,l]=[0.32,0.32,0.5]



g=lfn.metric_tensor(a,b,c,alpha,beta,gamma)

d_hkl=lfn.IntPlSpc(h,k,l,g)
g_hkl=d_hkl**-1
q_real = np.array([h, l]) @ np.linalg.inv(g_2d) #In terms of crystal basis vectors

#In terms of x-y coordinte system (Angstroms)
q_Ang = (q_real @ acl_rot.a_2d/np.linalg.norm(q_real @ acl_rot.a_2d)*d_hkl)    
#In terms of the crystal basis vectors
q_fract =q_Ang  @ np.linalg.inv(acl_rot.a_2d)
#In terms of image (pixel) coordinage system
q_img = q_fract @ acl_rot.alpha

#%%
'''1d PCFs'''
pcfs_1d ={}
filter_col = 'LatticeSite'
dr=0.01

A_box = (TEM_micro.shape[0] * TEM_micro.shape[1] 
         - 4*(20*(TEM_micro.shape[1]-20))) * len_per_pix**2

for site in sites:
    cols = acl_rot.at_cols[(acl_rot.at_cols[filter_col]
                           == site)].loc[:, 'x_fit': 'y_fit'].to_numpy()
    cols *= len_per_pix
    
    pcfs_1d[site] = pcf_radial(dr, cols[:, 0], cols[:, 1], total_area=A_box)
    
#%%
'''Plot 1d PCF'''
fig,axs = plt.subplots(ncols=1,figsize=(3.5,3), tight_layout=True)
site = 'A'
r = np.arange(0, pcfs_1d[site].shape[0] * dr, dr)
axs.plot(r, pcfs_1d[site], color='black', zorder=1, lw=0.5)
axs.fill_between(r, pcfs_1d[site], color='grey', label=site, zorder=0)

site = 'B'
r = np.arange(0, pcfs_1d[site].shape[0] * dr, dr)
axs.plot(r, pcfs_1d[site], color='black', lw=0.5)
axs.fill_between(r, pcfs_1d[site], color='orangered', label=r'$A2_1$',
                 alpha=0.65, zorder=2)

site = 'O'
r = np.arange(0, pcfs_1d[site].shape[0] * dr, dr)
axs.plot(r, pcfs_1d[site], color='black', lw=0.5)
axs.fill_between(r, pcfs_1d[site], color='orangered', label=r'$A2_1$',
                 alpha=0.65, zorder=2)

axs.legend(loc='upper right', fontsize=8)
axs.set_xlabel(r'r ($\AA$)', fontsize=10)
axs.set_ylabel(r'g(r)', fontsize=10)
axs.set_xlim((0, 10))
axs.set_ylim((0, 20))
axs.text(7.9, 12.5, r'$\|2\mathbf{\bar c}\|$', color='black', size=8, 
         ha='center')
axs.text(8.9, 11, r'$\|\frac{1}{2} \left[ 110 \right] \|$', color='black', 
         size=8, ha='center')
axs.text(9.7, 17.5, r'$\|\mathbf{\bar c} + \frac{1}{2} \left[ 110 \right] \|$', 
         color='black', size=8, ha='center')
axs.text(11.7, 13, r'$\|3\mathbf{\bar c}\|$', color='black', size=8, 
         ha='right')
axs.text(12.1, 13, r'$\|2\mathbf{\bar c} + \frac{1}{2} \left[ 110 \right] \|$', 
         color='black', size=8, ha='left')
axs.set_yticks([0,5,10,15,20])

plt.savefig(save_folder + 'Fig_3.pdf')

#%%
'''Calculate Pair-Pair Correlation Function'''
pair_pair_pcfs = {}
filter_col = 'LatticeSite'
sites = [site for site in pd.unique(uc.LatticeSite)]
sites.sort()
sites.remove('A2_2')
sites.remove('B1')
pair_pair = [[site1, site2]
             for count, site1 in enumerate(sites)
             for site2 in sites[count:]]#  if site1 != site2]

at_cols = acl_rot.at_cols[np.linalg.norm(
    (acl_rot.at_cols.loc[:, 'x_fit':'y_fit'].to_numpy()
     - acl_rot.at_cols.loc[:, 'x_ref':'y_ref'].to_numpy()), axis=1) < 8]

a1_start = -5.5
a1_stop = 5.5
a2_start = -5.5
a2_stop = 5.5

bin_dim = 0.02  #In Angstroms
dr = bin_dim/len_per_pix #convert to pixels

a1_mag = np.linalg.norm(acl_rot.a1)
a2_mag = np.linalg.norm(acl_rot.a2)

dx = np.arange(a1_start*a1_mag, a1_stop*a1_mag, dr)
dy = np.arange(a2_start*a2_mag, a2_stop*a2_mag, dr)

for pair in pair_pair:
    A_box = (TEM_micro.shape[0] * TEM_micro.shape[1] 
             - 4*(20*(TEM_micro.shape[1]-20))) * len_per_pix**2
    
    cols1 = at_cols[
        (at_cols[filter_col] == pair[0])].loc[:, 'x_fit': 'y_fit'].to_numpy()
    
    cols2 = at_cols[
        (at_cols[filter_col] == pair[1])].loc[:, 'x_fit': 'y_fit'].to_numpy()
    
    rho = cols2.shape[0]/A_box
    
    vects = np.array([cols1 - i for i in cols2])
    

    
    H, _, _ = np.histogram2d(vects[:,:,1].flatten(), vects[:,:,0].flatten(),
                             bins = [dy, dx])
    dx_0 = np.where(dx == 0)
    
    max_ind = np.unravel_index(np.argmax(H), H.shape)
    H[max_ind[0], max_ind[1]] = 0

    ppcf12 = H/rho          #Calculate PCF
    pair_pair_pcfs[pair[0] + '|' + pair[1]] = ppcf12
    
[pp_x0, pp_y0] = [H.shape[1]*(-a1_start)/(a1_stop - a1_start),
            H.shape[0]*(-a2_start)/(a2_stop - a2_start)]

# #%%
# plt.figure()
# plt.imshow(pair_pair_pcfs['AB'])
# plt.scatter(pp_x0, pp_y0, c='white')

#%%
'''Calculate peak moments '''
pcfs_ = pair_pair_pcfs
pcf_peaks = {}

# pcfs_ = partial_pcfs
# partial_pcf_peaks = {}

for site in pcfs_.keys():
    print(site)
    
    lattice_pt = (uc[uc.loc[:,'LatticeSite'] == site.split('|')[1]
                    ].loc[:, 'u':'v'].to_numpy()
                  - uc[uc.loc[:,'LatticeSite'] == site.split('|')[0]
                    ].loc[:, 'u':'v'].to_numpy()).flatten()
    lattice_pt = np.array([1 if ind == 0 else ind for ind in lattice_pt])
    
    a1_stop_ = a1_stop + 1 if lattice_pt[0] % 1 != 0 else a1_stop
    a2_stop_ = a2_stop + 1 if lattice_pt[1] % 1 != 0 else a2_stop

    uv = np.array([[i,j] + lattice_pt
                         for i in range(int(a1_start//1), int(a1_stop_//1)) 
                         for j in range(int(a2_start//1), int(a2_stop_//1))
                         if (list([i,j] + lattice_pt) != [0,0])])
    
    xy = uv @ acl_rot.a_2d/bin_dim + [pp_x0, pp_y0]
    
    pcf_peaks[site] = pd.DataFrame(columns=['u', 'v', 'x_ref', 'y_ref', 
                                            'x_fit', 'y_fit', 
                                            'sig_maj', 'sig_min', 
                                            'ecc', 'theta'],
                                   index = range(xy.shape[0]))
    
    pcf_peaks[site].loc[:, 'u':'y_ref'] = np.concatenate((uv,xy), axis=1)
    
    pcf_sm = ndimage.filters.gaussian_filter(pcfs_[site], sigma=20, truncate=19)
    masks, num_masks, slices, peaks = watershed_segment(pcf_sm, 
                                                        sigma=None, 
                                                        buffer=0)
    CoMs = peaks.loc[:, 'x':'y'].to_numpy()
    
    inds = [np.argmin(np.linalg.norm(CoMs - xy_, axis=1)) + 1 for xy_ in xy]
    
    for i, ind in tqdm(enumerate(inds)):

        pcf_masked = np.where(masks == ind, 1, 0)*pcfs_[site]

        x_fit, y_fit, ecc, theta, sig_maj, sig_min = img_ellip_param(pcf_masked)
        pcf_peaks[site].loc[i,'x_fit':] = [x_fit, y_fit, 
                                           sig_maj, sig_min, ecc, theta]
    pcf_peaks[site] = pcf_peaks[site].astype(float, copy=False)

#%%

pcfs = pair_pair_pcfs
[x0, y0] = [pp_x0, pp_y0]

fig = plt.figure(figsize=(26,13))
gs = gridspec.GridSpec(nrows=2, ncols=3, #width_ratios=[6,6,6], 
                       hspace=0.15, wspace=0.05)

font=14
axs = [i for i in range(len(pcf_peaks))]
cmap = plt.cm.plasma

ylim = [np.floor(y0 - 0.25*acl_rot.a_2d[1,1]/bin_dim), 
        np.ceil(y0 + acl_rot.a_2d[1,1]/bin_dim*10.75)]
xlim = [np.floor(x0 - 0.25*acl_rot.a_2d[0,0]/bin_dim), 
        np.ceil(x0 + acl_rot.a_2d[0,0]/bin_dim*10.75)]

for i, key in enumerate(pcfs.keys()):

    axs[i] = fig.add_subplot(gs[i//3, i%3])
    
    '''Eccentricity Plotting'''
    
    ells = [Ellipse(xy=[peak.x_fit, peak.y_fit], 
                    width=peak.sig_maj*15,
                    height=peak.sig_min*15,
                    angle=-peak.theta,
                    facecolor = cmap(peak.ecc),
                    lw=2,
                    zorder=1,
                    alpha = 0.7)
            for peak in pcf_peaks[key].itertuples()]
    
    for ell in ells:
        axs[i].add_artist(ell)
    
    hsv = np.ones((pcf_peaks[key].shape[0], 3))
    dxy = (pcf_peaks[key].loc[:,'x_fit':'y_fit'].to_numpy(dtype=float)
           - pcf_peaks[key].loc[:,'x_ref':'y_ref'].to_numpy(dtype=float))
    
    
    axs[i].imshow(pcfs[key], cmap= 'gist_yarg')
    # axs[i].set_ylim(ylim[0], ylim[1])
    # axs[i].set_xlim(xlim[0], xlim[1])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].text(0.05, 1.05,
        # x0 + acl_rot.a_2d[0,0]/bin_dim*0.5, 
        #         y0 + acl_rot.a_2d[1,1]/bin_dim*6.5, 
                key, color='black', size=font,
             horizontalalignment='center', weight='bold',
             transform=axs[i].transAxes).set_clip_on(False)
    axs[i].scatter(x0, y0, c='red')
    
    # axs[i].arrow(x0, y0, acl_rot.a_2d[1,0]/bin_dim, acl_rot.a_2d[1,1]/bin_dim,
    #               fc='red', ec='red', width=0.2, length_includes_head=True,
    #               head_width=30, head_length=50, label=r'[001]',
    #                   zorder=10)
    # axs[i].arrow(x0, y0, acl_rot.a_2d[0,0]/bin_dim, acl_rot.a_2d[0,1]/bin_dim,
    #               fc='green', ec='green', width=0.2, length_includes_head=True,
    #               head_width=30, head_length=50, label=r'frac{1}{2} [110]',
    #                   zorder=10)
    # if i == 1:
    #     scalebar = ScaleBar(bin_dim/10,'nm', font_properties={'size': 12},
    #                         pad=0.3,  border_pad=0.6,
    #                         box_color='dimgrey', height_fraction=0.02,
    #                         color='white', location='lower right', fixed_value=1)
    #     # scalebar.box_alpha = 0.7
    #     axs[i].add_artist(scalebar)
        
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(1.5)
        
# Colorbar
ax_cbar = fig.add_subplot(gs[0,3])
ax_cbar.axis('off')
im_ratio = pcfs[key].shape[0]/pcfs[key].shape[1]
cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), 
                                      cmap='plasma'),
                    ax=ax_cbar, 
                    orientation='vertical', 
                    fraction=0.4,
                    aspect=7, 
                    ticks = [0.0, 0.5, 1.0],
                    alpha=0.5)
ax_cbar.set_xticks([])
ax_cbar.set_yticks([])
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Eccentricity', fontsize = 16)

#%%
'''Fit Eccentricity vs phase'''

def sin_fn(phase, amp, offset):
    ecc_ = (amp) * np.abs(np.sin(phase*np.pi)) + offset
    return ecc_
site_ = 'A2_1'
fit_data = pcf_peaks[site_][(pcf_peaks[site_].x < H.shape[1]*(0.5 + 0.5*0.4)) 
                            & (pcf_peaks[site_].x > H.shape[1]*0.48)
                            & (pcf_peaks[site_].y < H.shape[0]*0.75) 
                            & (pcf_peaks[site_].y > H.shape[0]*0.48)]
xy = fit_data.loc[:, 'x':'y'].to_numpy(dtype=float) - np.array([x0, y0])
z = fit_data['ecc'].to_numpy(dtype=float)
def sin_fit(p0, xy, z, g):
    [h, l, amp, offset] = p0
    # print(h)
    # xy = np.vstack((x,y)).T

    d_hkl=lfn.IntPlSpc(h,h,l,g)
    q_real = np.array([h, l]) @ np.linalg.inv(g_2d) #In terms of crystal basis vectors
    
    #In terms of x-y coordinte system (Angstroms)
    q_Ang = (q_real @ acl_rot.a_2d/np.linalg.norm(q_real @ acl_rot.a_2d)*d_hkl)    
    q = np.array(q_Ang/bin_dim)

    phase = (xy @ q.T) / (np.linalg.norm(q)**2)
    
    # R = (np.pi/2 - np.abs(phase/2 - np.pi/2)) - np.arcsin(z)
    S = np.squeeze(np.array([sin_fn(ph, amp, offset) for ph in phase]))
    R = z - S
    sum_sq = np.sum(R**2)
    # print(sum_sq)
    return sum_sq

[max_, min_] = [np.max(pcf_peaks[site_].ecc), np.min(pcf_peaks[site_].ecc)]

'''Sigma major fitting'''
# p0 = np.array([0.325, 0.5, 4, 1])
# bounds = [(0.2, 0.4), (0.5, 0.5),
#           (2, 4), (0, 3)]

'''Eccentricity fitting'''
p0 = np.array([0.325, 0.5, 0.2, 0.74])
bounds = [(0.2, 0.4), (0.5, 0.5),
          (0, 1), (0, 1)]
params = minimize(sin_fit, p0, args=(xy, z, g), bounds=bounds,
                  method='Powell').x
print(params)


#%%
'''MCMC-DRAM parameter estimation'''


#%%
'''Modulation "planes" for 2D PDFs'''
# params[0] = 0.3219
[h,k,l]=[params_B[0], params_B[0], 0.5]
g=lfn.metric_tensor(a,b,c,alpha,beta,gamma)
d_hkl=lfn.IntPlSpc(h,k,l,g)
g_hkl=d_hkl**-1
q_real = np.array([h, l]) @ np.linalg.inv(g_2d) #In terms of crystal basis vectors
#In terms of x-y coordinte system (Angstroms)
q_Ang = (q_real @ acl_rot.a_2d/np.linalg.norm(q_real @ acl_rot.a_2d)*d_hkl)

q_pcf = q_Ang/bin_dim
m = -q_pcf[0,0]/q_pcf[0,1]
x_mod = np.linspace(0,H.shape[1])

b_ = []
y_mod = np.zeros((11, x_mod.shape[0]))
for n in range(-1, 10):
    b_ += [(y0 + n*q_pcf[0,1]) - m * (x0 + n*q_pcf[0,0])]
    y_mod[n,:] = m*x_mod + b_[-1]


#%% 
'''2D PCFs Figure 8 Sublattices'''

def sin_fn_line(phase, params):
    [alpha, amp, offset] = params
    ecc_ = (amp) * np.sqrt(np.abs(np.sin(phase*np.pi))) + offset
    return ecc_

fig = plt.figure(figsize=(12,8.5))
gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[6,6], 
                       height_ratios=[2,1,2,1.5], hspace=0.15, wspace=0.05)
# site_labels = [r'A$2_1$', r'A$1$', r'$\alpha$=0.31', r'$\alpha$=0.32',
#                 r'$\alpha=\frac{1}{3}$']
# site_labels = ['A1', r'A$2_1$', 'B1', 'B2']
# site_labels = ['A2_1']
site_labels = ['(a)', '(b)', '(c)', '(d)']
keys = ['A1', 'A2_1', 'B1', 'B2']
ax_ind = [[0,0],[0,1],[2,0],[2,1]]
font=14
axs = [i for i in range(4)]
cmap = plt.cm.plasma

ylim = [np.floor(y0 - 0.5*acl_rot.a_2d[1,1]/bin_dim), 
        np.ceil(y0 + acl_rot.a_2d[1,1]/bin_dim*8.5)]
xlim = [np.floor(x0 - 0.5*acl_rot.a_2d[0,0]/bin_dim), 
        np.ceil(x0 + acl_rot.a_2d[0,0]/bin_dim*10.5)]

for i, key in enumerate(keys):
    axs[i] = fig.add_subplot(gs[ax_ind[i][0], ax_ind[i][1]])
    
    '''Eccentricity Plotting'''
    
    ells = [Ellipse(xy=[peak.x, peak.y], 
                    width=peak.sig_maj*25,
                    height=peak.sig_min*25,
                    angle=-peak.theta,
                    facecolor = cmap(peak.ecc),
                    lw=2,
                    zorder=1)
            for peak in pcf_peaks[key].itertuples()]
    
    for ell in ells:
        axs[i].add_artist(ell)
        
    axs[i].imshow(pcfs[key], cmap= 'gist_yarg')
    axs[i].set_ylim(ylim[0], ylim[1])
    axs[i].set_xlim(xlim[0], xlim[1])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].text(0.05, 1.05,
        # x0 + acl_rot.a_2d[0,0]/bin_dim*0.5, 
        #         y0 + acl_rot.a_2d[1,1]/bin_dim*6.5, 
                site_labels[i], color='black', size=font,
             horizontalalignment='center', weight='bold',
             transform=axs[i].transAxes).set_clip_on(False)
    
    axs[i].arrow(x0, y0, acl_rot.a_2d[1,0]/bin_dim, acl_rot.a_2d[1,1]/bin_dim,
                  fc='red', ec='red', width=0.2, length_includes_head=True,
                  head_width=30, head_length=50, label=r'[001]',
                      zorder=10)
    axs[i].arrow(x0, y0, acl_rot.a_2d[0,0]/bin_dim, acl_rot.a_2d[0,1]/bin_dim,
                  fc='green', ec='green', width=0.2, length_includes_head=True,
                  head_width=30, head_length=50, label=r'frac{1}{2} [110]',
                      zorder=10)
    if i == 1:
        axs[i].arrow(x0, y0, q_pcf[0,0], q_pcf[0,1],
                      fc='purple', ec='purple', width=0.2, length_includes_head=True,
                      head_width=30, head_length=50, label=r'$q_{real}$',
                      zorder=10)

        scalebar = ScaleBar(bin_dim/10,'nm', font_properties={'size': 12},
                            pad=0.3,  border_pad=0.6,
                            box_color='dimgrey', height_fraction=0.02,
                            color='white', location='lower right', fixed_value=1)
        # scalebar.box_alpha = 0.7
        axs[i].add_artist(scalebar)
        
        # plot phase
        x, y, _ = np.meshgrid(np.arange(xlim[0], xlim[1]), 
                              np.arange(ylim[0], ylim[1]), [1])
        y = np.flipud(y)
        xy = np.concatenate((x,y), axis=2)
        phase = (((xy - np.array([x0, y0])) @ q_pcf.T) 
                 / np.linalg.norm(q_pcf)**2 % 1)
        extents = np.min(x), np.max(x), np.min(y), np.max(y)
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        cmap_ph = plt.get_cmap('Greys')
        new_cmap = truncate_colormap(cmap_ph, 0.2, 0)

        axs[i].imshow(phase, cmap=new_cmap, interpolation='bilinear', extent = extents)
    
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(1.5)
axs[0].set_zorder(10)
'''Axes for outsets and legends'''
center = gridspec.GridSpecFromSubplotSpec(1, 14, subplot_spec=gs[1,:],
                                         wspace=0.3, hspace=0.1)

# Colorbar
ax_cbar = fig.add_subplot(center[0,12:])
ax_cbar.axis('off')
im_ratio = pcfs[key].shape[0]/pcfs[key].shape[1]
cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1), 
                                      cmap='plasma'),
                    ax=ax_cbar, 
                    orientation='vertical', 
                    fraction=0.7,
                    aspect=7, 
                    ticks = [0.0, 0.5, 1.0])
ax_cbar.set_xticks([])
ax_cbar.set_yticks([])
cbar.ax.tick_params(labelsize=14)
cbar.set_label(label = 'Eccentricity', fontsize = 16)

# Legend
legend = [FancyArrow(10, -7, 0, 18,  fc='red', ec='red',
                     length_includes_head=True, head_width=7),
          FancyArrow(0, 3, 20, 0,  fc='green', ec='green',
                     length_includes_head=True, head_width=7),
          FancyArrow(0, 3, 20, 0,  fc='purple', ec='purple',
                     length_includes_head=True, head_width=7)]
def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = orig_handle
    return p

legend_ = fig.add_subplot(center[0, :2], anchor='C')
legend_.legend(handles = legend, loc='center', fontsize=12,
           labels = [r'[001]', r'$\frac{1}{2} [1 \bar 1 0]$', r'$q_{real}$'],
           handler_map={FancyArrow : HandlerPatch(patch_func=
                                                  make_legend_arrow),})
legend_.axis('off')


'''High eccentricity outset'''
ax_outset1 = fig.add_subplot(center[0,2:6])
zoom1 = ax_outset1.imshow(pcfs['A2_1'], cmap= 'gist_yarg')
ax_outset1.set_xticks([])
ax_outset1.set_yticks([])

x_crop = [x0 - 15, x0 +15] 
y_crop = [y0 + acl_rot.a_2d[1,1]/bin_dim*4 - 6,
          y0 + acl_rot.a_2d[1,1]/bin_dim*4 + 6]
ax_outset1.set_xlim(x_crop[0], x_crop[1])
ax_outset1.set_ylim(y_crop[0], y_crop[1])
    
for axis in ['top','bottom','left','right']:
    ax_outset1.spines[axis].set_linewidth(2)
    ax_outset1.spines[axis].set_color('red')
    
sq = (np.average(x_crop)-50, np.average(y_crop)-50)
rect = Rectangle(sq, 100, 100,
                 linewidth=2, edgecolor='red', facecolor='none')

axs[1].add_patch(rect)

xy_f = (sq[0], sq[1]+100)
xy_t = (x_crop[0], y_crop[1])
# con = ConnectionPatch(xyA=xy_f, xyB=xy_t, coordsA='data', coordsB='data',
#                       axesA=axs[1], axesB=ax_outset1, color="black",
#                       linewidth=2)
# axs[1].add_artist(con)
xy_f = (sq[0]+100, sq[1])
xy_t = (x_crop[1], y_crop[0])
# con = ConnectionPatch(xyA=xy_f, xyB=xy_t, coordsA='data', coordsB='data',
#                       axesA=axs[1], axesB=ax_outset1, color="black",
#                       linewidth=2)
# axs[1].add_artist(con)

'''Low eccentricity outset'''
ax_outset2 = fig.add_subplot(center[0,8:-2])
g_r = zoom2 = ax_outset2.imshow(pcfs['A2_1'], cmap= 'gist_yarg')
ax_outset2.set_xticks([])
ax_outset2.set_yticks([])

x_crop = [x0 - 15, x0 +15] 
y_crop = [y0 + acl_rot.a_2d[1,1]/bin_dim*7 - 6,
          y0 + acl_rot.a_2d[1,1]/bin_dim*7 + 6]

ax_outset2.set_xlim(x_crop[0], x_crop[1])
ax_outset2.set_ylim(y_crop[0], y_crop[1])

sq = (np.average(x_crop)-50, np.average(y_crop)-50)
rect2 = Rectangle((np.average(x_crop)-50, np.average(y_crop)-50), 100, 100,
                 linewidth=2, edgecolor='blue', facecolor='none')
axs[1].add_patch(rect2)

for axis in ['top','bottom','left','right']:
    ax_outset2.spines[axis].set_linewidth(2)
    ax_outset2.spines[axis].set_color('blue')

xy_f = (sq[0], sq[1])
xy_t = (x_crop[0], y_crop[0])
# con = ConnectionPatch(xyA=xy_f, xyB=xy_t, coordsA='data', coordsB='data',
#                       axesA=axs[1], axesB=ax_outset2, color="black",
#                       linewidth=2)
# axs[1].add_artist(con)

xy_f = (sq[0]+100, sq[1]+100)
xy_t = (x_crop[1], y_crop[1])
# con = ConnectionPatch(xyA=xy_f, xyB=xy_t, coordsA='data', coordsB='data',
#                       axesA=axs[1], axesB=ax_outset2, color="black",
#                       linewidth=2)
# axs[1].add_artist(con)


ax_gr_cbar = fig.add_subplot(center[0,6:8])
ax_gr_cbar.axis('off')
# im_ratio = pcfs[key].shape[0]/pcfs[key].shape[1]
cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, 
                                                  vmax=np.max(pcfs['A2_1'])), 
                                   cmap='gist_yarg'),
                    ax=ax_gr_cbar, 
                    orientation='vertical',
                    # location = 'right',
                    fraction=0.7,
                    aspect=7,
                    ticks = [0, 2500, 5000, 7500])
ax_cbar.set_xticks([])
ax_cbar.set_yticks([])
cbar.ax.tick_params(labelsize=9)
cbar.set_ticklabels([0, r'$2\cdot 10^3$', r'$4\cdot 10^3$',
                                    r'$6\cdot 10^3$'])
cbar.set_label(label = r'$ g(\vec r)$', fontsize = 12)

'''Eccentricity as a function of phase'''

ph_ = np.linspace(0, 9, 1000)
ecc_ = sin_fn_line(ph_, params_B)

d_hkl=lfn.IntPlSpc(h,k,l,g)
q_real = np.array([h, l]) @ np.linalg.inv(g_2d) #In terms of crystal basis vectors

#In terms of x-y coordinte system (Angstroms)
q_Ang = (q_real @ acl_rot.a_2d/np.linalg.norm(q_real @ acl_rot.a_2d)*d_hkl)    
# q_pcf = np.array(q_Ang/bin_dim)

# q_pcf = np.array(params[:2], ndmin=2)

pcf_fit = pcf_peaks['A2_1'][(pcf_peaks[site_].x < H.shape[1]*(0.5 + 0.5*.4)) 
                            & (pcf_peaks[site_].x > H.shape[1]*0.49)
                            & (pcf_peaks[site_].y < H.shape[0]*1) 
                            & (pcf_peaks[site_].y > H.shape[0]*0.49)]
pcf_coords = pcf_fit.loc[:, 'x':'y'].to_numpy(dtype=float) - np.array([x0, y0])
pcf_spot_phase = (pcf_coords @ q_pcf.T) / np.linalg.norm(q_pcf)**2
pcf_spot_phase.resize((pcf_coords.shape[0],))

ecc = pcf_fit['ecc'].to_numpy(dtype=float)

pcf_fit2 = pcf_peaks['A2_1'][(pcf_peaks[site_].x > H.shape[1]*(0.5 + 0.5*.4)) 
                              & (pcf_peaks[site_].y > H.shape[0]*0.49)]
pcf_coords2 = pcf_fit2.loc[:, 'x':'y'].to_numpy(dtype=float) - np.array([x0, y0])
pcf_spot_phase2 = (pcf_coords2 @ q_pcf.T) / np.linalg.norm(q_pcf)**2
pcf_spot_phase2.resize((pcf_coords2.shape[0],))

ecc2 = pcf_fit2['ecc'].to_numpy(dtype=float)


axs_f = fig.add_subplot(gs[3, :])
axs_f.set_aspect(1.5)
axs_f.scatter(pcf_spot_phase, ecc, zorder=10, c='tab:blue',
              label=r'$A2_1$ Peaks: x $ \leq 4 \cdot (\frac{1}{2} [1 \bar 1 0])$')
axs_f.scatter(pcf_spot_phase2, ecc2, zorder=1, c='silver',
              label=r'$A2_1$ Peaks: x $ > 4 \cdot (\frac{1}{2} [1 \bar 1 0])$')
axs_f.plot(ph_, ecc_, c='red', label=r'Model Fit: $\mathcal{A} ' 
           +'\sqrt{|\mathrm{sin}(\phi (\bar r)/2)|} + b$')
axs_f.set_xlim((0, 9))
axs_f.set_ylim((0, 1))
axs_f.set_xticks([i for i in range(10)])
axs_f.set_xticklabels([rf'${ph}\pi$' for ph in range(0, 20, 2)], size=14)
axs_f.set_yticks(np.linspace(0,1,5))
axs_f.tick_params(labelsize=14)

axs_f.set_xlabel ('Phase of Modulation Wave (rad)', size=16)
axs_f.set_ylabel ('Eccentricity', size=16)
# axs_f.text(-2.6, 0.1, r'$A2_1$', color='black', size=font,
#              horizontalalignment='center', weight='normal')
# axs_f.text(-2.6, 0.1, r'A$2_1$', color='black', size=font,
#              horizontalalignment='center', weight='normal')
axs_f.text(0.2, 1.05, r'(e)', color='black', size=font,
             horizontalalignment='center', weight='bold')
axs_f.legend(loc='lower right')
for axis in ['top','bottom','left','right']:
        axs_f.spines[axis].set_linewidth(1.5)

# plt.savefig(save_folder + 'Figure_8.pdf')

#%%
scalebar = ScaleBar(bin_dim/10,'nm', font_properties={'size': 24},
                    pad=0.4, height_fraction=0.02)
scalebar.box_alpha = 0.7
axs.add_artist(scalebar)

#%%
'''Eccentricity as a function of phase'''

[h,k,l]=[params[0], params[0], params[1]]
max_, min_ = params[2], params[3]
ph_ = np.linspace(-3, 3, 1000)

ecc_ = np.array([sin_fn(ph, max_, min_) for ph in ph_])

g=lfn.metric_tensor(a,b,c,alpha,beta,gamma)

d_hkl=lfn.IntPlSpc(h,k,l,g)
q_real = np.array([h, l]) @ np.linalg.inv(g_2d) #In terms of crystal basis vectors

#In terms of x-y coordinte system (Angstroms)
q_Ang = (q_real @ acl_rot.a_2d/np.linalg.norm(q_real @ acl_rot.a_2d)*d_hkl)    
q_pcf = np.array(q_Ang/bin_dim)

# q_pcf = np.array(params[:2], ndmin=2)
pcf_coords = pcf_peaks[site_].loc[:, 'x':'y'].to_numpy(dtype=float) - np.array([x0, y0])
pcf_spot_phase = (pcf_coords @ q_pcf.T) / np.linalg.norm(q_pcf)**2
pcf_spot_phase.resize((pcf_coords.shape[0],))

ecc = pcf_peaks[site_]['ecc'].to_numpy(dtype=float)
R = ecc - np.array([sin_fn(ph, max_, min_) for ph in pcf_spot_phase])
sum_sq = np.sum(R**2)
print(sum_sq)

fig,axs = plt.subplots(ncols=1, figsize=(8, 3), tight_layout=True)
axs.scatter(pcf_spot_phase, ecc)
axs.plot(ph_, ecc_, c='red')
axs.set_xlim((-3, 3))
axs.set_ylim((0, 1))
axs.set_xticks([-3, -2, -1, 0, 1, 2, 3])
axs.set_xticklabels([ r'-$6 \pi$', r'-$4 \pi$', r'$-2 \pi$', '0', r'$2 \pi$',
                     r'$4 \pi$', r'$6 \pi$'], size=14)
axs.tick_params(labelsize=14)

axs.set_xlabel ('Phase of Modulation Wave (rad)', size=16)
axs.set_ylabel ('Peak Eccentricity', size=16)

#%%
plt.figure(0)
for site in sites:
    plt.plot(rs[site],pcfs[site], label=site)

# plt.plot(r,pcf, label = r'A1 - $A2_1$')
# plt.plot(r,np.ones(len(r)))
plt.xlim(0,20)
# plt.ylim(0,10)
plt.rc('font',size=12)
plt.title(r'Pair Correlation Functions')
plt.xlabel('Distance ($\AA$)')
plt.ylabel('g(r)')
plt.legend()

#%%
'''Something related to superspace. Not sure if I still need this'''
mu_ = acl.at_cols[(acl.at_cols.atom_j == 8) | (acl.at_cols.atom_j == 1)]
x_mu = mu_.loc[:,'u':'v'].to_numpy()
u_mu = mu_.loc[:,'u_1':'u_2'].to_numpy()
x_4 = list(x_mu @ np.array([0.6,0.5]).T)
plt.figure()
plt.scatter(x_4,list(u_mu[:,1]))


#%%[558, 7, 560, 537]
pickle_out=open(f'{savedir}/{structure}_{za}_AtColumns.pkl', "wb")
pickle.dump(at_cols,pickle_out)
pickle_out.close()

pickle_out=open(f'{savedir}/{structure}_{za}_Image.pkl', "wb")
pickle.dump(TEM_micro,pickle_out)
pickle_out.close()

#%%
'''***[001] BSSN stuff***'''

rng = 200
pad = 10
[y_mid, x_mid] = [int(i/2) for i in TEM_micro.shape]
contour_crop = at_cols[((at_cols.y_ <= y_mid+rng-pad) &
                        (at_cols.y_ >= y_mid-rng+pad) &
                        (at_cols.x_ <= x_mid+rng-pad) &
                        (at_cols.x_ >= x_mid-rng+pad))]
plt.figure(figsize=(10, 10))
plt.imshow(TEM_micro[y_mid-rng:y_mid+rng,x_mid-rng:x_mid+rng], cmap='gray')
plt.scatter(contour_crop.loc[:,'x_']-(x_mid-rng),
            contour_crop.loc[:,'y_']-(y_mid-rng),
            c=contour_crop.loc[:,'channel'], cmap='RdYlGn', s=5)
Yin, Xin = np.mgrid[0:rng*2, 0:rng*2]
for ind in contour_crop.index:
    plt.contour(Xin, Yin, g2d.gaussian_2d(Xin, Yin, 
        contour_crop.at[ind,'x_']-(x_mid-rng),
        contour_crop.at[ind,'y_']-(y_mid-rng), 
        contour_crop.at[ind,'sig_maj'], contour_crop.at[ind,'sig_rat'], 
        np.radians(contour_crop.at[ind,'theta']),
        contour_crop.at[ind,'peak_int'])/contour_crop.at[ind,'peak_int'], 
        [0.5, 0.8], colors='w', linewidths=0.5, alpha=0.5)
'''Plot orientations of A2 gaussians'''
A2_crop = contour_crop[contour_crop.LatticeSite == "A2"]
major = plt.quiver(A2_crop.loc[:,'x_']-(x_mid-rng),
           A2_crop.loc[:,'y_']-(y_mid-rng),
           np.sin(np.radians(A2_crop.loc[:,'theta']+90))*A2_crop.loc[:,'sig_maj'],
           np.cos(np.radians(A2_crop.loc[:,'theta']+90))*A2_crop.loc[:,'sig_maj'],
           A2_crop.loc[:,'sig_rat'], scale = 0.5, scale_units ='xy',
           pivot='mid', headlength = 0, headwidth = 0, headaxislength=0,
           cmap = 'OrRd')
cbar = plt.colorbar(major, shrink = 0.5)#A2_crop.loc[:,'sig_rat'])
cbar.set_label(r'$\sigma_{major} / \sigma_{minor}$')

#%%
'''Plot magnitude of ellipticality vs orientation of A2 columns'''
eccentric = at_cols[at_cols.loc[:,'LatticeSite'] == 'A2']
# eccentric = A2_crop[A2_crop.loc[:,'sig_rat'] >1.2]
# eccentric = A2_crop[abs(A2_crop.loc[:,'theta']) < 0.01]

plt.figure()
plt.scatter(abs(eccentric.loc[:,'theta']), eccentric.loc[:,'sig_rat'])
# plt.hist(abs(eccentric.loc[:,'theta']), bins=2)

#%%
plt.figure(figsize=(6, 6))
plt.imshow(TEM_micro, cmap='gray')
plt.scatter(at_cols.loc[:,'x'],at_cols.loc[:,'y'], c='red', s=5)
plt.scatter(at_cols.loc[:,'x_ref'],at_cols.loc[:,'y_ref'],
            c=at_cols.loc[:,'delta_loc'], cmap='viridis', s=5)
# plt.scatter(params_[1],params_[0], c='green')
