import cv2
import time
import os
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import PyQt5
from scipy.ndimage import (
    gaussian_filter,
    gaussian_laplace,
    label,
    find_objects,
    # rotate,
    standard_deviation,
    map_coordinates,
    # center_of_mass,
    # binary_fill_holes,
    # binary_erosion,
)
import SingleOrigin as so


# %%
"""Location of .cif & image data"""
cif = 'CIFS/LaTe3.cif'
# save_path = PyQt5.QtWidgets.QFileDialog.getExistingDirectory()

# %% SET UP THE PROJECTED UNIT CELL

"""Project the unit cell along along a zone axis. 'a1' and 'a2' must be
directions that correspond to planes that obey the zone law for 'za'.
'za' -> 'a1' -> 'a2' must also obey the right hand rule in that order. """

za = [1, 0, 0]   # Zone Axis direction
a1 = [0, 0, 1]  # First projected image lattice vector
a2 = [0, 1, 0]   # Second projected image lattice vector

uc = so.UnitCell(cif)

uc.project_zone_axis(
    za, a1, a2,
    ignore_elements=[],
    reduce_proj_cell=False,
)

uc.combine_prox_cols(toler=1e-2)
uc.plot_unit_cell()

# %% LOAD THE IMAGE
# file_name = 'STEM_Images/09-07-2022_19.13.22_HAADFdrift_corr_HAADF_GRD.tif'

image, metadata, path = so.load_image(
    path=None,
    display_image=True,
    return_path=True,
    load_dset=0
)

# %% SET UP THE REFERENCE LATTICE AND GET IMAGE REGION TO ANALIZE

hrimage = so.HRImage(image)

rete3 = hrimage.add_lattice(
    'rete3',
    uc,
    origin_atom_column=None,
)

# %%
verts = rete3.get_roi_mask_polygon(return_vertices=True, show_mask=True)
# rete3.get_roi_mask_polygon(vertices=verts, show_mask=True)

# rete3.roi_mask = roi_mask.copy()

# %%
rete3.fft_get_basis_vect(
    a1_order=2,
    a2_order=2,
    sigma=5,
    thresh_factor=0.5
)

# %% REGISTER THE REFERENCE LATTICE

rete3.define_reference_lattice(plot_ref_lattice=True,
                               buffer=40,
                               zoom_factor=2,
                               )

# %% FIT THE ATOM COLUMNS
'''Here we use circular 2D Gaussians'''
# rete3.probe_fwhm = 0.8
rete3.fit_atom_columns(
    buffer=0,
    local_thresh_factor=1.1,
    pos_toler=1,
    peak_grouping_filter=None,
    peak_sharpening_filter='auto',
    parallelize=True,
    use_circ_gauss=True,
    watershed_line=True,
    use_bounds=False, pos_bound_dist=1,
)

# %% REFINE THE REFERENCE LATTICE

rete3.refine_reference_lattice(filter_by='symbol',
                               sites_to_use='Te1',
                               outlier_disp_cutoff=30)

# %% PLOT THE FITTED POSITIONS
# xlim, ylim, _ = so.binary_find_largest_rectangle(rete3.roi_mask)
fig, ax = hrimage.plot_atom_column_positions(
    filter_by='elem',
    sites_to_plot='all',
    fit_or_ref='fit',
    plot_masked_image=False,
    outlier_disp_cutoff=None,
    scalebar_len_nm=1,
    # color_dict={'Zn': 'blue',
    #             'O': 'red'},
    # x_lim=xlim,
    # y_lim=ylim,
)

# fig.savefig(os.path.join(save_path, f'Column_postions_ZnO_DPC.tif'),
#             bbox_inches='tight',
#             pad_inches=0.2, dpi=300)

# %% PLOT DISPLACEMENT VECTORS FROM (IDEAL) REFERENCE LATTICE POSITIONS

fig, axs = hrimage.plot_disp_vects(
    filter_by='elem',
    sites_to_plot='all',
    scalebar_len_nm=2,
    max_colorwheel_range_pm=25,
    arrow_scale_factor=1,
    outlier_disp_cutoff=None,
)

# %%

fig = rete3.get_fitting_residuals()

# %%

hrimage_rot, lattice_dict_rot = hrimage.rotate_image_and_data(
    align_basis='a1',
    align_dir='right',
    lattice_to_align='rete3'
)

rete3 = lattice_dict_rot['rete3']

# %%
# xlim, ylim, _ = so.binary_find_largest_rectangle(rete3.roi_mask)
fig, ax = hrimage_rot.plot_atom_column_positions(
    filter_by='elem',
    sites_to_plot='all',
    fit_or_ref='fit',
    plot_masked_image=False,
    outlier_disp_cutoff=None,
    # x_lim=xlim,
    # y_lim=ylim,
)
# %%
"""
Calculate the vPCFs from the atom column data
"""

rete3.get_vpcfs(
    xlim=[-2, 2],
    ylim=[-1, 1],
    d=0.02,
    area=None,
    filter_by='symbol',
    sublattice_list=None,
    get_only_partial_vpcfs=False,
    affine_transform=False,
)

# %%
"""
Calculate the peak shapes by moments or 2D Gaussian fitting
"""

rete3.get_vpcf_peak_params(
    sigma=20,
    buffer=20,
    method='moments',
    sigma_group=None,
    thresh_factor=0,
)

# %%
"""
Plot vPCFs
"""

fig, axs, axs_cbar = rete3.plot_vpcfs(
    vpcfs_to_plot='all',
    plot_equ_ellip=True,
    vpcf_cmap='Greys',
    # ellip_color_scale_param='sig_maj',
    ellip_scale_factor=1,
    unit_cell_box=True,
    scalebar_len=0.2
)

# fig.savefig(os.path.join(save_path, f'ZnO_DPC_vPCFs.tif'),
#             bbox_inches='tight',
#             pad_inches=0.2, dpi=300)
# %%
"""
Plot inter-atom column vector(s) chosen from vPCF
"""

"""
*** Choose 2 near neighbor peaks in the Al-N pair-pair vPCF. 
I have this set up to take the component of the Al-N dumbell distance along
the [001] direction only and plotting that distance.
"""

fig = rete3.plot_distances_from_vpcf_peak(
    'O-Zn',
    r=0.1,
    locate_by_fit_or_ref='ref',
    plot_fit_or_ref='ref',
    number_of_peaks_to_pick=1,
    # dist_along_vector=rete3.a2,
    # dist_along_vector=np.array([rete3.a2, rete3.a2]),
    deviation_or_absolute='deviation',
    plot_equ_ellip=False,
    center_deviation_at_zero=False,
    xlim=[hrimage.w*0.2, hrimage.w*0.4],
    ylim=[hrimage.h*0.2, hrimage.h*0.4],
)

fig.savefig(os.path.join(save_path, f'ZnO_DPC_Zn_vect6.tif'),
            bbox_inches='tight',
            pad_inches=0.2, dpi=300)


# %%
min_ = []
inds = []
xy1 = sub1.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float)
xy2 = sub2.loc[:, 'x_ref':'y_ref'].to_numpy(dtype=float)

for ind1, xy in enumerate(tqdm(xy1)):
    vect_err = norm(xy2 - xy - vect, axis=1)
    min_ = np.min(vect_err)
    if min_ < 1:
        inds += [ind1]

# %%
d_vects = xy2 - xy1[200, :]  # - vect
vect_err = norm(d_vects, axis=1)
min_ = np.min(vect_err)


test_inds = np.array([np.argmin(norm(xy2 - xy_, axis=1)) for xy_ in xy1])

vects = np.array([xy2[ind2] - xy1[ind1]
                 for ind1, ind2 in enumerate(test_inds)])
d_vects = np.where(norm(vects - vect, axis=1) < 1, 1, 0)


# %%
fig, ax = plt.subplots(1)
ax.imshow(hrimage_rot.image)
ax.scatter(xy1[4, 0], xy1[4, 1])
ax.scatter(xy2[4, 0], xy2[4, 1])
