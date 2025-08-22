import colorcet as cc
from ase.io import read
import os
# import datetime
import copy

import numpy as np
from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, scatter

import SingleOrigin as so

# %%
# Prepare dictionary for atom column position dataframes for mulitple images
at_cols_frames = {}

# %%
# Set folder to save figure results

# save_folder = so.select_folder()
save_folder = '/Users/funni/Documents/Academics/Grad_School/Research/Projects/TTBs/Experiment/DPC_results/BSSN_DPC_new'
# %% CHOOSE WORKING DIRECTORY
# data_folder = so.select_folder()
data_folder = '/Users/funni/Documents/Academics/Grad_School/Research/Projects/TTBs/Experiment/DPC_results/BSSN DPC images'

# %%
# Set up projected unit cell

za = [1, 1, 0]  # Zone Axis direction
a1 = [-1, 1, 0]  # Apparent horizontal axis in projection
a2 = [0, 0, 1]  # Most vertical axis in projection

cif = \
    '/Users/funni/Documents/Academics/Postdoc/Python Scripts/CIFS/BSSN Models/BSSN5137.cif'

uc = so.UnitCell(cif)


def LatticeSiteLabeling(unitcell):
    # Apply logical framework to categorize project atoms by sublattices
    lattice_site = []

    for ind in unitcell.index:
        if unitcell.at[ind, 'elem'] == 'Sm|Ba':
            lattice_site.append('A1')
        elif unitcell.at[ind, 'elem'] == 'Ba':
            # if np.isin(unitcell.at[ind, 'v'], [0.25, 0.75]):
            if unitcell.at[ind, 'u'] == 0.5:
                lattice_site.append('A2_1')
            else:
                lattice_site.append('A2_2')
        elif unitcell.at[ind, 'elem'] == 'Sn|Nb':
            # if np.isin(unitcell.at[ind, 'v'], [0.25, 0.75]):
            if unitcell.at[ind, 'u'] == 0.5:
                lattice_site.append('B1')
            else:
                lattice_site.append('B2')
        elif unitcell.at[ind, 'elem'] == 'O':
            if unitcell.at[ind, 'u'] < 0.5:
                lattice_site.append('O')
            elif unitcell.at[ind, 'u'] > 0.5:
                lattice_site.append('O')

        # else:
        #     print(unitcell.loc[ind, :])

    unitcell.loc[:, 'LatticeSite'] = lattice_site
    return unitcell


# Transform Unit Cell with and without oxygen
uc_O = copy.copy(uc)

uc.project_zone_axis(za, a1, a2, ignore_elements=['O'], reduce_proj_cell=True)
uc.combine_prox_cols(toler=1e-1)

uc_O.project_zone_axis(za, a1, a2, ignore_elements=[], reduce_proj_cell=True)
uc_O.combine_prox_cols(toler=1e-1)

# Filtering to remove unresolved oxygen columns in [110] TTB projection
df_O = uc_O.at_cols[(uc_O.at_cols.elem == 'O')].copy()
df_cation = uc.at_cols[uc.at_cols.elem != 'O'].copy()

df_O = df_O[(np.isclose(df_O.u, 0.2785, atol=0.001)) |
            (np.isclose(df_O.u, 0.7215, atol=0.001))]

df_all = pd.concat([df_cation, df_O], axis=0)


uc.at_cols = df_all
uc.at_cols.reset_index(drop=True, inplace=True)
uc.at_cols = LatticeSiteLabeling(uc.at_cols)

uc.at_cols.loc[:, 'u':'v'] = (
    uc.at_cols.loc[:, 'u':'v'] + np.array([0.5, 0.5])
) % 1

uc.at_cols.loc[:, 'x':'y'] = uc.at_cols.loc[:, 'u':'v'].to_numpy() @ uc.a_2d

label_dict = {'A1': 'A1',
              'A2_1': r'A2$_1$',
              'A2_2': r'A2$_2$',
              'B1': 'B1',
              'B2': 'B2',
              # 'O1': 'O1',
              # 'O2': 'O2',
              'O': 'O',
              }
color_dict = {'A1': 'royalblue',
              'A2_1': 'blue',
              'A2_2': 'deepskyblue',
              'B1': 'green',
              'B2': 'lawngreen',
              # 'O1': 'red',
              # 'O2': 'red',
              'O': 'red',
              }

# %%
uc.plot_unit_cell(
    label_by='LatticeSite',
    label_dict=label_dict,
    color_dict=color_dict
)

# plt.savefig(os.path.join(save_folder, 'DPC_projected_atom_columns.tif'),
#             bbox_inches='tight', pad_inches=0.25, dpi=300)

# %% Get all .tif filenames in the directory
file_list = []
for file in os.listdir(data_folder):
    if np.isin(file.split('.')[-1], ['tif']).item():
        file_list += [file]
    else:
        continue
file_list.sort()
file_list = file_list[1:]


file = iter([i for i in file_list])

# %% Cycle through images

filename = next(file)
[scan_num, scan_id] = [int(''.join(i for i in s if i.isdigit() or i in ''))
                       for s in filename.split('_')[1:3]]

image, _ = so.load_image(os.path.join(data_folder, filename),
                         display_image=False)

# %%
"""
Initiate image analysis object and get basis vectors
"""
hrimage = so.HRImage(image, pixel_size_cal=0.013184941*10 / 2)
bssn = hrimage.add_lattice('BSSN', uc,
                           origin_atom_column=None)

bssn.fft_get_peaks(thresh_factor=0.5,
                   sigma=5)

# %%
bssn.fft_get_basis_vect(a1_order=3, a2_order=1, sigma=5, thresh_factor=0.5)
# %%
"""
Get region mask for analysis region and define the reference lattice
"""
bssn.select_lattice_origin()

# %%
bssn.make_reference_lattice(plot_ref_lattice=True, buffer=10)

# %% FIT THE ATOM COLUMNS
"""
Fit the atom columns with 2D Gaussians
"""
bssn.fit_atom_columns(
    bkgd_thresh_factor=0.5,
    peak_sharpening_filter='auto',
    peak_grouping_filter='auto',
    parallelize=True,
    use_circ_gauss=True,
    use_bounds=True,
    pos_bound_dist=0.5,
    watershed_line=True,
)


# %%
# Refine the reference lattice using the fitted positions
bssn.refine_reference_lattice('LatticeSite', 'A1')

# %% Plot the positions

fig, ax = hrimage.plot_atom_column_positions(
    filter_by='LatticeSite',
    sites_to_plot='all',
    fit_or_ref='fit',
    plot_masked_image=False,
    # xlim=[0, hrimage.w*1],
    # ylim=[0, hrimage.h*1],
    legend_dict=label_dict,
    color_dict=color_dict,
    scalebar_len_nm=1,
    figax=True,
)

# fig.set_size_inches(10, 10)
# plt.savefig(
#     os.path.join(save_folder, _zoom.tif'),
#     bbox_inches='tight',
#     pad_inches=0,
#     dpi=300
# )
so.save_fig(fig, save_folder, f'scan{scan_num}_fitted_pos_fig.png', dpi=300)

# %% Rotate image and data

acl_rot, bssn_rot = hrimage.rotate_image_and_data(
    align_basis='a1',
    align_dir='right',
    lattice_to_align='BSSN')

# %%
"""
Calculate the vPCFs from the atom column data
"""

acl_rot.latt_dict['BSSN_rot'].get_vpcfs(
    xlim=[-2, 2],
    ylim=[-1.25, 1.75],
    d=0.02,
    area=None,
    filter_by='LatticeSite',
    sublattice_list=['O', 'A2_1'],
    get_only_partial_vpcfs=False,
    affine_transform=False,
    outlier_disp_cutoff=0.25,
    centrosymmetric=True,
)

# %%
"""
Calculate the peak shapes by moments or 2D Gaussian fitting
"""

acl_rot.latt_dict['BSSN_rot'].get_vpcf_peak_params(
    sigma=10,
    buffer=20,
    method='moments',
    sigma_group=None,
    thresh_factor=0.1,
)

# %%
"""
Plot vPCFs
"""

fig, axs, axs_cbar = acl_rot.latt_dict['BSSN_rot'].plot_vpcfs(
    vpcfs_to_plot='all',
    plot_equ_ellip=True,
    vpcf_cmap='Greys',
    ellip_colormap_param='sig_maj',
    # colormap_range=[6,12],
    ellip_scale_factor=3,
    unit_cell_box=True,
    scalebar_len=0.3
)

# fig.savefig(os.path.join(save_path, f'{filename[:-4]}_dDPC_vPCFs.tif'),
#             bbox_inches='tight',
#             pad_inches=0.2, dpi=300)

i = 0
# %%
"""
Plot inter-atom column vector(s) chosen from vPCF
"""

"""
*** Choose 2 near neighbor peaks in the Al-N pair-pair vPCF.
I have this set up to take the component of the Al-N dumbell distance along
the [001] direction only and plotting that distance.
"""
acl_rot.latt_dict['BSSN_rot'].vpcf_pick_peaks(
    'O-O',
    n_picks=1,
    plot_equ_ellip=True,
)

# %%

fig = acl_rot.latt_dict['BSSN_rot'].plot_vectors_on_image(
    r=.25,
    locate_by_fit_or_ref='ref',
    plot_fit_or_ref='fit',
    ref_vect=acl_rot.latt_dict['BSSN_rot'].dir_struct_matrix[0],
    vector_cmap_values='angle',
    plot_equ_ellip=False,
    outlier_disp_cutoff=0.5,
    # xlim=[464, 2015],
    # ylim=[478, 2002],
    cmap='RdBu_r'
)

# %%
so.save_fig(
    fig,
    save_folder,
    # f'scan{scan_num}_O_close_distances.png',
    # f'scan{scan_num}_O_far_distances.png',
    f'scan{scan_num}_A2_1_distances.png',
    dpi=300,
)

# %%
fig, ax = so.quickplot(np.where(bssn.peak_masks_orig == 0, -500, bssn.peak_masks_orig),
                       figax=True)

ax.scatter(
    bssn.at_cols_uncropped[bssn.at_cols_uncropped.elem == 'O'].loc[:, 'x_ref'],
    bssn.at_cols_uncropped[bssn.at_cols_uncropped.elem == 'O'].loc[:, 'y_ref'],
    s=5)

# %%

hrimage.plot_disp_vects(
    filter_by='elem',
    sites_to_plot=['O'],
    outlier_disp_cutoff=None,
    xlim=None,
    ylim=None,
    scalebar_len_nm=2,
    arrow_scale_factor=1,
    max_colorwheel_range_pm=None,
    label_dict=None,
    plot_fit_points=False,
    plot_ref_points=False
)

# %% Rotate image and data

acl_rot, bssn_rot = hrimage.rotate_image_and_data(
    align_basis='a1',
    align_dir='right',
    lattice_to_align='BSSN')

# %% Convert units and save to dictionary
"""
Convert units from pixel to Angstroms and correct lattice distortion.
Save result in the dataframe.
"""

xy_fit = bssn_rot.at_cols.loc[:, 'x_fit': 'y_fit'].to_numpy() \
    - np.array([bssn_rot.x0, bssn_rot.y0])

uv_fit = xy_fit @ bssn_rot.dir_struct_matrix \
    / norm(bssn_rot.dir_struct_matrix, axis=1)**2

xy_agstr = uv_fit @ bssn_rot.a_2d

bssn_rot.at_cols['x_Agstr'] = xy_agstr[:, 0]
bssn_rot.at_cols['y_Agstr'] = xy_agstr[:, 1]

at_cols_frames[str(scan_num)] = bssn_rot.at_cols.copy()


# %%
for key, df in at_cols_frames.items():
    df.to_csv(os.path.join(save_folder, f'scan{key}_at_cols.csv'))
