import SingleOrigin as so
import numpy as np
import matplotlib.pyplot as plt
import copy

"""This is an example script for using the multiple lattice functionality
in SingleOrigin. That is, fitting more than one lattice to a single image."""
# %% SET UP TWO CRYSTAL STRUCTURES FOR FILM AND SUBSTRATE

za = [1, 1, 0]  # Zone Axis direction
a2 = [-1, 1, 0]  # Apparent horizontal axis in projection
a3 = [0, 0, 1]  # Most vertical axis in projection

# !!! Load the STO.cif file here.
uc = so.UnitCell()

# Then we modify it for the right elements in each lattice.
uc_bsmo = copy.deepcopy(uc)
uc_bsmo.atoms.replace(['Sr', 'Ti'], ['Ba/Sr', 'Mn'], inplace=True)
uc_dso = copy.deepcopy(uc)
uc_dso.atoms.replace(['Sr', 'Ti'], ['Dy', 'Sc'], inplace=True)

uc_bsmo.transform_basis(za, a2, a3)
uc_dso.transform_basis(za, a2, a3)

uc_bsmo.project_uc_2d(proj_axis=0, ignore_elements=['O'])
uc_dso.project_uc_2d(proj_axis=0, ignore_elements=['O'])

uc_bsmo.combine_prox_cols(toler=1e-2)
uc_dso.combine_prox_cols(toler=1e-2)
uc_dso.plot_unit_cell()

# %% LOAD THE IMAGE
image, metadata = so.import_image(display_image=True)

# %% INITIATE HRIMAGE OBJECT

hrimage = so.HRImage(image)
# %% ADD LATTICE OBJECT

bsmo = hrimage.add_lattice('BSMO', uc_bsmo)
bsmo.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=4)

# %% MASK AREA TO FIT ATOM COLUMNS

vertices = np.array([[785, 0],
                     [bsmo.w, 0],
                     [bsmo.w, bsmo.h],
                     [780, bsmo.h]])

# %%
bsmo.get_region_mask_polygon(vertices=vertices,
                             buffer=0, invert=False, show_poly=True)

plt.figure()
plt.imshow(bsmo.region_mask)
# %% PICK ORIGIN ATOM COLUMN
bsmo.define_reference_lattice()

# %% FIT PEAKS
bsmo.fit_atom_columns(buffer=20, local_thresh_factor=0.5,
                      grouping_filter='auto', diff_filter='auto',
                      use_circ_gauss=True,
                      parallelize=True)

# %%
bsmo.refine_reference_lattice(filter_by='elem', sites_to_use='Ba/Sr',
                              outlier_disp_cutoff=30)

# %% ADD SECOND LATTICE OBJECT
dso = hrimage.add_lattice('DSO', uc_dso)
dso.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=4)

# %% MASK AREA TO FIT ATOM COLUMNS
dso.get_region_mask_polygon(vertices=vertices,
                            buffer=0, invert=True, show_poly=False)

# %% PICK ORIGIN ATOM COLUMN
dso.define_reference_lattice()

# %% FIT PEAKS
dso.fit_atom_columns(buffer=20, local_thresh_factor=0.5,
                     grouping_filter='auto', diff_filter='auto',
                     use_circ_gauss=True,
                     parallelize=True)

# %% REFINE THE REFERENCE LATTICE BASED ON THE FITTED POSITIONS
dso.refine_reference_lattice(filter_by='elem', sites_to_use='Dy',
                             outlier_disp_cutoff=30)

# %%
"""Plot Column positions
NOTE: Plotting funcions are methods of the HRImage class"""

fig, ax = hrimage.plot_atom_column_positions(filter_by='elem',
                                             sites_to_plot='all',
                                             fit_or_ref='fit',
                                             outlier_disp_cutoff=50,
                                             plot_masked_image=True,
                                             scatter_kwargs_dict={'s': 10})


# %%
"""Plot Column displacement vectors"""
fig, axs = hrimage.plot_disp_vects(filter_by='elem', sites_to_plot='all',
                                   scalebar_len_nm=2,
                                   max_colorwheel_range_pm=None,
                                   arrow_scale_factor=1, outlier_disp_cutoff=np.inf)
