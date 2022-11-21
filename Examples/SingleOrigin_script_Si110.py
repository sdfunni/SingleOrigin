import SingleOrigin as so
import numpy as np
# import matplotlib.pyplot as plt

# %% SET UP THE PROJECTED UNIT CELL

"""Project the unit cell along along a zone axis. 'a1' and 'a2' must be
directions that correspond to planes that obey the zone law for 'za'.
'za' -> 'a1' -> 'a2' must also obey the right hand rule in that order. """

za = [1, 1, 0]   # Zone Axis direction
a1 = [1, -1, 0]  # First projected image lattice vector
a2 = [0, 0, 1]   # Second projected image lattice vector

cif = 'CIFS/silicon.cif'
uc = so.UnitCell(cif)
uc.project_zone_axis(za, a1, a2, ignore_elements=[], unique_proj_cell=True)
uc.combine_prox_cols(toler=1e-2)
uc.plot_unit_cell()

# %% LOAD THE IMAGE
file_name = '09-07-2022_19.13.22_HAADFdrift_corr_HAADF_GRD.tif'
image, metadata = so.load_image(path=file_name, display_image=True)

# %% SET UP THE REFERENCE LATTICE AND GET IMAGE REGION TO ANALIZE

hrimage = so.HRImage(image)
silicon = hrimage.add_lattice('silicon', uc, probe_fwhm=1,
                              origin_atom_column=None)

silicon.fft_get_basis_vect(a1_order=2, a2_order=4, sigma=3)

silicon.get_region_mask_std(buffer=0)

# %% REGISTER THE REFERENCE LATTICE

silicon.define_reference_lattice()

# %% FIT THE ATOM COLUMNS
'''Here we use circular 2D Gaussians because this is silicon'''

silicon.fit_atom_columns(buffer=40, local_thresh_factor=0,
                         grouping_filter='auto', diff_filter='auto',
                         parallelize=True, use_circ_gauss=True)
# %% REFINE THE REFERENCE LATTICE

silicon.refine_reference_lattice()

# %% PLOT THE FITTED POSITIONS

hrimage.plot_atom_column_positions(filter_by='elem', sites_to_plot='all',
                                   fit_or_ref='fit',
                                   plot_masked_image=False)

# %% CHECK THE RESIDUALS

fig = silicon.get_fitting_residuals()

# %% PLOT DISPLACEMENT VECTORS FROM (IDEAL) REFERENCE LATTICE POSITIONS

fig, axs = hrimage.plot_disp_vects(filter_by='elem',
                                   sites_to_plot='all',
                                   scalebar_len_nm=2,
                                   max_colorwheel_range_pm=25,
                                   arrow_scale_factor=1,
                                   outlier_disp_cutoff=np.inf)
