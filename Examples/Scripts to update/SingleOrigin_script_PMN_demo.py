import SingleOrigin as so
import numpy as np

# %% SET UP TWO CRYSTAL STRUCTURE

za = [1, 1, 0]  # Zone Axis direction
a2 = [-1, 1, 0]  # Apparent horizontal axis in projection
a3 = [0, 0, 1]  # Most vertical axis in projection

# !!! Load the PMN.cif here
uc = so.UnitCell()

uc.transform_basis(za, a2, a3)

uc.project_uc_2d(proj_axis=0, ignore_elements=[])

uc.combine_prox_cols(toler=1e-2)
uc.at_cols.loc[:, 'LatticeSite'] = ['B', 'O', 'A']

uc.plot_unit_cell(label_by='LatticeSite')
sites = sorted(uc.at_cols.loc[:, 'LatticeSite'].to_list())

# %% LOAD THE IMAGE
image, metadata = so.import_image(display_image=True)

# %% INITIATE HRIMAGE OBJECT
hrimage = so.HRImage(image)

# %% ADD LATTICE OBJECT
lattice = hrimage.add_lattice('PMN', uc, probe_fwhm=0.8)
lattice.fft_get_basis_vect(a1_order=1, a2_order=1, sigma=4)

# %% PICK ORIGIN ATOM COLUMN
lattice.define_reference_lattice()

# %% FIT PEAKS
lattice.fit_atom_columns(buffer=10, local_thresh_factor=0.6,
                         grouping_filter='auto', diff_filter=3,
                         use_circ_gauss=False)

# %% REFINE THE REFERENCE LATTICE BASED ON THE FITTED POSITIONS
lattice.refine_reference_lattice(filter_by='LatticeSite', sites_to_use='B',
                                 outlier_disp_cutoff=30)

# %% PLOT THE ATOM COLUMN POSITIONS
fit, axs = hrimage.plot_atom_column_positions(filter_by='elem',
                                              sites_to_plot='all',
                                              fit_or_ref='fit',
                                              outlier_disp_cutoff=None,
                                              plot_masked_image=True)

# %% PLOT DISPLACEMENT VECTORS OF THE FITTED POSITIONS FROM THE REFERENCE
hrimage.plot_disp_vects(filter_by='elem',
                        sites_to_plot='all',
                        scalebar_len_nm=2,
                        max_colorwheel_range_pm=None,
                        arrow_scale_factor=2, outlier_disp_cutoff=30)
