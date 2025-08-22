import SingleOrigin as so
import numpy as np

# %%
"""Location of .cif & image data"""

cif = 'ZnO.cif'
path = 'ZnO_dDPC.tif'

# %% SET UP THE PROJECTED UNIT CELL

"""Project the unit cell along along a zone axis. 'a1' and 'a2' must be
directions that correspond to planes that obey the zone law for 'za'.
'za' -> 'a1' -> 'a2' must also obey the right hand rule in that order. """

za = [1, 1, 0]   # Zone Axis direction
a1 = [1, -1, 0]  # First projected image lattice vector
a2 = [0, 0, 1]   # Second projected image lattice vector


uc = so.UnitCell(cif)
uc.project_zone_axis(
    za, a1, a2,
    ignore_elements=[],
    unique_proj_cell=True,
)

uc.combine_prox_cols(toler=1e-2)
uc.plot_unit_cell()

# %% LOAD THE IMAGE
# file_name = '09-07-2022_19.13.22_HAADFdrift_corr_HAADF_GRD.tif'
image, metadata = so.load_image(
    path=path,
    display_image=True,
)

# %% SET UP THE REFERENCE LATTICE AND GET IMAGE REGION TO ANALIZE

hrimage = so.HRImage(image)

zno = hrimage.add_lattice(
    'zno',
    uc,
    probe_fwhm=0.8,
    origin_atom_column=None,
)

zno.fft_get_basis_vect(
    a1_order=1,
    a2_order=2,
    sigma=3
)

# zno.get_region_mask_std(buffer=0)
zno.region_mask = np.ones(image.shape)

# %% REGISTER THE REFERENCE LATTICE

zno.define_reference_lattice()

# %% FIT THE ATOM COLUMNS
'''Here we use circular 2D Gaussians because this is zno'''

zno.fit_atom_columns(
    buffer=20,
    local_thresh_factor=0.5,
    peak_grouping_filter='auto',
    peak_sharpening_filter='auto',
    parallelize=True,
    use_circ_gauss=False,
)

# %% REFINE THE REFERENCE LATTICE

zno.refine_reference_lattice()

# %% PLOT THE FITTED POSITIONS

hrimage.plot_atom_column_positions(
    filter_by='elem',
    sites_to_plot='all',
    fit_or_ref='fit',
    plot_masked_image=False
)

# %% CHECK THE RESIDUALS

fig = zno.get_fitting_residuals()

# %% PLOT DISPLACEMENT VECTORS FROM (IDEAL) REFERENCE LATTICE POSITIONS

fig, axs = hrimage.plot_disp_vects(
    filter_by='elem',
    sites_to_plot='all',
    scalebar_len_nm=2,
    max_colorwheel_range_pm=25,
    arrow_scale_factor=1,
    outlier_disp_cutoff=None,
)
