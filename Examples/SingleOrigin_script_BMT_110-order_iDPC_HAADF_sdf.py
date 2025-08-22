import SingleOrigin as so
from SingleOrigin import rotation_angle_bt_vectors
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm
from numpy.linalg import norm

#%%
"""Location of .cif & image data"""

cif = 'CIFS/BMT110order_vesta.cif'

# image_path = '/Users/funni/Downloads/ForSteph_20230306/DrProbe_BMT_order_iDPC_FlipHo.tif'

image_path = '/Users/funni/Downloads/ForSteph_20230306/DrProbe_BMT_order_HAADF_FlipHo.tif'



# %% SET UP THE PROJECTED UNIT CELL

"""Project the unit cell along along a zone axis. 'a1' and 'a2' must be
directions that correspond to planes that obey the zone law for 'za'.
'za' -> 'a1' -> 'a2' must also obey the right hand rule in that order. """

za = [0,0,-1]            #Zone Axis direction
a1 = [-1,0,0]         #Apparent horizontal axis in projection
a2 = [0,1,0]          #Most vertical axis in projection

uc = so.UnitCell(path=cif)

uc.project_zone_axis(
    za, a1, a2,
    ignore_elements=["O", "Mg"],          # For HAADF
    reduce_proj_cell=True,
)

uc.combine_prox_cols(toler=1e-2)

uc.plot_unit_cell()

# %% LOAD THE IMAGE
image, metadata = so.load_image(
    path=image_path,
    display_image=False,
    return_path=False,
    emd_velox=False,
)

"""
*** Crop tilied simulation image to single unit cell for analysis
"""

image_ = image[:144, :202]

plt.figure()
plt.imshow(image_)
# %% SET UP THE REFERENCE LATTICE

hrimage = so.HRImage(image_)

bmt = hrimage.add_lattice(
    'bmt',
    uc,
    probe_fwhm=0.8,
    # origin_atom_column=13, # iDPC
    origin_atom_column=6, # HAADF Lower left Ta column
)

#%% GET BASIS VECTORS

"""
Here I specify the basis vectors as the cropped image edge vectors. If more
than one unit cell in the image, you could just divide by the appropriate
number along each direction.
"""
bmt.specify_basis_vectors(a1=[202, 0], a2=[0, -144])

"""
For a regular image use the following. It is more or less as before,
applied tothe lattice object(which I called "bmt" here).:
"""
# %% REGISTER THE REFERENCE LATTICE

bmt.define_reference_lattice(
    plot_ref_lattice=True,
    # origin=[57, 84],    # For HAADF
    origin=[65, 108],    # For iDPC
)

# %% FIT THE ATOM COLUMNS

hrimage.probe_fwhm=0.8
bmt.probe_fwhm=0.8
bmt.fit_atom_columns(
    buffer=0,
    local_thresh_factor= 0.5, 
    peak_grouping_filter=None,
    peak_sharpening_filter='auto',
    use_circ_gauss=True,
    use_bounds=True,
    use_background_param=True,
    )

#%%
fig = bmt.show_masks(mask_to_show='grouping')

# %% REFINE THE REFERENCE LATTICE

#For iDPC
#bmt.refine_reference_lattice(filter_by='elem', sites_to_use=['Ba|O'], 
#                             outlier_disp_cutoff=30)
#For HAADF
bmt.refine_reference_lattice(filter_by='elem', sites_to_use='all',
                             outlier_disp_cutoff=30)

# %% PLOT THE FITTED POSITIONS
"""
Notice that the plotting functions are applied to the hrimage object, not a
specific lattice. This allows you to have more than one lattice in an image,
for example a substrate and a thin film. 
"""

hrimage.plot_atom_column_positions(
    filter_by='elem',
    sites_to_plot='all',
    fit_or_ref='fit',
    outlier_disp_cutoff=None,
    plot_masked_image=False,
    scalebar_len_nm=None,
    )

# %% PLOT DISPLACEMENT VECTORS FROM (IDEAL) REFERENCE LATTICE POSITIONS

fig, axs = hrimage.plot_disp_vects(filter_by='elem', sites_to_plot='all', #titles=None,
                          x_lim=[0, bmt.w], y_lim=[bmt.h, 0],
                          # x_crop=[0, 500], y_crop=[500, 0],
                          outlier_disp_cutoff=None,
                          # scalebar=True, 
                          scalebar_len_nm=0.5,
                          max_colorwheel_range_pm=10,
                          plot_fit_points=False, plot_ref_points=False,
                          arrow_scale_factor=2)

#%%
fig = bmt.get_fitting_residuals()