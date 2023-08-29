import os

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import SingleOrigin as so
from PyQt5.QtWidgets import QFileDialog as qfd

from tem_image_utils import (
    load_empad2_data,
    load_empad_data,
)

import hyperspy.api as hs
import py4DSTEM

# %%
"""
Select path to data

For EMPAD2, also select dark background and the empad2 calibration file
locatrions
"""

dataPath = so.select_file(ftypes='.raw')

# bkgdPath = so.select_file(ftypes='.raw')

# calibrationPath = '/Users/funni/Documents/Academics/Postdoc/Python Scripts/EMPAD scripts/EMPAD2_calibrations'

# %%
"""
Load and correct an EMPAD2 dataset, optionally bin pixels
"""

data = load_empad2_data(
    dataPath,
    bkgdPath,
    calibrationPath,
    scan_dims=None,
    bkgd_dims=None,
    bin_scan=None,
    bin_detector=None
)

# %%
"""
Load an EMPAD (1) dataset, optionally bin pixels
"""

data = load_empad_data(
    dataPath,
    scan_dims=None,
    bkgd_dims=None,
    bin_scan=None,
    bin_detector=None)

# %%
"""
Initiate the DataCube object for analysis
"""
datacube = so.DataCube(data)

# %%
"""
Set up analysis on the mean EWPC
"""

basis_picks = datacube.initalize_cepstral_analysis(
    pick_basis_order=(1, 1),        # Pick 2nd along 1st direction,
                                    # 1st peak along 2nd direction.
    use_only_basis_peaks=False,     # Use all peaks in the specified range
    measure_basis_order=(2, 1),     # Use 2nd, 1st peaks for measurements
    r_max=30,                       # Max radius peak considered
    r_min=5,                        # Min radius peak considered
    graphical_picking=False,        # Choose peak by clicking in plot
                                    # If true plots w/ labels
    min_order=1,                    # Min order of peak to attempt to find
    max_order=2,                    # Max order of peak to attempt to find
    scaling='pwr',                  # Contrast scaling type for display
    power=0.2,                      # Power to use for contrast scaling
    # pick_labels=[223, 269]          # Choose basis peaks by label #
                                    # If not passed, and graphical_picking is
                                    # False, plots with labels so you can
                                    # decide which to use.
)

# %%
"""
Run the cepstral strain analysis
"""

datacube.get_cepstral_strain(
    window_size=3,                  # Window size to bound peak finding
    rotation=78,                    # Rotation between scan and EWPC dimensions
    ref=np.s_[0:50, 150:200]        # Strain reference region
)

# %%
"""
Plot the strain results
"""

fig, axs = datacube.plot_strain_maps(
    normal_strain_lim=(-1, 1),      # Colorbar limits in % strain
    shear_strain_lim=(-1, 1),       # Colorbar limits in % strain
    theta_lim=(-1, 1),              # Colorbar limits in degrees
    return_fig=True,
    # figsize=(10, 5),
)

# %%
"""
Save the figure
"""

savepath = '/' + os.path.join(*dataPath.split('/')[:-1])

file_code = dataPath.split('/')[-2]
plt.savefig(os.path.join(savepath, f'strain_maps_latticefit_{file_code}.tif'),
            bbox_inches='tight', pad_inches=0.01, dpi=300)
