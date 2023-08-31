"""SingleOrigin is a module for atomic column position finding intended for 
    high resolution scanning transmission electron microscope images.
    Copyright (C) 2023  Stephen D. Funni

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see https://www.gnu.org/licenses"""


__version__ = "2.5"

from SingleOrigin.utils import (
    metric_tensor,
    bond_length,
    bond_angle,
    absolute_angle_bt_vectors,
    rotation_angle_bt_vectors,
    IntPlSpc,
    IntPlAng,
    get_astar_2d_matrix,
    TwoTheta,
    elec_wavelength,
    select_folder,
    select_file,
    load_image,
    image_norm,
    write_image_array_to_tif,
    band_pass_filter,
    fast_rotate_90deg,
    std_local,
    binary_find_largest_rectangle,
    binary_find_smallest_rectangle,
    fft_square,
    hann_2d,
    get_fft_pixel_size,
    get_feature_size,
    detect_peaks,
    watershed_segment,
    img_equ_ellip,
    img_ellip_param,
    gaussian_2d,
    gaussian_ellip_ss,
    gaussian_circ_ss,
    fit_gaussian_ellip,
    fit_gaussian_circ,
    pack_data_prefit,
    fit_gaussian_group,
    cft,
    find_cepstrum_peak,
    pcf_radial,
    v_pcf,
    pick_points,
    register_lattice_to_peaks,
    plot_basis,
    disp_vect_sum_squares,
    fit_lattice,
    fft_amplitude_area,
)

from SingleOrigin.cell_transform import UnitCell

from SingleOrigin.find_atom_columns import (
    HRImage,
    AtomicColumnLattice,
)

from SingleOrigin.reciprocal_lattice_analyzer import ReciprocalImage

from SingleOrigin.nbed4dstem import DataCube
