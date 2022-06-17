"""SingleOrigin is a module for atomic column position finding intended for 
    high resolution scanning transmission electron microscope images.
    Copyright (C) 2022  Stephen D. Funni

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


__version__ = "2.1.1"

from SingleOrigin.utils import (metric_tensor,
                                bond_length,
                                bond_angle,
                                absolute_angle_bt_vectors,
                                rotation_angle_bt_vectors,
                                IntPlSpc,
                                IntPlAng,
                                TwoTheta,
                                elec_wavelength,
                                import_image,
                                image_norm,
                                write_image_array_to_tif,
                                img_equ_ellip,
                                img_ellip_param,
                                gaussian_2d,
                                LoG_2d,
                                gaussian2d_ss,
                                fit_gaussian2D,
                                pcf_radial,
                                detect_peaks,
                                watershed_segment)

from SingleOrigin.cell_transform import UnitCell

from SingleOrigin.find_atom_columns import AtomicColumnLattice